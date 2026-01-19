import torch
import torch.nn.functional as F
from load_dsu_model import ModelInitializerLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import DynamicCache, LlamaForCausalLM


class DSUModel(ModelInitializerLoader):
    def __init__(self, config):
        super().__init__(config)

        if not hasattr(config, "num_dsus"):
            raise ValueError(
                "config.num_dsus must be set before initializing DSU model"
            )

        self.num_dsus = config.num_dsus
        self.text_stream = config.text_stream
        self.multi_text_stream = config.multi_text_stream
        self.use_speaker_embedding = config.use_speaker_embedding
        self.calc_loss_on_c1_only = config.calc_loss_on_c1_only

        # vocab sizes for text and audio
        self.text_vocab_size = self.get_output_embeddings().weight.size(0)
        self.audio_vocab_size = config.audio_vocab_size + 2

        # hidden size
        self.hidden_size = self.get_input_embeddings().embedding_dim

        # heads will be initalized later (if num_dsus > 0)
        self.dsu_head = None
        self.num_dsu_heads = None

        # text heads will be initialized later if multi_text_stream
        self.text_head = None
        self.num_text_heads = None
        if self.text_stream:
            self.num_text_streams = 1
        elif self.multi_text_stream:
            self.num_text_streams = 2
        else:
            self.num_text_streams = 0

        # audio embeds will be initialized later
        self.audio_embeds = None
        self.num_audio_embeds = None

        # set audio EOS to last element in vocab (0-index)
        self.audio_eos_id = self.audio_vocab_size - 1
        self.audio_delay_id = self.audio_vocab_size - 2

        # set main process for logging and debugging
        is_main_process = (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )
        self.is_main = is_main_process

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        dsu_ids=None,
        text_stream_ids=None,
        spk_emb=None,
        inference=False,
        need_loss=True,
        past_key_values=None,
        use_cache=None,
        **kwargs,
    ):

        if self.num_dsus < 1:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        labels = None  # was just a placeholder before and will be built below

        get_input_embeds_and_labels = (
            self.get_input_embeds_and_labels_padded
            if inference
            else self.get_input_embeds_and_labels
        )

        (
            prompt_dsu_input_embeds,
            prompt_dsu_attention_mask,
            dsu_labels,
            ts_labels,
        ) = get_input_embeds_and_labels(
            prompt_ids=input_ids,  # prompt ids
            prompt_att_mask=attention_mask,  # prompt attention_mask
            dsu_ids=dsu_ids,
            text_stream_ids=text_stream_ids,
            spk_emb=spk_emb,
            inference=inference,
        )

        if use_cache and len(past_key_values) > 0:
            prompt_dsu_input_embeds = prompt_dsu_input_embeds[:, -1:, :]

        outputs = super().forward(
            inputs_embeds=prompt_dsu_input_embeds,
            attention_mask=prompt_dsu_attention_mask,
            labels=None,
            output_hidden_states=True,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.hidden_states[-1]  # [B, L, Hdim]
        B, L_with_text, _ = hidden_states.shape

        if use_cache and len(past_key_values) > 0:
            audio_hidden_padded = hidden_states[:, -1:, :]
        else:
            # only get audio hidden states and repad
            prompt_lens = (
                prompt_dsu_attention_mask.sum(dim=1)
                - (dsu_labels != self.pad_token_id).sum(dim=-1)[:, 0]
            )

            audio_hidden = [hidden_states[b, prompt_lens[b] :, :] for b in range(B)]

            audio_hidden_padded = pad_sequence(
                audio_hidden, batch_first=True, padding_value=0.0
            )

        B, L, _ = audio_hidden_padded.shape

        # feed hiddens through heads

        dsu_logits = self.dsu_head(audio_hidden_padded).view(
            B, L, self.num_dsu_heads, -1
        )

        logits_labels_pairs = [
            (dsu_logits, dsu_labels, "dsus"),
        ]

        if self.multi_text_stream:
            ts_logits = torch.stack(
                [head(audio_hidden_padded) for head in self.text_head], dim=2
            )
            logits_labels_pairs.append((ts_logits, ts_labels, "text"))

        elif self.text_stream:
            ts_logits = outputs.logits.unsqueeze(2)  # [B, L, V] -> [B, L, 1, V]
            logits_labels_pairs.append((ts_logits, ts_labels, "text"))
        else:
            ts_logits = None

        total_loss, c1_dsu_loss, c1_text_loss = None, None, None

        if need_loss:
            total_loss, c1_dsu_loss, c1_text_loss = 0, 0, 0

            for logits, labels, loss_type in logits_labels_pairs:

                labels_padded = F.pad(labels, (0, 1), value=self.pad_token_id)
                labels_shifted = labels_padded[..., 1:].contiguous()

                if (
                    not self.model.training or self.calc_loss_on_c1_only
                ):  # only care about speaker 1 ppl
                    labels_shifted = labels_shifted[
                        :, : max(1, labels_shifted.shape[1] // 2), :
                    ]
                    logits = logits[:, :, : max(1, logits.shape[2] // 2), :]

                mask = labels_shifted != self.pad_token_id
                logits = logits.permute(0, 2, 1, 3)  # [B, L, H, V] -> [B, H, L, V]

                logits_flat = logits[mask]
                labels_flat = labels_shifted[mask]
                loss = F.cross_entropy(
                    logits_flat, labels_flat, ignore_index=self.pad_token_id
                )

                total_loss += loss
                if loss_type == "dsus":
                    c1_dsu_loss += loss
                elif loss_type == "text":
                    c1_text_loss += loss
                else:
                    raise NotImplementedError

            if not inference and total_loss is not None:
                total_loss = total_loss / self.grad_acc_steps

        return {
            "loss": total_loss,
            "c1_text_loss": c1_text_loss,
            "c1_dsu_loss": c1_dsu_loss,
            "logits": dsu_logits,
            "ts_logits": ts_logits,
            "past_key_values": past_key_values,
        }

    def check_vocab_bounds(self, prompt_ids, dsu_ids):
        if (prompt_ids >= self.text_vocab_size).any():
            raise ValueError("Prompt IDs contain out-of-vocabulary tokens")

        if self.audio_eos_id >= self.audio_vocab_size:
            raise ValueError("Audio EOS ID is out-of-vocabulary for audio embeddings")

    def get_input_embeds_and_labels(
        self,
        prompt_ids,
        prompt_att_mask,
        dsu_ids,
        text_stream_ids=None,
        spk_emb=None,
        inference=False,
    ):

        self.check_vocab_bounds(prompt_ids, dsu_ids)
        create_eos_token = lambda token_id, ref_tensor: torch.full(
            (ref_tensor.size(1), 1),  # batch_size x 1
            token_id,
            device=ref_tensor.device,
            dtype=ref_tensor.dtype,
        )
        audio_eos = create_eos_token(self.audio_eos_id, dsu_ids)
        text_eos = (
            create_eos_token(self.pad_token_id, text_stream_ids)
            if text_stream_ids is not None
            else None
        )

        embed_layer = self.get_input_embeddings()
        audio_embedding_lookup = self.audio_embeds

        B = prompt_ids.shape[0]

        if dsu_ids is None:
            dsu_ids = torch.full(
                (B, 1, 0),
                self.pad_token_id,
                device=prompt_ids.device,
                dtype=torch.long,
            )

        all_concat_embeds = []
        labels_all_dsu_heads = []
        labels_all_text_streams = []

        for b in range(B):
            # prompt
            prompt_ids_nopad = prompt_ids[b, prompt_att_mask[b].bool()]
            prompt_embeds = embed_layer(prompt_ids_nopad)

            # DSU
            dsu_pad_mask = dsu_ids[b] != self.pad_token_id
            non_pad_len = dsu_pad_mask.sum(dim=1)[0]
            dsu_ids_b = dsu_ids[b, :, :non_pad_len]

            if not inference:
                dsu_ids_b = torch.cat([dsu_ids_b, audio_eos], dim=1)

            dsu_embeds = torch.stack(
                [
                    audio_embedding_lookup[h](dsu_ids_b[h])  # per head embedding
                    for h in range(self.num_dsu_heads)
                ],
                dim=0,
            )

            summed_dsu = dsu_embeds.sum(dim=0)
            labels_all_dsu_heads.append(dsu_ids_b)

            if self.use_speaker_embedding:

                assert B == len(spk_emb)

                proj_spk = self.speaker_embed_proj(
                    torch.tensor(
                        spk_emb[b], device=self.model.device, dtype=self.model.dtype
                    )
                ).unsqueeze(0)
                proj_spk_text = text_eos
            else:
                proj_spk = torch.empty(
                    0,
                    self.hidden_size,
                    device=self.model.device,
                    dtype=self.model.dtype,
                )
                if self.text_stream or self.multi_text_stream:
                    proj_spk_text = torch.empty(
                        text_stream_ids.shape[1],
                        0,
                        device=self.model.device,
                        dtype=text_stream_ids.dtype,
                    )

            # Text stream (if any)
            if self.text_stream or self.multi_text_stream:
                text_stream_ids_b = text_stream_ids[b, :, :non_pad_len]
                if not inference:
                    text_stream_ids_b = torch.cat([text_stream_ids_b, text_eos], dim=1)

                summed_dsu += embed_layer(text_stream_ids_b).sum(dim=0)

                if self.text_stream:  # ues prompt in label
                    prompt_ids_expanded = prompt_ids_nopad.unsqueeze(0).expand(
                        text_stream_ids_b.shape[0], -1
                    )

                    labels_all_text_streams.append(
                        torch.cat(
                            [proj_spk_text, prompt_ids_expanded, text_stream_ids_b],
                            dim=1,
                        )
                    )

                if self.multi_text_stream:  # will have head without prompt
                    labels_all_text_streams.append(text_stream_ids_b)

            # Concatenate prompt and DSU embeddings

            all_concat_embeds.append(
                torch.cat([proj_spk, prompt_embeds, summed_dsu], dim=0)
            )

        # Pad batch to max sequence length
        padded_batch = pad_sequence(
            all_concat_embeds, batch_first=True, padding_value=0.0
        )

        # Build attention mask
        attention_mask = torch.zeros(
            padded_batch.shape[:2],
            device=padded_batch.device,
            dtype=torch.long,
        )
        for i, seq in enumerate(all_concat_embeds):
            attention_mask[i, : seq.size(0)] = 1

        labels_all_dsu_heads = pad_sequence(
            labels_all_dsu_heads,
            batch_first=True,
            padding_value=self.pad_token_id,
        )  # [B, H, L]

        if labels_all_text_streams:
            labels_all_text_streams = pad_sequence(
                labels_all_text_streams,
                batch_first=True,
                padding_value=self.pad_token_id,
            )
        else:
            labels_all_text_streams = None

        return (
            padded_batch,
            attention_mask,
            labels_all_dsu_heads,
            labels_all_text_streams,
        )

    def get_input_embeds_and_labels_padded(
        self,
        prompt_ids,
        prompt_att_mask,
        dsu_ids,
        text_stream_ids=None,
        spk_emb=None,
        inference=False,
    ):
        self.check_vocab_bounds(prompt_ids, dsu_ids)
        create_eos_token = lambda token_id, ref_tensor: torch.full(
            (ref_tensor.size(1), 1),  # batch_size x 1
            token_id,
            device=ref_tensor.device,
            dtype=ref_tensor.dtype,
        )
        audio_eos = create_eos_token(self.audio_eos_id, dsu_ids)
        text_eos = (
            create_eos_token(self.pad_token_id, text_stream_ids)
            if text_stream_ids is not None
            else None
        )

        embed_layer = self.get_input_embeddings()

        audio_embedding_lookup = self.audio_embeds

        B = prompt_ids.shape[0]

        if dsu_ids is None:
            dsu_ids = torch.full(
                (B, 1, 0),
                self.pad_token_id,
                device=prompt_ids.device,
                dtype=torch.long,
            )

        labels_all_dsu_heads = []
        labels_all_text_streams = []
        input_components = []

        for b in range(B):
            # prompt
            prompt_ids_nopad = prompt_ids[b, prompt_att_mask[b].bool()]
            prompt_embeds = embed_layer(prompt_ids_nopad)

            # DSU
            dsu_pad_mask = dsu_ids[b] != self.pad_token_id
            non_pad_len = dsu_pad_mask.sum(dim=1)[0]
            dsu_ids_b = dsu_ids[b, :, :non_pad_len]

            if not inference:
                dsu_ids_b = torch.cat([dsu_ids_b, audio_eos], dim=1)

            dsu_embeds = torch.stack(
                [
                    audio_embedding_lookup[h](dsu_ids_b[h])  # per head embedding
                    for h in range(self.num_dsu_heads)
                ],
                dim=0,
            )

            summed_dsu = dsu_embeds.sum(dim=0)
            labels_all_dsu_heads.append(dsu_ids_b)

            if self.use_speaker_embedding:

                assert B == len(spk_emb)

                proj_spk = self.speaker_embed_proj(
                    torch.tensor(
                        spk_emb[b], device=self.model.device, dtype=self.model.dtype
                    )
                ).unsqueeze(0)
                proj_spk_text = text_eos
            else:
                proj_spk = torch.empty(
                    0,
                    self.hidden_size,
                    device=self.model.device,
                    dtype=self.model.dtype,
                )
                proj_spk_text = torch.empty(
                    text_stream_ids.shape[1],
                    0,
                    device=self.model.device,
                    dtype=text_stream_ids.dtype,
                )

            # Text stream (if any)
            if self.text_stream or self.multi_text_stream:
                text_stream_ids_b = text_stream_ids[b, :, :non_pad_len]
                if not inference:
                    text_stream_ids_b = torch.cat([text_stream_ids_b, text_eos], dim=1)

                summed_dsu += embed_layer(text_stream_ids_b).sum(dim=0)

                prompt_t = None  # multi_text_stream have head without prompt
                if self.text_stream:  # ues prompt in label
                    prompt_ids_expanded = prompt_ids_nopad.unsqueeze(0).expand(
                        text_stream_ids_b.shape[0], -1
                    )
                    prompt_t = torch.cat([proj_spk_text, prompt_ids_expanded], dim=1)

                labels_all_text_streams.append([prompt_t, text_stream_ids_b])

            # Concatenate prompt and DSU embeddings
            input_components.append(
                [torch.cat([proj_spk, prompt_embeds], dim=0), summed_dsu]
            )

        def pad_attn(sequences, padding="right", pad_value=0.0):
            assert padding in ("left", "right")

            batch_size = len(sequences)
            max_len = max(seq.size(0) for seq in sequences)
            embed_dim = sequences[0].size(1)

            padded = torch.full(
                (batch_size, max_len, embed_dim),
                pad_value,
                dtype=self.model.dtype,
            )
            attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

            for i, seq in enumerate(sequences):
                length = seq.size(0)
                if padding == "right":
                    padded[i, :length] = seq
                    attention_mask[i, :length] = 1
                else:  # left padding
                    padded[i, max_len - length :] = seq
                    attention_mask[i, max_len - length :] = 1

            return padded, attention_mask

        padded_prompts, prompts_att_mask = pad_attn(
            [x for x, _ in input_components], padding="left"
        )
        padded_dsus, dsu_att_mask = pad_attn(
            [x for _, x in input_components], padding="right"
        )
        padded_batch = torch.cat([padded_prompts, padded_dsus], dim=1).to(self.device)
        attention_mask = torch.cat([prompts_att_mask, dsu_att_mask], dim=1).to(
            self.device
        )

        labels_all_dsu_heads = pad_sequence(
            labels_all_dsu_heads,
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side="right",
        )  # [B, H, L]

        if labels_all_text_streams:

            dsu_text = [
                x.transpose(1, 0) for _, x in labels_all_text_streams
            ]  # pad_sequence expects [L, H]
            labels_text_streams = pad_sequence(
                dsu_text,
                batch_first=True,
                padding_value=self.pad_token_id,
                padding_side="right",
            ).transpose(
                2, 1
            )  # [B, H, L]

            prompt_text = [
                x.transpose(1, 0) for x, _ in labels_all_text_streams if x != None
            ]  # pad_sequence expects [L, H]
            if prompt_text != []:
                prompt_text_padded = pad_sequence(
                    prompt_text,
                    batch_first=True,
                    padding_value=self.pad_token_id,
                    padding_side="left",
                ).transpose(
                    2, 1
                )  # [B, H, L]

                labels_text_streams = torch.cat(
                    [prompt_text_padded, labels_text_streams], dim=-1
                )

        else:
            labels_text_streams = None

        assert attention_mask.shape[-1] == padded_batch.shape[1]
        if labels_text_streams is not None:

            assert labels_text_streams.shape[-1] == padded_batch.shape[1]

        return (
            padded_batch,
            attention_mask,
            labels_all_dsu_heads,
            labels_text_streams,
        )

    @torch.no_grad()
    def sample_next_token(self, logits, temperature=1.0, top_k=0, top_p=1.0):
        """
        Apply temperature scaling, top-k filtering, and top-p (nucleus) filtering
        to logits, then return filtered probabilities.
        """
        # temperature

        if temperature != 1.0:
            logits = logits / temperature

        probs = F.softmax(logits, dim=-1)

        # top-k
        if top_k > 0:
            top_k_values, _ = torch.topk(probs, top_k, dim=-1)
            min_top_k = top_k_values[..., -1, None]
            probs = torch.where(probs < min_top_k, torch.zeros_like(probs), probs)

        # top-p
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # mask everything beyond p
            mask = cumulative_probs > top_p
            mask[..., 1:] = mask[..., :-1].clone()  # shift mask right
            mask[..., 0] = False

            sorted_probs = torch.where(
                mask, torch.zeros_like(sorted_probs), sorted_probs
            )
            probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)

        # renormalize
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    @torch.no_grad()
    def get_next_tokens(
        self, B, next_token_logits, do_sample, temperature, top_k, top_p
    ):
        if do_sample:
            probs = self.sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            next_tokens = torch.multinomial(
                probs.reshape([-1, next_token_logits.shape[-1]]), num_samples=1
            ).reshape(next_token_logits.shape[:-1])
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)  # [B, H] if DSU
        return next_tokens

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=4096,
        **kwargs,
    ):

        if self.num_dsus < 1:
            return super().generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                **kwargs,
            )

        past_key_values = DynamicCache()

        do_sample = kwargs.pop("do_sample")
        top_p = kwargs.pop("top_p", 1.0)
        top_k = kwargs.pop("top_k", 0)
        temperature = kwargs.pop("temperature", 1.0)
        use_speaker_sample = kwargs.pop("use_speaker_sample", 0)
        dsu_sample = kwargs.pop("dsu_sample", None)
        text_sample = kwargs.pop("text_sample", None)
        n_delay_audio_stream = kwargs.pop("n_delay_audio_stream", 0)
        n_delay_text_stream = kwargs.pop("n_delay_text_stream", 0)
        talk_to_itself = kwargs.pop("talk_to_itself", True)
        spk_emb = kwargs.pop("spk_emb", None)

        if self.text_stream and text_sample is not None and dsu_sample is None:
            raise ValueError("Need to add dsu sample if you want text sample!")

        if self.text_stream and dsu_sample is not None and text_sample is None:
            raise ValueError(
                "DSU sample is set and using text stream, but no text sample given!"
            )
        if n_delay_text_stream:
            raise NotImplementedError

        # define stop_ids
        tokenizer = kwargs.pop("tokenizer")
        stop_strings = kwargs.pop("stop_strings", [])
        text_stop_ids = [
            tokenizer(stop_str, add_special_tokens=False).input_ids[0]
            for stop_str in stop_strings
        ]
        text_stop_ids.append(self.pad_token_id)
        text_stop_ids = torch.tensor(text_stop_ids, device=input_ids.device)
        audio_stop_ids = torch.tensor([self.audio_eos_id], device=input_ids.device)

        B = dsu_sample.size(0)

        # Initialize per-head DSU sequences
        generated_dsu = torch.full(
            (B, self.num_dsu_heads, max_length),
            self.pad_token_id,
            device=input_ids.device,
            dtype=torch.long,
        )

        generated_text = torch.full(
            (B, self.num_text_streams, max_length),
            self.pad_token_id,
            device=input_ids.device,
            dtype=torch.long,
        )

        duplicate_with_head_rotation = lambda x: torch.cat(
            [x, torch.roll(x, -x.shape[1] // 2, dims=1)], dim=0
        )

        def interleave_halves(x):
            half = x.shape[1] // 2
            out = x.clone()
            out[0::2, half:] = x[1::2, :half].clone()
            out[1::2, half:] = x[0::2, :half].clone()
            return out

        len_context = use_speaker_sample

        # define max_length, start step and last step
        max_length = max_length + len_context
        start_step = len_context
        last_step = -1

        dsu_context = dsu_sample[:, :, :len_context]
        generated_dsu = torch.cat([dsu_context, generated_dsu], dim=-1)

        if talk_to_itself:
            # Duplicate DSU for two-speaker simulation
            generated_dsu = duplicate_with_head_rotation(generated_dsu)

        if self.num_text_streams > 0:
            if talk_to_itself:
                # Take first 2 text streams from context
                text_context = text_sample[:, :2, :len_context].permute(1, 0, 2)
                generated_text = torch.cat([generated_text, generated_text], dim=0)
            else:
                text_context = text_sample[:, : self.num_text_streams, :len_context]

            generated_text = torch.cat([text_context, generated_text], dim=-1)

        for step in range(start_step, max_length):
            # print(f"{step-start_step}/{max_length-use_speaker_sample}")

            dsu_ids_step = generated_dsu[:, :, :step].clone()  # [B, H, L_so_far]
            text_ids_step = generated_text[:, :, :step].clone()

            outputs = self.forward(
                input_ids=input_ids,  # prompt ids
                attention_mask=attention_mask,  # prompt attention mask
                dsu_ids=dsu_ids_step,
                text_stream_ids=text_ids_step,
                inference=True,
                need_loss=False,
                spk_emb=spk_emb,
                past_key_values=past_key_values,
                use_cache=True,
                **kwargs,
            )

            past_key_values = outputs["past_key_values"]
            dsu_logits = outputs["logits"][:, -1, :, :]  # [B, L_total, H,  V]

            if step >= n_delay_audio_stream:
                generated_dsu[:, :, step] = self.get_next_tokens(
                    B, dsu_logits, do_sample, temperature, top_k, top_p
                )
            else:
                generated_dsu[:, :, step] = self.audio_delay_id

            if self.text_stream or self.multi_text_stream:
                ts_logits = outputs["ts_logits"][:, -1, :, :]  # [B, L_total, H,  V]

                generated_text[:, :, step] = self.get_next_tokens(
                    B, ts_logits, do_sample, temperature, top_k, top_p
                )

            if talk_to_itself:
                generated_dsu = interleave_halves(generated_dsu)
                if self.num_text_streams > 1:
                    generated_text = interleave_halves(generated_text)

            # EOS in dsus
            if torch.isin(generated_dsu[:, :, step], audio_stop_ids).any():
                last_step = step + 1
                break

            # EOS in text
            if (
                self.num_text_streams > 1
                and torch.isin(generated_text[:, :, step], text_stop_ids).any()
            ):
                last_step = step + 1
                break

        # never do batch inference
        out_dsu = generated_dsu[0, :, :last_step].tolist()
        out_text = (
            generated_text[:, 0, :last_step].tolist()
            if self.num_text_streams > 0
            else None
        )
        return out_dsu, out_text


class DSULlama(DSUModel, LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(
            config
        )  # This now calls DSUModel.__init__, which calls LlamaForCausalLM.__init__
