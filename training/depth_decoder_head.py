import torch
import torch.nn as nn
from transformers import DynamicCache


class CsmDepthDecoderHead(nn.Module):
    """
    Alternative to the flat multi-head `dsu_head`: predicts the semantic (first)
    Mimi codebook with a small trainable linear head, then predicts the remaining
    acoustic codebooks with Sesame CSM's pretrained depth decoder, which is kept
    entirely frozen. A trainable adapter projects our backbone's hidden state into
    the depth decoder's expected input space, since our backbone was never jointly
    trained with it.

    This module only ever predicts a single speaker's `num_dsus` codebooks - it is
    only meant to be used for the first speaker's stream (see
    `DSUModel.calc_loss_on_c1_only` requirement in modeling_dsu.py), not this
    repo's dual-speaker "talk to itself" setup.

    Confirmed I/O contract from `transformers`' `CsmDepthDecoderModel.forward`
    (`codebook_idxs = clamp(position_ids - 1, min=0)`, `offset = codebook_idxs *
    vocab_size`, `inputs_embeds = embed_tokens(input_ids + offset)`, then
    `inputs_embeds[:, 0] = backbone_last_hidden_state`): position 0 is a
    throwaway placeholder (its embedding is immediately overwritten by the
    backbone hidden state); position k (k>=1) must hold codebook (k-1)'s REAL id,
    embedded via codebook (k-1)'s own embedding sub-table, and predicts codebook
    k's logits (via `codebooks_head`). So to predict codebooks 1..num_dsus-1 (a
    length num_dsus-1 sequence), position 1 holds the semantic/codebook-0 id,
    position 2 holds codebook-1's id, etc. - the semantic id IS fed into the
    depth decoder (just not as an output target, since our own semantic_head
    predicts it separately).
    """

    def __init__(
        self,
        hidden_size,
        audio_vocab_size,
        num_dsus,
        pretrained_path="sesame/csm-1b",
    ):
        super().__init__()

        from transformers import CsmForConditionalGeneration

        self.num_dsus = num_dsus
        self.audio_vocab_size = audio_vocab_size

        self.semantic_head = nn.Linear(hidden_size, audio_vocab_size)

        # `sesame/csm-1b` is only distributed as the full CsmForConditionalGeneration
        # checkpoint (backbone + depth decoder together) - there is no standalone
        # depth-decoder checkpoint/config on the Hub, so we load the full model and
        # keep only its `.depth_decoder` submodule (which carries its own correctly
        # nested CsmDepthDecoderConfig), discarding the backbone/text/codec parts.
        full_model = CsmForConditionalGeneration.from_pretrained(pretrained_path)
        self.depth_decoder = full_model.depth_decoder
        del full_model

        depth_config = self.depth_decoder.config

        if depth_config.vocab_size != audio_vocab_size:
            # audio_vocab_size here already includes the +2 reserved ids DSUModel
            # adds for its own audio_eos_id/audio_delay_id (see modeling_dsu.py),
            # so config.audio_vocab_size (the ModelArgs/config field) must be set to
            # exactly `depth_config.vocab_size - 2`.
            required_config_audio_vocab_size = depth_config.vocab_size - 2
            raise ValueError(
                f"audio_vocab_size ({audio_vocab_size}) must match the pretrained "
                f"depth decoder's vocab_size ({depth_config.vocab_size}). Set "
                f"config.audio_vocab_size = {required_config_audio_vocab_size} "
                "(the +2 for audio_eos_id/audio_delay_id is added automatically) - "
                "this assumes your Mimi-encoded DSU data only contains raw codebook "
                f"ids in [0, {required_config_audio_vocab_size})."
            )
        if num_dsus > depth_config.num_codebooks:
            raise ValueError(
                f"num_dsus ({num_dsus}) exceeds the pretrained depth decoder's "
                f"num_codebooks ({depth_config.num_codebooks})."
            )

        for p in self.depth_decoder.parameters():
            p.requires_grad_(False)
        self.depth_decoder.eval()

        # this repo trains with gradient_checkpointing=True, and Trainer/HF
        # recursively calls gradient_checkpointing_enable() on every nested
        # PreTrainedModel it finds - including this frozen depth_decoder, since it
        # is itself a full CsmDepthDecoderForCausalLM. That call registers a forward
        # hook (enable_input_require_grads) that forces the embedding *output* to
        # require grad even though the underlying weight is frozen, turning it into
        # a leaf tensor with requires_grad=True - which then crashes on CSM's own
        # in-place `inputs_embeds[:, 0] = backbone_last_hidden_state`. Make the
        # frozen depth decoder immune to this recursive (re-)enabling.
        self.depth_decoder.gradient_checkpointing_enable = lambda *a, **k: None
        self.depth_decoder.gradient_checkpointing_disable = lambda *a, **k: None

        self.backbone_adapter = nn.Linear(
            hidden_size, depth_config.backbone_hidden_size, bias=False
        )

    def _build_inputs_embeds(self, input_ids, context):
        """
        Replicates CsmDepthDecoderModel.forward's own
        `embed_tokens(input_ids + codebook_idx * vocab_size)` +
        `inputs_embeds[:, 0] = backbone_last_hidden_state` construction, but via
        `torch.cat` instead of an in-place write, so we can pass the result as
        `inputs_embeds=` directly (skipping their `input_ids`/
        `backbone_last_hidden_state` code path entirely, which crashes with
        `RuntimeError: a view of a leaf Variable that requires grad is being used
        in an in-place operation` in this training setup - gradient checkpointing
        plus DeepSpeed appears to force `embed_tokens`'s output to require grad
        even though its weight is frozen, which is fine for a fresh `cat` but not
        for their in-place assignment).

        input_ids: [N, L] token ids (position 0 is a placeholder, ignored/replaced).
        context: [N, backbone_hidden_size] already-projected backbone hidden state.
        """
        embed_tokens = self.depth_decoder.model.embed_tokens
        L = input_ids.shape[1]
        position_ids = torch.arange(L, device=input_ids.device)
        codebook_idxs = torch.clamp(position_ids - 1, min=0)
        offset = (codebook_idxs * self.audio_vocab_size).unsqueeze(0)
        raw_embeds = embed_tokens(input_ids + offset)  # [N, L, backbone_hidden_size]
        return torch.cat([context.unsqueeze(1), raw_embeds[:, 1:, :]], dim=1)

    def train(self, mode=True):
        super().train(mode)
        self.depth_decoder.eval()  # never let Model.train() unfreeze/un-eval the frozen decoder
        return self

    def semantic_logits(self, hidden_states):
        return self.semantic_head(hidden_states)  # [B, L, V]

    def forward(self, hidden_states, dsu_labels):
        """
        Teacher-forced training/eval path, first speaker only. Only meant to be
        called when a loss is actually needed - see `generate()` for the
        inference/sampling path.

        hidden_states: [B, L, hidden_size] backbone per-frame hidden states,
            where hidden_states[:, l] predicts frame (l+1)'s codebooks (standard
            next-frame/causal-LM shift, applied by the caller).
        dsu_labels: [B, num_dsus, L] ALREADY time-shifted so that dsu_labels[:, :, l]
            holds frame (l+1)'s own ground-truth codebook ids - i.e. the exact
            same `labels_shifted` tensor the caller's loss loop builds as the
            *target* for this method's output, reused here as this frame's
            *within-frame* teacher-forcing input (codebook (k-1)'s real id at
            depth-decoder position k - see class docstring). Passing the raw,
            unshifted per-frame labels here would condition each prediction on
            the wrong frame's codebooks.

        Returns logits of shape [B, L, num_dsus, audio_vocab_size], drop-in
        compatible with the existing `dsu_head(...).view(...)` output (restricted
        to the first speaker) so the rest of the loss computation in
        modeling_dsu.py doesn't need to change beyond also restricting to c1.
        """
        B, L, _ = hidden_states.shape
        assert dsu_labels.shape[1] == self.num_dsus, (
            f"expected dsu_labels for exactly {self.num_dsus} codebooks (first "
            f"speaker only), got {dsu_labels.shape[1]}"
        )

        semantic_logits = self.semantic_logits(hidden_states)  # [B, L, V]

        if self.num_dsus > 1:
            # batch/end-of-sequence padding ids (e.g. the tokenizer's pad/eos id, a
            # much larger id from the main LLM vocab) live outside the audio codebook
            # vocab entirely and are only ever used as *targets* (masked out by the
            # caller's loss weighting) - never as embedding lookup indices here.
            #
            # Our own DSUModel.audio_delay_id/audio_eos_id (= audio_vocab_size - 2/-1)
            # are legitimately < audio_vocab_size, so they pass through this clamp
            # unchanged and DO get embedded/predicted through the frozen depth decoder.
            # Because audio_vocab_size is required to equal the depth decoder's own
            # vocab_size (2051 for CSM), these land exactly on ids 2049/2050 - CSM's
            # own reserved top-of-vocab slots (2050 = codebook_pad_token_id, 2049 =
            # undocumented reserved headroom), never on a real trained Mimi code
            # (0-2047). CSM's own codebook_eos_token_id (=0, an ordinary Mimi code
            # reused as a whole-frame stop marker) and codebook_pad_token_id are a
            # separate convention we don't rely on - we only need our own reserved
            # ids to avoid colliding with real acoustic content, which this guarantees.
            labels = torch.where(
                dsu_labels < self.audio_vocab_size,
                dsu_labels,
                torch.zeros_like(dsu_labels),
            )

            projected = self.backbone_adapter(hidden_states)  # [B, L, backbone_hidden]
            context = projected.reshape(B * L, -1)

            # position 0 = placeholder (overwritten by the hidden state below, and
            # never predicted from - CsmDepthDecoderForCausalLM's default
            # logits_to_keep=0 drops position 0's logits entirely, since it only
            # ever seeds attention context); positions 1..num_dsus-1 = codebooks
            # 0..num_dsus-2's real ids, so that position k (k>=1) holds codebook
            # (k-1) - see class docstring. Total input length num_dsus yields
            # num_dsus-1 output logits (codebooks 1..num_dsus-1), as needed.
            placeholder = torch.zeros(B, 1, L, dtype=labels.dtype, device=labels.device)
            real_codebooks = labels[:, : self.num_dsus - 1, :]  # codebooks 0..num_dsus-2
            acoustic_input_ids = torch.cat([placeholder, real_codebooks], dim=1)
            acoustic_input_ids = acoustic_input_ids.permute(0, 2, 1).reshape(
                B * L, self.num_dsus
            )

            inputs_embeds = self._build_inputs_embeds(acoustic_input_ids, context)
            depth_outputs = self.depth_decoder(inputs_embeds=inputs_embeds)
            acoustic_logits = depth_outputs.logits.view(
                B, L, self.num_dsus - 1, self.audio_vocab_size
            )
        else:
            acoustic_logits = hidden_states.new_zeros(
                B, L, 0, self.audio_vocab_size
            )

        dsu_logits = torch.cat(
            [semantic_logits.unsqueeze(2), acoustic_logits], dim=2
        )  # [B, L, num_dsus, V]
        assert dsu_logits.shape[2] == self.num_dsus
        return dsu_logits

    @torch.no_grad()
    def generate(self, hidden_state_last, sample_fn):
        """
        Inference-time path for one frame, first speaker only: samples ALL
        `num_dsus` codebooks - first the semantic (codebook 0) id via
        `semantic_head` + `sample_fn`, then the acoustic codebooks 1..num_dsus-1
        autoregressively from the frozen depth decoder, conditioned on the
        semantic id and the backbone hidden state.

        hidden_state_last: [B, hidden_size] backbone hidden state for this frame.
        sample_fn: callable(logits) -> next_token_ids, e.g. wrapping
            DSUModel.get_next_tokens with the desired sampling params.

        Returns ids of shape [B, num_dsus].
        """
        semantic_logits = self.semantic_logits(hidden_state_last)  # [B, V]
        semantic_id = sample_fn(semantic_logits).view(-1, 1)  # [B, 1]

        if self.num_dsus <= 1:
            return semantic_id

        acoustic_ids = self._generate_acoustic_ids(
            hidden_state_last, semantic_id.squeeze(1), sample_fn
        )
        return torch.cat([semantic_id, acoustic_ids], dim=1)

    @torch.no_grad()
    def _seed(self, context, past_key_values):
        """
        Seed the KV cache with the backbone context as the depth decoder's
        position-0 embedding. Passed via `inputs_embeds=` rather than
        `input_ids=`/`backbone_last_hidden_state=`, which bypasses
        `embed_tokens` entirely - there is no token id involved at all, so no
        placeholder is needed. Its logits are never used (nothing has been
        predicted yet); `logits_to_keep=1` is still passed (rather than the
        default `0`, which would slice out an empty range for this
        length-1 call and crash `codebooks_head` on an empty stack) even
        though the resulting single logits row is discarded.

        Returns the updated past_key_values.
        """
        outputs = self.depth_decoder(
            inputs_embeds=context.unsqueeze(1),
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        return outputs.past_key_values

    @torch.no_grad()
    def _step(self, input_ids, past_key_values):
        """
        One incremental depth-decoder call: exactly one new token in (plus the
        running KV cache), exactly one logits row out. Lets the frozen
        `depth_decoder` do its own embedding lookup (`input_ids=`) - unsafe
        under the training-time gradient-checkpointing/DeepSpeed combination
        for the in-place position-0 write (see `_build_inputs_embeds`
        docstring), but that write never happens here (no
        `backbone_last_hidden_state` passed - see `_seed`), and generation runs
        under `@torch.no_grad()` regardless, so there is no autograd graph to
        break.

        Absolute position (and thus which codebook's embedding sub-table /
        output head to use) is derived by the model itself from
        `past_key_values`'s current length, so callers just need to call this
        once per position in order - no manual offset bookkeeping needed.

        `logits_to_keep=1` (rather than the model's default of `0`) is required
        here: the default assumes a single full-sequence call and drops local
        position 0 unconditionally (right for that case, since 0 is always the
        placeholder there) - but every one of these calls has exactly one
        *real* position at local index 0, which would otherwise be silently
        dropped.

        Returns (next_logits [B, V], updated past_key_values).
        """
        outputs = self.depth_decoder(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        return outputs.logits[:, -1, :], outputs.past_key_values

    @torch.no_grad()
    def _generate_acoustic_ids(self, hidden_state_last, semantic_id, sample_fn):
        """
        Autoregressively sample the acoustic codebooks 1..num_dsus-1 from the
        frozen depth decoder, conditioned on the backbone hidden state and the
        already-sampled semantic (codebook-0) id, which - per the class
        docstring - the depth decoder does consume as its position-1 input (just
        not as one of its own prediction targets, since our own semantic_head
        predicts it separately).

        Uses a fresh `DynamicCache` per frame and `_step` to feed exactly one
        new token per call: first a seed call carrying the backbone context
        directly (no zero-padded placeholder re-embedded on every step), then
        one call per remaining codebook, each attending over the cache instead
        of recomputing the whole growing prefix.

        hidden_state_last: [B, hidden_size] backbone hidden state for this frame.
        semantic_id: [B] (or [B, 1]) already-sampled codebook-0 id.
        sample_fn: callable(logits) -> next_token_ids.

        Returns acoustic ids of shape [B, num_dsus - 1].
        """
        B = hidden_state_last.shape[0]
        device = hidden_state_last.device

        if self.num_dsus <= 1:
            return torch.zeros(B, 0, dtype=torch.long, device=device)

        context = self.backbone_adapter(hidden_state_last)  # [B, backbone_hidden]
        past_key_values = DynamicCache()
        past_key_values = self._seed(context, past_key_values)

        next_id = semantic_id.view(B, 1)
        acoustic_ids = []
        for i in range(self.num_dsus - 1):
            next_logits, past_key_values = self._step(next_id, past_key_values)
            next_id = sample_fn(next_logits).view(B, 1)
            acoustic_ids.append(next_id)

        generated = torch.cat(acoustic_ids, dim=1)
        assert generated.shape[1] == self.num_dsus - 1
        return generated
