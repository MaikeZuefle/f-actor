import torch
from transformers import DataCollatorForLanguageModeling


class DSUDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        """
        A custom data collator that extends Hugging Face's DataCollatorForLanguageModeling
        to handle additional hierarchical DSU (Discrete Symbol Unit) token sequences and optional
        text streams.

        This collator performs the following:
        1. Pads standard fields (`input_ids`, `attention_mask`, `labels`) using the parent collator.
        2. Pads DSU sequences across the batch. Each example may have multiple DSUs, assumed
        to have the same length within a single example but possibly different lengths across examples.
        The padded DSU tensor has shape `(batch_size, num_quantizers, max_dsu_length_in_batch)`.
        3. Handles optional text streams (`text_stream`) and their attention masks
        (`text_stream_attention_mask`).

        Args:
            features (List[Dict[str, Any]]): A list of dataset examples, where each example is a dict containing:
                - 'input_ids': List[int]
                - 'attention_mask': List[int]
                - 'labels': List[int]
                - 'dsu_ids': List[List[int]] (DSU sequences)
                - Optional: 'text_stream_ids': np.ndarray or List[List[int]]

        Returns:
            Dict[str, torch.Tensor]: A batch dictionary containing:
                - Padded 'input_ids', 'attention_mask', 'labels' (handled by parent collator)
                - 'dsu_ids': torch.Tensor of shape (batch_size, num_quantizers, max_dsu_length)
                - Optional 'text_stream_ids' as tensors
        """

        # extract these so parent collator works
        all_dsu_ids = [f.pop("dsu_ids") for f in features]
        all_text_streams = [f.pop("text_stream_ids", None) for f in features]
        spk_emb = [f.pop("spk_emb", None) for f in features]

        # First, pad input_ids and other standard fields using parent collator
        # breakpoint()
        batch = super().__call__(features)
        if all(x is False for x in all_dsu_ids):
            return batch

        batch["spk_emb"] = spk_emb
        # process text and dsustreams
        batch_size = len(features)
        streams = [
            (
                all_dsu_ids,
                len(all_dsu_ids[0]),
                "dsu_ids",
            ),  # (data, num_streams, batch_key)
        ]

        if all_text_streams[0] is not None:
            streams.append(
                (all_text_streams, len(all_text_streams[0]), "text_stream_ids")
            )

        for data_list, num_streams, batch_key in streams:
            # compute max length across batch for this stream type
            max_len = max(len(sample[0]) for sample in data_list)

            # allocate padded tensor
            padded_tensor = torch.full(
                (batch_size, num_streams, max_len),
                fill_value=self.tokenizer.pad_token_id,
                dtype=torch.long,
            )

            # fill tensor
            for i, example_streams in enumerate(data_list):
                for idx, seq in enumerate(example_streams):
                    if seq is None:
                        continue
                    padded_tensor[i, idx, : len(seq)] = torch.tensor(seq)

            # assign to batch
            batch[batch_key] = padded_tensor

        return batch
