import numpy as np
from data_collator_dsu import DSUDataCollator
from datasets import DatasetDict, load_dataset
from dialogue_creation.get_prompt import build_prompt
from dialogue_creation.get_text_stream import adapt_to_text_stream
from dialogue_creation.utils import (
    COLUMNS_TO_REMOVE,
    SKIP_EXAMPLE_DICT_INFERENCE,
    SKIP_EXAMPLE_DICT_TRAIN,
    make_attention_mask,
    prepare_dsu,
)


def load_speech_data(
    model_args, data_args, audio_delay_id, logger, tokenizer, inference=False
):

    # get arguments
    max_length = model_args.max_length
    num_dsus = model_args.num_dsus
    text_stream = model_args.text_stream
    multi_text_stream = model_args.multi_text_stream

    def tokenize_speech(example):
        n_overflow_words = 0
        skip_example = False
        dsu_s, dsu_u, dsu_mono, orig_dsu_length, role_to_speaker_map = prepare_dsu(
            example,
            num_dsus,
            data_args.remove_start_silence,
            max_length,
            inference=inference,
        )

        dsu_ids_list = np.concatenate([dsu_s, dsu_u], axis=0)

        assert all(len(s) == len(dsu_ids_list[0]) for s in dsu_ids_list)

        if text_stream or multi_text_stream:

            dsu_ids_list, stacked_ts_ids, skip_example, n_overflow_words = (
                adapt_to_text_stream(
                    (
                        multi_text_stream if not inference else True
                    ),  # always two text streams in inference for talk to itself
                    dsu_ids_list,
                    data_args.n_delay_audio_stream,
                    data_args.n_delay_text_stream,
                    data_args.word_alignment,
                    orig_dsu_length,
                    example,
                    tokenizer,
                    max_length,
                    audio_delay_id,
                    role_to_speaker_map,
                    add_bc_token=data_args.add_bc_token,
                    add_interrupt_token=data_args.add_interrupt_token,
                    add_counting_tokens=data_args.add_counting_tokens,
                )
            )
            if skip_example:
                return (
                    SKIP_EXAMPLE_DICT_INFERENCE
                    if inference
                    else SKIP_EXAMPLE_DICT_TRAIN
                )  # empty dict

        prompt_system = build_prompt(
            example,
            max_length=max_length,
            orig_dsu_length=orig_dsu_length,
            role_to_speaker_map=role_to_speaker_map,
            use_system_narrative=data_args.use_system_narrative,
            speech=True,
        )

        speaker_embeds = [
            np.array(example["spk_emb_c1"]),
            np.array(example["spk_emb_c2"]),
        ]
        speaker_embed_system = speaker_embeds[role_to_speaker_map["system"]]

        if not inference:
            prompt_tokens = tokenizer(
                prompt_system,
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )["input_ids"]

            prompt_att_mask = make_attention_mask(prompt_tokens, tokenizer.pad_token_id)

            return_dict = {
                "input_ids": prompt_tokens,
                "attention_mask": prompt_att_mask,
                "labels": [-100],  # just a placeholder, will be changed later
                "dsu_ids": dsu_ids_list,
                "text_stream_ids": (
                    stacked_ts_ids if text_stream or multi_text_stream else None
                ),
                "skip_example": skip_example,
                "n_overflow_words": n_overflow_words,
                "spk_emb": speaker_embed_system,
            }

        else:
            prompt_user = build_prompt(
                example,
                max_length=max_length,
                orig_dsu_length=orig_dsu_length,
                role_to_speaker_map={
                    k: 1 - v for k, v in role_to_speaker_map.items()
                },  # change user and system
                use_system_narrative=data_args.use_system_narrative,
                speech=True,
            )
            speaker_embed_user = speaker_embeds[role_to_speaker_map["user"]]

            return_dict = {
                "input_text": prompt_system,  # raw prompt string system
                "prompt_s2": prompt_user,  # raw prompt string user
                "reference_text": dsu_ids_list,  # list of references for each head
                "reference_text_stream": (
                    stacked_ts_ids if text_stream or multi_text_stream else None
                ),  # text stream references
                "skip_example": skip_example,
                "spk_emb1": speaker_embed_system,
                "spk_emb2": speaker_embed_user,
                "n_overflow_words": n_overflow_words,
                "orig_narrative": example["narrative"],
            }

        return return_dict

    logger.info("Preprocessing speech")
    dataset = load_dataset(data_args.speech_path)

    split_keys = ["train", "validation"] if not inference else ["test"]
    tokenized_dataset = DatasetDict()
    for split_key in split_keys:
        data_split = dataset[split_key]

        if data_args.debug or data_args.train_on_subset:
            subset_size = (
                int(len(data_split) * data_args.train_on_subset)
                if data_args.train_on_subset
                else 30
            )
            data_split = data_split.shuffle(seed=42).select(range(subset_size))

        data_split = data_split.map(
            tokenize_speech,
            batched=False,
            load_from_cache_file=False,
        )
        logger.info(
            f"{split_key} dataset size before filtering invalid examples: {len(data_split)}"
        )
        data_split = data_split.filter(lambda x: not x["skip_example"])
        logger.info(
            f"Train dataset after filtering invalid examples: {len(data_split)}"
        )

        avg_overflow = np.mean(data_split["n_overflow_words"])

        logger.info(
            f"Average n_overflow_words per dialgoue in {split_key} dataset: {avg_overflow:.2f}"
        )
        tokenized_dataset[split_key] = data_split

    tokenized_dataset = tokenized_dataset.remove_columns(COLUMNS_TO_REMOVE)

    if not inference:
        data_collator = DSUDataCollator(tokenizer=tokenizer, mlm=False)
        train_dataset, validation_dataset = (
            tokenized_dataset["train"],
            tokenized_dataset["validation"],
        )

        return train_dataset, validation_dataset, data_collator
    else:
        return tokenized_dataset["test"]


def load_data(
    model_args, data_args, audio_delay_id, logger, tokenizer, inference=False
):
    logger.info(f"Loading dataset from {data_args.speech_path}.")
    num_dsus = model_args.num_dsus
    if num_dsus < 1:
        raise ValueError(f"Invalid config: num_dsus must be >= 1 (got {num_dsus}).")

    return load_speech_data(
        model_args,
        data_args,
        audio_delay_id,
        logger,
        tokenizer,
        inference=inference,
    )
