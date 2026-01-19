import numpy as np

COLUMNS_TO_REMOVE = [
    "file_name",
    "narrative",
    "new_narrative_s1",
    "new_narrative_s2",
    "speakers",
    "behaviors",
    "num_turns",
    "utterances",
    "tts_speaker_ids",
    "tts_genders",
    "statistics",
    "dsu_c1_path",
    "dsu_c2_path",
    "skip_example",
    "n_overflow_words",
]
SKIP_EXAMPLE_DICT_INFERENCE = {
    "input_text": None,
    "reference_text": None,
    "reference_text_stream": None,
    "skip_example": True,
    "spk_emb1": None,
    "spk_emb2": None,
    "prompt_s2": None,
    "n_overflow_words": None,
    "orig_narrative": None,
}

SKIP_EXAMPLE_DICT_TRAIN = {
    "skip_example": True,
    "input_ids": None,
    "attention_mask": None,
    "labels": None,
    "dsu_ids": None,
    "text_stream_ids": None,
    "n_overflow_words": None,
    "spk_emb": None,
}


def get_train_subset(train_dataset, train_on_subset, logger):
    logger.info(f"Only using {train_on_subset*100}% of the training dataset.")
    subset_size = int(len(train_dataset) * train_on_subset)
    train_dataset = train_dataset.shuffle(seed=42).select(range(subset_size))
    return train_dataset


def prepare_dsu(example, num_dsus, remove_start_silence, max_length, inference=False):
    if inference:
        role_to_speaker_map = {"system": 0, "user": 1}
    else:
        if np.random.rand() < 0.5:
            role_to_speaker_map = {"system": 0, "user": 1}
        else:
            role_to_speaker_map = {"system": 1, "user": 0}

    dsu_arrays = np.stack(
        [
            np.array(example["dsu_c1_path"]),
            np.array(example["dsu_c2_path"]),
        ],
        axis=0,
    )

    validate_dsu_shapes(dsu_arrays, num_dsus)

    # get dsu properties
    orig_dsu_length = dsu_arrays.shape[-1]
    dsu_start = get_dsu_start(example, orig_dsu_length) if remove_start_silence else 0

    # preprocess dsu
    dsu_arrays = cut_dsu(dsu_arrays, num_dsus, max_length, dsu_start=dsu_start)

    # assign channels to system or user
    dsu_s = dsu_arrays[role_to_speaker_map["system"]]
    dsu_u = dsu_arrays[role_to_speaker_map["user"]]

    if "dsu_mono_path" in example.keys():
        dsu_mono = np.array(example["dsu_mono_path"])
        dsu_mono = np.expand_dims(dsu_mono, axis=0)
        dsu_mono = cut_dsu(dsu_mono, num_dsus, max_length, dsu_start=dsu_start)[0]
    else:
        dsu_mono = None

    return dsu_s, dsu_u, dsu_mono, orig_dsu_length, role_to_speaker_map


def cut_dsu(dsu, num_dsus, max_length, dsu_start):
    dsu = dsu[:, :, dsu_start:]
    dsu = dsu[:, :num_dsus, :max_length]
    return dsu


def validate_dsu_shapes(dsus, num_dsus):
    assert (
        dsus.shape[1] >= num_dsus
    ), f"Need at least {num_dsus} DSUs, found {dsus.shape[1]}"


def get_dsu_start(example, orig_dsu_length):
    words = example["utterances"][0]["words"]

    if words != []:
        utterance_starts = words[0]["start"]
    else:
        utterance_starts = 0

    audio_duration = example["utterances"][-1]["end_time"]
    frames_per_sec = orig_dsu_length / audio_duration
    dsu_starts = int(frames_per_sec * utterance_starts)
    return dsu_starts


def get_speech_tokens_string(dsus):
    # we do +1 because our tokenizer script does not start from 0
    dsus = dsus[0].tolist()
    return [
        "".join(f"<|AO{int(x)+1}|>" for x in row.tolist()) + "<|EOS|>" for row in dsus
    ]


def make_attention_mask(input_ids, pad_token_id):
    return [1 if tok != pad_token_id else 0 for tok in input_ids]
