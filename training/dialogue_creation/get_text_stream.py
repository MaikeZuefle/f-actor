import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import re

import numpy as np
from special_tokens import (
    BC_COUNTS,
    BC_TOKEN,
    INTER_COUNTS,
    INTER_TOKEN,
    SILENCE_PAD,
    UTTERANCE_PAD,
    WORD_PAD,
)


def validate_word_timestamps(segment):
    words = segment["words"]
    if not words:
        return  # Nothing to validate

    first_word_start = words[0]["start"]
    last_word_end = words[-1]["start"] + words[-1]["dur"]

    if first_word_start < segment["start_time"] or last_word_end > segment["end_time"]:
        raise ValueError(
            f"Word alignment timestamps out of utterance bounds:\n"
            f"Utterance {segment['start_time']}–{segment['end_time']} | "
            f"Words {first_word_start}–{last_word_end}"
        )


def create_text_stream(
    n_dsu,
    example,
    tokenizer,
    speaker_to_use=0,
    delay_frames=2,
    max_length=4096,
    word_alignment=False,
    add_bc_token=False,
    add_interrupt_token=False,
    add_counting_tokens=False,
):

    audio_duration = example["utterances"][-1]["end_time"]  # total time of conversation

    frames_per_sec = n_dsu / audio_duration

    silence_pad_id = tokenizer.convert_tokens_to_ids(SILENCE_PAD)
    utterance_pad_id = tokenizer.convert_tokens_to_ids(UTTERANCE_PAD)
    word_pad_id = tokenizer.convert_tokens_to_ids(WORD_PAD)
    text_ids = np.full(n_dsu, silence_pad_id, dtype=int)

    return_skip_example = lambda: (text_ids, True, overflow_words)

    inter_bc_count = {0: {"inter": 0, "bc": 0}, 1: {"inter": 0, "bc": 0}}

    overflow_words = 0
    for utt in example["utterances"]:

        if utt["speaker_idx"] == speaker_to_use:
            segments = [utt]
            if utt["uttr_type"] == "interrupt":
                utt_type = "interrupt"
            else:
                utt_type = "normal"

        else:
            segments = utt.get("backchannels", [])
            utt_type = "bc"

        for segment in segments:
            words = segment["words"]
            if words == []:
                return return_skip_example()

            if word_alignment:

                validate_word_timestamps(segment)

                start_u_idx = int(words[0]["start"] * frames_per_sec)
                end_u_idx = int(
                    (words[-1]["start"] + words[-1]["dur"]) * frames_per_sec
                )

                # Fill the whole utterance region first with UTTERANCE_PAD
                text_ids[start_u_idx:end_u_idx] = utterance_pad_id
                last_end_idx = 0

                for wi, word_info in enumerate(words):
                    word = word_info["word"].lower()

                    start_idx = int(word_info["start"] * frames_per_sec)
                    end_idx = int(
                        (word_info["start"] + word_info["dur"]) * frames_per_sec
                    )
                    orig_dsu_span = end_idx - start_idx

                    if last_end_idx > start_idx:
                        overflow_words += 1

                    # get tokens
                    if add_counting_tokens:
                        if utt_type == "bc":
                            if wi == 0:
                                inter_bc_count[utt["speaker_idx"]]["bc"] += 1
                                current_bcs = inter_bc_count[utt["speaker_idx"]]["bc"]
                                word = f"{BC_COUNTS[current_bcs]} {word}"

                            else:
                                word = " " + word

                            # current_bcs = inter_bc_count[utt["speaker_idx"]]["bc"]
                            # word = BC_COUNTS[current_bcs] * (orig_dsu_span)

                        elif utt_type == "interrupt":

                            if wi == 0:
                                inter_bc_count[utt["speaker_idx"]]["inter"] += 1
                                current_inter = inter_bc_count[utt["speaker_idx"]][
                                    "inter"
                                ]
                                word = f"{INTER_COUNTS[current_inter]} {word}"

                            else:
                                word = " " + word
                        else:
                            if wi != 0:
                                word = " " + word
                    else:
                        if utt_type == "bc" and add_bc_token:
                            # word = BC_TOKEN * (orig_dsu_span)
                            if wi == 0:
                                word = f"{BC_TOKEN} {word}"
                            else:
                                word = " " + word
                        elif utt_type == "interrupt" and add_interrupt_token:
                            if wi == 0:
                                word = f"{INTER_TOKEN} {word}"
                            else:
                                word = " " + word
                        else:
                            if wi != 0:
                                word = " " + word
                    tokens = tokenizer(word, add_special_tokens=False)["input_ids"]
                    num_tokens = len(tokens)

                    # get new start and end index (if previous tokens too long)
                    start_idx = max(last_end_idx, start_idx)
                    end_idx = max(start_idx + num_tokens, end_idx)
                    span = end_idx - start_idx

                    if end_idx > len(text_ids):  # overflows over end
                        return return_skip_example()

                    text_ids[start_idx : start_idx + len(tokens)] = tokens
                    if num_tokens < span:
                        text_ids[start_idx + num_tokens : end_idx] = word_pad_id

                    last_end_idx = end_idx

            else:  # do utterance level speech-text alignment
                tts_text = segment["tts_text"].lower()
                tts_text = re.sub(r"[.,!?] ", " ", tts_text + " ").strip()

                if add_counting_tokens:
                    if utt_type == "bc":
                        inter_bc_count[utt["speaker_idx"]]["bc"] += 1
                        current_bcs = inter_bc_count[utt["speaker_idx"]]["bc"]
                        tts_text = BC_COUNTS[current_bcs] * len(tts_text)

                    elif utt_type == "interrupt":
                        inter_bc_count[utt["speaker_idx"]]["inter"] += 1
                        current_inter = inter_bc_count[utt["speaker_idx"]]["inter"]
                        tts_text = f"{INTER_COUNTS[current_inter]} {tts_text}"

                elif utt_type == "bc" and add_bc_token:
                    tts_text = len(tts_text) * BC_TOKEN
                    # tts_text = f"{BC_TOKEN}{tts_text}"
                elif utt_type == "interrupt" and add_interrupt_token:
                    tts_text = f"{INTER_TOKEN} {tts_text}"

                tokens = tokenizer(tts_text, add_special_tokens=False)["input_ids"]

                start_idx = int(words[0]["start"] * frames_per_sec)
                end_idx = int((words[-1]["start"] + words[-1]["dur"]) * frames_per_sec)

                span = end_idx - start_idx
                text_ids[start_idx:end_idx] = utterance_pad_id

                if len(tokens) > span:
                    return return_skip_example()

                text_ids[start_idx : start_idx + len(tokens)] = tokens

    # delay speech frame
    if delay_frames != 0:
        text_ids = np.concatenate(
            [
                np.full(delay_frames, silence_pad_id, dtype=int),
                text_ids[:-delay_frames],
            ]
        )
    assert len(text_ids) == n_dsu

    text_ids = text_ids[:max_length]
    return text_ids, False, overflow_words


def add_audio_delay(
    tokenizer,
    audio_delay,
    audio_delay_id,
    dsu_ids_list,
    text_stream_ids,
):
    delay_id = audio_delay_id
    delays_audio = np.full(
        (dsu_ids_list.shape[0], audio_delay),
        delay_id,
        dtype=dsu_ids_list.dtype,
    )

    dsu_ids_list = np.concatenate([delays_audio, dsu_ids_list], axis=-1)

    silence_pad_id = tokenizer.convert_tokens_to_ids(SILENCE_PAD)
    silence_text = np.full((audio_delay), silence_pad_id)
    text_stream_ids = np.concatenate([text_stream_ids, silence_text])
    return dsu_ids_list, text_stream_ids


def adapt_to_text_stream(
    multi_text_stream,
    dsu_ids_list,
    audio_delay,
    text_delay,
    word_alignment,
    orig_dsu_length,
    example,
    tokenizer,
    max_length,
    audio_delay_id,
    role_to_speaker_map,
    add_bc_token=False,
    add_interrupt_token=False,
    add_counting_tokens=False,
):

    min_delay = min(audio_delay, text_delay)
    audio_delay -= min_delay
    text_delay -= min_delay

    num_text_streams = 2 if multi_text_stream else 1
    text_stream_ids_list = []
    skip_examples = []

    total_overflow_words = 0
    for role in ["system", "user"][:num_text_streams]:
        ts_ids, skip, n_overflow_words = create_text_stream(
            orig_dsu_length,
            example,
            tokenizer,
            delay_frames=text_delay,
            max_length=max_length,
            speaker_to_use=role_to_speaker_map[role],
            word_alignment=word_alignment,
            add_bc_token=add_bc_token,
            add_interrupt_token=add_interrupt_token,
            add_counting_tokens=add_counting_tokens,
        )
        total_overflow_words += n_overflow_words

        if audio_delay > 0 and not skip:
            dsu_ids_list_with_delay, ts_ids = add_audio_delay(
                tokenizer,
                audio_delay,
                audio_delay_id,
                dsu_ids_list,
                ts_ids,
            )
        skip_examples.append(skip)
        text_stream_ids_list.append(ts_ids)

    dsu_ids_list = dsu_ids_list_with_delay if audio_delay and not skip else dsu_ids_list
    skip_example = any(skip_examples)
    stacked_ts_ids = (
        np.stack(text_stream_ids_list, axis=0) if not skip_example else None
    )

    return dsu_ids_list, stacked_ts_ids, skip_example, total_overflow_words
