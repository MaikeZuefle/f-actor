import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import re

import numpy as np
from special_tokens import (
    BC_TOKEN,
    EOU,
    EPAD,
    INTER_TOKEN,
    SILENCE_PAD,
    UTTERANCE_PAD,
    WORD_PAD,
)

# Class ids for the undelayed, output-only "event" head: predicts, from
# frame i-1, which of these events (if any) occurs at frame i. Unlike the
# EPAD/BC/INTERRUPT/EOU markers inserted into the text stream, this label is
# never shifted by delay_frames/audio_delay content-wise — it always
# reflects the true, undelayed position of the event so the head stays
# reactive rather than inheriting the text/audio channels' lookahead buffer.
EVENT_NONE = 0
EVENT_EPAD = 1
EVENT_BC = 2
EVENT_INTERRUPT = 3
EVENT_EOU = 4


def validate_word_timestamps(segment):
    words = segment["words"]
    if not words:
        return  # Nothing to validate

    first_word_start = words[0]["start"]
    last_word_end = words[-1]["start"] + words[-1]["dur"]

    if (first_word_start + 0.02) < segment["start_time"] or (last_word_end - 0.02) > segment["end_time"]:
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
    add_epad_token=False,
    add_eou_token=False,
):

    audio_duration = example["utterances"][-1]["end_time"]  # total time of conversation

    frames_per_sec = n_dsu / audio_duration

    silence_pad_id = tokenizer.convert_tokens_to_ids(SILENCE_PAD)
    utterance_pad_id = tokenizer.convert_tokens_to_ids(UTTERANCE_PAD)
    word_pad_id = tokenizer.convert_tokens_to_ids(WORD_PAD)
    epad_id = tokenizer.convert_tokens_to_ids(EPAD)
    eou_id = tokenizer.convert_tokens_to_ids(EOU)
    bc_token_id = tokenizer.convert_tokens_to_ids(BC_TOKEN) if add_bc_token else None
    inter_token_id = (
        tokenizer.convert_tokens_to_ids(INTER_TOKEN) if add_interrupt_token else None
    )
    text_ids = np.full(n_dsu, silence_pad_id, dtype=int)
    event_ids = np.full(n_dsu, EVENT_NONE, dtype=int)

    return_skip_example = lambda: (text_ids, event_ids, True, overflow_words)

    overflow_words = 0
    # tracks the end of the last word placed anywhere in the stream so far
    # (across segments/utterances), so overlap detection and marker-frame
    # placement both see the true prior position rather than resetting per
    # segment. Starts at 1 rather than 0 when a marker frame (EPAD, or a
    # BC/INTER segment marker) may be needed before the very first word, so
    # that word (if it starts at frame 0) is nudged to frame 1, leaving room
    # for the marker, instead of changing the stream-wide delay.
    last_end_idx = 1 if (add_epad_token or add_bc_token or add_interrupt_token) else 0
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

                if utt_type == "bc" and add_bc_token:
                    segment_marker_id = bc_token_id
                elif utt_type == "interrupt" and add_interrupt_token:
                    segment_marker_id = inter_token_id
                else:
                    segment_marker_id = None

                for wi, word_info in enumerate(words):
                    word = word_info["word"].lower()

                    start_idx = int(word_info["start"] * frames_per_sec)
                    end_idx = int(
                        (word_info["start"] + word_info["dur"]) * frames_per_sec
                    )

                    if last_end_idx > start_idx:
                        overflow_words += 1

                    if wi != 0:
                        word = " " + word
                    tokens = tokenizer(word, add_special_tokens=False)["input_ids"]
                    num_tokens = len(tokens)

                    # get new start and end index (if previous tokens too long)
                    start_idx = max(last_end_idx, start_idx)

                    # BC/INTER segment markers supersede EPAD on the segment's
                    # first word, since they already signal "a word is about
                    # to start" (plus the backchannel/interrupt event itself).
                    if wi == 0 and segment_marker_id is not None:
                        marker_id = segment_marker_id
                    elif add_epad_token:
                        marker_id = epad_id
                    else:
                        marker_id = None

                    if marker_id is not None:
                        marker_idx = start_idx - 1
                        if text_ids[marker_idx] in (
                            silence_pad_id,
                            utterance_pad_id,
                            word_pad_id,
                        ):
                            text_ids[marker_idx] = marker_id

                    end_idx = max(start_idx + num_tokens, end_idx)
                    span = end_idx - start_idx

                    if end_idx > len(text_ids):  # overflows over end
                        return return_skip_example()

                    text_ids[start_idx : start_idx + len(tokens)] = tokens
                    if num_tokens < span:
                        text_ids[start_idx + num_tokens : end_idx] = word_pad_id

                    last_end_idx = end_idx

                # mark the end of the utterance/segment at the frame right
                # after its last word's placed tokens, so the model gets an
                # explicit "speaker just finished" cue (the counterpart to
                # EPAD's "a word is about to start"). Unlike EPAD/BC/INTER,
                # EOU is always inserted: if the immediate frame clashes with
                # real content, it falls back to the next frame instead of
                # being dropped.
                if add_eou_token:
                    eou_idx = last_end_idx
                    if (
                        eou_idx < len(text_ids)
                        and text_ids[eou_idx]
                        not in (silence_pad_id, utterance_pad_id, word_pad_id)
                    ):
                        eou_idx += 1
                    if eou_idx < len(text_ids):
                        text_ids[eou_idx] = eou_id
                        last_end_idx = eou_idx + 1

            else:  # do utterance level speech-text alignment
                # No frame-level markers are placed in this mode, so
                # event_ids stays EVENT_NONE throughout — the event head
                # requires word_alignment=True.
                tts_text = segment["tts_text"].lower()
                tts_text = re.sub(r"[.,!?] ", " ", tts_text + " ").strip()

                if utt_type == "bc" and add_bc_token:
                    tts_text = len(tts_text) * BC_TOKEN
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

    # Derive event_ids from the undelayed text_ids via a token-id -> class
    # mapping, before the delay shift below is applied: the event head has
    # no delay, so its label must stay aligned to the true frame at which
    # the event occurs, not the (possibly marker-skipped-on-collision)
    # delayed text-stream position.
    event_map = {
        epad_id: EVENT_EPAD,
        eou_id: EVENT_EOU,
    }
    if bc_token_id is not None:
        event_map[bc_token_id] = EVENT_BC
    if inter_token_id is not None:
        event_map[inter_token_id] = EVENT_INTERRUPT
    event_ids = np.full(n_dsu, EVENT_NONE, dtype=int)
    for token_id, event_class in event_map.items():
        event_ids[text_ids == token_id] = event_class

    # delay speech frame
    if delay_frames != 0:
        text_ids = np.concatenate(
            [
                np.full(delay_frames, silence_pad_id, dtype=int),
                text_ids[:-delay_frames],
            ]
        )
    assert len(text_ids) == n_dsu
    assert len(event_ids) == n_dsu

    text_ids = text_ids[:max_length]
    event_ids = event_ids[:max_length]
    return text_ids, event_ids, False, overflow_words


def add_audio_delay(
    tokenizer,
    audio_delay,
    audio_delay_id,
    dsu_ids_list,
    text_stream_ids,
    event_ids=None,
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

    if event_ids is not None:
        # Padding only (no real event content), so this does not
        # reintroduce delay into the event labels themselves.
        event_pad = np.full((audio_delay), EVENT_NONE, dtype=event_ids.dtype)
        event_ids = np.concatenate([event_ids, event_pad])

    return dsu_ids_list, text_stream_ids, event_ids


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
    add_epad_token=False,
    add_eou_token=False,
):

    min_delay = min(audio_delay, text_delay)
    audio_delay -= min_delay
    text_delay -= min_delay

    num_text_streams = 2 if multi_text_stream else 1
    text_stream_ids_list = []
    event_ids_list = []
    skip_examples = []

    total_overflow_words = 0
    for role in ["system", "user"][:num_text_streams]:
        ts_ids, event_ids, skip, n_overflow_words = create_text_stream(
            orig_dsu_length,
            example,
            tokenizer,
            delay_frames=text_delay,
            max_length=max_length,
            speaker_to_use=role_to_speaker_map[role],
            word_alignment=word_alignment,
            add_bc_token=add_bc_token,
            add_interrupt_token=add_interrupt_token,
            add_epad_token=add_epad_token,
            add_eou_token=add_eou_token,
        )
        total_overflow_words += n_overflow_words

        if audio_delay > 0 and not skip:
            dsu_ids_list_with_delay, ts_ids, event_ids = add_audio_delay(
                tokenizer,
                audio_delay,
                audio_delay_id,
                dsu_ids_list,
                ts_ids,
                event_ids,
            )
        skip_examples.append(skip)
        text_stream_ids_list.append(ts_ids)
        event_ids_list.append(event_ids)

    dsu_ids_list = dsu_ids_list_with_delay if audio_delay and not skip else dsu_ids_list
    skip_example = any(skip_examples)
    stacked_ts_ids = (
        np.stack(text_stream_ids_list, axis=0) if not skip_example else None
    )
    stacked_event_ids = (
        np.stack(event_ids_list, axis=0) if not skip_example else None
    )

    return (
        dsu_ids_list,
        stacked_ts_ids,
        stacked_event_ids,
        skip_example,
        total_overflow_words,
    )
