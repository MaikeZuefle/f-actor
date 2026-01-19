import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import re

import numpy as np
from bc_interrupt.get_word_timestamps import get_parakeet_timestps
from scipy.stats import pearsonr
from tqdm import tqdm
from utils import get_soda_index, get_wav_pairs, load_asr_model


def get_segments(c1_speech_segments, c2_speech_segments):
    segments = []
    for channel_idx, timestamps in enumerate([c1_speech_segments, c2_speech_segments]):
        for entry in timestamps:
            start = entry["start"]
            end = entry["end"]
            channel = f"c{channel_idx+1}"
            text = entry.get("segment", "")
            segments.append((start, end, channel, text))
    return segments


def segment_words(word_timestamps):
    segments = []
    for channel in ["c1", "c2"]:
        last_end = -1
        segment = []

        for word in word_timestamps[channel]:
            if (
                word["start"] - last_end > UTTERANCE_SPLIT_THRESHOLD
                and len(segment) > 0
            ):
                start = segment[0]["start"]
                end = segment[-1]["end"]
                text = " ".join(w.get("word", w.get("text", "")) for w in segment)
                segments.append((start, end, channel, text))
                segment = []

            segment.append(word)
            last_end = word["end"]

        if len(segment) > 0:
            start = segment[0]["start"]
            end = segment[-1]["end"]
            text = " ".join(w.get("word", w.get("text", "")) for w in segment)
            segments.append((start, end, channel, text))

    return segments


def label_segments(segments):
    labelled_segments = []
    last_end = {"c1": 0, "c2": 0}
    other = {"c1": "c2", "c2": "c1"}
    for start, end, channel, text in sorted(segments, key=lambda x: x[0]):
        last_other_end = last_end[other[channel]]

        if start < last_other_end and end < last_other_end:
            labelled_segments.append((start, end, channel, text, "#BACKCHANNEL"))
        elif start < last_other_end - INTERRUPTION_THRESHOLD and last_other_end < end:
            labelled_segments.append((start, end, channel, text, "#INTERRUPTION"))
        else:
            labelled_segments.append((start, end, channel, text, ""))

        last_end[channel] = end

    return labelled_segments


def overlaps_with_any(segment, segments):
    for other_segment in segments:
        if other_segment[2] != segment[2]:
            continue

        if other_segment[4] != segment[4]:
            continue

        if (
            other_segment[0] - OVERLAP_TOLERANCE < segment[0]
            and segment[0] < other_segment[1] + OVERLAP_TOLERANCE
        ):
            return True

        if (
            other_segment[0] - OVERLAP_TOLERANCE < segment[1]
            and segment[1] < other_segment[1] + OVERLAP_TOLERANCE
        ):
            return True

    return False


def compare_gold_pred(segments, gold_stats, overall_stats):

    pred_stats = {
        "c1_#BACKCHANNEL": 0,
        "c2_#BACKCHANNEL": 0,
        "c1_#INTERRUPTION": 0,
        "c2_#INTERRUPTION": 0,
    }

    assert pred_stats.keys() == gold_stats.keys()

    for segment in segments:
        if not segment[4]:
            continue
        channel = segment[2]
        pred_stats[f"{channel}_{segment[4]}"] += 1

    example_stats = {}

    for category, pred in pred_stats.items():
        channel, label = category.split("_")
        gold = gold_stats[category]

        match = min(pred, gold)
        deviation = abs(pred - gold)

        # update global stats
        overall_stats[f"matching_{channel}"][label] += match
        overall_stats[f"deviation_{channel}"][label] += deviation

        # save per-example
        example_stats[category] = {
            "pred": pred,
            "gold": gold,
            "match": match,
            "deviation": deviation,
            "missing": max(0, gold - pred),
            "extra": max(0, pred - gold),
        }

    return overall_stats, example_stats


def get_bc_inter_from_prompt(prompt):

    backchannels = int(re.search(r"backchannels:\s*(\d+)", prompt).group(1))
    interruptions = int(re.search(r"interruptions:\s*(\d+)", prompt).group(1))
    return backchannels, interruptions


def main(wavs_dir, reference_path, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    # define hyperparams
    global UTTERANCE_SPLIT_THRESHOLD, INTERRUPTION_THRESHOLD, OVERLAP_TOLERANCE
    UTTERANCE_SPLIT_THRESHOLD = 0.565
    INTERRUPTION_THRESHOLD = 0.405
    OVERLAP_TOLERANCE = 0.435

    # load data
    wav_pairs = get_wav_pairs(wavs_dir)
    with open(reference_path, "r") as f:
        reference_utterances = json.load(f)
        reference_utterances = {x["soda_index"]: x for x in reference_utterances}

    # Load model
    print("Loading model...")
    parakeet_model = load_asr_model()

    # defining outputs
    overall_stats = {
        "matching_c1": {"#BACKCHANNEL": 0, "#INTERRUPTION": 0},
        "deviation_c1": {"#BACKCHANNEL": 0, "#INTERRUPTION": 0},
        "matching_c2": {"#BACKCHANNEL": 0, "#INTERRUPTION": 0},
        "deviation_c2": {"#BACKCHANNEL": 0, "#INTERRUPTION": 0},
    }

    per_example_stats = []
    # scoring segments
    print(f"Processing {len(wav_pairs)} dialogue pairs...")

    for ch1_path, ch2_path in tqdm(wav_pairs):
        soda_index = get_soda_index([ch1_path])

        # get gold stats
        s1_bc, s1_inter = get_bc_inter_from_prompt(
            reference_utterances[soda_index]["instruction_s1"]
        )
        s2_bc, s2_inter = get_bc_inter_from_prompt(
            reference_utterances[soda_index]["instruction_s2"]
        )
        gold_stats = {
            "c1_#BACKCHANNEL": s1_bc,
            "c2_#BACKCHANNEL": s2_bc,
            "c1_#INTERRUPTION": s1_inter,
            "c2_#INTERRUPTION": s2_inter,
        }

        # get word timestamps
        c1_speech_words = get_parakeet_timestps(
            ch1_path, parakeet_model, return_words_tmsp=True
        )
        c2_speech_words = get_parakeet_timestps(
            ch2_path, parakeet_model, return_words_tmsp=True
        )
        word_timestamps = {"c1": c1_speech_words, "c2": c2_speech_words}

        # merge to segments
        segments = segment_words(word_timestamps)

        # label bc and inter
        labelled_segments_words = label_segments(segments)

        # compare to gold
        overall_stats, example_stats = compare_gold_pred(
            labelled_segments_words, gold_stats, overall_stats
        )

        per_example_stats.append({"soda_index": soda_index, "stats": example_stats})
    # compute means & stds
    categories = next(iter(per_example_stats))["stats"].keys()
    mean_std_stats = {}

    for category in categories:
        deviations = [e["stats"][category]["deviation"] for e in per_example_stats]
        match_percents = []
        for e in per_example_stats:
            pred = e["stats"][category]["pred"]
            gold = e["stats"][category]["gold"]
            match_percent = (pred / gold * 100) if gold > 0 else 0.0
            match_percents.append(match_percent)

        total_match = sum(e["stats"][category]["match"] for e in per_example_stats)
        total_missing = sum(e["stats"][category]["missing"] for e in per_example_stats)
        total_extra = sum(e["stats"][category]["extra"] for e in per_example_stats)

        TOTAL_gold = total_match + total_missing  # gold reference events
        TOTAL_in_audio = total_match + total_extra  # predicted events

        mean_std_stats[category] = {
            "MATCH_mean (in percent)": float(np.mean(match_percents)),
            "std match (in percent)": float(np.std(match_percents)),
            "DIFFERENCE_mean (in example)": float(np.mean(deviations)),
            "std difference": float(np.std(deviations)),
            "TOTAL_gold": int(TOTAL_gold),
            "TOTAL_in_audio": int(TOTAL_in_audio),
        }
    correlation_stats = {}
    for category in categories:
        preds = [x["stats"][category]["pred"] for x in per_example_stats]
        golds = [x["stats"][category]["gold"] for x in per_example_stats]
        corr, p_value = pearsonr(preds, golds)
        correlation_stats[category] = {"corr": corr, "p_value": p_value}

    # prepare final output
    output = {
        "correlation_stats": correlation_stats,
        "overall_stats": overall_stats,
        "mean_std_stats": mean_std_stats,
        "per_example_stats": per_example_stats,
    }

    for key, value in correlation_stats.items():
        print(key, ":", value)

    for key, value in overall_stats.items():
        print(key, ":", value)

    print()

    for key, value in mean_std_stats.items():
        print(key, ":", value)

    out_path = f"{out_dir}/results_bc_interrupt.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Done. Results saved to {out_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute UTMOS scores for stereo dialogue WAV pairs using ASR timestamps."
    )
    parser.add_argument(
        "--wavs_dir",
        type=str,
        required=True,
        help="Path to directory containing WAV files (paired per dialogue)",
    )
    parser.add_argument(
        "--text_stream_json",
        type=str,
        required=True,
        help="Directory containing text stream JSON files",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output JSON"
    )

    args = parser.parse_args()

    main(
        wavs_dir=args.wavs_dir,
        reference_path=args.text_stream_json,
        out_dir=args.output_dir,
    )
