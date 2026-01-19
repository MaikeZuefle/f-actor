import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import json
import re

import jiwer
from utils import load_asr_model

from training.special_tokens import TEXT_STREAM_TOKENS

TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.RemoveMultipleSpaces(),
    ]
)


def remove_pads(text, text_stream_tokens):
    # Build regex for both repeated and single pad tokens
    token_patterns = [rf"(?:\d+x)?{re.escape(token)}" for token in text_stream_tokens]
    pattern = "|".join(token_patterns)

    # Remove all pad tokens (with or without repetitions)
    cleaned = re.sub(pattern, " ", text)

    # Normalize spaces — collapse multiple spaces and trim
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        cleaned = " "

    return cleaned


def main(wavs_dir, text_stream_json, out_dir):
    # List all WAV files
    wavs = [
        os.path.join(wavs_dir, f)
        for f in os.listdir(wavs_dir)
        if f.lower().endswith(".wav")
    ]

    if not wavs:
        raise ValueError("No wavs found!")

    with open(text_stream_json, "r") as f:
        text_stream_data = json.load(f)  # list of dicts

    asr_model = load_asr_model()

    # Pair WAV files with their text_stream_0
    wer_scores_c1 = []
    wer_scores_c2 = []
    empty_ref = 0
    for entry in text_stream_data:
        soda_index = entry["soda_index"]
        ref_text_c1 = remove_pads(
            entry["text_stream_0"]["generated"], TEXT_STREAM_TOKENS
        )
        ref_text_c2 = remove_pads(
            entry["text_stream_1"]["generated"], TEXT_STREAM_TOKENS
        )

        wav_file_c1 = f"{wavs_dir}/soda_index_{soda_index}_c1.wav"
        wav_file_c2 = f"{wavs_dir}/soda_index_{soda_index}_c2.wav"

        if not wav_file_c1 in wavs or not wav_file_c2 in wavs:
            raise ValueError(f"Wav file {wav_file_c1} missing!")

        pred = asr_model.transcribe([wav_file_c1, wav_file_c2], timestamps=True)
        pred_c1 = pred[0].text.strip()
        pred_c2 = pred[1].text.strip()


        try:
            wer_c1 = jiwer.wer(TRANSFORM(ref_text_c1), TRANSFORM(pred_c1))
            wer_c2 = jiwer.wer(TRANSFORM(ref_text_c2), TRANSFORM(pred_c2))
        except:
            empty_ref += 1
            continue
        wer_scores_c1.append(wer_c1)
        wer_scores_c2.append(wer_c2)

    avg_c1 = sum(wer_scores_c1) / len(wer_scores_c1)
    avg_c2 = sum(wer_scores_c2) / len(wer_scores_c2)

    print(f"Average WER Channel 1: {avg_c1:.3f}")
    print(f"Average WER Channel 2: {avg_c2:.3f}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "wer_results.json")
    results = {
        "average_c1": avg_c1,
        "average_c2": avg_c2,
        "empty_ref": empty_ref,
        "wer_c1": wer_scores_c1,
        "wer_c2": wer_scores_c2,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved WER results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pair WAVs with text stream JSON entries."
    )
    parser.add_argument(
        "--wavs_dir", type=str, required=True, help="Directory containing WAV files"
    )
    parser.add_argument(
        "--text_stream_json",
        type=str,
        required=True,
        help="Directory containing text stream JSON files",
    )
    parser.add_argument(
        "--out_dir", type=str, required=True, help="Directory to save paired output"
    )
    args = parser.parse_args()

    main(args.wavs_dir, args.text_stream_json, args.out_dir)
