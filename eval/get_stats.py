import argparse
import json
import os

import numpy as np
import soundfile as sf
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
from tqdm import tqdm
from utils import get_wav_pairs


def get_duration_seconds(wav_path):
    """Return total duration of a .wav file in seconds."""
    info = sf.info(wav_path)
    return info.duration


def get_speech_time(wav_path, model):
    """
    Return total speaking time (in seconds) for a single speaker's wav
    using Silero VAD.
    """
    wav = read_audio(wav_path)
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=True,
    )

    speech_time = sum(seg["end"] - seg["start"] for seg in speech_timestamps)
    return speech_time


def main(wavs_dir, out_dir):
    # Get (c1, c2, merged) tuples
    wav_pairs = get_wav_pairs(wavs_dir, include_merged_wav=True)
    os.makedirs(out_dir, exist_ok=True)

    # Load VAD model once
    model = load_silero_vad()

    total_lengths = []
    c1_speech_times = []
    c2_speech_times = []
    speech_diffs = []

    for c1_wav, c2_wav, merged_wav in tqdm(wav_pairs):
        # Compute total dialogue length
        total_length = get_duration_seconds(merged_wav)

        # Compute how much each speaker speaks
        c1_speech = get_speech_time(c1_wav, model)
        c2_speech = get_speech_time(c2_wav, model)

        # Store for averaging
        total_lengths.append(total_length)
        c1_speech_times.append(c1_speech)
        c2_speech_times.append(c2_speech)
        speech_diffs.append(abs(c1_speech - c2_speech))

    # Compute averages
    avg_total = np.mean(total_lengths)
    avg_c1_speech = np.mean(c1_speech_times)
    avg_c2_speech = np.mean(c2_speech_times)
    avg_diff = np.mean(speech_diffs)

    # Compute balance (%)
    c1_pct = avg_c1_speech / avg_total * 100
    c2_pct = avg_c2_speech / avg_total * 100
    diff_pct = avg_diff / avg_total * 100

    # Print summary
    print(f"\n📊 Dialogue Statistics:")
    print(f"Average total dialogue length: {avg_total:.2f}s")
    print(
        f"Average C1 speaking time: {avg_c1_speech:.2f}s ({avg_c1_speech / avg_total * 100:.1f}%)"
    )
    print(
        f"Average C2 speaking time: {avg_c2_speech:.2f}s ({avg_c2_speech / avg_total * 100:.1f}%)"
    )
    print(
        f"Average speaking time difference: {avg_diff:.2f}s ({avg_diff / avg_total * 100:.1f}% of total)"
    )

    # Prepare results dict
    results = {
        "num_dialogues": len(wav_pairs),
        "average_total_length_s": avg_total,
        "average_c1_speech_s": avg_c1_speech,
        "average_c2_speech_s": avg_c2_speech,
        "average_speech_difference_s": avg_diff,
        "c1_percentage": round(c1_pct, 1),
        "c2_percentage": round(c2_pct, 1),
        "difference_percentage": round(diff_pct, 1),
    }

    output_path = os.path.join(out_dir, "speech_statistics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to {output_path}")


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
        "--out_dir", type=str, required=True, help="Directory to save output JSON"
    )
    args = parser.parse_args()

    main(args.wavs_dir, args.out_dir)
