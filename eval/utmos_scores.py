import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from utils import get_wav_pairs, load_asr_model


def get_utmos_from_asr_segments(wav_path, asr_model, predictor):
    """Compute UTMOS scores for all ASR speech segments in one channel."""
    audio_np, sr = sf.read(wav_path)
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)

    # Run ASR to get timestamped speech segments
    transcript = asr_model.transcribe(audio_np, timestamps=True)
    segments = transcript[0].timestamp["segment"]

    scores = []
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        seg_audio = audio_np[start_sample:end_sample]

        if len(seg_audio) == 0:
            continue

        audio_tensor = torch.from_numpy(seg_audio).float().unsqueeze(0).to("cuda")
        score = predictor(audio_tensor, sr).item()
        scores.append(score)

    return scores


def main(wavs_dir, out_dir):
    wav_pairs = get_wav_pairs(wavs_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load ASR and UTMOS models
    print("Loading models...")
    asr_model = load_asr_model()
    predictor = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    ).to("cuda")

    file_scores = {}
    all_scores = []
    speaker1_means = []
    speaker2_means = []

    print(f"Processing {len(wav_pairs)} dialogue pairs...")
    for ch1_path, ch2_path in tqdm(wav_pairs):
        # Speaker 1
        scores_s1 = get_utmos_from_asr_segments(ch1_path, asr_model, predictor)
        mean_s1 = float(np.mean(scores_s1)) if scores_s1 else 0.0
        speaker1_means.append(mean_s1)
        file_scores[os.path.basename(ch1_path)] = mean_s1

        # Speaker 2
        scores_s2 = get_utmos_from_asr_segments(ch2_path, asr_model, predictor)
        mean_s2 = float(np.mean(scores_s2)) if scores_s2 else 0.0
        speaker2_means.append(mean_s2)
        file_scores[os.path.basename(ch2_path)] = mean_s2

        # Combined dialogue mean
        if scores_s1 or scores_s2:
            dialogue_mean = np.mean([x for x in [mean_s1, mean_s2] if x > 0])
            all_scores.append(dialogue_mean)

    # Global stats
    scores_np = np.array(all_scores) if all_scores else np.array([0.0])
    final_score_mean = float(np.mean(scores_np))
    final_score_std = float(np.std(scores_np))
    min_score = float(np.min(scores_np))
    max_score = float(np.max(scores_np))

    mean_speaker1 = float(np.mean(speaker1_means)) if speaker1_means else 0.0
    mean_speaker2 = float(np.mean(speaker2_means)) if speaker2_means else 0.0

    # Print results
    print("UTMOS Scores:")
    print(f"Mean (overall): {final_score_mean:.3f}")
    print(f"Std: {final_score_std:.3f}")
    print(f"Min: {min_score:.3f}")
    print(f"Max: {max_score:.3f}")
    print(f"Mean Speaker 1: {mean_speaker1:.3f}")
    print(f"Mean Speaker 2: {mean_speaker2:.3f}")

    # Save to JSON
    out_file = os.path.join(out_dir, "utmos_scores.json")
    result_dict = {
        "mean": final_score_mean,
        "std": final_score_std,
        "min": min_score,
        "max": max_score,
        "mean_speaker1": mean_speaker1,
        "mean_speaker2": mean_speaker2,
        "scores_per_file": file_scores,
    }

    with open(out_file, "w") as f:
        json.dump(result_dict, f, indent=4)

    print(f"Saved results to {out_file}")


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
