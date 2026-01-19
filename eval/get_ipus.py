import argparse
import json
import os

import soundfile as sf
from pyannote.audio import Pipeline
from tqdm import tqdm
from utils import get_wav_pairs
import torchaudio


def get_duration_seconds(wav_path):
    """Return total duration of a .wav file in seconds."""
    info = sf.info(wav_path)
    return info.duration


def ipu_metrics_per_min(ipus, audio_duration_sec):
    """
    Returns:
      - Number of IPUs per minute (IPU/min)
      - Cumulative IPU duration per minute (sec/min)
    """
    n_ipus = len(ipus)
    total_ipu_duration = sum(ipu["duration"] for ipu in ipus)

    duration_min = audio_duration_sec / 60.0

    metrics = {
        "ipu_per_min": n_ipus / duration_min,
        "cumulative_ipu_duration_per_min": total_ipu_duration / duration_min,
    }

    return metrics


def get_ipus_from_speech_timestamps(speech_timestamps, min_pause=0.2):
    ipus = []
    n = len(speech_timestamps)

    for i, curr in enumerate(speech_timestamps):
        prev_end = speech_timestamps[i - 1]["end"] if i > 0 else float("-inf")
        next_start = speech_timestamps[i + 1]["start"] if i < n - 1 else float("inf")

        pre_silence = curr["start"] - prev_end
        post_silence = next_start - curr["end"]

        if pre_silence >= min_pause and post_silence >= min_pause:
            ipus.append(
                {
                    "start": curr["start"],
                    "end": curr["end"],
                    "duration": curr["end"] - curr["start"],
                    "pre_silence": pre_silence,
                    "post_silence": post_silence,
                }
            )
    return ipus


def get_ipu_pause_gap_overlap(wav_path1, wav_path2, total_length, pipeline):

    # Step 1: Load audio and get speech timestamps per channel
    ipus_channels = []
    for wav_path in [wav_path1, wav_path2]:

        waveform, sr = torchaudio.load(wav_path)
        outputs = pipeline({"waveform":waveform, "sample_rate":sr})
        speech_timestamps = [
            {"start": s.start, "end": s.end} for s in outputs.get_timeline()
        ]
        ipus = get_ipus_from_speech_timestamps(speech_timestamps, min_pause=0.2)
        ipus_channels.append(ipus)

    duration_min = total_length / 60.0
    metrics = {}

    # Step 2: IPU per channel (sum durations)

    ipu_cumsec_per_min_channels = []
    for i, ipus in enumerate(ipus_channels):
        ipu_total_sec = sum(ipu["duration"] for ipu in ipus)
        ipu_cumsec_per_min_channels.append(ipu_total_sec)

        metrics[f"channel{i+1}_ipu_per_min"] = len(ipus) / duration_min
        metrics[f"channel{i+1}_ipu_cumsec_per_min"] = ipu_total_sec / duration_min

    metrics["total_ipu_cumsec_per_min"] = (
        sum(ipu_cumsec_per_min_channels) / duration_min
    )

    # Step 3: Merge IPUs from both channels
    merged_ipus = sorted(
        [
            (ipu["start"], ipu["end"], ch)
            for ch, ipus in enumerate(ipus_channels)
            for ipu in ipus
        ],
        key=lambda x: x[0],
    )

    # Step 4: Compute pauses and gaps on merged timeline
    pause_sec = 0.0
    gap_sec = 0.0
    prev_start, prev_end, prev_ch = merged_ipus[0]
    for curr_start, curr_end, curr_ch in merged_ipus[1:]:
        silence = max(0, curr_start - prev_end)
        if prev_ch == curr_ch:
            pause_sec += silence  # same speaker → pause
        else:
            gap_sec += silence  # different speaker → gap
        prev_start, prev_end, prev_ch = curr_start, curr_end, curr_ch

    metrics["pause_cumsec_per_min"] = pause_sec / duration_min
    metrics["gap_cumsec_per_min"] = gap_sec / duration_min

    # Step 5: Compute overlap
    overlap_sec = 0.0
    i, j = 0, 0
    ipus_ch1, ipus_ch2 = ipus_channels
    while i < len(ipus_ch1) and j < len(ipus_ch2):
        start1, end1 = ipus_ch1[i]["start"], ipus_ch1[i]["end"]
        start2, end2 = ipus_ch2[j]["start"], ipus_ch2[j]["end"]
        ov_start = max(start1, start2)
        ov_end = min(end1, end2)
        if ov_end > ov_start:
            overlap_sec += ov_end - ov_start
        if end1 <= end2:
            i += 1
        else:
            j += 1

    metrics["overlap_cumsec_per_min"] = overlap_sec / duration_min
    return metrics


def main(wavs_dir, out_dir):
    # Get (c1, c2, merged) tuples
    wav_pairs = get_wav_pairs(wavs_dir, include_merged_wav=True)
    os.makedirs(out_dir, exist_ok=True)

    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")

    results = {}
    # Lists to compute averages
    ipu_list, pause_list, gap_list, overlap_list = [], [], [], []

    for c1_wav, c2_wav, merged_wav in tqdm(wav_pairs):
        total_length = get_duration_seconds(merged_wav)
        metrics = get_ipu_pause_gap_overlap(c1_wav, c2_wav, total_length, pipeline)

        # Extract metrics
        ipu = metrics["total_ipu_cumsec_per_min"]
        pause = metrics["pause_cumsec_per_min"]
        gap = metrics["gap_cumsec_per_min"]
        overlap = metrics["overlap_cumsec_per_min"]

        # Append to lists for average
        ipu_list.append(ipu)
        pause_list.append(pause)
        gap_list.append(gap)
        overlap_list.append(overlap)

        # Save per-example metrics using merged_wav as key
        results[merged_wav] = metrics

    # Compute averages
    avg_metrics = {
        "average_total_ipu_cumsec_per_min": sum(ipu_list) / len(ipu_list),
        "average_pause_cumsec_per_min": sum(pause_list) / len(pause_list),
        "average_gap_cumsec_per_min": sum(gap_list) / len(gap_list),
        "average_overlap_cumsec_per_min": sum(overlap_list) / len(overlap_list),
    }

    # Insert averages at the top of the results dictionary
    output_dict = {"average": avg_metrics, "examples": results}

    # Save to JSON
    output_path = os.path.join(out_dir, "ipu_statistics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)

    print("\n✅ Results saved to", output_path)
    print("\nAverage metrics:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.3f}")


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
