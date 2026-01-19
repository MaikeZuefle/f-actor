import argparse
import json
import logging
import os
import random
import re

import nemo.collections.asr.models as asr_models
import numpy as np
import torch
import torchaudio
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import set_seed
from utils import get_soda_index, get_wav_pairs

set_seed(42)


def get_speech_snippet(wav, sr, seg):
    """Helper to slice a snippet from segment dict."""
    start_sample = int(seg["start"] * sr)
    end_sample = int(seg["end"] * sr)
    return wav[start_sample:end_sample].contiguous()


def get_random_speech_snippet(wav_path, vad_model, min_dur=3.0, max_dur=5.0):
    """
    Return:
        - random speech snippet
        - first speech snippet
        - last speech snippet
        - sampling rate
    based on Silero VAD.
    """
    random.seed(42)
    sr = 16000
    wav = read_audio(wav_path, sampling_rate=sr)

    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        return_seconds=True,
    )

    # No speech → return first 5s for all
    if not speech_timestamps:
        end_sample = int(min(5.0, len(wav) / sr) * sr)
        snippet = wav[:end_sample].contiguous()
        return snippet, snippet, snippet, sr

    # Filter segments within duration range
    valid_segments = [
        seg
        for seg in speech_timestamps
        if min_dur <= (seg["end"] - seg["start"]) <= max_dur
    ]

    # Pick random segment
    if valid_segments:
        random_seg = random.choice(valid_segments)
    else:
        random_seg = max(speech_timestamps, key=lambda s: s["end"] - s["start"])

    # First + last segments (based on time)
    first_seg = speech_timestamps[0]
    last_seg = speech_timestamps[-1]

    # Extract waveforms
    random_snippet = get_speech_snippet(wav, sr, random_seg)
    first_snippet = get_speech_snippet(wav, sr, first_seg)
    last_snippet = get_speech_snippet(wav, sr, last_seg)

    return random_snippet, first_snippet, last_snippet, sr


def compute_speaker_embedding(
    snippet, speaker_model, sr_in=16000, device=torch.device("cuda")
):
    """
    Compute speaker embedding for a mono snippet (torch.Tensor).
    Automatically resamples to the model's target SR if needed.
    """
    snippet = snippet.to(dtype=torch.float32)
    model = speaker_model.module if hasattr(speaker_model, "module") else speaker_model
    target_sr = model.preprocessor.featurizer.sample_rate

    if sr_in != target_sr:
        snippet = torchaudio.functional.resample(
            snippet.unsqueeze(0), orig_freq=sr_in, new_freq=target_sr
        ).squeeze(0)

    input_signal = snippet.unsqueeze(0).to(device)  # [1, T]
    input_length = torch.tensor([snippet.shape[-1]], dtype=torch.int64).to(device)

    with torch.no_grad():
        logits, embeddings = model.forward(
            input_signal=input_signal, input_signal_length=input_length
        )

    emb = embeddings[0].cpu().numpy()
    return emb


def get_speaker_embeds(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    speaker_emb1 = {entry["soda_index"]: np.array(entry["spk_emb1"]) for entry in data}
    speaker_emb2 = {entry["soda_index"]: np.array(entry["spk_emb2"]) for entry in data}
    return speaker_emb1, speaker_emb2


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    # normalized dot
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return (a_norm @ b_norm.T).squeeze()


def load_speaker_model():
    speaker_model = asr_models.EncDecSpeakerLabelModel.from_pretrained(
        "ecapa_tdnn", map_location=torch.device("cuda")
    )
    speaker_model.eval()
    rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.device_count() > 1:
        # Only wrap in DDP if multiple GPUs
        speaker_model = DDP(speaker_model, device_ids=[rank])
    return speaker_model


def main(args, logger):
    logger.info(f"Reading generated output from: {args.wavs_dir}")

    speaker_emb1, speaker_emb2 = get_speaker_embeds(args.speaker_emb_file)

    wavs_pairs = get_wav_pairs(args.wavs_dir)
    silero_vad_model = load_silero_vad()
    speaker_model = load_speaker_model()

    sims_c1_list = []
    sims_c2_list = []

    for c1_wav, c2_wav in tqdm(wavs_pairs):
        soda_index = get_soda_index([c1_wav])

        c1_rand, c1_first, c1_last, sr1 = get_random_speech_snippet(
            c1_wav, silero_vad_model
        )
        c2_rand, c2_first, c2_last, sr2 = get_random_speech_snippet(
            c2_wav, silero_vad_model
        )

        c1_emb_rand = compute_speaker_embedding(c1_rand, speaker_model, sr_in=sr1)
        c1_emb_first = compute_speaker_embedding(c1_first, speaker_model, sr_in=sr1)
        c1_emb_last = compute_speaker_embedding(c1_last, speaker_model, sr_in=sr1)

        c2_emb_rand = compute_speaker_embedding(c2_rand, speaker_model, sr_in=sr2)
        c2_emb_first = compute_speaker_embedding(c2_first, speaker_model, sr_in=sr2)
        c2_emb_last = compute_speaker_embedding(c2_last, speaker_model, sr_in=sr2)

        c1_prompt_emb = speaker_emb1[soda_index]
        c2_prompt_emb = speaker_emb2[soda_index]

        sim_c1_random = float(cosine_similarity(c1_emb_rand, c1_prompt_emb))
        sim_c2_random = float(cosine_similarity(c2_emb_rand, c2_prompt_emb))

        sim_c1_shift = 1 - float(cosine_similarity(c1_emb_first, c1_emb_last))
        sim_c2_shift = 1 - float(cosine_similarity(c2_emb_first, c2_emb_last))

        # sim_c1_random = float(cosine_similarity(c1_emb_rand, c2_prompt_emb))
        # sim_c2_random = float(cosine_similarity(c2_emb_rand, c1_prompt_emb))

        # sim_c1_shift = 1 - float(cosine_similarity(c1_emb_first, c2_emb_last))
        # sim_c2_shift = 1 - float(cosine_similarity(c2_emb_first, c1_emb_last))

        sims_c1_list.append(
            {
                "soda_index": soda_index,
                "sim_random_vs_prompt": sim_c1_random,
                "sim_first_vs_last": sim_c1_shift,
            }
        )

        sims_c2_list.append(
            {
                "soda_index": soda_index,
                "sim_random_vs_prompt": sim_c2_random,
                "sim_first_vs_last": sim_c2_shift,
            }
        )

    avg_c1_random = float(np.mean([x["sim_random_vs_prompt"] for x in sims_c1_list]))
    avg_c2_random = float(np.mean([x["sim_random_vs_prompt"] for x in sims_c2_list]))

    avg_c1_shift = float(np.mean([x["sim_first_vs_last"] for x in sims_c1_list]))
    avg_c2_shift = float(np.mean([x["sim_first_vs_last"] for x in sims_c2_list]))

    logger.info(f"Average C1 random→prompt similarity: {avg_c1_random:.4f}")
    logger.info(f"Average C2 random→prompt similarity: {avg_c2_random:.4f}")
    logger.info(f"Average C1 speaker shift: {avg_c1_shift:.4f}")
    logger.info(f"Average C2 speaker shift: {avg_c2_shift:.4f}")
    # Save to JSON
    output_path = os.path.join(args.output_dir, "speaker_sims.json")
    results_dict = {
        "c1_avg_random_vs_prompt": avg_c1_random,
        "c2_avg_random_vs_prompt": avg_c2_random,
        "c1_avg_shift": avg_c1_shift,
        "c2_avg_shift": avg_c2_shift,
        "sims_c1": sims_c1_list,
        "sims_c2": sims_c2_list,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)

    logger.info(f"Saved speaker similarities to {output_path}")
    logger.info("Everything done! Goodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process generated output and evaluate with a judge."
    )

    parser.add_argument(
        "--wavs_dir",
        type=str,
        required=True,
        help="Path to directory containing WAV files",
    )
    parser.add_argument(
        "--speaker_emb_file",
        type=str,
        required=True,
        help="Path to directory containing instructions (and general outputs)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save the result"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "llm_judge.log")),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    main(args, logger)
