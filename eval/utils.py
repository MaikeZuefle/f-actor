import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re

import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def get_soda_index(paths):
    path = paths[0]
    match = re.search(r"soda_index_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


def load_asr_model():
    import nemo.collections.asr as nemo_asr

    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )

    return asr_model


def get_wav_pairs(wavs_dir, include_merged_wav=False):
    all_wavs = [f for f in os.listdir(wavs_dir) if f.lower().endswith(".wav")]

    pairs = {}
    for wav in all_wavs:
        path = os.path.join(wavs_dir, wav)
        if "_c1.wav" in wav:
            key = wav.replace("_c1.wav", "")
            pairs.setdefault(key, [None, None, None])[0] = path
        elif "_c2.wav" in wav:
            key = wav.replace("_c2.wav", "")
            pairs.setdefault(key, [None, None, None])[1] = path
        elif include_merged_wav and not ("_c1.wav" in wav or "_c2.wav" in wav):
            # only include merged wavs when explicitly requested
            key = wav.replace(".wav", "")
            pairs.setdefault(key, [None, None, None])[2] = path

    wavs_pairs = [tuple(p[:3] if include_merged_wav else p[:2]) for p in pairs.values()]

    return wavs_pairs
