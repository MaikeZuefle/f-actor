import argparse
import json
import os
import re

import numpy as np
from arguments.parse_arguments import parse_args
from inference_audio.mimi_decode import convert_to_audio as convert_to_audio_mimi
from inference_audio.mimi_decode import load_model as load_model_mimi
from inference_audio.nano_decode import convert_to_audio as convert_to_audio_nano
from inference_audio.nano_decode import load_model as load_model_nano


def process_sample(s):
    """Split a sample into text tokens, speech tokens, and all tokens."""
    return {
        "text_tokens": [],
        "speech_tokens": s,
        "all_tokens": s,
    }


def load_all_heads(file_path, num_heads):
    """
    Load and process samples from all heads.
    Returns a list of dicts, one per sample, where each dict contains:
      - per-head speech tokens (list of lists)
      - per-head text tokens
      - per-head all tokens
    """

    with open(file_path, "r") as f:
        data_list = json.load(f)

    all_samples = []
    for data in data_list:
        soda_index = data["soda_index"]

        head_keys = [k for k in data if "head" in k]

        head_keys = sorted(head_keys, key=lambda x: int(x.split("_")[1]))

        all_heads_list = [data[k]["generated"] for k in head_keys]
        assert len(all_heads_list) == num_heads

        all_heads_samples = []
        for head_samples in all_heads_list:

            processed = process_sample(head_samples)
            all_heads_samples.append(processed)

        sample_data = {
            "speech_tokens": [
                all_heads_samples[h]["speech_tokens"] for h in range(num_heads)
            ],
            "text_tokens": [
                all_heads_samples[h]["text_tokens"] for h in range(num_heads)
            ],
            "all_tokens": [
                all_heads_samples[h]["all_tokens"] for h in range(num_heads)
            ],
        }

        all_samples.append((soda_index, sample_data))

    return all_samples


def align_dsus(speech_tokens_per_head, pad_value=0):
    """
    Convert list of lists (speech tokens per head) into a 2D array
    of shape (sequence_length, num_heads) by padding shorter sequences.
    """
    num_heads = len(speech_tokens_per_head)
    max_len = max(len(t) for t in speech_tokens_per_head)
    aligned = np.full((max_len, num_heads), pad_value, dtype=np.int32)
    for h, tokens in enumerate(speech_tokens_per_head):
        aligned[: len(tokens), h] = tokens
    return aligned


def main(model_args, inference_args):
    inf_dir = inference_args.inf_output_dir
    if "mimi" in inf_dir:
        load_model = load_model_mimi
        convert_to_audio = convert_to_audio_mimi
    else:
        load_model = load_model_nano
        convert_to_audio = convert_to_audio_nano

    suffix = (
        f"-{inference_args.do_sample}"
        f"_T{inference_args.temperature:.2f}"
        f"_k{inference_args.top_k}"
        f"_p{inference_args.top_p:.2f}"
    ).replace(".", "_")

    # create output dir
    out_dir = os.path.join(inf_dir, f"outputs_wavs{suffix}")
    os.makedirs(out_dir, exist_ok=True)

    # load DSUs
    num_heads = model_args.num_dsus * 2
    inf_dsu_file_path = os.path.join(inf_dir, f"outputs_samples{suffix}.json")

    all_samples = load_all_heads(
        inf_dsu_file_path,
        num_heads,
    )

    # load code model
    codec_model = load_model(num_codebooks=model_args.num_dsus)

    # decode with codec model
    for sample_idx, (soda_idx, sample) in enumerate(all_samples):

        dsus_array = align_dsus(sample["speech_tokens"])  # shape: (length, num_heads)
        convert_to_audio(
            codec_model,
            dsus_array,
            os.path.join(out_dir, f"soda_index_{soda_idx}.wav"),
        )
        print(f"Encoded sample {sample_idx+1}/{len(all_samples)}!")
        print(os.path.join(out_dir, f"soda_index_{soda_idx}.wav"))
        print()


if __name__ == "__main__":
    model_args, _, _, inference_args = parse_args(include_inference=True)
    main(model_args, inference_args)
