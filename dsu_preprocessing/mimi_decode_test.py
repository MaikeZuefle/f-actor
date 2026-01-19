import argparse
import os
import random
from glob import glob

import numpy as np
import soundfile as sf
import torch
from transformers import MimiModel


def main(input_dir, output_dir, num_pairs=5, seed=None):
    os.makedirs(output_dir, exist_ok=True)

    # Find all _c1.npy files
    c1_files = glob(os.path.join(input_dir, "*_c1.npy"))
    if seed is not None:
        random.seed(seed)
    random.shuffle(c1_files)

    selected = c1_files[:num_pairs]
    if len(selected) < num_pairs:
        print(f"Warning: only found {len(selected)} pairs.")

    # Initialize your decoder (adjust to your actual library usage)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    with torch.no_grad():

        for idx, c1_path in enumerate(selected, 1):
            c2_path = c1_path.replace("_c1.npy", "_c2.npy")
            if not os.path.isfile(c2_path):
                print(f"Skipping {c1_path}: partner file not found ({c2_path}).")
                continue

            c1 = torch.from_numpy(np.load(c1_path)).to(device)

            c2 = torch.from_numpy(np.load(c2_path)).to(device)

            audio_c1 = model.decode(c1[0:1, :].unsqueeze(0))[0].cpu().numpy()
            audio_c2 = model.decode(c2[0:1, :].unsqueeze(0))[0].cpu().numpy()

            audio_c1 = np.squeeze(audio_c1)
            audio_c2 = np.squeeze(audio_c2)

            output_path = os.path.join(output_dir, f"{idx:03d}.wav")
            stereo_audio = np.stack([audio_c1, audio_c2], axis=-1)
            sf.write(output_path, stereo_audio, samplerate=24000)
            print(f"Saved decoded stereo audio to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode mimi encodings to wav files")
    parser.add_argument(
        "input_dir", type=str, help="Directory with numpy _c1.npy and _c2.npy files"
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory to save decoded wav files"
    )
    parser.add_argument(
        "--num_pairs", type=int, default=5, help="Number of pairs to decode"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.num_pairs, args.seed)
