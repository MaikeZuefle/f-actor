import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
from nemo.collections.tts.models import AudioCodecModel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    track,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchaudio.functional import resample
from tqdm import tqdm


class AudioDataset(Dataset):
    def __init__(self, audio_filepaths, target_sample_rate, is_mono=False):
        self.audio_filepaths = audio_filepaths
        self.target_sample_rate = target_sample_rate
        self.is_mono = is_mono

    def __len__(self):
        return len(self.audio_filepaths)

    def __getitem__(self, idx):
        audio_path = self.audio_filepaths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            waveform = resample(
                waveform, orig_freq=sample_rate, new_freq=self.target_sample_rate
            )

        # If mono flag is set, and audio is stereo, convert to mono by averaging channels
        if self.is_mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Now, waveform is guaranteed to be mono if self.is_mono is True

        # Separate channels for stereo processing or handle mono
        if waveform.shape[0] > 1:  # Stereo
            c1_waveform = waveform[0, :].unsqueeze(0)
            c2_waveform = waveform[1, :].unsqueeze(0)
        else:  # Mono
            c1_waveform = waveform
            c2_waveform = None

        return c1_waveform, c2_waveform, audio_path


def collate_fn(batch, channel_idx=0, is_mono=False):
    c1_waveforms, c2_waveforms, paths = zip(*batch)

    waveforms = []
    new_paths = []
    lengths = []

    # Select which channel to process
    target_waveforms = c1_waveforms if channel_idx == 0 else c2_waveforms

    # Filter out None values which occur for mono files on the second channel pass
    valid_waveforms = []
    valid_paths = []
    for w, p in zip(target_waveforms, paths):
        if w is not None:
            valid_waveforms.append(w)
            valid_paths.append(p)

    # If the batch is empty after filtering (e.g., all mono files on channel 2 pass), return empty tensors
    if not valid_waveforms:
        return torch.tensor([]), [], torch.tensor([])

    MAX_AUDIO_LEN = max([w.shape[-1] for w in valid_waveforms])

    for i, w in enumerate(valid_waveforms):
        cur_path = valid_paths[i].replace(".flac", ".wav")
        orig_len = w.shape[-1]
        lengths.append(torch.tensor(orig_len))

        if orig_len < MAX_AUDIO_LEN:
            pad_len = MAX_AUDIO_LEN - orig_len
            w = torch.nn.functional.pad(w, (0, pad_len))

        waveforms.append(w)

        # Adjust filename based on mono or stereo processing
        if is_mono:
            # For mono, don't add a channel suffix
            new_filename = cur_path
        else:
            # For stereo, add channel suffix
            new_filename = cur_path.replace(".wav", f"_c{channel_idx+1}.wav")
        new_paths.append(new_filename)

    waveforms = torch.stack(waveforms)
    lengths = torch.stack(lengths)

    return waveforms, new_paths, lengths


def get_files_in_subdir(subdir_name, data_dir, audio_extensions=(".wav", ".flac")):
    subdir_file_paths = []
    full_subdir_path = os.path.join(data_dir, subdir_name)
    for root, _, files in os.walk(full_subdir_path):
        for f in files:
            if f.lower().endswith(tuple(ext.lower() for ext in audio_extensions)):
                subdir_file_paths.append(os.path.join(root, f))
    return subdir_file_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help="Directory of data to be preprocessed",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory of procesed data to be saved to",
    )
    parser.add_argument("--batch_size", required=False, type=int, default=1)
    parser.add_argument(
        "--codec_ckpt",
        required=False,
        type=str,
        default=None,
        help="Path to codec model checkpoint",
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="If set, process audio as a single mono channel",
    )

    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ["MASTER_ADDR"] = (
        "localhost"  # Or the IP of the master node in multi-node
    )
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size
    )  # NCCL for NVIDIA GPUs
    torch.cuda.set_device(rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.codec_ckpt:
        model = AudioCodecModel.from_pretrained(args.codec_ckpt).eval().to(device)
    else:
        model = (
            AudioCodecModel.from_pretrained(
                "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps"
            )
            .eval()
            .to(device)
        )
    model = DDP(model, device_ids=[rank])

    subdirs = [
        d
        for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ]
    file_paths = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_subdir = {
            executor.submit(get_files_in_subdir, subdir, args.data_dir): subdir
            for subdir in subdirs
        }

        if rank == 0:
            iterator = tqdm(
                as_completed(future_to_subdir),
                total=len(subdirs),
                desc="Obtaining files in sub-directories (threaded)",
            )
        else:
            iterator = as_completed(future_to_subdir)

        for future in iterator:
            found_files = future.result()
            file_paths.extend(found_files)

    if rank == 0:
        print(f"Finished collecting {len(file_paths)} audio files.")

    # Pass the args.mono flag to the dataset
    audio_data = AudioDataset(file_paths, model.module.sample_rate, is_mono=args.mono)
    sampler = DistributedSampler(audio_data, num_replicas=world_size, rank=rank)

    num_channels = 1 if args.mono else 2
    for channel_idx in range(num_channels):
        if rank == 0:
            print(f"Processing channel {channel_idx + 1}")

        # Use a lambda to pass the channel index and mono flag to the collate function
        def channel_collate_fn(batch):
            return collate_fn(batch, channel_idx=channel_idx, is_mono=args.mono)

        dataloader = DataLoader(
            audio_data,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=5,
            collate_fn=channel_collate_fn,
        )

        if rank == 0:
            batch_iter = tqdm(
                dataloader,
                desc=f"Processing Channel {channel_idx + 1} through Codec Model",
            )
        else:
            batch_iter = dataloader

        for batch in batch_iter:
            inputs, filenames, lengths = batch

            inputs = inputs.squeeze(1)  # Remove channel dimension

            # Skip empty batches that might result from filtering
            if inputs.numel() == 0:
                continue

            with torch.no_grad():
                DSUs, DSU_lengths = model.module.encode(
                    audio=inputs.to(device), audio_len=lengths.to(device)
                )

            for j, DSU_sequence in enumerate(DSUs):
                unpadded_length = DSU_lengths[j]
                unpadded_DSU_seq = DSU_sequence[:, :unpadded_length]

                output_filename = os.path.basename(filenames[j]).replace(".wav", ".npy")
                output_path = os.path.join(args.output_dir, output_filename)

                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                np.save(output_path, unpadded_DSU_seq.cpu().numpy())


if __name__ == "__main__":
    main()
