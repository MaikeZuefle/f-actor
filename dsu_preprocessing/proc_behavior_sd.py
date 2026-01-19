import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchaudio.functional import resample
from tqdm import tqdm
from transformers import AutoFeatureExtractor, MimiModel


class AudioDataset(Dataset):
    def __init__(self, audio_filepaths, target_sample_rate):
        self.audio_filepaths = audio_filepaths
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.audio_filepaths)

    def __getitem__(self, idx):
        audio_path = self.audio_filepaths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            c1_waveform = waveform[0, :]
            c2_waveform = waveform[1, :]
        else:
            c1_waveform = waveform
            c2_waveform = None
        return c1_waveform, c2_waveform, audio_path


def collate_fn(batch):
    c1_waveforms, c2_waveforms, paths = zip(*batch)

    waveforms = []
    new_paths = []
    lengths = []

    MAX_AUDIO_LEN = max([w.shape[-1] for w in c1_waveforms + c2_waveforms])

    for i, (w1, w2) in enumerate(zip(c1_waveforms, c2_waveforms)):
        cur_path = paths[i].replace(".flac", ".wav")

        for ch_idx, w in enumerate([w1, w2]):
            # w = w.unsqueeze(0)

            orig_len = w.shape[-1]
            lengths.append(orig_len)

            if orig_len < MAX_AUDIO_LEN:
                pad_len = MAX_AUDIO_LEN - orig_len
                w = torch.nn.functional.pad(w, (0, pad_len))

            waveforms.append(w)
            new_paths.append(cur_path.replace(".wav", f"_c{ch_idx+1}.wav"))

    # Stack into one tensor of shape (batch_size*2, MAX_AUDIO_LEN)
    waveforms = torch.stack(waveforms).unsqueeze(1)  # Add channel dimension

    return waveforms, new_paths, lengths


def get_files_in_subdir(subdir_name, data_dir, audio_extensions=(".wav", ".flac")):
    subdir_file_paths = []
    full_subdir_path = os.path.join(data_dir, subdir_name)
    for entry in os.scandir(full_subdir_path):
        if entry.is_file() and entry.name.endswith(audio_extensions):
            subdir_file_paths.append(entry.path)
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
    parser.add_argument("--num_quantizers", required=False, type=int, default=8)
    parser.add_argument("--downsample_factor", required=False, type=int, default=1280)
    parser.add_argument("--batch_size", required=False, type=int, default=2)
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
    model = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    model = DDP(model, device_ids=[rank])
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    subdirs = [
        d
        for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ]
    file_paths = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit each subdirectory to the executor
        # The executor.submit() method returns a Future object
        future_to_subdir = {
            executor.submit(get_files_in_subdir, subdir, args.data_dir): subdir
            for subdir in subdirs
        }

        # Use tqdm to monitor the progress of the submitted tasks
        for future in tqdm(
            as_completed(future_to_subdir),
            total=len(subdirs),
            desc="Obtaining files in sub-directories (threaded)",
        ):
            subdir_name = future_to_subdir[future]
            # Get the results from the completed future
            found_files = future.result()
            file_paths.extend(found_files)

    print(f"Finished collecting {len(file_paths)} audio files.")

    audio_data = AudioDataset(file_paths, feature_extractor.sampling_rate)
    sampler = DistributedSampler(audio_data, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        audio_data,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=5,
        collate_fn=collate_fn,
    )
    for batch in tqdm(dataloader, desc="Processing Batches through Mimi"):
        inputs, filenames, lengths = batch

        encoded_inputs = model.module.encode(
            inputs.to(device), num_quantizers=args.num_quantizers
        )

        for j, encoded_input in enumerate(encoded_inputs.audio_codes):
            unpadded_length = lengths[j]
            # Unpad the encoded input to match the original length
            unpadded_encoded_input = encoded_input[:, :unpadded_length]

            fname = os.path.join(
                args.output_dir, os.path.basename(filenames[j]).replace(".wav", ".npy")
            )
            np.save(fname, unpadded_encoded_input.cpu().numpy())


if __name__ == "__main__":
    main()
