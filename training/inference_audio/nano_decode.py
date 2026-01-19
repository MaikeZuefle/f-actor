import argparse
import os
import random
from glob import glob

import numpy as np
import soundfile as sf
import torch
from nemo.collections.tts.models import AudioCodecModel


def load_model(num_codebooks=8):
    if num_codebooks == 8:
        model = AudioCodecModel.from_pretrained(
            "nvidia/low-frame-rate-speech-codec-22khz"
        )
    elif num_codebooks == 4:
        model = AudioCodecModel.from_pretrained(
            "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"
        )
    else:
        raise NotImplementedError
    model = model.eval()
    return model


def get_audio(model, dsus):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        c = torch.from_numpy(dsus).to(device)
        c_length = torch.tensor(c.shape[-1]).unsqueeze(0).to(device)
        audio = (
            model.decode(tokens=c.unsqueeze(0), tokens_len=c_length)[0].cpu().numpy()
        )
        audio = np.squeeze(audio)
    return audio


def convert_to_audio(model, dsus, out_file):
    dsus = dsus.T  # from (seq,head) -> (head,seq)
    num_heads = dsus.shape[0]

    dsus_c1 = dsus[0 : (num_heads // 2)]
    dsus_c2 = dsus[(num_heads // 2) :]

    audio_c1 = get_audio(model, dsus_c1)
    audio_c2 = get_audio(model, dsus_c2)

    # Combine into stereo
    stereo_audio = np.stack([audio_c1, audio_c2], axis=-1)
    sf.write(out_file, stereo_audio, samplerate=22050)

    # Write individual channels
    base, ext = os.path.splitext(out_file)
    out_file_c1 = f"{base}_c1{ext}"
    out_file_c2 = f"{base}_c2{ext}"

    sf.write(out_file_c1, audio_c1, samplerate=22050)
    sf.write(out_file_c2, audio_c2, samplerate=22050)
