import argparse
import os
import random
from glob import glob

import numpy as np
import soundfile as sf
import torch
from transformers import MimiModel


def load_model(num_codebooks=8):
    model = MimiModel.from_pretrained("kyutai/mimi").to("cuda").eval()
    return model


def get_audio(model, dsus):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        c = torch.from_numpy(dsus).to(device)
        audio = model.decode(c.unsqueeze(0))[0].cpu().numpy()
        audio = np.squeeze(audio)
    return audio


def convert_to_audio(model, dsus, out_file):
    dsus = dsus.T  # from (seq,head) -> (head,seq)

    num_heads = dsus.shape[0]

    dsus_c1 = dsus[0 : (num_heads // 2)]
    dsus_c2 = dsus[(num_heads // 2) :]
    codebook_size = model.quantizer.semantic_residual_vector_quantizer.codebook_size

    # if last token is bigger than codebook size, it's EOS token and can be removed
    if dsus_c1[:, -1].max() >= codebook_size or dsus_c2[:, -1].max() >= codebook_size:
        dsus_c1 = dsus_c1[:, :-1]
        dsus_c2 = dsus_c2[:, :-1]

    audio_c1 = get_audio(model, dsus_c1)
    audio_c2 = get_audio(model, dsus_c2)

    # Combine into stereo
    stereo_audio = np.stack([audio_c1, audio_c2], axis=-1)
    sf.write(out_file, stereo_audio, samplerate=24000)

    # Write individual channels
    base, ext = os.path.splitext(out_file)
    out_file_c1 = f"{base}_c1{ext}"
    out_file_c2 = f"{base}_c2{ext}"

    sf.write(out_file_c1, audio_c1, samplerate=24000)
    sf.write(out_file_c2, audio_c2, samplerate=24000)
