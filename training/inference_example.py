import argparse
import json

import numpy as np
import torch
from huggingface_hub import snapshot_download
from inference_audio.nano_decode import convert_to_audio
from inference_audio.nano_decode import load_model as load_nanocodecs
from special_tokens import TEXT_STREAM_TOKENS
from modeling_dsu import DSULlama
from transformers import AutoConfig, AutoTokenizer
import re

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = snapshot_download(
        repo_id="maikezu/f-actor", local_dir=f"./hf_models/f_actor"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, legacy=False, rust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model_cls = DSULlama

    # add arguments to config
    config = AutoConfig.from_pretrained(model_id)
    config.num_dsus = 4
    config.text_stream = True
    config.multi_text_stream = False
    config.audio_vocab_size = 4032
    config.use_speaker_embedding = True
    config.calc_loss_on_c1_only = False

    # load model (if num_dsu < 1, this will be the normal model)
    model = model_cls.from_pretrained(
        model_id,
        config=config,
    ).to(device)

    # set some useful paremters needed for forward passes
    model.pad_token_id = tokenizer.pad_token_id
    model.grad_acc_steps = 8

    model.init_or_load_speaker_embed_proj(model_path=model_id)
    model.init_or_load_audio_heads(model_path=model_id)
    model.init_or_load_audio_embeds(model_path=model_id)
    model.eval()
    return model, tokenizer


def prepare_prompt(
    system_speaker, user_speaker, narrative, speaks_first, backchannels, interruptions
):
    prompt = f"""Generate a dialogue between you ({system_speaker}) and another speaker ({user_speaker}) based on the given narrative. Follow the specific behavior instructions for you.

Narrative:
{narrative}

Your behaviors:
- backchannels: {backchannels}
- interruptions: {interruptions}
- starts the dialogue: {speaks_first}

Ensure that the dialogue reflects the behaviours of you.\n"""

    prompt += "<|SOS|>"
    return prompt


def prepare_input(speaker1, speaker2):
    prompt_speaker1 = prepare_prompt(
        system_speaker=speaker1["name"],
        user_speaker=speaker2["name"],
        narrative=speaker1["narrative"],
        speaks_first=speaker1["starts"],
        backchannels=speaker1.get("backchannels", 2),
        interruptions=speaker1.get("interruptions", 2),
    )
    prompt_speaker2 = prepare_prompt(
        system_speaker=speaker2["name"],
        user_speaker=speaker1["name"],
        narrative=speaker2["narrative"],
        speaks_first=speaker2["starts"],
        backchannels=speaker2.get("backchannels", 2),
        interruptions=speaker2.get("interruptions", 2),
    )

    with open("training/example_speakers/speaker_embeddings.json", "r") as f:
        spk_to_emb = json.load(f)

    speaker_embed_speaker1 = np.array(spk_to_emb[speaker1["name"].lower()])
    speaker_embed_speaker2 = np.array(spk_to_emb[speaker2["name"].lower()])

    return {
        "prompts": [prompt_speaker1, prompt_speaker2],
        "spk_embeds": [speaker_embed_speaker1, speaker_embed_speaker2],
    }


def remove_special_tokens(text):
    token_patterns = [rf"(?:\d+x)?{re.escape(token)}" for token in TEXT_STREAM_TOKENS]
    pattern = "|".join(token_patterns)
    cleaned = re.sub(pattern, " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def run_inference(config):
    speaker1 = config["speaker1"]
    speaker2 = config["speaker2"]
    out_audio_file = config.get("output_file", "dialogue.wav")

    # load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model()
    nanocodec_model = load_nanocodecs(num_codebooks=4)

    # prepare instructions
    print("Preparing instructions...")
    example = prepare_input(speaker1, speaker2)
    inputs = tokenizer(
        example["prompts"],
        return_tensors="pt",
        max_length=4096,
        truncation=True,
        padding=True,
    )
    input_ids = inputs.input_ids.cuda()
    attention_mask = inputs.attention_mask.cuda()

    # generate dialogue
    print("Generating dialogue...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=config.get("max_length", 1024),
            do_sample=True,
            temperature=config.get("temperature", 0.9),
            top_k=config.get("top_k", 40),
            top_p=config.get("top_p", 1.0),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer,
            stop_strings=["<|EOS|>"],
            use_speaker_sample=0,
            dsu_sample=torch.zeros(1, 8, 1, dtype=torch.int64).to(
                "cuda"
            ),  # placeholder
            text_sample=torch.zeros(1, 2, 1, dtype=torch.int64).to(
                "cuda"
            ),  # placeholder
            n_delay_text_stream=0,
            n_delay_audio_stream=2,
            talk_to_itself=True,
            spk_emb=example["spk_embeds"],
        )

    generated_dsu_ids, generated_text_ids = generated_ids

    # generate speech sample
    print("Decoding speech sample...")
    generated_dsus = [
        generated_dsu_ids[head_idx][2:] for head_idx in range(model.num_dsu_heads)
    ]
    dsus = np.array(generated_dsus).T

    convert_to_audio(nanocodec_model, dsus, out_audio_file)
    print("Dialogue saved to ", out_audio_file)

    # decode generated text
    print("Decoding generated text")
    text_speaker1 = remove_special_tokens(tokenizer.decode(generated_text_ids[0], skip_special_tokens=False))
    text_speaker2 = remove_special_tokens(tokenizer.decode(generated_text_ids[1], skip_special_tokens=False))
    return text_speaker1, text_speaker2


def main():
    parser = argparse.ArgumentParser(
        description="Generate a custom dialogue with F-Actor from a JSON config."
    )
    parser.add_argument(
        "--config",
        default="confs/example_dialogue.json",
        help="Path to a dialogue config JSON (see confs/example_dialogue.json).",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    speakers = [
        "Rebeka",
        "Gweneth",
        "Brian",
        "Tom",
    ]

    for spk in (config["speaker1"], config["speaker2"]):
        if spk["name"] not in speakers:
            raise KeyError(
                f"Invalid speaker '{spk['name']}', please choose speakers from {speakers}!"
            )

    text_speaker1, text_speaker2 = run_inference(config)

    print("\n=== Transcripts ===")
    print(f"{config['speaker1']['name']}: {text_speaker1}")
    print(f"{config['speaker2']['name']}: {text_speaker2}")


if __name__ == "__main__":
    main()
