import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
Modify the tokenizer offline by adding new regular or special tokens
"""


def add_new_tokens(tokenizer, new_tokens, new_special_tokens, number_audio_tokens):
    if not new_tokens and not new_special_tokens and number_audio_tokens <= 0:
        print(f"Neither new regular, special tokens nor audio_tokens to add")
        return

    exist_vocab = tokenizer.get_vocab()

    if new_tokens:
        tokens_to_add = [tok for tok in new_tokens if tok not in exist_vocab]
        if tokens_to_add:
            added = tokenizer.add_tokens(tokens_to_add)
            print(f"{added} tokens added - {tokens_to_add}")

    if new_special_tokens:
        special_tokens_to_add = [
            tok for tok in new_special_tokens if tok not in exist_vocab
        ]
        print(f"{special_tokens_to_add}")
        if special_tokens_to_add:
            added = tokenizer.add_special_tokens(
                {"additional_special_tokens": special_tokens_to_add}
            )
            print(f"{added} special tokens added - {special_tokens_to_add}")

    if number_audio_tokens > 0:
        tokens_to_add = []
        for idx in range(number_audio_tokens):
            tok = f"<|AO{idx+1}|>"
            if tok not in exist_vocab:
                tokens_to_add.append(tok)
        if tokens_to_add:
            added = tokenizer.add_tokens(tokens_to_add)
            print(f"{added} speech tokens added - {tokens_to_add}")

    return tokenizer


def save_tokenizer_and_model(tokenizer, model, save_path):
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"Tokenizer + model saved to {save_path}")


def load_tokenizer_and_model(model_id, set_pad_to_eos=True):
    print("Trying to load tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    print("Loaded tokenizer + model!")

    if set_pad_to_eos:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def main(args):
    tokenizer, model = load_tokenizer_and_model(args.model_id, args.set_pad_to_eos)
    tokenizer = add_new_tokens(
        tokenizer,
        args.new_tokens,
        args.new_special_tokens,
        args.number_audio_tokens,
    )

    if tokenizer:
        model.resize_token_embeddings(len(tokenizer))
        save_tokenizer_and_model(tokenizer, model, args.save_path)

    print(f"Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="HF's model_id or cache_dir")
    parser.add_argument(
        "--save_path", type=str, help="Output path of the new tokenizer"
    )
    parser.add_argument(
        "--set_pad_to_eos",
        action="store_true",
        help="set tokenizer.pad_token=tokenizer.eos_token",
    )
    parser.add_argument("--new_tokens", type=json.loads)
    parser.add_argument("--new_special_tokens", type=json.loads)
    parser.add_argument("--number_audio_tokens", type=int, default=0)

    args = parser.parse_args()
    main(args)
