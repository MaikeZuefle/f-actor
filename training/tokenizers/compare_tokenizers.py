import argparse
import json
import os
from functools import partial

import torch
from transformers import AutoTokenizer

"""
Sample scripts for testing and comparing two tokenizers
"""

tokenizer1 = None
tokenizer2 = None

ADDED_TOKENS = [
    "<|SOT|>",
    "<|SOS|>",
    "<|EOS|>",
    "[laughter]",
    "<laughter>",
    "</laughter>",
    "<OVERLAP>",
    "<BC_S>",
    "<BC_E>",
    "<GAP>",
    "<LISTEN>",
    "<SPEAK>",
]
# ["<OVERLAP>", "<BC_S>", "<BC_E>", "<GAP>", "<LISTEN>", "<SPEAK>"]

TEXT = [
    "Hello world!",
    "Hello",
    "<|SOT|>",
    "<|SOS|>",
    "<|EOS|>",
    "[laughter]",
    "<laughter>",
    "</laughter>",
    "<OVERLAP>",
    "<BC_S>",
    "<BC_E>",
    "AUDIO1",
    "AUDIO2",
    "<|AO1|>",
    "<|AO283|>",
    "<LISTEN>",
    "<GAP>",
    "<SPEAK>",
    "<|SOT|> this is conversation",
    "<|SOT|> ... conversation <OVERLAP> What!? <BC_S>",
    "<OVERLAP> <BC_S> <BC_E> AUDIO1 AUDIO2 <LISTEN> <GAP> <SPEAK>",
]


def test_tokenization(text):
    tokens1 = tokenizer1.tokenize(text)
    tokens2 = tokenizer2.tokenize(text)

    assert isinstance(tokens1, list)
    assert isinstance(tokens2, list)

    print(f"\n---\nText: {text}")
    print(f"tokenizer1 gives: {tokens1}")
    print(f"tokenizer2 gives: {tokens2}")


def test_encode_decode(text):
    ids1 = tokenizer1.encode(text)
    ids2 = tokenizer2.encode(text)

    decode1 = tokenizer1.decode(ids1)
    decode2 = tokenizer2.decode(ids2)

    print(f"\n---\nText: {text}")
    print(f"tokenizer1 gives encode: {ids1}")
    print(f" which is decoded back to: {decode1}")
    print(f"tokenizer2 gives encode: {ids2}")
    print(f" which is decoded back to: {decode2}")


def load_tokenizer(model_id, set_pad_to_eos=True):
    try:
        print("Trying to load tokenizer...")
        # without legacy=False, extra white spaces are generated around the special tokens
        tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
        print(f"Tokenizer loaded!")
    except Exception as e:
        print(f"AutoTokenizer load failed: {e}")

    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def init_tokenizers(model_id, model2_id):
    global tokenizer1, tokenizer2
    tokenizer1 = load_tokenizer(model_id)
    tokenizer2 = load_tokenizer(model2_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="HF's model_id or cache_dir")
    parser.add_argument("--model2_id", type=str, help="HF's model_id or cache_dir")

    args = parser.parse_args()

    init_tokenizers(args.model_id, args.model2_id)

    list(map(test_tokenization, TEXT))
    list(map(test_encode_decode, TEXT))

    print("Finished!")


if __name__ == "__main__":
    main()
