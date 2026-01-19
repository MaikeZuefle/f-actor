import os

import torch
from huggingface_hub import snapshot_download
from modeling_dsu import DSULlama
from special_tokens import TEXT_STREAM_TOKENS
from transformers import AutoConfig, AutoTokenizer


def add_new_tokens(model, tokenizer, new_tokens, new_special_tokens, logger=None):
    resized = False
    if new_tokens:
        added = tokenizer.add_tokens(new_tokens)
        if added > 0:
            logger.info(f"Added {added} new tokens.")
            resized = True

    if new_special_tokens:
        special_token_dict = {"additional_special_tokens": new_special_tokens}
        added = tokenizer.add_special_tokens(special_token_dict)
        if added > 0:
            logger.info(f"Added {added} special tokens.")
            resized = True

    if resized:
        model.resize_token_embeddings(len(tokenizer))
        model.text_vocab_size = model.get_output_embeddings().weight.size(0)
        logger.info("Resized token embeddings.")


def load_model(model_args, grad_acc_steps=1, logger=None, inference=False):

    model_id = model_args.model_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model and tokenizer from '{model_id}'.")

    if model_id != "meta-llama/Llama-3.2-1B-Instruct" and not os.path.isdir(model_id):
        logger.info(
            f"Model ID '{model_id}' is not a local directory. Downloading from HuggingFace Hub..."
        )
        model_id = snapshot_download(
            repo_id=model_id, local_dir=f"./hf_models/{model_id.replace('/', '_')}"
        )
        logger.info(f"Downloaded to '{model_id}'.")

    best_model_path = os.path.join(model_id, "best_model")
    if os.path.isdir(best_model_path):
        logger.info(f"Loading best checkpoint '{best_model_path}'.")
        model_id = best_model_path

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, legacy=False, rust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model_cls = DSULlama

    # add arguments to config
    config = AutoConfig.from_pretrained(model_id)
    config.num_dsus = model_args.num_dsus
    config.text_stream = model_args.text_stream
    config.multi_text_stream = model_args.multi_text_stream
    config.audio_vocab_size = model_args.audio_vocab_size
    config.use_speaker_embedding = model_args.use_speaker_embedding
    config.calc_loss_on_c1_only = model_args.calc_loss_on_c1_only

    # load model (if num_dsu < 1, this will be the normal model)
    model = model_cls.from_pretrained(
        model_id,
        config=config,
    ).to(device)

    if tokenizer.pad_token_id < model_args.audio_vocab_size:
        # text tokenier pad also used to pad audio, so should not be in audio vocab
        new_pad = "<EOS>"
        add_new_tokens(
            model, tokenizer, new_tokens=[], new_special_tokens=[new_pad], logger=logger
        )
        tokenizer.eos_token = new_pad
        tokenizer.pad_token = tokenizer.eos_token

    # set some useful paremters needed for forward passes
    model.pad_token_id = tokenizer.pad_token_id
    model.grad_acc_steps = grad_acc_steps

    if model_args.text_stream or model_args.multi_text_stream:
        add_new_tokens(
            model,
            tokenizer,
            new_tokens=[],
            new_special_tokens=TEXT_STREAM_TOKENS,
            logger=logger,
        )

    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        model.text_vocab_size = model.get_output_embeddings().weight.size(0)
        logger.info(
            "#model_embeddings < #tokenizer_embeddings --> resize model_embeddings."
        )
        assert (
            model.get_input_embeddings().weight is model.get_output_embeddings().weight
        )

    if model_args.use_speaker_embedding:
        model.init_or_load_speaker_embed_proj(model_path=model_id)

    if model_args.multi_text_stream:
        model.init_or_load_text_heads(model_path=model_id)

    if model.num_dsus > 0:
        model.init_or_load_audio_heads(
            model_path=model_id
        )  # loads if dsu_head exists, else initializes
        model.init_or_load_audio_embeds(
            model_path=model_id
        )  # loads if audio_embeds exist, else initializes

    if inference:
        model.eval()

    return model, tokenizer
