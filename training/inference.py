import logging
import os
import sys
import time

import torch
from arguments.parse_arguments import parse_args
from tqdm import tqdm
from transformers import set_seed

seed = 42
set_seed(seed, deterministic=True)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

from data import load_data
from inference_utils import (
    compress_selected_pads,
    compute_rouge_scores,
    save_outputs_to_json,
)
from model import load_model
from special_tokens import TEXT_STREAM_TOKENS


def compute_perplexity(
    model,
    tokenizer,
    dataset,
    logger,
    max_length=4096,
    text_stream_exists=False,
):

    logger.info("Computing perplexity on test dataset.")
    losses = []
    text_losses = []
    dsu_losses = []
    data_indices = []
    meta = {}

    for example in tqdm(dataset, desc="Calculate perplexity"):

        idx = example["soda_index"]
        data_indices.append(idx)
        meta[idx] = {
            "instruction_s1": example["input_text"],
            "instruction_s2": example["prompt_s2"],
            "narrative": example["orig_narrative"],
            "spk1": example["spk_emb1"],
            "spk2": example["spk_emb2"],
        }

        prompt = example["input_text"]
        ref_texts = example["reference_text"]
        ref_text_stream = example["reference_text_stream"]

        prompt_tokens = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )
        input_ids = prompt_tokens.input_ids.cuda()
        attention_mask = prompt_tokens.attention_mask.cuda()

        dsu_ids_list = [
            torch.tensor(x, dtype=torch.long).unsqueeze(0).cuda() for x in ref_texts
        ]

        dsu_ids = torch.stack(dsu_ids_list, dim=1).cuda()
        labels = None  # compute loss per head

        if text_stream_exists:
            ref_text_stream = (
                torch.tensor(ref_text_stream, dtype=torch.long).unsqueeze(0).cuda()
            )
            ref_text_stream = ref_text_stream[:, 0 : model.num_text_streams, :]

        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dsu_ids=dsu_ids,
                text_stream_ids=ref_text_stream,  # will be None if no ref_text
                spk_emb=[example["spk_emb1"]],  # wrap in list for batch size 1
                labels=labels,
                inference=True,
            )
            losses.append(outputs["loss"].item())
            dsu_losses.append(outputs["c1_dsu_loss"].item())
            if text_stream_exists:
                text_losses.append(outputs["c1_text_loss"].item())

    ppls = {}
    ppls["Perplexity"] = torch.exp(torch.tensor(losses).mean()).item()
    ppls["Perplexity DSU"] = torch.exp(torch.tensor(dsu_losses).mean()).item()
    if text_stream_exists:
        ppls["Perplexity Text"] = torch.exp(torch.tensor(text_losses).mean()).item()

    logger.info(", ".join(f"{k}: {v:.4f}" for k, v in ppls.items()))

    return ppls, data_indices, meta


def generate_outputs(
    model,
    tokenizer,
    dataset,
    max_length=2048,
    do_sample=False,
    temperature=1,
    top_k=0,
    top_p=1,
    use_speaker_sample=0,
    text_stream_exists=False,
    n_delay_text_stream=0,
    n_delay_audio_stream=0,
    talk_to_itself=False,
    return_gold=False,
):
    """Generate texts for all DSU heads. Returns list of lists: [head][examples]."""

    make_lists = lambda n: [[] for _ in range(max(1, n))]
    num_output_texts = 2 if (talk_to_itself or model.num_text_streams == 2) else 1
    all_generated_texts = make_lists(model.num_dsu_heads)
    all_reference_texts = make_lists(model.num_dsu_heads)
    generated_text_streams = make_lists(num_output_texts)
    ref_text_streams = make_lists(num_output_texts)

    for example in tqdm(dataset, desc="Generate Output"):

        ref_texts = example["reference_text"]
        if text_stream_exists:
            ref_text_stream = (
                torch.tensor(example["reference_text_stream"], dtype=torch.long)
                .unsqueeze(0)
                .cuda()
            )

        # Tokenize main input
        prompt = [example["input_text"]]
        spk_emb = [example["spk_emb1"]]

        if talk_to_itself:  # also give prompt + speaker embed to second speaker
            prompt.append(example["prompt_s2"])
            spk_emb.append(example["spk_emb2"])

        # breakpoint()
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )

        input_ids = inputs.input_ids.cuda()
        attention_mask = inputs.attention_mask.cuda()

        dsu_ids_list = [
            torch.tensor(x, dtype=torch.long).unsqueeze(0).cuda() for x in ref_texts
        ]

        dsu_sample = torch.stack(dsu_ids_list, dim=1).cuda()
        text_sample = ref_text_stream if text_stream_exists else None

        if not return_gold:
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    tokenizer=tokenizer,
                    stop_strings=["<|EOS|>"],
                    use_speaker_sample=use_speaker_sample,
                    dsu_sample=dsu_sample,
                    text_sample=text_sample,
                    n_delay_text_stream=n_delay_text_stream,
                    n_delay_audio_stream=n_delay_audio_stream,
                    talk_to_itself=talk_to_itself,
                    spk_emb=spk_emb,
                )

            input_len = input_ids.shape[1]

        else:
            generated_ids = (dsu_sample[0].tolist(), text_sample[0].tolist())

        if model.num_dsus > 0:
            generated_dsu_ids, generated_text_ids = generated_ids

            for head_idx in range(model.num_dsu_heads):
                gen_dsus = generated_dsu_ids[head_idx][n_delay_audio_stream:]

                all_generated_texts[head_idx].append(gen_dsus)
                all_reference_texts[head_idx].append(ref_texts[head_idx])

            if text_stream_exists:
                for ts_idx in range(num_output_texts):

                    gen_text = generated_text_ids[ts_idx][n_delay_text_stream:]
                    decoded_text_stream = tokenizer.decode(
                        gen_text,
                        skip_special_tokens=False,
                    )

                    decoded_text_ref = tokenizer.decode(
                        ref_text_stream[0][ts_idx],
                        skip_special_tokens=False,
                    )
                    generated_text_streams[ts_idx].append(decoded_text_stream)
                    ref_text_streams[ts_idx].append(decoded_text_ref)

        else:  # just text output
            gen_text = tokenizer.decode(
                generated_ids[0][input_len:],
                skip_special_tokens=False if model.num_dsus < 1 else True,
            )
            all_generated_texts[0].append(gen_text)
            all_reference_texts[0].append(ref_texts)

    return (
        all_generated_texts,
        all_reference_texts,
        generated_text_streams,
        ref_text_streams,
    )


def prepare_inference(model_args, data_args, inference_args, logger):
    model, tokenizer = load_model(model_args, logger=logger, inference=True)

    text_stream_exists = model_args.text_stream or model_args.multi_text_stream

    dataset = load_data(
        model_args=model_args,
        data_args=data_args,
        audio_delay_id=model.audio_delay_id,
        logger=logger,
        tokenizer=tokenizer,
        inference=True,
    )

    # Restrict dataset to a percentage if requested
    if inference_args.inference_on_subset:
        subset_size = max(
            1, int(len(dataset) * 0.01 * inference_args.inference_on_subset)
        )
        logger.info(
            f"Inference on {inference_args.inference_on_subset}% data: {subset_size} instances"
        )
        # dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(subset_size))

    if len(dataset) > 500:
        dataset = dataset.select(range(500))

    return model, tokenizer, text_stream_exists, dataset


def eval(model_args, data_args, inference_args, logger):
    logger.info(f"Starting Inference with with:\n{inference_args}")

    model, tokenizer, text_stream_exists, dataset = prepare_inference(
        model_args, data_args, inference_args, logger
    )
    # calculate ppl
    ppls, data_indices, meta = compute_perplexity(
        model,
        tokenizer,
        dataset,
        logger,
        text_stream_exists=text_stream_exists,
    )

    (
        generated_outputs_per_head,
        reference_outputs_per_head,
        generated_text_streams,
        ref_text_streams,
    ) = generate_outputs(
        model,
        tokenizer,
        dataset,
        max_length=model_args.max_length,
        do_sample=inference_args.do_sample,
        temperature=inference_args.temperature,
        top_k=inference_args.top_k,
        top_p=inference_args.top_p,
        use_speaker_sample=inference_args.use_speaker_sample,
        text_stream_exists=text_stream_exists,
        n_delay_text_stream=data_args.n_delay_text_stream,
        n_delay_audio_stream=data_args.n_delay_audio_stream,
        talk_to_itself=inference_args.talk_to_itself,
        return_gold=inference_args.return_gold,
    )

    # Compute ROUGE per head + average
    rouge_outputs = compute_rouge_scores(
        generated_outputs_per_head, reference_outputs_per_head, logger
    )
    rouge_text_stream = None

    if text_stream_exists:
        rouge_text_stream = compute_rouge_scores(
            generated_text_streams, ref_text_streams, logger
        )
        generated_text_streams = compress_selected_pads(
            generated_text_streams, TEXT_STREAM_TOKENS
        )
        ref_text_streams = compress_selected_pads(ref_text_streams, TEXT_STREAM_TOKENS)

    # save outputs
    output_dir = inference_args.inf_output_dir
    suffix = (
        f"-{inference_args.do_sample}"
        f"_T{inference_args.temperature:.2f}"
        f"_k{inference_args.top_k}"
        f"_p{inference_args.top_p:.2f}"
    ).replace(".", "_")
    gen_outputs_path = os.path.join(output_dir, f"outputs_samples{suffix}.json")

    save_outputs_to_json(
        gen_outputs_path,
        data_indices,
        generated_outputs_per_head,
        reference_outputs_per_head,
        generated_text_streams=generated_text_streams,
        reference_text_streams=ref_text_streams,
        text_stream_exists=text_stream_exists,
        meta=meta,
    )

    # save metric results
    results_path = os.path.join(output_dir, "eval_results.json")
    results = {
        "perplexities": ppls,
        "dsu_rouge": rouge_outputs,
    }
    if text_stream_exists:
        results["text_stream_rouge"] = rouge_text_stream

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(
        "\n=== Evaluation Summary ==="
        f"\nPerplexities:\n{ppls}"
        f"\n\nROUGE (Outputs):\n{rouge_outputs}"
        f"\n\nROUGE (Text Stream):\n{rouge_text_stream}"
        f"\n\nSaved results → {results_path}"
        f"\nSaved generations → {gen_outputs_path}"
        "\nAll done! Bye :)"
    )


def main():
    model_args, data_args, _, inference_args = parse_args(include_inference=True)

    os.makedirs(inference_args.inf_output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(inference_args.inf_output_dir, "eval.log")
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)
    eval(model_args, data_args, inference_args, logger)


if __name__ == "__main__":
    main()
