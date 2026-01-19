import json
import logging

import evaluate


def read_generated_output(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        samples = []
        for sample in root.findall("sample"):
            sample_data = sample.text.strip()
            samples.append(sample_data)
        return samples
    except ET.ParseError as e:
        raise ValueError(f"Error parsing XML: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")


def save_outputs_to_json(
    output_path,
    data_indices,
    generated_outputs_per_head,
    reference_outputs_per_head,
    generated_text_streams=None,
    reference_text_streams=None,
    text_stream_exists=False,
    meta=None,
):
    """
    Save generated and reference outputs (per head + optional text streams) into a single JSON file.
    """

    outputs_to_save = [
        (generated_outputs_per_head, reference_outputs_per_head, "head_")
    ]

    if text_stream_exists:
        outputs_to_save.append(
            (generated_text_streams, reference_text_streams, "text_stream_")
        )

    results = []
    num_examples = len(generated_outputs_per_head[0])
    assert num_examples == len(data_indices)

    for example_idx in range(num_examples):
        soda_index = data_indices[example_idx]
        example_dict = {"soda_index": soda_index}
        example_dict["instruction_s1"] = meta[soda_index]["instruction_s1"]
        example_dict["instruction_s2"] = meta[soda_index]["instruction_s2"]
        example_dict["narrative"] = meta[soda_index]["narrative"]
        example_dict["spk_emb1"] = meta[soda_index]["spk1"]
        example_dict["spk_emb2"] = meta[soda_index]["spk2"]

        for gen_list, ref_list, prefix in outputs_to_save:
            for head_idx, (gen, ref) in enumerate(zip(gen_list, ref_list)):
                example_dict[f"{prefix}{head_idx}"] = {
                    "generated": gen[example_idx],
                    "reference": ref[example_idx],
                }

        results.append(example_dict)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def save_judgements_to_json(samples, judgements, output_path):
    results = []
    for i, (sample, (response, score)) in enumerate(zip(samples, judgements)):
        results.append(
            {
                "index": i,
                "sample": sample,
                "response": response,
                "score": score,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def save_summary_to_json(judge, summary_dict, output_path):
    summary_dict["judge"] = judge  # just add the judge key
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False)


def average_results(judgements):
    valid_scores = []
    invalid_count = 0

    for _, score in judgements:
        if score.isdigit():
            valid_scores.append(int(score))
        else:
            invalid_count += 1
    total = len(judgements)
    num_valid = total - invalid_count
    avg_score = sum(valid_scores) / num_valid if num_valid > 0 else 0
    invalid_pct = (invalid_count / total) * 100
    return avg_score, invalid_pct, invalid_count, total


def compute_rouge_scores(all_generated_texts, reference_texts, logger):
    """Compute ROUGE scores per head and average."""
    rouge = evaluate.load("rouge")
    num_heads = len(all_generated_texts)
    head_scores = []

    for head_idx in range(num_heads):
        gen_text = all_generated_texts[head_idx]
        ref_text = reference_texts[head_idx]

        # Convert token lists to strings if needed
        if isinstance(gen_text[0][0], int):
            gen_text = [" ".join(map(str, seq)) for seq in gen_text]
        if isinstance(ref_text[0][0], int):
            ref_text = [" ".join(map(str, seq)) for seq in ref_text]

        # Compute ROUGE for this head
        scores = rouge.compute(predictions=gen_text, references=ref_text)
        head_scores.append(scores)
        logger.info(f"ROUGE for head {head_idx}: {scores}")

    avg_scores = {
        key: sum(head[key] for head in head_scores) / num_heads
        for key in head_scores[0]
    }

    return {
        "per_head": head_scores,
        "average": avg_scores,
    }


def compress_selected_pads(text_list, pad_tokens):
    """
    text_list: list of strings (or list of list of strings)
    pad_tokens: list of pad token names, e.g. ["UTTERANCE_PAD", "SILENCE_PAD"]

    Returns the same structure with consecutive repeats of the pad tokens
    replaced with 'Nx [PAD_TOKEN]'.
    """

    def compress(text):
        text_len = len(text)
        i = 0
        sequences = []  # list of (start, end, token)

        # Iterate through the string
        while i < text_len:
            for token in pad_tokens:
                pad_str = token
                if text.startswith(pad_str, i):
                    start = i
                    while i < text_len and text.startswith(pad_str, i):
                        i += len(pad_str)
                    sequences.append((start, i, pad_str))
                    break
            else:
                i += 1  # move forward if no pad matches

        # Replace sequences from the end
        text_list_chars = list(text)  # mutable
        for start, end, pad_str in reversed(sequences):
            count = (end - start) // len(pad_str)
            replacement = f" {count}x{pad_str} " if count > 1 else pad_str
            text_list_chars[start:end] = [
                replacement
            ]  # replace slice with single string
        return "".join(text_list_chars).strip()

    # handle nested list
    return [[compress(t) for t in sublist] for sublist in text_list]
