# F-Actor: Controllable Conversational Behavior in Full-Duplex Models

[![arXiv](https://img.shields.io/badge/arXiv-2601.11329-b31b1b.svg)](https://arxiv.org/abs/2601.11329)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/maikezu/f-actor)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/maikezu/f-actor-behavior-sd-nanocodec)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/maikezu/f-actor-behavior-sd-mimi)


> **Work in Progress**

## Overview

This repository contains the code accompanying the paper
**[F-Actor: Controllable Conversational Behaviour in Full-Duplex Models](https://arxiv.org/abs/2601.11329)**.

Spoken conversational systems require more than accurate speech generation to have human-like conversations: to feel natural and engaging, they must produce conversational behaviour that adapts dynamically to the context. Current spoken conversational systems, however, rarely allow such customization, limiting their naturalness and usability. In this work, we present the first open, instruction-following full-duplex conversational speech model that can be trained efficiently under typical academic resource constraints. By keeping the audio encoder frozen and finetuning only the language model, our model requires just 2,000 hours of data, without relying on large-scale pretraining or multi-stage optimization. The model can follow explicit instructions to control speaker voice, conversation topic, conversational behaviour (e.g., backchanneling and interruptions), and dialogue initiation. We propose a single-stage training protocol and systematically analyze design choices. Both the model and training code will be released to enable reproducible research on controllable full-duplex speech systems.


## Released Resources

- 🤗 **Model**:  https://huggingface.co/maikezu/f-actor

- 🤗 **Dataset (Behavior-SD, NanoCodec)**: https://huggingface.co/datasets/maikezu/f-actor-behavior-sd-nanocodec

- 🤗 **Dataset (Behavior-SD, Mimi)**: https://huggingface.co/datasets/maikezu/f-actor-behavior-sd-mimi
---

## Requirements

```
conda create -n factor python=3.10
conda activate factor
cd f-actor
pip install .
```

## Training

Example training scripts are located in [`scripts/train`](scripts/train/).

### Usage

1. Adapt an existing training script to your needs:
   ```
   scripts/train/your-train-script.sh
   ```

2. Run the training:
   ```bash
   bash scripts/train/your-train-script.sh
   ```

---

## Inference

Example inference scripts for generating dialogues using two instances of the model and prompts from Behavior-SD can be found in [`scripts/inference_eval`](scripts/inference_eval/). If you like to run inference with F-Actor from HuggingFace, please refer to `scripts/inference_eval/inference_nanocodec_special_tokens.sh`.

### Usage

1. Adapt an inference script:
   ```
   scripts/inference_eval/your-inference-script.sh
   ```

2. Run inference:
   ```bash
   bash scripts/inference_eval/your-inference-script.sh
   ```
   The generated stores will be stored in the output directory that is specified in the script.

---

## Evaluation

To run the same evaluation metrics as reported in the paper:

1. Adapt the evaluation script:
   ```
   scripts/inference_eval/eval.sh
   ```
   Add the output directory that was used during inference, where the genereated dialogues are stored.

2. Run:
   ```bash
   bash scripts/inference_eval/eval.sh
   ```

---

## Example Dialogues

Example dialogues generated with F-Actor can be found in the [`example_dialogues`](example_dialogues/) folder.

---
## Citation

If you use this work, please cite:

```bibtex
@misc{züfle2026factorcontrollableconversationalbehaviour,
      title={F-Actor: Controllable Conversational Behaviour in Full-Duplex Models},
      author={Maike Züfle and Ondrej Klejch and Nicholas Sanders and Jan Niehues and Alexandra Birch and Tsz Kin Lam},
      year={2026},
      eprint={2601.11329},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.11329},
}
```
