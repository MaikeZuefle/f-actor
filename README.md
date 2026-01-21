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

Models and datasets will be released on HuggingFace soon.

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

1. Adapt an existing training script to your needs using an example script from [`scripts/train`](scripts/train/). More parameters can be found in [`arguments.py`](training/arguments/arguments.py).

2. Run the training:
   ```bash
   bash scripts/train/your-train-script.sh
   ```

---

## Inference

Example inference scripts for generating dialogues using two instances of the model and prompts from Behavior-SD can be found in [`scripts/inference_eval`](scripts/inference_eval/). If you like to run inference with F-Actor from HuggingFace, please refer to `scripts/inference_eval/inference_nanocodec_special_tokens.sh`.

### Usage

1. Adapt an inference script to your needs using an example script from [`scripts/inference_eval`](scripts/inference_eval/).
2. Run inference:
   ```bash
   bash scripts/inference_eval/your-inference-script.sh
   ```
   The generated dialogues will be stored in the output directory that is specified in the script.

---

## Evaluation

To run the same evaluation metrics as reported in the paper:

1. Add the path of your model to the evaluation script in [`scripts/inference_eval`](scripts/inference_eval/).
   Add the output directory that was used during inference.

2. Run:
   ```bash
   bash scripts/inference_eval/eval.sh
   ```

---
## Generate Your Own Dialogues
You can generate custom dialogues using the script
[`training/inference_example.py`](training/inference_example.py).
Before running the script, configure the following options at the bottom of the file:

- **Speaker selection** (determines the voice used for each character)
  - Four example speaker voices from the original Behaviour-SD are provided below.
    Select any two of these voices for your dialogue.
    - **Tom**
      <audio controls src="training/example_speakers/tom.wav"></audio>
    - **Brian**
      <audio controls src="training/example_speakers/brian.wav"></audio>
    - **Gweneth**
      <audio controls src="training/example_speakers/gweneth.wav"></audio>
    - **Rebeka**
      <audio controls src="training/example_speakers/rebeka.wav"></audio>
- **Starting speaker** (which speaker begins the conversation)
- **Narrative context** (background or setup for the dialogue)

An example can be found in
[`training/inference_example.py`](training/inference_example.py).

To run the script, use:

```bash
python training/inference_example.py
```


## Example Dialogues

Here are three example dialogues generated with F-Actor. The corresponding prompts can be found in this file: [`example_prompts`](example_dialogues/instructions_for_example_1_2_3.txt).

<audio controls src="example_dialogues/example_1.wav"></audio>
<audio controls src="example_dialogues/example_2.wav"></audio>
<audio controls src="example_dialogues/example_3.wav"></audio>

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
