import json
import logging
import os
import random

import numpy as np
import torch
from arguments.parse_arguments import parse_args
from model import load_model
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)

from data import load_data

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
set_seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class EvalLossLoggerCallback(TrainerCallback):
    def on_evaluate(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        **kwargs,
    ):
        # Only log if eval_loss is present
        if "eval_loss" in metrics:
            logging.info(
                f"[Eval @ epoch {state.epoch}] eval_loss: {metrics['eval_loss']:.4f}"
            )


def train(args, logger):
    logger.info("Welcome to behaviour-sd full-duplex training! :)")

    model_args, data_args, training_args = args

    model, tokenizer = load_model(
        model_args,
        grad_acc_steps=training_args.gradient_accumulation_steps,
        logger=logger,
    )

    train_dataset, eval_dataset, data_collator = load_data(
        model_args=model_args,
        data_args=data_args,
        audio_delay_id=model.audio_delay_id,
        logger=logger,
        tokenizer=tokenizer,
    )
    # Training arguments
    hf_training_args = TrainingArguments(
        output_dir=training_args.output_dir,
        eval_strategy=training_args.eval_strategy,
        eval_steps=training_args.eval_steps,
        save_strategy=training_args.save_strategy,
        save_steps=training_args.save_steps,
        logging_strategy=training_args.logging_strategy,
        logging_steps=training_args.logging_steps,
        report_to="wandb",
        run_name=model_args.model_name,
        per_device_train_batch_size=training_args.train_batch_size,
        per_device_eval_batch_size=training_args.eval_batch_size,
        max_steps=training_args.max_steps,
        num_train_epochs=training_args.num_train_epochs,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        gradient_checkpointing=training_args.gradient_checkpointing,
        max_grad_norm=training_args.gradient_clipping,
        weight_decay=training_args.weight_decay,
        save_total_limit=training_args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=training_args.learning_rate,
        fp16=training_args.use_fp16,
        bf16=training_args.use_bf16,
        auto_find_batch_size=training_args.auto_find_batch_size,
        remove_unused_columns=False,
    )

    # log the training args
    combined_config = {
        **vars(model_args),
        **vars(data_args),
        **vars(training_args),
        **hf_training_args.to_dict(),
    }
    config_path = os.path.join(training_args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(combined_config, f, indent=4)

    logger.info(f"Saved training config to {config_path}")
    logger.info("Starting training.")
    # Trainer
    trainer = Trainer(
        model=model,
        args=hf_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience
            ),
            EvalLossLoggerCallback(),
        ],
    )

    # Train
    trainer.train()
    logging.info(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    trainer.save_model(os.path.join(training_args.output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(training_args.output_dir, "best_model"))

    logger.info(f"Trainig finished: Final model saved to {training_args.output_dir}.")
    logger.info("All done! Bye :)")


def main():
    args = parse_args()
    model_args, data_args, training_args = args

    os.makedirs(training_args.output_dir, exist_ok=True)

    # Configure logging at the start of your script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(training_args.output_dir, "training.log")),
            logging.StreamHandler(),  # still log to console too
        ],
    )
    logger = logging.getLogger(__name__)

    train(args, logger)


if __name__ == "__main__":
    main()
