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


class DepthDecoderLRGateCallback(TrainerCallback):
    """Keeps the depth decoder's optimizer param group (see
    DSUTrainer.create_optimizer, always the last group) at lr=0 until
    global_step reaches unfreeze_after_steps, then leaves it alone so the
    normal LR scheduler (already tracking that group's own base
    depth_decoder_lr) takes over.

    This is a deliberate LR-only gate, not a requires_grad toggle: DeepSpeed
    ZeRO stage 1/2 flattens each optimizer param group's parameters into one
    fixed contiguous buffer at initialization and requires that group's set of
    trainable params to stay constant for the whole run (see
    CsmDepthDecoderHead's start_frozen docstring) - so the depth decoder's
    params are requires_grad=True from step 0 onward whenever this feature is
    used, and "frozen until step N" is simulated purely by forcing this
    group's lr to 0 every step beforehand, overriding whatever the scheduler
    just computed for it.

    Runs on `on_step_begin` (before that step's optimizer.step()) specifically
    so it wins against the scheduler's `on_step_end`-adjacent `.step()` call
    from the previous iteration, which would otherwise silently reactivate a
    nonzero lr for this group each step.

    Reads/writes `trainer.optimizer` directly rather than the `optimizer`
    kwarg callbacks receive - that kwarg is CallbackHandler's own copy,
    snapshotted once at construction time (before create_optimizer ever
    runs) and never refreshed, so it stays None/stale for the whole run.
    """

    def __init__(self, trainer, unfreeze_after_steps, logger):
        self.trainer = trainer
        self.unfreeze_after_steps = unfreeze_after_steps
        self.logger = logger
        self._activated = False

    def on_step_begin(self, args, state, control, **kwargs):
        if self._activated:
            return
        if state.global_step < self.unfreeze_after_steps:
            self.trainer.optimizer.param_groups[-1]["lr"] = 0.0
        else:
            self._activated = True
            self.logger.info(
                "Depth decoder optimizer group's LR schedule now active at "
                f"step {state.global_step}."
            )


class DSUTrainer(Trainer):
    """Trainer that, when depth_decoder_lr is set, gives depth_decoder_head's
    pretrained decoder its own optimizer param group at that LR - independent
    of the main learning_rate - instead of Trainer's default single-LR setup.

    The depth decoder's params are included regardless of requires_grad
    filtering (unlike the other two groups): whenever depth_decoder_lr is set,
    CsmDepthDecoderHead was constructed with start_frozen=False, so they're
    requires_grad=True for the whole run anyway (see its docstring for why -
    DeepSpeed ZeRO can't tolerate requires_grad changing after the optimizer
    is built). Actual gating of when this group starts learning is done via
    DepthDecoderLRGateCallback, not by excluding params here.
    """

    def __init__(self, *args, depth_decoder_lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_decoder_lr = depth_decoder_lr

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        if not (
            getattr(opt_model, "use_depth_decoder", False)
            and self.depth_decoder_lr is not None
        ):
            return super().create_optimizer()

        depth_prefix = "depth_decoder_head.depth_decoder."
        decay_parameters = self.get_decay_parameter_names(opt_model)

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if n in decay_parameters
                    and p.requires_grad
                    and not n.startswith(depth_prefix)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if n not in decay_parameters
                    and p.requires_grad
                    and not n.startswith(depth_prefix)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if n.startswith(depth_prefix)
                ],
                "weight_decay": 0.0,
                "lr": self.depth_decoder_lr,
            },
        ]

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, opt_model
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


def train(args, logger):
    logger.info("Welcome to behaviour-sd full-duplex training! :)")

    model_args, data_args, training_args = args

    model, tokenizer = load_model(
        model_args,
        grad_acc_steps=training_args.gradient_accumulation_steps,
        logger=logger,
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

    with hf_training_args.main_process_first():
        train_dataset, eval_dataset, data_collator = load_data(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            audio_delay_id=model.audio_delay_id,
            logger=logger,
            tokenizer=tokenizer,
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

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=training_args.early_stopping_patience
        ),
        EvalLossLoggerCallback(),
    ]

    unfreeze_depth_decoder = (
        model_args.use_depth_decoder
        and model_args.depth_decoder_unfreeze_after_steps > 0
    )

    # Trainer
    trainer = DSUTrainer(
        model=model,
        args=hf_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        depth_decoder_lr=(
            model_args.depth_decoder_unfreeze_lr if unfreeze_depth_decoder else None
        ),
    )

    if unfreeze_depth_decoder:
        # Added after construction (rather than passed into callbacks=) since
        # it needs a live reference to `trainer` itself - see
        # DepthDecoderLRGateCallback's docstring for why it can't just use the
        # `optimizer` kwarg callbacks normally receive.
        trainer.add_callback(
            DepthDecoderLRGateCallback(
                trainer, model_args.depth_decoder_unfreeze_after_steps, logger
            )
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
