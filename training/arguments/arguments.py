from dataclasses import dataclass, field
from typing import Optional


# Model arguments
@dataclass
class ModelArgs:
    model_name: str = "eurollm-finetune-run"
    model_id: str = "utter-project/EuroLLM-1.7B"
    num_dsus: int = 0
    max_length: int = 4096
    text_stream: bool = False
    multi_text_stream: bool = False
    audio_vocab_size: int = 4032
    use_speaker_embedding: bool = False
    calc_loss_on_c1_only: bool = False
    first_codebook_weight: float = 1.0
    text_padding_weight: float = 1.0
    use_depth_decoder: bool = False
    depth_decoder_pretrained_path: str = "sesame/csm-1b"
    use_event_head: bool = False
    event_focal_gamma: float = 2.0
    # per-class focal-loss alpha, ordered [none, epad, bc, interrupt, eou];
    # None (the default) falls back to uniform weighting in DSUModel.__init__.
    event_focal_alpha: Optional[list] = field(default=None)

    def __post_init__(self):
        if self.text_stream and self.multi_text_stream:
            raise ValueError(
                "Only one of `text_stream` or `multi_text_stream` can be True, not both."
            )


# Data arguments
@dataclass
class DataArgs:
    # One or more HF dataset repos, comma-separated (e.g. "org/ds1,org/ds2").
    # Only the first dataset's "validation" split is used for eval; all
    # datasets' "train" splits are concatenated for training.
    speech_path: Optional[str] = None
    train_on_subset: Optional[float] = None
    n_delay_text_stream: int = 0
    n_delay_audio_stream: int = 0
    word_alignment: bool = False
    add_bc_token: bool = False
    add_interrupt_token: bool = False
    add_epad_token: bool = False
    add_eou_token: bool = False
    debug: bool = False
    use_system_narrative: bool = False
    remove_start_silence: bool = False
    preprocessing_num_workers: Optional[int] = None


# Training arguments
@dataclass
class TrainingArgs:
    train_batch_size: int = 1
    eval_batch_size: int = 1
    early_stopping_patience: int = 10
    max_steps: int = 100000
    num_train_epochs: int = 0
    learning_rate: float = 5e-5
    output_dir: str = "./outputs"
    gradient_accumulation_steps: int = 8
    gradient_clipping: float = 1.0
    gradient_checkpointing: bool = True
    auto_find_batch_size: bool = False
    precision: str = "bf16"  # choices: fp32, fp16, bf16
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 1000
    logging_strategy: str = "steps"
    logging_steps: int = 1
    weight_decay: float = 0.01
    save_total_limit: int = 3

    # convenience helpers
    @property
    def use_fp16(self) -> bool:
        return self.precision == "fp16"

    @property
    def use_bf16(self) -> bool:
        return self.precision == "bf16"


@dataclass
class InferenceArgs:
    inf_output_dir: str = "eval_output"
    inference_on_subset: Optional[float] = None  # percentage [0,100]
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    use_speaker_sample: int = 0
    talk_to_itself: bool = False
    return_gold: bool = False
