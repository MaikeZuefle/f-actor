from dataclasses import dataclass
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

    def __post_init__(self):
        if self.text_stream and self.multi_text_stream:
            raise ValueError(
                "Only one of `text_stream` or `multi_text_stream` can be True, not both."
            )


# Data arguments
@dataclass
class DataArgs:
    speech_path: Optional[str] = None
    train_on_subset: Optional[float] = None
    n_delay_text_stream: int = 0
    n_delay_audio_stream: int = 0
    word_alignment: bool = False
    add_bc_token: bool = False
    add_interrupt_token: bool = False
    add_counting_tokens: bool = False
    debug: bool = False
    use_system_narrative: bool = False
    remove_start_silence: bool = False

    def __post_init__(self):
        if (self.add_bc_token and self.add_counting_tokens) or (
            self.add_interrupt_token and self.add_counting_tokens
        ):
            raise ValueError(
                "add_counting_tokens cannot be set together with add_bc_token/add_interrupt_token."
            )


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
