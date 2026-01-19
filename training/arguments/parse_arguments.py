import argparse
from dataclasses import dataclass
from typing import Optional, Union, get_args, get_origin

from arguments.arguments import DataArgs, InferenceArgs, ModelArgs, TrainingArgs


def parse_args(include_inference: bool = False):
    parser = argparse.ArgumentParser(description="Finetune Full-duplex Model")

    dataclasses = [
        ("model", ModelArgs),
        ("data", DataArgs),
        ("training", TrainingArgs),
    ]

    if include_inference:
        dataclasses.append(("inference", InferenceArgs))

    for group_name, cls in dataclasses:
        for field, field_def in cls.__dataclass_fields__.items():

            if field == "precision":
                parser.add_argument(
                    "--precision",
                    type=str,
                    choices=["fp32", "fp16", "bf16"],
                    default=field_def.default,
                    help="Mixed precision mode (fp32, fp16, bf16)",
                )
                continue

            # Unwrap Optional[...] types
            field_type = field_def.type
            origin = get_origin(field_type)
            if origin is Union:
                args = get_args(field_type)
                # Optional[T] is actually Union[T, NoneType]
                if type(None) in args:
                    # extract the actual type (exclude NoneType)
                    field_type = next(a for a in args if a is not type(None))

            # Boolean fields use store_true
            if field_type == bool:
                parser.add_argument(
                    f"--{field}",
                    action="store_true",
                    default=field_def.default,
                    help=f"{field} (flag, default: {field_def.default})",
                )
            else:
                parser.add_argument(
                    f"--{field}",
                    type=field_type,
                    default=field_def.default,
                    help=f"{field} (default: {field_def.default})",
                )

    args = parser.parse_args()

    if args.precision in ["fp16", "bf16"]:
        import torch

        assert (
            torch.cuda.is_available()
        ), "CUDA is required for mixed precision training."

    parsed = (
        ModelArgs(
            **{k: getattr(args, k) for k in ModelArgs.__dataclass_fields__.keys()}
        ),
        DataArgs(**{k: getattr(args, k) for k in DataArgs.__dataclass_fields__.keys()}),
        TrainingArgs(
            **{k: getattr(args, k) for k in TrainingArgs.__dataclass_fields__.keys()}
        ),
    )

    if include_inference:
        parsed += (
            InferenceArgs(
                **{
                    k: getattr(args, k)
                    for k in InferenceArgs.__dataclass_fields__.keys()
                }
            ),
        )

    return parsed


if __name__ == "__main__":
    model_args, data_args, training_args = parse_args()
    print(training_args)
