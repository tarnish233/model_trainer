from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="bert-base-chinese")
    tokenizer_name_or_path: Optional[str] = field(default="bert-base-chinese")
    use_fast_tokenizer: Optional[bool] = field(default=True)
    problem_type: Optional[str] = field(default="single_label_classification")
    loss_type: Optional[str] = field(default="CrossEntropyLoss")
    model_type: Optional[str] = field(default="bert")
    label_weights: Optional[str] = field(default=None)


@dataclass
class DataTrainingArguments:
    train_file: str = field(default=None)
    eval_file: str = field(default=None)
    test_file: str = field(default=None)
    label2id_file: str = field(default=None)
    token2id_file: str = field(default=None)
    max_length: int = field(default=128)
    preprocessing_num_workers: int = field(default=4)
    padding: Optional[bool] = field(default=True)
    truncation: Optional[bool] = field(default=True)
    cache_dir: Optional[str] = field(default=None)
    task_type: Optional[str] = field(
        default="classification",
        metadata={"help": "mrc_for_classification or classification"},
    )


@dataclass
class MyTrainingArguments(TrainingArguments):
    flash_attn: Optional[bool] = field(default=False)
    wandb_project: Optional[str] = field(default="default")
    wandb_name: Optional[str] = field(default="default")
    wandb_notes: Optional[str] = field(default="default")
