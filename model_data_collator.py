from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass
class DataCollatorForSequenceAndTokenClassification(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        token_label_name = (
            "token_labels" if "token_labels" in features[0].keys() else None
        )
        token_labels = (
            [feature[token_label_name] for feature in features]
            if token_label_name in features[0].keys()
            else None
        )

        no_token_labels_features = [
            {k: v for k, v in feature.items() if k != token_label_name}
            for feature in features
        ]

        batch = self.tokenizer.pad(
            no_token_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if token_labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        if padding_side == "right":
            batch[token_label_name] = [
                to_list(label)
                + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in token_labels
            ]
        else:
            batch[token_label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label))
                + to_list(label)
                for label in token_labels
            ]

        batch[token_label_name] = torch.tensor(
            batch[token_label_name], dtype=torch.int64
        )
        return batch
