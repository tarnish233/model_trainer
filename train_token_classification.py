import csv
import json
import logging
import os
import sys

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (  # DataCollatorForTokenClassification,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    set_seed,
)

import wandb
from arguments import DataTrainingArguments, ModelArguments, MyTrainingArguments
from logging_utils import logger
from modeling import (
    BertForTokenClassification,
    CustomerBertForSequenceClassification,
    MultiTaskBertForSequenceAndTokenClassification,
)
from tools import (
    eval_multi_label_classification_metrics,
    token_label_classification_metrics,
)

MODEL_CLASSES = {
    "bert": BertForSequenceClassification,
    "customerBert": CustomerBertForSequenceClassification,
    "multitaskbert": MultiTaskBertForSequenceAndTokenClassification,
    "tokenbert": BertForTokenClassification,
}


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_yaml_file(
        yaml_file=os.path.abspath(sys.argv[1])
    )
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


set_seed(training_args.seed)


def load_label2id(label2id_file):
    with open(label2id_file) as f:
        label2id = json.load(f)
        id2label = {value: key for key, value in label2id.items()}
        num_labels = len(label2id)
    return label2id, id2label, num_labels


def build_dataset(
    train_data_file,
    eval_data_file,
    test_data_file,
    cache_dir,
    seed,
    tokenizer,
    preprocessing_num_workers,
):
    datasets = load_dataset(
        "json",
        data_files={
            "train": [train_data_file],
            "eval": [eval_data_file],
            "test": [test_data_file],
        },
        cache_dir=cache_dir,
    )

    shuffled_datasets = datasets.shuffle(seed=seed)

    def process_function(examples):
        batch_tokens = [list(sentence) for sentence in examples["sentence"]]
        tokenized_examples = tokenizer(
            batch_tokens,
            max_length=data_args.max_length,
            truncation=data_args.truncation,
            is_split_into_words=True,
            # padding=data_args.padding,
        )
        # token label
        token_labels = []
        for i, label in enumerate(examples["token_labels"]):
            word_ids = tokenized_examples.word_ids(batch_index=i)
            token_label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    token_label_ids.append(-100)
                else:
                    token_label_ids.append(label[word_id])
            token_labels.append(token_label_ids)
        tokenized_examples["labels"] = token_labels

        return tokenized_examples

    tokenized_datasets = shuffled_datasets.map(
        process_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        # remove_columns=datasets["train"].column_names,
        remove_columns=["label"],
    )

    logger.info("tokenized_datasets: {}".format(tokenized_datasets))

    return tokenized_datasets


def load_class_modelmetrics():
    acc_metric = evaluate.load("accuracy")
    f1_metirc = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    return acc_metric, f1_metirc, precision_metric, recall_metric
    # evaluate_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def token_eval_metric(eval_predict):
    logits, labels = eval_predict
    token_predictions = np.argmax(logits, axis=-1)
    token_scores, _ = token_label_classification_metrics(
        id2label,
        token_predictions,
        labels,
        average="micro",
    )
    scores = {}
    scores["precision"] = token_scores["token_precision"]
    scores["recall"] = token_scores["token_recall"]
    scores["f1"] = token_scores["token_f1"]

    return scores


if __name__ == "__main__":
    label2id, id2label, num_labels = load_label2id(data_args.label2id_file)

    # mlb = MultiLabelBinarizer(classes=range(num_labels))
    config = BertConfig.from_pretrained(model_args.model_name_or_path)
    config.loss_type = model_args.loss_type
    config.num_labels = num_labels
    # label_weights
    config.label_weights = None
    if model_args.label_weights is not None:
        with open(model_args.label_weights) as f:
            data_count = json.load(f)
            train_data_count = list(data_count.values())
            config.label_weights = [
                max(train_data_count) / item if item != 0 else 0
                for item in train_data_count
            ]
    # load model
    model = MODEL_CLASSES[model_args.model_type].from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )
    model.config.id2label = id2label
    model.config.label2id = label2id
    tokenizer = BertTokenizerFast.from_pretrained(model_args.tokenizer_name_or_path)

    # load data
    tokenized_datasets = build_dataset(
        data_args.train_file,
        data_args.eval_file,
        data_args.test_file,
        data_args.cache_dir,
        training_args.data_seed,
        tokenizer,
        data_args.preprocessing_num_workers,
    )

    logger.info(tokenized_datasets["train"][0])
    # logger.info(len(tokenized_datasets["train"][0]["token_labels"]))
    # logger.info(len(tokenized_datasets["train"][0]["input_ids"]))

    my_data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    eval_metric = token_eval_metric

    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        data_collator=my_data_collator,
        compute_metrics=eval_metric,
    )
    if trainer.is_world_process_zero():
        logging.info("***** this is the global main process *****")
        logging.info("***** wandb init *****")
        wandb.init(
            project=training_args.wandb_project,
            name=training_args.wandb_name,
            notes=training_args.wandb_notes,
        )

    # train
    if training_args.do_train:
        logger.info("***** Running training *****")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
            logger.info("***** resume from checkpoint *****")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(tokenized_datasets["train"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        trainer.save_model()
        # save tokenizer
        tokenizer.save_pretrained(training_args.output_dir)

    # eval
    if training_args.do_eval:
        logger.info("***** Running evaluation *****")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(tokenized_datasets["eval"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("***** Running prediction *****")

        predictions = trainer.predict(tokenized_datasets["test"])
        metrics = predictions.metrics
        token_labels = predictions.label_ids
        token_logits = predictions.predictions

        # token_label
        token_predictions = np.argmax(token_logits, axis=-1)
        # 将预测结果转换为标签
        token_predictions = [
            [id2label[item] for item in label_ids] for label_ids in token_predictions
        ]
        token_labels = [
            [id2label[item] for item in label_ids] for label_ids in token_labels
        ]
        # 保存预测结果
        pd.DataFrame(
            {
                "sentence": tokenized_datasets["test"]["sentence"],
                "labels": token_labels,
                "pred_labels": token_predictions,
            }
        ).to_csv(
            os.path.join(training_args.output_dir, "test_data_token_predictions.csv"),
            mode="w",
            header=True,
        )

        # metrics
        token_scores, token_report = token_label_classification_metrics(
            id2label,
            token_predictions,
            token_labels,
            average="micro",
        )
        token_report = pd.DataFrame(token_report).T
        # 将precision、recall、f1保存为百分数
        token_report["precision"] = token_report["precision"].apply(
            lambda x: "{:.2f}%".format(x * 100)
        )
        token_report.to_csv(
            os.path.join(training_args.output_dir, "token_classification_report.csv"),
            mode="w",
            header=True,
        )

        metrics["test_samples"] = len(tokenized_datasets["test"])
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
