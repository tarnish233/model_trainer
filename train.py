import json
import logging
import os
import sys

import evaluate
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    set_seed,
)

from arguments import DataTrainingArguments, ModelArguments, MyTrainingArguments
from logging_utils import logger
from modeling import CustomerBertForSequenceClassification
from tools import eval_multi_label_classification_metrics

MODEL_CLASSES = {
    "bert": BertForSequenceClassification,
    "customerBert": CustomerBertForSequenceClassification,
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
    tokenizer_path,
    use_fast_tokenizer,
    preprocessing_num_workers,
    task_type="classification",
    problem_type="single_label_classification",
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

    tokenizer = BertTokenizer.from_pretrained(
        tokenizer_path, use_fast=use_fast_tokenizer
    )

    if task_type == "classification":

        def process_function(examples):
            tokenized_examples = tokenizer(
                examples["sentence"],
                max_length=data_args.max_length,
                truncation=data_args.truncation,
                # padding=data_args.padding,
            )
            if problem_type == "single_label_classification":
                tokenized_examples["labels"] = examples["label"]
            else:
                labels = mlb.fit_transform(examples["label"]).astype("float32")
                tokenized_examples["labels"] = labels

            return tokenized_examples

    elif task_type == "mrc_for_classification":

        def process_function(examples):
            tokenized_examples = tokenizer(
                examples["query"],
                examples["context"],
                max_length=data_args.max_length,
                truncation=data_args.truncation,
                # padding=data_args.padding,
            )
            if problem_type == "single_label_classification":
                tokenized_examples["labels"] = examples["label"]
            else:
                labels = mlb.fit_transform(examples["label"]).astype("float32")
                tokenized_examples["labels"] = labels
            return tokenized_examples

    tokenized_datasets = shuffled_datasets.map(
        process_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        # remove_columns=datasets["train"].column_names,
        remove_columns=["label"],
    )

    logger.info("tokenized_datasets: {}".format(tokenized_datasets))

    return tokenized_datasets, tokenizer


def load_class_modelmetrics():
    acc_metric = evaluate.load("accuracy")
    f1_metirc = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    return acc_metric, f1_metirc, precision_metric, recall_metric
    # evaluate_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def eval_metric(eval_predict):
    logits, labels = eval_predict
    if model_args.problem_type == "single_label_classification":
        predictions = np.argmax(logits, axis=-1)
    elif model_args.problem_type == "multi_label_classification":
        predictions = torch.sigmoid(torch.tensor(logits))
        predictions = (predictions > 0.5).int().numpy()
    scores, _ = eval_multi_label_classification_metrics(
        predictions,
        labels,
        labels=list(label2id.values()),
        target_names=list(label2id.keys()),
        average="micro",
    )
    return scores
    # acc = acc_metric.compute(predictions=predictions, references=labels)
    # f1 = f1_metirc.compute(predictions=predictions, references=labels, average="macro")
    # precision = precision_metric.compute(
    #     predictions=predictions, references=labels, average="macro"
    # )
    # recall = recall_metric.compute(
    #     predictions=predictions, references=labels, average="macro"
    # )
    # acc.update(f1)
    # acc.update(precision)
    # acc.update(recall)
    # return acc


if __name__ == "__main__":
    label2id, id2label, num_labels = load_label2id(data_args.label2id_file)
    mlb = MultiLabelBinarizer(classes=range(num_labels))
    config = BertConfig.from_pretrained(model_args.model_name_or_path)
    config.loss_type = model_args.loss_type
    config.problem_type = model_args.problem_type
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

    # load metrics
    acc_metric, f1_metirc, precision_metric, recall_metric = load_class_modelmetrics()

    # load data
    tokenized_datasets, tokenizer = build_dataset(
        data_args.train_file,
        data_args.eval_file,
        data_args.test_file,
        data_args.cache_dir,
        training_args.data_seed,
        model_args.tokenizer_name_or_path,
        model_args.use_fast_tokenizer,
        data_args.preprocessing_num_workers,
        data_args.task_type,
        model_args.problem_type,
    )

    logger.info(tokenized_datasets["train"][0])

    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
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
        label_ids = predictions.label_ids
        metrics = predictions.metrics
        predictions = predictions.predictions
        if model_args.problem_type == "single_label_classification":
            predictions = np.argmax(predictions, axis=-1)
        elif model_args.problem_type == "multi_label_classification":
            predictions = torch.sigmoid(torch.tensor(predictions))
            predictions = (predictions > 0.5).int().numpy()
            # 将预测结果转换为标签
            prediction_label_ids = [np.where(item == 1)[0] for item in predictions]
            with open(
                os.path.join(training_args.output_dir, "test_data_predictions.csv"), "w"
            ) as f:
                for sentence, item_label_ids in zip(
                    tokenized_datasets["test"]["sentence"], prediction_label_ids
                ):
                    labels = [id2label[item] for item in item_label_ids]
                    f.write(sentence + "\t" + ",".join(labels) + "\n")
        scores, report = eval_multi_label_classification_metrics(
            predictions,
            label_ids,
            labels=list(label2id.values()),
            target_names=list(label2id.keys()),
            average="weighted",
        )
        report = pd.DataFrame(report).T
        # 将precision、recall、f1保存为百分数
        report["precision"] = report["precision"].apply(
            lambda x: "{:.2f}%".format(x * 100)
        )
        report.to_csv(
            os.path.join(training_args.output_dir, "report.csv"), mode="w", header=True
        )
        metrics["test_samples"] = len(tokenized_datasets["test"])
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
