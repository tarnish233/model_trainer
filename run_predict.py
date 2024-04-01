import argparse
import csv
import json
import os

import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from tools import eval_multi_label_classification_metrics

args = argparse.ArgumentParser()
args.add_argument("--data_file", type=str, default="data.csv")
args.add_argument("--output_file", type=str, default="data_model_res.csv")
args.add_argument("--model_path", type=str)
args.add_argument("--tokenizer_path", type=str)
args.add_argument("--label2id", type=str, default="label2id.json")
args.add_argument("--max_length", type=int, default=512)
args.add_argument("--problem_type", type=str, default="single_label_classification")
args = args.parse_args()


def inference(
    sentences,
    model,
    tokenizer,
    problem_type="single_label_classification",
):
    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        if problem_type == "single_label_classification":
            logits = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1).cpu().numpy().tolist()
        elif problem_type == "multi_label_classification":
            logits = torch.sigmoid(logits)
            pred = (logits > 0.5).int().cpu().numpy()[0]
        return pred


def main(data_file, output_dir, problem_type, model, tokenizer):
    model_res_file = os.path.join(output_dir, "model_res.csv")
    model_metrics_file = os.path.join(output_dir, "model_metrics.csv")
    with open(data_file, encoding="utf-8-sig") as f_r, open(model_res_file, "w") as f_w:
        writer = csv.DictWriter(f_w, fieldnames=["sentence", "label", "model"])
        writer.writeheader()
        all_labels, all_probs = [], []
        for line in tqdm(f_r.readlines()):
            line = json.loads(line.strip())
            sentence = line["sentence"]
            label = line["label"]
            pred = inference(sentence, model, tokenizer, problem_type)
            if problem_type == "single_label_classification":
                all_labels.append(int(label))
                all_probs.append(pred)
                pred_label = id2label[pred]
                true_label = id2label[int(label)]

            elif problem_type == "multi_label_classification":
                all_probs.append(pred)
                print(label)
                true_label_id = mlb.fit_transform(label).astype("float32")
                all_labels.append(true_label_id)
                pred = pred.nonzero()[0].tolist()
                true_label = [id2label[int(l)] for l in label]
                pred_label = [id2label[p] for p in pred]
            writer.writerow(
                {"sentence": sentence, "label": true_label, "model": pred_label}
            )

    scores, report = eval_multi_label_classification_metrics(
        all_probs,
        all_labels,
        labels=list(label2id.values()),
        target_names=list(label2id.keys()),
        average="weighted",
    )
    report = pd.DataFrame(report).transpose()
    report.to_csv(model_metrics_file)


if __name__ == "__main__":
    with open(args.label2id, "r") as f:
        label2id = json.load(f)
        id2label = {v: k for k, v in label2id.items()}
        num_labels = len(label2id)

    mlb = MultiLabelBinarizer(classes=range(num_labels))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=num_labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    main(args.data_file, args.output_file, args.problem_type, model, tokenizer)
    # sentence = "销售：您好，欢迎咨询[马自达CX-5 2017款 2.0L 自动两驱智享型 国V]，请问有什么可以帮您？\n客户：这辆车车况如何？\n客户：这辆车车况如何？\n客户：这辆车到手价多少？\n销售：当前不支持查看此类消息，请使用新版懂车帝APP查看\n客户：我的联系方式已提交，请优先加微信沟通吧，微信号同手机号\n销售：收到，我会尽快联系您\n客户：车在哪\n客户：在吗\n销售：车刚卖 正在办理过户 还没来得及下架\n客户：好的👌🏻"
    # res = inference(sentence, None, model, tokenizer, args.problem_type)
    # print(res)
