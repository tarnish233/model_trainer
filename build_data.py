import argparse
import csv
import json
import os
import random

import matplotlib.pyplot as plt
from tqdm import tqdm

random.seed(42)

args = argparse.ArgumentParser()
args.add_argument("--data_file", type=str, default="data.csv")
args.add_argument("--output_dir", type=str, default="data")
args.add_argument("--train_ratio", type=float, default=0.8)
args.add_argument("--dev_ratio", type=float, default=0.1)
args.add_argument("--test_ratio", type=float, default=0.1)
args.add_argument("--label2id", type=str, default="label2id.json")
args.add_argument("--problem_type", type=str, default="single_label_classification")

args = args.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)


def read_data(data_file, problem_type, label2id):
    data = []
    with open(data_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for line in reader:
            sentence = line["sentence"]
            label = line["label"]
            if problem_type == "single_label_classification":
                print(sentence + "," + label)
                labelid = label2id[label] if label in label2id else label2id["无意图标签"]
            elif problem_type == "multi_label_classification":
                label = label.split(",")
                labelid = [label2id[l] for l in label if l in label2id]
                if labelid == []:
                    labelid = [label2id["无意图标签"]]
            data.append({"sentence": sentence, "label": labelid})
    return data


def write_data(output_dir, data, data_ratio):
    train_data_file = os.path.join(output_dir, "train.json")
    eval_data_file = os.path.join(output_dir, "eval.json")
    test_data_file = os.path.join(output_dir, "test.json")
    label_weight_file = os.path.join(output_dir, "label_weight.json")
    sentence_len_file = os.path.join(output_dir, "sentence_len.png")

    sentence_len = []
    label_weight = {}

    random.shuffle(data)
    with open(train_data_file, "w") as f_train, open(
        eval_data_file, "w"
    ) as f_eval, open(test_data_file, "w") as f_test:
        for line in data:
            # label_wight
            label = line["label"]
            if isinstance(label, list):
                for l in label:
                    if l not in label_weight:
                        label_weight[l] = 1
                    else:
                        label_weight[l] += 1
            else:
                if label not in label_weight:
                    label_weight[label] = 1
                else:
                    label_weight[label] += 1
            # sentence_len
            sentence = line["sentence"]
            sentence_len.append(len(sentence))

            random_num = random.random()
            if random_num <= data_ratio["train"]:
                f_train.write(json.dumps(line, ensure_ascii=False) + "\n")
            elif random_num > data_ratio["train"] and random_num <= (
                data_ratio["train"] + data_ratio["eval"]
            ):
                f_eval.write(json.dumps(line, ensure_ascii=False) + "\n")
            else:
                f_test.write(json.dumps(line, ensure_ascii=False) + "\n")

    with open(label_weight_file, "w") as f:
        json.dump(label_weight, f, ensure_ascii=False, indent=4)

    with open(sentence_len_file, "wb") as f:
        plt.hist(sentence_len, bins=100)
        plt.savefig(f)


if __name__ == "__main__":
    with open(args.label2id, "r") as f:
        label2id = json.load(f)

    data_ratio = {
        "train": args.train_ratio,
        "eval": args.dev_ratio,
        "test": args.test_ratio,
    }
    data = read_data(args.data_file, args.problem_type, label2id)
    write_data(args.output_dir, data, data_ratio)
