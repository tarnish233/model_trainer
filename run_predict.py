import csv
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# step = 1724

model_path = "/mnt/bn/fulei-v6-hl-nas-mlx/mlx/workspace/fake_vehicle_source/model_train_hf/model/d0906/mrc_for_cls_sample_focal_loss_a75_0906"

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2,
)
tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/bn/fulei-v6-hl-nas-mlx/mlx/workspace/models/chinese-macbert-large"
)

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

data_path = "/mnt/bn/fulei-v6-hl-nas-mlx/mlx/workspace/fake_vehicle_source/data/correct_data/d0904_im_data/process_data/hf_sample_data/fake_im_corrent_test_mrc_data.json"  # noqa: E501

write_path = "/mnt/bn/fulei-v6-hl-nas-mlx/mlx/workspace/fake_vehicle_source/data/correct_data/d0904_im_data/process_data/hf_sample_data/model_res"

if not os.path.exists(write_path):
    os.makedirs(write_path)

write_path = os.path.join(write_path, "fake_im_corrent_test_mrc_data.json")

id2label = {0: "已出售", 1: "未出售"}


def inference(query, context):
    with torch.no_grad():
        inputs = tokenizer(
            query, context, max_length=512, truncation=True, return_tensors="pt"
        )
        outputs = model(**inputs)
        logits = outputs.logits
        logits = logits.softmax(dim=-1)
        logits = logits.cpu().numpy()
        return logits


with open(data_path, encoding="utf-8-sig") as f_r, open(write_path, "w") as f_w:
    writer = csv.writer(f_w)
    if data_path.endswith("json"):
        writer.writerow(["conv", "label", "model", "score"])
        for line in f_r.readlines():
            line = json.loads(line.strip())
            query = line["query"]
            context = line["context"]
            label = line["label"]
            true_label = id2label[int(label)]
            inference(query, context)

            print(_res)
            res = _res["label"]
            model_label = res
            socre = _res["score"]
            # if socre < 0.95:
            #     res = "非软文"
            writer.writerow([context, true_label, model_label, socre])
