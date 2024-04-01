from seqeval.metrics import accuracy_score as seqeval_accuracy_score
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.scheme import IOB2
from sklearn.metrics import accuracy_score, classification_report


def eval_multi_label_classification_metrics(
    predictions,
    references,
    labels=None,
    target_names=None,
    zero_division=0,
    digits=4,
    average="weighted",
):
    accuracy = accuracy_score(references, predictions)
    report = classification_report(
        references,
        predictions,
        labels=labels,
        target_names=target_names,
        zero_division=zero_division,
        digits=digits,
        output_dict=True,
    )
    scores = {}
    overall_score = report[average + " avg"]

    scores["precision"] = overall_score["precision"]
    scores["recall"] = overall_score["recall"]
    scores["f1"] = overall_score["f1-score"]
    scores["accuracy"] = accuracy

    return scores, report


def token_label_classification_metrics(
    token_id2label, y_preds, y_trues, average="weighted"
):
    """
    实体词任务的评价指标
    :param y_true: 真实标签，shape为(N, C)
    :param y_pred: 预测标签，shape为(N, C)
    :param average: 按哪种方式计算评价指标，可选'micro'、'macro'、'weighted'
    :return: 准确率（accuracy）、精确率（precision）、召回率（recall）、F1值（f1_score）
    """
    # print("y_preds", y_preds)
    # print("y_trues", y_trues)
    pred_labels = [
        [token_id2label[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(y_preds, y_trues)
    ]
    true_labels = [
        [token_id2label[l] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(y_preds, y_trues)
    ]
    # print("pred_labels", pred_labels)
    # print("pred_labels size: ", len(pred_labels))
    # print("true_labels", true_labels)
    # print("true_labels size: ", len(true_labels))

    accuracy = seqeval_accuracy_score(true_labels, pred_labels)
    report = seqeval_classification_report(
        true_labels,
        pred_labels,
        mode="strict",
        zero_division=0,
        digits=4,
        output_dict=True,
        scheme=IOB2,
    )
    scores = {}
    overall_score = report[average + " avg"]

    scores["token_precision"] = overall_score["precision"]
    scores["token_recall"] = overall_score["recall"]
    scores["token_f1"] = overall_score["f1-score"]
    scores["token_accuracy"] = accuracy

    if average not in ["micro", "macro", "weighted"]:
        raise ValueError("average parameter should be 'micro', 'macro', or 'weighted'.")
    return scores, report


if __name__ == "__main__":
    predictions = [1, 0, 0, 0, 1, 1, 0, 0, 2]
    references = [1, 0, 0, 0, 1, 1, 0, 0, 2]
    labels = ["label1", "label2", "label3"]
    scores, report = eval_multi_label_classification_metrics(
        predictions,
        references,
        labels=[0, 1, 2],
        target_names=labels,
        average="weighted",
    )
    print(scores)
    print(report)

    # import csv

    # f_r = "/mnt/bn/fulei-v6-hl-nas-mlx/mlx/workspace/model_trainer/data/data.csv"
    # r = open(f_r, "r", encoding="utf-8-sig")
    # reader = csv.reader(r, delimiter="\t")
    # predictions = []
    # references = []
    # labels = ["低", "中", "高"]
    # for line in reader:
    #     reference = line[0]
    #     if reference == "低":
    #         references.append([1, 0, 0])
    #     elif reference == "中":
    #         references.append([0, 1, 0])
    #     elif reference == "高":
    #         references.append([0, 0, 1])

    #     prediction = line[1]
    #     if prediction == "低":
    #         predictions.append([1, 0, 0])
    #     elif prediction == "中":
    #         predictions.append([0, 1, 0])
    #     elif prediction == "高":
    #         predictions.append([0, 0, 1])

    # scores, report = eval_multi_label_classification_metrics(
    #     predictions,
    #     references,
    #     labels=[0, 1, 2],
    #     target_names=labels,
    #     average="weighted",
    # )
    # print(scores)
    # print(report)
    # import pandas as pd

    # report = pd.DataFrame(report).transpose()
    # report.to_csv("report.csv")
