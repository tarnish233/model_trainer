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


if __name__ == "__main__":
    # predictions = [
    #     [0, 1, 1, 0],
    #     [1, 0, 1, 0],
    #     [0, 1, 1, 0],
    #     [1, 0, 1, 0],
    # ]
    # references = [
    #     [0, 1, 1, 0],
    #     [1, 0, 1, 0],
    #     [0, 1, 1, 0],
    #     [1, 0, 1, 0],
    # ]
    # labels = ["label1", "label2", "label3", "label4"]
    # scores, report = eval_multi_label_classification_metrics(
    #     predictions, references, labels, average="weighted"
    # )
    # print(scores)
    # print(report)
    import csv

    f_r = "/mnt/bn/fulei-v6-hl-nas-mlx/mlx/workspace/model_trainer/data/data.csv"
    r = open(f_r, "r", encoding="utf-8-sig")
    reader = csv.reader(r, delimiter="\t")
    predictions = []
    references = []
    labels = ["低", "中", "高"]
    for line in reader:
        reference = line[0]
        if reference == "低":
            references.append([1, 0, 0])
        elif reference == "中":
            references.append([0, 1, 0])
        elif reference == "高":
            references.append([0, 0, 1])

        prediction = line[1]
        if prediction == "低":
            predictions.append([1, 0, 0])
        elif prediction == "中":
            predictions.append([0, 1, 0])
        elif prediction == "高":
            predictions.append([0, 0, 1])

    scores, report = eval_multi_label_classification_metrics(
        predictions,
        references,
        labels=[0, 1, 2],
        target_names=labels,
        average="weighted",
    )
    import pandas as pd

    report = pd.DataFrame(report).transpose()
    report.to_csv("report.csv")
