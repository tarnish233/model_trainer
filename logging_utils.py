import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("Model")


def metrics_logging(metrics, logger, mode="Train"):
    try:
        # label
        acc = metrics["label"]["accuracy"]
        precision = metrics["label"]["precision"]
        recall = metrics["label"]["recall"]
        f1 = metrics["label"]["f1"]
        logger.info(
            f"{mode} Accuracy: {(100*acc):>0.2f}%...\t{mode} Precision: {(100*precision):>0.2f}%...\t{mode} Recall:  {(100*recall):>0.2f}%...\t{mode} F1:  {(100*f1):>0.2f}%..."
        )
    except:
        pass
    try:
        # token label
        token_acc = metrics["token_label"]["accuracy"]
        token_precision = metrics["token_label"]["precision"]
        token_recall = metrics["token_label"]["recall"]
        token_f1 = metrics["token_label"]["f1"]
        logger.info(
            f"{mode} Token Accuracy: {(100*token_acc):>0.2f}%...\t{mode} Token Precision: {(100*token_precision):>0.2f}%...\t{mode} Token Recall:  {(100*token_recall):>0.2f}%...\t{mode} Token F1:  {(100*token_f1):>0.2f}%..."
        )
    except:
        pass
