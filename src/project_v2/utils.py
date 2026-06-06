import torch
from sklearn.metrics import auc, average_precision_score, precision_recall_curve


def make_len_mask(inp):
    return inp == 0


def make_mask(mask):
    return (~mask).detach().type(torch.uint8)


def pr_auc_trapz_score(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def compute_pr_metrics(y_true, y_score):
    return {
        "ap": average_precision_score(y_true, y_score),
        "pr_auc": pr_auc_trapz_score(y_true, y_score),
    }


def auc_score(y_true, y_score, metric="ap"):
    metrics = compute_pr_metrics(y_true, y_score)
    if metric not in metrics:
        raise ValueError(f"Unsupported metric: {metric}")
    return metrics[metric]
