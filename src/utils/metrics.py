import numpy as np


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred, num_classes: int = 10):
    """
    Macro-averaged precision, recall, and F1-score.
    Returns: (precision, recall, f1) — all Python floats.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return (
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(f1s)),
    )


def compute_all_metrics(y_true, y_pred, num_classes: int = 10):
    acc           = accuracy(y_true, y_pred)
    prec, rec, f1 = precision_recall_f1(y_true, y_pred, num_classes)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
