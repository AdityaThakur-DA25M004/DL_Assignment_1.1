import numpy as np


def _to_one_hot(y, num_classes):
    """Convert integer labels (N,) → one-hot (N, C). Passes through if already 2-D."""
    y = np.asarray(y)
    if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
        y = y.ravel().astype(int)
        oh = np.zeros((len(y), num_classes), dtype=np.float64)
        oh[np.arange(len(y)), y] = 1.0
        return oh
    return y.astype(np.float64)


def softmax(logits):
    """Numerically stable row-wise softmax."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MeanSquaredError:
    """
    MSE loss.
    forward(y_pred, y_true) — y_pred are raw logits; y_true are integers or one-hot.
    backward(y_pred, y_true) — returns dL/d(logits).
    """

    def forward(self, y_pred, y_true):
        N, C = y_pred.shape
        y_oh = _to_one_hot(y_true, C)
        return float(np.mean((y_pred - y_oh) ** 2))

    def backward(self, y_pred, y_true):
        """dL/d(y_pred) for MSE: np.mean averages over N*C elements."""
        N, C = y_pred.shape
        y_oh = _to_one_hot(y_true, C)
        # np.mean sums over N*C, so gradient = 2*(y_pred - y_oh) / (N*C)
        return (2.0 / (N * C)) * (y_pred - y_oh)


class CrossEntropyLoss:
    """
    Softmax cross-entropy.
    forward() — computes softmax internally, returns scalar loss.
    backward() — returns (softmax(logits) - y_one_hot) / N  (gradient at logits).
    """

    def forward(self, logits, y_true):
        probs = softmax(logits)
        N, C  = probs.shape
        y_oh  = _to_one_hot(y_true, C)
        log_p = np.log(np.clip(probs, 1e-12, 1.0))
        return float(-np.sum(y_oh * log_p) / N)

    def backward(self, logits, y_true):
        """Returns dL/d(logits) = (softmax(logits) - y_oh) / N,  shape (N, C)."""
        probs = softmax(logits)
        N, C  = probs.shape
        y_oh  = _to_one_hot(y_true, C)
        return (probs - y_oh) / N


def get_loss(name: str):
    name = name.lower()
    if name in ("mse", "mean_squared_error"):
        return MeanSquaredError()
    elif name in ("cross_entropy", "ce"):
        return CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss: {name}. Choose 'cross_entropy' or 'mse'.")
