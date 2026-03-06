import numpy as np
import os
import gzip
import struct
import urllib.request
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Direct download fallback (no keras / tensorflow required)
# ---------------------------------------------------------------------------

_URLS = {
    "mnist": {
        "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
    },
    "fashion_mnist": {
        "train_images": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz",
        "train_labels": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz",
        "test_images":  "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz",
        "test_labels":  "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz",
    },
}


def _parse_images(raw: bytes) -> np.ndarray:
    _, n, h, w = struct.unpack(">IIII", raw[:16])
    return np.frombuffer(raw[16:], dtype=np.uint8).reshape(n, h * w)


def _parse_labels(raw: bytes) -> np.ndarray:
    _, n = struct.unpack(">II", raw[:8])
    return np.frombuffer(raw[8:], dtype=np.uint8)


def _download_raw(dataset_name: str):
    """Download IDX files and return (train_images, train_labels, test_images, test_labels)."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".datasets", dataset_name)
    os.makedirs(cache_dir, exist_ok=True)

    urls = _URLS[dataset_name]
    data = {}
    for key, url in urls.items():
        fname  = url.split("/")[-1]
        fpath  = os.path.join(cache_dir, fname)
        if not os.path.exists(fpath):
            print(f"Downloading {fname} …")
            urllib.request.urlretrieve(url, fpath)
        with gzip.open(fpath, "rb") as f:
            raw = f.read()
        data[key] = raw

    x_tr = _parse_images(data["train_images"]).astype(np.float64) / 255.0
    y_tr = _parse_labels(data["train_labels"]).astype(np.int64)
    x_te = _parse_images(data["test_images"]).astype(np.float64) / 255.0
    y_te = _parse_labels(data["test_labels"]).astype(np.int64)
    return x_tr, y_tr, x_te, y_te


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(dataset: str = "mnist", val_split: float = 0.1, seed: int = 42):
    """
    Load MNIST or Fashion-MNIST.

    Tries keras/tensorflow first, falls back to direct download.

    Returns
    -------
    x_train, y_train, x_val, y_val, x_test, y_test
        x arrays : float64, shape (N, 784), values in [0, 1]
        y arrays : int64,   shape (N,)  — raw integer class labels
    """
    dataset = dataset.lower().replace("-", "_")
    if dataset not in ("mnist", "fashion_mnist"):
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'mnist' or 'fashion_mnist'.")

    x_tr = y_tr = x_te = y_te = None

    # --- Attempt 1: keras (may not be available) ---
    try:
        if dataset == "mnist":
            from keras.datasets import mnist as ds
        else:
            from keras.datasets import fashion_mnist as ds
        (x_tr, y_tr), (x_te, y_te) = ds.load_data()
        x_tr = x_tr.reshape(len(x_tr), -1).astype(np.float64) / 255.0
        x_te = x_te.reshape(len(x_te), -1).astype(np.float64) / 255.0
        y_tr = y_tr.astype(np.int64)
        y_te = y_te.astype(np.int64)
    except Exception:
        pass

    # --- Attempt 2: tensorflow.keras ---
    if x_tr is None:
        try:
            import tensorflow as tf
            if dataset == "mnist":
                (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
            else:
                (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
            x_tr = x_tr.reshape(len(x_tr), -1).astype(np.float64) / 255.0
            x_te = x_te.reshape(len(x_te), -1).astype(np.float64) / 255.0
            y_tr = y_tr.astype(np.int64)
            y_te = y_te.astype(np.int64)
        except Exception:
            pass

    # --- Attempt 3: direct download ---
    if x_tr is None:
        x_tr, y_tr, x_te, y_te = _download_raw(dataset)

    # Stratified validation split
    x_train, x_val, y_train, y_val = train_test_split(
        x_tr, y_tr,
        test_size=val_split,
        random_state=seed,
        stratify=y_tr,
    )

    print(
        f"Dataset: {dataset} | "
        f"Train: {x_train.shape[0]} | Val: {x_val.shape[0]} | Test: {x_te.shape[0]}"
    )
    return x_train, y_train, x_val, y_val, x_te, y_te
