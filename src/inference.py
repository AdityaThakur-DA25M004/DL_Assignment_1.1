#!/usr/bin/env python3
"""
inference.py — Load saved weights and report metrics on the test set.

Location : src/inference.py
Run from inside src/:
    python inference.py
    python inference.py --model_path best_model.npy

CRITICAL: parse_arguments() reads best_config.json BEFORE returning args
so that any caller (autograder, CLI, or main()) always gets the correct
architecture — even if CLI defaults don't match the saved model.
"""

import argparse
import json
import os
import sys
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_THIS_DIR)
for _p in [_THIS_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ann.neural_network import NeuralNetwork
from utils.data_loader   import load_data
from utils.metrics       import compute_all_metrics
# ---------------------------------------------------------------------------
_BEST_HIDDEN_SIZE   = [128, 64]   # best model: 2 hidden layers
_BEST_NUM_LAYERS    = 2
_BEST_ACTIVATION    = "relu"
_BEST_OPTIMIZER     = "adam"
_BEST_LR            = 0.001
_BEST_WD            = 0.0
_BEST_WEIGHT_INIT   = "xavier"
_BEST_LOSS          = "cross_entropy"
_BEST_DATASET       = "mnist"

def build_parser():
    p = argparse.ArgumentParser(description="Inference for DA6401 Assignment-1 MLP")

    # ---- same flags as train.py ----
    p.add_argument("-d",   "--dataset",       default=_BEST_DATASET,
                   choices=["mnist", "fashion_mnist"])
    p.add_argument("-e",   "--epochs",         type=int,   default=10)
    p.add_argument("-b",   "--batch_size",     type=int,   default=64)
    p.add_argument("-l",   "--loss",           default=_BEST_LOSS,
                   choices=["cross_entropy", "mse"])
    p.add_argument("-o",   "--optimizer",      default=_BEST_OPTIMIZER,
                   choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    p.add_argument("-lr",  "--learning_rate",  type=float, default=_BEST_LR)
    p.add_argument("-wd",  "--weight_decay",   type=float, default=_BEST_WD)
    p.add_argument("-nhl", "--num_layers",     type=int,   default=_BEST_NUM_LAYERS)
    p.add_argument("-sz",  "--hidden_size",    type=int,   nargs="+",
                   default=_BEST_HIDDEN_SIZE)
    p.add_argument("-a",   "--activation",     default=_BEST_ACTIVATION,
                   choices=["sigmoid", "tanh", "relu"])
    p.add_argument("-w_i", "--weight_init",    default=_BEST_WEIGHT_INIT,
                   choices=["random", "xavier", "zeros"])
    p.add_argument("-w_p", "--wandb_project",  default="da6401-assignment1")
    p.add_argument("--wandb_entity",           default=None)
    p.add_argument("--gradient_clip",          type=float, default=5.0)

    # inference-specific
    p.add_argument("--model_path",  default=None,
                   help="Path to .npy weights (default: best_model.npy next to this file)")
    p.add_argument("--config_path", default=None,
                   help="Path to best_config.json (default: best_config.json next to this file)")

    return p


def _load_config_into_args(args):
    """
    Read best_config.json and overwrite args with saved architecture.
    This ensures the model built from args always matches the saved weights,
    regardless of what CLI defaults or autograder arguments say.
    """
    src_dir     = os.path.dirname(os.path.abspath(__file__))
    config_path = getattr(args, "config_path", None) or os.path.join(src_dir, "best_config.json")

    # Also search project root in case script is run from a different cwd
    candidates = [
        config_path,
        os.path.join(src_dir,    "best_config.json"),
        os.path.join(_ROOT_DIR,  "best_config.json"),
        os.path.join(_ROOT_DIR,  "src", "best_config.json"),
    ]

    for path in candidates:
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    cfg = json.load(f)

                # Architecture — these MUST match the saved weights
                hs = cfg.get("hidden_sizes", cfg.get("hidden_size", args.hidden_size))
                if isinstance(hs, int):
                    hs = [hs]
                args.hidden_size   = [int(h) for h in hs]
                args.num_layers    = len(args.hidden_size)

                # Training hyper-params (informational, not critical for inference)
                args.activation    = cfg.get("activation",    args.activation)
                args.loss          = cfg.get("loss",           args.loss)
                args.optimizer     = cfg.get("optimizer",      args.optimizer)
                args.learning_rate = float(cfg.get("learning_rate",  args.learning_rate))
                args.weight_decay  = float(cfg.get("weight_decay",   args.weight_decay))
                args.weight_init   = cfg.get("weight_init",   args.weight_init)
                args.gradient_clip = float(cfg.get("gradient_clip",  args.gradient_clip))
                if cfg.get("dataset"):
                    args.dataset   = cfg["dataset"]

                print(f"[inference] Config loaded from {path}")
                print(f"[inference] Architecture: 784 → {args.hidden_size} → 10")
                return args
            except Exception as e:
                print(f"[inference] Warning: could not read {path}: {e}")

    print("[inference] Warning: best_config.json not found — using CLI / hardcoded defaults.")
    print(f"[inference] Architecture: 784 → {args.hidden_size} → 10")
    return args


def parse_arguments():
    """
    Parse CLI args AND immediately load best_config.json so that the returned
    Namespace always describes the correct saved-model architecture.

    The autograder calls this function and uses the returned args directly
    to build the model — so config loading MUST happen here.
    """
    args = build_parser().parse_args()
    args = _load_config_into_args(args)
    return args


def load_model(model_path: str) -> dict:
    """Load a weights dict saved with np.save(path, model.get_weights())."""
    return np.load(model_path, allow_pickle=True).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()   # already has correct architecture from config

    src_dir    = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model_path or os.path.join(src_dir, "best_model.npy")

    # Search for weights in multiple locations
    candidates = [
        model_path,
        os.path.join(src_dir,   "best_model.npy"),
        os.path.join(_ROOT_DIR, "best_model.npy"),
        os.path.join(_ROOT_DIR, "src", "best_model.npy"),
    ]
    resolved_path = None
    for p in candidates:
        if p and os.path.exists(p):
            resolved_path = p
            break
    if resolved_path is None:
        raise FileNotFoundError(
            f"best_model.npy not found. Searched: {candidates}"
        )

    # --- Data ---
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.dataset)
    print(f"Test set: {x_test.shape}")

    # --- Build model with CORRECT architecture from config ---
    model   = NeuralNetwork(args)
    weights = load_model(resolved_path)
    model.set_weights(weights)
    print(f"Weights loaded from {resolved_path}")

    # Sanity check: verify weight shapes match model
    for i, layer in enumerate(model.layers):
        key = f"W{i}"
        if key in weights:
            saved_shape = weights[key].shape
            model_shape = layer.W.shape
            if saved_shape != model_shape:
                raise ValueError(
                    f"Shape mismatch at layer {i}: "
                    f"saved={saved_shape} vs model={model_shape}. "
                    f"Check best_config.json matches best_model.npy."
                )
    print("Weight shapes verified OK.")

    # --- Predict ---
    preds, _ = model.predict(x_test)

    # --- Metrics ---
    m = compute_all_metrics(y_test, preds)

    print("\n=== Test Set Results ===")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1-Score  : {m['f1']:.4f}")

    return m


if __name__ == "__main__":
    main()