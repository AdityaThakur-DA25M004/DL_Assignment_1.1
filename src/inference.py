#!/usr/bin/env python3
"""
inference.py — Load saved weights and report metrics on the test set.

Location : src/inference.py
Run from inside src/:
    python inference.py
    python inference.py --model_path best_model.npy
"""

import argparse
import json
import os
import sys
import numpy as np

# Ensure both src/ and project root are on the path.
# Handles running as `python src/inference.py` (from root) or
# `python inference.py` (from inside src/).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_THIS_DIR)
for _p in [_THIS_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ann.neural_network import NeuralNetwork
from utils.data_loader   import load_data
from utils.metrics       import compute_all_metrics


# ---------------------------------------------------------------------------
# CLI — mirrors train.py exactly (best config values as defaults)
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Inference for DA6401 Assignment-1 MLP")

    p.add_argument("-d",   "--dataset",       default="fashion_mnist",
                   choices=["mnist", "fashion_mnist"])
    p.add_argument("-e",   "--epochs",         type=int,   default=10)
    p.add_argument("-b",   "--batch_size",     type=int,   default=64)
    p.add_argument("-l",   "--loss",           default="cross_entropy",
                   choices=["cross_entropy", "mse"])
    p.add_argument("-o",   "--optimizer",      default="adam",
                   choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    p.add_argument("-lr",  "--learning_rate",  type=float, default=0.001)
    p.add_argument("-wd",  "--weight_decay",   type=float, default=1e-4)
    p.add_argument("-nhl", "--num_layers",     type=int,   default=3)
    p.add_argument("-sz",  "--hidden_size",    type=int,   nargs="+", default=[128])
    p.add_argument("-a",   "--activation",     default="relu",
                   choices=["sigmoid", "tanh", "relu"])
    p.add_argument("-w_i", "--weight_init",    default="xavier",
                   choices=["random", "xavier", "zeros"])
    p.add_argument("-w_p", "--wandb_project",  default="da6401-assignment1")
    p.add_argument("--wandb_entity",           default=None)
    p.add_argument("--gradient_clip",          type=float, default=5.0)

    # Inference-specific paths
    p.add_argument("--model_path",  default=None,
                   help="Path to .npy weights file (default: best_model.npy in src/)")
    p.add_argument("--config_path", default=None,
                   help="Path to best_config.json (default: best_config.json in src/)")

    return p


# Keep this name so autograder can call inference.parse_arguments()
def parse_arguments():
    return build_parser().parse_args()


def load_model(model_path: str) -> dict:
    """Load a weights dict saved with np.save(path, model.get_weights())."""
    return np.load(model_path, allow_pickle=True).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()

    # Expand single hidden_size to num_layers copies
    if len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size * args.num_layers

    src_dir     = os.path.dirname(os.path.abspath(__file__))
    model_path  = args.model_path  or os.path.join(src_dir, "best_model.npy")
    config_path = args.config_path or os.path.join(src_dir, "best_config.json")

    # Override architecture from saved config if available
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        args.hidden_size   = cfg.get("hidden_sizes",   args.hidden_size)
        args.activation    = cfg.get("activation",     args.activation)
        args.loss          = cfg.get("loss",            args.loss)
        args.optimizer     = cfg.get("optimizer",       args.optimizer)
        args.learning_rate = cfg.get("learning_rate",   args.learning_rate)
        args.weight_decay  = cfg.get("weight_decay",    args.weight_decay)
        args.weight_init   = cfg.get("weight_init",     args.weight_init)
        args.gradient_clip = cfg.get("gradient_clip",   args.gradient_clip)
        if cfg.get("dataset"):
            args.dataset   = cfg["dataset"]
        print(f"Config loaded from {config_path}")

    args.num_layers = len(args.hidden_size)

    # --- Data ---
    _, _, _, _, x_test, y_test = load_data(args.dataset)
    print(f"Test set: {x_test.shape}")

    # --- Build model and restore weights ---
    model   = NeuralNetwork(args)
    weights = load_model(model_path)
    model.set_weights(weights)
    print(f"Weights loaded from {model_path}")

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
