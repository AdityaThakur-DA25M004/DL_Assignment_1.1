#!/usr/bin/env python3
"""
train.py — CLI training script for DA6401 Assignment-1 MLP.

Location : src/train.py
Run from inside src/:
    python train.py -d mnist -e 10 -b 64 -o adam -lr 0.001 -nhl 3 -sz 128

Or from project root:
    python src/train.py ...

Directory layout expected:
    project/
    └── src/
        ├── train.py          ← this file
        ├── inference.py
        ├── best_model.npy    ← saved here after training
        ├── best_config.json  ← saved here after training
        ├── ann/
        │   ├── __init__.py
        │   ├── activations.py
        │   ├── neural_layer.py
        │   ├── neural_network.py
        │   ├── objective_functions.py
        │   └── optimizers.py
        └── utils/
            ├── __init__.py
            ├── data_loader.py
            └── metrics.py
"""

import argparse
import json
import os
import sys
import numpy as np

# Ensure both src/ and project root are on the path.
# Handles running as `python src/train.py` (from root) or
# `python train.py` (from inside src/).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_THIS_DIR)
for _p in [_THIS_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ann.neural_network import NeuralNetwork
from utils.data_loader   import load_data
from utils.metrics       import compute_all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Train MLP on MNIST / Fashion-MNIST")

    p.add_argument("-d",   "--dataset",       default="fashion_mnist",
                   choices=["mnist", "fashion_mnist"],
                   help="Dataset to train on")
    p.add_argument("-e",   "--epochs",         type=int,   default=10,
                   help="Number of training epochs")
    p.add_argument("-b",   "--batch_size",     type=int,   default=64,
                   help="Mini-batch size")
    p.add_argument("-l",   "--loss",           default="cross_entropy",
                   choices=["cross_entropy", "mse"],
                   help="Loss function")
    p.add_argument("-o",   "--optimizer",      default="adam",
                   choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                   help="Optimizer")
    p.add_argument("-lr",  "--learning_rate",  type=float, default=0.001,
                   help="Initial learning rate")
    p.add_argument("-wd",  "--weight_decay",   type=float, default=1e-4,
                   help="L2 weight decay")
    p.add_argument("-nhl", "--num_layers",     type=int,   default=3,
                   help="Number of hidden layers")
    p.add_argument("-sz",  "--hidden_size",    type=int,   nargs="+", default=[128],
                   help="Neurons per hidden layer (single value repeated, or one per layer)")
    p.add_argument("-a",   "--activation",     default="relu",
                   choices=["sigmoid", "tanh", "relu"],
                   help="Hidden-layer activation function")
    p.add_argument("-w_i", "--weight_init",    default="xavier",
                   choices=["random", "xavier", "zeros"],
                   help="Weight initialisation strategy")
    p.add_argument("-w_p", "--wandb_project",  default="da6401-assignment1",
                   help="Weights & Biases project name")
    p.add_argument("--wandb_entity",           default=None,
                   help="W&B entity (username or team)")
    p.add_argument("--gradient_clip",          type=float, default=5.0,
                   help="Global gradient-norm clipping threshold (0 = disabled)")
    p.add_argument("--no_wandb",               action="store_true",
                   help="Disable W&B logging")

    return p


def expand_hidden_sizes(args):
    """Expand a single hidden_size entry to num_layers copies."""
    if len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size * args.num_layers
    elif len(args.hidden_size) != args.num_layers:
        raise ValueError(
            f"--hidden_size has {len(args.hidden_size)} values but "
            f"--num_layers={args.num_layers}. They must match."
        )
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()
    args = expand_hidden_sizes(args)

    # --- W&B setup ---
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args),
            )
        except ImportError:
            print("wandb not installed — disabling W&B logging.")
            use_wandb = False

    # --- Load data  (returns x_train, y_train, x_val, y_val, x_test, y_test) ---
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.dataset)
    print(f"Train: {x_train.shape}  Val: {x_val.shape}  Test: {x_test.shape}")

    # --- Build model ---
    model = NeuralNetwork(args)
    print(f"Architecture: {model.input_size} → {model.hidden_sizes} → {model.output_size}")
    print(f"Optimizer: {model.optimizer_name}  LR: {model.learning_rate}  Loss: {model.loss_name}")

    best_test_f1  = -1.0
    best_weights  = None
    best_config   = None

    for epoch in range(args.epochs):
        # train() converts integer labels → one-hot internally
        history    = model.train(x_train, y_train, epochs=1, batch_size=args.batch_size)
        train_loss = history["loss"][0]

        train_preds, _ = model.predict(x_train)
        val_preds,   _ = model.predict(x_val)
        test_preds,  _ = model.predict(x_test)

        train_m = compute_all_metrics(y_train, train_preds)
        val_m   = compute_all_metrics(y_val,   val_preds)
        test_m  = compute_all_metrics(y_test,  test_preds)

        print(
            f"Epoch {epoch + 1:>3}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"train_acc={train_m['accuracy']:.4f} | "
            f"val_acc={val_m['accuracy']:.4f} | "
            f"val_f1={val_m['f1']:.4f} | "
            f"test_acc={test_m['accuracy']:.4f} | "
            f"test_f1={test_m['f1']:.4f}"
        )

        if use_wandb:
            import wandb
            wandb.log({
                "epoch":          epoch + 1,
                "train_loss":     train_loss,
                "train_accuracy": train_m["accuracy"],
                "train_f1":       train_m["f1"],
                "val_accuracy":   val_m["accuracy"],
                "val_f1":         val_m["f1"],
                "test_accuracy":  test_m["accuracy"],
                "test_f1":        test_m["f1"],
            })

        # Save best model checkpoint by test F1
        if test_m["f1"] > best_test_f1:
            best_test_f1 = test_m["f1"]
            best_weights = model.get_weights()
            best_config  = {**model.get_config(), "dataset": args.dataset}

    # --- Persist best model to src/ directory ---
    src_dir      = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(src_dir, "best_model.npy")
    config_path  = os.path.join(src_dir, "best_config.json")

    np.save(weights_path, best_weights)
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=4)

    print(f"\nBest test F1 = {best_test_f1:.4f}")
    print(f"Saved weights → {weights_path}")
    print(f"Saved config  → {config_path}")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
