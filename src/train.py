#!/usr/bin/env python3
"""
train.py — CLI training script for DA6401 Assignment-1 MLP.

Location : src/train.py
Run from inside src/:
    python train.py -d mnist -e 10 -b 64 -o adam -lr 0.001 -nhl 3 -sz 128

Logs to W&B:
  - per-epoch: loss, accuracy, F1 (train / val / test)
  - per-layer gradient norms  (section 2.4 vanishing gradients)
  - dead-neuron counts        (section 2.5)
  - per-neuron gradients      (section 2.9 symmetry)
  - sample image table        (section 2.1)
  - confusion matrix          (section 2.8)
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
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Train MLP on MNIST / Fashion-MNIST")

    p.add_argument("-d",   "--dataset",        default="fashion_mnist",
                   choices=["mnist", "fashion_mnist"])
    p.add_argument("-e",   "--epochs",          type=int,   default=10)
    p.add_argument("-b",   "--batch_size",      type=int,   default=64)
    p.add_argument("-l",   "--loss",            default="cross_entropy",
                   choices=["cross_entropy", "mse"])
    p.add_argument("-o",   "--optimizer",       default="adam",
                   choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    p.add_argument("-lr",  "--learning_rate",   type=float, default=0.001)
    p.add_argument("-wd",  "--weight_decay",    type=float, default=1e-4)
    p.add_argument("-nhl", "--num_layers",      type=int,   default=3)
    p.add_argument("-sz",  "--hidden_size",     type=int,   nargs="+", default=[128])
    p.add_argument("-a",   "--activation",      default="relu",
                   choices=["sigmoid", "tanh", "relu"])
    p.add_argument("-w_i", "--weight_init",     default="xavier",
                   choices=["random", "xavier", "zeros"])
    p.add_argument("-w_p", "--wandb_project",   default="da6401-assignment1")
    p.add_argument("--wandb_entity",            default=None)
    p.add_argument("--wandb_run_name",          default=None,
                   help="Override W&B run name")
    p.add_argument("--gradient_clip",           type=float, default=5.0)
    p.add_argument("--no_wandb",                action="store_true",
                   help="Disable W&B logging")

    # Special logging flags for specific report sections
    p.add_argument("--log_images",              action="store_true",
                   help="Log sample image table to W&B (Section 2.1)")
    p.add_argument("--log_gradients",           action="store_true",
                   help="Log per-layer gradient norms (Section 2.4)")
    p.add_argument("--log_dead_neurons",        action="store_true",
                   help="Log dead-neuron stats (Section 2.5)")
    p.add_argument("--log_symmetry",            action="store_true",
                   help="Log per-neuron gradients for symmetry analysis (Section 2.9)")
    p.add_argument("--log_confusion",           action="store_true",
                   help="Log confusion matrix at end of training (Section 2.8)")

    return p


def expand_hidden_sizes(args):
    if len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size * args.num_layers
    elif len(args.hidden_size) != args.num_layers:
        raise ValueError(
            f"--hidden_size has {len(args.hidden_size)} values but "
            f"--num_layers={args.num_layers}. They must match."
        )
    return args


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def log_image_table(wandb, x_data, y_data, dataset_name, num_per_class=5):
    """Section 2.1 — 5 sample images per class."""
    if dataset_name == "mnist":
        class_names = [str(i) for i in range(10)]
    else:
        class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal",  "Shirt",   "Sneaker",  "Bag",   "Ankle boot"]

    columns = ["class_id", "class_name"] + [f"image_{i+1}" for i in range(num_per_class)]
    table   = wandb.Table(columns=columns)

    for cls in range(10):
        idx   = np.where(y_data == cls)[0]
        picks = idx[:num_per_class]
        imgs  = []
        for p in picks:
            pixel = (x_data[p].reshape(28, 28) * 255).astype(np.uint8)
            imgs.append(wandb.Image(pixel, caption=f"class {cls}"))
        while len(imgs) < num_per_class:
            imgs.append(None)
        table.add_data(cls, class_names[cls], *imgs)

    wandb.log({"data_exploration/sample_images": table})
    print("  [W&B] Logged sample image table.")


def log_confusion_matrix(wandb, y_true, y_pred, dataset_name):
    """Section 2.8 — confusion matrix."""
    if dataset_name == "mnist":
        class_names = [str(i) for i in range(10)]
    else:
        class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal",  "Shirt",   "Sneaker",  "Bag",   "Ankle boot"]
    wandb.log({
        "error_analysis/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true.tolist(),
            preds=y_pred.tolist(),
            class_names=class_names,
        )
    })
    print("  [W&B] Logged confusion matrix.")


def log_gradient_norms(wandb, model, epoch):
    """Section 2.4 — per-layer gradient norms."""
    log_dict = {"epoch": epoch}
    for i, layer in enumerate(model.layers):
        if layer.grad_W is not None:
            log_dict[f"gradients/layer_{i}_norm"] = float(np.linalg.norm(layer.grad_W))
    wandb.log(log_dict)


def log_dead_neurons(wandb, model, x_sample, epoch):
    """Section 2.5 — dead neuron counts per layer."""
    dead_info = model.get_dead_neurons(x_sample[:512])
    log_dict  = {"epoch": epoch}
    for layer_idx, info in dead_info.items():
        log_dict[f"dead_neurons/layer_{layer_idx}_count"] = info["num_dead"]
        pct = info["num_dead"] / max(len(info["activation_rates"]), 1) * 100
        log_dict[f"dead_neurons/layer_{layer_idx}_pct"]   = pct
    if len(log_dict) > 1:
        wandb.log(log_dict)


_symmetry_iter = [0]

def log_symmetry_gradients(wandb, model, max_iters=50):
    """Section 2.9 — absolute gradient of 5 neurons in first hidden layer."""
    if _symmetry_iter[0] >= max_iters:
        return
    layer    = model.layers[0]
    if layer.grad_W is None:
        return
    log_dict = {"symmetry/iteration": _symmetry_iter[0]}
    for n in range(min(5, layer.grad_W.shape[1])):
        log_dict[f"symmetry/layer0_neuron{n}_grad"] = float(
            np.abs(layer.grad_W[:, n]).mean()
        )
    wandb.log(log_dict)
    _symmetry_iter[0] += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()
    args = expand_hidden_sizes(args)

    _symmetry_iter[0] = 0   # reset for each run

    # --- W&B setup ---
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args),
                reinit=True,
            )
        except ImportError:
            print("wandb not installed — disabling W&B logging.")
            use_wandb = False

    # --- Load data ---
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.dataset)
    print(f"Train: {x_train.shape}  Val: {x_val.shape}  Test: {x_test.shape}")

    # Section 2.1 — log image table once before training
    if use_wandb and args.log_images:
        import wandb as _wandb
        log_image_table(_wandb, x_train, y_train, args.dataset)

    # --- Build model ---
    model = NeuralNetwork(args)
    print(f"Architecture: {model.input_size} → {model.hidden_sizes} → {model.output_size}")
    print(f"Optimizer: {model.optimizer_name}  LR: {model.learning_rate}  Loss: {model.loss_name}")

    best_test_f1    = -1.0
    best_weights    = None
    best_config     = None
    best_test_preds = None

    for epoch in range(args.epochs):

        # When logging symmetry we need batch-level hooks; otherwise use train()
        if use_wandb and args.log_symmetry:
            import wandb as _wandb
            one_hot = np.zeros((len(x_train), model.output_size))
            one_hot[np.arange(len(x_train)), y_train.astype(int)] = 1
            indexes      = np.random.permutation(len(x_train))
            epoch_losses = []
            for start in range(0, len(indexes), args.batch_size):
                batch = indexes[start:start + args.batch_size]
                loss  = model.train_step(x_train[batch], one_hot[batch])
                epoch_losses.append(loss)
                log_symmetry_gradients(_wandb, model, max_iters=50)
            train_loss = float(np.mean(epoch_losses))
        else:
            history    = model.train(x_train, y_train, epochs=1, batch_size=args.batch_size)
            train_loss = history["loss"][0]

        # --- Evaluate ---
        train_preds, _ = model.predict(x_train)
        val_preds,   _ = model.predict(x_val)
        test_preds,  _ = model.predict(x_test)

        train_m = compute_all_metrics(y_train, train_preds)
        val_m   = compute_all_metrics(y_val,   val_preds)
        test_m  = compute_all_metrics(y_test,  test_preds)

        print(
            f"Epoch {epoch+1:>3}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"train_acc={train_m['accuracy']:.4f} | "
            f"val_acc={val_m['accuracy']:.4f} | "
            f"val_f1={val_m['f1']:.4f} | "
            f"test_acc={test_m['accuracy']:.4f} | "
            f"test_f1={test_m['f1']:.4f}"
        )

        if use_wandb:
            import wandb as _wandb
            _wandb.log({
                "epoch":          epoch + 1,
                "train_loss":     train_loss,
                "train_accuracy": train_m["accuracy"],
                "train_f1":       train_m["f1"],
                "val_accuracy":   val_m["accuracy"],
                "val_f1":         val_m["f1"],
                "test_accuracy":  test_m["accuracy"],
                "test_f1":        test_m["f1"],
            })

            if args.log_gradients:
                log_gradient_norms(_wandb, model, epoch + 1)

            if args.log_dead_neurons:
                log_dead_neurons(_wandb, model, x_train, epoch + 1)

        # Track best checkpoint
        if test_m["f1"] > best_test_f1:
            best_test_f1    = test_m["f1"]
            best_weights    = model.get_weights()
            best_config     = {**model.get_config(), "dataset": args.dataset}
            best_test_preds = test_preds.copy()

    # --- Section 2.8: confusion matrix ---
    if use_wandb and args.log_confusion and best_test_preds is not None:
        import wandb as _wandb
        log_confusion_matrix(_wandb, y_test, best_test_preds, args.dataset)

    # --- Save best model ---
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
        import wandb as _wandb
        _wandb.finish()

    return best_test_f1


if __name__ == "__main__":
    main()
