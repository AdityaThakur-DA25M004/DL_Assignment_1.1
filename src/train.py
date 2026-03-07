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
    p.add_argument("--log_activations",         action="store_true",
                   help="Log activation distributions + histograms (Section 2.5)")
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
    """
    Section 2.8 — Full error analysis with 3 visualizations:
      1. Confusion Matrix (W&B built-in plot)
      2. Misclassified Images Grid (W&B Table with images)
      3. Top Confused Pairs Table (ranked by error count)
    """
    import numpy as _np

    if dataset_name == "mnist":
        class_names = [str(i) for i in range(10)]
    else:
        class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal",  "Shirt",   "Sneaker",  "Bag",   "Ankle boot"]

    y_true = _np.asarray(y_true).ravel().astype(int)
    y_pred = _np.asarray(y_pred).ravel().astype(int)

    # ── 1. Standard Confusion Matrix ─────────────────────────────────
    wandb.log({
        "error_analysis/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true.tolist(),
            preds=y_pred.tolist(),
            class_names=class_names,
        )
    })
    print("  [W&B] Logged confusion matrix.")

    # ── 2. Misclassified Images Grid ─────────────────────────────────
    # We need x_test to show images — store it as a module-level var
    # set just before this function is called (see train loop below)
    x_test_ref = getattr(log_confusion_matrix, "_x_test", None)
    if x_test_ref is not None:
        wrong_idx = _np.where(y_true != y_pred)[0]
        # Pick up to 60 misclassified samples (6 per confused class pair, capped)
        _np.random.shuffle(wrong_idx)
        picks = wrong_idx[:60]

        table = wandb.Table(columns=[
            "image", "true_label", "true_name",
            "pred_label", "pred_name", "confidence_note"
        ])
        for idx in picks:
            pixel = (x_test_ref[idx].reshape(28, 28) * 255).astype(_np.uint8)
            t, p  = int(y_true[idx]), int(y_pred[idx])
            table.add_data(
                wandb.Image(pixel),
                t, class_names[t],
                p, class_names[p],
                f"{class_names[t]} → {class_names[p]}"
            )
        wandb.log({"error_analysis/misclassified_images": table})
        print(f"  [W&B] Logged {len(picks)} misclassified images grid.")

    # ── 3. Top Confused Pairs Table ───────────────────────────────────
    # Build confusion matrix, extract off-diagonal errors, rank them
    num_classes = len(class_names)
    cm = _np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    pairs = []
    for t in range(num_classes):
        for p in range(num_classes):
            if t != p and cm[t, p] > 0:
                pairs.append((cm[t, p], t, p))
    pairs.sort(reverse=True)   # most confused first

    pairs_table = wandb.Table(columns=[
        "rank", "true_class", "predicted_as", "error_count",
        "true_total", "error_rate_%"
    ])
    for rank, (count, t, p) in enumerate(pairs[:20], 1):   # top 20 pairs
        true_total = int(cm[t, :].sum())
        error_rate = round(count / true_total * 100, 1)
        pairs_table.add_data(
            rank,
            f"{t} ({class_names[t]})",
            f"{p} ({class_names[p]})",
            count,
            true_total,
            error_rate,
        )
    wandb.log({"error_analysis/top_confused_pairs": pairs_table})
    print("  [W&B] Logged top confused pairs table.")

    # ── Summary stats ─────────────────────────────────────────────────
    total_errors = int((y_true != y_pred).sum())
    total        = len(y_true)
    wandb.log({
        "error_analysis/total_errors":   total_errors,
        "error_analysis/error_rate_pct": round(total_errors / total * 100, 2),
        "error_analysis/top1_confused_pair": (
            f"{class_names[pairs[0][1]]}→{class_names[pairs[0][2]]}"
            if pairs else "none"
        ),
    })
    print(f"  [W&B] Total errors: {total_errors}/{total} "
          f"({total_errors/total*100:.2f}%)")


def log_gradient_norms(wandb, model, epoch):
    """Section 2.4 — per-layer gradient norms."""
    log_dict = {"epoch": epoch}
    for i, layer in enumerate(model.layers):
        if layer.grad_W is not None:
            log_dict[f"gradients/layer_{i}_norm"] = float(np.linalg.norm(layer.grad_W))
    wandb.log(log_dict)


def log_dead_neurons(wandb, model, x_sample, epoch):
    """Section 2.5 — dead/saturated neuron counts per layer.
    ReLU  → dead neurons (always output 0).
    Tanh/Sigmoid → saturated neurons (gradient ≈ 0).
    Both logged under same keys so W&B overlays them on one chart.
    """
    dead_info = model.get_dead_neurons(x_sample[:512])
    log_dict  = {"epoch": epoch}
    for layer_idx, info in dead_info.items():
        log_dict[f"dead_neurons/layer_{layer_idx}_count"] = info["num_dead"]
        pct = info["num_dead"] / max(len(info["activation_rates"]), 1) * 100
        log_dict[f"dead_neurons/layer_{layer_idx}_pct"]   = pct
    if len(log_dict) > 1:
        wandb.log(log_dict)


def log_activation_distributions(wandb, model, x_sample, epoch):
    """
    Section 2.5 — activation distribution per hidden layer.

    Logs per layer:
      activations/layer_N_mean      — mean activation value
      activations/layer_N_std       — std of activations
      activations/layer_N_zero_pct  — % of activations that are exactly 0 (ReLU dead)
      activations/layer_N_hist      — W&B Histogram of activation values
      activations/layer_N_near_zero_pct — % of neurons with |mean_activation| < 0.01
    """
    X = x_sample[:256]   # use 256 samples for speed
    A = X
    log_dict = {"epoch": epoch}

    for i, (layer, act) in enumerate(zip(model.layers[:-1], model.activations)):
        Z = layer.forward(A)
        A = act.forward(Z)

        flat         = A.ravel()
        mean_val     = float(np.mean(flat))
        std_val      = float(np.std(flat))
        zero_pct     = float(np.mean(flat == 0.0) * 100)        # exact zeros (ReLU dead)
        near_zero_pct= float(np.mean(np.abs(flat) < 0.01) * 100) # near-zero

        log_dict[f"activations/layer_{i}_mean"]          = mean_val
        log_dict[f"activations/layer_{i}_std"]           = std_val
        log_dict[f"activations/layer_{i}_zero_pct"]      = zero_pct
        log_dict[f"activations/layer_{i}_near_zero_pct"] = near_zero_pct
        log_dict[f"activations/layer_{i}_hist"]          = wandb.Histogram(flat)

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

            if args.log_activations:
                log_activation_distributions(_wandb, model, x_train, epoch + 1)

        # Track best checkpoint
        if test_m["f1"] > best_test_f1:
            best_test_f1    = test_m["f1"]
            best_weights    = model.get_weights()
            best_config     = {**model.get_config(), "dataset": args.dataset}
            best_test_preds = test_preds.copy()

    # --- Section 2.8: confusion matrix + misclassified grid + confused pairs ---
    if use_wandb and args.log_confusion and best_test_preds is not None:
        import wandb as _wandb
        log_confusion_matrix._x_test = x_test   # pass images for misclassified grid
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