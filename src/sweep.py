#!/usr/bin/env python3

# sweep.py
# Runs W&B hyperparameter sweep for Assignment-1 (Section 2.7)
#
# Run from inside src/:
#     python sweep.py --project da6401-assignment1 --count 100
#
# This creates a sweep and runs multiple experiments with different
# hyperparameter combinations. W&B will later generate plots like
# Parallel Coordinates for analysis.

import os
import sys
import argparse
import numpy as np

# make sure src and project root are in python path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)

for _p in [_THIS_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ------------------------------------------------------------
# sweep configuration (hyperparameters to explore)
# ------------------------------------------------------------
SWEEP_CONFIG = {

    # using bayesian search since it is better than random for limited runs
    "method": "bayes",

    # metric W&B should try to maximize
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize",
    },

    "parameters": {

        # learning rate search range
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-2,
        },

        "batch_size": {
            "values": [16, 32, 64, 128],
        },

        "optimizer": {
            "values": ["sgd", "momentum", "nag", "rmsprop"],
        },

        "activation": {
            "values": ["sigmoid", "tanh", "relu"],
        },

        "weight_init": {
            "values": ["random", "xavier"],
        },

        # number of hidden layers
        "num_layers": {
            "values": [1, 2, 3, 4, 5],
        },

        # neurons per hidden layer
        "hidden_size": {
            "values": [32, 64, 128],
        },

        "weight_decay": {
            "values": [0.0, 1e-4, 1e-3],
        },

        "loss": {
            "values": ["cross_entropy", "mse"],
        },
    },
}


# ------------------------------------------------------------
# function executed for every sweep run
# ------------------------------------------------------------
def sweep_train():

    import wandb
    from ann.neural_network import NeuralNetwork
    from utils.data_loader import load_data
    from utils.metrics import compute_all_metrics

    wandb.init()
    cfg = wandb.config

    # create hidden layer sizes list
    hidden_sizes = [int(cfg.hidden_size)] * int(cfg.num_layers)

    # load MNIST data
    x_train, y_train, x_val, y_val, x_test, y_test = load_data("mnist")

    # build argument namespace for model
    import argparse
    args = argparse.Namespace(
        hidden_size   = hidden_sizes,
        num_layers    = int(cfg.num_layers),
        activation    = cfg.activation,
        loss          = cfg.loss,
        optimizer     = cfg.optimizer,
        learning_rate = float(cfg.learning_rate),
        weight_decay  = float(cfg.weight_decay),
        weight_init   = cfg.weight_init,
        gradient_clip = 5.0,
        input_size    = 784,
        output_size   = 10,
    )

    model = NeuralNetwork(args)

    best_val_acc = 0.0
    EPOCHS = 10

    # training loop
    for epoch in range(EPOCHS):

        history = model.train(
            x_train,
            y_train,
            epochs=1,
            batch_size=int(cfg.batch_size)
        )

        train_loss = history["loss"][0]

        # predictions for metrics
        train_preds, _ = model.predict(x_train)
        val_preds, _   = model.predict(x_val)
        test_preds, _  = model.predict(x_test)

        train_m = compute_all_metrics(y_train, train_preds)
        val_m   = compute_all_metrics(y_val, val_preds)
        test_m  = compute_all_metrics(y_test, test_preds)

        # log results to wandb
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

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]

    wandb.finish()


# ------------------------------------------------------------
# main entry point
# ------------------------------------------------------------
def main():

    p = argparse.ArgumentParser(description="W&B sweep for Assignment-1")

    p.add_argument(
        "--project",
        default="da6401-assignment1",
        help="wandb project name"
    )

    p.add_argument(
        "--entity",
        default=None,
        help="wandb username or team"
    )

    p.add_argument(
        "--count",
        type=int,
        default=100,
        help="number of runs in sweep"
    )

    args = p.parse_args()

    import wandb

    # create sweep
    sweep_id = wandb.sweep(
        SWEEP_CONFIG,
        project=args.project,
        entity=args.entity,
    )

    print(f"Sweep ID: {sweep_id}")
    print(f"Running {args.count} sweep agents ...")

    # start agents
    wandb.agent(
        sweep_id,
        function=sweep_train,
        count=args.count
    )

    print("Sweep complete.")


if __name__ == "__main__":
    main()