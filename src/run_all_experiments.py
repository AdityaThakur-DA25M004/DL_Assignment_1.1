#!/usr/bin/env python3

import subprocess
import sys
import os
import argparse

# ---------------------------------------------------
# default W&B project 
# ---------------------------------------------------
DEFAULT_PROJECT = "Da6401-assignment1.1"
DEFAULT_ENTITY  = None


# ---------------------------------------------------
# helper function to run one experiment
# ---------------------------------------------------
def run(label, *args, project, entity):

    # build command for train.py
    cmd = [sys.executable, "train.py"] + list(args) + ["-w_p", project]

    if entity:
        cmd += ["--wandb_entity", entity]

    print()
    print("=" * 60)
    print(f"  RUNNING: {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, check=True)

    print(f"  DONE: {label}")

    return result


def main():

    # read optional CLI args
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--entity",  default=DEFAULT_ENTITY)

    # useful when testing (avoids running 100 sweep runs)
    p.add_argument("--skip_sweep", action="store_true")

    args = p.parse_args()

    P = args.project
    E = args.entity

    # ---------------------------------------------------
    # quick checks before running experiments
    # ---------------------------------------------------
    print("=" * 60)
    print("  DA6401 Assignment-1 — Full W&B Experiment Suite")
    print(f"  Project : {P}")
    print(f"  cwd     : {os.getcwd()}")
    print("=" * 60)

    # check wandb
    try:
        import wandb
        print(f"  wandb version : {wandb.__version__}")
    except ImportError:
        print("ERROR: wandb not installed. Run: pip install wandb && wandb login")
        sys.exit(1)

    # check ann package
    try:
        from ann.neural_network import NeuralNetwork
        print("  ann package   : OK")
    except ImportError:
        print("ERROR: ann package not found. Run from inside src/")
        sys.exit(1)

    print("  Pre-flight checks passed.\n")

    # ===================================================
    # 2.1 Data exploration (MNIST)
    # ===================================================
    run("2.1 Data Exploration (MNIST)",
        "-d", "mnist",
        "-e", "1",
        "-b", "64",
        "-l", "cross_entropy",
        "-o", "adam",
        "-lr", "0.001",
        "-wd", "1e-4",
        "-nhl", "2",
        "-sz", "128", "64",
        "-a", "relu",
        "-w_i", "xavier",
        "--wandb_run_name", "2.1_data_exploration_mnist",
        "--log_images",
        project=P, entity=E)

    # ===================================================
    # 2.3 Compare optimizers (same architecture)
    # ===================================================
    for opt, name in [
        ("sgd", "2.3_optimizer_sgd"),
        ("momentum", "2.3_optimizer_momentum"),
        ("nag", "2.3_optimizer_nag"),
        ("rmsprop", "2.3_optimizer_rmsprop")
    ]:

        run(f"2.3 Optimizer: {opt.upper()}",
            "-d", "mnist",
            "-e", "10",
            "-b", "32",
            "-l", "cross_entropy",
            "-o", opt,
            "-lr", "0.01",
            "-wd", "0",
            "-nhl", "3",
            "-sz", "128", "128", "128",
            "-a", "relu",
            "-w_i", "xavier",
            "--gradient_clip", "0",
            "--wandb_run_name", name,
            project=P, entity=E)

    # ===================================================
    # 2.4 Vanishing gradient experiment (deep network)
    # ===================================================
    for act, name in [
        ("sigmoid", "2.4_vanishing_sigmoid_deep"),
        ("relu", "2.4_vanishing_relu_deep")
    ]:

        run(f"2.4 Vanishing Grad: {act}",
            "-d", "mnist",
            "-e", "20",
            "-b", "32",
            "-l", "cross_entropy",
            "-o", "rmsprop",
            "-lr", "0.001",
            "-wd", "0",
            "-nhl", "5",
            "-sz", "128", "128", "128", "128", "128",
            "-a", act,
            "-w_i", "xavier",
            "--gradient_clip", "1.0",
            "--wandb_run_name", name,
            "--log_gradients",
            project=P, entity=E)

    # ===================================================
    # 2.5 Dead neuron experiment (high learning rate)
    # ===================================================
    for act, name in [
        ("relu", "2.5_dead_neurons_relu_lr0.1"),
        ("tanh", "2.5_dead_neurons_tanh_lr0.1")
    ]:

        run(f"2.5 Dead Neurons: {act}",
            "-d", "mnist",
            "-e", "20",
            "-b", "32",
            "-l", "cross_entropy",
            "-o", "sgd",
            "-lr", "0.1",
            "-wd", "0",
            "-nhl", "3",
            "-sz", "128", "128", "128",
            "-a", act,
            "-w_i", "xavier",
            "--gradient_clip", "0",
            "--wandb_run_name", name,
            "--log_dead_neurons",
            project=P, entity=E)

    # ===================================================
    # 2.6 Compare loss functions
    # ===================================================
    for loss, name in [
        ("cross_entropy", "2.6_loss_cross_entropy"),
        ("mse", "2.6_loss_mse")
    ]:

        run(f"2.6 Loss: {loss}",
            "-d", "mnist",
            "-e", "20",
            "-b", "32",
            "-l", loss,
            "-o", "adam",
            "-lr", "0.001",
            "-wd", "1e-4",
            "-nhl", "2",
            "-sz", "128", "64",
            "-a", "relu",
            "-w_i", "xavier",
            "--wandb_run_name", name,
            project=P, entity=E)

    # ===================================================
    # 2.8 Error analysis (best model)
    # ===================================================
    run("2.8 Error Analysis",
        "-d", "mnist",
        "-e", "50",
        "-b", "32",
        "-l", "cross_entropy",
        "-o", "adam",
        "-lr", "0.001",
        "-wd", "1e-4",
        "-nhl", "3",
        "-sz", "128", "128", "64",
        "-a", "relu",
        "-w_i", "xavier",
        "--wandb_run_name", "2.8_error_analysis_best_model",
        "--log_confusion",
        project=P, entity=E)

    # ===================================================
    # 2.9 Weight initialization experiment
    # ===================================================
    for wi, name in [
        ("zeros", "2.9_symmetry_zeros_init"),
        ("xavier", "2.9_symmetry_xavier_init")
    ]:

        run(f"2.9 Init: {wi}",
            "-d", "mnist",
            "-e", "5",
            "-b", "32",
            "-l", "cross_entropy",
            "-o", "adam",
            "-lr", "0.001",
            "-wd", "0",
            "-nhl", "2",
            "-sz", "128", "64",
            "-a", "relu",
            "-w_i", wi,
            "--gradient_clip", "0",
            "--wandb_run_name", name,
            "--log_symmetry",
            project=P, entity=E)

    # ===================================================
    # 2.10 Fashion-MNIST experiments
    # ===================================================
    fashion_configs = [
        ("2.10 Fashion Config-A",
         ["-o", "rmsprop", "-lr", "0.001", "-nhl", "3",
          "-sz", "128", "128", "64", "-a", "relu",
          "--wandb_run_name", "2.10_fashion_configA_relu_rmsprop"]),

        ("2.10 Fashion Config-B",
         ["-o", "adam", "-lr", "0.0005", "-nhl", "4",
          "-sz", "128", "128", "128", "64", "-a", "relu",
          "--wandb_run_name", "2.10_fashion_configB_relu_adam"]),

        ("2.10 Fashion Config-C",
         ["-o", "momentum", "-lr", "0.01", "-nhl", "3",
          "-sz", "128", "128", "128", "-a", "tanh",
          "--wandb_run_name", "2.10_fashion_configC_tanh_momentum"]),
    ]

    for label, extra in fashion_configs:

        run(label,
            "-d", "fashion_mnist",
            "-e", "30",
            "-b", "32",
            "-l", "cross_entropy",
            "-wd", "1e-4",
            "-w_i", "xavier",
            *extra,
            project=P, entity=E)

    # ===================================================
    # 2.7 Hyperparameter sweep (runs last)
    # ===================================================
    if args.skip_sweep:
        print("\nSkipping sweep (--skip_sweep used)")
    else:
        print()
        print("=" * 60)
        print("  Running Hyperparameter Sweep (100 runs)")
        print("=" * 60)

        sweep_cmd = [sys.executable, "sweep.py", "--project", P]

        if E:
            sweep_cmd += ["--entity", E]

        sweep_cmd += ["--count", "100"]

        subprocess.run(sweep_cmd, check=True)

    print()
    print("=" * 60)
    print("  ALL EXPERIMENTS COMPLETE")

    entity_str = E or "<your-username>"

    print(f"  View results: https://wandb.ai/{entity_str}/{P}")
    print("=" * 60)


if __name__ == "__main__":
    main()