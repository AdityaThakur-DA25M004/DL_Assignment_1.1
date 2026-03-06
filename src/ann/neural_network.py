import numpy as np
import os
import json

from .neural_layer import DenseLayer
from .activations import get_activation
from .objective_functions import get_loss, softmax
from .optimizers import get_optimizer


class NeuralNetwork:
    """
    Configurable Multi-Layer Perceptron built entirely with NumPy.

    Constructor accepts either:
      - an argparse.Namespace  (cli_args)
      - keyword arguments directly

    Backward convention (matches autograder):
        self.grad_W[0] = output-layer  weight gradient
        self.grad_W[1] = last-hidden   weight gradient
        ...
        self.grad_W[-1] = first-hidden weight gradient
        (same ordering for self.grad_b)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, cli_args=None, **kwargs):
        import argparse

        # Support both NeuralNetwork(cli_args) and NeuralNetwork(input_size=...) styles
        if cli_args is not None and isinstance(cli_args, argparse.Namespace):
            ns = cli_args
        elif cli_args is not None and not isinstance(cli_args, argparse.Namespace):
            # Treat cli_args as input_size (legacy positional call)
            ns = argparse.Namespace(
                input_size=cli_args,
                hidden_size=kwargs.get("hidden_sizes", kwargs.get("hidden_size", [128])),
                output_size=kwargs.get("output_size", 10),
                activation=kwargs.get("activation", "relu"),
                weight_init=kwargs.get("weight_init", "xavier"),
                loss=kwargs.get("loss", "cross_entropy"),
                optimizer=kwargs.get("optimizer", "adam"),
                learning_rate=kwargs.get("learning_rate", 0.001),
                weight_decay=kwargs.get("weight_decay", 0.0),
                gradient_clip=kwargs.get("gradient_clip", 5.0),
                num_layers=kwargs.get("num_layers", None),
            )
        else:
            ns = argparse.Namespace(
                input_size=kwargs.get("input_size", 784),
                hidden_size=kwargs.get("hidden_sizes", kwargs.get("hidden_size", [128])),
                output_size=kwargs.get("output_size", 10),
                activation=kwargs.get("activation", "relu"),
                weight_init=kwargs.get("weight_init", "xavier"),
                loss=kwargs.get("loss", "cross_entropy"),
                optimizer=kwargs.get("optimizer", "adam"),
                learning_rate=kwargs.get("learning_rate", 0.001),
                weight_decay=kwargs.get("weight_decay", 0.0),
                gradient_clip=kwargs.get("gradient_clip", 5.0),
                num_layers=kwargs.get("num_layers", None),
            )

        # --- Resolve hidden sizes ---
        raw_hs = getattr(ns, "hidden_size", None) or getattr(ns, "hidden_sizes", None) or [128]
        if isinstance(raw_hs, (int, np.integer)):
            raw_hs = [int(raw_hs)]
        raw_hs = [int(h) for h in raw_hs]

        num_layers = getattr(ns, "num_layers", None)
        if num_layers is not None and len(raw_hs) == 1:
            raw_hs = raw_hs * int(num_layers)

        self.input_size      = int(getattr(ns, "input_size", 784))
        self.output_size     = int(getattr(ns, "output_size", 10))
        self.hidden_sizes    = raw_hs
        self.activation_name = getattr(ns, "activation", "relu")
        self.loss_name       = getattr(ns, "loss", "cross_entropy")
        self.weight_init     = getattr(ns, "weight_init", "xavier")
        self.optimizer_name  = getattr(ns, "optimizer", "adam")
        self.learning_rate   = float(getattr(ns, "learning_rate", 0.001))
        self.weight_decay    = float(getattr(ns, "weight_decay", 0.0))
        self.gradient_clip   = float(getattr(ns, "gradient_clip", 5.0))

        # --- Build layers and activations ---
        layer_sizes      = [self.input_size] + self.hidden_sizes + [self.output_size]
        self.layers      = []
        self.activations = []   # len = num_hidden_layers (no activation on output)

        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                DenseLayer(layer_sizes[i], layer_sizes[i + 1], self.weight_init)
            )
            if i < len(layer_sizes) - 2:
                self.activations.append(get_activation(self.activation_name))

        self.loss_fn   = get_loss(self.loss_name)
        self.optimizer = get_optimizer(self.optimizer_name, self.learning_rate, self.weight_decay)

        # Public gradient containers (set after every backward())
        self.grad_W = None
        self.grad_b = None

        # Analysis helpers
        self.gradient_norms   = {i: [] for i in range(len(self.layers))}
        self.activation_stats = {i: [] for i in range(len(self.layers))}

    # ------------------------------------------------------------------
    # Forward — returns raw logits (no softmax applied)
    # ------------------------------------------------------------------
    def forward(self, X):
        """
        X : (N, input_size)
        Returns raw logits : (N, output_size)   ← NO softmax
        """
        A = X
        for i, layer in enumerate(self.layers):
            Z = layer.forward(A)
            # Apply activation only on hidden layers
            A = self.activations[i].forward(Z) if i < len(self.activations) else Z
        self.last_output = A
        return A

    # ------------------------------------------------------------------
    # Backward
    #
    # y_true : integer labels (N,) OR one-hot (N, C)
    # y_pred : raw logits from forward()
    #
    # Returns: (grad_W, grad_b) — numpy object arrays
    #   index 0  = output layer
    #   index 1  = last hidden layer
    #   index -1 = first hidden layer
    # ------------------------------------------------------------------
    def backward(self, y_true, y_pred):
        """
        Compute gradients via backpropagation.

        The method computes and stores:
            self.grad_W  — object array, index 0 = output layer
            self.grad_b  — object array, index 0 = output layer

        Each layer also has its own .grad_W and .grad_b updated.

        Returns (self.grad_W, self.grad_b).
        """
        N = y_pred.shape[0]

        # --- Gradient at the output (pre-softmax logits) ---
        if self.loss_name == "cross_entropy":
            dL_dZ_out = self.loss_fn.backward(y_pred, y_true)   # (probs - y_oh) / N
        else:
            # MSE: use logits directly vs one-hot
            dL_dZ_out = self.loss_fn.backward(y_pred, y_true)   # (2/N)(logits - y_oh)

        grad_W_list = []
        grad_b_list = []

        # --- Output layer backward ---
        out_idx = len(self.layers) - 1
        dL_dA   = self.layers[out_idx].backward(dL_dZ_out)
        grad_W_list.append(self.layers[out_idx].grad_W)   # index 0 → output layer
        grad_b_list.append(self.layers[out_idx].grad_b)
        self.gradient_norms[out_idx].append(
            float(np.linalg.norm(self.layers[out_idx].grad_W))
        )

        # --- Hidden layers: reverse order (last hidden → first hidden) ---
        for i in reversed(range(len(self.layers) - 1)):
            dL_dZ = self.activations[i].backward(dL_dA)
            dL_dA = self.layers[i].backward(dL_dZ)
            grad_W_list.append(self.layers[i].grad_W)   # appends in reverse hidden order
            grad_b_list.append(self.layers[i].grad_b)
            self.gradient_norms[i].append(
                float(np.linalg.norm(self.layers[i].grad_W))
            )
            self.activation_stats[i].append(
                np.abs(self.layers[i].grad_W).mean(axis=0)[:5].tolist()
            )

        # Pack into object arrays — index 0 = output layer, index -1 = first hidden
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------
    def update_weights(self):
        self.optimizer.step()
        for layer in self.layers:
            self.optimizer.update(layer)

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------
    def train_step(self, X, Y):
        """Standard forward → loss → backward → update."""
        if self.optimizer_name == "nag":
            return self.train_step_nag(X, Y)
        logits = self.forward(X)
        loss   = self.compute_loss(logits, Y)
        self.backward(Y, logits)
        if self.gradient_clip > 0:
            self.clip_gradients()
        self.update_weights()
        return loss

    def train_step_nag(self, X, Y):
        """
        NAG: compute gradient at θ - β·v (lookahead point),
        then update velocities and weights from the original θ.
        """
        beta  = getattr(self.optimizer, "beta", 0.9)
        saved = [(l.W.copy(), l.b.copy()) for l in self.layers]

        # Lookahead: temporarily move to θ - β·v
        for layer in self.layers:
            lid = id(layer)
            vW  = self.optimizer.velocities_W.get(lid, np.zeros_like(layer.W))
            vb  = self.optimizer.velocities_b.get(lid, np.zeros_like(layer.b))
            layer.W = layer.W - beta * vW
            layer.b = layer.b - beta * vb

        logits = self.forward(X)
        loss   = self.compute_loss(logits, Y)
        self.backward(Y, logits)

        # Restore original weights before velocity update
        for layer, (W0, b0) in zip(self.layers, saved):
            layer.W = W0
            layer.b = b0

        if self.gradient_clip > 0:
            self.clip_gradients()
        self.update_weights()
        return loss

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self, x_train, y_train, epochs: int = 1, batch_size: int = 32):
        """
        Trains the network for the given number of epochs.

        y_train : integer labels (N,) — converted to one-hot once before batching.
        Returns : dict with key "loss" → list of per-epoch mean losses.
        """
        # Convert integer labels to one-hot once
        if y_train.ndim == 1:
            one_hot = np.zeros((len(x_train), self.output_size))
            one_hot[np.arange(len(x_train)), y_train.astype(int)] = 1
            y_train = one_hot

        history = {"loss": []}
        for epoch in range(epochs):
            # Reset per-epoch stats
            for k in self.gradient_norms:
                self.gradient_norms[k] = []
            for k in self.activation_stats:
                self.activation_stats[k] = []

            indexes      = np.random.permutation(len(x_train))
            epoch_losses = []
            for start in range(0, len(indexes), batch_size):
                batch = indexes[start:start + batch_size]
                epoch_losses.append(
                    self.train_step(x_train[batch], y_train[batch])
                )
            history["loss"].append(float(np.mean(epoch_losses)))
        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X):
        """Returns (predicted_class_indices, softmax_probabilities)."""
        logits    = self.forward(X)
        probs     = softmax(logits)
        return np.argmax(probs, axis=1), probs

    def compute_loss(self, logits, y_true):
        """Compute scalar loss.  Always operates on raw logits."""
        return self.loss_fn.forward(logits, y_true)

    def evaluate(self, X, labels):
        """Returns accuracy (float)."""
        preds, _ = self.predict(X)
        y_int    = labels if labels.ndim == 1 else np.argmax(labels, axis=1)
        return float(np.mean(preds == y_int))

    # ------------------------------------------------------------------
    # Gradient clipping (global norm)
    # ------------------------------------------------------------------
    def clip_gradients(self):
        total_sq = sum(
            np.sum(layer.grad_W ** 2) + np.sum(layer.grad_b ** 2)
            for layer in self.layers if layer.grad_W is not None
        )
        global_norm = np.sqrt(total_sq + 1e-10)
        if global_norm > self.gradient_clip:
            coef = self.gradient_clip / global_norm
            for layer in self.layers:
                if layer.grad_W is not None:
                    layer.grad_W *= coef
                    layer.grad_b *= coef

    # ------------------------------------------------------------------
    # Weight serialisation — used by autograder
    # ------------------------------------------------------------------
    def get_weights(self):
        """Returns a plain dict: {W0, b0, W1, b1, ...}"""
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        """Restore weights from a dict produced by get_weights()."""
        if isinstance(weight_dict, np.ndarray) and weight_dict.ndim == 0:
            weight_dict = weight_dict.item()
        for i, layer in enumerate(self.layers):
            if f"W{i}" in weight_dict:
                layer.W = np.array(weight_dict[f"W{i}"]).copy()
            if f"b{i}" in weight_dict:
                layer.b = np.array(weight_dict[f"b{i}"]).copy()

    def save_weights(self, filepath):
        np.save(filepath, self.get_weights())

    def load_weights(self, filepath):
        self.set_weights(np.load(filepath, allow_pickle=True).item())

    def get_config(self):
        return {
            "input_size":    self.input_size,
            "hidden_sizes":  self.hidden_sizes,
            "output_size":   self.output_size,
            "activation":    self.activation_name,
            "loss":          self.loss_name,
            "weight_init":   self.weight_init,
            "optimizer":     self.optimizer_name,
            "learning_rate": float(self.learning_rate),
            "weight_decay":  float(self.weight_decay),
            "gradient_clip": float(self.gradient_clip),
        }

    def save_config(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.get_config(), f, indent=4)

    def save(self, weights_path, config_path=None):
        os.makedirs(os.path.dirname(os.path.abspath(weights_path)), exist_ok=True)
        self.save_weights(weights_path)
        if config_path:
            self.save_config(config_path)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------
    def get_layer_output(self, X, layer_idx: int):
        A = X
        for i in range(layer_idx + 1):
            Z = self.layers[i].forward(A)
            A = self.activations[i].forward(Z) if i < len(self.activations) else Z
        return A

    def get_dead_neurons(self, X, threshold: float = 0.01):
        dead_info = {}
        A = X
        for idx, (layer, act) in enumerate(zip(self.layers[:-1], self.activations)):
            Z = layer.forward(A)
            A = act.forward(Z)
            if act.get_name().lower() == "relu":
                activation_rate = np.sum(A > 0, axis=0) / A.shape[0]
                dead_neurons    = np.where(activation_rate < threshold)[0]
                dead_info[idx]  = {
                    "dead_neurons":     dead_neurons.tolist(),
                    "num_dead":         len(dead_neurons),
                    "activation_rates": activation_rate.tolist(),
                }
        return dead_info
