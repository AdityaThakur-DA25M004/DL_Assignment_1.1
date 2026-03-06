import numpy as np


class DenseLayer:
    """
    A single fully-connected (linear) layer.

    forward : Z = X @ W + b
    backward: given upstream gradient dZ (w.r.t. pre-activation Z),
              computes self.grad_W, self.grad_b and returns dA (gradient w.r.t. input).

    Note: activation is handled by NeuralNetwork, NOT inside this layer.
    """

    def __init__(self, in_features: int, out_features: int, weight_init: str = "xavier"):
        self.in_features  = in_features
        self.out_features = out_features

        if weight_init == "xavier":
            std = np.sqrt(2.0 / (in_features + out_features))
            self.W = np.random.randn(in_features, out_features) * std
        elif weight_init == "zeros":
            self.W = np.zeros((in_features, out_features))
        else:  # "random"
            self.W = np.random.randn(in_features, out_features) * 0.01

        self.b = np.zeros((1, out_features))

        # Gradients — available after every backward() call
        self.grad_W = None
        self.grad_b = None

        # Cache input for backward pass
        self._input = None

    def forward(self, X):
        """Z = X @ W + b  →  shape (N, out_features)"""
        self._input = X
        return X @ self.W + self.b

    def backward(self, dZ):
        """
        dZ  : upstream gradient w.r.t. pre-activation Z  (N, out_features)

        Sets:
            self.grad_W  (in_features, out_features)
            self.grad_b  (1, out_features)

        Returns:
            dA : gradient w.r.t. layer input  (N, in_features)
        """
        # dZ is already normalised (loss backward divides by N),
        # so grad_W = X.T @ dZ  gives the correct mean gradient.
        self.grad_W = self._input.T @ dZ               # (in, out)
        self.grad_b = dZ.sum(axis=0, keepdims=True)    # (1, out)
        dA = dZ @ self.W.T                              # (N, in)
        return dA
