import numpy as np


class DenseLayer:
    """
    Simple fully connected layer.

    Forward:  Z = X @ W + b
    Backward: gets gradient dZ from next layer and computes gradients
              for W and b, then sends gradient back to previous layer.

    Note: activation is applied outside this class (handled in NeuralNetwork).
    """

    def __init__(self, in_features: int, out_features: int, weight_init: str = "xavier"):
        self.in_features = in_features
        self.out_features = out_features

        # initialize weights depending on chosen method
        if weight_init == "xavier":
            std = np.sqrt(2.0 / (in_features + out_features))
            self.W = np.random.randn(in_features, out_features) * std
        elif weight_init == "zeros":
            self.W = np.zeros((in_features, out_features))
        else:  # random small values
            self.W = np.random.randn(in_features, out_features) * 0.01

        # bias starts as zero
        self.b = np.zeros((1, out_features))

        # gradients will be filled during backward pass
        self.grad_W = None
        self.grad_b = None

        # store input during forward pass (needed for gradient calculation)
        self._input = None

    def forward(self, X):
        # save input for use in backward pass
        self._input = X

        # linear transformation
        return X @ self.W + self.b

    def backward(self, dZ):
        """
        dZ : gradient coming from next layer (shape N x out_features)

        Computes:
        grad_W, grad_b and returns gradient w.r.t input.
        """

        # gradient of weights
        self.grad_W = self._input.T @ dZ

        # gradient of bias
        self.grad_b = dZ.sum(axis=0, keepdims=True)

        # gradient to pass to previous layer
        dA = dZ @ self.W.T

        return dA