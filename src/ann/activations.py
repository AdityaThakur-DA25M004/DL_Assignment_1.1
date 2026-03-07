import numpy as np


# -----------------------------
# Sigmoid Activation Function
# -----------------------------
class Sigmoid:
    def __init__(self):
        # store output during forward pass for use in backward pass
        self._out = None

    def get_name(self):
        return "sigmoid"

    def forward(self, Z):
        # apply sigmoid with clipping to avoid overflow in exp
        self._out = 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
        return self._out

    def backward(self, dA):
        # derivative of sigmoid: s * (1 - s)
        return dA * self._out * (1.0 - self._out)


# -----------------------------
# Tanh Activation Function
# -----------------------------
class Tanh:
    def __init__(self):
        # store output for gradient computation
        self._out = None

    def get_name(self):
        return "tanh"

    def forward(self, Z):
        # apply tanh activation
        self._out = np.tanh(Z)
        return self._out

    def backward(self, dA):
        # derivative of tanh: 1 - tanh^2
        return dA * (1.0 - self._out ** 2)


# -----------------------------
# ReLU Activation Function
# -----------------------------
class ReLU:
    def __init__(self):
        # store input to determine where gradient flows
        self._Z = None

    def get_name(self):
        return "relu"

    def forward(self, Z):
        # save input for backward pass
        self._Z = Z
        return np.maximum(0.0, Z)

    def backward(self, dA):
        # gradient flows only where Z > 0
        return dA * (self._Z > 0).astype(float)


# -----------------------------
# Identity Activation
# (no change to input)
# -----------------------------
class Identity:
    def get_name(self):
        return "identity"

    def forward(self, Z):
        # simply return input
        return Z

    def backward(self, dA):
        # derivative is 1
        return dA


# -----------------------------
# Helper function to get the
# activation object by name
# -----------------------------
def get_activation(name: str):
    name = name.lower()

    if name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return Tanh()
    elif name == "relu":
        return ReLU()
    elif name in ("identity", "none"):
        return Identity()
    else:
        raise ValueError(f"Unknown activation: {name}")