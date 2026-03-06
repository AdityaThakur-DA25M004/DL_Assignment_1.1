import numpy as np


class Sigmoid:
    def __init__(self):
        self._out = None

    def get_name(self):
        return "sigmoid"

    def forward(self, Z):
        self._out = 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
        return self._out

    def backward(self, dA):
        return dA * self._out * (1.0 - self._out)


class Tanh:
    def __init__(self):
        self._out = None

    def get_name(self):
        return "tanh"

    def forward(self, Z):
        self._out = np.tanh(Z)
        return self._out

    def backward(self, dA):
        return dA * (1.0 - self._out ** 2)


class ReLU:
    def __init__(self):
        self._Z = None

    def get_name(self):
        return "relu"

    def forward(self, Z):
        self._Z = Z
        return np.maximum(0.0, Z)

    def backward(self, dA):
        return dA * (self._Z > 0).astype(float)


class Identity:
    def get_name(self):
        return "identity"

    def forward(self, Z):
        return Z

    def backward(self, dA):
        return dA


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
