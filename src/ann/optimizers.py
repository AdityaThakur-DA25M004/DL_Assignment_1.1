import numpy as np


class SGD:
    """
    Vanilla mini-batch SGD with optional L2 weight decay.
    sgd = simple gradient descent, processes batched inputs.
    """

    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0):
        self.lr           = lr
        self.weight_decay = weight_decay

    def step(self):
        pass  # stateless

    def update(self, layer):
        gW = layer.grad_W + self.weight_decay * layer.W
        gb = layer.grad_b
        layer.W -= self.lr * gW
        layer.b -= self.lr * gb


class MomentumSGD:
    """SGD with classical momentum."""

    def __init__(self, lr: float = 0.01, beta: float = 0.9, weight_decay: float = 0.0):
        self.lr           = lr
        self.beta         = beta
        self.weight_decay = weight_decay
        self.velocities_W = {}
        self.velocities_b = {}

    def step(self):
        pass

    def update(self, layer):
        lid = id(layer)
        vW  = self.velocities_W.get(lid, np.zeros_like(layer.W))
        vb  = self.velocities_b.get(lid, np.zeros_like(layer.b))

        gW = layer.grad_W + self.weight_decay * layer.W
        gb = layer.grad_b

        vW = self.beta * vW + self.lr * gW
        vb = self.beta * vb + self.lr * gb

        self.velocities_W[lid] = vW
        self.velocities_b[lid] = vb

        layer.W -= vW
        layer.b -= vb


class NAG:
    """
    Nesterov Accelerated Gradient.
    The lookahead step is handled externally in NeuralNetwork.train_step_nag().
    This optimizer only performs the velocity update + weight correction.
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9, weight_decay: float = 0.0):
        self.lr           = lr
        self.beta         = beta
        self.weight_decay = weight_decay
        self.velocities_W = {}
        self.velocities_b = {}

    def step(self):
        pass

    def update(self, layer):
        lid = id(layer)
        vW  = self.velocities_W.get(lid, np.zeros_like(layer.W))
        vb  = self.velocities_b.get(lid, np.zeros_like(layer.b))

        gW = layer.grad_W + self.weight_decay * layer.W
        gb = layer.grad_b

        vW = self.beta * vW + self.lr * gW
        vb = self.beta * vb + self.lr * gb

        self.velocities_W[lid] = vW
        self.velocities_b[lid] = vb

        layer.W -= vW
        layer.b -= vb


class RMSProp:
    """RMSProp optimizer with optional weight decay."""

    def __init__(self, lr: float = 0.001, beta: float = 0.9,
                 epsilon: float = 1e-8, weight_decay: float = 0.0):
        self.lr           = lr
        self.beta         = beta
        self.epsilon      = epsilon
        self.weight_decay = weight_decay
        self.cache_W      = {}
        self.cache_b      = {}

    def step(self):
        pass

    def update(self, layer):
        lid = id(layer)
        cW  = self.cache_W.get(lid, np.zeros_like(layer.W))
        cb  = self.cache_b.get(lid, np.zeros_like(layer.b))

        gW = layer.grad_W + self.weight_decay * layer.W
        gb = layer.grad_b

        cW = self.beta * cW + (1.0 - self.beta) * gW ** 2
        cb = self.beta * cb + (1.0 - self.beta) * gb ** 2

        self.cache_W[lid] = cW
        self.cache_b[lid] = cb

        layer.W -= self.lr * gW / (np.sqrt(cW) + self.epsilon)
        layer.b -= self.lr * gb / (np.sqrt(cb) + self.epsilon)


class Adam:
    """Adam optimizer."""

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, weight_decay: float = 0.0):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.epsilon      = epsilon
        self.weight_decay = weight_decay
        self.m_W = {}; self.v_W = {}
        self.m_b = {}; self.v_b = {}
        self.t   = 0

    def step(self):
        self.t += 1

    def update(self, layer):
        lid = id(layer)
        t   = max(self.t, 1)

        mW = self.m_W.get(lid, np.zeros_like(layer.W))
        vW = self.v_W.get(lid, np.zeros_like(layer.W))
        mb = self.m_b.get(lid, np.zeros_like(layer.b))
        vb = self.v_b.get(lid, np.zeros_like(layer.b))

        gW = layer.grad_W + self.weight_decay * layer.W
        gb = layer.grad_b

        mW = self.beta1 * mW + (1.0 - self.beta1) * gW
        vW = self.beta2 * vW + (1.0 - self.beta2) * gW ** 2
        mb = self.beta1 * mb + (1.0 - self.beta1) * gb
        vb = self.beta2 * vb + (1.0 - self.beta2) * gb ** 2

        self.m_W[lid] = mW; self.v_W[lid] = vW
        self.m_b[lid] = mb; self.v_b[lid] = vb

        mW_hat = mW / (1.0 - self.beta1 ** t)
        vW_hat = vW / (1.0 - self.beta2 ** t)
        mb_hat = mb / (1.0 - self.beta1 ** t)
        vb_hat = vb / (1.0 - self.beta2 ** t)

        layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
        layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.epsilon)


class Nadam:
    """Nadam (Nesterov + Adam) optimizer."""

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, weight_decay: float = 0.0):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.epsilon      = epsilon
        self.weight_decay = weight_decay
        self.m_W = {}; self.v_W = {}
        self.m_b = {}; self.v_b = {}
        self.t   = 0

    def step(self):
        self.t += 1

    def update(self, layer):
        lid = id(layer)
        t   = max(self.t, 1)

        mW = self.m_W.get(lid, np.zeros_like(layer.W))
        vW = self.v_W.get(lid, np.zeros_like(layer.W))
        mb = self.m_b.get(lid, np.zeros_like(layer.b))
        vb = self.v_b.get(lid, np.zeros_like(layer.b))

        gW = layer.grad_W + self.weight_decay * layer.W
        gb = layer.grad_b

        mW = self.beta1 * mW + (1.0 - self.beta1) * gW
        vW = self.beta2 * vW + (1.0 - self.beta2) * gW ** 2
        mb = self.beta1 * mb + (1.0 - self.beta1) * gb
        vb = self.beta2 * vb + (1.0 - self.beta2) * gb ** 2

        self.m_W[lid] = mW; self.v_W[lid] = vW
        self.m_b[lid] = mb; self.v_b[lid] = vb

        vW_hat = vW / (1.0 - self.beta2 ** t)
        vb_hat = vb / (1.0 - self.beta2 ** t)

        # Nesterov look-ahead moment (bias corrected at t+1)
        mW_nag = (self.beta1 * mW + (1.0 - self.beta1) * gW) / (1.0 - self.beta1 ** (t + 1))
        mb_nag = (self.beta1 * mb + (1.0 - self.beta1) * gb) / (1.0 - self.beta1 ** (t + 1))

        layer.W -= self.lr * mW_nag / (np.sqrt(vW_hat) + self.epsilon)
        layer.b -= self.lr * mb_nag / (np.sqrt(vb_hat) + self.epsilon)


def get_optimizer(name: str, lr: float = 0.001, weight_decay: float = 0.0):
    name = name.lower()
    if name == "sgd":
        return SGD(lr=lr, weight_decay=weight_decay)
    elif name == "momentum":
        return MomentumSGD(lr=lr, weight_decay=weight_decay)
    elif name == "nag":
        return NAG(lr=lr, weight_decay=weight_decay)
    elif name == "rmsprop":
        return RMSProp(lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return Adam(lr=lr, weight_decay=weight_decay)
    elif name == "nadam":
        return Nadam(lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}. Choose from: sgd, momentum, nag, rmsprop, adam, nadam")
