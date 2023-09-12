"""Activation functions available in dl-framework"""

import numpy as np


class Relu:
    def __init__(self):
        pass

    def __call__(self, x):
        """Retrun same value if greater than zero"""
        self._output = (x > 0) * x
        return self._output

    def backprop(self, delta, lr):
        """lr was added for consistency of interface, it has no use here"""
        lr = 1
        return (self._output > 0) * delta * lr


class Sigmoid:
    def __init__(self):
        pass

    def __call__(self, x):
        self._output = 1 / (1 + np.exp(-x))
        return self._output

    def backprop(self, delta, lr):
        lr = 1
        return self._output * (1 - self._output) * delta * lr


class Tanh:
    def __init__(self):
        pass

    def __call__(self, x):
        self._output = np.tanh(x)
        return self._output

    def backprop(self, delta, lr):
        lr = 1
        return (1 - self._output**2) * lr
