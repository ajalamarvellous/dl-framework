"""Activation functions available in dl-framework"""

import numpy as np


class Relu:
    def __init__(self):
        pass

    def __call__(self, x):
        """Retrun same value if greater than zero"""
        self.output = (x > 0) * x
        return self.output

    def backprop(self, delta, lr):
        """lr was added for consistency of interface, it has no use here"""
        lr = 1
        return (self.output > 0) * delta * lr


class Sigmoid:
    def __init__(self):
        pass

    def __call__(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backprop(self, delta, lr):
        lr = 1
        return self.output * (1 - self.output) * delta * lr
