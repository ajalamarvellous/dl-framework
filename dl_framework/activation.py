"""Activation functions available in dl-framework"""

import numpy as np


class Relu:
    """
    ReLU activation function implementation (Hahnloser et al., 2000)
    y = max(value, 0)
    Returns zero for negative numbers and original number for positive numbers

    Usage
    ---------
    from activation import Relu


    model = Sequential([
        Linear(input_dim, n_nodes), Relu(), Linear(n_nodes, n_outputs)
    ])
    ...
    """

    def __init__(self):
        pass

    def __call__(self, x):
        """Implementation call"""
        self._output = (x > 0) * x
        return self._output

    def backprop(self, delta, lr):
        """
        Backpropagation Implementation
        delta : float
            Gradient propagated to the lyer from layers infront
        lr : float
            learning rate (not needed for ReLU backprop, it was only added
            for consistency of interface)
        """
        lr = 1
        return (self._output > 0) * delta * lr


class Sigmoid:
    """
    Sigmoid activation function implementation
    y = 1/(1 + e-x)
    Sigmoid forces all numbers to be between 0 and 1

    Usage
    ---------
    from activation import Sigmoid


    model = Sequential([
        Linear(input_dim, n_nodes), Sigmoid(), Linear(n_nodes, n_outputs)
    ])
    ...
    """

    def __init__(self):
        pass

    def __call__(self, x):
        """Implementation call"""
        self._output = 1 / (1 + np.exp(-x))
        return self._output

    def backprop(self, delta, lr):
        """
        Backpropagation Implementation
        delta : float
            Gradient propagated to the lyer from layers infront
        lr : float
            learning rate (not needed for ReLU backprop, it was only added
            for consistency of interface)
        """
        lr = 1
        return self._output * (1 - self._output) * delta * lr


class Tanh:
    """
    Tanh activation function implementation
    y = Tanh(x)
    Tanh forces all values to be between -1 and 1

    Usage
    ---------
    from activation import Tanh


    model = Sequential([
        Linear(input_dim, n_nodes), Tanh(), Linear(n_nodes, n_outputs)
    ])
    ...
    """

    def __init__(self):
        pass

    def __call__(self, x):
        """Implementation call"""
        self._output = np.tanh(x)
        return self._output

    def backprop(self, delta, lr):
        """
        Backpropagation Implementation
        delta : float
            Gradient propagated to the lyer from layers infront
        lr : float
            learning rate (not needed for ReLU backprop, it was only added
            for consistency of interface)
        """
        lr = 1
        return (1 - self._output**2) * lr
