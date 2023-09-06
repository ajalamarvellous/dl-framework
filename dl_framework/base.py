"""Base functions and classes will be found here"""
import numpy as np


class NN:
    """A neural network implementation in numpy"""

    def __init__(self, input_size: int, output_size: int):
        """
        Initialising the weights for the neural networks
        input_size: size of input
        output_size: size of output
        """
        self._weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand()

    def __call__(self, input):
        self.input = input
        self.output = self.input @ self._weights + self.bias
        return self.output


class Sequential:
    """
    A sortof graph tracker to arrange the order for feedforward or Backprop
    """

    def __init__(self):
        self.layers = []

    def __call__(self, node):
        if isinstance(node, list):
            self.layers.extend(node)
        else:
            self.layers.append(node)

    def add(self, node):
        if isinstance(node, list):
            self.layers.extend(node)
        else:
            self.layers.append(node)
