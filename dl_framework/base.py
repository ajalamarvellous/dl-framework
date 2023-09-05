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
        return input @ self._weights + self.bias
