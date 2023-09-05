"""Base functions and classes will be found here"""
import numpy as np


class nn:
    """A neural network implementation in numpy"""

    def __init__(self, no_output: int):
        """Initialising the weights for the neural networks"""
        self._no_output = no_output
        self._weights_set = False

    def forward(self, input):
        if not self._weights_set:
            self._weights = np.random.randn((input.shape[1], self._no_output))
            self.bias = np.random.randn()
        return self._input @ self._weight + self.bias
