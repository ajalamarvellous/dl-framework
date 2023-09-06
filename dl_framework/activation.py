"""Activation functions available in dl-framework"""


class Relu:
    def __init__(self):
        pass

    def __call__(self, x):
        """Retrun same value if greater than zero"""
        self.output = (x > 0) * x
        return self.output

    def backprop(self, delta):
        return (self.output > 0) * delta
