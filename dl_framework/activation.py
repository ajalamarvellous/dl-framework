"""Activation functions available in dl-framework"""


class Relu:
    def __init__(self):
        pass

    def __call__(self, x):
        """Retrun same value if greater than zero"""
        return (x > 0) * x
