"""implementation of different optimizations"""
import numpy as np
from eval import RMSE


class Dropout:
    def __init__(self, percentage):
        """Randomly set certain percentage to zero"""
        self.p = percentage
        pass

    def __call__(self, inputs):
        self.mask = np.random.choice(
            2, size=inputs.shape, p=[self.p, 1 - self.p]
        )  # noqa
        return inputs * self.mask

    def backprop(self, delta, lr):
        return delta * self.mask


class GradientDescent:
    def __init__(
        self, alpha: float = 0.01, iterations: int = 500, loss_fn="mse"
    ):  # noqa
        """
        Implementation of gradient descent optimization
        Argument(s)
        ---------------
        alpha: int =      scaling value of how big step should be
        iterations: int = Number of iterations to make
        loss_fn: str =    Loss function to use for the GradientDescent
                          Available options
                          mse = Mean Square Error

        """

        self._alpha = alpha
        self._iterations = iterations
        if loss_fn == "mse":
            self._loss_fn = RMSE

    def __call__(self, x_train, y_train, model):
        model = model()
        for n in range(self._iterations):
            y_pred = model.forward(x_train)
            error = self._loss_fn(y_train, y_pred)
            model._weights -= error * x_train * self._alpha
        return model
