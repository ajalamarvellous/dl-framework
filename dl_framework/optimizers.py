"""implementation of different optimizations"""
import copy

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


class EarlyStoppage:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_score = None
        self.count = 0
        self.early_stoppage = False

    def __call__(self, model, error):
        self.count += 1
        if self.best_score is None:
            self.best_score = abs(error)
            self.model = copy.deepcopy(model)
        elif self.count >= self.patience:
            print("Stopping training now....")
            self.early_stoppage = True
        elif abs(error) < self.best_score:
            self.best_score = abs(error)
            self.model = copy.deepcopy(model)
            self.count = 0
        elif abs(error) >= self.best_score:
            print(f"Error not improving {self.count}/{self.patience}")
            print(f"Error: {abs(error)}, best error: {self.best_score}")
        else:
            pass


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
