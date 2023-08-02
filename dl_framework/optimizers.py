"""implementation of different optimizations"""

from eval import RMSE


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
            error = self._loss_fn(y_train - y_pred)
            model._weights -= error * x_train * self._alpha
        return model
