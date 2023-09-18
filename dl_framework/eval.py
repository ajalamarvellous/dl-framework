"""Script to hold some evaluation metrics function"""

from typing import List

import numpy as np


class RMSE:
    """
    Root mean square implementation
    RMSE = sqrt(
        (y_true0 - y_hat0)^2 + (y_true1 - y_hat0)^2 +...+(y_truen - y_hatn)^2
    ) / n

    Usage
    --------
    from eval import RMSE

    error_func = RMSE()

    y_true = np.array(...)
    y_hat = np.array(...)

    error = error_function(y_true, y_hat)

    or

    y_true = 2.0
    y_hat = 3.5

    error = error_function(y_true, y_hat)
    print(error)
    > 1.5
    """

    def __init__(self):
        pass

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> List:
        n = 1

        # confirm that both the y_true and y_pred are of the same type,
        # floats or arrays
        assert type(y_true) == type(
            y_pred
        ), f"Different type formats: y_true type {type(y_true)} y_pred {type(y_pred)}"  # noqa
        if isinstance(type(y_true), np.ndarray):
            # confirm that y_pred and y_true have the same shape
            assert (
                y_true.shape == y_pred.shape
            ), f"The len of y_true {len(y_true)} does not match the len of y_pred{len(y_pred)}"  # noqa
            n = y_true.shape[0]

        self._mse = (np.array(y_true) - np.array(y_pred)) ** 2
        self._rmse = np.sqrt(np.sum(self._mse)) / n
        return self._rmse


class LogLikelihood:
    def __init__(self):
        pass

    def __call__(self):
        pass
