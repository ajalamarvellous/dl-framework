"""Script to hold some evaluation metrics function"""

from typing import List

import numpy as np


class RMSE:
    def __init__(self):
        """Root mean square implementation"""
        pass

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> List:
        n = 1
        assert type(y_true) == type(
            y_pred
        ), f"Different type formats: y_true type {type(y_true)} y_pred {type(y_pred)}"  # noqa
        if isinstance(type(y_true), np.ndarray):
            assert (
                y_true.shape[0] == y_pred.shape[0]
            ), f"The len of y_true {len(y_true)} does not match the len of y_pred{len(y_pred)}"  # noqa
            n = y_true.shape[0]
        self.mse_ = (np.array(y_true) - np.array(y_pred)) ** 2
        self.rmse_ = np.sqrt(np.sum(self.mse_)) / n
        return self.rmse_


class LogLikelihood:
    def __init__(self):
        pass

    def __call__(self):
        pass
