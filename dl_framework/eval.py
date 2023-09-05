"""Script to hold some evaluation metrics function"""

from typing import List

import numpy as np


class RMSE:
    def __init__(self):
        """Root mean square implementation"""
        pass

    def __call__(self, y_true: List[float], y_pred: List[float]) -> List:
        if type(y_true) == list:
            assert (
                y_pred == list
            ), "Only one prediction returned and list of true values given"  # noqa
            assert len(y_true) != len(
                y_pred
            ), f"The len of y_true {len(y_true)} does not match the len of y_pred{len(y_pred)}"  # noqa
            n = len(y_true)
        else:
            n = 1
        self.mse_ = (np.array(y_true) - np.array(y_pred)) ** 2
        self.rmse_ = np.sqrt(np.sum(self.mse_)) / n
        return self.rmse_
