"""Script to hold some evaluation metrics function"""

from typing import List

import numpy as np


class RMSE:
    def __init__(self):
        """Root mean square implementation"""
        pass

    def __call__(self, y_true: List[float], y_pred: List[float]) -> List:
        if len(y_true) != len(y_pred):
            raise Exception(
                f"The len of y_true {len(y_true)} does not match the len of y_pred{len(y_pred)}"  # noqa
            )
        self.mse_ = (np.array(y_true) - np.array(y_pred)) ** 2
        self.rmse_ = np.average(np.sqrt(self.mse_))
        return np.average(self.mse_)
