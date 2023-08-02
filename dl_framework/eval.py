"""Script to hold some evaluation metrics function"""

import numpy as np


class RMSE:
    def __init__(self):
        """Root mean square implementation"""
        pass

    def __call__(self, target: int, value: int) -> int:
        self.mse_ = (target - value) ** 2
        self.rmse_ = np.sqrt(self.mse_)
        return self.mse_
