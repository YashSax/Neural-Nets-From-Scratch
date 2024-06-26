import numpy as np
from abc import ABC, abstractmethod


class LossFn(ABC):
    @abstractmethod
    def __call__(self, y_true: np.array, y_pred: np.array):
        pass


class MSELoss(LossFn):
    def __call__(self, y_true: np.array, y_pred: np.array):
        assert len(y_true.shape) < 3 and len(y_pred.shape) < 3, "y_true and y_pred must be either 1-D or 2-D!"
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape!"

        self.y_true = y_true
        self.y_pred = y_pred
        return np.sum((y_true - y_pred) ** 2) / np.prod(y_true.shape)

    def backward(self):
        self.grad = 2 / self.y_pred.shape[0] * (self.y_pred - self.y_true)
        return self.grad

class BCELoss(LossFn):
    def __call__(self, y_true: np.array, y_pred: np.array):
        assert len(y_true.shape) < 3 and len(y_pred.shape) < 3, "y_true and y_pred must be either 1-D or 2-D!"

        batch_size = 1 if len(y_true.shape) == 1 else y_true.shape[0]
        self.y_true = y_true
        self.y_pred = y_pred
        total_prod = np.transpose(y_true) @ np.log(y_pred)

        return -1 / batch_size * np.sum(total_prod * np.eye(total_prod.shape[0]))

    def backward(self):
        self.grad = self.y_true / self.y_pred
        return self.grad
