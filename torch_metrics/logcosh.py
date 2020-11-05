import torch

class LogCoshError:
    """
    Computes Logarithm of the hyperbolic cosine of the prediction error.

    Args:
        y_true: Tensor of Ground truth values.
        y_pred: Tensor of Predicted values.

    Returns:
        Tensor of Logcosh error
    """

    def __call__(self, y_pred, y_true):
        diff = y_pred - y_true
        return torch.mean(torch.log((torch.exp(diff) + torch.exp(-1. * diff)) / 2.))