import torch
from torch_metrics.utils import check_same_shape
from torch_metrics.utils import convert_to_tensor


class LogCoshError:
    """
    Computes Logarithm of the hyperbolic cosine of the prediction error.

    Args:
        y_true: Tensor of Ground truth values. shape = [batch_size, d0, .., dN]
        y_pred: Tensor of Predicted values. shape = [batch_size, d0, .., dN]

    Returns:
        Tensor of Logcosh error
    """

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = convert_to_tensor(y_pred)
        y_true = convert_to_tensor(y_true)

        check_same_shape(y_pred, y_true)

        diff = y_pred - y_true
        return torch.mean(torch.log((torch.exp(diff) + torch.exp(-1.0 * diff)) / 2.0))
