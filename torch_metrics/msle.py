import torch
from torch_metrics.utils import check_same_shape
from torch_metrics.utils import convert_to_tensor


class MeanSquaredLogarithmicError:
    """
    Computes the mean squared logarithmic error between `y_true` and `y_pred`.

    Args:
        y_true: Tensor of Ground truth values. shape = [batch_size, d0, .., dN]
        y_pred: Tensor of Predicted values. shape = [batch_size, d0, .., dN]

    Returns:
        Tensor of mean squared logarithmic error
    """

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = convert_to_tensor(y_pred)
        y_true = convert_to_tensor(y_true)

        check_same_shape(y_pred, y_true)

        squared_log = torch.pow(torch.log1p(y_pred) - torch.log1p(y_true), 2)

        return torch.mean(squared_log)
