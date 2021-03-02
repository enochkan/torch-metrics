import torch
from torch_metrics.utils import check_same_shape, convert_to_tensor


class Huber:
    """
    Computes the huber loss between `y_true` and `y_pred`.

    Args:
        y_true: Tensor of Ground truth values. shape = [batch_size, d0, .., dN]
        y_pred: Tensor of Predicted values. shape = [batch_size, d0, .., dN]
        delta: A float, the point where the Huber loss function changes from a
                quadratic to linear. default: `1.0`

    Returns:
        Tensor of Huber loss
    """

    def __call__(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, delta: float = 1.0
    ) -> torch.Tensor:
        y_pred = convert_to_tensor(y_pred)
        y_true = convert_to_tensor(y_true)

        check_same_shape(y_pred, y_true)

        abs_error = torch.abs(y_pred - y_true)
        quadratic = torch.clamp(abs_error, max=delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic.pow(2) + delta * linear
        return loss.mean()
