import torch

class KLDivergence:
    """
    Computes Kullback-Leibler divergence metric between `y_true` and `y_pred`.

    Args:
        y_true: Tensor of Ground truth values.
        y_pred: Tensor of Predicted values.

    Returns:
        Tensor of Kullback-Leibler divergence metric
    """

    def __init__(self):
        self.epsilon = 1e-10

    def __call__(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1)
        y_true = torch.clamp(y_true, self.epsilon, 1)
        kld = torch.sum(y_true * torch.log(y_true / y_pred), axis=-1)
        return torch.mean(kld)