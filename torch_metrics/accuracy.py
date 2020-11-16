import torch


class Accuracy:
    """
    Computes how often predictions equals true labels.

    Args:
        y_true: Tensor of Ground truth values. shape = [batch_size, d0, .., dN]
        y_pred: Tensor of The predicted values. shape = [batch_size, d0, .., dN]
        threshold: Threshold value for binary or multi-label logits. default: `0.5`
        from_logits: If the predictions are logits/probabilites or actual labels. default: `True`
            * `True` for Logits
            * `False` for Actual labels

    Returns:
        Tensor of Accuracy metric
    """

    def __init__(self, threshold: float = 0.5, from_logits: bool = True):
        self.threshold = threshold
        self.from_logits = from_logits

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            y_pred, y_true = self._conversion(y_pred, y_true, self.threshold)
        return torch.mean((y_pred == y_true).float())

    def _conversion(self, y_pred, y_true, threshold):
        if len(y_pred.shape) == len(y_true.shape) + 1:
            y_pred = torch.argmax(y_pred, dim=1)

        if len(y_pred.shape) == len(y_true.shape) and y_pred.dtype == torch.float:
            y_pred = (y_pred >= threshold).float()

        return y_pred, y_true
