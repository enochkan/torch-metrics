import torch


class Accuracy:
    """
    Computes how often predictions equals true labels.

    Args:
        y_true: Tensor of Ground truth values.
        y_pred: Tensor of The predicted values.
        threshold: Threshold value for binary or multi-label logits. default: `0.5`
        logits: If the predictions are logits/probabilites or actual labels. default: `True`
            * `True` for Logits
            * `False` for Actual labels

    Returns:
        Tensor of Accuracy metric
    """

    def __init__(self, threshold=0.5, logits=True):
        self.threshold = threshold
        self.logits = logits

    def __call__(self, y_pred, y_true):
        if self.logits:
            y_pred, y_true = self.conversion(y_pred, y_true, self.threshold)
        return torch.mean((y_pred == y_true).float())

    def conversion(self, y_pred, y_true, threshold):
        if len(y_pred.shape) == len(y_true.shape) + 1:
            y_pred = torch.argmax(y_pred, dim=1)

        if len(y_pred.shape) == len(y_true.shape) and y_pred.dtype == torch.float:
            y_pred = (y_pred >= threshold).float()

        return y_pred, y_true
