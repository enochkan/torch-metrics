import torch


class Precision:
    """
    Computes precision of the predictions with respect to the true labels.

    Args:
        y_true: Tensor of Ground truth values.
        y_pred: Tensor of Predicted values.
        epsilon: Fuzz factor to avoid division by zero. default: `1e-10`

    Returns:
        Tensor of precision score
    """

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def __call__(self, y_pred, y_true):
        true_positives = torch.sum(torch.round(torch.clip(y_pred * y_true, 0, 1)))
        predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + self.epsilon)
        return precision


class Recall:
    """
    Computes recall of the predictions with respect to the true labels.

    Args:
        y_true: Tensor of Ground truth values.
        y_pred: Tensor of Predicted values.
        epsilon: Fuzz factor to avoid division by zero. default: `1e-10`

    Returns:
        Tensor of recall score
    """

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def __call__(self, y_pred, y_true):
        true_positives = torch.sum(torch.round(torch.clip(y_pred * y_true, 0, 1)))
        actual_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
        recall = true_positives / (actual_positives + self.epsilon)
        return recall
