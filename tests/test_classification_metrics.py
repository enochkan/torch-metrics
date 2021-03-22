import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from sklearn.metrics import accuracy_score, f1_score, precision_score
from torch_metrics.classification import Accuracy, F1Score, Precision

from input_data import binary_prob_inputs, binary_raw_inputs

torch.manual_seed(42)


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.8])
@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        (binary_raw_inputs.target, binary_raw_inputs.preds),
        (binary_prob_inputs.target, binary_prob_inputs.preds),
    ],
)
def test_accuracy(y_true, y_pred, threshold):
    sk_preds = (y_pred.view(-1).numpy() >= threshold).astype(np.uint8)
    sk_target = y_true.view(-1).numpy()
    sk_score = accuracy_score(y_true=sk_target, y_pred=sk_preds)
    torch_metric = Accuracy(threshold=threshold)
    tm_score = torch_metric(y_pred, y_true)
    assert_allclose(sk_score, tm_score)
