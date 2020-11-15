import torch
import pytest
import numpy as np
from numpy.testing import assert_allclose
from torch import tensor
from input_datas import uniform_regression_inputs, normal_regression_inputs
from torch_metrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    MeanSquaredLogarithmicError,
    RootMeanSquaredError,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
)

torch.manual_seed(42)


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        (uniform_regression_inputs.target, uniform_regression_inputs.preds),
        (normal_regression_inputs.target, normal_regression_inputs.preds),
    ],
)
def test_mae(y_true, y_pred):
    sk_preds = y_pred.numpy()
    sk_target = y_true.numpy()
    sk_score = mean_absolute_error(y_true=sk_target, y_pred=sk_preds)
    torch_metric = MeanAbsoluteError()
    tm_score = torch_metric(y_pred, y_true)
    assert_allclose(sk_score, tm_score)


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        (uniform_regression_inputs.target, uniform_regression_inputs.preds),
        (normal_regression_inputs.target, normal_regression_inputs.preds),
    ],
)
def test_mse(y_true, y_pred):
    sk_preds = y_pred.numpy()
    sk_target = y_true.numpy()
    sk_score = mean_squared_error(y_true=sk_target, y_pred=sk_preds)
    torch_metric = MeanSquaredError()
    tm_score = torch_metric(y_pred, y_true)
    assert_allclose(sk_score, tm_score)


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        (uniform_regression_inputs.target, uniform_regression_inputs.preds),
    ],
)
def test_msle(y_true, y_pred):
    sk_preds = y_pred.numpy()
    sk_target = y_true.numpy()
    sk_score = mean_squared_log_error(y_true=sk_target, y_pred=sk_preds)
    torch_metric = MeanSquaredLogarithmicError()
    tm_score = torch_metric(y_pred, y_true)
    assert_allclose(sk_score, tm_score)


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        (uniform_regression_inputs.target, uniform_regression_inputs.preds),
        (normal_regression_inputs.target, normal_regression_inputs.preds),
    ],
)
def test_rmse(y_true, y_pred):
    sk_preds = y_pred.numpy()
    sk_target = y_true.numpy()
    sk_score = np.sqrt(mean_squared_error(y_true=sk_target, y_pred=sk_preds))
    torch_metric = RootMeanSquaredError()
    tm_score = torch_metric(y_pred, y_true)
    assert_allclose(sk_score, tm_score)
