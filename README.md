# Torch-metrics: model evaluation metrics for Pytorch
[![PyPI version](https://badge.fury.io/py/torchsummary.svg)](https://badge.fury.io/py/torchsummary)

As summarized in this [issue](https://github.com/pytorch/pytorch/issues/22439), Pytorch does not have a built-in libary `torch.metrics` for model evaluation metrics. This python library serves as a custom library to provide common evaluation metrics in Pytorch, similar to `tf.keras.metrics`. 

### Usage

- `pip install --upgrade torch_metrics` or 
- `git clone https://github.com/chinokenochkan/torch-metrics`

```python
from torch_metrics import RSquaredMetric
ground_truth = torch.tensor([2., 41., 55., 65.])
model_out = model(torch.tensor([1., 2., 3.]))
r2 = RSquaredMetric(model_out, ground_truth)
```


