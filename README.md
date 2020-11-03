# Torch-metrics: model evaluation metrics for Pytorch
[![PyPI version](https://badge.fury.io/py/torch-metrics.svg)](https://badge.fury.io/py/torch-metrics&kill_cache=1)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  



As summarized in this [issue](https://github.com/pytorch/pytorch/issues/22439), Pytorch does not have a built-in libary `torch.metrics` for model evaluation metrics. This python library serves as a custom library to provide common evaluation metrics in Pytorch, similar to `tf.keras.metrics`. 

### Usage

- `pip install --upgrade torch-metrics` or 
- `git clone https://github.com/chinokenochkan/torch-metrics`

```python
from torch_metrics import RSquaredMetric
## define metric ##
metric = RSquaredMetric()
ground_truth = torch.tensor([2., 41., 55., 65.])
model_out = model(torch.tensor([1., 2., 3.]))
r2 = metric(tensor1=model_out, tensor2=ground_truth)
```


