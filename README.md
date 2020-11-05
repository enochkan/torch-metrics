# Torch-metrics: model evaluation metrics for PyTorch
[![PyPI version](https://badge.fury.io/py/torch-metrics.svg)](https://badge.fury.io/py/torch-metrics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

As summarized in this [issue](https://github.com/pytorch/pytorch/issues/22439), Pytorch does not have a built-in libary `torch.metrics` for model evaluation metrics. This python library serves as a custom library to provide common evaluation metrics in Pytorch, similar to `tf.keras.metrics`. This is similar to the metrics library in [PyTorch Lightning](https://github.com/enochkan/torch-metrics/issues).

### Usage

- `pip install --upgrade torch-metrics` or 
- `git clone https://github.com/chinokenochkan/torch-metrics`

```python
from torch_metrics import Accuracy
## define metric ##
metric = Accuracy()
ground_truth = torch.tensor([2., 41., 55., 65., 4., 0.4, 0.8, 0.25])
model_out = model(torch.tensor([1.4, 2.2, 0.3, 0.6, 0.4, 0.7, 0.21]))
r2 = metric(tensor1=model_out, tensor2=ground_truth)
```

### Implementation

Metrics from `tf.keras.metrics` and other metrics that are already implemented vs. to-do

- [X] MeanSquaredError class
- [X] RootMeanSquaredError class
- [X] MeanAbsoluteError class
- [X] Precision class
- [X] Recall class
- [X] MeanIoU class
- [X] DSC class (Dice Similarity Coefficient)
- [X] F1Score class
- [X] RSquared class
- [X] Hinge class
- [X] SquaredHinge class
- [X] LogCoshError class
- [X] Accuracy class
- [ ] BinaryAccuracy class
- [ ] CosineSimilarity class
- [ ] AUC class
- [ ] BinaryCrossEntropy class
- [ ] CategoricalCrossEntropy class
- [ ] SparseCategoricalCrossentropy class
- [ ] KLDivergence class

Please raise issues or feature requests [here](https://github.com/enochkan/torch-metrics/issues). 
