from torch_metrics import MeanSquaredError
import torch

metric = MeanSquaredError()
t1 = torch.tensor([1., 2., 3.])
t2 = torch.tensor([1., 5., 25.])
print(metric(t1, t2))