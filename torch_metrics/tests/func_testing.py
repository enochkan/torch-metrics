from torch_metrics import RSquaredMetric
import torch
t1 = torch.tensor([1., 2., 3.])
t2 = torch.tensor([1., 5., 25.])
print(RSquaredMetric(t1, t2))