from torch_metrics import MSEMetric, MAEMetric, RSquaredMetric
import torch

metric = MSEMetric()
t1 = torch.tensor([1., 2., 3.])
t2 = torch.tensor([1., 5., 25.])
print(metric(t1, t2))