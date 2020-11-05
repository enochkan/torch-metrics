import torch

class Accuracy:
    def __call__(self, tensor1, tensor2):
        return torch.mean((t1==t2).type(torch.float64))

# metric = DSCMetric()
# t1 = torch.tensor([1., 2., 3., 4.])
# t2 = torch.tensor([0., 2., 3., 4.])
# print(metric(t1, t2))