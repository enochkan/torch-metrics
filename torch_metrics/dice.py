import torch

class DSCMetric:
    def __init__(self):
        self.smooth = 1.
    def __call__(self, tensor1, tensor2):
        iflat = tensor1.view(-1)
        tflat = tensor2.view(-1)
        intersection = (iflat * tflat).sum()
        
        return 1 - ((2. * intersection + self.smooth) /
                (iflat.sum() + tflat.sum() + self.smooth))


# metric = DSCMetric()
# t1 = torch.tensor([1., 0., 1., 0., 0.])
# t2 = torch.tensor([0., 0., 1., 1., 0.])
# print(metric(t1, t2))