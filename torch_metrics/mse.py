import torch

class MSEMetric:
    def __call__(self, tensor1, tensor2):
        """
        Arguments
        ---------
        x : torch.Tensor
        y : torch.Tensor
        """
        return torch.mean((tensor1 - tensor2)**2)