import torch

class RMSEMetric:
    def __call__(self, tensor1, tensor2):
        """
        Returns the root mean squared error (RMSE) of two tensors.

        Arguments
        ---------
        x : torch.Tensor
        y : torch.Tensor
        """
        return torch.sqrt(torch.mean((tensor1 - tensor2)**2))