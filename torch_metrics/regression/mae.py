import torch

class MeanAbsoluteError:
    def __call__(self, tensor1, tensor2):
        """
        Arguments
        ---------
        x : torch.Tensor
        y : torch.Tensor
        """
        return torch.mean(torch.abs(tensor1 - tensor2))
