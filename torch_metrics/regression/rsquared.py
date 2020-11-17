import torch

def corrcoef(tensor1, tensor2):
    """
    Arguments
    ---------
    x : torch.Tensor
    y : torch.Tensor
    """
    xm = tensor1.sub(torch.mean(tensor1))
    ym = tensor2.sub(torch.mean(tensor2))
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

class RSquared:
    def __init__(self):
        self.corrcoef = corrcoef
    def __call__(self, tensor1, tensor2):
        """
        Arguments
        ---------
        x : torch.Tensor
        y : torch.Tensor
        """
        return (corrcoef(tensor1, tensor2))**2