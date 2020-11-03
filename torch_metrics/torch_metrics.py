import torch

def corrcoef(tensor1, tensor2):
    xm = tensor1.sub(torch.mean(tensor1))
    ym = tensor2.sub(torch.mean(tensor2))
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

def RSquaredMetric(tensor1, tensor2):
    return (corrcoef(tensor1, tensor2))**2

def MSEMetric(tensor1, tensor2):
    return torch.mean((tensor1 - tensor2)**2)

def MAEMetric(tensor1, tensor2):
    return torch.mean((tensor1 - tensor2))