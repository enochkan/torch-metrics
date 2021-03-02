import torch


class SquareHingeMetric:
    def __call__(self, tensor1, tensor2):
        """
        Arguments
        ---------
        pred : torch.Tensor
        ground_truth : torch.Tensor [-1 or 1]
        """
        if 0.0 in torch.unique(tensor2):
            tensor2[tensor2 == 0.0] = -1.0
        hinge_loss = 1 - torch.mul(tensor1, tensor2)
        hinge_loss[hinge_loss < 0] = 0
        return torch.mean(hinge_loss ** 2)


# metric = SquareHingeMetric()
# t1 = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
# t2 = torch.tensor([[0., 1.], [0., 0.]])
# print(metric(t1, t2))
