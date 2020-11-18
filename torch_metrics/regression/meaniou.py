import torch


class MeanIoU:
    def __init__(self):
        self.epsilon = 1e-10

    def __call__(self, tensor1, tensor2):
        # if single dimension
        if len(tensor1.shape) == 1 and len(tensor2.shape) == 1:
            inter = torch.sum(torch.squeeze(tensor1 * tensor2))
            union = torch.sum(torch.squeeze(tensor1 + tensor2)) - inter
        else:
            inter = torch.sum(
                torch.sum(torch.squeeze(tensor1 * tensor2, axis=3), axis=2), axis=1
            )
            union = (
                torch.sum(
                    torch.sum(torch.squeeze(tensor1 + tensor2, axis=3), axis=2), axis=1
                )
                - inter
            )
        return torch.mean((inter + self.epsilon) / (union + self.epsilon))


# metrics = MeanIoUMetric()
# t1 = torch.tensor([0., 0., 1., 1.])
# t2 = torch.tensor([0., 1., 0., 1.])
# print(metrics(t1,t2))
