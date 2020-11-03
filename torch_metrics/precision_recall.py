import torch

class Precision:
    def __init__(self):
        self.epsilon = 1e-10
    def __call__(self, tensor1, tensor2):
        true_positives = torch.sum(torch.round(torch.clip(tensor1 * tensor2, 0, 1)))
        predicted_positives = torch.sum(torch.round(torch.clip(tensor2, 0, 1)))
        precision = true_positives / (predicted_positives + self.epsilon)
        return precision


class Recall:
    def __init__(self):
        self.epsilon = 1e-10
    def __call__(self, tensor1, tensor2):
        true_positives = torch.sum(torch.round(torch.clip(tensor1 * tensor2, 0, 1)))
        possible_positives = torch.sum(torch.round(torch.clip(tensor1, 0, 1)))
        recall = true_positives / (possible_positives + self.epsilon)
        return recall
    
