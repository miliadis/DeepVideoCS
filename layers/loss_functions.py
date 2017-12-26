import torch.nn.functional as F


class _LossFunctions(object):
    def __init__(self):
        return None


class EuclideanDistance(_LossFunctions):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def compute_loss(self, input, target):
        loss = F.mse_loss(input, target, size_average=False)
        return loss / (2 * self.batch_size)
