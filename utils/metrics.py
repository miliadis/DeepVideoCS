from math import log10
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def psnr_accuracy(output, target):
    """Computes the PSNR accuracy per frame"""

    pix_elements = output[0, 0, 0].flatten().shape
    avg_psnr = 0
    for batch in range(target.shape[0]):
        for video_slice in range(target.shape[1]):
            for frame in range(target.shape[2]):
                mse = np.sum(((output[batch, video_slice, frame].flatten(
                ) - target[batch, video_slice, frame].flatten()) ** 2)) / pix_elements
                psnr = 10 * log10(1 / mse)
                avg_psnr += psnr

    avg_psnr /= target.shape[0] * target.shape[1] * target.shape[2]

    return avg_psnr
