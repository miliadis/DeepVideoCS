import torch
import torch.nn as nn
from torch.autograd import Variable
import layers as mynn

__all__ = ['FC8Net', 'fc8net']


class FC8Net(nn.Module):

    def __init__(self, block_opts, mask_path, mean, std, noise_level):
        super(FC8Net, self).__init__()

        self.spatial_size = block_opts[1]
        self.temporal_size = block_opts[0]
        self.vectorized = self.spatial_size * self.spatial_size * self.temporal_size

        self.measurements = mynn.Measurements(
            self.spatial_size, self.temporal_size, overlapping=block_opts[3], mask=mask_path, mean=mean, std=std, noise=noise_level)

        self.reconstruction = nn.Sequential(
            nn.Linear(self.spatial_size * self.spatial_size, self.vectorized),
            nn.ReLU(inplace=True),
            nn.Linear(self.vectorized, self.vectorized),
            nn.ReLU(inplace=True),
            nn.Linear(self.vectorized, self.vectorized),
            nn.ReLU(inplace=True),
            nn.Linear(self.vectorized, self.vectorized),
            nn.ReLU(inplace=True),
            nn.Linear(self.vectorized, self.vectorized),
            nn.ReLU(inplace=True),
            nn.Linear(self.vectorized, self.vectorized),
            nn.ReLU(inplace=True),
            nn.Linear(self.vectorized, self.vectorized),
            nn.ReLU(inplace=True),
            nn.Linear(self.vectorized, self.vectorized)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, pad_frame_size, patch_shape):

        height = x.size(3)
        width = x.size(4)

        # Compute measurements
        x = self.measurements(x, pad_frame_size, patch_shape).detach()

        # Map measurements to video blocks
        out = Variable(torch.zeros(x.size(0), x.size(1), self.vectorized))
        for i in range(x.size(1)):
            out[:, i, :] = self.reconstruction(x[:, i, :])

        output_patches = out.data.view(out.size(0), self.measurements.patches_size[0],
                                       self.measurements.patches_size[1], self.measurements.patches_size[2], self.temporal_size, self.spatial_size, self.spatial_size).permute(0, 1, 4, 2, 3, 6, 5)

        # Reconstruct video blocks to video
        reconstructed_video = Variable(
            self.measurements.fold(output_patches)[0].cuda())

        # Crop padding
        reconstructed_video = reconstructed_video[:, :, :, 0:height, 0:width]

        return reconstructed_video


def fc8net(block_size, pretrained=None, mask_path=None, mean=None, std=None, noise=None):

    model = FC8Net(block_size, mask_path, mean, std, noise)
    if pretrained is not None:
        checkpoint = torch.load(pretrained)
        model.load_state_dict(checkpoint['state_dict'])
    return model
