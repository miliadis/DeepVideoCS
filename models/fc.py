import torch
import torch.nn as nn
from torch.autograd import Variable
import layers as mynn
import math
from collections import OrderedDict

__all__ = ['FCNet', 'fcnet']


class FCNet(nn.Module):

    def __init__(self, block_opts, mask_path, mean, std, noise_level, encoder_learn, bernoulli_p, K):
        super(FCNet, self).__init__()

        self.encoder_learn = encoder_learn

        self.spatial_size = block_opts[1]
        self.temporal_size = block_opts[0]
        self.vectorized = self.spatial_size * self.spatial_size * self.temporal_size

        self.pad_frame_size = None
        self.patch_shape = None

        self.measurements = mynn.Measurements(
            self.spatial_size, self.temporal_size, bernoulli_p, overlapping=block_opts[3], mask=mask_path, mean=mean, std=std, noise=noise_level)

        layers = OrderedDict()
        layers['linear' + str(0)] = nn.Linear(self.spatial_size *
                                              self.spatial_size, self.vectorized)
        for i in range(1, K + 1):
            layers['relu' + str(i)] = nn.ReLU(inplace=True)
            layers['linear' +
                   str(i)] = nn.Linear(self.vectorized, self.vectorized)
        self.reconstruction = nn.Sequential(layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.in_features
                scale = math.sqrt(3. / n)
                m.weight.data.uniform_(-scale, scale)
                m.bias.data.zero_()

    def forward(self, x):

        height = x.size(3)
        width = x.size(4)

        # Compute measurements
        x, y = self.measurements(x, self.pad_frame_size, self.patch_shape)

        if self.encoder_learn is False:
            x = x.detach()
            y = y.detach()

        # Map measurements to video blocks
        out = Variable(torch.zeros(
            x.size(0), x.size(1), self.vectorized)).cuda()
        for i in range(x.size(1)):
            out[:, i, :] = self.reconstruction(x[:, i, :])

        output_patches = out.view(out.size(0), self.measurements.patches_size[0],
                                  self.measurements.patches_size[1], self.measurements.patches_size[2], self.temporal_size, self.spatial_size, self.spatial_size).permute(0, 1, 4, 2, 3, 6, 5)

        # Reconstruct video blocks to video
        reconstructed_video = self.measurements.fold(output_patches)[0]

        # Crop padding
        reconstructed_video = reconstructed_video[:, :, :, 0:height, 0:width]

        return reconstructed_video, y


def fcnet(block_size, pretrained=None, mask_path=None, mean=None, std=None, noise=None, encoder_learn=False, p=50, K=7):
    model = FCNet(block_size, mask_path, mean, std, noise, encoder_learn, p, K)
    if pretrained is not None:
        checkpoint = torch.load(pretrained)
        state_dict = checkpoint['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    return model
