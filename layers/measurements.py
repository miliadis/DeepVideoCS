import math
import torch
import h5py
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


class Measurements(nn.Module):

    def __init__(self, spatial_size, temporal_size, overlapping, mask=None, mean=None, std=None, noise=None):
        super(Measurements, self).__init__()
        self.spatial_size = spatial_size
        self.temporal_size = temporal_size
        self.in_size = (spatial_size, spatial_size, temporal_size)
        self.out_size = (spatial_size, spatial_size)
        self.overlapping = overlapping
        self.noise = noise
        self.patches_size = None
        self.mean = None
        self.std = None

        if self.overlapping:
            self.step = self.spatial_size / 2
        else:
            self.step = self.spatial_size

        self.input_padded_size = None
        self.input_padded_numel = None

        self.weight = nn.Parameter(torch.Tensor(
            temporal_size, self.step, self.step))

        if mean is not None:
            self.mean = Variable(torch.from_numpy(h5py.File(
                mean).items()[0][1].value).float().cuda(), requires_grad=False)
            self.std = Variable(torch.from_numpy(h5py.File(
                std).items()[0][1].value).float().cuda(), requires_grad=False)

        if mask is not None:
            mask_data = np.load(mask)
            self.weight.data.copy_(torch.from_numpy(
                mask_data).double().permute(0, 2, 1))
        else:
            self.reset_parameters()

    def reset_parameters(self):
        n = self.spatial_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, pad_frame_size, patch_shape):

        n_patch_h = patch_shape[0][0].cpu().data.numpy()[0]
        n_patch_w = patch_shape[0][1].cpu().data.numpy()[0]

        input_padded = torch.zeros((input.size(0), input.size(1), input.size(
            2), pad_frame_size[0][0].cpu().data.numpy()[0], pad_frame_size[0][1].cpu().data.numpy()[0]))
        input_padded[:, :, :, 0:input.size(3), 0:input.size(4)] = input.data

        # save dimensions
        self.input_padded_size = input_padded.size()
        self.input_padded_numel = input_padded.numel()

        weight = self.weight.repeat(input.size(0), input.size(
            1), 1, n_patch_h + 1, n_patch_w + 1)

        output = torch.mul(input_padded.cuda(), weight.data).sum(2)

        if self.noise is not None:
            output = self.add_noise(output, input.size(), self.noise)

        output_patches = output.unfold(2, self.spatial_size, self.step).unfold(
            3, self.spatial_size, self.step)

        self.patches_size = (output_patches.size(
            1), output_patches.size(2), output_patches.size(3))
        output_patches = Variable(output_patches.permute(0, 1, 2, 3, 5, 4).contiguous(
        ).view((output_patches.size(0), -1, self.spatial_size**2)))

        if self.mean is not None:
            mean = self.mean.repeat(
                output_patches.size(0), output_patches.size(1), 1)
            std = self.std.repeat(output_patches.size(0),
                                  output_patches.size(1), 1)
            output_patches -= mean
            output_patches = output_patches / std

        return output_patches

    # This is the opposite of unfold:
    # Combines video blocks into video frames
    # and averages the result due to overlapping (if true)
    def fold(self, patches):
        idx = patches.new().long()
        torch.arange(0, self.input_padded_numel, out=idx)
        idx = idx.view(self.input_padded_size)
        idx_unfolded = idx.unfold(3, self.spatial_size, self.step).unfold(
            4, self.spatial_size, self.step)
        idx_unfolded = idx_unfolded.contiguous().view(-1)
        video = patches.new(self.input_padded_numel).zero_()
        video_ones = patches.new(self.input_padded_numel).zero_()
        patches_ones = torch.zeros(patches.size()) + 1
        patches = patches.contiguous().view(-1)
        patches_ones = patches_ones.contiguous().view(-1)
        video.index_add_(0, idx_unfolded, patches)
        video_ones.index_add_(0, idx_unfolded, patches_ones)
        return (video / video_ones).view(self.input_padded_size), None, None, None

    def add_noise(self, input, input_or_size, SNR):
        # x : input signal (image: M x N, or vector: M*N x 1)
        # SNR: Singal to Noise Ratio

        input_or = input[:, :, 0:input_or_size[3], 0:input_or_size[4]]
        variance = torch.zeros(input_or.size())

        # Noise variance, calculated through SNR
        # Create the noise element (multiply by standard deviation sqrt(Ps))
        for batch in range(input_or.shape[0]):
            for slice in range(input_or.shape[1]):
                Ps = input_or[batch, slice, :, :].var()
                variance_scalar = torch.FloatTensor([Ps / 10**(SNR / 10)])
                variance[batch, slice, :, :] = variance_scalar.repeat(
                    input_or.size(2), input_or.size(3))

        mult = torch.randn(input_or.size(), out=None)
        n = torch.mul(mult, torch.sqrt(variance)).cuda()

        # Add noise to the input
        y = input_or + n

        y_pad = torch.zeros(input.size()).cuda()
        y_pad[:, :, 0:input_or_size[3], 0:input_or_size[4]] = y

        return y_pad

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_size) + ' -> ' \
            + str(self.out_size) + ')'
