from __future__ import division
import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import random


class Measurements(nn.Module):

    def __init__(self, spatial_size, temporal_size, bernoulli_p, overlapping, mask=None, mean=None, std=None, noise=None):
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
            self.step = int(self.spatial_size / 2)
        else:
            self.step = self.spatial_size

        self.input_padded_size = None
        self.input_padded_numel = None

        self.weight = nn.Parameter(torch.Tensor(
            temporal_size, self.step, self.step))

        if mean is not None:
            self.mean = np.load(mean)
            self.std = np.load(std)

        if mask is not None:
            mask_data = np.load(mask)
            self.weight.data.copy_(torch.from_numpy(
                mask_data).double().permute(0, 2, 1))
        else:
            self.reset_parameters(bernoulli_p)

        self.save_weight = self.weight.data.clone()

    def reset_parameters(self, p):
        bernoulli_weights = torch.FloatTensor(
            self.weight.size()).bernoulli_(p / 100)
        n = self.temporal_size
        stdv = 1. / math.sqrt(n)

        weights_zero = bernoulli_weights[bernoulli_weights ==
                                         0].uniform_(-stdv, 0)
        weights_one = bernoulli_weights[bernoulli_weights == 1].uniform_(
            0, stdv)
        bernoulli_weights[bernoulli_weights == 0] = weights_zero
        bernoulli_weights[bernoulli_weights == 1] = weights_one

        self.weight.data.copy_(bernoulli_weights)

    def binarization(self):
        self.weight.data.clamp(-1.0, 1.0, out=self.weight.data)
        self.save_weight.copy_(self.weight.data)
        self.weight.data = 0.5 * (self.weight.data.sign() + 1)
        self.weight.data[self.weight.data == 0.5] = 1

    def restore(self):
        self.weight.data.copy_(self.save_weight)

    def forward(self, input, pad_frame_size, patch_shape):

        n_patch_h = patch_shape[0][0]
        n_patch_w = patch_shape[0][1]

        input_padded = Variable(torch.zeros((input.size(0), input.size(
            1), input.size(2), pad_frame_size[0][0], pad_frame_size[0][1]))).cuda()
        input_padded[:, :, :, 0:input.size(3), 0:input.size(4)] = input
        # save dimensions
        self.input_padded_size = input_padded.size()
        self.input_padded_numel = input_padded.numel()

        # This is the compressed frame!
        weight = self.weight.repeat(input.size(0), input.size(
            1), 1, n_patch_h + 1, n_patch_w + 1)
        output = torch.mul(input_padded, weight).sum(2)
        if self.noise is not None:
            output = self.add_noise(output, input.size(), self.noise)

        # Create patches from compressed frame
        output_patches = output.unfold(2, self.spatial_size, self.step).unfold(
            3, self.spatial_size, self.step)
        self.patches_size = (output_patches.size(
            1), output_patches.size(2), output_patches.size(3))
        output_patches = output_patches.permute(0, 1, 2, 3, 5, 4).contiguous().view(
            (output_patches.size(0), -1, self.spatial_size**2))

        if self.mean is not None:
            mean_var = Variable(torch.from_numpy(self.mean)).float().cuda()
            std_var = Variable(torch.from_numpy(self.std)).float().cuda()
            mean = mean_var.repeat(output_patches.size(
                0), output_patches.size(1), 1)
            std = std_var.repeat(output_patches.size(0),
                                 output_patches.size(1), 1)
            output_patches = output_patches - mean
            output_patches = output_patches / std

        return output_patches, output[:, :, 0:input.size(3), 0:input.size(4)]

    # This is the opposite of unfold:
    # Combines video blocks into video frames
    # and averages the result due to overlapping (if true)
    def fold(self, patches):
        idx = patches.data.new().long()
        torch.arange(0, self.input_padded_numel, out=idx)
        idx = idx.view(self.input_padded_size)
        idx_unfolded = idx.unfold(3, self.spatial_size, self.step).unfold(
            4, self.spatial_size, self.step)
        idx_unfolded = idx_unfolded.contiguous().view(-1)

        video = Variable(patches.data.new(self.input_padded_numel).zero_())
        video_ones = Variable(patches.data.new(
            self.input_padded_numel).zero_())
        patches_ones = Variable(torch.zeros(patches.size()) + 1).cuda()

        patches = patches.contiguous().view(-1)
        patches_ones = patches_ones.contiguous().view(-1)
        video.index_add_(0, Variable(idx_unfolded), patches)
        video_ones.index_add_(0, Variable(idx_unfolded), patches_ones)
        return (video / video_ones).view(self.input_padded_size), None, None, None

    def add_noise(self, input, input_or_size, SNR):
        # x : input signal (image: M x N, or vector: M*N x 1)
        # SNR: Signal to Noise Ratio

        if self.training:
            SNR = random.uniform(SNR - 10, SNR + 10)

        input_or = input[:, :, 0:input_or_size[3], 0:input_or_size[4]]

        # Noise variance, calculated through SNR
        input_var = input_or.contiguous().view(input_or.size(0), input_or.size(1), input_or.size(2) * input_or.size(3)).var(dim=2,
                                                                                                                            keepdim=True) / 10 ** (SNR / 10)
        variance = input_var.repeat(1, 1, input_or.size(2) * input_or.size(3)).view(input_or.size(0), input_or.size(1),
                                                                                    input_or.size(2), input_or.size(3))
        mult = torch.randn(input_or.size()).cuda()
        n = Variable(mult.mul(variance.data.sqrt())).cuda()

        # Add noise to the input
        y = input_or + n

        y_pad = Variable(torch.zeros(input.size())).cuda()
        y_pad[:, :, 0:input_or_size[3], 0:input_or_size[4]] = y

        return y_pad

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_size) + ' -> ' \
            + str(self.out_size) + ')'
