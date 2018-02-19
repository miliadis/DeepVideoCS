import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from utils.log import write_video
import torchvision.transforms as transforms
import datasets
import models
import utils.metrics as metrics
import numpy as np


parser = argparse.ArgumentParser(
    description='PyTorch Video Compressive Sensing - Testing')
parser.add_argument('data', help='path to testing dataset')
parser.add_argument('arch', help='choose model name', default='fcnet'),
parser.add_argument('layers_k', type=int, default=7,
                    help='number of FC layers in decoder')
parser.add_argument('pretrained_net', help='pre-trained model path'),
parser.add_argument('--block_opts', type=int, nargs='+',
                    help='Item order: (temporal size, spatial size, video chunks)', default=[16, 8, 2])
parser.add_argument('--block_overlap', action='store_false',
                    help='overlapping blocks or not')
parser.add_argument('--noise', type=int,
                    help='Noise Level in dB: e.g., 20, 30, 40', default=None)
parser.add_argument('--mean', default=None,
                    help='Mean file'),
parser.add_argument('--std', default=None,
                    help='Standard deviation file')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--save_videos', default=None, help='path to save videos'),
parser.add_argument('--save_format', default='avi',
                    help='format for saving video file: avi, gif'),
parser.add_argument('--gpu_id', type=int, default=0, help='choose gpu id')


def main():
    global args
    args = parser.parse_args()

    # massage args
    block_opts = []
    block_opts = args.block_opts
    block_opts.append(args.block_overlap)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](
        block_opts, pretrained=args.pretrained_net, mask_path=None, mean=args.mean, std=args.std,
        noise=args.noise, K=args.layers_k)
    model = torch.nn.DataParallel(model, device_ids=[args.gpu_id]).cuda()

    # switch to evaluate mode
    model.eval()
    cudnn.benchmark = True

    # Data loading code
    testdir = os.path.join(args.data)

    test_loader = torch.utils.data.DataLoader(
        datasets.videocs.VideoCS(testdir, block_opts, transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)

    batch_time = metrics.AverageMeter()
    psnr = metrics.AverageMeter()

    # binarize weights
    model_weights = model.module.measurements.weight.data
    if ((model_weights == 0) | (model_weights == 1)).all() == False:
        model.module.measurements.binarization()

    end = time.time()
    for i, (video_frames, pad_frame_size, patch_shape) in enumerate(test_loader):
        video_input = Variable(video_frames.cuda(async=True), volatile=True)
        print(test_loader.dataset.videos[i])

        # compute output
        model.module.pad_frame_size = pad_frame_size.numpy()
        model.module.patch_shape = patch_shape.numpy()
        reconstructed_video, y = model(video_input)

        # original video
        reconstructed_video = reconstructed_video.cpu().data.numpy()
        original_video = video_input.cpu().data.numpy()

        # measure accuracy and record loss
        psnr_video = metrics.psnr_accuracy(reconstructed_video, original_video)
        psnr.update(psnr_video, video_frames.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'PSNR {psnr.val:.3f} ({psnr.avg:.3f})'.format(
                  i + 1, len(test_loader), batch_time=batch_time,
                  psnr=psnr))

        if args.save_videos is not None:
            save_path = os.path.join(args.save_videos, args.save_format)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            y_repeat = torch.zeros(
                *y.size()).unsqueeze(2).repeat(1, 1, args.block_opts[0], 1, 1)
            for j in range(y.size(1)):
                y_repeat[:, j, :, :, :] = y[:, j, :, :].repeat(
                    1, args.block_opts[0], 1, 1).data
            y_repeat = y_repeat.numpy()

            original_video = np.reshape(
                original_video, (original_video.shape[0] * original_video.shape[1] * original_video.shape[2], original_video.shape[3], original_video.shape[4]))
            reconstructed_video = np.reshape(reconstructed_video, (reconstructed_video.shape[0] * reconstructed_video.shape[1] *
                                                                   reconstructed_video.shape[2], reconstructed_video.shape[3], reconstructed_video.shape[4])) / np.max(reconstructed_video)
            y_repeat = np.reshape(y_repeat, (y_repeat.shape[0] * y_repeat.shape[1] *
                                             y_repeat.shape[2], y_repeat.shape[3], y_repeat.shape[4])) / np.max(y_repeat)

            write_video(save_path, test_loader.dataset.videos[i], np.dstack(
                (original_video, y_repeat, reconstructed_video)), args.save_format)

    print(' * PSNR {psnr.avg:.3f}'.format(psnr=psnr))


if __name__ == '__main__':
    main()
