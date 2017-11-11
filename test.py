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
import torchvision.transforms as transforms
import datasets
import models
import utils.metrics as metrics


parser = argparse.ArgumentParser(
    description='PyTorch Video Compressive Sensing - Testing')
parser.add_argument('data',
                    help='path to dataset')
parser.add_argument('arch', help='choose model name', default='fc8net'),
parser.add_argument('pretrained_net', help='pre-trained model path'),
parser.add_argument('--block_opts', nargs='+',
                    help='Item order: (temporal size, spatial size, video chunks, overlapping)', default=[16, 8, 1, True])
parser.add_argument('--noise', type=int,
                    help='Noise Level in dB: e.g., 20, 30, 40', default=None)
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--mean', default=None,
                    help='Mean file: for now we support only .mat files'),
parser.add_argument('--std', default=None,
                    help='Standard deviation file: for now we only support .mat files')
parser.add_argument('--gpu_id', type=int, default=0, help='choose gpu id')


def main():
    global args
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](
        args.block_opts, pretrained=args.pretrained_net, mask_path=None, mean=args.mean, std=args.std, noise=args.noise)
    model = torch.nn.DataParallel(model, device_ids=[args.gpu_id]).cuda()

    # switch to evaluate mode
    model.eval()

    cudnn.benchmark = True

    # Data loading code
    testdir = os.path.join(args.data)

    test_loader = torch.utils.data.DataLoader(
        datasets.videocs.VideoCS(testdir, args.block_opts, transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)

    batch_time = metrics.AverageMeter()
    psnr = metrics.AverageMeter()

    end = time.time()
    for i, (video_frames, pad_frame_size, patch_shape,
            idx) in enumerate(test_loader):
        video_input = Variable(video_frames.cuda(async=True), volatile=True)
        print(idx)

        # compute output
        reconstructed_video = model(
            video_input,
            Variable(pad_frame_size),
            Variable(patch_shape))

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

    print(' * PSNR {psnr.avg:.3f}'.format(psnr=psnr))


if __name__ == '__main__':
    main()
