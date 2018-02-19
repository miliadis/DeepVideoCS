import argparse
import os
import time
import logging
from utils.log import setup_logging, ResultsLog, save_checkpoint, results_add
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim.lr_scheduler as sc
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import models
import datasets
import numpy as np
import utils.metrics as metrics
import random
from torch.autograd import Variable
import layers.loss_functions as loss
from bisect import bisect_right


parser = argparse.ArgumentParser(
    description='PyTorch Video Compressive Sensing - Training')


parser.add_argument('data_train', help='path to training dataset')
parser.add_argument('data_val', help='path to validation dataset')
parser.add_argument('--hdf5', action='store_true', default=False)
parser.add_argument('--mean', default=None, help='Mean file')
parser.add_argument('--std', default=None, help='Standard deviation file')
parser.add_argument('--workers', default=0, type=int,
                    help='number of data loading workers (default: 0)')
parser.add_argument('--gpus', type=int, nargs='+',
                    help='GPUs list: e.g., 0 1', default=[0])


# Model params
parser.add_argument('arch', help='choose model name', default='fcnet')
parser.add_argument('layers_k', type=int, default=7,
                    help='number of FC layers in decoder')
parser.add_argument('--pretrained_net', help='pre-trained model path')
parser.add_argument('--mask_path', default=None,
                    help='provide a pre-defined compressive sensing mask')
parser.add_argument('--bernoulli_p', type=int, default=40,
                    help='percentage of 1s for creating mask')
parser.add_argument('--block_opts', type=int, nargs='+',
                    help='Item order: (temporal size, spatial size, video chunks)', default=[16, 8, 1])
parser.add_argument('--block_overlap', action='store_false',
                    help='overlapping blocks or not')
parser.add_argument('--noise', type=int,
                    help='Noise Level in dB: e.g., 20, 30, 40', default=None)
parser.add_argument('--seed', type=int, default=5347, help='random seed')


# Optimization
parser.add_argument('--epochs', default=1000, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=200, type=int,
                    help='mini-batch size (default: 200)')
parser.add_argument('--encoder_lr', default=0.1, type=float,
                    help='initial learning rate for encoder')
parser.add_argument('--decoder_lr', default=0.01, type=float,
                    help='initial learning rate for decoder')
parser.add_argument('--encoder_annual', type=float, nargs='+',
                    help='Item order: (divide by, for every # epochs, until epoch #, then lr=0)', default=[0.5, 10, 400])
parser.add_argument('--decoder_annual', type=float, nargs='+',
                    help='Item order: (divide by, at epoch [#])', default=[0.1, 400])
parser.add_argument('--gradient_clipping', default=10, type=int,
                    help='gradient clipping to prevent explosion')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight-decay', default=0, type=float,
                    help='weight decay (default: 0)')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')

# Monitoring
parser.add_argument('--print-freq', default=1000, type=int,
                    help='print frequency (default: 1000)')
parser.add_argument('--results_dir', default='./results', help='results dir')
parser.add_argument('--save', default='', help='folder to save checkpoints')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')


best_psnr = 0


def main():
    global args, best_psnr
    args = parser.parse_args()

    # massage args
    block_opts = []
    block_opts = args.block_opts
    block_opts.append(args.block_overlap)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.save is '':
        args.save = time_stamp
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log_%s.txt' % time_stamp))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.encoder_lr > 0:
        encoder_learn = True
    else:
        encoder_learn = False

    # create model
    if args.pretrained_net is not None:
        logging.info("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](
            block_opts, pretrained=args.pretrained_net, mask_path=args.mask_path, mean=args.mean, std=args.std,
            noise=args.noise, encoder_learn=encoder_learn, p=args.bernoulli_p, K=args.layers_k)
    else:
        logging.info("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](
            block_opts, mask_path=args.mask_path, mean=args.mean, std=args.std,
            noise=args.noise, encoder_learn=encoder_learn, p=args.bernoulli_p, K=args.layers_k)
        model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    # define loss function (criterion) and optimizer
    mseloss = loss.EuclideanDistance(args.batch_size)

    # annual scedule
    if encoder_learn:
        optimizer = torch.optim.SGD([
            {'params': model.module.measurements.parameters(), 'lr': args.encoder_lr},
            {'params': model.module.reconstruction.parameters()}],
            args.decoder_lr, momentum=args.momentum, weight_decay=args.weight_decay)

        def lambda1(epoch): return 0.0 if epoch >= args.encoder_annual[2] else (
            args.encoder_annual[0] ** bisect_right(range(args.encoder_annual[1], args.encoder_annual[2], args.encoder_annual[1]), epoch))

        def lambda2(
            epoch): return args.decoder_annual[0] ** bisect_right([args.decoder_annual[1]], epoch)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[lambda1, lambda2])
    else:
        optimizer = torch.optim.SGD([
            {'params': model.module.reconstruction.parameters()}],
            args.decoder_lr, momentum=args.momentum, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[args.decoder_annual[1]], gamma=args.decoder_annual[0])

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_psnr = checkpoint['best_psnr']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info("=> loaded checkpoint '{}' (epoch {})"
                         .format(args.resume, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        datasets.videocs.VideoCS(args.data_train, args.block_opts, transforms.Compose([
            transforms.ToTensor(),
        ]), hdf5=args.hdf5),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.videocs.VideoCS(args.data_val, args.block_opts, transforms.Compose([
            transforms.ToTensor(),
        ]), hdf5=False),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)

    # Save initial mask
    if encoder_learn:
        initial_weights = binarization(
            model.module.measurements.weight.clone())
        perc_1 = initial_weights.mean().cpu().data.numpy()[0]
        logging.info('Percentage of 1: {}'.format(perc_1))
        np.save(save_path + '/initial_mask.npy',
                model.module.measurements.weight.clone())
    else:
        # binarize weights
        model.module.measurements.binarization()
        perc_1 = model.module.measurements.weight.clone().mean().cpu().data.numpy()[
            0]
        logging.info('Percentage of 1: {}'.format(perc_1))

    # perform first validation
    validate(val_loader, model, encoder_learn)

    for epoch in range(args.start_epoch, args.epochs):

        # Annual schedule enforcement
        scheduler.step()

        logging.info(scheduler.get_lr())

        if encoder_learn:
            save_binary_weights_before = binarization(
                model.module.measurements.weight.clone())

        # train for one epoch
        train_loss = train(train_loader, model, optimizer, epoch,
                           mseloss, encoder_learn, args.gradient_clipping)

        if encoder_learn:
            save_binary_weights_after = binarization(
                model.module.measurements.weight.clone())
            diff = np.int(torch.abs(save_binary_weights_after -
                                    save_binary_weights_before).sum().cpu().data.numpy())
            perc_1 = save_binary_weights_after.mean().cpu().data.numpy()[0]
            logging.info(
                'Binary Weights Changed: {} - Percentage of 1: {}'.format(diff, perc_1))
        else:
            perc1 = model.module.measurements.weight.clone().mean().cpu().data.numpy()[0]
            logging.info('Percentage of 1: {}'.format(perc_1))

        # evaluate on validation set
        psnr = validate(val_loader, model, encoder_learn)

        # remember best psnr and save checkpoint
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_psnr': best_psnr,
            'optimizer': optimizer.state_dict(),
        }, is_best, path=save_path)
        results_add(epoch, results, train_loss, psnr)

        if encoder_learn:
            model.module.measurements.restore()


def binarization(weights):
    weights = weights.clamp(-1.0, 1.0)
    weights = 0.5 * (weights.sign() + 1)
    weights[weights == 0.5] = 1
    return weights


def train(train_loader, model, optimizer, epoch, mseloss, encoder_learn, gradient_clip):
    batch_time = metrics.AverageMeter()
    data_time = metrics.AverageMeter()
    losses = metrics.AverageMeter()
    psnr = metrics.AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (video_blocks, pad_block_size, block_shape) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = video_blocks.cuda(async=True)
        input_var = Variable(video_blocks.cuda())
        target_var = Variable(target)

        # compute output
        model.module.pad_frame_size = pad_block_size.numpy()
        model.module.patch_shape = block_shape.numpy()

        if encoder_learn:
            model.module.measurements.binarization()

        output, y = model(input_var)
        loss = mseloss.compute_loss(output, target_var)
        # record loss
        losses.update(loss.data[0], video_blocks.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        if encoder_learn:
            # restore real-valued weights
            model.module.measurements.restore()
            nn.utils.clip_grad_norm(model.module.parameters(), gradient_clip)
        else:
            nn.utils.clip_grad_norm(
                model.module.reconstruction.parameters(), gradient_clip)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                             epoch, i, len(train_loader), batch_time=batch_time,
                             data_time=data_time, loss=losses))
    return losses.avg


def validate(val_loader, model, encoder_learn):
    batch_time = metrics.AverageMeter()
    psnr = metrics.AverageMeter()

    # switch to evaluate mode
    model.cuda()
    model.eval()

    # binarize weights
    if encoder_learn:
        model.module.measurements.binarization()

    end = time.time()
    for i, (video_frames, pad_frame_size, patch_shape) in enumerate(val_loader):
        video_input = Variable(video_frames.cuda(async=True), volatile=True)
        print(val_loader.dataset.videos[i])

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

        logging.info('Test: [{0}/{1}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'PSNR {psnr.val:.3f} ({psnr.avg:.3f})'.format(
                         i + 1, len(val_loader), batch_time=batch_time,
                         psnr=psnr))

    # restore real-valued weights
    if encoder_learn:
        model.module.measurements.restore()

    print(' * PSNR {psnr.avg:.3f}'.format(psnr=psnr))

    return psnr.avg


if __name__ == '__main__':
    main()
