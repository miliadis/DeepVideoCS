import torch.utils.data as data
import os.path
import imageio
import numpy as np
import torch
import math


def make_dataset(dir):
    videos = os.listdir(dir)
    return videos


def find_new_dims(frame_size, patch_size, step):
    new_h = int(
        (math.ceil((frame_size[1] - patch_size) / float(step)) * step) + patch_size)
    new_w = int(
        (math.ceil((frame_size[2] - patch_size) / float(step)) * step) + patch_size)
    new_n_patches_h = (new_h // step) - 1
    new_n_patches_w = (new_w // step) - 1
    return (new_h, new_w), (new_n_patches_h, new_n_patches_w)


def make_video_blocks(video, chunks, t_frames):
    slices = np.int(np.floor(video.shape[0] / t_frames))
    video = video[0:slices * t_frames, :, :]
    video = np.reshape(
        video, (slices, t_frames, video.shape[1], video.shape[2]))
    video = video[0:chunks, :, :, :]
    return video


def default_loader(path):
    reader = imageio.get_reader(path)
    video = np.zeros((reader._meta['nframes'], reader._meta['size']
                      [1], reader._meta['size'][0]), dtype=np.uint8)
    for i, im in enumerate(reader):
        video[i, :, :] = im.mean(2)
    return video


class VideoCS(data.Dataset):

    def __init__(self, root, block_size, transform=None, loader=default_loader):
        videos = make_dataset(root)
        if len(videos) == 0:
            raise(RuntimeError("Found 0 videos in subfolders of: " + root + "\n"))

        # Parameters
        self.video_patch = [block_size[1], block_size[1], block_size[0]]
        self.chunks = block_size[2]
        self.overlapping = block_size[3]

        self.root = root
        self.videos = videos
        self.transform = transform
        self.loader = loader
        if self.overlapping:
            self.overlap = self.video_patch[0] / 2
        self.or_frame_size = None
        self.pad_frame_size = None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            array: video chunks.
        """
        path = os.path.join(self.root, self.videos[index])
        video_np = self.loader(path)

        self.or_frame_size = (video_np.shape[1], video_np.shape[2])
        frame_size, patch_shape = find_new_dims(
            video_np.shape, self.video_patch[0], self.overlap)
        self.pad_frame_size = frame_size

        frames = make_video_blocks(video_np, self.chunks, self.video_patch[2])
        if self.transform is not None:
            frames_tensor = self.transform(frames.reshape(
                (-1, frames.shape[2], frames.shape[3])).transpose(1, 2, 0))
        frames = frames_tensor.view((frames.shape))

        return frames, np.asarray(self.pad_frame_size), np.asarray(patch_shape), self.videos[index]

    def __len__(self):
        return len(self.videos)
