import torch.utils.data as data
import os.path
import imageio
import numpy as np
import torch
import math
import h5py
import glob


def get_hdf5_all_samples(videos):
    count = 0
    for video in videos:
        count += h5py.File(video, 'r')['label'].shape[0]
    return count


def make_dataset(dir, hdf5):
    if hdf5:
        videos = [x for x in glob.glob(os.path.join(dir, '*.h5'))]
        number_samples = get_hdf5_all_samples(videos)
    else:
        videos = os.listdir(dir)
        number_samples = len(videos)
    return videos, number_samples


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


def hdf5_loader(hdf5_file, video_patch):
    h5_file = h5py.File(hdf5_file, 'r')
    dataset = h5_file['label']
    data = np.transpose(np.squeeze(np.asarray(dataset).reshape(
        (len(dataset), video_patch[0], video_patch[1], video_patch[2]), order="F")), (0, 3, 1, 2))
    data = np.uint8(data * 255)
    return data


class VideoCS(data.Dataset):

    def __init__(self, root, block_size, transform=None, loader=default_loader, hdf5=False):
        self.hdf5 = hdf5
        videos, number_of_samples = make_dataset(root, self.hdf5)
        if len(videos) == 0:
            raise(RuntimeError("Found 0 videos in subfolders of: " + root + "\n"))

        # Parameters
        self.video_patch = [block_size[1], block_size[1], block_size[0]]
        self.chunks = block_size[2]
        self.overlapping = block_size[3]

        if self.hdf5:
            self.videos = sorted(videos)
            self.hdf5_index = 0
            self.data = hdf5_loader(
                self.videos[self.hdf5_index], self.video_patch)
            self.hdf5_limit = self.data.shape[0]
            self.data_size = self.data.shape[0]
        else:
            self.videos = videos
            self.root = root

        self.number_of_samples = number_of_samples
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
        if self.hdf5:
            if index >= self.hdf5_limit:
                self.hdf5_index = (self.hdf5_index + 1) % len(self.videos)
                self.data = hdf5_loader(
                    self.videos[self.hdf5_index], self.video_patch)
                self.hdf5_limit += self.data.shape[0]
                self.data_size = self.data.shape[0]
            if index == (self.number_of_samples - 1):
                self.hdf5_limit = 0
            video_np = self.data[index % self.data_size]
        else:
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

        return frames, np.asarray(self.pad_frame_size, dtype=np.int), np.asarray(patch_shape, dtype=np.int)

    def __len__(self):
        return self.number_of_samples
