#! /usr/bin/python3

import os
import numpy as np
from torch.utils.data import Dataset
import torchvision
import torch
import h5py

# code adopted from https://github.com/soyers/ddff-pytorch/blob/master/python/ddff/dataproviders/datareaders/FocalStackDDFFH5Reader.py


class DDFF12Loader(Dataset):

    def __init__(self, hdf5_filename,  stack_key="stack_train", disp_key="disp_train", transform=None,
                 n_stack=10, min_disp=0.02, max_disp=0.28, b_test=False):
        """
        Args:
            root_dir_fs (string): Directory with all focal stacks of all image datasets.
            root_dir_depth (string): Directory with all depth images of all image datasets.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Disable opencv threading since it leads to deadlocks in PyTorch DataLoader
        self.hdf5 = h5py.File(hdf5_filename, 'r')
        self.stack_key = stack_key
        self.disp_key = disp_key
        self.max_n_stack = 10
        self.b_test = b_test

        assert n_stack <= self.max_n_stack, 'DDFF12 has maximum 10 images per stack!'
        self.n_stack = n_stack
        self.disp_dist = torch.linspace(max_disp,min_disp, steps=self.max_n_stack)

        if transform is None:
            if 'train' in self.stack_key:
                self.transform = self.__create_preprocessing(crop_size=(224, 224), cliprange=None, b_filp=True)
            else:
                transform_test = [DDFF12Loader.ToTensor(),
                                  DDFF12Loader.PadSamples((384, 576)),
                                  DDFF12Loader.Normalize(mean_input=[0.485, 0.456, 0.406],
                                                         std_input=[0.229, 0.224, 0.225])]
                self.transform =  torchvision.transforms.Compose(transform_test) #self.__create_preprocessing()
        else:
            self.transform = transform

    def __len__(self):
        return self.hdf5[self.stack_key].shape[0]

    def __getitem__(self, idx):
        # Create sample dict
        try:
            if 'test' in self.stack_key:
                sample =  {'input': self.hdf5[self.stack_key][idx].astype(float), 'output': np.ones([2,2])}
            else:
                sample = {'input': self.hdf5[self.stack_key][idx].astype(float), 'output': self.hdf5[self.disp_key][idx]}
        except:
            sample = None
            for _ in range(100):
                sample = {'input': self.hdf5[self.stack_key][idx].astype(float), 'output': self.hdf5[self.disp_key][idx]}
                if sample is not None:
                    break
            if sample is None:
                a =  self.hdf5[self.stack_key][idx].astype(float)
                b = self.hdf5[self.disp_key][idx]
                print(len(self.hdf5[self.stack_key]), idx, a is None, b is None)
                exit(1)

        # Transform sample with data augmentation transformers
        if self.transform :
            sample_out = self.transform(sample)

        # we do not experiment more than 10
        if self.n_stack < self.max_n_stack:
            if 'train' in self.disp_key:
                rand_idx = np.random.choice(self.max_n_stack, self.n_stack, replace=False) # this will shuffle order as well
                rand_idx = np.sort(rand_idx)
            else:
                rand_idx = np.linspace(0, 9, self.n_stack)

            out_imgs = sample_out['input'][rand_idx]
            out_disp = sample_out['output']
            disp_dist = self.disp_dist[rand_idx]
        else:
            out_imgs = sample_out['input']
            out_disp = sample_out['output']
            disp_dist = self.disp_dist

        if 'val' in self.disp_key and (not self.b_test):
            out_disp = out_disp[:, :256, :256]
            out_imgs = out_imgs[:,:, :256, :256]

        return out_imgs, out_disp, disp_dist #sample_out['input'], sample_out['output'].squeeze()#, sample['input']

    def __create_preprocessing(self, crop_size=None, cliprange=[0.0202, 0.2825], mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225], b_filp=True):
        # The real data mean and std
        # mean [0.40134945 0.48795037 0.45252803], std [0.14938343, 0.15709994, 0.1496986]
        if b_filp:
            transform = [self.RandomFilp(), self.ToTensor()]
        else:
            transform = [self.ToTensor()]

        if cliprange is not None:
            transform += [self.ClipGroundTruth(cliprange[0], cliprange[1])]
        if crop_size is not None:
            transform += [self.RandomCrop(crop_size)]
        if mean is not None and std is not None:
            transform += [self.Normalize(mean_input=mean, std_input=std)]
        transform = torchvision.transforms.Compose(transform)
        return transform

    def get_stack_size(self):
        return self.__getitem__(0)['input'].shape[0]

    class ToTensor(object):
        """Convert ndarrays in sample to Tensors."""

        def __call__(self, sample):
            # Add color dimension to depth map
            sample['output'] = np.expand_dims(sample['output'], axis=0)
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            sample['input'] = torch.from_numpy(sample['input'].transpose((0, 3, 1, 2))).float().div(255) #I add div 255
            sample['output'] = torch.from_numpy(sample['output']).float()
            return sample

    class Normalize(object):
        def __init__(self, mean_input, std_input, mean_output=None, std_output=None):
            self.mean_input = mean_input
            self.std_input = std_input
            self.mean_output = mean_output
            self.std_output = std_output

        def __call__(self, sample):
            img_lst = []
            samples = sample['input']

            for i, sample_input in enumerate(samples):
                img_lst.append(torchvision.transforms.functional.normalize(sample_input, mean=self.mean_input, std=self.std_input))
            input_images = torch.stack(img_lst)

            if self.mean_output is None or self.std_output is None:
                output_image = sample['output']
            else:
                output_image = torchvision.transforms.functional.normalize(sample['output'], mean=self.mean_output,
                                                                    std=self.std_output)


            return {'input': input_images, 'output': output_image}

    class ClipGroundTruth(object):
        def __init__(self, lower_bound, upper_bound):
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

        def __call__(self, sample):
            sample['output'][sample['output'] < self.lower_bound] = 0.0
            sample['output'][sample['output'] > self.upper_bound] = 0.0
            return sample

    class RandomFilp(object):
        """ Randomly crop images
        """

        def __init__(self, ratio=0.5):
            self.ratio = ratio

        def __call__(self, sample):
            inputs, target = sample['input'], sample['output']

            # hori filp
            if np.random.binomial(1, self.ratio):
                inputs = inputs[:, :, ::-1]
                target = target[:, ::-1]

            # vert flip
            if np.random.binomial(1, self.ratio):
                inputs = inputs[:, ::-1]
                target = target[::-1]

            return {'input': np.ascontiguousarray(inputs), 'output': np.ascontiguousarray(target)}


    class RandomCrop(object):
        def __init__(self, output_size, valid_crop_threshold=0.8):
            assert isinstance(output_size, (int, tuple))
            if isinstance(output_size, int):
                self.output_size = (output_size, output_size)
            else:
                assert len(output_size) == 2
                self.output_size = output_size
            self.valid_crop_threshold = valid_crop_threshold

        def __is_valid_crop(self, output_image, valid_pixel_cond=lambda x: x >= 0.01):
            valid_occurrances = valid_pixel_cond(output_image).sum()
            all_occurances = np.prod(output_image.shape)
            return (float(valid_occurrances) / float(all_occurances)) >= self.valid_crop_threshold

        def __call__(self, sample):
            h, w = sample['input'].shape[2:4]
            new_h, new_w = self.output_size

            # Generate list of possible random crops
            candidates = np.asarray([(x, y) for y in range(h - new_h) for x in range(w - new_w)])
            np.random.shuffle(candidates)

            # Iterate through candidates and choose forst valid crop
            for x, y in candidates:
                output_image = sample['output'][:, y:(y + new_h), x:(x + new_w)]
                if self.__is_valid_crop(output_image):
                    input_images = sample['input'][:, :, y:(y + new_h), x:(x + new_w)]
                    return {'input': input_images, 'output': output_image}

            # No valid crop found. Return any crop
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            input_images =  sample['input'][:, :, top:(top + new_h), left:(left + new_w)]
            output_image = sample['output'][:, top:(top + new_h), left:(left + new_w)]
            return {'input': input_images, 'output': output_image}

    class PadSamples(object):
        def __init__(self, output_size, ground_truth_pad_value=0.0):
            assert isinstance(output_size, (int, tuple))
            if isinstance(output_size, int):
                self.output_size = (output_size, output_size)
            else:
                assert len(output_size) == 2
                self.output_size = output_size
            self.ground_truth_pad_value = ground_truth_pad_value

        def __call__(self, sample):
            h, w = sample['input'].shape[2:4]
            new_h, new_w = self.output_size
            padh = np.int32(new_h - h)
            padw = np.int32(new_w - w)
            sample['input'] = torch.stack(
                [torch.from_numpy(np.pad(sample_input.numpy(), ((0, 0), (0, padh), (0, padw)), mode="reflect")).float()
                 for sample_input in sample['input']])
            sample['output'] = torch.from_numpy(
                np.pad(sample['output'].numpy(), ((0, 0), (0, padh), (0, padw)), mode="constant",
                       constant_values=self.ground_truth_pad_value)).float()

            return sample

    class RandomSubStack(object):
        def __init__(self, output_size):
            self.output_size = output_size

        def __call__(self, sample):
            sample['input'] = torch.stack([sample['input'][i] for i in
                                           np.random.choice(sample['input'].shape[0], self.output_size, replace=False)])
            return sample
