import os
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import numpy as np
import torch

class Logger(object):

    def __init__(self, log_dir, name=None):
        """Create a summary writer logging to log_dir."""
        if name is None:
            name = 'temp'

        self.savepth = os.path.join(log_dir, name)
        if not os.path.isdir(self.savepth):
            os.makedirs(self.savepth)
        self.writer = SummaryWriter(self.savepth)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step, n_log=4):

        if 'uint8' not in str(images.dtype):
            images = images / images.max()


        if len(images.shape) < 3:
            print('Tensor input is needed (dim>=3)')
            return
        elif len(images.shape) == 3:
            if images.shape[0] == 3 or images.shape[0] == 1:
                # we treat it as a single image
                self.writer.add_image(tag, images, step)
                return
            else:
                # we treat it as multiple image with channel 1:
                if isinstance(images, (np.ndarray, np.generic)):
                    images = np.expand_dims(images, axis=1)
                    # images = np.concatenate([images, images, images], axis=1)
                else:
                    images = images.unsqueeze(1)
                    # images = torch.cat([images, images, images], dim=1)
        images = make_grid(images,padding=5)
        # print(images.max(), images.min(), images.shape)
        self.writer.add_image(tag, images, step)

    # def val2uint8(mat, maxVal, minVal=0):
    #     maxVal_mat = np.ones(mat.shape) * maxVal
    #     minVal_mat = np.ones(mat.shape) * minVal
    #
    #     mat_vis = np.where(mat > maxVal_mat, maxVal_mat, mat)
    #     mat_vis = np.where(mat < minVal_mat, minVal_mat, mat_vis)
    #     return (mat_vis * 255. / maxVal).astype(np.uint8)
