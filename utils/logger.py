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
                else:
                    images = images.unsqueeze(1)
        images = make_grid(images,padding=5)
        self.writer.add_image(tag, images, step)
