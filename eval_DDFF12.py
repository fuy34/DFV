import argparse
import cv2
from models import DFFNet
import numpy as np
import os
import skimage.filters as skf
import time
from models.submodule import *

import  matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torchvision


'''
Code for Ours-FV and Ours-DFV evaluation on DDFF-12 dataset  
'''

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser(description='DFVDFF')
parser.add_argument('--data_path', default='/data/DFF/my_ddff_trainVal.h5',help='test data path')
parser.add_argument('--loadmodel', default=None, help='model path')
parser.add_argument('--outdir', default='./DDFF12/',help='output dir')

parser.add_argument('--max_disp', type=float ,default=0.28, help='maxium disparity')
parser.add_argument('--min_disp', type=float ,default=0.02, help='minium disparity')

parser.add_argument('--stack_num', type=int ,default=5, help='num of image in a stack, please take a number in [2, 10], change it according to the loaded checkpoint!')
parser.add_argument('--use_diff', default=1, choices=[0,1], help='if use differential images as input, change it according to the loaded checkpoint!')

parser.add_argument('--level', type=int, default=4, help='num of layers in network, please take a number in [1, 4]')
args = parser.parse_args()

# !!! Only for users who download our pre-trained checkpoint, comment the next four line if you are not !!!
if os.path.basename(args.loadmodel) == 'DFF-DFV.tar' :
    args.use_diff = 1
else:
    args.use_diff = 0

# dataloader
from dataloader import DDFF12Loader

# construct model
model = DFFNet(clean=False,level=args.level, use_diff=args.use_diff)
model = nn.DataParallel(model)
model.cuda()
ckpt_name = os.path.basename(os.path.dirname(args.loadmodel))

if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('run with random init')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def calmetrics( pred, target, mse_factor, accthrs, bumpinessclip=0.05, ignore_zero=True):
    metrics = np.zeros((1, 7 + len(accthrs)), dtype=float)

    if target.sum() == 0:
        return metrics

    pred_ = np.copy(pred)
    if ignore_zero:
        pred_[target == 0.0] = 0.0
        numPixels = (target > 0.0).sum()  # number of valid pixels
    else:
        numPixels = target.size

    # euclidean norm
    metrics[0, 0] = np.square(pred_ - target).sum() / numPixels * mse_factor

    # RMS
    metrics[0, 1] = np.sqrt(metrics[0, 0])

    # log RMS
    logrms = (np.ma.log(pred_) - np.ma.log(target))
    metrics[0, 2] = np.sqrt(np.square(logrms).sum() / numPixels)

    # absolute relative
    metrics[0, 3] = np.ma.divide(np.abs(pred_ - target), target).sum() / numPixels

    # square relative
    metrics[0, 4] = np.ma.divide(np.square(pred_ - target), target).sum() / numPixels

    # accuracies
    acc = np.ma.maximum(np.ma.divide(pred_, target), np.ma.divide(target, pred_))
    for i, thr in enumerate(accthrs):
        metrics[0, 5 + i] = (acc < thr).sum() / numPixels * 100.

    # badpix
    metrics[0, 8] = (np.abs(pred_ - target) > 0.07).sum() / numPixels * 100.

    # bumpiness -- Frobenius norm of the Hessian matrix
    diff = np.asarray(pred - target, dtype='float64')  # PRED or PRED_
    chn = diff.shape[2] if len(diff.shape) > 2 else 1
    bumpiness = np.zeros_like(pred_).astype('float')
    for c in range(0, chn):
        if chn > 1:
            diff_ = diff[:, :, c]
        else:
            diff_ = diff
        dx = skf.scharr_v(diff_)
        dy = skf.scharr_h(diff_)
        dxx = skf.scharr_v(dx)
        dxy = skf.scharr_h(dx)
        dyy = skf.scharr_h(dy)
        dyx = skf.scharr_v(dy)
        hessiannorm = np.sqrt(np.square(dxx) + np.square(dxy) + np.square(dyy) + np.square(dyx))
        bumpiness += np.clip(hessiannorm, 0, bumpinessclip)
    bumpiness = bumpiness[target > 0].sum() if ignore_zero else bumpiness.sum()
    metrics[0, 9] = bumpiness / chn / numPixels * 100.

    return metrics


def main(image_size = (383, 552)):
    model.eval()

    # Calculate pad size for images
    test_pad_size = (np.ceil((image_size[0] / 32)) * 32, np.ceil((image_size[1] / 32)) * 32)

    # Create test set transforms
    transform_test = [DDFF12Loader.ToTensor(),
                      DDFF12Loader.PadSamples(test_pad_size),
                      DDFF12Loader.Normalize(mean_input=[0.485, 0.456, 0.406],std_input=[0.229, 0.224, 0.225])]
    transform_test = torchvision.transforms.Compose(transform_test)

    test_set = DDFF12Loader(args.data_path, stack_key="stack_val", disp_key="disp_val", transform=transform_test,
                            n_stack=args.stack_num,
                            min_disp=args.min_disp, max_disp=args.max_disp, b_test=True
                            )
    dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)


    # metric prepare
    accthrs = [1.25, 1.25 ** 2, 1.25 ** 3]
    avgmetrics = np.zeros((1, 7 + len(accthrs) + 1), dtype=float)
    test_num = len(dataloader)
    time_rec = np.zeros(test_num)
    for inx, (img_stack, disp, foc_dist) in enumerate(dataloader):
        # if inx not in [34, 19, 25, 74, 108]: continue # paper viz images
        if inx % 10 == 0:
            print('processing: {}/{}'.format(inx, test_num))

        # print(img_stack.shape, disp.shape, foc_dist)

        img_stack = Variable(torch.FloatTensor(img_stack)).cuda()
        gt_disp = Variable(torch.FloatTensor(disp))

        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            pred_disp, std, focusMap = model(img_stack, foc_dist)
            torch.cuda.synchronize()
            ttime = (time.time() - start_time); #print('time = %.2f' % (ttime*1000) )
            time_rec[inx] = ttime

        pred_disp = pred_disp.squeeze().cpu().numpy()[:image_size[0], :image_size[1]]
        gt_disp = gt_disp.squeeze().numpy()[:image_size[0], :image_size[1]]

        # uncomment if need to generate viz shown in the paper
        # if inx in [116]:
        #     # plt.imshow(std[:image_size[0], :image_size[1]].squeeze().detach().cpu().numpy())
        #     # plt.show()
        #     img_save_pth = os.path.join(args.outdir, ckpt_name) #'figure_paper'#
        #     if not os.path.isdir(img_save_pth):
        #         os.makedirs(img_save_pth)
        #     img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=img_stack.dtype).view(3, 1, 1)
        #     img_std = torch.tensor([0.229, 0.224, 0.225], dtype=img_stack.dtype).view(3, 1, 1)
        #     # print(img_stack.shape, img_std.shape, img_mean.shape)
        #     cv2.imwrite('{}/{}_img.png'.format(img_save_pth, inx),
        #                 ((img_stack[0,1,:,:image_size[0], :image_size[1]].detach().cpu() * img_std + img_mean)*255).numpy().transpose(1,2,0)[:,:,::-1])
        #
        #     MAX_DISP, MIN_DISP = 0.28, 0.02
        #     # pred_disp = pred_disp.squeeze().detach().cpu().numpy()
        #     plt.imshow(pred_disp, vmax=MAX_DISP, vmin=MIN_DISP)  # val2uint8(, MAX_DISP, MIN_DISP)
        #     plt.axis('off')
        #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #     plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        #     plt.margins(0, 0)
        #     plt.savefig('{}/{}_pred_viz.png'.format(img_save_pth, inx, args.level), bbox_inches='tight', pad_inches=0)
        #     plt.close()
        #
        #     plt.imshow(gt_disp, vmax=MAX_DISP, vmin=MIN_DISP)  # val2uint8(, MAX_DISP, MIN_DISP)
        #     plt.axis('off')
        #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #     plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        #     plt.margins(0, 0)
        #     plt.savefig('{}/{}_gt_viz.png'.format(img_save_pth, inx), bbox_inches='tight', pad_inches=0)
        #     plt.close()
        #
        #     print(std.max(), std.min(), std.shape)
        #     plt.imshow(std[:image_size[0], :image_size[1]].squeeze().detach().cpu().numpy(), vmax=0.1, vmin=0)  # val2uint8(, MAX_DISP, MIN_DISP)
        #     plt.axis('off')
        #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #     plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        #     plt.margins(0, 0)
        #     plt.savefig('{}/{}_std_viz.png'.format(img_save_pth, inx,  args.level), bbox_inches='tight', pad_inches=0)
        #     plt.close()
        #
        #     for i in range(args.stack_num):
        #         MAX_DISP, MIN_DISP = 1, 0
        #         plt.imshow(focusMap[i][:image_size[0], :image_size[1]].squeeze().detach().cpu().numpy(), vmax=MAX_DISP, vmin=MIN_DISP, cmap='jet')  # val2uint8(, MAX_DISP, MIN_DISP)
        #         plt.axis('off')
        #         plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #         plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #         plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        #         plt.margins(0, 0)
        #         plt.savefig('{}/{}_{}_prob_dist.png'.format(img_save_pth, inx, i), bbox_inches='tight', pad_inches=0)
        #
        #     mask = (gt_disp > 0)#.float()
        #     err = (np.abs(pred_disp - gt_disp) * mask).clip(0, 0.1)
        #
        #     cv2.imwrite('{}/{}_err.png'.format(img_save_pth, inx), err*2550)

        metrics = calmetrics( pred_disp, gt_disp, 1.0, accthrs, bumpinessclip=0.05, ignore_zero=True)
        avgmetrics[:,:-1] += metrics
        avgmetrics[:, -1] += std.mean().detach().cpu().numpy()

        torch.cuda.empty_cache()

    final_res = (avgmetrics /test_num)[0]
    final_res = np.delete(final_res, 8) # remove badpix result, we do not use it in our paper
    print('==============  Final result =================')
    print("\n  " + ("{:>10} | " * 10).format("MSE", "RMS", "log RMS", "Abs_rel", "Sqr_rel", "a1", "a2", "a3", "bump", "avgUnc"))
    print(("  {: 2.6f}  " * 10).format(*final_res.tolist()) )
    print('runtime mean', np.mean(time_rec[1:])) # first one usually very large due to the pytorch warm up, discard


if __name__ == '__main__':
    main()

