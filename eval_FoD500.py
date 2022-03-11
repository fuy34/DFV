import numpy as np
import skimage.filters as skf
import argparse
from glob import glob
import cv2

'''
Code for Ours-FV and Ours-DFV evaluation on FoD500 dataset  
'''

# For FoD500 eval, please run ''FoD_test.py'' first and set res_path as the outdir path in ''FoD_test.py''
parser = argparse.ArgumentParser(description='DFVDFF')
parser.add_argument('--res_path', default=None, help='test result path')
args = parser.parse_args()



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

if __name__ == '__main__':
    # init metric
    accthrs = [1.25, 1.25 ** 2, 1.25 ** 3]
    avgmetrics = np.zeros((1, 7 + len(accthrs) + 1), dtype=float)

    # locate results
    pred_list = glob(args.res_path + '/*_pred.png')
    gt_list = glob(args.res_path + '/*_gt.png')

    pred_list.sort()
    gt_list.sort()
    test_num = len(gt_list)

    for pred_pth, gt_pth in zip(pred_list, gt_list):

        pred = (cv2.imread(pred_pth, -1) / 10000.)
        gt = (cv2.imread(gt_pth, -1) / 10000.)

        pred = pred.clip(0, 1.5)
        gt = gt.clip(0, 1.5)


        metrics = calmetrics(pred, gt, 1.0, accthrs, bumpinessclip=0.05, ignore_zero=True)
        avgmetrics[:, :-1] += metrics

    # final_res = avgmetrics / test_num
    # print('final result', final_res)

    final_res = (avgmetrics / test_num)[0]
    final_res = np.delete(final_res, 8)  # remove badpix result, we do not use it
    print('==============  Final result =================')
    print("\n  " + ("{:>10} | " * 9).format("MSE", "RMS", "log RMS", "Abs_rel", "Sqr_rel", "a1", "a2", "a3", "bump"))
    print(("  {: 2.6f}  " * 9).format(*final_res[:-1].tolist()))