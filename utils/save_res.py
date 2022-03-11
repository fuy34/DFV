import shutil
import torch
import sys

def write_log(viz, img1, img2, disp, logger, step, num_log_img=4, b_train=True, thres=1e-4):

    n_log = min(num_log_img, img1.shape[0])

    mean_values = torch.tensor([0.485, 0.456, 0.406], dtype=img1.dtype).view(3, 1, 1)
    std_values = torch.tensor([0.229, 0.224, 0.225], dtype=img1.dtype).view(3, 1, 1)

    logger.image_summary('img_first', (img1[0:n_log] * std_values + mean_values).clamp(0,1), step)
    logger.image_summary('img_last', (img2[0:n_log] * std_values+ mean_values).clamp(0,1), step)
    logger.image_summary('gt_disp', disp[0:n_log], step)
    logger.image_summary('pred_disp', viz['pred'][0:n_log], step)

    err = (torch.abs( disp[0:n_log] - viz['pred'][0:n_log]) * viz['mask'][0:n_log].type(torch.float)).cpu()
    logger.image_summary('disp_err', val2uint8(err, thres), step)


def save_ckpt(state, save_path, epoch, b_best):
    torch.save(state, save_path + '/epoch_%d.tar'%epoch)
    if b_best:
        shutil.copyfile(save_path + '/epoch_%d.tar'%epoch, save_path + '/best_model.tar')


def val2uint8(mat,maxVal, minVal=0):
    maxVal_mat = torch.ones(mat.shape) * maxVal
    minVal_mat = torch.ones(mat.shape) * minVal

    mat_vis = torch.where(mat > maxVal, maxVal_mat, mat)
    mat_vis = torch.where(mat < minVal, minVal_mat, mat_vis)
    ret =  (mat_vis * 255. / maxVal).type(torch.uint8)
    return ret



'''
Save a Numpy array to a PFM file.
'''
def save_pfm(file, image, scale = 1):
  color = None

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write('%f\n' % scale)

  image.tofile(file)
