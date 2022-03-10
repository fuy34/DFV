from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import time
from models import DFFNet
from utils import logger, write_log
torch.backends.cudnn.benchmark=True
from glob import glob

'''
Main code for Ours-FV and Ours-DFV training 
'''


parser = argparse.ArgumentParser(description='DFVDFF')
# === dataset =====
parser.add_argument('--dataset', default=['FoD500','DDFF12'], nargs='+',  help='data Name')
parser.add_argument('--DDFF12_pth', default=None, help='DDFF12 data path')
parser.add_argument('--FoD_pth', default=None, help='FOD data path')
parser.add_argument('--FoD_scale', default=0.2,
                    help='FoD dataset gt scale for loss balance, because FoD_GT: 0.1-1.5, DDFF12_GT 0.02-0.28, '
                         'empirically we find this scale help improve the model performance for our method and DDFF')
# ==== hypo-param =========
parser.add_argument('--stack_num', type=int ,default=5, help='num of image in a stack, please take a number in [2, 10]')
parser.add_argument('--level', type=int ,default=4, help='num of layers in network, please take a number in [1, 4]')
parser.add_argument('--use_diff', default=1, type=int, choices=[0,1], help='if use differential feat, 0: None,  1: diff cost volume')
parser.add_argument('--lvl_w', nargs='+', default=[8./15, 4./15, 2./15, 1./15],  help='for std weight')

parser.add_argument('--lr', type=float, default=0.0001,  help='learning rate')
parser.add_argument('--epochs', type=int, default=700, help='number of epochs to train')
parser.add_argument('--batchsize', type=int, default=20, help='samples per batch')

# ====== log path ==========
parser.add_argument('--loadmodel', default=None,   help='path to pre-trained checkpoint if any')
parser.add_argument('--savemodel', default=None, help='save path')
parser.add_argument('--seed', type=int, default=2021, metavar='S',  help='random seed (default: 2021)')

args = parser.parse_args()
args.logname = '_'.join(args.dataset)


# ============ init ===============
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

start_epoch = 1
best_loss = 1e5
total_iter = 0

model = DFFNet(clean=False,level=args.level, use_diff=args.use_diff)
model = nn.DataParallel(model)
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# ========= load model if any ================
if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()  } #if ('disp' not in k)
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)

    if 'epoch' in pretrained_dict:
        start_epoch = pretrained_dict['epoch']

    if 'iters' in pretrained_dict:
        total_iter = pretrained_dict['iters']

    if 'best' in pretrained_dict:
        best_loss = pretrained_dict['best']

    if 'optimize' in pretrained_dict:
        optimizer.load_state_dict(pretrained_dict['optimize'])

    print('load model from {}, start epoch {}, best_loss {}'.format(args.loadmodel, start_epoch, best_loss))

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# ============ data loader ==============
#Create data loader
if  'DDFF12' in args.dataset:
    from dataloader import DDFF12Loader
    database = '/data/DFF/my_ddff_trainVal.h5' if args.DDFF12_pth is None else  args.DDFF12_pth
    DDFF12_train = DDFF12Loader(database, stack_key="stack_train", disp_key="disp_train", n_stack=args.stack_num,
                                 min_disp=0.02, max_disp=0.28)
    DDFF12_val = DDFF12Loader(database, stack_key="stack_val", disp_key="disp_val", n_stack=args.stack_num,
                                      min_disp=0.02, max_disp=0.28, b_test=False)
    DDFF12_train, DDFF12_val = [DDFF12_train], [DDFF12_val]
else:
    DDFF12_train, DDFF12_val = [], []

if 'FoD500' in args.dataset:
    from dataloader import FoD500Loader
    database = '/data/DFF/baseline/defocus-net/data/fs_6/' if args.FoD_pth is None else  args.FoD_pth
    FoD500_train, FoD500_val = FoD500Loader(database, n_stack=args.stack_num, scale=args.FoD_scale)
    FoD500_train, FoD500_val =  [FoD500_train], [FoD500_val]
else:
    FoD500_train, FoD500_val = [], []

dataset_train = torch.utils.data.ConcatDataset(DDFF12_train  + FoD500_train )
dataset_val = torch.utils.data.ConcatDataset(DDFF12_val) # we use the model perform better on  DDFF12_val

TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=4, batch_size=args.batchsize, shuffle=True, drop_last=True)
ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val, num_workers=1, batch_size=12, shuffle=False, drop_last=True)

print('%d batches per epoch'%(len(TrainImgLoader)))


# =========== Train func. =========
def train(img_stack_in, disp, foc_dist):
    model.train()
    img_stack_in   = Variable(torch.FloatTensor(img_stack_in))
    gt_disp    = Variable(torch.FloatTensor(disp))
    img_stack, gt_disp, foc_dist = img_stack_in.cuda(),  gt_disp.cuda(), foc_dist.cuda()

    #---------
    max_val = torch.where(foc_dist>=100, torch.zeros_like(foc_dist), foc_dist) # exclude padding value
    min_val = torch.where(foc_dist<=0, torch.ones_like(foc_dist)*10, foc_dist)  # exclude padding value
    mask = (gt_disp >= min_val.min(dim=1)[0].view(-1,1,1,1)) & (gt_disp <= max_val.max(dim=1)[0].view(-1,1,1,1)) #
    mask.detach_()
    #----

    optimizer.zero_grad()
    beta_scale = 1 # smooth l1 do not have beta in 1.6, so we increase the input to and then scale back -- no significant improve according to our trials
    stacked, stds, _ = model(img_stack, foc_dist)


    loss = 0
    for i, (pred, std) in enumerate(zip(stacked, stds)):
        _cur_loss = F.smooth_l1_loss(pred[mask] * beta_scale, gt_disp[mask]* beta_scale, reduction='none') / beta_scale
        loss = loss + args.lvl_w[i] * _cur_loss.mean()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    vis = {}
    vis['pred'] = stacked[0].detach().cpu()
    vis['mask'] = mask.type(torch.float).detach().cpu()
    lossvalue = loss.data

    del stacked
    del loss
    return lossvalue,vis


def valid(img_stack_in,disp, foc_dist):
    model.eval()
    img_stack = Variable(torch.FloatTensor(img_stack_in))
    gt_disp = Variable(torch.FloatTensor(disp))
    img_stack, gt_disp, foc_dist = img_stack.cuda() , gt_disp.cuda(), foc_dist.cuda()

    #---------
    mask = gt_disp > 0
    mask.detach_()
    #----
    with torch.no_grad():
        pred_disp, _, _ = model(img_stack, foc_dist)
        loss = (F.mse_loss(pred_disp[mask] , gt_disp[mask] , reduction='mean')) # use MSE loss for val

    vis = {}
    vis['mask'] = mask.type(torch.float).detach().cpu()
    vis["pred"] = pred_disp.detach().cpu()

    return loss, vis



def adjust_learning_rate(optimizer, epoch):
    # turn out we do not need adjust lr, the results is already good enough
    if epoch <= args.epochs:
        lr = args.lr
    else:
        lr = args.lr * 0.1 #1e-5  will not used in this project
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def main():
    global  start_epoch, best_loss, total_iter
    saveName = args.logname + "_scale{}_nsck{}_lr{}_ep{}_b{}_lvl{}".format(args.FoD_scale,args.stack_num,
                                                                         args.lr, args.epochs, args.batchsize, args.level)
    if args.use_diff > 0:
        saveName = saveName + '_diffFeat{}'.format(args.use_diff)


    train_log = logger.Logger( os.path.abspath(args.savemodel), name=saveName + '/train')
    val_log = logger.Logger( os.path.abspath(args.savemodel), name=saveName + '/val')

    total_iters = total_iter

    for epoch in range(start_epoch, args.epochs+1):
        total_train_loss = 0
        lr_ = adjust_learning_rate(optimizer,epoch)
        train_log.scalar_summary('lr_epoch', lr_, epoch)

        ## training ##
        for batch_idx, (img_stack, gt_disp, foc_dist) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss, vis = train(img_stack, gt_disp, foc_dist)

            if total_iters %10 == 0:
                torch.cuda.synchronize()
                print('epoch %d:  %d/ %d train_loss = %.6f , time = %.2f' % (epoch, batch_idx, len(TrainImgLoader), loss, time.time() - start_time))
                train_log.scalar_summary('loss_batch',loss, total_iters)

            total_train_loss += loss
            total_iters += 1

        # record the last batch
        write_log(vis, img_stack[:, 0], img_stack[:, -1], gt_disp, train_log, epoch, thres=0.05)
        train_log.scalar_summary('avg_loss', total_train_loss / len(TrainImgLoader), epoch)

        # save model
        torch.save({
            'epoch': epoch + 1,
            'iters': total_iters + 1,
            'best': best_loss,
            'state_dict': model.state_dict(),
            'optimize':optimizer.state_dict(),
        },  os.path.abspath(args.savemodel) + '/' + saveName +'/model_{}.tar'.format(epoch))

        # save top 5 ckpts only
        list_ckpt = glob(os.path.join( os.path.abspath(args.savemodel) + '/' + saveName, 'model_*'))
        list_ckpt.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        if len(list_ckpt) > 5:
            os.remove(list_ckpt[0])

        # Vaild
        if epoch % 5 == 0:
            total_val_loss = 0
            for batch_idx, (img_stack, gt_disp, foc_dist) in enumerate(ValImgLoader):
                with torch.no_grad():
                    start_time = time.time()
                    val_loss, viz = valid(img_stack, gt_disp, foc_dist)

                if batch_idx %10 == 0:
                    torch.cuda.synchronize()
                    print('[val] epoch %d : %d/%d val_loss = %.6f , time = %.2f' % (epoch, batch_idx, len(ValImgLoader), val_loss, time.time() - start_time))
                total_val_loss += val_loss


            avg_val_loss = total_val_loss / len(ValImgLoader)
            err_thres = 0.05 # for validation purpose
            write_log(viz, img_stack[:, 0], img_stack[:, -1], gt_disp, val_log, epoch, thres=err_thres)
            val_log.scalar_summary('avg_loss', avg_val_loss, epoch)

            # save best
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'iters': total_iters + 1,
                    'best': best_loss,
                    'state_dict': model.state_dict(),
                    'optimize': optimizer.state_dict(),
                },  os.path.abspath(args.savemodel) + '/' + saveName + '/best.tar')


        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
