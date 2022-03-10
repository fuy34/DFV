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
Code for Ours-FV and Ours-DFV evaluation on DDFF-12 test set submission  
'''

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='DFVDFF')
parser.add_argument('--data_path', default='/data/DFF/ddff-dataset-test.h5',help='test data path')
parser.add_argument('--loadmodel', default=None, help='model path')
parser.add_argument('--outdir', default=None,help='output dir')

parser.add_argument('--max_disp', type=float ,default=0.28, help='maxium disparity')
parser.add_argument('--min_disp', type=float ,default=0.02, help='minium disparity')

parser.add_argument('--stack_num', type=int ,default=5, help='num of image in a stack, please take a number in [2, 10], change it according to the loaded checkpoint!')
parser.add_argument('--use_diff', default=1, choices=[0, 1], help='if use differential images as input, change it according to the loaded checkpoint!')

parser.add_argument('--level', type=int, default=1,
                    help='output level of output, default is level 1 (stage 3),\
                          can also use level 2 (stage 2) or level 3 (stage 1)')
args = parser.parse_args()

scene_name = {0: 'lockeroom', 1:'cafeteria', 2:'library', 3:'spencerlab', 4:'office44', 5:'magistrale' }
# dataloader
from dataloader import DDFF12Loader

# construct model
model = DFFNet( clean=False,level=args.level, use_diff=args.use_diff)
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


def main(image_size = (383, 552)):
    model.eval()

    # Calculate pad size for images
    test_pad_size = (np.ceil((image_size[0] / 32)) * 32, np.ceil((image_size[1] / 32)) * 32)  # 32=2**numPoolings(=5)
    # Create test set transforms
    transform_test = [DDFF12Loader.ToTensor(),
                      DDFF12Loader.ClipGroundTruth(0.0202, 0.2825),
                      DDFF12Loader.PadSamples(test_pad_size),
                      DDFF12Loader.Normalize(mean_input=[0.485, 0.456, 0.406],std_input=[0.229, 0.224, 0.225])]
    transform_test = torchvision.transforms.Compose(transform_test)

    test_set = DDFF12Loader(args.data_path, stack_key="stack_test", disp_key="disp_test", transform=transform_test,
                            n_stack=args.stack_num,
                            min_disp=args.min_disp, max_disp=args.max_disp
                            )
    dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)


    # metric prepare
    test_num = len(dataloader)
    time_list = []
    for inx, (img_stack, disp, foc_dist) in enumerate(dataloader):

        scene_id = inx // 20
        img_id = inx % 20 + 1
        if img_id == 1:
            print('processing: {}: {}'.format(scene_id, inx))


        img_stack = Variable(torch.FloatTensor(img_stack)).cuda()


        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            pred_disp, std, _ = model(img_stack, foc_dist)
            torch.cuda.synchronize()
            ttime = (time.time() - start_time); print('time = %.2f' % (ttime*1000) )

        pred_disp = pred_disp.squeeze().cpu().numpy()[:image_size[0], :image_size[1]]

        img_save_pth = os.path.join(args.outdir, ckpt_name, scene_name[scene_id]) #'figure_paper'#
        if not os.path.isdir(img_save_pth + '_viz'):
            os.makedirs(img_save_pth + '_viz')
        img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=img_stack.dtype).view(3, 1, 1)
        img_std = torch.tensor([0.229, 0.224, 0.225], dtype=img_stack.dtype).view(3, 1, 1)

        # # save the original image -- no need for submission
        # cv2.imwrite('{}_viz/{}_img.png'.format(img_save_pth, img_id),
        #             ((img_stack[0,0, :, :image_size[0], :image_size[1]].detach().cpu() * img_std + img_mean)*255).numpy().transpose(1,2,0))
        #
        # # pred viz -- no need for submission
        # MAX_DISP, MIN_DISP = 0.28, 0.02
        # plt.figure()
        # plt.imshow(pred_disp, vmax=MAX_DISP, vmin=MIN_DISP)  # val2uint8(, MAX_DISP, MIN_DISP)
        # plt.axis('off')
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.savefig('{}_viz/{}_pred_viz.png'.format(img_save_pth, img_id), bbox_inches='tight', pad_inches=0)
        # plt.close()
        #
        # # std viz -- no need for submission
        # plt.imshow(std.squeeze().detach().cpu().numpy(), vmax=0.1, vmin=0)  # val2uint8(, MAX_DISP, MIN_DISP)
        # plt.axis('off')
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.savefig('{}_viz/{}_std_viz.png'.format(img_save_pth, img_id,  args.level), bbox_inches='tight', pad_inches=0)

        if not os.path.isdir(img_save_pth ):
            os.makedirs(img_save_pth )

        # npy for eval submission
        np.save('{}/DISP_{:04d}.npy'.format(img_save_pth, img_id), pred_disp)

        # time
        time_list.append('{}/DISP_{:04d} {}\n'.format(scene_name[scene_id], img_id, ttime))

    with open('{}/{}/runtime.txt'.format(args.outdir, ckpt_name), 'w') as f:
        for line in time_list:
            f.write(line)


if __name__ == '__main__':
    main()

