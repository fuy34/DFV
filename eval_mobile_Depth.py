import argparse
import cv2
from models import DFFNet
import os
import time
from models.submodule import *
import  matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from glob import glob

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

'''
Code for Ours-FV and Ours-DFV test on Mobile depth dataset  
'''

parser = argparse.ArgumentParser(description='DFVDFF')
parser.add_argument('--data_path', default='/data/DFF/MobileDepth/',help='test data path')
parser.add_argument('--outdir', default='./mobileDepth/',help='output save path')
parser.add_argument('--loadmodel', default='/data/large_download/DFF-DFV.tar', help='model path')

parser.add_argument('--stack_num', type=int ,default=5, help='num of image in a stack, please take a number in [2, 10], change it according to the loaded checkpoint!')
parser.add_argument('--use_diff', default=1, choices=[0, 1], help='if use differential images as input, change it according to the loaded checkpoint!')

parser.add_argument('--level', type=int, default=4, help='num of layers in network, please take a number in [1, 4]')
args = parser.parse_args()
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

if not os.path.isdir(args.outdir):
    os.makedirs(args.outdir)


# !!! Only for users who download our pre-trained checkpoint, comment the next four line if you are not !!!
if os.path.basename(args.loadmodel) == 'DFF-DFV.tar' :
    args.use_diff = 1
elif os.path.basename(args.loadmodel) == 'DFF-FV.tar' :
    args.use_diff = 0


# construct model
model = DFFNet(clean=False,level=args.level, use_diff=args.use_diff)
model = nn.DataParallel(model)
model.to(device)
ckpt_name = os.path.basename(os.path.dirname(args.loadmodel))

if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('run with random init')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

base_width = 32

def main():
    model.eval()

    for dir in os.listdir(args.data_path):
        # img load
        img_lst = glob(args.data_path + '/' + dir + '/a*.jpg')
        img_lst.sort(key=lambda x: int(os.path.basename(x)[1:-4]))

        mats_input = []
        img_mean = np.array([0.485, 0.456, 0.406]).reshape([1,1,3])
        img_std = np.array([[0.229, 0.224, 0.225]]).reshape([1,1,3])

        for pth in img_lst:
            im = cv2.imread(pth) / 255
            h, w, _  = im.shape
            zoomx, zoomy = 224/ h, 224/ w
            zoom = max(zoomx, zoomy)
            rsz_img = cv2.resize(im, dsize=None, fx=zoom, fy=zoom)

            max_h = int(rsz_img.shape[0] // base_width * base_width)
            max_w = int(rsz_img.shape[1] // base_width * base_width)
            if max_h < rsz_img.shape[0]: max_h += base_width
            if max_w < rsz_img.shape[1]: max_w += base_width

            top_pad = max_h - rsz_img.shape[0]
            left_pad = max_w - rsz_img.shape[1]
            mat_all = np.lib.pad(rsz_img, ( (top_pad, 0), (0, left_pad), (0, 0)), mode='constant', constant_values=0)

            mat_all = (mat_all - img_mean) / img_std

            mats_input.append(mat_all)

        mats_input = np.stack(mats_input)
        img_num, h, w, _ = mats_input.shape

        # focus load
        focus_dist_np = np.genfromtxt(args.data_path + '/' + dir + '/focus_dpth.txt')

        if dir == 'metals':  # metals' distance esitimation is oppsoite, as we only care about relative dist, directly take minus
            focus_dist_np = - focus_dist_np

        # sort image and dist
        img_dist = [(i, focus_dist_np[x]) for i, x in enumerate(range(img_num))]
        # already sorted in the s
        sort_img_dist = sorted(img_dist, key=lambda x: x[1])
        img_stack = torch.from_numpy( np.stack([mats_input[x[0]] for x in sort_img_dist])).float().permute([0, 3, 1, 2]).to(device)

        # Some focal distance estimation has extremely large scale difference, up to ~10^6, and
        # In the window scene, we observe a fairly large focal distance range from frames, but the estimated focal distance does not change much.
        # For these scenes,  we found use a linear distance distrbution w.r.t. the relative order leads to better visual results.
        # All baselines method presented in the paper use the same in this dataset, except mobileDFF and AiFDepthNet
        # MobileDFF use the author provided results, AiFDepthNet we use their code without any modification
        if sort_img_dist[0][1] * 10 < sort_img_dist[-1][1] or dir == 'window':
            focus_dist = torch.linspace(0, 1., img_num)
        else:
            focus_dist = torch.from_numpy( np.stack([x[1] - sort_img_dist[0][1] for x in sort_img_dist])).float() # torch.linspace(0, 1., img_num) #
            focus_dist = focus_dist / (sort_img_dist[-1][1] - sort_img_dist[0][1])

        # select image evenly
        if img_num > args.stack_num:
            idx = np.linspace(0, img_num-1, args.stack_num).round().astype(np.int)
            focus_dist = focus_dist[idx]
            img_stack = img_stack[idx]


        img_stack = img_stack.unsqueeze(0)
        focus_dist = focus_dist.unsqueeze(0)
        print(focus_dist.shape, img_stack.shape)

        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            pred_dpth, std, focusMap = model(img_stack, focus_dist)
            torch.cuda.synchronize()
            ttime = (time.time() - start_time); print('time = %.2f' % (ttime*1000) )

        pred_dpth = pred_dpth.squeeze().cpu().numpy()[top_pad:, :-left_pad]

        std = std.squeeze().cpu().numpy()[top_pad:, :-left_pad]

        # pred viz
        img_save_pth = os.path.abspath(args.outdir) + '/mobile_depth_diff{}_std_correct/'.format(args.use_diff)
        if not os.path.isdir(img_save_pth):
            os.makedirs(img_save_pth)
        MAX_DISP, MIN_DISP = pred_dpth.max(), pred_dpth.min()
        plt.figure()
        plt.imshow(pred_dpth, vmax=MAX_DISP, vmin=MIN_DISP)
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig('{}/{}_pred_viz_diff{}.png'.format(img_save_pth, dir, args.use_diff), bbox_inches='tight', pad_inches=0)
        plt.close()

        for i in range(args.stack_num):
            MAX_DISP, MIN_DISP = 1, 0
            plt.imshow(focusMap[i][top_pad:, :-left_pad].squeeze().detach().cpu().numpy(), vmax=MAX_DISP, vmin=MIN_DISP,
                       cmap='jet')
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig('{}/{}_{}_prob_dist.png'.format(img_save_pth, dir, i), bbox_inches='tight', pad_inches=0)


        cv2.imwrite('{}/{}_img.png'.format(img_save_pth, dir), im*255)



if __name__ == '__main__':
    main()

