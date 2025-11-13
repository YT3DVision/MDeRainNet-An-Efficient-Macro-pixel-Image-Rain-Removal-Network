import cv2
import os
import argparse
import glob
import torch
from torch.autograd import Variable
from utils import *
from DataforDemoTest import *
from model import *
import torch.backends.cudnn as cudnn
import time


def parse_args():
    parser = argparse.ArgumentParser(description="DerainNet_Test")
    parser.add_argument("--logdir", type=str, default="./log/", help='path to model and log files')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--n_groups", type=int, default=2, help="number of Inter-Groups")
    # parser.add_argument("--n_blocks", type=int, default=4, help="number of Inter-Blocks")
    parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
    # transformer
    parser.add_argument('--patch_dim', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0)
    parser.add_argument('--no_norm', action='store_true')
    parser.add_argument('--freeze_norm', action='store_true')
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_mlp', action='store_true')
    parser.add_argument('--pos_every', type=bool, default=True)
    parser.add_argument('--no_pos', action='store_true')

    parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
    # parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    # parser.add_argument("--n_groups", type=int, default=4, help="number of Inter-Groups")
    parser.add_argument('--patch_size', type=int, default=320, help='output patch size')
    parser.add_argument('--crop', type=bool, default=True, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--patchsize", type=int, default=64, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--minibatch", type=int, default=4, help="LFs are cropped into patches to save GPU memory")
    # parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
    # parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
    parser.add_argument('--input_dir', type=str, default='/data/HE/Datasets/LMY/real/test')
    parser.add_argument('--save_path', type=str, default='results/outputs/epoch66')

    return parser.parse_args()


def demo_test(cfg):
    # Build model
    print('Loading model ...\n')
    net = TransNet(args=cfg, angRes=cfg.angRes, n_groups=cfg.n_groups)
    cudnn.benchmark = True
    net.to(cfg.device)
    model_name = 'LF-Derain-TransNet_epoch_66'
    model = torch.load(cfg.logdir + model_name + '.pth.tar', map_location={'cuda:0': cfg.device})
    net.load_state_dict(model['state_dict'])
    net.eval()

    factor = 3
    time_test = 0
    count = 0
    global img_h, img_w, stride_w, stride_h, out_inter_list
    input_path = os.path.join(cfg.input_dir, 'input')

    test_scenes_num = 0
    list_Scenes = os.listdir(input_path)  # 列出对应的不同场景的文件夹   /input/
    list_Scenes.sort(key=lambda i: int(i[0:]))
    for Scenes_num in range(len(list_Scenes)):
        print('Working on scene: ' + list_Scenes[Scenes_num] + '...')
        input_dir_name = os.path.join(input_path, list_Scenes[Scenes_num])  # /input/1
        input_img_list = os.listdir(input_dir_name)  # 获得不同场景下的input images
        input_img_list.sort(key=lambda i: int(i[0:-4]))

        input_list = []
        input_patch_list = []
        if cfg.crop == False:
            for i in range(len(input_img_list)):
                input_file = "%s" % (input_img_list[i])
                input_img = cv2.imread(os.path.join(input_dir_name, input_file))
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])

                input_img = np.float32(normalize(input_img))
                img_h, img_w, c = input_img.shape
                stride_w = img_h
                stride_h = img_w
                input_list.append(input_img)

            input_SAI = input2SAI(np.array(input_list), img_w, img_h, stride_w, stride_h, cfg.angRes)

            input_SAI = Variable(torch.Tensor(input_SAI))
            input_SAI = input_SAI.unsqueeze(0)
            input_SAI = input_SAI.permute(0, 3, 1, 2)
            # print(input_SAI.shape)
            with torch.no_grad():  #
                start_time = time.time()
                out = net(input_SAI.to(cfg.device))
                # out = torch.clamp(out, 0., 1.)
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

                print(out.shape)
                print('Scenes Name: {}，Run Time:{}', list_Scenes[Scenes_num], dur_time)
        else:
            for i in range(len(input_img_list)):
                input_file = "%s" % (input_img_list[i])
                input_img = cv2.imread(os.path.join(input_dir_name, input_file))
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])

                input_img = np.float32(normalize(input_img))
                input_patches = Im2Patchfortest(input_img.transpose(2, 0, 1), win=cfg.patchsize, stride=cfg.patchsize)
                input_patch_list.append(input_patches)

            input_SAI = patches2SAIfortest(input_patch_list, cfg.patchsize, cfg.angRes)
            input_SAI = Variable(torch.Tensor(input_SAI))
            out_list = []
            assert input_SAI.shape[0] % cfg.minibatch == 0
            with torch.no_grad():
                start_time = time.time()
                for i in range(input_SAI.shape[0] // cfg.minibatch):
                    index0 = i * cfg.minibatch
                    index1 = index0 + cfg.minibatch
                    out = net(input_SAI[index0:index1, :, :, :].to(cfg.device))
                    out_list.extend(out.data.cpu().numpy())
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

            out_list = np.array(out_list)
            # print(':', out_list.shape)
            print('Scenes Name: {}，Run Time:{}。'.format(list_Scenes[Scenes_num], dur_time))

            out_inter_list = np.array(LFintegrate(out_list, cfg.angRes, cfg.patchsize))

        if not os.path.exists(cfg.save_path + '/' + list_Scenes[Scenes_num]):
            os.makedirs(cfg.save_path + '/' + list_Scenes[Scenes_num])
        for i in range(len(out_inter_list)):
            out_file = "%s" % (input_img_list[i])
            orig_img = out_inter_list[i, :, :, :]
            orig_img = np.clip(orig_img, 0., 1.)
            orig_img = 255 * orig_img
            orig_img = np.uint8(orig_img)

            b, g, r = cv2.split(orig_img)
            orig_img = cv2.merge([r, g, b])
            cv2.imwrite(os.path.join(cfg.save_path, list_Scenes[Scenes_num] + '/' + out_file), orig_img)

        test_scenes_num += 1
        count += 1
    print('Finished! \n')
    print('demo_test set, # Scenes %d\n' % test_scenes_num)
    print('Avg. time:', time_test / count)


if __name__ == '__main__':
    cfg = parse_args()
    demo_test(cfg)
