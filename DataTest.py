import os
import os.path
import random
import h5py
import torch
import cv2
import glob
import math
import torch.utils.data as udata
from utils import *
import numpy as np


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def patches2SAI(input_patch_list, target_patch_list, patch_size, angs):
    input_patch_list = np.array(input_patch_list)
    input_patch_list = np.transpose(input_patch_list,(0, 4, 2, 3, 1))  # (25,4,48,48,3)
    target_patch_list = np.array(target_patch_list)
    target_patch_list = np.transpose(target_patch_list, (0, 4, 2, 3, 1))  # (25,4,48,48,3)
    input_sai_list = []
    target_sai_list = []
    img_size_w = patch_size * angs
    img_size_h = patch_size * angs
    n = int(math.sqrt(len(input_patch_list)))  # 5
    for k in range(input_patch_list.shape[1]):
        img = np.zeros((img_size_h, img_size_w, 3))
        for ix in range(n):
            for iy in range(n):
                ind = iy + (n * ix)  # 0
                iby = int(iy * patch_size)  # 0
                iey = min(img_size_w, iby + patch_size)  # 48
                ibx = int(ix * patch_size)  # 0
                iex = min(img_size_h, ibx + patch_size)  # 48
                img[ibx:iex, iby:iey, :] = input_patch_list[ind, k, :, :, :]
        input_sai_list.append(img)

    for k in range(target_patch_list.shape[1]):
        img = np.zeros((img_size_h, img_size_w, 3))
        for ix in range(n):
            for iy in range(n):
                ind = iy + (n * ix)  # 0
                iby = int(iy * patch_size)  # 0
                iey = min(img_size_w, iby + patch_size)  # 48
                ibx = int(ix * patch_size)  # 0
                iex = min(img_size_h, ibx + patch_size)  # 48
                img[ibx:iex, iby:iey, :] = target_patch_list[ind, k, :, :, :]
        target_sai_list.append(img)
    input_sai_list = np.array(input_sai_list)
    input_sai_list = np.transpose(input_sai_list, (3, 1, 2, 0))
    target_sai_list = np.array(target_sai_list)
    target_sai_list = np.transpose(target_sai_list, (3, 1, 2, 0))
    return input_sai_list, target_sai_list


def prepare_test_data_syn(data_path, patch_size, stride, angs):
    # train
    if os.path.exists(os.path.join(data_path, 'val_target.h5')) == 0 and os.path.exists(
            os.path.join(data_path, 'val_input.h5')) == 0:
        print('process Verification syn_data')
        input_path = os.path.join(data_path, 'input')
        target_path = os.path.join(data_path, 'gt')

        save_target_path = os.path.join(data_path, 'val_target.h5')
        save_input_path = os.path.join(data_path, 'val_input.h5')

        target_h5f = h5py.File(save_target_path, 'w')
        input_h5f = h5py.File(save_input_path, 'w')

        train_num = 0
        # list_rain_Scenes = os.listdir(input_path)   # 列出不同场景的雨图文件夹
        list_Scenes = os.listdir(target_path)  # 列出对应的不同场景的文件夹   /gt/
        list_Scenes.sort(key=lambda i: int(i[0:]))
        for Scenes_num in range(len(list_Scenes)):
            input_dir_name = os.path.join(input_path, list_Scenes[Scenes_num])  # /input/1
            target_dir_name = os.path.join(target_path, list_Scenes[Scenes_num])  # /gt/1

            input_img_list = os.listdir(input_dir_name)  # 获得不同场景下的input images
            target_img_list = os.listdir(target_dir_name)  # 获得不同场景下的target images

            input_img_list.sort(key=lambda i: int(i[:-4]))
            target_img_list.sort(key=lambda i: int(i[:-4]))

            input_patch_list = []
            target_patch_list = []
            for i in range(len(input_img_list)):
                target_file = "%s" % (target_img_list[i])
                target = cv2.imread(os.path.join(target_dir_name, target_file))
                b, g, r = cv2.split(target)
                target = cv2.merge([r, g, b])

                input_file = "%s" % (input_img_list[i])
                input_img = cv2.imread(os.path.join(input_dir_name, input_file))
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])

                target_img = target
                target_img = np.float32(normalize(target_img))
                target_patches = Im2Patch(target_img.transpose(2, 0, 1), win=patch_size, stride=stride)
                target_patch_list.append(target_patches)

                input_img = np.float32(normalize(input_img))
                input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)
                input_patch_list.append(input_patches)

            input_patches_SAI, target_patches_SAI = patches2SAI(input_patch_list, target_patch_list, patch_size, angs)
            print("target dir: %s # samples: %d" % (list_Scenes[Scenes_num], target_patches_SAI.shape[3]))

            for n in range(target_patches_SAI.shape[3]):
                target_data = target_patches_SAI[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches_SAI[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)
                train_num += 1

        target_h5f.close()
        input_h5f.close()
        print('Verification set_syn, # samples %d\n' % train_num)


class Test_Dataset_syn(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Test_Dataset_syn, self).__init__()

        self.data_path = data_path

        target_path = os.path.join(self.data_path, 'val_target.h5')
        input_path = os.path.join(self.data_path, 'val_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        target_path = os.path.join(self.data_path, 'val_target.h5')
        input_path = os.path.join(self.data_path, 'val_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])

        target_h5f.close()
        input_h5f.close()

        return torch.Tensor(input), torch.Tensor(target)
