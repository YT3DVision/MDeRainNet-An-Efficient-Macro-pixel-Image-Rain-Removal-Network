import math
import numpy as np


def input2SAI(data, size_w, size_h, stride_w, stride_h, angs):
    img_size_w = size_w * angs
    img_size_h = size_h * angs
    n = int(math.sqrt(len(data)))
    img = np.zeros((img_size_h, img_size_w, 3))
    for ix in range(n):
        for iy in range(n):
            ind = iy + (n * ix)
            iby = int(iy * stride_w)
            iey = min(img_size_w, iby + stride_w)
            ibx = int(ix * stride_h)
            iex = min(img_size_h, ibx + stride_h)
            img[ibx:iex, iby:iey, :] = data[ind, :, :, :]
    return img


def Im2Patchfortest(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    assert img.shape[1] % win == 0 and img.shape[2] % win == 0
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def patches2SAIfortest(input_patch_list, patch_size, angs):
    input_patch_list = np.array(input_patch_list)
    input_patch_list = np.transpose(input_patch_list, (0, 4, 2, 3, 1))  # (25,4,48,48,3)
    input_sai_list = []
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
    input_sai_list = np.array(input_sai_list)
    input_sai_list = np.transpose(input_sai_list, (0, 3, 1, 2))
    return input_sai_list


def Merge(img_patch_list, n, patch_size):
    img_size_w = patch_size * n
    img_size_h = patch_size * n
    img = np.zeros((3, img_size_h, img_size_w))
    for ix in range(n):
        for iy in range(n):
            ind = iy + (n * ix)
            ibx = int(ix * patch_size)
            iex = min(img_size_h, ibx + patch_size)
            iby = int(iy * patch_size)
            iey = min(img_size_w, iby + patch_size)
            img[:, ibx:iex, iby:iey] = img_patch_list[ind, :, :, :]
    return img


def LFintegrate(data, img_angs, patch_size):
    n = int(math.sqrt(len(data)))
    inter_list = []
    for row in range(img_angs):
        ibx = row * patch_size
        iex = ibx + patch_size
        for col in range(img_angs):
            iby = col * patch_size
            iey = iby + patch_size
            img_patch_list = []
            for patch in range(len(data)):
                img_patch_list.append(data[patch, :, ibx:iex, iby:iey])
            orig_img = Merge(np.array(img_patch_list), n, patch_size)
            orig_img = np.float32(orig_img)
            orig_img = orig_img.transpose(1, 2, 0)
            inter_list.append(orig_img)
    return inter_list
