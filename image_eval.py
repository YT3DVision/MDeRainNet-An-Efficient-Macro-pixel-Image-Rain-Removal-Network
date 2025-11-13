import argparse
import math
import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# 将RGB通道图片转换为yCbCr
def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


#########################calc_metrics#############################
def calc_metrics(img1, img2, crop_border, test_Y=True):
    #
    img1 = img1 / 255.
    img2 = img2 / 255.

    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2

    if im1_in.ndim == 3:
        cropped_im1 = im1_in[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_im2 = im2_in[crop_border:-crop_border, crop_border:-crop_border, :]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:-crop_border, crop_border:-crop_border]
        cropped_im2 = im2_in[crop_border:-crop_border, crop_border:-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    psnr = calc_psnr(cropped_im1 * 255, cropped_im2 * 255)
    ssim = calc_ssim(cropped_im1 * 255, cropped_im2 * 255)
    return psnr, ssim


def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


parser = argparse.ArgumentParser(description='image_eval')
parser.add_argument('--target_path', help='path to clean(target) image dataset',
                    default='E:/DeRain/1_Light field/LMY/syn/demo_test/gt')
parser.add_argument('--de_rained_path', help='path to de_rain(predict) image dataset',
                    default='results/outputs')
parser.add_argument('--image_format', help='format of the image', default='png')
opt = parser.parse_args()

list_Scenes = os.listdir(opt.de_rained_path)  # 列出对应的不同场景的文件夹   /outputs/
list_Scenes.sort(key=lambda i: int(i[0:]))

Scenes_Names = []
psnr_record = []
ssim_record = []

print('PSNR and SSIM are being calculated.........')
for Scenes_num in range(len(list_Scenes)):
    input_dir_name = os.path.join(opt.de_rained_path, list_Scenes[Scenes_num])  # /outputs/3
    target_dir_name = os.path.join(opt.target_path, list_Scenes[Scenes_num])  # /gt/3

    input_img_list = os.listdir(input_dir_name)  # 获得不同场景下的input images
    target_img_list = os.listdir(target_dir_name)  # 获得不同场景下的target images

    input_img_list.sort(key=lambda i: int(i[:-4]))
    target_img_list.sort(key=lambda i: int(i[:-4]))
    psnr_Scenes = []
    ssim_Scenes = []
    for idx in range(len(input_img_list)):
        img_target = cv2.imread(os.path.join(input_dir_name, input_img_list[idx]))
        img_de_rain = cv2.imread(os.path.join(target_dir_name, target_img_list[idx]))
        PSNR = calc_psnr(img_target, img_de_rain)
        SSIM = calc_ssim(img_target, img_de_rain)

        psnr_Scenes.append(PSNR)
        ssim_Scenes.append(SSIM)

    Scenes_Names.append(list_Scenes[Scenes_num])
    psnr_record.append(np.mean(psnr_Scenes))
    ssim_record.append(np.mean(ssim_Scenes))



if __name__ == "__main__":
    dit = {'Scenes_name': Scenes_Names, 'PSNR': psnr_record, 'SSIM': ssim_record}
    df = pd.DataFrame(dit)
    # df.to_csv(r'./result.csv',columns=['image_number','psnr','ssim'],index=False,sep=',')
    print(df)
    avg_psnr = df['PSNR'].mean()
    avg_ssim = df['SSIM'].mean()
    print('\n----------------PSNR and SSIM average in all scenes----------------\n')
    print('PSNR:', avg_psnr)
    print('SSIM:', avg_ssim)
