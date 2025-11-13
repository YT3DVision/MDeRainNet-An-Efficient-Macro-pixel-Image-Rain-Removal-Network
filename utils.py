import numpy as np
from skimage.metrics import peak_signal_noise_ratio


def normalize(data):
    return data / 255.


# def batch_PSNR(img, imclean, data_range):
#     Img = img.data.cpu().numpy().astype(np.float32)
#     Iclean = imclean.data.cpu().numpy().astype(np.float32)
#     PSNR = 0
#     for i in range(Img.shape[0]):
#         PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
#     return (PSNR / Img.shape[0])


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    h = imclean.shape[2] // 5
    w = imclean.shape[3] // 5
    for i in range(Img.shape[0]):
        psnr_per_batch = []
        for row in range(5):
            ibx = row * h
            iex = ibx + h
            for col in range(5):
                iby = col * w
                iey = iby + w
                value_per_SAI = peak_signal_noise_ratio(Iclean[i, :, ibx:iex, iby:iey], Img[i, :, ibx:iex, iby:iey], data_range=data_range)
                psnr_per_batch.append(value_per_SAI)

        PSNR += sum(psnr_per_batch) / len(psnr_per_batch)
    return (PSNR / Img.shape[0])


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
