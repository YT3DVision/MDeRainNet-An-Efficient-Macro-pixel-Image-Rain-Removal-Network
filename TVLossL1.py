import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from grad_conv import grad_conv_hor, grad_conv_vet
from torch.nn.functional import l1_loss


# img must be variable with grad and of dim N*C*W*H
# def TVLossL1(img):
#     hor = grad_conv_hor()(img)
#     vet = grad_conv_vet()(img)
#     target = torch.autograd.Variable(torch.FloatTensor(img.shape).zero_().cuda())
#     loss_hor = l1_loss(hor, target, size_average=False)
#     loss_vet = l1_loss(vet, target, size_average=False)
#     loss = loss_hor + loss_vet
#     return loss

def TVLossL1(img):
    loss_total = []
    h = img.shape[2] // 5
    w = img.shape[3] // 5
    for row in range(5):
        ibx = row * h
        iex = ibx + h
        for col in range(5):
            iby = col * w
            iey = iby + w
            hor = grad_conv_hor()(img[:, :, ibx:iex, iby:iey])
            vet = grad_conv_vet()(img[:, :, ibx:iex, iby:iey])
            target = torch.autograd.Variable(torch.FloatTensor(img[:, :, ibx:iex, iby:iey].shape).zero_().cuda())
            loss_hor = l1_loss(hor, target, size_average=False)
            loss_vet = l1_loss(vet, target, size_average=False)
            loss = loss_hor + loss_vet
            loss_total.append(loss)
    return sum(loss_total) / len(loss_total)


if __name__ == "__main__":
    img = Image.open('E:/LMY_ComparedData/LMY_TEST/test/input/224/041.png')
    img = transforms.ToTensor()(img)[None, :, :, :]
    img = torch.autograd.Variable(img, requires_grad=True).cuda()
    loss = TVLossL1(img)
    print('loss: ', loss)
