from torch import nn
from SSIM import SSIM


class criterion2(nn.Module):
    def __init__(self):
        super(criterion2, self).__init__()
        self.ssim = SSIM()

    def forward(self, pred_im, gt):
        loss_total = []
        h = gt.shape[2] // 5
        w = gt.shape[3] // 5
        for row in range(5):
            ibx = row * h
            iex = ibx + h
            for col in range(5):
                iby = col * w
                iey = iby + w
                value = self.ssim(pred_im[:, :, ibx:iex, iby:iey], gt[:, :, ibx:iex, iby:iey])
                loss_total.append(value)
        ssim_value = sum(loss_total) / len(loss_total)
        ssim_loss = (1 - ssim_value)
        return ssim_loss
