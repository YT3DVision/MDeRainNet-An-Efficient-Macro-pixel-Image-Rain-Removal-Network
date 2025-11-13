import torch.nn.functional as F
from torch import nn


class criterion1(nn.Module):
    def __init__(self):
        super(criterion1, self).__init__()
        self.L1 = F.smooth_l1_loss

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
                value = self.L1(pred_im[:, :, ibx:iex, iby:iey], gt[:, :, ibx:iex, iby:iey])
                loss_total.append(value)
        return sum(loss_total) / len(loss_total)
