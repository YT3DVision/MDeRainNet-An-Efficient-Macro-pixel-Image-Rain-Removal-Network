# --- Imports --- #
import torch
import torch.nn.functional as F


# --- Perceptual loss network  --- #
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

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
                loss_per_SAI = []
                pred_im_features = self.output_features(pred_im[:, :, ibx:iex, iby:iey])
                gt_features = self.output_features(gt[:, :, ibx:iex, iby:iey])
                for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
                    loss_per_SAI.append(F.mse_loss(pred_im_feature, gt_feature))
                loss_total.append(sum(loss_per_SAI) / len(loss_per_SAI))

        return sum(loss_total) / len(loss_total)
