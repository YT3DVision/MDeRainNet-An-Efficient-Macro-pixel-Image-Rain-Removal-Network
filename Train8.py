import os
import torch
import argparse
from model import TransNet
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
from DataTrain import *
from DataTest import *
from Loss_Function import criterion1
from SSIM_LOSS import criterion2
from torchvision.models import vgg16
from perceptual import LossNetwork
from TVLossL1 import *
from CR import ContrastLoss
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


# Training settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_syn", type=bool, default=False, help='run prepare_data or not')
    parser.add_argument("--preprocess_real", type=bool, default=False, help='run prepare_data or not')
    parser.add_argument('--is_semi', type=bool, default=True)
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda:0')
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
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--n_groups", type=int, default=2, help="number of Inter-Groups")
    parser.add_argument('--patch_size', type=int, default=320, help='output patch size')
    parser.add_argument('--is_ab', type=bool, default=False)
    # Model specifications
    parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
    parser.add_argument('-train_batch_size', help='Set the training batch size', default=6, type=int)
    parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=6, type=int)
    parser.add_argument('-lambda_p', help='Set the lambda in loss function', default=0.04, type=float)
    parser.add_argument('-lambda_ssim', help='Set the lambda in loss function', default=1, type=float)
    parser.add_argument('-lambda_cr_syn', help='Set the lambda in loss function', default=0.1, type=float)
    parser.add_argument('-lambda_kl', help='Set the lambda in loss function', default=10, type=float)
    parser.add_argument('-lambda_tv', help='Set the lambda in loss function', default=0.000001, type=float)
    parser.add_argument('-lambda_cr_real', help='Set the lambda in loss function', default=0.1, type=float)
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=270, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=45, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
    parser.add_argument("--DataPreprocess_patchsize", type=int, default=64,
                        help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--stride", type=int, default=32, help="LFs are cropped into patches to save GPU memory")  # 70

    parser.add_argument('--trainset_dir_syn', type=str, default='/data/HE/Datasets/LMY/syn/train')
    parser.add_argument('--trainset_dir_real', type=str, default='/data/HE/Datasets/LMY/real/train')
    parser.add_argument('--testset_dir_syn', type=str, default='/data/HE/Datasets/LMY/syn/test')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--model_name', type=str, default='LF-Derain-TransNet')
    parser.add_argument('--load_pretrain', type=bool, default=True)
    parser.add_argument('--model_path', type=str, default='./log/net_lasted.pth.tar')
    parser.add_argument('-seed', help='set random seed', default=19, type=int)

    return parser.parse_args()


def train_syn(train_loader, val_loader, cfg):
    # set seed
    seed = cfg.seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        print('Seed:\t{}'.format(seed))

    print('--- Hyper-parameters for training ---')
    print(
        'learning_rate: {}\nangular_resolution: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nlambda_ssim: {}\nlambda_cr_syn: {}\nlambda_kl: {}\nlambda_tv: {}\nlambda_cr_real: {}'.format(
            cfg.lr, cfg.angRes, cfg.DataPreprocess_patchsize, cfg.train_batch_size, cfg.val_batch_size, cfg.lambda_ssim,
            cfg.lambda_cr_syn, cfg.lambda_kl, cfg.lambda_tv, cfg.lambda_cr_real))

    net = TransNet(args=cfg, angRes=cfg.angRes, n_groups=cfg.n_groups)
    net.to(cfg.device)
    # net.apply(weights_init_xavier)
    cudnn.benchmark = True

    initial_epoch = 0
    total_train_time = 0  # 总训练时间

    if cfg.load_pretrain:
        # load the lastest model
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
            initial_epoch = model["epoch"]
            print("继续训练！\n")
            print('resuming by loading epoch %d' % initial_epoch)
            net.load_state_dict(model['state_dict'])
        else:
            print("=> no model found at '{}'\n".format(cfg.model_path))
            print("从头训练！\n")

    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    # criterion_Loss = torch.nn.L1Loss().to(cfg.device)

    # --- Define the smooth L1-loss --- #
    criterion_smooth_l1 = criterion1().to(cfg.device)

    # --- Define the perceptual loss network --- #
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(cfg.device)
    # vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_network = LossNetwork(vgg_model)
    loss_network.eval()

    # --- Define the CR-loss --- #
    criterion_CR = ContrastLoss(ablation=cfg.is_ab)

    # --- Define the ssim-loss --- #
    criterion_ssim = criterion2().to(cfg.device)

    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    scheduler._step_count = initial_epoch
    psnr_epoch = []
    loss_epoch = []
    loss_list = []
    psnr_list = []

    # record training
    writer = SummaryWriter('RECORD')
    # 开始训练
    step = 0  # 记录训练次数
    for idx_epoch in range(initial_epoch, cfg.n_epochs):
        net.train()
        time_start = time.time()
        for idx_iter, (data, label) in enumerate(train_loader):
            data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)
            out, _ = net(data)
            out_train = out
            # --- supervised loss --- #
            # loss = criterion_Loss(out, label)
            smooth_loss = criterion_smooth_l1(out, label)
            # perceptual_loss = loss_network(out, label)
            CR_loss = criterion_CR(out, label, data)
            ssim_loss = criterion_ssim(out, label)
            loss = smooth_loss + cfg.lambda_cr_syn * CR_loss + cfg.lambda_ssim * ssim_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch.append(loss.data.cpu())
            # psnr_epoch.append(batch_PSNR(out.data.cpu(), label.data.cpu()))

            # training curve
            net.eval()
            # out_train = net(data)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, label, 1.)
            psnr_epoch.append(psnr_train)
            print("已完成第{}轮训练的{:.2f}%，目前损失值为{:.6f}, PSNR: {:.4f}。".format(idx_epoch + 1,
                                                                         (idx_iter + 1) / len(train_loader) * 100,
                                                                         loss.item(), psnr_train))
            if step % 100 == 0:
                # Log the scalar values
                writer.add_scalar('Loss of every batch on training set', loss.item(), step)
                writer.add_scalar('PSNR of every batch on training set', psnr_train, step)
            step += 1

        # save model
        if idx_epoch % 1 == 0:
            writer.add_scalar('Loss of every epoch on training set', float(np.array(loss_epoch).mean()), idx_epoch + 1)
            writer.add_scalar('PSNR of every epoch on training set', float(np.array(psnr_epoch).mean()), idx_epoch + 1)
            # for name, param in net.named_parameters():
            # writer.add_histogram('epoch_' + name + '_param', param, idx_epoch + 1)  # 记录参数
            # writer.add_histogram('epoch_' + name + '_grad', param.grad, idx_epoch + 1)
            loss_list.append(float(np.array(loss_epoch).mean()))
            psnr_list.append(float(np.array(psnr_epoch).mean()))
            print('Epoch----%5d, loss---%f, PSNR---%f' % (
                idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(psnr_epoch).mean())))

            save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'loss': loss_list,
                'psnr': psnr_list,
            }, save_path='./log/', filename=cfg.model_name + '_epoch_' + str(idx_epoch + 1) + '.pth.tar')

            save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'loss': loss_list,
                'psnr': psnr_list,
            }, save_path='./log/', filename='net_lasted.pth.tar')

            psnr_epoch = []
            loss_epoch = []

        ''' evaluation '''
        net.eval()
        if cfg.eval:  # 构建验证评估类
            print('Verify..............')
            with torch.no_grad():
                PSNR_val = []
                # Loss_val = []
                for idx_iter, (data, label) in enumerate(val_loader):
                    data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)
                    out_val, _ = net(data)
                    # val_loss = criterion_Loss(out_val, label)
                    # Loss_val.append(val_loss.data.cpu())
                    out_val = torch.clamp(out_val, 0., 1.)
                    psnr_batch_val = batch_PSNR(out_val, label, 1.)
                    PSNR_val.append(psnr_batch_val)
                # print(time.ctime()[4:-5] + ' , Valid on Epoch----%5d, loss---%f, PSNR---%f' % (idx_epoch + 1,
                #                                                                                float(np.array(
                #                                                                                    Loss_val).mean()),
                #                                                                                float(np.array(
                #                                                                                    PSNR_val).mean())))
                print(time.ctime()[4:-5] + ' , Valid on Epoch----%5d, PSNR---%f' % (idx_epoch + 1, float(np.array(
                    PSNR_val).mean())))

                # Log the scalar values
                # writer.add_scalar('Loss of every epoch on Verification set', float(np.array(Loss_val).mean()),
                #                   idx_epoch + 1)
                writer.add_scalar('PSNR of every epoch on Verification set', float(np.array(PSNR_val).mean()),
                                  idx_epoch + 1)

        scheduler.step()

        # 记录每一轮训练时间
        time_end = time.time()
        seconds = time_end - time_start
        m, s = divmod(seconds, 60)
        print('训练该轮所花的时间：%f分钟' % m)
        total_train_time = total_train_time + m
        ## epoch training end

    # 打印训练总时间
    print('起始轮次：{},训练总时间：{}分钟'.format(initial_epoch, total_train_time))
    writer.close()


def train_real(train_loader_syn, train_loader_real, val_loader_syn, cfg):
    # set seed
    seed = cfg.seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        print('Seed:\t{}'.format(seed))

    print('--- Hyper-parameters for training ---')
    print(
        'learning_rate: {}\nangular_resolution: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nlambda_ssim: {}\nlambda_cr_syn: {}\nlambda_kl: {}\nlambda_tv: {}\nlambda_cr_real: {}'.format(
            cfg.lr, cfg.angRes, cfg.DataPreprocess_patchsize, cfg.train_batch_size, cfg.val_batch_size, cfg.lambda_ssim,
            cfg.lambda_cr_syn, cfg.lambda_kl, cfg.lambda_tv, cfg.lambda_cr_real))

    net = TransNet(args=cfg, angRes=cfg.angRes, n_groups=cfg.n_groups)
    net.to(cfg.device)
    # net.apply(weights_init_xavier)
    cudnn.benchmark = True

    initial_epoch = 0
    total_train_time = 0  # 总训练时间

    if cfg.load_pretrain:
        # load the lastest model
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
            initial_epoch = model["epoch"]
            print("继续训练！\n")
            print('resuming by loading epoch %d' % initial_epoch)
            net.load_state_dict(model['state_dict'])
        else:
            print("=> no model found at '{}'\n".format(cfg.model_path))
            print("从头训练！\n")

    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    # criterion_Loss = torch.nn.L1Loss().to(cfg.device)

    # --- Define the smooth L1-loss --- #
    criterion_smooth_l1 = criterion1().to(cfg.device)

    # --- Define the perceptual loss network --- #
    # vgg_model = vgg16(pretrained=True).features[:16]
    # vgg_model = vgg_model.to(cfg.device)
    # # vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
    # for param in vgg_model.parameters():
    #     param.requires_grad = False
    # loss_network = LossNetwork(vgg_model)
    # loss_network.eval()

    # --- Define the CR-loss in supervised branch --- #
    criterion_CR_syn = ContrastLoss(ablation=cfg.is_ab)

    # --- Define the ssim-loss --- #
    criterion_ssim = criterion2().to(cfg.device)

    # --- Define the CR-loss in unsupervised branch --- #
    criterion_CR_real = ContrastLoss(ablation=cfg.is_ab)

    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    scheduler._step_count = initial_epoch
    psnr_epoch = []
    loss_epoch = []
    loss_list = []
    psnr_list = []

    loss_syn_epoch = []
    kl_loss_epoch = []
    tv_loss_epoch = []
    cr_real_loss_epoch = []

    # record training
    writer = SummaryWriter('RECORD')
    # 开始训练
    step = 0  # 记录训练次数
    t = 0
    for idx_epoch in range(initial_epoch, cfg.n_epochs):
        net.train()
        time_start = time.time()
        real_data = []
        real_residual = []
        for idx_iter_real, (data, residual) in enumerate(train_loader_real):
            real_data.append(data)
            real_residual.append(residual)
        for idx_iter, (data_syn, label_syn) in enumerate(train_loader_syn):
            data_syn, label_syn = Variable(data_syn).to(cfg.device), Variable(label_syn).to(cfg.device)
            out_syn, _, syn_code2, syn_code3 = net(data_syn)
            out_train = out_syn

            # --- supervised loss --- #
            smooth_loss = criterion_smooth_l1(out_syn, label_syn)
            # perceptual_loss = loss_network(out_syn, label_syn)
            cr_loss_syn = criterion_CR_syn(out_syn, label_syn, data_syn)
            ssim_loss = criterion_ssim(out_syn, label_syn)

            loss_syn = smooth_loss + cfg.lambda_cr_syn * cr_loss_syn + cfg.lambda_ssim * ssim_loss

            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()

            # --- unsupervised loss --- #
            data_real, residual_real = Variable(real_data[idx_iter % len(train_loader_real)]).to(cfg.device), Variable(
                real_residual[idx_iter % len(train_loader_real)]).to(cfg.device)
            out_real, _, real_code2, real_code3 = net(data_real)
            # KL-Divgence Loss
            # syn_code1 = syn_code1.detach()
            syn_code2 = syn_code2.detach()
            syn_code3 = syn_code3.detach()

            # logp_real_code1 = F.log_softmax(real_code1)
            # p_syn_code1 = F.softmax(syn_code1)
            # kl_loss1 = F.kl_div(logp_real_code1, p_syn_code1, reduction='mean')

            logp_real_code2 = F.log_softmax(real_code2)
            p_syn_code2 = F.softmax(syn_code2)
            kl_loss2 = F.kl_div(logp_real_code2, p_syn_code2, reduction='mean')

            logp_real_code3 = F.log_softmax(real_code3)
            p_syn_code3 = F.softmax(syn_code3)
            kl_loss3 = F.kl_div(logp_real_code3, p_syn_code3, reduction='mean')

            kl_streak_loss = kl_loss2 + kl_loss3

            # Adversarial loss
            # TV Loss
            tv_loss = TVLossL1(out_real)
            # CR Loss
            cr_loss_real = criterion_CR_real(out_real, residual_real, data_real)

            loss_real = (t / 1) * kl_streak_loss + cfg.lambda_tv * tv_loss + (1 / (1 + t)) * cr_loss_real

            optimizer.zero_grad()
            loss_real.backward()
            optimizer.step()

            loss = loss_syn + loss_real
            loss_epoch.append(loss.data.cpu())
            loss_syn_epoch.append(loss_syn.data.cpu())
            kl_loss_epoch.append(kl_streak_loss.data.cpu())
            tv_loss_epoch.append(tv_loss.data.cpu())
            cr_real_loss_epoch.append(cr_loss_real.data.cpu())
            # psnr_epoch.append(batch_PSNR(out.data.cpu(), label.data.cpu()))

            # training curve
            net.eval()
            # out_train = net(data)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, label_syn, 1.)
            psnr_epoch.append(psnr_train)
            print(
                "已完成第{}轮训练的{:.2f}%，目前损失值为{:.6f}, 合成损失值为{:.6f}, smooth损失值为{:.6f}, CR_syn损失值为{:.6f}, SSIM损失值为{:.6f}, 真实损失值为{:.6f}, kl损失值为{:.6f},tv损失值为{:.6f},CR_real损失值为{:.6f}, PSNR: {:.4f},t: {:.2f}。".format(
                    idx_epoch + 1,
                    (idx_iter + 1) / len(train_loader_syn) * 100,
                    loss.item(),
                    loss_syn.item(),
                    smooth_loss.item(),
                    cr_loss_syn.item(),
                    ssim_loss.item(),
                    loss_real.item(),
                    kl_streak_loss.item(),
                    tv_loss.item(),
                    cr_loss_real.item(),
                    psnr_train,
                    t))
            if step % 100 == 0:
                # Log the scalar values
                writer.add_scalar('Loss of every batch on training set', loss.item(), step)
                writer.add_scalar('PSNR of every batch on training set', psnr_train, step)
            step += 1
            t += 1
            # save model
            if (idx_iter + 1) % 3750 == 0:
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'iter': idx_iter + 1,
                    'state_dict': net.state_dict(),
                }, save_path='./log/', filename=cfg.model_name + '_epoch_' + str(idx_epoch + 1) + '_iter_' + str(idx_iter + 1) + '.pth.tar')
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'iter': idx_iter + 1,
                    'state_dict': net.state_dict(),
                }, save_path='./log/', filename='iter_' + str(idx_iter + 1) + '_net_lasted.pth.tar')

        # save model
        if idx_epoch % 1 == 0:
            writer.add_scalar('Loss of every epoch on training set', float(np.array(loss_epoch).mean()), idx_epoch + 1)
            writer.add_scalar('PSNR of every epoch on training set', float(np.array(psnr_epoch).mean()), idx_epoch + 1)
            writer.add_scalar('SYN LOSS of every epoch on training set', float(np.array(loss_syn_epoch).mean()),
                              idx_epoch + 1)
            writer.add_scalar('KL LOSS of every epoch on training set', float(np.array(kl_loss_epoch).mean()),
                              idx_epoch + 1)
            writer.add_scalar('TV LOSS of every epoch on training set', float(np.array(tv_loss_epoch).mean()),
                              idx_epoch + 1)
            writer.add_scalar('CR_REAL LOSS of every epoch on training set', float(np.array(cr_real_loss_epoch).mean()),
                              idx_epoch + 1)
            # for name, param in net.named_parameters():
            # writer.add_histogram('epoch_' + name + '_param', param, idx_epoch + 1)  # 记录参数
            # writer.add_histogram('epoch_' + name + '_grad', param.grad, idx_epoch + 1)
            loss_list.append(float(np.array(loss_epoch).mean()))
            psnr_list.append(float(np.array(psnr_epoch).mean()))
            print('Epoch----%5d, loss---%f, PSNR---%f' % (
                idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(psnr_epoch).mean())))

            save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'loss': loss_list,
                'psnr': psnr_list,
            }, save_path='./log/', filename=cfg.model_name + '_epoch_' + str(idx_epoch + 1) + '.pth.tar')

            save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'loss': loss_list,
                'psnr': psnr_list,
            }, save_path='./log/', filename='net_lasted.pth.tar')

            psnr_epoch = []
            loss_epoch = []

        ''' evaluation '''
        net.eval()
        if cfg.eval:  # 构建验证评估类
            print('Verify..............')
            with torch.no_grad():
                PSNR_val = []
                # Loss_val = []
                for idx_iter, (data, label) in enumerate(val_loader_syn):
                    data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)
                    out_val, _, _, _ = net(data)
                    # val_loss = criterion_Loss(out_val, label)
                    # Loss_val.append(val_loss.data.cpu())
                    out_val = torch.clamp(out_val, 0., 1.)
                    psnr_batch_val = batch_PSNR(out_val, label, 1.)
                    PSNR_val.append(psnr_batch_val)
                # print(time.ctime()[4:-5] + ' , Valid on Epoch----%5d, loss---%f, PSNR---%f' % (idx_epoch + 1,
                #                                                                                float(np.array(
                #                                                                                    Loss_val).mean()),
                #                                                                                float(np.array(
                #                                                                                    PSNR_val).mean())))
                print(time.ctime()[4:-5] + ' , Valid on Epoch----%5d, PSNR---%f' % (idx_epoch + 1, float(np.array(
                    PSNR_val).mean())))

                # Log the scalar values
                # writer.add_scalar('Loss of every epoch on Verification set', float(np.array(Loss_val).mean()),
                #                   idx_epoch + 1)
                writer.add_scalar('PSNR of every epoch on Verification set', float(np.array(PSNR_val).mean()),
                                  idx_epoch + 1)

        scheduler.step()

        # 记录每一轮训练时间
        time_end = time.time()
        seconds = time_end - time_start
        m, s = divmod(seconds, 60)
        print('训练该轮所花的时间：%f分钟' % m)
        total_train_time = total_train_time + m
        ## epoch training end

    # 打印训练总时间
    print('起始轮次：{},训练总时间：{}分钟'.format(initial_epoch, total_train_time))
    writer.close()


# def cal_psnr(img1, img2):
#     _, _, h, w = img1.size()
#     mse = torch.sum((img1 - img2) ** 2) / img1.numel()
#     psnr = 10 * log10(1 / mse)
#     return psnr


def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)


def main(cfg):
    # 加载数据集
    if cfg.is_semi == False:
        print('Loading synthetic train dataset ...\n')
        dataset_train_syn = Train_Dataset_syn(data_path=cfg.trainset_dir_syn)
        loader_train_syn = DataLoader(dataset=dataset_train_syn, num_workers=cfg.num_workers,
                                      batch_size=cfg.train_batch_size,
                                      shuffle=True)
        print("# of synthetic training samples: %d\n" % int(len(dataset_train_syn)))  # 训练样本数
        if cfg.eval:
            print('Loading synthetic val dataset ...\n')
            dataset_val_syn = Test_Dataset_syn(data_path=cfg.testset_dir_syn)
            loader_val_syn = DataLoader(dataset=dataset_val_syn, num_workers=cfg.num_workers,
                                        batch_size=cfg.val_batch_size,
                                        shuffle=True)
            print("# of synthetic val samples: %d\n" % int(len(dataset_val_syn)))  # 验证样本数
        train_syn(loader_train_syn, loader_val_syn, cfg)
    else:
        print('Loading synthetic train dataset ...\n')
        dataset_train_syn = Train_Dataset_syn(data_path=cfg.trainset_dir_syn)
        loader_train_syn = DataLoader(dataset=dataset_train_syn, num_workers=cfg.num_workers,
                                      batch_size=cfg.train_batch_size,
                                      shuffle=True)
        print("# of synthetic training samples: %d\n" % int(len(dataset_train_syn)))  # 训练样本数

        print('Loading real-world train dataset ...\n')
        dataset_train_real = Train_Dataset_real(data_path=cfg.trainset_dir_real)
        loader_train_real = DataLoader(dataset=dataset_train_real, num_workers=cfg.num_workers,
                                       batch_size=cfg.train_batch_size,
                                       shuffle=True)
        print("# of real-world training samples: %d\n" % int(len(dataset_train_real)))  # 训练样本数

        if cfg.eval:
            print('Loading synthetic val dataset ...\n')
            dataset_val_syn = Test_Dataset_syn(data_path=cfg.testset_dir_syn)
            loader_val_syn = DataLoader(dataset=dataset_val_syn, num_workers=cfg.num_workers,
                                        batch_size=cfg.val_batch_size,
                                        shuffle=True)
            print("# of synthetic val samples: %d\n" % int(len(dataset_val_syn)))  # 验证样本数
        train_real(loader_train_syn, loader_train_real, loader_val_syn, cfg)


if __name__ == '__main__':
    cfg = parse_args()
    if cfg.is_semi == False:
        if cfg.preprocess_syn:
            prepare_train_data_syn(data_path=cfg.trainset_dir_syn, patch_size=cfg.DataPreprocess_patchsize,
                                   stride=cfg.stride,
                                   angs=cfg.angRes)
            if cfg.eval:
                prepare_test_data_syn(data_path=cfg.testset_dir_syn, patch_size=cfg.DataPreprocess_patchsize,
                                      stride=cfg.stride,
                                      angs=cfg.angRes)
    else:
        if cfg.preprocess_syn:
            prepare_train_data_syn(data_path=cfg.trainset_dir_syn, patch_size=cfg.DataPreprocess_patchsize,
                                   stride=cfg.stride,
                                   angs=cfg.angRes)
            if cfg.eval:
                prepare_test_data_syn(data_path=cfg.testset_dir_syn, patch_size=cfg.DataPreprocess_patchsize,
                                      stride=cfg.stride,
                                      angs=cfg.angRes)

        if cfg.preprocess_real:
            prepare_train_data_real(data_path=cfg.trainset_dir_real, patch_size=cfg.DataPreprocess_patchsize,
                                    stride=cfg.stride,
                                    angs=cfg.angRes)
    main(cfg)
