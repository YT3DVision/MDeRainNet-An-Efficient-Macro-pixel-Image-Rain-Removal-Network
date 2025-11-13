import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math


class TransNet(nn.Module):
    def __init__(self, args, angRes, n_groups):
        super(TransNet, self).__init__()
        self.args = args
        n_feats = args.n_feats
        # Feature Extraction
        self.angRes = angRes
        self.init_conv = nn.Conv2d(args.n_colors, n_feats, kernel_size=3, stride=1, dilation=angRes, padding=angRes,
                                   bias=False)
        self.disentg0 = DisentgBlock(angRes, n_feats)

        self.disentg1 = DisentgBlock(angRes, n_feats)
        self.Downsample1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 2 // 2, kernel_size=1, stride=1, padding=0, bias=False),
            PixelUnshuffle(2))

        self.disentg2 = DisentgBlock(angRes, n_feats)
        self.Downsample2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 2 // 2, kernel_size=1, stride=1, padding=0, bias=False),
            PixelUnshuffle(2))

        self.disentg3 = DisentgBlock(angRes, n_feats)
        self.Downsample3 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 2 // 2, kernel_size=1, stride=1, padding=0, bias=False),
            PixelUnshuffle(2))

        self.angFE1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=int(self.angRes), stride=int(self.angRes), padding=0,
                      bias=False),
            nn.Conv2d(n_feats, int(angRes * angRes * n_feats), kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        self.spaFE1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, dilation=int(self.angRes),
                                padding=int(self.angRes),
                                bias=False)
        self.angFE2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=int(self.angRes), stride=int(self.angRes), padding=0,
                      bias=False),
            nn.Conv2d(n_feats, int(angRes * angRes * n_feats), kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        self.spaFE2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, dilation=int(self.angRes),
                                padding=int(self.angRes),
                                bias=False)
        self.angFE3 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=int(self.angRes), stride=int(self.angRes), padding=0,
                      bias=False),
            nn.Conv2d(n_feats, int(angRes * angRes * n_feats), kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        self.spaFE3 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, dilation=int(self.angRes),
                                padding=int(self.angRes),
                                bias=False)
        # Spatial-Angular Interaction
        self.interaction1 = VisionTransformer(img_dim=args.patch_size // 2, patch_dim=args.patch_dim,
                                              num_channels=n_feats,
                                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                              num_heads=args.num_heads,
                                              num_layers=1,
                                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                              # num_queries=args.num_queries,
                                              dropout_rate=args.dropout_rate,
                                              mlp=args.no_mlp,
                                              pos_every=args.pos_every, no_pos=args.no_pos, no_norm=args.no_norm)
        # Fusion and Reconstruction
        self.bottleneck1 = BottleNeck1(angRes, n_groups, n_feats)
        self.interaction2 = VisionTransformer(img_dim=args.patch_size // 4, patch_dim=args.patch_dim,
                                              num_channels=n_feats,
                                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                              num_heads=args.num_heads,
                                              num_layers=2,
                                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                              # num_queries=args.num_queries,
                                              dropout_rate=args.dropout_rate,
                                              mlp=args.no_mlp,
                                              pos_every=args.pos_every, no_pos=args.no_pos, no_norm=args.no_norm)
        # Fusion and Reconstruction
        self.bottleneck2 = BottleNeck2(angRes, n_groups, n_feats)
        self.interaction3 = VisionTransformer(img_dim=args.patch_size // 8, patch_dim=args.patch_dim,
                                              num_channels=n_feats,
                                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                              num_heads=args.num_heads,
                                              num_layers=args.num_layers,
                                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                              # num_queries=args.num_queries,
                                              dropout_rate=args.dropout_rate,
                                              mlp=args.no_mlp,
                                              pos_every=args.pos_every, no_pos=args.no_pos, no_norm=args.no_norm)
        # Fusion and Reconstruction
        self.bottleneck3 = BottleNeck3(angRes, n_groups, n_feats)

        self.Upsample1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 2 * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2))
        self.reduce_chan_level3 = nn.Conv2d(2 * n_feats, n_feats, kernel_size=3, stride=1,
                                            dilation=int(self.angRes),
                                            padding=int(self.angRes),
                                            bias=False)
        self.disentg4 = DisentgBlock(angRes, n_feats)

        self.Upsample2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 2 * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2))
        self.reduce_chan_level2 = nn.Conv2d(2 * n_feats, n_feats, kernel_size=3, stride=1,
                                            dilation=int(self.angRes),
                                            padding=int(self.angRes),
                                            bias=False)
        self.disentg5 = DisentgBlock(angRes, n_feats)

        self.Upsample3 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 2 * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2))
        self.reduce_chan_level1 = nn.Conv2d(2 * n_feats, n_feats, kernel_size=3, stride=1,
                                            dilation=int(self.angRes),
                                            padding=int(self.angRes),
                                            bias=False)
        self.disentg6 = DisentgBlock(angRes, n_feats)

        self.reconstruction = ReconBlock(angRes, n_feats)

    def forward(self, x):
        orig_x = x
        x = SAI2MacPI(x, self.angRes)
        buffer = self.init_conv(x)
        buffer2 = self.disentg0(buffer)

        down_level1_x = self.Downsample1(buffer2)
        xa1, xs1 = self.angFE1(down_level1_x), self.spaFE1(down_level1_x)
        buffer_a1, buffer_s1 = self.interaction1(xa1, xs1)
        buffer_a1, buffer_s1 = torch.cat((buffer_a1, xa1), 1), torch.cat((buffer_s1, xs1), 1)
        down_level1_x_im = self.bottleneck1(buffer_a1, buffer_s1) + down_level1_x
        down_level1_x = self.disentg1(down_level1_x_im)

        # latent space
        latent_vector1 = down_level1_x

        down_level2_x = self.Downsample2(down_level1_x)
        xa2, xs2 = self.angFE2(down_level2_x), self.spaFE2(down_level2_x)
        buffer_a2, buffer_s2 = self.interaction2(xa2, xs2)
        buffer_a2, buffer_s2 = torch.cat((buffer_a2, xa2), 1), torch.cat((buffer_s2, xs2), 1)
        down_level2_x_im = self.bottleneck2(buffer_a2, buffer_s2) + down_level2_x
        down_level2_x = self.disentg2(down_level2_x_im)

        # latent space
        latent_vector2 = down_level2_x

        down_level3_x = self.Downsample3(down_level2_x)
        xa3, xs3 = self.angFE3(down_level3_x), self.spaFE3(down_level3_x)
        buffer_a3, buffer_s3 = self.interaction3(xa3, xs3)
        buffer_a3, buffer_s3 = torch.cat((buffer_a3, xa3), 1), torch.cat((buffer_s3, xs3), 1)
        down_level3_x_im = self.bottleneck3(buffer_a3, buffer_s3) + down_level3_x
        down_level3_x = self.disentg3(down_level3_x_im)

        # latent space
        latent_vector3 = down_level3_x

        up_level2_x = self.Upsample1(down_level3_x)
        up_level2_x = torch.cat((down_level2_x_im, up_level2_x), 1)
        up_level2_x = self.reduce_chan_level3(up_level2_x)
        up_level2_x = self.disentg4(up_level2_x)
        up_level1_x = self.Upsample2(up_level2_x)
        up_level1_x = torch.cat((down_level1_x_im, up_level1_x), 1)
        up_level1_x = self.reduce_chan_level2(up_level1_x)
        up_level1_x = self.disentg5(up_level1_x)
        up_level0_x = self.Upsample3(up_level1_x)
        up_level0_x = torch.cat((buffer2, up_level0_x), 1)
        up_level0_x = self.reduce_chan_level1(up_level0_x)
        up_level0_x = self.disentg6(up_level0_x)

        out = self.reconstruction(up_level0_x)
        out = orig_x - out
        return out, latent_vector1, latent_vector2, latent_vector3


class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            # num_queries,
            # positional_encoding_type="learned",
            dropout_rate=0,
            no_norm=False,
            mlp=False,
            pos_every=False,
            no_pos=False
    ):
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels

        self.out_dim = patch_dim * patch_dim * num_channels

        self.no_pos = no_pos

        if self.mlp == False:
            self.linear_encoding1 = nn.Linear(self.flatten_dim, embedding_dim)
            self.linear_encoding2 = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head1 = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate)
            )
            self.mlp_head2 = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate)
            )
            # self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)

        ang_encoder_layer = TransformerEncoderLayer_ang(embedding_dim, num_heads, hidden_dim, dropout_rate,
                                                        self.no_norm)
        spa_encoder_layer = TransformerEncoderLayer_spa(embedding_dim, num_heads, hidden_dim, dropout_rate,
                                                        self.no_norm)
        self.encoder = TransformerEncoder(ang_encoder_layer, spa_encoder_layer, num_layers)

        if not self.no_pos:
            self.position_encoding_a = LearnedPositionalEncoding_a(
                self.seq_length, self.embedding_dim, self.seq_length
            )
            self.position_encoding_s = LearnedPositionalEncoding_s(
                self.seq_length, self.embedding_dim, self.seq_length
            )

        self.dropout_layer1 = nn.Dropout(dropout_rate)
        self.dropout_layer2 = nn.Dropout(dropout_rate)

        # if no_norm:
        #     for m in self.modules():
        #         if isinstance(m, nn.Linear):
        #             nn.init.normal_(m.weight, std = 1/m.weight.size(1))

    def forward(self, xa, xs):
        buffer_a = torch.nn.functional.unfold(xa, self.patch_dim, stride=self.patch_dim).transpose(1, 2).transpose(0,
                                                                                                                   1).contiguous()
        buffer_s = torch.nn.functional.unfold(xs, self.patch_dim, stride=self.patch_dim).transpose(1,
                                                                                                   2).transpose(0,
                                                                                                                1).contiguous()
        if self.mlp == False:
            buffer_a = self.dropout_layer1(self.linear_encoding1(buffer_a)) + buffer_a
            buffer_s = self.dropout_layer2(self.linear_encoding2(buffer_s)) + buffer_s

        if not self.no_pos:
            pos_a = self.position_encoding_a(buffer_a).transpose(0, 1)
            pos_s = self.position_encoding_s(buffer_s).transpose(0, 1)

        if self.pos_every:
            buffer_a, buffer_s = self.encoder(buffer_a, buffer_s, pos_a=pos_a, pos_s=pos_s)
        elif self.no_pos:
            buffer_a, buffer_s = self.encoder(buffer_a, buffer_s)
        else:
            buffer_a, buffer_s = self.encoder(buffer_a + pos_a, buffer_s + pos_s)

        if self.mlp == False:
            buffer_a[self.num_layers - 1] = self.mlp_head1(buffer_a[self.num_layers - 1]) + buffer_a[
                self.num_layers - 1]
            buffer_s[self.num_layers - 1] = self.mlp_head2(buffer_s[self.num_layers - 1]) + buffer_s[
                self.num_layers - 1]

        for i in range(self.num_layers):
            buffer_a[i] = buffer_a[i].transpose(0, 1).contiguous().view(buffer_a[i].size(1), -1, self.flatten_dim)
            buffer_s[i] = buffer_s[i].transpose(0, 1).contiguous().view(buffer_s[i].size(1), -1, self.flatten_dim)
            buffer_a[i] = torch.nn.functional.fold(buffer_a[i].transpose(1, 2).contiguous(), int(self.img_dim),
                                                   self.patch_dim,
                                                   stride=self.patch_dim)
            buffer_s[i] = torch.nn.functional.fold(buffer_s[i].transpose(1, 2).contiguous(), int(self.img_dim),
                                                   self.patch_dim,
                                                   stride=self.patch_dim)
        buffer_a = torch.cat(buffer_a, 1)
        buffer_s = torch.cat(buffer_s, 1)
        return buffer_a, buffer_s


class LearnedPositionalEncoding_a(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding_a, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return position_embeddings


class LearnedPositionalEncoding_s(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding_s, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return position_embeddings


class TransformerEncoder(nn.Module):

    def __init__(self, ang_encoder_layer, spa_encoder_layer, num_layers):
        super().__init__()
        self.ang_layers = _get_clones(ang_encoder_layer, num_layers)
        self.spa_layers = _get_clones(spa_encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src_ang, src_spa, pos_a=None, pos_s=None):
        output_ang = src_ang
        output_spa = src_spa
        out_a = []
        out_s = []
        for num in range(self.num_layers):
            output_ang1 = self.ang_layers[num](output_ang, output_spa, pos_a=pos_a, pos_s=pos_s)
            output_spa1 = self.spa_layers[num](output_ang, output_spa, pos_a=pos_a, pos_s=pos_s)
            output_ang = output_ang1
            output_spa = output_spa1

            out_a.append(output_ang)
            out_s.append(output_spa)
        return out_a, out_s


class TransformerEncoderLayer_ang(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_multihead_attn = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout_self_attn = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm4 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm5 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation1 = _get_activation_fn(activation)
        self.activation2 = _get_activation_fn(activation)

        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.multihead_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src_ang, src_spa, pos_a=None, pos_s=None):
        src_ang = self.norm1(src_ang)
        src_spa = self.norm2(src_spa)
        q = self.with_pos_embed(src_spa, pos_s)
        k = self.with_pos_embed(src_ang, pos_a)
        src2 = self.multihead_attn(q, k, src_ang)
        src = self.dropout1(src2[0])
        src2 = self.norm3(src)
        src2 = self.linear2(self.dropout_multihead_attn(self.activation1(self.linear1(src2))))
        src = src + self.dropout2(src2)
        src = src_ang + src

        src_ang2 = self.norm4(src)
        q = k = src_ang2
        src_ang2 = self.self_attn(q, k, value=src_ang2)[0]
        src_ang = src + self.dropout3(src_ang2)
        src_ang2 = self.norm5(src_ang)
        src_ang2 = self.linear4(self.dropout_self_attn(self.activation2(self.linear3(src_ang2))))
        src_ang = src_ang + self.dropout4(src_ang2)
        return src_ang


class TransformerEncoderLayer_spa(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_multihead_attn = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout_self_attn = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm4 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm5 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation1 = _get_activation_fn(activation)
        self.activation2 = _get_activation_fn(activation)

        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.multihead_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src_ang, src_spa, pos_a=None, pos_s=None):
        src_ang = self.norm1(src_ang)
        src_spa = self.norm2(src_spa)
        q = self.with_pos_embed(src_ang, pos_a)
        k = self.with_pos_embed(src_spa, pos_s)
        src2 = self.multihead_attn(q, k, src_spa)
        src = self.dropout1(src2[0])
        src2 = self.norm3(src)
        src2 = self.linear2(self.dropout_multihead_attn(self.activation1(self.linear1(src2))))
        src = src + self.dropout2(src2)
        src = src_spa + src

        src_spa2 = self.norm4(src)
        q = k = src_spa2
        src_spa2 = self.self_attn(q, k, value=src_spa2)[0]
        src_spa = src + self.dropout3(src_spa2)
        src_spa2 = self.norm5(src_spa)
        src_spa2 = self.linear4(self.dropout_self_attn(self.activation2(self.linear3(src_spa2))))
        src_spa = src_spa + self.dropout4(src_spa2)
        return src_spa


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class DisentgBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(DisentgBlock, self).__init__()
        SpaChannel, AngChannel = channels, channels // 2

        self.SpaConv = nn.Sequential(
            nn.Conv2d(channels, SpaChannel, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(SpaChannel, SpaChannel, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.AngConv = nn.Sequential(
            nn.Conv2d(channels, AngChannel, kernel_size=angRes, stride=angRes, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(AngChannel, angRes * angRes * AngChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.PixelShuffle(angRes),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(SpaChannel + AngChannel, channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.SpaConv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False),
        )

    def forward(self, x):
        feaSpa = self.SpaConv(x)
        feaAng = self.AngConv(x)
        buffer = torch.cat((feaSpa, feaAng), dim=1)
        buffer = self.fuse(buffer)
        y = self.SpaConv2(buffer) + buffer
        return y + x


class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler
    Upscales the last dimension (i.e., W) of a tensor by reducing its channel length
    inout: x of size [b, factor*c, h, w]
    output: y of size [b, c, h, w*factor]
    """

    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor
        x = x.contiguous().view(b, self.factor, c, h, w)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # b, c, h, w, factor
        y = x.view(b, c, h, w * self.factor)
        return y


def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)


class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''

        return pixel_unshuffle(input, self.downscale_factor)


class BottleNeck1(nn.Module):
    def __init__(self, angRes, n_blocks, channels):
        super(BottleNeck1, self).__init__()

        self.AngBottle = nn.Conv2d((1 + 1) * channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.SpaBottle = nn.Conv2d((1 + 2) * channels, channels, kernel_size=3, stride=1, dilation=int(angRes),
                                   padding=int(angRes), bias=False)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, xa, xs):
        xa = self.ReLU(self.AngBottle(xa))
        xs = torch.cat((xs, xa), 1)
        out = self.ReLU(self.SpaBottle(xs))
        return out


class BottleNeck2(nn.Module):
    def __init__(self, angRes, n_blocks, channels):
        super(BottleNeck2, self).__init__()

        self.AngBottle = nn.Conv2d((1 + 2) * channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.SpaBottle = nn.Conv2d((1 + 3) * channels, channels, kernel_size=3, stride=1, dilation=int(angRes),
                                   padding=int(angRes), bias=False)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, xa, xs):
        xa = self.ReLU(self.AngBottle(xa))
        xs = torch.cat((xs, xa), 1)
        out = self.ReLU(self.SpaBottle(xs))
        return out


class BottleNeck3(nn.Module):
    def __init__(self, angRes, n_blocks, channels):
        super(BottleNeck3, self).__init__()

        self.AngBottle = nn.Conv2d((n_blocks + 1) * channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.SpaBottle = nn.Conv2d((n_blocks + 2) * channels, channels, kernel_size=3, stride=1, dilation=int(angRes),
                                   padding=int(angRes), bias=False)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, xa, xs):
        xa = self.ReLU(self.AngBottle(xa))
        xs = torch.cat((xs, xa), 1)
        out = self.ReLU(self.SpaBottle(xs))
        return out


class ReconBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(ReconBlock, self).__init__()
        self.PreConv = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                                 dilation=int(angRes), padding=int(angRes), bias=False)
        self.FinalConv = nn.Conv2d(int(channels), 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.angRes = angRes

    def forward(self, x):
        buffer = self.PreConv(x)
        bufferSAI_LR = MacPI2SAI(buffer, self.angRes)
        bufferSAI_HR = bufferSAI_LR
        out = self.FinalConv(bufferSAI_HR)
        return out


def MacPI2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out


def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out
