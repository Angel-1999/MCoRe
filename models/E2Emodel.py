import os
import argparse
from contextlib import nullcontext
import random
import numpy as np
from tabulate import tabulate
import torch

torch.backends.cudnn.benchmark = True
from models.common import  BaseRGBModel
from models.shift import make_temporal_shift
from models.modules import *
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm


# Prevent the GRU params from going too big (cap it at a RegNet-Y 800MF)
MAX_GRU_HIDDEN_DIM = 768

class E2EModel(nn.Module):

    def __init__(self, feature_arch, clip_len, modality):
        super().__init__()
        is_rgb = modality == 'rgb'
        in_channels = {'flow': 2, 'bw': 1, 'rgb': 3}[modality]

        if feature_arch.startswith(('rn18', 'rn50')):
            resnet_name = feature_arch.split('_')[0].replace('rn', 'resnet')
            features = getattr(
                torchvision.models, resnet_name)(pretrained=is_rgb)
            feat_dim = features.fc.in_features
            features.fc = nn.Identity()
            # import torchsummary
            # print(torchsummary.summary(features.to('cuda'), (3, 224, 224)))

            # Flow has only two input channels
            if not is_rgb:
                # FIXME: args maybe wrong for larger resnet
                features.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=(7, 7), stride=(2, 2),
                    padding=(3, 3), bias=False)

        elif feature_arch.startswith(('rny002', 'rny008')):
            features = timm.create_model({
                                             'rny002': 'regnety_002',
                                             'rny008': 'regnety_008',
                                         }[feature_arch.rsplit('_', 1)[0]], pretrained=is_rgb)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()

            if not is_rgb:
                features.stem.conv = nn.Conv2d(
                    in_channels, 32, kernel_size=(3, 3), stride=(2, 2),
                    padding=(1, 1), bias=False)

        elif 'convnextt' in feature_arch:
            features = timm.create_model('convnext_tiny', pretrained=is_rgb)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()

            if not is_rgb:
                features.stem[0] = nn.Conv2d(
                    in_channels, 96, kernel_size=4, stride=4)

        else:
            raise NotImplementedError(feature_arch)

        # Add Temporal Shift Modules
        self._require_clip_len = -1
        if feature_arch.endswith('_tsm'):
            make_temporal_shift(features, clip_len, is_gsm=False)
            self._require_clip_len = clip_len
        elif feature_arch.endswith('_gsm'):
            make_temporal_shift(features, clip_len, is_gsm=True)
            self._require_clip_len = clip_len

        self._features = features
        self._feat_dim = feat_dim


    def forward(self, x):
        # print(x.shape) #[8, 100, 3, 224, 224]) 改为 [8, 3, 96, 112, 112])
        batch_size, channels, true_clip_len, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4)

        clip_len = true_clip_len
        if self._require_clip_len > 0:
            # TSM module requires clip len to be known
            assert true_clip_len <= self._require_clip_len, \
                'Expected {}, got {}'.format(
                    self._require_clip_len, true_clip_len)
            if true_clip_len < self._require_clip_len:
                x = F.pad(
                    x, (0,) * 7 + (self._require_clip_len - true_clip_len,))
                clip_len = self._require_clip_len

        # print(x.size()) ([16, 96, 3, 112, 112])
        im_featmap = self._features(
            x.reshape(-1, channels, height, width)
        )
        im_feat = im_featmap.reshape(batch_size, clip_len, self._feat_dim)

        if true_clip_len != clip_len:
            # Undo padding
            im_feat = im_feat[:, :true_clip_len, :]

        # print(im_featmap.size(),im_feat.size()) torch.Size([1536, 368]) torch.Size([16, 96, 368])
        return im_feat

class SpotModel(nn.Module):
    def __init__(self, num_classes, temporal_arch, feat_dim = 368):
        super().__init__()

        if 'gru' in temporal_arch:  # default
            hidden_dim = feat_dim
            if hidden_dim > MAX_GRU_HIDDEN_DIM:
                hidden_dim = MAX_GRU_HIDDEN_DIM
                print('Clamped GRU hidden dim: {} -> {}'.format(
                    feat_dim, hidden_dim))
            if temporal_arch in ('gru', 'deeper_gru'):
                self._pred_fine = GRUPrediction(
                    feat_dim, num_classes, hidden_dim,
                    num_layers=3 if temporal_arch[0] == 'd' else 1)
            else:
                raise NotImplementedError(temporal_arch)
        elif temporal_arch == 'mstcn':
            self._pred_fine = TCNPrediction(feat_dim, num_classes, 3)
        elif temporal_arch == 'asformer':
            self._pred_fine = ASFormerPrediction(feat_dim, num_classes, 3)
        elif temporal_arch == '':
            self._pred_fine = FCPrediction(feat_dim, num_classes)
        else:
            raise NotImplementedError(temporal_arch)

    def forward(self,x):

        pred = self._pred_fine(x)
        # print(pred.size(),self._pred_fine)

        return pred


class EnhModel (nn.Module):
    def __init__(self,feat_dim, hidden_dim, num_layers=3):
        super().__init__()

        self.enhance = nn.GRU(feat_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self._dropout = nn.Dropout()

    def forward(self, x):
        # x: 8, dim, 96
        # (batch_size, sequence_length, input_size)
        x = x.permute(0, 2, 1)
        # print(x.size())
        y, _ = self.enhance(x)
        return self._dropout(y)


# class E2EModel(BaseRGBModel):
#
#     class Impl(nn.Module):
#
#         def __init__(self, num_classes, feature_arch, temporal_arch, clip_len,
#                      modality):
#             super().__init__()
#             is_rgb = modality == 'rgb'
#             in_channels = {'flow': 2, 'bw': 1, 'rgb': 3}[modality]
#
#             if feature_arch.startswith(('rn18', 'rn50')):
#                 resnet_name = feature_arch.split('_')[0].replace('rn', 'resnet')
#                 features = getattr(
#                     torchvision.models, resnet_name)(pretrained=is_rgb)
#                 feat_dim = features.fc.in_features
#                 features.fc = nn.Identity()
#                 # import torchsummary
#                 # print(torchsummary.summary(features.to('cuda'), (3, 224, 224)))
#
#                 # Flow has only two input channels
#                 if not is_rgb:
#                     #FIXME: args maybe wrong for larger resnet
#                     features.conv1 = nn.Conv2d(
#                         in_channels, 64, kernel_size=(7, 7), stride=(2, 2),
#                         padding=(3, 3), bias=False)
#
#             elif feature_arch.startswith(('rny002', 'rny008')):
#                 features = timm.create_model({
#                     'rny002': 'regnety_002',
#                     'rny008': 'regnety_008',
#                 }[feature_arch.rsplit('_', 1)[0]], pretrained=is_rgb)
#                 feat_dim = features.head.fc.in_features
#                 features.head.fc = nn.Identity()
#
#                 if not is_rgb:
#                     features.stem.conv = nn.Conv2d(
#                         in_channels, 32, kernel_size=(3, 3), stride=(2, 2),
#                         padding=(1, 1), bias=False)
#
#             elif 'convnextt' in feature_arch:
#                 features = timm.create_model('convnext_tiny', pretrained=is_rgb)
#                 feat_dim = features.head.fc.in_features
#                 features.head.fc = nn.Identity()
#
#                 if not is_rgb:
#                     features.stem[0] = nn.Conv2d(
#                         in_channels, 96, kernel_size=4, stride=4)
#
#             else:
#                 raise NotImplementedError(feature_arch)
#
#             # Add Temporal Shift Modules
#             self._require_clip_len = -1
#             if feature_arch.endswith('_tsm'):
#                 make_temporal_shift(features, clip_len, is_gsm=False)
#                 self._require_clip_len = clip_len
#             elif feature_arch.endswith('_gsm'):
#                 make_temporal_shift(features, clip_len, is_gsm=True)
#                 self._require_clip_len = clip_len
#
#             self._features = features
#             self._feat_dim = feat_dim
#
#             if 'gru' in temporal_arch: # default
#                 hidden_dim = feat_dim
#                 if hidden_dim > MAX_GRU_HIDDEN_DIM:
#                     hidden_dim = MAX_GRU_HIDDEN_DIM
#                     print('Clamped GRU hidden dim: {} -> {}'.format(
#                         feat_dim, hidden_dim))
#                 if temporal_arch in ('gru', 'deeper_gru'):
#                     self._pred_fine = GRUPrediction(
#                         feat_dim, num_classes, hidden_dim,
#                         num_layers=3 if temporal_arch[0] == 'd' else 1)
#                 else:
#                     raise NotImplementedError(temporal_arch)
#             elif temporal_arch == 'mstcn':
#                 self._pred_fine = TCNPrediction(feat_dim, num_classes, 3)
#             elif temporal_arch == 'asformer':
#                 self._pred_fine = ASFormerPrediction(feat_dim, num_classes, 3)
#             elif temporal_arch == '':
#                 self._pred_fine = FCPrediction(feat_dim, num_classes)
#             else:
#                 raise NotImplementedError(temporal_arch)
#
#         def forward(self, x):
#             # print(x.shape) [8, 100, 3, 224, 224]) 改为 [8, 3, 96, 112, 112])
#             batch_size, channels, true_clip_len, height, width = x.shape
#
#             clip_len = true_clip_len
#             if self._require_clip_len > 0:
#                 # TSM module requires clip len to be known
#                 assert true_clip_len <= self._require_clip_len, \
#                     'Expected {}, got {}'.format(
#                         self._require_clip_len, true_clip_len)
#                 if true_clip_len < self._require_clip_len:
#                     x = F.pad(
#                         x, (0,) * 7 + (self._require_clip_len - true_clip_len,))
#                     clip_len = self._require_clip_len
#
#             im_feat = self._features(
#                 x.view(-1, channels, height, width)
#             ).reshape(batch_size, clip_len, self._feat_dim)
#
#             if true_clip_len != clip_len:
#                 # Undo padding
#                 im_feat = im_feat[:, :true_clip_len, :]
#
#             out = self._pred_fine(im_feat)
#
#             return out,im_feat
#
#         def print_stats(self):
#             print('Model params:',
#                 sum(p.numel() for p in self.parameters()))
#             print('  CNN features:',
#                 sum(p.numel() for p in self._features.parameters()))
#             print('  Temporal:',
#                 sum(p.numel() for p in self._pred_fine.parameters()))
#
#     def __init__(self, num_classes, feature_arch, temporal_arch, clip_len,
#                  modality,device = 'cuda', multi_gpu=False):
#         self.device = device
#         self._multi_gpu = multi_gpu
#         self._model = E2EModel.Impl(
#             num_classes, feature_arch, temporal_arch, clip_len, modality)
#         self._model.print_stats()
#
#         if multi_gpu:
#             self._model = nn.DataParallel(self._model)
#
#         self._model.to(device)
#         self._num_classes = num_classes
#
#     def forward(self, frame):
#         pred, total_feamap, total_feature = self._model(frame)
#
#         return pred, total_feamap, total_feature
#
#
#
#     # def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None,
#     #           acc_grad_iter=1, fg_weight=5):
#     #     if optimizer is None:
#     #         self._model.eval()
#     #     else:
#     #         optimizer.zero_grad()
#     #         self._model.train()
#     #
#     #     ce_kwargs = {}
#     #     if fg_weight != 1:
#     #         ce_kwargs['weight'] = torch.FloatTensor(
#     #             [1] + [fg_weight] * (self._num_classes - 1)).to(self.device)
#     #
#     #     epoch_loss = 0.
#     #     with torch.no_grad() if optimizer is None else nullcontext():
#     #         for batch_idx, batch in enumerate(tqdm(loader)):
#     #             # print(batch['frame'].size(),batch['label'].size(),batch['mix_weight'].size())
#     #             frame = loader.dataset.load_frame_gpu(batch, self.device)
#     #             label = batch['label'].to(self.device)
#     #             # print(batch['mix_weight'])
#     #             # print(self.device, frame.size(),label.size())
#     #             # print(label[0])
#     #
#     #             # Depends on whether mixup is used
#     #             label = label.flatten() if len(label.shape) == 2 \
#     #                 else label.view(-1, label.shape[-1])
#     #
#     #             with torch.cuda.amp.autocast():
#     #                 pred = self._model(frame)
#     #
#     #                 loss = 0.
#     #                 if len(pred.shape) == 3:
#     #                     pred = pred.unsqueeze(0)
#     #
#     #                 for i in range(pred.shape[0]):
#     #                     loss += F.cross_entropy(
#     #                         pred[i].reshape(-1, self._num_classes), label,
#     #                         **ce_kwargs)
#     #
#     #             if optimizer is not None:
#     #                 step(optimizer, scaler, loss / acc_grad_iter,
#     #                      lr_scheduler=lr_scheduler,
#     #                      backward_only=(batch_idx + 1) % acc_grad_iter != 0)
#     #
#     #             epoch_loss += loss.detach().item()
#     #
#     #     return epoch_loss / len(loader)     # Avg loss
#     #
#     # def predict(self, seq, use_amp=True):
#     #     if not isinstance(seq, torch.Tensor):
#     #         seq = torch.FloatTensor(seq)
#     #     if len(seq.shape) == 4: # (L, C, H, W)
#     #         seq = seq.unsqueeze(0)
#     #     if seq.device != self.device:
#     #         seq = seq.to(self.device)
#     #
#     #     self._model.eval()
#     #     with torch.no_grad():
#     #         with torch.cuda.amp.autocast() if use_amp else nullcontext():
#     #             pred = self._model(seq)
#     #         if isinstance(pred, tuple):
#     #             pred = pred[0]
#     #         if len(pred.shape) > 3:
#     #             pred = pred[-1]
#     #         pred = torch.softmax(pred, axis=2)
#     #         pred_cls = torch.argmax(pred, axis=2)
#     #         return pred_cls.cpu().numpy(), pred.cpu().numpy()
#     #


