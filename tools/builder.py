import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch
import torch.optim as optim
from models import I3D_backbone
from models.PS import PSNet
from utils.misc import import_class
from torchvideotransforms import video_transforms, volume_transforms
from models import decoder_fuser
from models import MLP_score
from models.E2Emodel import E2EModel,SpotModel,EnhModel


def get_video_trans():
    train_trans = video_transforms.Compose([
        video_transforms.RandomHorizontalFlip(),
        video_transforms.Resize((200, 112)),
        video_transforms.RandomCrop(112),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_trans = video_transforms.Compose([
        video_transforms.Resize((200, 112)),
        video_transforms.CenterCrop(112),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_trans, test_trans


def dataset_builder(args):
    train_trans, test_trans = get_video_trans()
    Dataset = import_class("datasets." + args.benchmark)
    if args.test:
        test_dataset = Dataset(args, transform=test_trans, subset='test')
        return test_dataset
    else:
        train_dataset = Dataset(args, transform=train_trans, subset='train')
        test_dataset = Dataset(args, transform=test_trans, subset='test')
        return train_dataset, test_dataset



def model_builder(args):
    feat_dim = 368  # rny002 368  rny008/convnextt 768
    classes = 2 # 一共2次切换

    base_model = E2EModel(args.feature_arch, clip_len=args.clip_len, modality=args.modality,)
    PSNet_model = SpotModel(classes, args.temporal_arch, feat_dim=feat_dim )
    # print(PSNet_model)
    # base_model = I3D_backbone(I3D_class=400)
    # base_model.load_pretrain(args.pretrained_i3d_weight)
    # PSNet_model = PSNet(n_channels=9)
    EnhNet_model = EnhModel(feat_dim=feat_dim, hidden_dim=feat_dim//2, num_layers=3)
    Decoder_vit = decoder_fuser(dim=feat_dim, num_heads=8, num_layers=3)
    Regressor_delta = MLP_score(in_channel=feat_dim, out_channel=1)
    return EnhNet_model,base_model,PSNet_model, Decoder_vit, Regressor_delta

def build_opti_sche(EnhNet_model,base_model,psnet_model, decoder, regressor_delta, args):
    if args.optimizer == 'Adam':
        optimizer = optim.Adam([
            {'params': EnhNet_model.parameters()},
            {'params': base_model.parameters(), 'lr': args.base_lr * args.lr_factor},
            {'params': psnet_model.parameters()},
            {'params': decoder.parameters()},
            {'params': regressor_delta.parameters()}
        ], lr=args.base_lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError()

    scheduler = None
    return optimizer, scheduler


def resume_train(base_model, psnet_model, decoder, regressor_delta, optimizer, args):
    ckpt_path = os.path.join(args.experiment_path, 'last.pth')
    if not os.path.exists(ckpt_path):
        print('no checkpoint file from path %s...' % ckpt_path)
        return 0, 0, 0, 1000, 1000
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # parameter resume of base model
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    psnet_ckpt = {k.replace("module.", ""): v for k, v in state_dict['psnet_model'].items()}
    psnet_model.load_state_dict(psnet_ckpt)

    decoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['decoder'].items()}
    decoder.load_state_dict(decoder_ckpt)

    regressor_delta_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor_delta'].items()}
    regressor_delta.load_state_dict(regressor_delta_ckpt)

    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

    # parameter
    start_epoch = state_dict['epoch'] + 1
    epoch_best_aqa = state_dict['epoch_best_aqa']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']

    return start_epoch, epoch_best_aqa, rho_best, L2_min, RL2_min


def load_model(EnhNet_model,base_model, psnet_model, decoder, regressor_delta, args):
    ckpt_path = args.ckpts
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path,map_location='cpu')

    # parameter resume of base model
    # EnhNet_ckpt = {k.replace("module.", ""): v for k, v in state_dict['EnhNet_model'].items()}
    # EnhNet_model.load_state_dict(EnhNet_ckpt)
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)
    psnet_model_ckpt = {k.replace("module.", ""): v for k, v in state_dict['psnet_model'].items()}
    psnet_model.load_state_dict(psnet_model_ckpt)
    decoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['decoder'].items()}
    decoder.load_state_dict(decoder_ckpt)
    regressor_delta_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor_delta'].items()}
    regressor_delta.load_state_dict(regressor_delta_ckpt)

    epoch_best_aqa = state_dict['epoch_best_aqa']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']
    print('ckpts @ %d epoch(rho = %.4f, L2 = %.4f , RL2 = %.4f)' % (epoch_best_aqa - 1, rho_best,  L2_min, RL2_min))
    return