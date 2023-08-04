import os, yaml, cv2
import json, wandb
import logging, argparse

import torch, time
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from timm import create_model
from data import create_dataset, create_dataloader

from models import MemSeg, MemoryBank
from focal_loss import FocalLoss, DiceLoss, BinaryDiceLoss
from train_utils import training
from utils import setup_default_logging
from utils import torch_seed
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scheduler import CosineAnnealingWarmupRestarts
from prodigyopt import Prodigy

import warnings
warnings.filterwarnings("ignore")


_logger = logging.getLogger('train')

def run(cfg):

    # setting seed and device
    setup_default_logging()
    torch_seed(cfg['SEED'])
    torch.set_num_threads(1)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # savedir
    cfg['EXP_NAME'] = cfg['EXP_NAME'] + f"-{cfg['DATASET']['target']}"
    savedir = os.path.join(cfg['RESULT']['savedir'], cfg['EXP_NAME'])
    os.makedirs(savedir, exist_ok=True)

    
    # wandb
    if cfg['TRAIN']['use_wandb']:
        wandb.init(name=cfg['EXP_NAME'], project='MemSeg', config=cfg)

    # build datasets
    trainset = create_dataset(
        datadir                = cfg['DATASET']['datadir'],
        target                 = cfg['DATASET']['target'], 
        train                  = True,
        resize                 = cfg['DATASET']['resize'],
        texture_source_dir     = cfg['DATASET']['texture_source_dir'],
        structure_grid_size    = cfg['DATASET']['structure_grid_size'],
        transparency_range     = cfg['DATASET']['transparency_range'],
        perlin_scale           = cfg['DATASET']['perlin_scale'], 
        min_perlin_scale       = cfg['DATASET']['min_perlin_scale'], 
        perlin_noise_threshold = cfg['DATASET']['perlin_noise_threshold']
    )

    memoryset = create_dataset(
        datadir                = cfg['DATASET']['datadir'],
        target                 = cfg['DATASET']['target'], 
        train                  = True,
        to_memory              = True,
        resize                 = cfg['DATASET']['resize'],
        texture_source_dir     = cfg['DATASET']['texture_source_dir'],
        structure_grid_size    = cfg['DATASET']['structure_grid_size'],
        transparency_range     = cfg['DATASET']['transparency_range'],
        perlin_scale           = cfg['DATASET']['perlin_scale'], 
        min_perlin_scale       = cfg['DATASET']['min_perlin_scale'], 
        perlin_noise_threshold = cfg['DATASET']['perlin_noise_threshold']
    )

    testset = create_dataset(
        datadir                = cfg['DATASET']['datadir'],
        target                 = cfg['DATASET']['target'], 
        train                  = False,
        resize                 = cfg['DATASET']['resize'],
        texture_source_dir     = cfg['DATASET']['texture_source_dir'],
        structure_grid_size    = cfg['DATASET']['structure_grid_size'],
        transparency_range     = cfg['DATASET']['transparency_range'],
        perlin_scale           = cfg['DATASET']['perlin_scale'], 
        min_perlin_scale       = cfg['DATASET']['min_perlin_scale'], 
        perlin_noise_threshold = cfg['DATASET']['perlin_noise_threshold']
    )
    
    # build dataloader
    trainloader = create_dataloader(
        dataset     = trainset,
        train       = True,
        batch_size  = cfg['DATALOADER']['batch_size'],
        num_workers = cfg['DATALOADER']['num_workers']
    )
    
    testloader = create_dataloader(
        dataset     = testset,
        train       = False,
        batch_size  = cfg['DATALOADER']['batch_size'],
        num_workers = cfg['DATALOADER']['num_workers']
    )

    inputs, masks, targets = next(iter(trainloader))
    tik = str(time.time())
    for k in range(inputs.shape[0]):
        img = inputs[k]
        img = (((img - img.min()) / (img.max() - img.min())) * 255).to(torch.uint8)
        img = img.permute(1, 2, 0)
        output_path = "./samples/DEBUG/ANOMALY"
        os.makedirs(output_path, exist_ok=True)
        if len(os.listdir(output_path)) < 21:
            img_path = f"./samples/DEBUG/ANOMALY/img_{tik}_{k}.png"
            # cv2.imwrite(img_path, cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # build feature extractor
    feature_extractor = feature_extractor = create_model(
        cfg['MODEL']['feature_extractor_name'], 
        pretrained    = True, 
        features_only = True
    ).to(device)
    ## freeze weight of layer1,2,3
    for l in ['layer1','layer2','layer3']:
        for p in feature_extractor[l].parameters():
            p.requires_grad = False

    # build memory bank
    memory_bank = MemoryBank(
        normal_dataset   = memoryset,
        nb_memory_sample = cfg['MEMORYBANK']['nb_memory_sample'],
        device           = device
    )
    ## update normal samples and save
    memory_bank.update(feature_extractor=feature_extractor)
    torch.save(memory_bank, os.path.join(savedir, f'memory_bank.pt'))
    _logger.info('Update {} normal samples in memory bank'.format(cfg['MEMORYBANK']['nb_memory_sample']))

    # build MemSeg
    model = MemSeg(memory_module=memory_bank,
                   encoder = feature_extractor
    ).to(device)

    # Set training
    l1_criterion = nn.L1Loss()
    f_criterion = FocalLoss(
        gamma = cfg['TRAIN']['focal_gamma'], 
        alpha = cfg['TRAIN']['focal_alpha']
    )
    dice_criterion = DiceLoss()

    if cfg['OPTIMIZER']['use_prodigy']:
        optimizer = Prodigy(params       = filter(lambda p: p.requires_grad, model.parameters()),
                            lr           = 1.,
                            weight_decay = cfg['OPTIMIZER']['weight_decay'],
                            )
    else:
        optimizer = torch.optim.AdamW(
            params       = filter(lambda p: p.requires_grad, model.parameters()), 
            lr           = cfg['OPTIMIZER']['lr'], 
            weight_decay = cfg['OPTIMIZER']['weight_decay'],
        )

    if cfg['SCHEDULER']['use_scheduler']:
        if cfg['SCHEDULER']['use_prodigy']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['TRAIN']['num_training_steps'])
        else:
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer, 
                first_cycle_steps = cfg['TRAIN']['num_training_steps'],
                max_lr = cfg['OPTIMIZER']['lr'],
                min_lr = cfg['SCHEDULER']['min_lr'],
                warmup_steps   = int(cfg['TRAIN']['num_training_steps'] * cfg['SCHEDULER']['warmup_ratio'])
            )
    else:
        scheduler = None

    # Fitting model
    training(
        model              = model, 
        num_training_steps = cfg['TRAIN']['num_training_steps'], 
        trainloader        = trainloader, 
        validloader        = testloader, 
        criterion          = [l1_criterion, f_criterion], 
        loss_weights       = [cfg['TRAIN']['l1_weight'], cfg['TRAIN']['focal_weight']],
        optimizer          = optimizer,
        scheduler          = scheduler,
        log_interval       = cfg['LOG']['log_interval'],
        eval_interval      = cfg['LOG']['eval_interval'],
        savedir            = savedir,
        device             = device,
        use_wandb          = cfg['TRAIN']['use_wandb']
    )


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='MemSeg Defect Detection')
    parser.add_argument('--yaml_config_dir', type=str, default='./configs', help='Config file directory')
    parser.add_argument('--object_name', type=str, default='capsule', help='Object in MVTec dataset')
    args = parser.parse_args()

    # Config
    cfg = yaml.load(open(os.path.join(args.yaml_config_dir, f'{args.object_name}.yaml'),'r'), Loader=yaml.FullLoader)

    run(cfg)
