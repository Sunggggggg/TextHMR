# The code is largely borrowd from https://github.com/mkocabas/VIBE
# Adhere to their licence to use this script
import os
import os.path as osp
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import pprint
import random
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.backends.cudnn as cudnn
from lib.dataset._motion_dataset import read_pkl
from lib.core.config import parse_args
from lib.utils.utils import prepare_output_dir, create_logger, get_optimizer
from lib.dataset._loaders_only3d import get_data_loaders
from lib.utils.lr_scheduler import CosineAnnealingWarmupRestarts

from lib.core.loss_TextHMR_v4 import Loss
from lib.core.trainer_TextHMR_v4 import Trainer
from lib.TextHMR.model_v4 import Model

def main(cfg):
    if cfg.SEED_VALUE >= 0:
        print(f'Seed value for the experiment {cfg.SEED_VALUE}')
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)
        torch.cuda.manual_seed(cfg.SEED_VALUE)
        torch.cuda.manual_seed_all(cfg.SEED_VALUE)
        
    logger = create_logger(cfg.LOGDIR, phase='train')

    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    logger.info(pprint.pformat(cfg))

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # ========= Dataloaders ========= #
    data_loaders = get_data_loaders(cfg)

    # ========= Textloader ========= #
    # text_embeds = read_pkl(os.path.join(cfg.TEXT.data_root, 'total_description_embedding.pkl'))
    # num_motion = len(text_embeds)

    # ========= Compile Loss ========= #
    loss = Loss(
        e_loss_weight=cfg.LOSS.KP_2D_W,
        e_3d_loss_weight=cfg.LOSS.KP_3D_W,
        e_pose_loss_weight=cfg.LOSS.POSE_W,
        e_shape_loss_weight=cfg.LOSS.SHAPE_W,
        vel_or_accel_2d_weight = cfg.LOSS.vel_or_accel_2d_weight,
        vel_or_accel_3d_weight = cfg.LOSS.vel_or_accel_3d_weight
    )

    # ========= Model ========= #
    model = Model().to(cfg.DEVICE)
    logger.info(f'net: {model}')

    if cfg.TRAIN.PRETRAINED :
        pretrained_dict = torch.load(cfg.TRAIN.PRETRAINED)['gen_state_dict']

        model.load_state_dict(pretrained_dict)
        print(f'=> loaded pretrained model from \'{cfg.TRAIN.PRETRAINED}\'')

    # ========= Optimizer, Scheduler ========= #
    net_params = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info(f'params num: {net_params}')
    gen_optimizer = get_optimizer(
        model=model,
        optim_type=cfg.TRAIN.GEN_OPTIM,
        lr=cfg.TRAIN.GEN_LR,
        weight_decay=cfg.TRAIN.GEN_WD,
        momentum=cfg.TRAIN.GEN_MOMENTUM,
    )
    
    lr_scheduler = CosineAnnealingWarmupRestarts(
        gen_optimizer,
        first_cycle_steps = cfg.TRAIN.END_EPOCH,
        max_lr=cfg.TRAIN.GEN_LR,
        min_lr=cfg.TRAIN.GEN_LR * 0.1,
        warmup_steps=cfg.TRAIN.LR_PATIENCE,
    )

    # ========= Start Training ========= #
    Trainer(
        cfg=cfg,
        data_loaders=data_loaders,
        generator=model,
        criterion=loss,
        gen_optimizer=gen_optimizer,
        lr_scheduler=lr_scheduler,
        val_epoch=cfg.TRAIN.val_epoch
    ).fit()

if __name__ == '__main__':
    cfg, cfg_file, _ = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)

    main(cfg)