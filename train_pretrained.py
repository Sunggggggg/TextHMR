import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from lib.pre_train.model import Model
from lib.core.loss_pretrain import *
from lib.dataset._motion_dataset import MotionDataset3D
from lib.utils.utils import AverageMeter

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--exp_name', type=str, default='pre_train')
    parser.add_argument('--exp_root', type=str, default='./pre_trained_experiment/')
    parser.add_argument('--data_root', type=str, default='/mnt/SKY/AMASS_proc/processed_16frames/')
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--subset_list', type=list, default=['HUMAN4D' ,'KIT', 'ACCAD', 'BioMotionLab_NTroje'])
    parser.add_argument('--lambda_3d_velocity', type=float, default=20.0)
    parser.add_argument('--lambda_scale', type=float, default=0.5)
    parser.add_argument('--lambda_text', type=float, default=10.0)

    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(args):
    # 
    exp_dir = os.path.join(args.exp_root, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print('Make dir', exp_dir)

    ### Dataset
    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 4,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
        }
    num_motions = train_dataset.num_motions
    train_loader_3d = DataLoader(train_dataset, **trainloader_params)
    print('Load dataset #of motion :', num_motions)

    ### Model
    model = Model(num_total_motion=num_motions)
    model.to('cuda')
    
    ### Optim
    net_params = sum(map(lambda x: x.numel(), model.parameters()))
    optimizer = torch.optim.AdamW(lr=0.0001, params=model.parameters(), weight_decay=0.9)
    print('Build model #of param. :', net_params)

    losses_total = AverageMeter()
    losses_3d_pos = AverageMeter()
    losses_3d_scale = AverageMeter()
    losses_3d_velocity = AverageMeter()
    losses_text = AverageMeter()

    ### Train
    train_3d_iter = iter(train_loader_3d)
    model.train()
    for i in range(args.epoch):
        for itr in tqdm(range(1000)):
            try:
                target_3d = next(train_3d_iter)
            except StopIteration:
                train_3d_iter = iter(train_loader_3d)
                target_3d = next(train_3d_iter)
            
            (motion_2d, inp_text), (motion_3d, gt_class) = target_3d
            motion_2d = motion_2d.cuda()
            inp_text = inp_text.cuda()
            motion_3d = motion_3d.cuda()        # root relative
            gt_class = gt_class.cuda()

            pred_text, pred_kp_3d = model(motion_2d[..., :2], inp_text)
            
            ### loss
            loss_3d_pos = loss_mpjpe(pred_kp_3d, motion_3d)
            loss_3d_scale = n_mpjpe(pred_kp_3d, motion_3d)
            loss_3d_velocity = loss_velocity(pred_kp_3d, motion_3d)
            loss_lv = loss_limb_var(pred_kp_3d)
            loss_lg = loss_limb_gt(pred_kp_3d, motion_3d)
            loss_a = loss_angle(pred_kp_3d, motion_3d)
            loss_av = loss_angle_velocity(pred_kp_3d, motion_3d)
            loss_text = loss_cross_entropy(pred_text, gt_class)

            loss_total = loss_3d_pos + (args.lambda_scale * loss_3d_scale) + (args.lambda_3d_velocity * loss_3d_velocity)\
                        + (args.lambda_text *loss_text)

            optimizer.zero_grad()           
            loss_total.backward()
            optimizer.step()

            losses_total.update(loss_total, motion_2d.size(0))
            losses_3d_pos.update(loss_3d_pos, motion_2d.size(0))
            losses_3d_scale.update(args.lambda_scale * loss_3d_scale, motion_2d.size(0))
            losses_3d_velocity.update(args.lambda_3d_velocity * loss_3d_velocity, motion_2d.size(0))
            losses_text.update(args.lambda_text *loss_text, motion_2d.size(0))

            summary_string = f'({i + 1}/{args.epoch}) | {itr}/1000 | loss: {losses_total.avg:.2f} ' \
                                f'| 3d: {losses_3d_pos.avg:.2f} | 3d_scale: {losses_3d_scale.avg:.2f} | 3d_vel: {losses_3d_velocity.avg:.2f} ' \
                                f'| text: {losses_text.avg:.2f} '
            print(summary_string)

            if i % 5 == 0 :
                save_dict = {
                    'epoch': i,
                    'gen_state_dict': model.state_dict(),
                    'gen_optimizer': optimizer.state_dict(),
                }
                filename = os.path.join(exp_dir, f'Epoch{i}_checkpoint.pth.tar')
                torch.save(save_dict, filename)

    
    


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(123)
    main(args)