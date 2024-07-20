import random
import pandas as pd
import os
import numpy as np
import copy
import pickle
import torch
from torch.utils.data import Dataset
from lib.data_utils._moition_utils import crop_scale_3d, crop_scale_2d, read_pkl

class Augmenter3D(object):
    """
        Make 3D augmentations when dataloaders get items. NumPy single motion version.
    """
    def __init__(self, args):
        if hasattr(args, "scale_range_pretrain"):
            self.scale_range_pretrain = args.scale_range_pretrain
        else:
            self.scale_range_pretrain = None
    
    def augment3D(self, motion_3d):
        if self.scale_range_pretrain:
            motion_3d = crop_scale_3d(motion_3d, self.scale_range_pretrain)
        return motion_3d

class MotionDataset(Dataset):
    def __init__(self, args, subset_list, data_split): # data_split: train/test
        np.random.seed(0)
        self.data_root = args.data_root # /mnt/SKY/AMASS_proc/processed/
        self.subset_list = subset_list  # ['HUMAN4D' ,'KIT', ...]
        self.data_split = data_split

        text_candidate = pd.read_csv(os.path.join(self.data_root, 'total_description.csv'), header=None)
        self.text_candidate = list(text_candidate[0][1:])
        self.text_embeds = read_pkl(os.path.join(self.data_root, 'total_description_embedding.pkl'))

        file_list_all = []
        text_dic = {}
        for subset in self.subset_list:
            data_path = os.path.join(self.data_root, subset, self.data_split)   # /mnt/SKY/AMASS_proc/processed/KIT/train
            motion_list = sorted(os.listdir(data_path))
            for i in motion_list:
                file_list_all.append(os.path.join(data_path, i))                # /mnt/SKY/AMASS_proc/processed/KIT/train/0000.pkl

            dec_path = os.path.join(self.data_root, subset, 'amass_joints_coco_60.pkl')
            text_dic[subset] = read_pkl(dec_path)['all_description']
        
        self.file_list = file_list_all
        self.text_dic = text_dic
        self.num_motions = len(self.text_candidate)
   
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_list)

    def __getitem__(self, index):
        raise NotImplementedError 

class MotionDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split):
        super(MotionDataset3D, self).__init__(args, subset_list, data_split)
        self.synthetic = True
        self.aug = Augmenter3D(args)
        self.gt_2d = False

        self.max_len = 36

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)                   # /mnt/SKY/AMASS_proc/processed/KIT/train/00000000.pkl
        motion_3d = motion_file["data_label"]               # [243, 17, 3]
        if self.data_split=="train":
            if self.synthetic or self.gt_2d:
                #motion_3d = self.aug.augment3D(motion_3d)
                motion_3d = crop_scale_3d(motion_3d, [0.5, 1.0])
                motion_2d = np.zeros(motion_3d.shape, dtype=np.float32)
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1                        # No 2D detection, use GT xy and c=1.
            elif motion_file["data_input"] is not None:     # Have 2D detection 
                motion_2d = motion_file["data_input"]
            else:
                raise ValueError('Training illegal.') 
        elif self.data_split=="test":                                           
            motion_2d = motion_file["data_input"]
            if self.gt_2d:
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1
        else:
            raise ValueError('Data split unknown.') 

        # Text
        motion_text_range = motion_file["split_id"]
        subset = file_path.split('/')[-3]

        # Padding
        idx_list = [self.text_candidate.index(self.text_dic[subset][i]) for i in motion_text_range]
        idx = np.unique(idx_list)
        text_feat = self.text_embeds[int(idx)][0]   # [1, n, 768]
        caption_len = text_feat.shape[0]
        caption_mask = np.ones((self.max_len))
        caption_mask[:caption_len] = 0.
        inp_text = np.concatenate([text_feat] + [np.zeros_like(text_feat[0:1]) for _ in range(self.max_len-caption_len)], axis=0)

        # 
        #gt_class = np.zeros((self.num_motions))
        gt_class = idx

        motion_2d = torch.from_numpy(motion_2d).float()
        inp_text = torch.from_numpy(inp_text).float()
        caption_mask = torch.from_numpy(caption_mask).bool()
        motion_3d = torch.from_numpy(motion_3d).float()
        gt_class = torch.from_numpy(gt_class).long()
        
        return (motion_2d, inp_text, caption_mask), (motion_3d, gt_class)