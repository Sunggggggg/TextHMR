#!/bin/bash

python train_pretrained.py --exp_name coco_joint_64 --data_root /mnt/SKY/AMASS_proc/processed_64frames_coco_16_fps10/ --batch_size 8 --seqlen 64
python train_v2.py --cfg configs/repr_table1_3dpw.yaml


python train_pretrained.py --exp_name coco_joint_16_proposed --data_root /mnt/SKY/AMASS_proc/processed_16frames_coco_16_fps10/ --batch_size 8 --seqlen 16
#python train_pretrained.py --exp_name coco_joint_64_KIT --data_root /mnt/SKY/AMASS_proc/processed_64frames_coco_16_fps10_KIT/ --batch_size 8 --seqlen 64