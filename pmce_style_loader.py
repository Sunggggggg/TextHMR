from lib.core.config import cfg

from torch.utils.data import DataLoader
from data.multiple_datasets import MultipleDatasets

import data.Human36M.dataset as Human36M
import data.COCO.dataset as COCO
import data.PW3D.dataset as PW3D
import data.MPII3D.dataset as MPII3D
import data.MPII.dataset as MPII

def get_dataloader(args, dataset_names, is_train):
    dataset_split = 'TRAIN' if is_train else 'TEST'
    batch_per_dataset = cfg[dataset_split].batch_size // len(dataset_names)
    dataset_list, dataloader_list = [], []

    print(f"==> Preparing {dataset_split} Dataloader...")
    for name in dataset_names:
        dataset = eval(f'{name}')(dataset_split.lower(), args=args)
        print("# of {} {} data: {}".format(dataset_split, name, len(dataset)))
        dataloader = DataLoader(dataset,
                                batch_size=batch_per_dataset,
                                shuffle=cfg[dataset_split].shuffle,
                                num_workers=cfg.DATASET.workers,
                                pin_memory=False)
        dataset_list.append(dataset)
        dataloader_list.append(dataloader)

    if not is_train:
        return dataset_list, dataloader_list
    else:
        trainset_loader = MultipleDatasets(dataset_list, make_same_len=True)
        batch_generator = DataLoader(dataset=trainset_loader, \
                          batch_size=batch_per_dataset * len(dataset_names), \
                          shuffle=cfg[dataset_split].shuffle, \
                          num_workers=cfg.DATASET.workers, pin_memory=False)
        return dataset_list, batch_generator