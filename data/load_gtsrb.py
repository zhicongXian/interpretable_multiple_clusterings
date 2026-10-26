

import json
import os
import pickle
from PIL import Image
import shutil
from collections import Counter
from torch.utils.data import Dataset
import numpy as np

import torch
from collections import OrderedDict
from torchvision import datasets
import torchvision.transforms as transforms
import argparse

"""
Code adapted from https://github.com/mesnico/RelationNetworks-CLEVR/edit/master/clevr_dataset_connector.py
to parse the json files easily.
"""


from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
def clean_labels(labels):
    l = np.array(labels)
    print("Labels: ")
    for col_idx in range(l.shape[1]):
        label_list = l[:, col_idx].tolist()
        unique_labels = list(set(label_list))
        unique_labels.sort()
        print(unique_labels)
        mapping = {}
        for label_idx, label_i in enumerate(unique_labels):
            mapping[label_i] = label_idx

        l[:, col_idx] = [mapping[i] for i in label_list]
    l = l.astype(np.int64)
    return l


def get_nr_labels(labels):
    labels_nr = []
    for l_i in labels:
        if l_i == 0:
            labels_nr.append(["03_speed_limit_70", "red_white_black"])
        elif l_i == 1:
            labels_nr.append(["09", "red_white_black"])
        elif l_i == 2:
            labels_nr.append(["35_ahead_only", "blue_white"])
        elif l_i == 3:
            labels_nr.append(["38_keep_right", "blue_white"])
    return labels_nr
def get_pt_data(dl):
    pt_data = []
    pt_labels = []
    for batch in dl:
        img, label = batch
        pt_data.append(img)
        pt_labels.append(label)
    pt_data = torch.cat(pt_data, dim=0).view((-1, 3, 32, 32))
    pt_labels = torch.cat(pt_labels)
    pt_labels = get_nr_labels(pt_labels)
    pt_labels = clean_labels(pt_labels)
    return pt_data, pt_labels

class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

class AutoencoderDataset(Dataset):
    def __init__(
        self,
        files,
        image_transform=None,
        normalize=None,
        image_mode="RGB",
    ):
        self.files = list(files)
        self.image_transform = image_transform
        self.normalize = normalize
        self.image_mode = image_mode
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            image = image.convert(self.image_mode)

        # Converts to [C, H, W], with values in [0, 1].
        image = self.to_tensor(image)

        # Apply augmentation only once so input and target stay aligned.
        if self.image_transform is not None:
            image = self.image_transform(image)

        target = image.clone()
        x = self.normalize(image) if self.normalize is not None else image

        return x, target


@torch.no_grad()
def channel_stats(dl, device):
    """Streaming per-channel statistics for [B, C, H, W] batches."""
    count = 0
    mean = None
    m2 = None

    for xb, _ in dl:
        x = xb.to(device=device, dtype=torch.float64)

        n = x.shape[0] * x.shape[2] * x.shape[3]
        batch_var, batch_mean = torch.var_mean(
            x, dim=(0, 2, 3), correction=0
        )
        batch_m2 = batch_var * n

        if mean is None:
            mean = batch_mean
            m2 = batch_m2
            count = n
        else:
            total = count + n
            delta = batch_mean - mean

            m2 = (
                m2
                + batch_m2
                + delta.square() * (count * n / total)
            )
            mean = mean + delta * (n / total)
            count = total

    if count <= 1:
        raise ValueError("At least two pixels per channel are required.")

    # Matches torch.std(..., correction=1).
    std = (m2 / (count - 1)).clamp_min(0).sqrt()
    std = torch.where(std == 0, torch.ones_like(std), std)

    return mean.float().cpu(), std.float().cpu()



def load_gtsrb(args):

    src_path = args.dataset_path #"C:\\Users\\zhicong\\Documents\\Data\\enrc_data\\enrc_data\\nr_objects"
    # args.dataset_path = clevr_dir
    image_mode = args.image_mode
    use_gpu = True
    bs = 64
    stats_bs = 64
    image_size = None

    # def create_dataloader(
    #         src_path,
    #         bs=16,
    #         stats_bs=64,
    #         image_size=None,
    #         augmentations=None,
    #         use_gpu=True,
    #         image_mode="RGB",
    # ):
    device = torch.device(
        "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    )

    # Recursively collect PNG files.
    files = sorted(
        path
        for path in Path(src_path).rglob("*")
        if path.is_file() and path.suffix.lower() == ".png"
    )

    if not files:
        raise ValueError(f"No PNG images found in {src_path}")

    # All images must have equal dimensions for batching.
    preprocessing = []
    if image_size is not None:
        preprocessing.append(
            transforms.Resize(image_size, antialias=True)
        )

    base_transform = transforms.Compose(preprocessing)

    # Compute statistics without normalization or random augmentations.
    stats_dataset = AutoencoderDataset(
        files,
        image_transform=base_transform,
        image_mode=image_mode,
    )

    stats_loader = DataLoader(
        stats_dataset,
        batch_size=stats_bs,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    mean, std = channel_stats(stats_loader, device)
    stats = [mean, std]


    # This is a good important discovery !!!
    augmentations = [
        transforms.RandomApply(
            [transforms.RandomRotation(
                degrees=10,
                interpolation=InterpolationMode.BILINEAR,
            )],
            p=0.75,
        ),
        transforms.RandomApply(
            [transforms.RandomAffine(
                degrees=0,
                scale=(1.0, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            )],
            p=0.75,
        ),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.2)],
            p=0.75,
        ),
        transforms.RandomApply(
            [transforms.ColorJitter(contrast=0.2)],
            p=0.75,
        ),
    ]

    train_transforms = preprocessing + (
        list(augmentations) if augmentations is not None else []
    )

    pretrain_dataset = AutoencoderDataset(
        files,
        image_transform=transforms.Compose(train_transforms),
        normalize=transforms.Normalize(
            mean=mean.tolist(),
            std=std.tolist(),
        ),
        image_mode=image_mode,
    )

    # for train datasets:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

    trainset = datasets.ImageFolder(
        root=src_path,
        transform=transform,
        is_valid_file=lambda path: path.lower().endswith(".png"),
    )

    helperloader = DataLoader(
        trainset,
        batch_size=bs,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    # All files are used for training, matching IndexSplitter([]).
    # data = DataLoader(
    #     train_dataset,
    #     batch_size=bs,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=False,
    #     pin_memory=(device.type == "cuda"),
    # )
    pt_data, pt_labels = get_pt_data(helperloader)
    base_dataset = torch.utils.data.TensorDataset(*(pt_data, torch.from_numpy(pt_labels)))
    train_dataset = TransformDataset(base_dataset,transform)

    pretrainloader = torch.utils.data.DataLoader(pretrain_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             pin_memory=(device.type == "cuda"),
                                             num_workers=0)

    testloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             pin_memory=(device.type == "cuda"),
                                             num_workers=0)



    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              drop_last=False, num_workers=0,   pin_memory=(device.type == "cuda"))

    return pretrainloader, trainloader, testloader

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--image-mode", type=str, default="RGB")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pretrain_loader, trainloader, testloader = load_gtsrb(args)

    for data, label in trainloader:
        print(f"data: {data}, label: {label}")
