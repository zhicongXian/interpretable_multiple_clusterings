
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

def get_pt_data(dl, class_dict):
    pt_data = []
    pt_labels = []
    for batch in dl:
        img, label = batch
        pt_data.append(img)
        for label_i in label.tolist():
            cl = class_dict[label_i].split("_")
            cl = torch.tensor([int(cl[0]), int(cl[1])])
            pt_labels.append(cl)
    pt_data = torch.cat(pt_data, dim=0)
    pt_labels = torch.stack(pt_labels)
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

class GrayscaleAutoencoderDataset(Dataset):
    def __init__(self, root, transform=None):
        extensions = {
            ".png", ".jpg", ".jpeg", ".bmp",
            ".tif", ".tiff", ".webp",
        }
        self.files = sorted(
            path
            for path in Path(root).rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )

        if not self.files:
            raise ValueError(f"No images found in {root}")

        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            image = image.convert("L")  # Grayscale

        x = self.transform(image)      # Shape: [1, H, W]
        return x, x.clone()             # Same image as target




def load_cmnist(args):

    src_path = args.dataset_path # "C:\\Users\\zhicong\\Documents\\Data\\enrc_data\\enrc_data\\nr_objects"
    # args.dataset_path = clevr_dir
    image_mode = args.image_mode
    use_gpu = True
    bs = 64
    stats_bs = 64
    image_size = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats_dataset = GrayscaleAutoencoderDataset(root=src_path)

    # All images are used for training, matching IndexSplitter([]).




    # All images must have equal dimensions for batching.
    preprocessing = []
    if image_size is not None:
        preprocessing.append(
            transforms.Resize(image_size, antialias=True)
        )

    base_transform = transforms.Compose(preprocessing)

    # Compute statistics without normalization or random augmentations.

    # --- Remove calculating the mean and std by himself ##########
    # stats_loader = DataLoader(
    #     stats_dataset,
    #     batch_size=stats_bs,
    #     shuffle=False,
    #     num_workers=0,
    #     drop_last=False,
    #     pin_memory=(device.type == "cuda"),
    # )
    #
    # mean, std = channel_stats(stats_loader, device)
    # stats = [mean, std]
    #
    #
    # # This is a good important discovery !!!
    # mean = torch.as_tensor(stats[0]).detach().cpu().reshape(-1).tolist()
    # std = torch.as_tensor(stats[1]).detach().cpu().reshape(-1).tolist()
    ############################
    std = (0.2274,)
    mean = (0.0653,)

    tfms = [
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
        transforms.RandomPerspective(
            distortion_scale=0.2,
            p=0.75,
            interpolation=InterpolationMode.BILINEAR,
        ),
    ]

    transform = transforms.Compose([
        # Uncomment if your images have different sizes:
        # transforms.Resize((128, 128)),
        transforms.ToTensor(),
        *tfms,
        transforms.Normalize(mean=mean, std=std),
    ])

    pretrain_dataset = GrayscaleAutoencoderDataset(
        root=src_path,
        transform=transform,
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # # for train datasets:
    # std = (0.2274,)
    # mean = (0.0653,)
    # now the training transform

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    trainset = datasets.ImageFolder(
        root=src_path,
        transform=transform,
    )
    preprocessing_dataloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              drop_last=False, num_workers=0,   pin_memory=(device.type == "cuda"))


    classes = trainset.classes
    class_dict = {i: c_i for i, c_i in enumerate(classes)}
    pt_data, pt_labels = get_pt_data(preprocessing_dataloader, class_dict=class_dict)

    base_dataset = torch.utils.data.TensorDataset(*(pt_data, pt_labels))
    tensor_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(mean=mean, std=std),
    ])
    train_dataset = TransformDataset(base_dataset ,tensor_transform)

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
    parser.add_argument("--dataset", type=str, default="cmnist")
    parser.add_argument("--dataset-path", type=Path, default=r"C:\Users\erikc\Documents\Data\enrc_data\enrc_data\tmp_cmnist")
    parser.add_argument("--image-mode", type=str, default="RGB")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pretrain_loader, trainloader, testloader = load_cmnist(args)

    for data, label in trainloader:
        print(f"data: {data}, label: {label}")


    for data, label in pretrain_loader:
        print(f"data: {data}, label: {label}")

    for data, label in testloader:
        print(f"data: {data}, label: {label}")