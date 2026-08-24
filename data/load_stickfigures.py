import numpy as np
import os
from utils.general_utils import setup_directory
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets.folder import pil_loader
import torch
from torch.utils.data import TensorDataset
def get_pt_data(dl):
    pt_data = []
    for batch in dl:
        img, label = batch
        pt_data.append(img)
    pt_data = torch.cat(pt_data, dim=0)
    return pt_data

def load_stickfigures(args):



    # Dataset statistics
    data_dir = args.dataset_path
    std = (0.3332, )
    mean = (0.2585,)
    bs = args.batch_size

    def to_grey(x):
        return x.convert('L')

    transform = transforms.Compose([to_grey,
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean,
                                                         std),
                                    ])

    np_data = np.loadtxt(os.path.join(
        data_dir, 'stickfigures_3sub.data'), delimiter=";")

    np_labels = np_data[:, [0, 1, 2]].astype(np.int64)
    trainset = datasets.DatasetFolder(
        root=data_dir, loader=pil_loader, extensions=(".png",), transform=transform)
    testloader = torch.utils.data.DataLoader(
        trainset, batch_size=bs, shuffle=False, drop_last=False)


    # match data and labels
    pt_data = get_pt_data(testloader)
    pt_labels_idx = [int(i[0].split("//")[-1].split("_")[-1].strip(".png"))
                     for i in trainset.samples]
    pt_labels = np_labels[pt_labels_idx, :]
    pt_labels = pt_labels[:, [0, 1]]

    # the dataloader do not contain labels?
    dataset = TensorDataset(pt_data, torch.from_numpy(pt_labels))
    final_train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs,shuffle=True, drop_last=False)
    final_test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=False)

    return final_train_dataloader, final_test_dataloader

if __name__ == "__main__":

    bs = 32
    n_iterations = 2000
    cluster_lr = 1e-2
    pretrain_lr = cluster_lr / 4.0  # initial lr 1e-2
    # Dataset statistics
    data_dir = "C:\\Users\\zhicong\\Documents\\Data\\enrc_data\\enrc_data\\stickfigures"
    std = (0.3332, )
    mean = (0.2585,)


    def to_grey(x):
        return x.convert('L')

    transform = transforms.Compose([to_grey,
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean,
                                                         std),
                                    ])

    np_data = np.loadtxt(os.path.join(
        data_dir, 'stickfigures_3sub.data'), delimiter=";")

    np_labels = np_data[:, [0, 1, 2]].astype(np.int64)
    trainset = datasets.DatasetFolder(
        root=data_dir, loader=pil_loader, extensions=(".png",), transform=transform)
    testloader = torch.utils.data.DataLoader(
        trainset, batch_size=bs, shuffle=False, drop_last=False)
    trainloader= testloader

    # match data and labels
    pt_data = get_pt_data(testloader)
    pt_labels_idx = [int(i[0].split("//")[-1].split("_")[-1].strip(".png"))
                     for i in trainset.samples]
    pt_labels = np_labels[pt_labels_idx, :]
    pt_labels = pt_labels[:, [0, 1]]
    print(f"pt_labels: {pt_labels}")

    dataset = TensorDataset(pt_data, torch.from_numpy(pt_labels))
    final_train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs,shuffle=True, drop_last=False)
    final_test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=False)
    for img, label in final_train_dataloader:
        print(f"label: {label}")