import argparse
import os
import random

import torch
import torchvision
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data_path', metavar='DIR', default='/home/krojerb/scratch/imagenet/',
                    help='path to dataset (default: /home/krojerb/scratch/imagenet/)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')

imagenet_val_data = torchvision.datasets.ImageNet('/home/krojerb/scratch/imagenet/val_data')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,)

valdir = os.path.join(args.data, 'val_data')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

val_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, sampler=None)