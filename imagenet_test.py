import argparse
import os
import random
import time

import torch
import torchvision
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from torch.utils.data import Subset
print('imports done!')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_path', metavar='DIR', default='./datasets/imagenet',
                    help='path to dataset (default: /home/krojerb/scratch/imagenet/)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
args = parser.parse_args()



print('creating data loader')
valdir = os.path.join(args.data_path, 'val')

# TODO: see why this is necessary and perhaps we don't need it?
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

val_dataset = datasets.ImageFolder(valdir,)
    # transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     normalize,
    # ]))
# print(val_dataset)
# print(val_dataset.class_to_idx)

for i in range(60):
    print(i)
    img, target = val_dataset[i]
    print(f"type of img: {type(img)}")
    print(f"type of target: {type(target)}")
    print(f"target:\n{target}")
    # compute output

# val_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=args.batch_size, shuffle=False,
#         pin_memory=True, sampler=None)

# print('iterating though dataloader')
# for i, (images, target) in enumerate(val_loader):
#     if torch.cuda.is_available():
#         images = images.cuda(args.gpu, non_blocking=True)
#         target = target.cuda(args.gpu, non_blocking=True)
#     print(f"type of images: {type(images)}")
#     print(f"shape of images: {images.shape}")
#     print(f"shape of images[0]: {images[0].shape}")
#     print(f"images[0]:\n{images[0]}")
#     print(f"type of target: {type(target)}")
#     print(f"shape of target: {target.shape}")
#     print(f"target:\n{target}")
#     # compute output  
#     break     