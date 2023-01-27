import torch
import torchvision


imagenet_data = torchvision.datasets.ImageNet('/home/krojerb/scratch/imagenet')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,)
