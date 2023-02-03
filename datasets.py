import json
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import PIL
import numpy as np
import torch
from torchvision import datasets

def get_dataset(dataset_name, root_dir, transform=None, split='valid', resize=512):
    if dataset_name == 'winoground':
        return WinogroundDataset(root_dir, transform, resize=resize)
    elif dataset_name == 'imagecode':
        return ImageCoDeDataset(root_dir, split, transform, resize=resize)
    elif dataset_name == 'flickr30k':
        return Flickr30KDataset(root_dir, split, transform, resize=resize)
    elif dataset_name == 'imagenet':
        return ImagenetDataset(root_dir, transform, resize=resize)
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

def diffusers_preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = image.squeeze(0)
    return 2.0 * image - 1.0

class ImagenetDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512):
        self.root_dir = root_dir
        self.data = datasets.ImageFolder(root_dir)
        self.loader = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        self.resize = resize
        self.transform = transform
        self.classes = json.load(open(f'./imagenet_classes.json', 'r')).values()


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, class_id = self.data[idx]
        return img, self.classes, class_id

class WinogroundDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512):
        self.root_dir = root_dir
        self.data = json.load(open(f'{root_dir}/data.json', 'r'))
        self.resize = resize
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        cap0 = ex['caption_0']
        cap1 = ex['caption_1']
        img_id = ex['id']
        img_path0 = f'{self.root_dir}/images/ex_{img_id}_img_0.png'
        img_path1 = f'{self.root_dir}/images/ex_{img_id}_img_1.png'

        img0 = Image.open(img_path0).convert("RGB")
        img1 = Image.open(img_path1).convert("RGB")
        img0_resize = img0.resize((self.resize, self.resize))
        img1_resize = img1.resize((self.resize, self.resize))
        img0_resize = diffusers_preprocess(img0_resize)
        img1_resize = diffusers_preprocess(img1_resize)

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        else:
            img0 = transforms.ToTensor()(img0)
            img1 = transforms.ToTensor()(img1)
        
        imgs = [img0, img1]
        text = [cap0, cap1]

        return (imgs, [img0_resize, img1_resize]), text, img_id

class ImageCoDeDataset(Dataset):
    def __init__(self, root_dir, split, transform, resize=512):
        self.root_dir = root_dir
        self.resize = resize
        self.dataset = self.load_data(root_dir, split)
        self.transform = transform

    @staticmethod
    def load_data(data_dir, split, static_only=True):
        with open(f'{data_dir}/{split}_data.json') as f:
            json_file = json.load(f)
        img_path = f'{data_dir}/image-sets'

        dataset = []
        for img_dir, data in json_file.items():
            img_files = list((Path(f'{img_path}/{img_dir}')).glob('*.jpg'))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            for img_idx, text in data.items():
                static = 'open-images' in img_dir
                if static_only:
                    if static:
                        dataset.append((img_dir, img_files, int(img_idx), text))
                else:
                    dataset.append((img_dir, img_files, int(img_idx), text))
        
        return dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img_dir, img_files, img_idx, text = self.dataset[idx]
        imgs = [Image.open(img_path).convert("RGB") for img_path in img_files]
        imgs_resize = [img.resize((self.resize, self.resize)) for img in imgs]
        imgs_resize = [diffusers_preprocess(img) for img in imgs_resize]

        if self.transform:
            imgs = [self.transform(img) for img in imgs]
        else:
            imgs = [transforms.ToTensor()(img) for img in imgs]

        return (imgs, imgs_resize), [text], img_dir, img_idx

class Flickr30KDataset(Dataset):
    def __init__(self, root_dir, split, transform, resize=512):
        self.root_dir = root_dir
        self.resize = resize
        self.data = json.load(open(f'{root_dir}/top10_RN50x64.json', 'r'))
        self.data = list(self.data.items())
        self.data = self.remove_impossible(self.data)
        self.transform = transform
    

    @staticmethod
    def remove_impossible(data):
        new_data = []
        for caption, imgs in data:
            correct, imgs = imgs[0], imgs[1]
            if correct in imgs:
                new_data.append((caption, [correct, imgs]))
        return new_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        text = ex[0]
        correct_path = ex[1][0]
        img_paths = ex[1][1]
        img_idx = img_paths.index(correct_path)
        imgs = [Image.open(f'datasets/{img_path}').convert("RGB") for img_path in img_paths]
        imgs_resize = [img.resize((self.resize, self.resize)) for img in imgs]
        imgs_resize = [diffusers_preprocess(img) for img in imgs_resize]

        if self.transform:
            imgs = [self.transform(img) for img in imgs]
        else:
            imgs = [transforms.ToTensor()(img) for img in imgs]

        return (imgs, imgs_resize), [text], img_idx