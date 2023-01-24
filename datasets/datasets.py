import json
import os
import random
from torch.utils.data import Dataset
from PIL import Image

def get_dataset(dataset_name, root_dir, split='valid', resize=512):
    if dataset_name == 'winoground':
        return WinogroundDataset(root_dir, resize=resize)
    elif dataset_name == 'imagecode':
        return ImageCoDeDataset(root_dir, split, resize=resize)
    elif daataset_name == 'flickr30k':
        return Flickr30KDataset(root_dir, split, resize=resize)
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

class WinogroundDataset(Dataset):
    def __init__(self, root_dir, resize=512):
        self.root_dir = root_dir
        self.data = json.load(open(f'{root_dir}/data.json', 'r'))
        self.resize = resize

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
        img0 = img0.resize((self.resize, self.resize))
        img1 = img1.resize((self.resize, self.resize))

        imgs = [img0, img1]
        text = [cap0, cap1]

        return imgs, text

class ImageCoDeDataset(Dataset):
    def __init__(self, root_dir, split, resize=512):
        self.root_dir = root_dir
        self.resize = resize
        dataset = self.load_data(root_dir, split)

    @staticmethod
    def load_data(data_dir, split, video_only=False):
        with open(f'{data_dir}/{split}_data.json') as f:
            json_file = json.load(f)
        img_path = f'{data_dir}/image-sets'

        dataset = []
        for img_dir, data in json_file.items():
            img_files = list((Path(f'{img_path}/{img_dir}')).glob('*.jpg'))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            for img_idx, text in data.items():
                static = 'open-images' in img_dir
                if video_only:
                    if not static:
                        dataset.append((img_dir, img_files, int(img_idx), text))
                else:
                    dataset.append((img_dir, img_files, int(img_idx), text))
        
        return dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img_dir, img_files, img_idx, text = self.dataset[idx]
        imgs = [Image.open(img_path).convert("RGB").resize(self.resize, self.resize) for img_path in img_files]

        return imgs, [text], img_dir, img_idx

class Flickr30KDataset(Dataset):
    def __init__(self, root_dir, split, resize=512):
        self.root_dir = root_dir
        self.resize = resize
        self.data = json.load(open(f'{root_dir}/top10_RN50x64.json', 'r'))
        self.data = list(self.data.items())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        text = ex[0]
        correct_path = f'datasets/{ex[1][0]}'
        img_paths = f'datasets/{ex[1][1]}'
        img_idx = img_paths.index(correct_path)
        imgs = [Image.open(img_path).convert("RGB").resize(self.resize, self.resize) for img_path in img_paths]
        
        return imgs, [text], img_idx