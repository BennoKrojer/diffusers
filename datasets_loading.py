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

def get_dataset(dataset_name, root_dir, transform=None, split='valid', resize=512, scoring_only=False, tokenizer=None):
    if dataset_name == 'winoground':
        return WinogroundDataset(root_dir, transform, resize=resize, scoring_only=scoring_only)
    elif dataset_name == 'imagecode':
        return ImageCoDeDataset(root_dir, split, transform, resize=resize, scoring_only=scoring_only)
    elif dataset_name == 'flickr30k':
        return Flickr30KDataset(root_dir, split, transform, resize=resize, scoring_only=scoring_only)
    elif dataset_name == 'lora_flickr30k':
        return LoRaFlickr30KDataset(root_dir, split, transform, resize=resize, tokenizer=tokenizer)
    elif dataset_name == 'imagenet':
        return ImagenetDataset(root_dir, transform, resize=resize, scoring_only=scoring_only)
    elif dataset_name == 'svo':
        return SVOClassificationDataset(root_dir, transform, resize=resize, scoring_only=scoring_only)
    elif dataset_name == 'clevr':
        return CLEVRDataset(root_dir, transform, resize=resize, scoring_only=scoring_only)
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
    
    def __len__(self):
        return len(self.data)

lora_train_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512) if True else transforms.RandomCrop(512),
            transforms.RandomHorizontalFlip() if True else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

class ImagenetDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False):
        self.root_dir = root_dir
        self.data = datasets.ImageFolder(root_dir + '/val')
        # self.loader = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        self.resize = resize
        self.transform = transform
        self.classes = list(json.load(open(f'./imagenet_classes.json', 'r')).values())
        if True:
            prompted_classes = []
            for c in self.classes:
                class_text = 'a photo of a ' + c
                prompted_classes.append(class_text)
            self.classes = prompted_classes
        self.scoring_only = scoring_only

    def __getitem__(self, idx):
        if not self.scoring_only:
            img, class_id = self.data[idx]
            img = img.convert("RGB")
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
        else:
            class_id = idx // 50

        if self.scoring_only:
            return self.classes, class_id
        else:
            return ([img], [img_resize]), self.classes, class_id

class WinogroundDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False):
        self.root_dir = root_dir
        self.data = json.load(open(f'{root_dir}/data.json', 'r'))
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        cap0 = ex['caption_0']
        cap1 = ex['caption_1']
        img_id = ex['id']
        img_path0 = f'{self.root_dir}/images/ex_{img_id}_img_0.png'
        img_path1 = f'{self.root_dir}/images/ex_{img_id}_img_1.png'
        if not self.scoring_only:
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
        if self.scoring_only:
            return text, img_id
        else:
            return (imgs, [img0_resize, img1_resize]), text, img_id

class ImageCoDeDataset(Dataset):
    def __init__(self, root_dir, split, transform, resize=512, scoring_only=False):
        self.root_dir = root_dir
        self.resize = resize
        self.dataset = self.load_data(root_dir, split)
        self.transform = transform
        self.scoring_only = scoring_only

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
        if not self.scoring_only:
            imgs = [Image.open(img_path).convert("RGB") for img_path in img_files]
            imgs_resize = [img.resize((self.resize, self.resize)) for img in imgs]
            imgs_resize = [diffusers_preprocess(img) for img in imgs_resize]

            if self.transform:
                imgs = [self.transform(img) for img in imgs]
            else:
                imgs = [transforms.ToTensor()(img) for img in imgs]

        if self.scoring_only:
            return text, img_dir, img_idx
        else:
            return (imgs, imgs_resize), [text], img_dir, img_idx

class Flickr30KDataset(Dataset):
    def __init__(self, root_dir, split, transform, resize=512, scoring_only=False):
        self.root_dir = root_dir
        self.resize = resize
        self.data = json.load(open(f'{root_dir}/val_top10_RN50x64.json', 'r'))
        self.data = list(self.data.items())
        self.data = self.remove_impossible(self.data)
        self.transform = transform
        self.scoring_only = scoring_only
    

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
        if not self.scoring_only:
            imgs = [Image.open(f'datasets/{img_path}').convert("RGB") for img_path in img_paths]
            imgs_resize = [img.resize((self.resize, self.resize)) for img in imgs]
            #convert pillow to numpy array
            # imgs_resize = [np.array(img) for img in imgs_resize]
            imgs_resize = [diffusers_preprocess(img) for img in imgs_resize]

            if self.transform:
                imgs = [self.transform(img) for img in imgs]
            else:
                imgs = [transforms.ToTensor()(img) for img in imgs]
        if self.scoring_only:
            return [text], img_idx
        else:
            return (imgs, imgs_resize), [text], img_idx

class LoRaFlickr30KDataset(Dataset):
    def __init__(self, root_dir, split, transform, resize=512, tokenizer=None):
        self.root_dir = root_dir
        self.resize = resize
        self.data = json.load(open(f'{root_dir}/train_top10_RN50x64.json', 'r'))
        self.data = list(self.data.items())
        self.transform = lora_train_transforms
        self.tokenizer = tokenizer
    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        text = ex[0]
        text = self.tokenizer([text], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        text = text.input_ids.squeeze(0)
        img_paths = ex[1]
        img_idx = 0
        imgs = [Image.open(img_path).convert("RGB") for img_path in img_paths]
        #convert pillow to numpy array
        # imgs_resize = [np.array(img) for img in imgs]
        imgs_resize = [img.resize((self.resize, self.resize)) for img in imgs]
        imgs_resize = [diffusers_preprocess(img) for img in imgs_resize]

        return imgs_resize, text, img_idx

class SVOClassificationDataset(Dataset):

    def __init__(self, root_dir, transform, resize=512, scoring_only=False):
        self.transform = transform
        self.root_dir = root_dir
        self.data = self.load_data(root_dir)
        self.resize = resize
        self.scoring_only = scoring_only

    def load_data(self, data_dir):
        dataset = []
        split_file = os.path.join(data_dir, 'val.json')
        with open(split_file) as f:
            json_file = json.load(f)

        for i, row in enumerate(json_file):
            pos_id = str(row['pos_id'])
            neg_id = str(row['neg_id'])
            sentence = row['sentence']
            # get two different images
            pos_file = os.path.join(data_dir, "images", pos_id)
            neg_file = os.path.join(data_dir, "images", neg_id)
            dataset.append((pos_file, neg_file, sentence))

        return dataset
    
    def __getitem__(self, idx):
        file0, file1, text = self.data[idx]
        img0 = Image.open(file0).convert("RGB")
        img1 = Image.open(file1).convert("RGB")
        if not self.scoring_only:
            imgs = [img0, img1]
            imgs_resize = [img.resize((self.resize, self.resize)) for img in imgs]
            imgs_resize = [diffusers_preprocess(img) for img in imgs_resize]
            if self.transform:
                imgs = [self.transform(img) for img in imgs]
            else:
                imgs = [transforms.ToTensor()(img) for img in imgs]
        if self.scoring_only:
            return [text], 0
        else:
            return (imgs, imgs_resize), [text], 0
        
    def __len__(self):
        return len(self.data)

class CLEVRDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False):
        root_dir = '../clevr/output'
        self.root_dir = root_dir
        subtasks = ['pair_binding_size', 'pair_binding_color', 'recognition_color', 'recognition_shape', 'spatial', 'binding_color_shape', 'binding_shape_color']
        data_ = []
        for subtask in subtasks:
            self.data = json.load(open(f'{root_dir}/{subtask}.json', 'r')).items()
            for k, v in self.data:
                for i in range(len(v)):
                    if 'subtask' == 'spatial':
                        texts = [v[i][1], v[i][2]]
                    else:
                        texts = [v[i][0], v[i][1]]
                    data_.append((k, texts))
        self.data = data_
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        cap0 = ex[1][0]
        cap1 = ex[1][1]
        img_id = ex[0]
        img_path0 = f'{self.root_dir}/images/{img_id}'
        if not self.scoring_only:
            img0 = Image.open(img_path0).convert("RGB")
            img0_resize = img0.resize((self.resize, self.resize))
            img0_resize = diffusers_preprocess(img0_resize)

            if self.transform:
                img0 = self.transform(img0)
            else:
                img0 = transforms.ToTensor()(img0)
            
            imgs = [img0]
        text = [cap0, cap1]
        if self.scoring_only:
            return text, 0
        else:
            return (imgs, [img0_resize]), text, 0
