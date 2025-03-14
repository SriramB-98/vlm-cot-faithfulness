# %%
import functools
import cv2
import numpy as np
from torchvision.datasets import ImageFolder
import os   
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision.datasets.folder import default_loader
from torchvision.datasets import ImageNet
# from torch.utils.data import Subset
from pathlib import Path
from functools import wraps
import json

class CelebAHQ(Dataset):
    def __init__(self, root, split, label_type='all', classify=None, confounder=None, 
                 img_transform=None):
        assert os.path.isdir(root)
        self.root = root
        self.split = split
        self.names = []
        self.classify = classify
        self.confounder = confounder
        if split != "ALL":
            hq_to_orig_mapping = dict()
            orig_to_hq_mapping = dict()
            mapping_file = os.path.join(
                root, 'CelebA-HQ-to-CelebA-mapping.txt')
            assert os.path.exists(mapping_file)
            for s in open(mapping_file, 'r'):
                if '.jpg' not in s:
                    continue
                idx, _, orig_file = s.split()
                hq_to_orig_mapping[int(idx)] = orig_file
                orig_to_hq_mapping[orig_file] = int(idx)

            # load partition
            partition_file = os.path.join(root, 'list_eval_partition.txt')
            assert os.path.exists(partition_file)
            for s in open(partition_file, 'r'):
                if '.jpg' not in s:
                    continue
                orig_file, group = s.split(',')
                group = int(group)
                if orig_file not in orig_to_hq_mapping:
                    continue
                hq_id = orig_to_hq_mapping[orig_file]
                if split == "TRAIN" and group == 0:
                    self.names.append(str(hq_id))
                elif split == "VAL" and group == 1:
                    self.names.append(str(hq_id))
                elif split == "TEST" and group == 2:
                    self.names.append(str(hq_id))
                elif split == "TOY":
                    self.names.append(str(hq_id))
                    if len(self.names) >= 10:
                        break
        else:
            self.names = [
                n[:-(len('.jpg'))]
                for n in os.listdir(os.path.join(self.root, 'CelebA-HQ-img'))
                if n.endswith('.jpg')
            ]
            
        self.img_transform = img_transform

        attr_list = pd.read_csv(os.path.join(root, 'CelebAMask-HQ-attribute-anno.txt') , sep=r'\s+')
        self.attr_names = list(attr_list.columns)[1:]
        self.classify_ind = self.attr_names.index(classify)
        self.confounder_ind = self.attr_names.index(confounder)
        self.attributes = (attr_list.to_numpy()[:,1:].astype(int) > 0)
        self.label_setting = {
            'human': {
                'suffix': [
                    'neck', 'skin', 'cloth', 'l_ear', 'r_ear', 'l_brow', 'r_brow',
                    'l_eye', 'r_eye', 'nose', 'mouth', 'l_lip', 'u_lip', 'hair'
                ],
                'names': [
                    'bg', 'neck', 'face', 'cloth', 'rr', 'lr', 'rb', 'lb', 're',
                    'le', 'nose', 'imouth', 'llip', 'ulip', 'hair'
                ]
            },
            'aux': {
                'suffix': [
                    'eye_g', 'hat', 'ear_r', 'neck_l',
                ],
                'names': [
                    'normal', 'glass', 'hat', 'earr', 'neckl'
                ]
            },
            'all': {
                'suffix': [
                    'neck', 'skin', 'cloth', 'l_ear', 'r_ear', 'l_brow', 'r_brow',
                    'l_eye', 'r_eye', 'nose', 'mouth', 'l_lip', 'u_lip', 'hair',
                    'eye_g', 'hat', 'ear_r', 'neck_l',
                ],
                'names': [
                    'bg', 'neck', 'face', 'cloth', 'rr', 'lr', 'rb', 'lb', 're',
                    'le', 'nose', 'imouth', 'llip', 'ulip', 'hair',
                    'glass', 'hat', 'earr', 'neckl'
                ]
            }
        }[label_type]

        subgroup_lists = dict()
        for i, name in enumerate(self.names):
            target_index = int(self.attributes[int(name)][self.classify_ind])
            target = self.classify.lower().replace('_', ' ')
            if target_index == 0:
                target = 'not ' + target

            confounder_index = int(self.attributes[int(name)][self.confounder_ind])
            confounder = self.confounder.lower().replace('_', ' ')
            if confounder_index == 0:
                confounder = 'not ' + confounder

            if (target, confounder) not in subgroup_lists:
                subgroup_lists[(target, confounder)] = []
            subgroup_lists[(target, confounder)].append(i)

        self.subgroups = list(subgroup_lists.keys())
        self.subgroup_indices = list(subgroup_lists.values())

        for sg, sg_inds in subgroup_lists.items():
            print(f"{sg}: {len(sg_inds)}")

    def __getitem__(self, index):
        new_index = self.subgroup_indices[index%len(self.subgroups)][index//len(self.subgroups)]
        name = self.names[new_index]
        target, confounder = self.subgroups[index%len(self.subgroups)]

        image = Image.open(os.path.join(self.root, 'CelebA-HQ-img', name + '.jpg'))
        if self.img_transform:
            image = self.img_transform(image)
        
        return new_index, image, target, confounder

    def get_sample_at_index(self, index):
        name = self.names[index]
        image = Image.open(os.path.join(self.root, 'CelebA-HQ-img', name + '.jpg'))
        
        target_index = int(self.attributes[int(name)][self.classify_ind])
        target = self.classify.lower().replace('_', ' ')
        if target_index == 0:
            target = 'not ' + target

        confounder_index = int(self.attributes[int(name)][self.confounder_ind])
        confounder = self.confounder.lower().replace('_', ' ')
        if confounder_index == 0:
            confounder = 'not ' + confounder

        return image, target, confounder

    def __len__(self):
        return len(self.subgroups)*min(len(sg_inds) for sg_inds in self.subgroup_indices)

    def sample_name(self, index):
        return self.names[index]

    @property
    def label_names(self):
        return self.label_setting['names']
    
@functools.lru_cache()
def _cached_imread(fname, flags=None):
    return cv2.imread(fname, flags=flags)


class CelebAHQCorrected(CelebAHQ):
    def __init__(self, *args, **kwargs):
        self.apply_corr = kwargs.pop('apply_corr', True)
        if self.apply_corr:
            self.correction = kwargs.pop('correction')
            region = kwargs.pop('region')
        super().__init__(*args, **kwargs)
        if self.apply_corr:
            self.region_ind = self.label_setting['suffix'].index(region)
  
    def make_label(self, index, ordered_label_suffix):
        label = np.zeros((512, 512), np.uint8)
        name = self.names[index]
        name_id = int(name)
        name5 = '%05d' % name_id
        p = os.path.join(self.root, 'CelebAMask-HQ-mask-anno',
                         str(name_id // 2000), name5)
        for i, label_suffix in enumerate(ordered_label_suffix):
            label_value = i + 1
            label_fname = os.path.join(p + '_' + label_suffix + '.png')
            if os.path.exists(label_fname):
                mask = _cached_imread(label_fname, cv2.IMREAD_GRAYSCALE)
                label = np.where(mask > 0,
                                 np.ones_like(label) * label_value, label)
        return label
    
    def __getitem__(self, index):
        new_index, image, target, confounder = super().__getitem__(index)
        if self.apply_corr and confounder == 'male' and target == 'blond hair':
            mask = self.make_label(new_index, self.label_setting['suffix'])
            mask = cv2.resize(mask, image.size, interpolation=cv2.INTER_NEAREST)
            region_mask = (mask == self.region_ind + 1)
            if np.sum(region_mask) > 0:
                image = np.array(image, dtype=np.float32)
                image[region_mask] = image[region_mask] + self.correction
                image = np.clip(image, 0, 255)
                image = Image.fromarray(image.astype(np.uint8))
        return new_index, image, target, confounder

class CleanWaterbirdsDataset(ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_names = ["landbird", "waterbird"]
        class_confounders = [(self.class_names[target], path.split('.')[-2].split('_')[-1]) 
                             for path, target in self.samples]
        subgroup_lists = dict()
        for index, (class_name, confounder) in enumerate(class_confounders):
            if (class_name, confounder) not in subgroup_lists:
                subgroup_lists[(class_name, confounder)] = []
            subgroup_lists[(class_name, confounder)].append(index)
        
        self.subgroups = list(subgroup_lists.keys())
        self.subgroup_indices = list(subgroup_lists.values())
        
    def __getitem__(self, index):
        new_index = self.subgroup_indices[index%len(self.subgroups)][index//len(self.subgroups)]
        path, _ = self.samples[new_index]
        sample = self.loader(path)
        class_name, confounder = self.subgroups[index%len(self.subgroups)]
        return new_index, sample, class_name, confounder
    
    def __len__(self):
        return len(self.subgroups)*min(len(sg_inds) for sg_inds in self.subgroup_indices)
    
    def get_sample_at_index(self, index):
        path, target = self.samples[index]
        class_name = self.class_names[target]
        confounder = path.split('.')[-2].split('_')[-1]
        sample = self.loader(path)
        return sample, class_name, confounder



class SpuriousOnlyDataset(Dataset):
    def __init__(self, path, transform):
        imgs = []
        targets = []
        components = []

        subdirs = next(os.walk(path))[1]
        for subdir in subdirs:
            subdir_class = int(subdir.split('_')[2])
            subdir_component = int(subdir.split('_')[-1])
            for file in os.listdir(os.path.join(path, subdir)):
                imgs.append(os.path.join(path, subdir, file))
                targets.append(subdir_class)
                components.append(subdir_component)

        self.internal_idcs = torch.argsort(torch.LongTensor(targets))
        #print(f'SpuriousImageNet - {len(subdirs)} classes - {len(imgs)} images')
        self.transform = transform
        self.imgs = imgs
        self.targets = targets
        self.components = components
        self.included_classes = list(set(list(self.targets)))

    def get_class_component_pairings(self):
        class_components = {}
        for idx in self.internal_idcs:
            idx = idx.item()
            class_components[(self.targets[idx], self.components[idx])] = None

        return list(class_components.keys())

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        internal_idx = self.internal_idcs[idx].item()
        img = default_loader((self.imgs[internal_idx]))
        if self.transform is not None:
            img = self.transform(img)

        label = self.targets[internal_idx]
        return img, label 

def file_cache(filename):
    """Decorator to cache the output of a function to disk."""
    def decorator(f):
        @wraps(f)
        def decorated(self, directory, *args, **kwargs):
            filepath = Path('/nfshomes/sriramb') / filename
            if filepath.is_file():
                out = json.loads(filepath.read_text())
            else:
                out = f(self, directory, *args, **kwargs)
                filepath.write_text(json.dumps(out))
            return out
        return decorated
    return decorator

class CachedImageNet(ImageNet):
    @file_cache(filename="cached_classes.json")
    def find_classes(self, directory, *args, **kwargs):
        classes = super().find_classes(directory, *args, **kwargs)
        return classes

    @file_cache(filename="cached_structure.json")
    def make_dataset(self, directory, *args, **kwargs):
        dataset = super().make_dataset(directory, *args, **kwargs)
        return dataset

class SpuriousImagenetDataset(Dataset):
    def __init__(self, imgnet_path, spurious_path, class_name, transform=None):
        self.class_name = class_name
        with open('imagenet_classes.txt', 'r') as f:
            self.imgnet_classes = [line.strip() for line in f.readlines()]
        class_idx = self.imgnet_classes.index(class_name)
        self.imgnet_dataset = CachedImageNet(imgnet_path, split="train", transform=transform)
        self.imgnet_rel_indices = [i for i, t in enumerate(self.imgnet_dataset.targets) if t == class_idx]
        
        self.spur_dataset = SpuriousOnlyDataset(spurious_path, transform=None)
        internal_idcs = [i.item() for i in self.spur_dataset.internal_idcs]
        self.spur_rel_indices = [internal_idcs.index(i) for i, t in enumerate(self.spur_dataset.targets) if t == class_idx]
        
    def __len__(self):
        return 2*min(len(self.imgnet_rel_indices), len(self.spur_rel_indices))
    
    def __getitem__(self, idx):
        if idx % 2 == 0:
            ind = self.imgnet_rel_indices[idx // 2]
            return ind, self.imgnet_dataset[ind][0], "yes", 'spurious'
        else:
            ind = self.spur_rel_indices[idx // 2]
            return ind, self.spur_dataset[ind][0], "no", 'spurious'        


def load_dataset(dataset_name, class_name=None):
    if dataset_name == 'clean_waterbirds':
        return CleanWaterbirdsDataset(root='/nfshomes/sriramb/projects/vit_decompose/dataset_archives/uncorr/')
    elif dataset_name == 'celeba_blond':
        return CelebAHQ(root='/cmlscratch/sriramb/CelebAMask-HQ', split='ALL', label_type='all', classify='Blond_Hair', confounder='Male')
    elif dataset_name == 'celeba_blond_corrected':
        return CelebAHQCorrected(root='/cmlscratch/sriramb/CelebAMask-HQ', split='ALL', label_type='all', classify='Blond_Hair', confounder='Male', region='hair', correction=np.array([14.05237131, 12.63626369, 10.36256402]))
    elif 'spurious_imagenet' in dataset_name:
        return SpuriousImagenetDataset('/fs/cml-datasets/ImageNet/ILSVRC2012/', '/cmlscratch/sriramb/spurious_imagenet/images/', class_name)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

# %%
