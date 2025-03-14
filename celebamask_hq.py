# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import cv2
import functools
from typing import Dict, List
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import _log_api_usage_once
import torch
from PIL import Image

@functools.lru_cache()
def _cached_imread(fname, flags=None):
    return cv2.imread(fname, flags=flags)

BICUBIC = transforms.functional.InterpolationMode('bicubic')
NEAREST = transforms.functional.InterpolationMode('nearest')

class ToTensor:
    def __init__(self) -> None:
        _log_api_usage_once(self)

    def __call__(self, pic):
        pic = np.array(pic)
        pic = pic[np.newaxis, ...]
        return torch.Tensor(pic)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class CelebAMaskHQ(Dataset):
    def __init__(self, root, split, label_type='all', classify=None, confounder=None, 
                 img_transform=None, mask_transform=None):
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
        self.mask_transform = mask_transform
        
        # if mask_transform is None:
        #     self.mask_transform = transforms.Compose([
        #         transforms.ToPILImage(),
        #         transforms.Resize(224, interpolation=NEAREST, max_size=None, antialias=None),
        #         # transforms.CenterCrop(size=(224, 224)),
        #         ToTensor()
        #     ])
        
        # if img_transform is None:
        #     self.img_transform = transforms.Compose([
        #         transforms.Resize(224, interpolation=BICUBIC, max_size=None, antialias=None),
        #         # transforms.CenterCrop(size=(224, 224)),
        #         ToTensor()
        #     ])

        
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
        name = self.names[index]

        image = Image.open(os.path.join(self.root, 'CelebA-HQ-img', name + '.jpg'))
        if self.img_transform:
            image = self.img_transform(image)
#         image = None
        mask = self.make_label(index, self.label_setting['suffix'])
        if self.mask_transform:
            mask = self.mask_transform(mask)
        attributes = torch.Tensor(self.attributes[int(name)])
#         attributes = None
        return (image, mask, attributes)

    def __len__(self):
        return len(self.names)

    def sample_name(self, index):
        return self.names[index]

    @property
    def label_names(self) -> List[str]:
        return self.label_setting['names']