import math
import time
import os
import random
import argparse
import glob
import json

import torch
from torch.utils import data

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage, signal

import cv2
from PIL import Image


def temporal_transform(frame_indices, sample_range):
    tmp = np.random.randint(0,len(frame_indices)-sample_range)
    return frame_indices[tmp:tmp+sample_range]

DAVIS_2016 = ['bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 
              'car-turn', 'dance-jump', 'dog-agility', 'drift-turn', 'elephant',
              'flamingo', 'hike', 'hockey', 'horsejump-low', 'kite-walk', 
              'lucia', 'mallard-fly', 'mallard-water', 'motocross-bumps', 
              'motorbike', 'paragliding', 'rhino', 'rollerblade', 
              'scooter-gray', 'soccerball', 'stroller', 'surf', 'swing', 
              'tennis', 'train',' blackswan', 'bmx-trees', 'breakdance', 
              'camel', 'car-roundabout', 'car-shadow', 'cows', 'dance-twirl',
              'dog', 'drift-chicane', 'drift-straight', 'goat', 
              'horsejump-high', 'kite-surf', 'libby', 'motocross-jump', 
              'paragliding-launch', 'parkour', 'scooter-black', 'soapbox']

class DAVIS(data.Dataset):
    def __init__(self, root, imset='2016/train.txt', 
                 resolution='480p', size=(256,256), sample_duration=0):
        self.sample_duration = sample_duration
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.size = size
        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(
                    self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(
                    self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        num_objects = 1
        info['num_objects'] = num_objects        
        
        images = []
        masks = []
        struct = ndimage.generate_binary_structure(2, 2)

        f_list = list(range(self.num_frames[video]))
        if self.sample_duration >0:
            f_list = temporal_transform(f_list,self.sample_duration)

        for f in f_list:
                            
            img_file = os.path.join(
                self.image_dir, video, '{:05d}.jpg'.format(f))
            image_ = cv2.resize(
                cv2.imread(img_file), self.size, cv2.INTER_CUBIC)
            image_ = np.float32(image_)/255.0
            images.append(torch.from_numpy(image_))

            try:
                mask_file = os.path.join(
                    self.mask_dir, video, '{:05d}.png'.format(f))
            except:
                mask_file = os.path.join(self.mask_dir, video, '00000.png')
            mask_ = np.array(Image.open(mask_file).convert('P'), np.uint8)
            mask_ = cv2.resize(mask_,self.size, cv2.INTER_NEAREST)

            if video in DAVIS_2016:
                mask_ = (mask_ != 0)
            else:
                select_mask = min(1,mask_.max())
                mask_ = (mask_==select_mask).astype(np.float)
            
            w_k = np.ones((10,6))                
            mask2 = signal.convolve2d(mask_.astype(np.float), w_k, 'same')
            mask2 = 1 - (mask2 == 0)
            mask_ = np.float32(mask2)
            masks.append( torch.from_numpy(mask_) )

        masks = torch.stack(masks)
        masks = ( masks == 1 ).type(torch.FloatTensor).unsqueeze(0)
        images = torch.stack(images).permute(3,0,1,2)

        return images, masks, info

class Each_Data_Loader:
    def __init__(self, data_name, label_name, size=(256,256), sample_duration=0):
        self.sample_duration = sample_duration
        self.data_name = data_name
        self.label_name = label_name
        self.size = size

        self.train_image_dir = f'./data/train_data/{self.data_name}/{self.label_name}'
        self.label_image_dir = f'./data/train_label/{self.data_name}/{self.label_name}'
        
        self.train_data = glob.glob(os.path.join(self.train_image_dir,'*.jpg'))
        self.train_data.sort()
        
        self.train_label = glob.glob(os.path.join(self.label_image_dir,'*.png'))
        self.train_label.sort()

        self.num_frames = len(self.train_data)
        
    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        video = self.label_name
        
        info = {}
        info['name'] = self.label_name
        info['num_frames'] = self.num_frames
        
        images = []
        masks = []

        img_file = self.train_data[index]
        image = cv2.resize(cv2.imread(img_file), self.size, cv2.INTER_CUBIC)
        image = np.float32(image)/255.0

        mask_file = self.train_label[index]

        mask = np.array(Image.open(mask_file).convert('P'), np.uint8)
        mask = cv2.resize(mask,self.size, cv2.INTER_NEAREST)
        mask = (mask != 0)

        mask = torch.tensor(mask).unsqueeze(-1)
        mask = ( mask == 1 ).type(torch.FloatTensor)
        image = torch.tensor(image)

        image = image.permute(2,0,1)
        mask = mask.permute(2,0,1)

        return image, mask

class Custom_Data_Loader:
    def __init__(self, data_name, size=(256,256), sample_duration=0):
        self.sample_duration = sample_duration
        self.data_name = data_name
        self.size = size

        self.label_names = []

        self.loaders = []

        with open(f'./data/label_txt/{self.data_name}.txt') as f:
            label = f.readline()
            label = label.rstrip('\n')
            self.label_names.append(label)

        for label in self.label_names:
            self.loaders.append(Each_Data_Loader(self.data_name, label, size=self.size, sample_duration=self.sample_duration))

        self.num_objects = len(self.label_names)

    def __len__(self):
        return self.num_objects

    def __getitem__(self, index):
        return self.loaders[index]

    def get_train_input(self, index, frame_interval=3):
        last_index = len(self.loaders[index]) - 4 * frame_interval - 1
        
        if index > last_index:
            print("your input frame index is fault !")
        
        X_6l, X_6l_mask = self.loaders[index][index]
        X_6l = X_6l.unsqueeze(0)
        X_6l_mask = X_6l_mask.unsqueeze(0)

        X_3l, X_3l_mask = self.loaders[index][index + 1 * frame_interval]
        X_3l = X_3l.unsqueeze(0)
        X_3l_mask = X_3l_mask.unsqueeze(0)

        X, X_mask = self.loaders[index][index + 2 * frame_interval]
        X = X.unsqueeze(0)
        X_mask = X_mask.unsqueeze(0)

        X_3r, X_3r_mask = self.loaders[index][index + 3 * frame_interval]
        X_3r = X_3r.unsqueeze(0)
        X_3r_mask = X_3r_mask.unsqueeze(0)

        X_6r, X_6r_mask = self.loaders[index][index + 4 * frame_interval]
        X_6r = X_6r.unsqueeze(0)
        X_6r_mask = X_6r_mask.unsqueeze(0)

        return torch.stack([X_6l, X_3l, X, X_3r, X_6r], dim=2), torch.stack([X_6l_mask, X_3l_mask, X_mask, X_3r_mask, X_6r_mask], dim=2)
        

if __name__=='__main__':
    test = Custom_Data_Loader('sample')
    test[0][0]
    x, y = test.get_train_input(0,0)
    print(x.size())
    print(y.size())

        

