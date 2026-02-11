import os
import numpy as np
import json
import glob
from os import path as osp
from torch.utils.data import Dataset
import torch
import torchvision.transforms as tf
from PIL import Image
import copy
import random

from configs.options import Options 
from datasets.transform_utils import NormalizeImage, PrepareForNet, Resize

class Co3DDataset(Dataset):
    def __init__(self, 
                 opt: Options, 
                 training=True,
                 shuffle=False,
                 nearby_range=5,
                 ):
        self.opt = opt
        self.training = training
        self.shuffle = shuffle
        if nearby_range < opt.output_frames - 1:
            print(f"Warning: nearby_range is less than output_frames - 1 on co3d, nearby_range: {nearby_range}, output_frames: {opt.output_frames}, change nearby_range to {opt.output_frames - 1}")
            nearby_range = opt.output_frames - 1
        self.nearby_range = nearby_range

        self.split = ["train", "val"] if training else ["train"]
        self.set_transform()

        self.categories = [d for d in os.listdir(opt.root_path) if osp.isdir(osp.join(opt.root_path, d))]
        assert len(self.categories) > 0, "No categories found in the data path"

        self.set_lists = []
        for category in self.categories:
            set_list_path = osp.join(opt.root_path, category, "set_lists")
            if osp.exists(set_list_path):
                for set_list_file in glob.glob(osp.join(set_list_path, "*.json")):
                    if training and "manyview_dev" in set_list_file:
                        with open(set_list_file, "r") as f:
                            self.set_lists.append(json.load(f))
                    elif not training and "manyview_test" in set_list_file:
                        with open(set_list_file, "r") as f:
                            self.set_lists.append(json.load(f))


        self.frame_infos = []
        self._all_frames = []
        self.set_len = []
        self.ret_frames = []
        for set_idx, set_list in enumerate(self.set_lists):
            seq_frame_infos = []
            for split in self.split:
                for idx, frame_info in enumerate(set_list[split]):
                    seq_frame_infos.append(frame_info)
            seq_frame_infos = sorted(seq_frame_infos, key=lambda x: x[1])

            self.frame_infos.append(seq_frame_infos)
            self.set_len.append(len(seq_frame_infos))
            
            for idx, frame_info in enumerate(seq_frame_infos):
                self._all_frames.append((frame_info, set_idx, idx))

                has_enough_frames = True
                if self.opt.output_frames > 1 and idx + self.opt.output_frames > len(seq_frame_infos):
                    has_enough_frames = False
                
                if has_enough_frames:
                    self.ret_frames.append((frame_info, set_idx, idx))

    
    def set_transform(self):
        self.transform = tf.Compose([
            tf.Resize(self.opt.down_resolution),
            tf.ToTensor(),
        ])
        
        self.depth_transform = tf.Compose([
                tf.Resize(self.opt.down_resolution, interpolation=tf.InterpolationMode.NEAREST),
                tf.ToTensor(), 
            ])

    def __len__(self):
        return len(self.ret_frames)
    
    def get_predicted_depth_path(self, image_path):
        pred_dir = image_path.replace("images", "predict_depths")
        pred_dir = osp.dirname(pred_dir)
        os.makedirs(pred_dir, exist_ok=True)
        base_name = osp.splitext(osp.basename(image_path))[0] + "_pred.png"
        return osp.join(pred_dir, base_name)
    
    def __getitem__(self, index):
        frame_info, set_idx, current_idx = self.ret_frames[index]
        frames = []
        depths = []
        timestamps_list = [] 

        # frame[0]
        image_path = frame_info[2]
        image_path = osp.join(self.opt.root_path, image_path)
        
        depth_path = self.get_predicted_depth_path(image_path)
        
        if not osp.exists(image_path) or not osp.exists(depth_path):
            print(f"Warning: Missing file {image_path} or {depth_path}")
        else:
            frame = Image.open(image_path)
            depth = Image.open(depth_path)
            depth = self.depth_transform(depth)
            frame = self.transform(frame)
            frames.append(frame)
            depths.append(depth)
            timestamps_list.append(0.0)

        seq_length = self.set_len[set_idx]
        if not self.shuffle:
            if current_idx + self.nearby_range >= seq_length:
                interval = (seq_length - current_idx) // (self.opt.output_frames - 1)
            else:
                interval = self.nearby_range // (self.opt.output_frames - 1)
            assert interval > 0, "Interval must be greater than 0"
            offsets = [i * interval for i in range(1, self.opt.output_frames)]
        else:
            if current_idx + self.nearby_range >= seq_length:
                offsets = random.sample(range(1, seq_length - current_idx), self.opt.output_frames - 1)
            else:
                offsets = random.sample(range(1, self.nearby_range + 1), self.opt.output_frames - 1)
        
        offsets.sort()
        if self.opt.output_frames > 1:
            assert len(offsets) == self.opt.output_frames - 1, "offsets length must be equal to output_frames - 1"
        
        for offset in offsets:
            pair_idx = current_idx + offset
            image_path_pair = self.frame_infos[set_idx][pair_idx][2]
            image_path_pair = osp.join(self.opt.root_path, image_path_pair)
            
            depth_path_pair = self.get_predicted_depth_path(image_path_pair)
            
            if not osp.exists(image_path_pair) or not osp.exists(depth_path_pair):
                print(f"Warning: Missing file {image_path_pair} or {depth_path_pair}")
                continue
            else:
                frame_pair = Image.open(image_path_pair)
                depth_pair = Image.open(depth_path_pair)
                depth_pair = self.depth_transform(depth_pair)
                frame_pair = self.transform(frame_pair)
                frames.append(frame_pair)
                depths.append(depth_pair)
                timestamps_list.append(float(offset))
        
        frames = torch.stack(frames)
        depths = torch.stack(depths)
        timestamps_tensor = torch.tensor(timestamps_list)
        
        results = {
            "frames": frames,
            "depths": depths,
            "timestamps": timestamps_tensor
        }

        return results
