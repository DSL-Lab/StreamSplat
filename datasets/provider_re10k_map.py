import os
import numpy as np
import json
import pickle
import torch
import glob 
from typing import Optional, Dict, List, Tuple
from os import path as osp
from torch.utils.data import Dataset
import torchvision.transforms as tf
from PIL import Image 
from io import BytesIO
import random 

class Re10kMapDataset(Dataset):
    def __init__(self, 
                 opt,
                 shuffle=False,
                 training=True,
                 nearby_range=15):
        self.opt = opt
        self.shuffle = shuffle
        self.training = training
        self.seed = 42
        if nearby_range < opt.output_frames - 1:
            print(f"Warning: nearby_range is less than output_frames - 1 on re10k, nearby_range: {nearby_range}, output_frames: {opt.output_frames}, change nearby_range to {opt.output_frames - 1}")
            nearby_range = opt.output_frames - 1
        self.nearby_range = nearby_range

        self.root_path = opt.root_path
        self.augmentation = self.opt.augmentation
        self.set_transform()

        if self.training:
            self.root_path = osp.join(self.root_path, "train")
        else:
            self.root_path = osp.join(self.root_path, "test")

        assert os.path.exists(os.path.join(self.root_path, "index.json")), "index.json not found"
        with open(osp.join(self.root_path, "index.json"), "r") as f:
            self.index = json.load(f)
        
        self.src_chunk_paths = sorted(glob.glob(osp.join(self.root_path, "*.torch")))
        assert len(self.src_chunk_paths) > 0, "No chunks found in the data path"
        self._get_depth_path = lambda x: osp.join(osp.dirname(x), 'depth_{}'.format(osp.basename(x).replace('.torch', '.pt')))

        _unordered_depth_paths = set([self._get_depth_path(path) for path in self.src_chunk_paths])
        assert len([glob.glob(path) for path in _unordered_depth_paths]) == len(self.src_chunk_paths), "Depth paths do not match"

        # Pre-compute all frame indices
        self._all_frames = []
        self.chunk_data = {}
        self.set_len = []
        self.ret_frames = []
        self.key = {}
        
        for chunk_idx, src_chunk_path in enumerate(self.src_chunk_paths):
            src_chunk = torch.load(src_chunk_path)
            depth_chunk = pickle.load(open(self._get_depth_path(src_chunk_path), "rb"))
            assert [file['key'] for file in src_chunk] == depth_chunk['key']
            depth_chunk = depth_chunk['predicted_depth_norm']
            
            # Store chunk data for later access
            self.chunk_data[chunk_idx] = {
                'src_chunk': src_chunk,
                'depth_chunk': depth_chunk
            }
            self.set_len.append(len(src_chunk))
            # Create frame indices for this chunk
            for i_vid, src_video_info in enumerate(src_chunk):
                num_frames = len(src_video_info["images"])
                self.key[src_video_info["key"]] = (chunk_idx, i_vid)
                for i_frame in range(num_frames):
                    self._all_frames.append((chunk_idx, i_vid, i_frame))

                    has_enough_frames = True
                    if self.opt.output_frames > 1 and i_frame + self.opt.output_frames > num_frames:
                        has_enough_frames = False
                    
                    if has_enough_frames:
                        self.ret_frames.append((chunk_idx, i_vid, i_frame))

    def set_transform(self):
        self.transform = tf.Compose([
            tf.Resize(self.opt.down_resolution),
            tf.ToTensor(),
        ])
        self.depth_transform = tf.Compose([
            tf.Resize(self.opt.down_resolution, interpolation=tf.InterpolationMode.NEAREST),
            tf.ToTensor(),
        ])
        self.depth_transform_gt = tf.Compose([
            tf.Resize(self.opt.down_resolution, interpolation=tf.InterpolationMode.NEAREST),
            tf.ToTensor(),
        ])

        self.to_tensor = tf.ToTensor()
        self.down_resize = tf.Resize(self.opt.down_resolution)

    def __len__(self):
        return len(self.ret_frames)

    def __getitem__(self, idx):
        chunk_idx, i_vid, i_frame = self.ret_frames[idx]
        chunk_data = self.chunk_data[chunk_idx]
        src_chunk = chunk_data['src_chunk']
        depth_chunk = chunk_data['depth_chunk']
        
        src_video_info = src_chunk[i_vid]
        image_tensors = self.convert_images(src_video_info["images"])
        depth_tensors = torch.from_numpy(depth_chunk[i_vid]).unsqueeze(1)

        if isinstance(image_tensors, torch.Tensor):
            assert len(self.transform.transforms) == 2
            func_img_transform = self.transform.transforms[0]
            func_depth_transform = self.depth_transform.transforms[0]
        else:
            func_img_transform = self.transform
            func_depth_transform = self.depth_transform
        
        depth_tensors = depth_tensors.to(torch.float32) / 255.0
        
        frames = []
        depths = []
        timestamps_list = [0.0]
        
        current_frame = func_img_transform(image_tensors[i_frame])
        current_depth = func_depth_transform(depth_tensors[i_frame])
        frames.append(current_frame)
        depths.append(current_depth)
        
        if self.opt.output_frames > 1:
            seq_length = len(image_tensors)

            if not self.shuffle:
                interval = self.nearby_range
                offsets = []
                for i in range(1, 1000):
                    offset = i * interval
                    if i_frame + offset < seq_length:
                        offsets.append(offset)
                    else:
                        break
            else:
                if i_frame + self.nearby_range >= seq_length:
                    offsets = random.sample(range(1, seq_length - i_frame), self.opt.output_frames - 1)
                else:
                    offsets = random.sample(range(1, self.nearby_range + 1), self.opt.output_frames - 1)

            offsets.sort()
            for offset in offsets:
                shifted_idx = i_frame + offset
                frame = func_img_transform(image_tensors[shifted_idx])
                depth = func_depth_transform(depth_tensors[shifted_idx])
                frames.append(frame)
                depths.append(depth)
                timestamps_list.append(float(offset))
        
        frames = torch.stack(frames)
        depths = torch.stack(depths)
        mask_tensor = torch.zeros_like(depths).float()  # dummy mask
        timestamps_list = torch.tensor(timestamps_list)
        
        return {
            "frames": frames,
            "depths": depths,
            "supv_masks": mask_tensor,
            "timestamps": timestamps_list
        }

    def convert_images(self, images):
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)
