import os
import glob
import random
from os import path as osp
from torch.utils.data import Dataset
import torch
import torchvision.transforms as tf
from PIL import Image
import numpy as np

class VOSDataset(Dataset):
    def __init__(self, opt, training=True, shuffle=False, nearby_range=3):
        self.opt = opt
        self.training = training
        self.shuffle = shuffle
        if nearby_range < opt.output_frames - 1:
            print(f"Warning: nearby_range is less than output_frames - 1 on vos, nearby_range: {nearby_range}, output_frames: {opt.output_frames}, change nearby_range to {opt.output_frames - 1}")
            nearby_range = opt.output_frames - 1
        self.nearby_range = nearby_range

        self.split = "train" if training else "val"
        
        base = osp.join(opt.root_path, "train" if training else "valid")
        self.images_root = osp.join(base, "JPEGImages")
        self.mask_root   = osp.join(base, "Annotations") if training else None
        
        self.set_names = [
            d for d in os.listdir(self.images_root)
            if osp.isdir(osp.join(self.images_root, d))
        ]

        self.frame_infos = []
        self._all_frames = []
        self.set_len = [] 
        self.ret_frames = []
        
        for set_idx, set_name in enumerate(self.set_names):
            set_dir = osp.join(self.images_root, set_name)
            image_files = glob.glob(osp.join(set_dir, "*.jpg"))
            if not image_files:
                print(f"Warning: No jpg images found in {set_dir}")
                continue

            def extract_timestamp(file_path):
                base = osp.basename(file_path)
                num_str = osp.splitext(base)[0]
                try:
                    return float(num_str)
                except ValueError:
                    return 0.0

            image_files = sorted(image_files, key=lambda x: extract_timestamp(x))
            
            frames_for_set = []
            for idx, img_path in enumerate(image_files):
                timestamp = extract_timestamp(img_path)
                frames_for_set.append((img_path, timestamp))
                self._all_frames.append((set_idx, img_path, timestamp, idx))

                has_enough_frames = True
                if self.opt.output_frames > 1 and idx + self.opt.output_frames > len(image_files):
                    has_enough_frames = False
                if has_enough_frames:
                    self.ret_frames.append((set_idx, img_path, timestamp, idx))

            self.frame_infos.append(frames_for_set)
            self.set_len.append(len(frames_for_set))
        
        self.set_transform()

    def set_transform(self):
        self.transform = tf.Compose([
            tf.Resize(self.opt.down_resolution),
            tf.ToTensor(),
        ])
        self.depth_transform = tf.Compose([
            tf.Resize(self.opt.down_resolution, interpolation=tf.InterpolationMode.NEAREST),
            tf.ToTensor(),
        ])
        self.mask_transform = tf.Compose([
            tf.Resize(self.opt.down_resolution, interpolation=tf.InterpolationMode.NEAREST),
        ])

    def transform_mask(self, mask):
        mask = self.mask_transform(mask)
        mask = np.array(mask, dtype=np.uint8)
        mask = torch.from_numpy(mask)
        return mask
    
    def get_predicted_depth_path(self, image_path):
        pred_dir = image_path.replace("JPEGImages", "depthImages")
        pred_dir = osp.dirname(pred_dir)
        os.makedirs(pred_dir, exist_ok=True)
        base_name = osp.splitext(osp.basename(image_path))[0] + "_pred.png"
        return osp.join(pred_dir, base_name)
    
    def __len__(self):
        return len(self.ret_frames)

    def __getitem__(self, index):
        set_idx, image_path, timestamp, frame_idx = self.ret_frames[index]
        
        if not osp.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        main_image = Image.open(image_path).convert("RGB")
        depth_path = self.get_predicted_depth_path(image_path)
        depth_image = Image.open(depth_path)
        main_image = self.transform(main_image)
        depth_image = self.depth_transform(depth_image)

        if self.training:
            mask_path = image_path.replace(self.images_root, self.mask_root).replace(".jpg", ".png")
            if not osp.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            mask_image = Image.open(mask_path)
            mask_image = self.transform_mask(mask_image).unsqueeze(0)
        else:
            mask_image = torch.ones_like(depth_image)
        
        frames = [main_image]
        depths = [depth_image]
        masks = [mask_image]
        timestamps_list = [0.0]
        
        seq_length = self.set_len[set_idx]
        current_idx = frame_idx
        
        if self.opt.output_frames > 1 and seq_length > 1:
            if not self.shuffle:
                # pick frame with fixed interval
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
            
            for offset in offsets:
                pair_idx = current_idx + offset
                img_file_pair, ts_pair = self.frame_infos[set_idx][pair_idx]
                if not osp.exists(img_file_pair):
                    print(f"Warning: Missing file {img_file_pair}")
                    continue
                pair_image = Image.open(img_file_pair).convert("RGB")
                pair_depth_path = self.get_predicted_depth_path(img_file_pair)
                pair_depth_image = Image.open(pair_depth_path)
                pair_image = self.transform(pair_image)
                pair_depth_image = self.depth_transform(pair_depth_image)

                if self.training:
                    pair_mask_path = img_file_pair.replace(self.images_root, self.mask_root).replace(".jpg", ".png")
                    if not osp.exists(pair_mask_path):
                        print(f"Warning: Missing file {pair_mask_path}")
                        continue
                    pair_mask_image = Image.open(pair_mask_path)
                    pair_mask_image = self.transform_mask(pair_mask_image).unsqueeze(0)
                else:
                    pair_mask_image = torch.ones_like(pair_depth_image)

                frames.append(pair_image)
                depths.append(pair_depth_image)
                masks.append(pair_mask_image)
                timestamps_list.append(offset)
        
        frames_tensor = torch.stack(frames)
        depths_tensor = torch.stack(depths)
        masks_tensor = torch.stack(masks).float()
        timestamps_tensor = torch.tensor(timestamps_list, dtype=torch.float)

    
        results = {
            "frames": frames_tensor,
            "depths": depths_tensor,
            "supv_masks": masks_tensor,
            "timestamps": timestamps_tensor
        }
        return results
    