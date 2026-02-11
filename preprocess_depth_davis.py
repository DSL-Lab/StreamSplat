import os
import glob
import json
import random
from os import path as osp

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms as tf
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import numpy as np
import torch.nn.functional as F

from configs.options import Options, AllConfigs
from model.depth_wrapper import DepthAnythingWrapper

def resize_to_multiple_of_14(image: Image.Image) -> Image.Image:
    """
    Resize a PIL image so that its height and width are the closest multiples of 14.
    The resizing is done with bilinear interpolation.
    """
    w, h = image.size
    # Compute the nearest multiples of 14 for width and height.
    new_w = round(w / 14) * 14
    new_h = round(h / 14) * 14
    return TF.resize(image, size=(new_h, new_w), interpolation=tf.InterpolationMode.BILINEAR)

def resize_to_shorter_side(image: Image.Image, target_size: int = 518) -> Image.Image:
    """
    Resize the input PIL image so that its shorter side equals target_size (default 518)
    while keeping the aspect ratio.
    """
    w, h = image.size
    if w <= h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))
    return TF.resize(image, size=(new_h, new_w), interpolation=tf.InterpolationMode.BILINEAR)

class DAVISDataset(Dataset):
    def __init__(self, opt, training=True, shuffle=False):
        self.opt = opt
        self.training = training
        self.shuffle = shuffle
        self.split = "train" if training else "val"
        
        self.set_txt_path = osp.join(opt.root_path, "ImageSets", "2017", f"{self.split}.txt")
        self.images_root = osp.join(opt.root_path, "JPEGImages", "Full-Resolution")
        
        with open(self.set_txt_path, "r") as f:
            self.set_names = [line.strip() for line in f.readlines() if line.strip()]
        if len(self.set_names) == 0:
            raise RuntimeError(f"No set names found in {self.set_txt_path}")

        self.frame_infos = []
        self.all_frames = []
        self.set_len = []  # number of frames per set
        
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
                self.all_frames.append((set_idx, img_path, timestamp, idx))
            self.frame_infos.append(frames_for_set)
            self.set_len.append(len(frames_for_set))
        
        self.set_transform()

    def set_transform(self):
        self.transform = tf.Compose([
            tf.ToTensor(),
        ])
    
    def get_predicted_depth_path(self, image_path):
        pred_dir = image_path.replace("JPEGImages", "depthImages")
        pred_dir = osp.dirname(pred_dir)
        os.makedirs(pred_dir, exist_ok=True)
        base_name = osp.splitext(osp.basename(image_path))[0] + "_pred.png"
        return osp.join(pred_dir, base_name)

    def __len__(self):
        return len(self.all_frames)

    def __getitem__(self, index):
        set_idx, image_path, timestamp, frame_idx = self.all_frames[index]
        
        if not osp.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        frame = Image.open(image_path).convert("RGB")
        depth = Image.open(image_path).convert("RGB")

        depth = resize_to_shorter_side(depth)

        depth = resize_to_multiple_of_14(depth)

        frame = self.transform(frame)
        depth = self.transform(depth)
        
        results = {
            "frames": frame,
            "depths": depth,
            "predicted_depth_paths": self.get_predicted_depth_path(image_path),
        }
        return results

class Predictor(torch.nn.Module):
    def __init__(self, opt: Options, **model_kwargs):
        super().__init__()
        self.opt = opt
        self.depth_prior = DepthAnythingWrapper(opt.depth_model_name)
    
    def forward(self, frames, depths, cond_times=None):
        with torch.no_grad():
            predicted_depth = self.depth_prior(depths).detach()  # [B, H, W]
            predicted_depth = F.interpolate(predicted_depth[:, None], size=frames.shape[-2:], mode="bilinear", align_corners=True)
        return predicted_depth  # [B, C, H, W]

def run_inference(opt: Options):
    train_dataset = DAVISDataset(opt, training=True)
    test_dataset = DAVISDataset(opt, training=False)
    dataloader_set = [DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8), 
                  DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
                  ]
    
    model = Predictor(opt)
    model.eval()
    device = "cuda"
    model.to(device)

    print("Running inference...")
    
    with torch.no_grad():
        for dataloader in dataloader_set:
            for batch in dataloader:
                frames = batch["frames"].to(device)
                depths = batch["depths"].to(device)
                predicted_depth = model(frames, depths)  # shape: [B, C, H, W]

                
                for i in range(predicted_depth.shape[0]):
                    depth_img = predicted_depth[i].cpu().numpy()
                    depth = depth_img

                    if depth_img.ndim == 3 and depth_img.shape[0] > 1:
                        depth_img = depth_img[0]
                    elif depth_img.ndim == 3:
                        depth_img = depth_img[0]
                    
                    depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX) 
                    depth_norm = depth_norm.astype(np.uint8) 

                    save_path = batch["predicted_depth_paths"][i]
                    os.makedirs(osp.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, depth_norm)
                    print(f"Saved predicted depth image to {save_path}")

if __name__ == "__main__":
    import tyro
    opt = tyro.cli(AllConfigs)
    run_inference(opt)