import torch
from torch.utils.data import Dataset
import torchvision.transforms as tf
from .provider_co3d import Co3DDataset
from .provider_re10k_map import Re10kMapDataset
from .provider_davis import DAVISDataset
from .provider_vos import VOSDataset
import numpy as np
import random
from copy import deepcopy

class CombinedDataset(Dataset):
    def __init__(self, opt, training=True, shuffle=False, override_nearby_range=None):
        """
        Combined dataset that includes Co3D, Re10k and DAVIS datasets.
        
        Args:
            opt: Configuration object that includes:
                - dataset_weights: Dict of dataset names to their weights
                - root_paths: Dict of dataset names to their root paths
            training: Whether in training mode
            shuffle: Whether to shuffle the dataset
            override_nearby_range: Override the nearby range for the dataset
        """
        super().__init__()
        self.opt = opt
        self.training = training
        self.shuffle = shuffle
        self.rcvd = hasattr(opt, 'vos_path') and opt.vos_path != ""
        self.override_nearby_range = override_nearby_range
        if override_nearby_range is not None:
            nearby_range_kwargs = {"nearby_range": override_nearby_range}
        else:
            nearby_range_kwargs = {}

        if self.rcvd:
            print("Using Re10k, Co3D, DAVIS and VOS datasets")
        else:
            print("Using Re10k, Co3D and DAVIS datasets")

        self.set_transform()

        self.datasets = {}
        self.dataset_lengths = {}
        self.dataset_names = []
        
        # Setup Co3D dataset if path exists
        if hasattr(opt, 'co3d_path'):
            opt_co3d = deepcopy(opt)
            opt_co3d.root_path = opt_co3d.co3d_path
            dataset = Co3DDataset(opt=opt_co3d, training=training, shuffle=shuffle, **nearby_range_kwargs)
            dataset.transform = self.transform
            dataset.depth_transform = self.depth_transform
            self.datasets['co3d'] = dataset
            self.dataset_lengths['co3d'] = len(dataset)
            
        # Setup Re10k dataset if path exists
        if hasattr(opt, 're10k_path'):
            opt_re10k = deepcopy(opt)
            opt_re10k.root_path = opt_re10k.re10k_path
            dataset = Re10kMapDataset(opt=opt_re10k, training=training, shuffle=shuffle, **nearby_range_kwargs)
            dataset.transform = self.transform
            dataset.depth_transform = self.depth_transform
            self.datasets['re10k'] = dataset
            self.dataset_lengths['re10k'] = len(dataset)
            
        # Setup DAVIS dataset if path exists
        if hasattr(opt, 'davis_path'):
            opt_davis = deepcopy(opt)
            opt_davis.root_path = opt_davis.davis_path
            dataset = DAVISDataset(opt=opt_davis, training=training, shuffle=shuffle, **nearby_range_kwargs)
            dataset.transform = self.transform
            dataset.depth_transform = self.depth_transform
            self.datasets['davis'] = dataset
            self.dataset_lengths['davis'] = len(dataset)

        if hasattr(opt, 'vos_path') and self.rcvd:
            opt_vos = deepcopy(opt)
            opt_vos.root_path = opt_vos.vos_path
            dataset = VOSDataset(opt=opt_vos, training=training, shuffle=shuffle, **nearby_range_kwargs)
            dataset.transform = self.transform
            dataset.depth_transform = self.depth_transform
            self.datasets['vos'] = dataset
            self.dataset_lengths['vos'] = len(dataset)

        if not self.datasets:
            raise ValueError("No datasets were initialized. Check the paths in config.")
            
        self.dataset_names = []
        
        for name, length in self.dataset_lengths.items():
            self.dataset_names.append(name)
            print("Dataset {} has {} samples.".format(name, length))
            

        self.dataset_boundaries = []
        current_boundary = 0
        for name in self.dataset_names:
            self.dataset_boundaries.append(current_boundary)
            current_boundary += self.dataset_lengths[name]
        self.dataset_boundaries.append(current_boundary)  # Add final boundary

        self.seed = getattr(opt, 'seed', 42)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def __len__(self):
        return sum(self.dataset_lengths.values())

    def __getitem__(self, idx):
        for i, (start, end) in enumerate(zip(self.dataset_boundaries[:-1], self.dataset_boundaries[1:])):
            if start <= idx < end:
                dataset_name = self.dataset_names[i]
                dataset = self.datasets[dataset_name]
                index_in_dataset = idx - start
                break
        else:
            raise IndexError(f"Index {idx} out of range")

        data = dataset[index_in_dataset]

        required_keys = {'frames', 'depths', 'timestamps'}
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise KeyError(f"Missing required keys {missing_keys} in dataset {dataset_name}")

        if 'supv_masks' not in data:
            data['supv_masks'] = torch.zeros_like(data['depths'], dtype=torch.float32)
        else:
            data['supv_masks'] = data['supv_masks'].to(torch.float32)

        data['src_nm'] = dataset_name

        return data

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