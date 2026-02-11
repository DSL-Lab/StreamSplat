import cv2
import torch
import numpy as np
import matplotlib
import os 


from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

filename = "gt_image_0_7_0.jpg"

raw_img = cv2.imread(f'example/{filename}')
depth = model.infer_image(raw_img, 518) # 518 is the default input size

depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth = depth.astype(np.uint8)
depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

cv2.imwrite(os.path.join(os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)