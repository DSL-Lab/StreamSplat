import os 

import torch 
import torch.nn as nn

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


class DepthAnythingWrapper(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        assert model_name in model_configs.keys(), f"model_name should be in {model_configs.keys()}"
        self.model = self._build_depth_anything(model_name)
        self._freeze()
    
    def _build_depth_anything(self, model_name):
        from importlib import import_module
        da2_hub = import_module(".depth_anything.depth_anything_v2.dpt", package=__package__)
        model_fn = getattr(da2_hub, "DepthAnythingV2")
        model = model_fn(**model_configs[model_name])
        checkpoint_path = os.path.join(os.path.dirname(__file__), f"../checkpoints/depth_anything_v2_{model_name}.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
        return model

    def _freeze(self):
        # logger.warning(f"======== Freezing Dinov2Wrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        
    def forward(self, x):
        return self.model(x)