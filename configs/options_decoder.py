import tyro
from dataclasses import dataclass, field
from typing import Literal, Dict, Optional, List
from configs.options import Options as BaseOptions

encoder_path: str = "PATH_TO_ENCODER_CHECKPOINT"

# specify the root path for each dataset here
root_path_re10k: str = "PATH_TO_RE10K"
root_path_co3d: str = "PATH_TO_CO3D"
root_path_davis: str = "PATH_TO_DAVIS"
root_path_vos: str = "PATH_TO_Youtube-VOS"
root_path_combined: str = "combined"             # placeholder without any dataset

@dataclass
class Options(BaseOptions):
    model_type = "Decoder"
    mixed_precision: str = 'bf16'
    gradient_accumulation_steps: int = 1
    
    # Dataset paths
    root_path: str = ""
    re10k_path: str = ""
    co3d_path: str = ""
    davis_path: str = ""
    vos_path: str = ""
    batch_size: int = 16
    num_workers: int = 8
    resume: Optional[str] = None

    lr: float = 5e-4
    enable_depth: bool = True 
    num_epochs: int = 200
    warmup_iters: int = 20000
    lr_decay_epochs: int = 200
    gradient_clip: float = 1.0
    forder: int = 1
    output_frames: int = 6 
    lambda_lpips: float = 0.05
    lambda_depth: float = 0.01
    lambda_reg: float = 0.0
    lambda_mask: float = 3.0
    ignore_large_loss: float = 0.3
    lpips_start_epoch : int = 50
    depth_start_epoch : int = 0
    # workspace
    workspace: str = './workspace'

    # Model architecture
    patch_size: int = 8
    decoder_num_layers: int = 10
    decoder_hidden_dim: int = 768
    decoder_ratio: float = 3.0
    opacity_activation: str = "sigmoid"
    use_augmentation: bool = True
    
    # Probabilistic sampling
    use_pm: bool = True
    
    # Decoder-specific training
    encoder_path: str = ""
    use_dino: bool = True
    drop_path_rate: float = 0.0
    pm_dynamic: bool = True
    skip: bool = False
    fix_opacity: bool = False
    
    # Training options
    compile: bool = False


    
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['combined'] = ''
config_defaults['combined_rcvd'] = Options(root_path=root_path_combined, re10k_path=root_path_re10k, co3d_path=root_path_co3d, davis_path=root_path_davis, vos_path=root_path_vos)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
