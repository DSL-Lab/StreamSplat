import tyro
from dataclasses import dataclass, field
from typing import Tuple, Literal, Dict, Optional, List

# specify the root path for each dataset here
root_path_re10k: str = "PATH_TO_RE10K"
root_path_co3d: str = "PATH_TO_CO3D"
root_path_davis: str = "PATH_TO_DAVIS"
root_path_vos: str = "PATH_TO_Youtube-VOS"
root_path_combined: str = "combined"             # placeholder without any dataset

@dataclass
class Options:
    model_type = "Encoder"
    mixed_precision: str = 'bf16'
    gradient_accumulation_steps: int = 1

    # dataset 
    root_path: str = ""
    re10k_path: str = ""
    co3d_path: str = ""
    davis_path: str = ""
    vos_path: str = ""

    batch_size: int = 32
    num_workers: int = 8
    resume: Optional[str] = None

    # Data augmentation control
    use_augmentation: bool = True

    lr: float = 5e-4
    enable_depth: bool = True 
    depth_loss_type: str = "ssitrim" 
    depth_model_name: str = 'vitl'
    num_epochs: int = 250
    warmup_iters: int = 50000
    lr_decay_epochs: int = 250
    gradient_clip: float = 1.0
    orth_proj: bool = True
    lambda_dssim: float = 0.2
    forder: int = 1
    dynamic_type: str = "poly" 
    image_height: int = 288
    image_width: int = 512
    down_resolution: Tuple[int, int] = (288, 512)
    input_frames: int = 1
    output_frames: int = 1 
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    lambda_lpips: float = 0.05
    lambda_depth: float = 0.05
    lambda_reg: float = 0.0
    ignore_large_loss: float = 0.0
    lpips_start_epoch : int = 50
    depth_start_epoch : int = 0
    dino_name: str = 'dinov2_vitb14_reg'
    workspace: str = './workspace'

    # GS Encoder
    in_channels: int = 3
    patch_size: int = 8
    checkpointing: bool = True
    num_layers: int = 10
    hidden_dim: int = 768
    bwindow_size: Optional[int] = None  # encoder window size
    window_size: int = 2304  # upsampler window size
    decoder_ratio: float = 2.0

    # GS Decoder
    decoder_dim: int = 64 
    nr_mix: int = 1
    fix_keys: List[str] = field(default_factory=lambda: ["rot_static", "rot_dynamic", "xyz_dynamic"])
    sample_keys: List[str] = field(default_factory=lambda: ["xyz_static"])
    pred_keys: List[str] = field(default_factory=lambda: ["rgb", "opacity", "scale"])
    opacity_activation: str = "sigmoid" 
    pixel_align: bool = True
    depth_downsample: bool = False

    # for probablistic sampling
    use_pm: bool = True

    # for decoder training
    keep_dynamic: bool = True

    # torch compile ops
    compile: bool = False

    pred_inverse: bool = True
    
    
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['davis'] = 'davis dataset'
config_defaults['davis'] = Options(root_path=root_path_davis)

config_doc['combined'] = 'combined dataset'
config_defaults['combined'] = Options(root_path=root_path_combined, re10k_path=root_path_re10k, co3d_path=root_path_co3d, davis_path=root_path_davis, vos_path=root_path_vos)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
