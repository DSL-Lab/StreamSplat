import tyro
from dataclasses import dataclass, field
from typing import Literal, Dict, Optional, List
from configs.options_decoder import Options as BaseOptions

@dataclass
class Options(BaseOptions):
    model_type = "Inference"
    use_augmentation: bool = False
    fix_keys: List[str] = field(default_factory=lambda: ["rot_static", "rot_dynamic"])
    sample_keys: List[str] = field(default_factory=list)
    pred_keys: List[str] = field(default_factory=lambda: ["rgb", "opacity", "scale", "xyz_static"])
    compile: bool = True

AllConfigs = Options
