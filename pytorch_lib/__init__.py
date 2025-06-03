from .trainer import Trainer
from .net import Net
from .dataset_lib import*
from .loss_lib import*
from .transform_lib import*
from .visualization import*
from .module_lib import*
from .vision_language import*
from .inferencer import*
from .pretrained_model import*
from .utility import*

import os,torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print_GPU_info()
