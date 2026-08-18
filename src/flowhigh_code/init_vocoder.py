import json
import torch
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
# file is at src/flowhigh_code/init_vocoder.py -> go up 2 to repo root
repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
# Add repo root to path so `import vocoder.*` works from any CWD/entry point, quick fix
# :TODO: Use a proper package structure and entry point instead of this hack
sys.path.insert(0, repo_root)

from vocoder.BIGVGAN.bigvgan.models import BigVGAN
from vocoder.BIGVGAN.bigvgan.env import AttrDict   


def init_bigvgan(config, checkpoint, vocoder_freeze=False):    

    with open(config) as f:
        h = AttrDict(json.load(f))

    vocoder = BigVGAN(h)
    # CPU-compatible: map to available device instead of hard-coded "cuda"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dict = torch.load(checkpoint, map_location=device)
    vocoder.load_state_dict(checkpoint_dict['generator'])
    vocoder = vocoder.to(device).eval()
    vocoder.remove_weight_norm()

    if vocoder_freeze == True:    
        for param in vocoder.parameters():
            param.requires_grad = False

    return vocoder
