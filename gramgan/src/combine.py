import re
from pathlib import Path

import numpy as np
import torch
import torchvision
import hydra
from omegaconf import DictConfig

from models import Noise, Generator, sample_bilinear

@hydra.main(version_base=None, config_path="..", config_name="config")
def combine(cfg: DictConfig):
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    noise = Noise(cfg.model.noise_features, cfg.model.noise_res, device=device)

    generator = Generator(
        cfg.model.hidden_features,
        cfg.model.noise_features,
        cfg.model.output_features,
        cfg.model.num_hidden_layers,
    ).to(device)
    dict = torch.load(f"models/test_cond/train_state_4000.pt")
    generator.load_state_dict(dict["generator"])

    generator.eval()
    for p in generator.parameters():
        p.requires_grad = False

    base_terrain = torchvision.io.read_image("terrain/terrain.png", torchvision.io.ImageReadMode.GRAY).to(torch.float32).to(device)
    base_terrain -= base_terrain.mean()
    base_terrain /= base_terrain.std()

    condition_map = torch.zeros([9, *base_terrain.shape[1:]], device=device)

    for tile_y in range(1, base_terrain.shape[2] - 1):
        for tile_x in range(1, base_terrain.shape[1] - 1):
                conditions = base_terrain[:,tile_y-1:tile_y+2,tile_x-1:tile_x+2].flatten()
                condition_map[:, tile_y, tile_x] = conditions

    scale_factor = 64
    terrain: torch.Tensor = torch.nn.functional.interpolate(
        base_terrain.unsqueeze(0),
        scale_factor=scale_factor,
        mode='bilinear',    
    ).squeeze(0)

    details = torch.empty_like(terrain)

    # generate details row by row to have a reasonable batch size
    for y in range(base_terrain.shape[1] * scale_factor):
        coords_x = torch.linspace(0.0, base_terrain.shape[2], base_terrain.shape[2] * scale_factor, device=device)
        coords_y = (y / scale_factor) * torch.ones_like(coords_x, device=device)

        coords = torch.stack([coords_x, coords_y], dim=1)
        conditions = sample_bilinear(condition_map, coords.unsqueeze(1))
        
        row = generator(conditions, noise, coords)
        details[0,y,:] = row[:,0]

    terrain -= terrain.min()
    terrain /= terrain.max()

    details -= details.min()
    details /= details.max()

    np.save("terrain/terrain.npy", terrain.cpu().numpy())
    np.save("terrain/details.npy", details.cpu().numpy())
    
    
    


if __name__ == "__main__":
    combine()