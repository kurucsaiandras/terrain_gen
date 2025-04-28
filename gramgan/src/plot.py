from pathlib import Path

import torch
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from models import Generator, Discriminator, NoiseTransforms, sample_bilinear
from data import RealHeightMap
from train import find_latest_epoch

def plot_height_examples(generator: Generator, discriminator: Discriminator, cfg: DictConfig, device=None):
    torch.manual_seed(1234)

    height_map = RealHeightMap(device=device) 

    with torch.no_grad():
        real_patches = height_map.get_patches(cfg.train.batch_size, cfg.model.image_res).to(device)
        real_logits, _ = discriminator(real_patches)

        coords = torch.stack(torch.meshgrid(
            torch.linspace(0.0, 1.0, cfg.model.image_res, device=device),
            torch.linspace(0.0, 1.0, cfg.model.image_res, device=device),
            indexing='xy',
        )).permute([1, 2, 0])
        translations = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], device=device)
        patch_coords = coords + translations.view([4, 1, 1, 2])
        
        noise = torch.randn(cfg.model.noise_features, cfg.model.noise_res, cfg.model.noise_res, device=device)
        fake_patches = generator(noise, patch_coords)
        fake_patches = fake_patches.permute([0, 3, 1, 2])
        fake_logits, _ = discriminator(fake_patches)

        fig, axs = plt.subplots(2, 4)

        for j in range(4):
            axs[0, j].set_title(f"D: {real_logits[j,0]:.2E}")
            axs[0, j].imshow(real_patches[j,0,:,:].cpu().numpy())
            
            axs[1, j].set_title(f"D: {fake_logits[j,0]:.2E}")
            axs[1, j].imshow(fake_patches[j,0,:,:].detach().cpu().numpy())

        fig.savefig(f"reports/height_examples_{cfg.train.name}.pdf")

def plot_sample_bilinear():
    noise = torch.rand(1, 4, 6)
    coords = torch.stack(torch.meshgrid(
        torch.linspace(-4.0, 12.0, 128),
        torch.linspace(-6.0, 12.0, 128),
        indexing='xy',
    )).permute([1, 2, 0]).unsqueeze(2)

    image = sample_bilinear(noise, coords)
    
    fig = plt.figure()
    ax = fig.subplots()
    ax.imshow(image.cpu().numpy())
    fig.savefig("reports/sample_bilinear.pdf")

def plot_generator_noise_transforms(generator: Generator, cfg: DictConfig, device=None):
    
    noise = torch.randn(cfg.model.noise_features, cfg.model.noise_res, cfg.model.noise_res, device=device)

    coords = torch.stack(torch.meshgrid(
        torch.linspace(0.0, 1.0, cfg.model.image_res, device=device),
        torch.linspace(0.0, 1.0, cfg.model.image_res, device=device),
        indexing='xy',
    )).permute([1, 2, 0])

    noise_coords = generator.noise_transforms(coords)
    heights = sample_bilinear(noise, noise_coords)

    fig = plt.figure()
    ax = fig.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            ax[i, j].imshow(heights[...,idx].detach().cpu().numpy(), vmin=-2.0, vmax=2.0)
    fig.tight_layout()
    fig.savefig(f"reports/noise_transforms_{cfg.train.name}.pdf")

def plot_noise_transforms(cfg: DictConfig, device = None):
    noise_transforms = NoiseTransforms(cfg.model.noise_features).to(device)
    noise = torch.randn(cfg.model.noise_features, cfg.model.noise_res, cfg.model.noise_res, device=device)

    coords = torch.stack(torch.meshgrid(
        torch.linspace(0.0, 1.0, cfg.model.image_res, device=device),
        torch.linspace(0.0, 1.0, cfg.model.image_res, device=device),
        indexing='xy',
    )).permute([1, 2, 0])

    noise_coords = noise_transforms(coords)
    heights = sample_bilinear(noise, noise_coords)

    fig = plt.figure()
    ax = fig.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            ax[i, j].imshow(heights[...,idx].detach().cpu().numpy(), vmin=-2.0, vmax=2.0)
    fig.tight_layout()
    fig.savefig(f"reports/noise_transforms.pdf")

def load_models(
    save_dir: Path,
    generator: Generator,
    discriminator: Discriminator,
    epoch: int,
):
    dict = torch.load(save_dir / f"train_state_{epoch}.pt")

    generator.load_state_dict(dict["generator"])
    discriminator.load_state_dict(dict["discriminator"])



@hydra.main(version_base=None, config_path="..", config_name="config")
def all_plots(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    save_dir: Path = Path("models") / cfg.train.name

    generator = Generator(cfg.model.hidden_features, cfg.model.noise_features, cfg.model.output_features).to(device)
    discriminator = Discriminator(cfg.model.output_features).to(device)

    epoch = find_latest_epoch(save_dir)
    load_models(
        save_dir,
        generator,
        discriminator,
        epoch,
    )

    plot_sample_bilinear()
    plot_noise_transforms(cfg, device)
    plot_generator_noise_transforms(generator, cfg, device)
    plot_height_examples(generator, discriminator, cfg, device)

if __name__ == "__main__":
    all_plots()