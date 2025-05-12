import re
from pathlib import Path

import torch
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from models import Noise, Generator, Discriminator, NoiseTransforms, sample_bilinear
from data import HeightMap, Texture
from train import random_patch_coords

def plot_losses(cfg: DictConfig):
    save_dir: Path = Path("models") / cfg.train.name
    stats = torch.load(save_dir / "stats.pt")

    loss_g: torch.Tensor = stats["generator_loss"]
    loss_d: torch.Tensor = stats["discriminator_loss"]

    steps = torch.arange(loss_g.shape[0])

    loss_d_mean = loss_d.mean(dim=1)
    loss_d_std = loss_d.std(dim=1)
    

    fig = plt.figure()
    ax = fig.subplots()

    ax.plot(steps.numpy(), loss_g.numpy(), label="Generator", color='b')
    ax.plot(steps.numpy(), loss_d_mean.numpy(), label="Discriminator", color='r')
    ax.fill_between(
        steps.numpy(),
        (loss_d_mean - loss_d_std).numpy(),
        (loss_d_mean + loss_d_std).numpy(),
        color='r',
        alpha=0.2,
    )
    ax.set_ylim(-500.0, 500.0)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"reports/losses_{cfg.train.name}.pdf")

def plot_texture_examples(cfg: DictConfig, device=None):
    torch.manual_seed(1234)

    texture = Texture(cfg.model.image_res, device=device) 
    noise = Noise(cfg.model.noise_features, cfg.model.noise_res, device=device)

    generator = Generator(cfg.model.hidden_features, cfg.model.noise_features, cfg.model.output_features).to(device)
    discriminator = Discriminator(cfg.model.output_features).to(device)

    save_dir: Path = Path("models") / cfg.train.name
    epochs = sorted(saved_epochs(save_dir))

    with torch.no_grad():

        fig = plt.figure(figsize=(24, 6))
        axs = fig.subplots(2, len(epochs))

        for j, epoch in enumerate(epochs):

            load_models(
                save_dir,
                generator,
                discriminator,
                epoch,
            )
            
            real_patches = texture.get_batch(1)
            fake_patches = generator(noise, random_patch_coords(cfg, device)).permute([0, 3, 1, 2])

            real_logits, _ = discriminator(real_patches)
            fake_logits, _ = discriminator(fake_patches)

            real_img = real_patches[0,:,:,:].permute([1, 2, 0]).cpu().numpy()
            fake_img = fake_patches[0,:,:,:].permute([1, 2, 0]).cpu().numpy()

            axs[0, j].set_title(f"D: {real_logits[0,0]:.2E}")
            axs[0, j].imshow(real_img)
            axs[0, j].axis('off')

            axs[1, j].set_title(f"D: {fake_logits[0,0]:.2E}")
            axs[1, j].imshow(fake_img)
            axs[1, j].axis('off')

        fig.tight_layout()
        fig.savefig(f"reports/texture_examples_{cfg.train.name}.pdf")

def plot_height_examples(cfg: DictConfig, device=None):
    torch.manual_seed(1234)

    height_map = HeightMap(cfg.model.image_res, device=device) 
    noise = Noise(cfg.model.noise_features, cfg.model.noise_res, device=device)

    generator = Generator(
        cfg.model.hidden_features,
        cfg.model.noise_features,
        cfg.model.output_features,
        cfg.model.num_hidden_layers,
    ).to(device)
    discriminator = Discriminator(cfg.model.output_features).to(device)

    save_dir: Path = Path("models") / cfg.train.name
    epochs = sorted(saved_epochs(save_dir))

    with torch.no_grad():

        fig = plt.figure(figsize=(24, 6))
        axs = fig.subplots(2, len(epochs))

        for j, epoch in enumerate(epochs):

            load_models(
                save_dir,
                generator,
                discriminator,
                epoch,
            )
            
            real_patches, conditions = height_map.get_batch(1)
            fake_patches = generator(conditions, noise, random_patch_coords(1, cfg.model.image_res, device)).permute([0, 3, 1, 2])

            real_logits, _ = discriminator(conditions, real_patches)
            fake_logits, _ = discriminator(conditions, fake_patches)

            real_img = real_patches[0,:,:,:].permute([1, 2, 0]).cpu().numpy()
            fake_img = fake_patches[0,:,:,:].permute([1, 2, 0]).cpu().numpy()

            axs[0, j].set_title(f"D: {real_logits[0,0]:.2E}")
            axs[0, j].imshow(real_img, vmin=-4.0, vmax=4.0)
            axs[0, j].axis('off')

            axs[1, j].set_title(f"D: {fake_logits[0,0]:.2E}")
            axs[1, j].imshow(fake_img, vmin=-4.0, vmax=4.0)
            axs[1, j].axis('off')

        fig.tight_layout()
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
    
    noise = Noise(cfg.model.noise_features, cfg.model.noise_res, device=device)

    coords = torch.stack(torch.meshgrid(
        torch.linspace(0.0, 1.0, cfg.model.image_res, device=device),
        torch.linspace(0.0, 1.0, cfg.model.image_res, device=device),
        indexing='xy',
    )).permute([1, 2, 0])

    noise_coords = generator.noise_transforms(coords)
    heights = noise(noise_coords)

    fig = plt.figure()
    ax = fig.subplots(2, 4)
    for i in range(2):
        for j in range(4):
            idx = i * 4 + j
            ax[i, j].imshow(heights[...,idx].detach().cpu().numpy(), vmin=-2.0, vmax=2.0)
    fig.tight_layout()
    fig.savefig(f"reports/noise_transforms_{cfg.train.name}.pdf")

def plot_noise_transforms(cfg: DictConfig, device = None):
    noise_transforms = NoiseTransforms(cfg.model.noise_features).to(device)
    noise = Noise(cfg.model.noise_features, cfg.model.noise_res, device=device)

    coords = torch.stack(torch.meshgrid(
        torch.linspace(0.0, 1.0, cfg.model.image_res, device=device),
        torch.linspace(0.0, 1.0, cfg.model.image_res, device=device),
        indexing='xy',
    )).permute([1, 2, 0])

    noise_coords = noise_transforms(coords)
    heights = noise(noise_coords)

    fig = plt.figure()
    ax = fig.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            ax[i, j].imshow(heights[...,idx].detach().cpu().numpy(), vmin=-2.0, vmax=2.0)
            ax[i, j].axis('off')
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

def saved_epochs(save_dir: Path) -> list[int]:
    saved_epochs = []    
    for p in save_dir.iterdir():
        match = re.match(r"train_state_(\d+)", p.stem)
        if match is None:
            continue
        saved_epochs.append(int(match.group(1)))

    return saved_epochs


def latest_epoch(save_dir: Path) -> int | None:
    return max(saved_epochs(save_dir), default=None)

@hydra.main(version_base=None, config_path="..", config_name="config")
def all_plots(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # plot_sample_bilinear()
    plot_noise_transforms(cfg, device)
    # plot_generator_noise_transforms(generator, cfg, device)
    # plot_height_examples(cfg, device)
    # plot_texture_examples(cfg, device)
    # plot_losses(cfg)

if __name__ == "__main__":
    all_plots()