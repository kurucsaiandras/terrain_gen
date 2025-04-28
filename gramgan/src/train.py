from pathlib import Path
import re

import torch
import hydra
from omegaconf import DictConfig

from models import Generator, Discriminator
from data import RealHeightMap

def noise_image(cfg: DictConfig, device = None) -> torch.Tensor:
    return torch.randn(cfg.model.noise_features, cfg.model.noise_res, cfg.model.noise_res, device=device)

def random_patch_coords(cfg: DictConfig, device = None) -> torch.Tensor:
    coords = torch.stack(torch.meshgrid(
        torch.linspace(0.0, 1.0, cfg.model.image_res, device=device),
        torch.linspace(0.0, 1.0, cfg.model.image_res, device=device),
        indexing='xy',
    )).permute([1, 2, 0])

    translations = torch.rand(cfg.train.batch_size, 1, 1, 2, device=device) * 2.0 - 1.0
    
    return coords + translations

def gradient_penalty(
    discriminator: Discriminator,
    real_patches: torch.Tensor,
    fake_patches: torch.Tensor,
) -> torch.Tensor:
    device = real_patches.device
    batch_size = real_patches.shape[0]

    theta = torch.rand(batch_size, 1, 1, 1, device=device)
    interp_patches = torch.lerp(real_patches, fake_patches, theta)
    interp_patches.requires_grad = True
    interp_logits = discriminator(interp_patches)[0][:,0]

    gradients = torch.autograd.grad(
        outputs=interp_logits,
        inputs=interp_patches,
        grad_outputs=torch.ones(interp_logits.size(), device=device),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    
    return ((gradients.norm(2, dim=1) - 1.0)**2).mean()

def gram(features: torch.Tensor) -> torch.Tensor:
    """
    :param features: a batch of feature tensors [batch_size, channels, height, width]
    :return: a gram matrix for the features [batch_size, channels, channels]
    """
    batch_size, channels, height, width = features.size()
    num_pixels = height * width

    features = features.view(batch_size, channels, num_pixels)
    return torch.matmul(features, features.transpose(1, 2)) / num_pixels


def style_loss(real_features: list[torch.Tensor], fake_features: list[torch.Tensor]) -> torch.Tensor:
    """
    :param real_features: a list of feature tensors [batch_size, channels, height, width]
    """
    total = 0.0
    for r, f in zip(real_features, fake_features):
        total = total + torch.nn.functional.l1_loss(gram(f), gram(r))

    return total / len(real_features)

def discriminator_iter(
        generator: Generator,
        discriminator: Discriminator,
        height_map: RealHeightMap,
        optimizer: torch.optim.Optimizer, 
        cfg: DictConfig,
        device = None
):
    discriminator.zero_grad()
    discriminator_loss = 0.0

    noise = noise_image(cfg, device=device)

    real_patches = height_map.get_patches(cfg.train.batch_size, cfg.model.image_res)
    real_logits, _ = discriminator(real_patches)
    real_loss = -real_logits.mean()
    real_loss.backward()
    discriminator_loss += real_loss.item()

    coords = random_patch_coords(cfg, device=device)
    fake_patches = generator(noise, coords).permute([0, 3, 1, 2])
    fake_logits, _ = discriminator(fake_patches)
    fake_loss = fake_logits.mean()
    fake_loss.backward()
    discriminator_loss += fake_loss.item()

    penalty = gradient_penalty(discriminator, real_patches, fake_patches)
    penalty_loss = cfg.train.gp_weight * penalty 
    penalty_loss.backward()
    discriminator_loss += penalty_loss.item()

    optimizer.step()

def save_train_state(
    save_dir: Path,
    generator: Generator,
    discriminator: Discriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int | None = None,
):
    torch.save({
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict()
    }, save_dir / f"train_state_{epoch}.pt")

def load_train_state(
    save_dir: Path,
    generator: Generator,
    discriminator: Discriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int,
):
    dict = torch.load(save_dir / f"train_state_{epoch}.pt")

    generator.load_state_dict(dict["generator"])
    discriminator.load_state_dict(dict["discriminator"])
    optimizer_g.load_state_dict(dict["optimizer_g"])
    optimizer_d.load_state_dict(dict["optimizer_d"])
    
def find_latest_epoch(save_dir: Path) -> int | None:
    saved_epochs = [int(re.match(r"train_state_(\d+)", p.stem).group(1)) for p in save_dir.iterdir()]
    return max(saved_epochs, default=None)

@hydra.main(version_base=None, config_path="..", config_name="config")
def train(cfg: DictConfig):
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    save_dir: Path = Path("models") / cfg.train.name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"saving intermediate models to {save_dir}")

    height_map = RealHeightMap(device=device) 
    
    generator = Generator(cfg.model.hidden_features, cfg.model.noise_features, cfg.model.output_features).to(device)
    discriminator = Discriminator(cfg.model.output_features).to(device)
    
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-3)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    epoch_range = range(cfg.train.epochs)

    if cfg.train.resume:
        start_epoch = find_latest_epoch(save_dir)
        if start_epoch is not None:

            epoch_range = range(start_epoch, start_epoch + cfg.train.epochs)
            load_train_state(
                save_dir,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                start_epoch,
            )

    print(f"starting training from epoch {epoch_range.start}")
    
    for epoch in epoch_range:

        if epoch % cfg.train.save_frequency == 0:
            print(f"finished {epoch} epochs, saving train state")
            save_train_state(
                save_dir,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                epoch=epoch,
            )

        # disable gradient computation for generator
        for p in generator.parameters():
            p.requires_grad = False

        for _ in range(cfg.train.discriminator_iters):
            discriminator_iter(generator, discriminator, height_map, optimizer_d, cfg, device)
        
        # reenable gradient computation for generator
        for p in generator.parameters():
            p.requires_grad = True

        # disable gradient computation for discriminator
        for p in discriminator.parameters():
            p.requires_grad = False

        generator.zero_grad()
    
        real_patches = height_map.get_patches(cfg.train.batch_size, cfg.model.image_res)
        _, real_features = discriminator(real_patches)

        noise = noise_image(cfg, device=device)
        patches: torch.Tensor = generator(noise, random_patch_coords(cfg, device))
        patches = patches.permute([0, 3, 1, 2])
        logits, features = discriminator(patches)
        
        wgan = -logits.mean()
        style = style_loss(real_features, features)

        loss = cfg.train.wgan_weight * wgan + cfg.train.style_weight * style
        loss.backward()

        optimizer_g.step()
        
        # reenable gradient computation for discriminator
        for p in discriminator.parameters():
            p.requires_grad = True

    print(f"finished training at epoch {epoch_range.stop}")

    save_train_state(
        save_dir,
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        epoch=epoch_range.stop,
    )

if __name__ == "__main__":
    train()
