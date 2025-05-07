from pathlib import Path
import re

import torch
import hydra
from omegaconf import DictConfig

from models import Noise, Generator, Discriminator
from data import HeightMap, Texture


def random_patch_coords(batch_size: int, image_res: int, device = None) -> torch.Tensor:
    coords = torch.stack(torch.meshgrid(
        torch.linspace(0.0, 1.0, image_res, device=device),
        torch.linspace(0.0, 1.0, image_res, device=device),
        indexing='xy',
    )).permute([1, 2, 0])

    translations = torch.rand(batch_size, 1, 1, 2, device=device) * 16.0
    
    return coords + translations

def gradient_penalty(
    discriminator: Discriminator,
    conditions: torch.Tensor,
    real_patches: torch.Tensor,
    fake_patches: torch.Tensor,
) -> torch.Tensor:
    device = real_patches.device
    batch_size = real_patches.shape[0]

    theta = torch.rand(batch_size, 1, 1, 1, device=device)
    interp_patches = torch.lerp(real_patches, fake_patches, theta)
    interp_patches.requires_grad = True
    interp_logits = discriminator(conditions, interp_patches)[0][:,0]

    gradients = torch.autograd.grad(
        outputs=interp_logits,
        inputs=interp_patches,
        grad_outputs=torch.ones(interp_logits.size(), device=device),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    
    return ((gradients.norm(2, dim=1) - 1.0 + 1e-4)**2).mean()

def discriminator_loss(
    discriminator: Discriminator,
    conditions: torch.Tensor,
    real_patches: torch.Tensor,
    fake_patches: torch.Tensor,
    cfg: DictConfig,
) -> torch.Tensor:
    
    real_logits, _real_features = discriminator(conditions, real_patches)
    fake_logits, _fake_features = discriminator(conditions, fake_patches)

    wgan = fake_logits.mean() - real_logits.mean()
    gp = gradient_penalty(discriminator, conditions, real_patches, fake_patches)

    return wgan + cfg.train.gp_weight * gp 


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

def generator_loss(
    discriminator: Discriminator,
    conditions: torch.Tensor,
    real_patches: torch.Tensor,
    fake_patches: torch.Tensor,
    cfg: DictConfig,
) -> torch.Tensor:

    _real_logits, real_features = discriminator(conditions, real_patches)
    fake_logits, fake_features = discriminator(conditions, fake_patches)

    wgan = -fake_logits.mean()
    style = style_loss(real_features, fake_features)

    return cfg.train.wgan_weight * wgan + cfg.train.style_weight * style


def save_train_state(
    save_dir: Path,
    generator: Generator,
    discriminator: Discriminator,
    epoch: int | None = None,
):
    torch.save({
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
    }, save_dir / f"train_state_{epoch}.pt")

@hydra.main(version_base=None, config_path="..", config_name="config")
def train(cfg: DictConfig):
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    save_dir: Path = Path("models") / cfg.train.name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"saving intermediate models to {save_dir}")

    height_map = HeightMap(cfg.model.image_res, device=device) 
    
    generator = Generator(
        cfg.model.hidden_features,
        cfg.model.noise_features,
        cfg.model.output_features,
        cfg.model.num_hidden_layers,
    ).to(device)
    discriminator = Discriminator(cfg.model.output_features).to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), cfg.train.generator_lr, betas=(0.0, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), cfg.train.discriminator_lr, betas=(0.0, 0.999))

    stats = {
        "generator_loss":  torch.empty(cfg.train.epochs),
        "discriminator_loss":  torch.empty(cfg.train.epochs, cfg.train.discriminator_iters),
    }

    for epoch in range(cfg.train.epochs):

        if epoch % cfg.train.save_frequency == 0:
            print(f"finished {epoch} epochs, saving train state")
            save_train_state(
                save_dir,
                generator,
                discriminator,
                epoch=epoch,
            )
            torch.save({key: stat[:epoch] for key, stat in stats.items()}, save_dir / "stats.pt")

        for i in range(cfg.train.discriminator_iters):
            real_patches, conditions = height_map.get_batch(cfg.train.batch_size)
            noise = Noise(cfg.model.noise_features, cfg.model.noise_res, device=device)
            patch_coords = random_patch_coords(cfg.train.batch_size, cfg.model.image_res, device=device)
            fake_patches = generator(conditions, noise, patch_coords)
            fake_patches = fake_patches.detach().permute([0, 3, 1, 2])
    
            loss_d = discriminator_loss(discriminator, conditions, real_patches, fake_patches, cfg)
            loss_d.backward()
            optimizer_d.step()
            discriminator.zero_grad()
            stats["discriminator_loss"][epoch, i] = loss_d.item()

        # disable gradient computation for discriminator
        discriminator.eval()
        for p in discriminator.parameters():
            p.requires_grad = False

        real_patches, conditions = height_map.get_batch(cfg.train.batch_size)
        noise = Noise(cfg.model.noise_features, cfg.model.noise_res, device=device)
        patch_coords = random_patch_coords(cfg.train.batch_size, cfg.model.image_res, device=device)
        fake_patches = generator(conditions, noise, patch_coords)
        fake_patches = fake_patches.permute([0, 3, 1, 2])

        loss_g = generator_loss(discriminator, conditions, real_patches, fake_patches, cfg)
        loss_g.backward()
        optimizer_g.step()
        generator.zero_grad()
        stats["generator_loss"][epoch] = loss_g.item()
        
        # reenable gradient computation for discriminator
        discriminator.train()
        for p in discriminator.parameters():
            p.requires_grad = True

    print(f"finished training")

    save_train_state(
        save_dir,
        generator,
        discriminator,
        epoch=cfg.train.epochs,
    )

    torch.save(stats, save_dir / "stats.pt")

if __name__ == "__main__":
    train()
