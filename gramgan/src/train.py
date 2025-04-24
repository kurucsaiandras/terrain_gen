
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
    interp_logits = discriminator(interp_patches)[:,0]

    gradients = torch.autograd.grad(
        outputs=interp_logits,
        inputs=interp_patches,
        grad_outputs=torch.ones(interp_logits.size(), device=device),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    
    return ((gradients.norm(2, dim=1) - 1.0)**2).mean()

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
    real_logits: torch.Tensor = discriminator(real_patches)
    real_loss = -real_logits.mean()
    real_loss.backward()
    discriminator_loss += real_loss.item()

    coords = random_patch_coords(cfg, device=device)
    fake_patches = generator(noise, coords).permute([0, 3, 1, 2])
    fake_logits: torch.Tensor = discriminator(fake_patches)
    fake_loss = fake_logits.mean()
    fake_loss.backward()
    discriminator_loss += fake_loss.item()

    penalty = gradient_penalty(discriminator, real_patches, fake_patches)
    penalty_loss = cfg.train.gp_weight * penalty 
    penalty_loss.backward()
    discriminator_loss += penalty_loss.item()

    optimizer.step()


@hydra.main(version_base=None, config_path="..", config_name="config")
def train(cfg: DictConfig):
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    height_map = RealHeightMap(device=device) 

    generator = Generator(cfg.model.hidden_features, cfg.model.noise_features).to(device)
    discriminator = Discriminator().to(device)
    
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-3)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    for epoch in range(cfg.train.epochs):

        if epoch % cfg.train.save_frequency == 0:
            torch.save(generator.state_dict(), f"models/generator_{cfg.train.name}_{epoch}.pt")
            torch.save(discriminator.state_dict(), f"models/discriminator_{cfg.train.name}_{epoch}.pt")

        for p in generator.parameters():
            p.requires_grad = False

        for _ in range(cfg.train.discriminator_iters):
            discriminator_iter(generator, discriminator, height_map, optimizer_d, cfg, device)
        
        for p in generator.parameters():
            p.requires_grad = True


        for p in discriminator.parameters():
            p.requires_grad = False

        generator.zero_grad()
    
        noise = noise_image(cfg, device=device)
        patches = generator(noise, random_patch_coords(cfg, device))
        patches = patches.permute([0, 3, 1, 2])
        loss: torch.Tensor = -discriminator(patches).mean()
        loss.backward()

        optimizer_g.step()
        
        for p in discriminator.parameters():
            p.requires_grad = True


    torch.save(generator.state_dict(), f"models/generator_{cfg.train.name}.pt")
    torch.save(discriminator.state_dict(), f"models/discriminator_{cfg.train.name}.pt")
    

if __name__ == "__main__":
    train()