from pathlib import Path

import torch
import rasterio
from torchvision.transforms import v2

class RealHeightMap:
    def __init__(self, tiff_path: Path = Path("exemplars/yosemite.tif"), device=None):
        with rasterio.open(tiff_path) as dataset:
            self.image = torch.tensor(dataset.read(1), device=device).unsqueeze(0)
        
        self.image -= self.image.mean()
        self.image /= self.image.std()

    def get_patches(self, batch_size: int, resolution: int) -> torch.Tensor:

        transforms = v2.Compose([  
            v2.Grayscale(),
            v2.RandomCrop(resolution),
        ])

        patches = torch.empty(
            [batch_size, 1, resolution, resolution],
            device=self.image.device
        )

        for i in range(batch_size):
            patches[i,:,:,:] = transforms(self.image)

        return patches

    def get_patches_and_conditions(self, batch_size: int, resolution: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param batch_size: number of patches to return
        :param resolution: size of patches to return
        :return: tuple with patches and conditions ([batch_size, 1, resolution, resolution], [batch_size, 4])
        """

        patches = self.get_patches(batch_size, resolution)

        grad_x = (patches[:,:,1:-1,2:] - patches[:,:,1:-1,:-2]) * (128.0 / 2.0)
        grad_y = (patches[:,:,2:,1:-1] - patches[:,:,:-2,1:-1]) * (128.0 / 2.0)
        grad_norm = (grad_x * grad_x + grad_y * grad_y).sqrt()

        conditions = torch.stack([c.mean(dim=(1, 2, 3)) for c in [patches, grad_x, grad_y, grad_norm]])

        return patches, conditions

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    height_map = RealHeightMap(device=device)
    
    patches, conditions = height_map.get_patches_and_conditions(16, 128)

    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(1, 4)

    for j in range(4):
        condition_text = (
            f"mean: {conditions[j,0]:.2E}\n"
            f"grad_x: {conditions[j,1]:.2E}\n"
            f"grad_y: {conditions[j,2]:.2E}\n"
            f"slope: {conditions[j,3]:.2E}"
        )

        axs[j].imshow(patches[j,0,:,:].cpu().numpy(), vmin=-2.0, vmax=2.0)
        axs[j].text(
            64.0,
            -10.0,
            condition_text,
            horizontalalignment='center'
        )
        axs[j].axis('off')
    fig.tight_layout()
    fig.savefig(f"reports/conditions.pdf")
