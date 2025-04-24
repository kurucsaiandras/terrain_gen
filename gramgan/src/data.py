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
