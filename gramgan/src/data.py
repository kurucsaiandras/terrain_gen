from pathlib import Path

import torch
import rasterio
from torchvision.transforms import v2

class RealHeightMap:
    def __init__(self, tile_size: int, tiff_path: Path = Path("exemplars/yosemite.tif"), device=None):
        self.tile_size = tile_size
        
        with rasterio.open(tiff_path) as dataset:
            self.image = torch.tensor(dataset.read(1), device=device).unsqueeze(0)
        
        self.image -= self.image.mean()
        self.image /= self.image.std()

        self.tile_averages = torch.nn.functional.avg_pool2d(self.image, tile_size)

    def get_patches(self, batch_size: int) -> torch.Tensor:
        """
        :param batch_size: number of patches to return
        :param resolution: size of patches to return
        :return: tuple with patches and conditions ([batch_size, 1, tile_size, tile_size], [batch_size, 9])
        """

        tiles_x = torch.randint(self.image.shape[2] // self.tile_size, [batch_size])
        tiles_y = torch.randint(self.image.shape[1] // self.tile_size, [batch_size])

        x0 = tiles_x * self.tile_size
        x1 = tiles_x * self.tile_size + self.tile_size
        y0 = tiles_y * self.tile_size
        y1 = tiles_y * self.tile_size + self.tile_size

        patches = torch.empty([batch_size, 1, self.tile_size, self.tile_size], device=self.image.device)  

        for i in range(batch_size):
            patches[i,:,:,:] = self.image[0,y0[i]:y1[i],x0[i]:x1[i]]


        return patches


