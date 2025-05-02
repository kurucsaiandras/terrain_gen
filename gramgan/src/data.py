from pathlib import Path

import torch
import rasterio
from torchvision.transforms import v2

class RealHeightMap:
    def __init__(self, tile_size: int, tiff_path: Path = Path("exemplars/yosemite.tif"), device=None):
        self.tile_size = tile_size
        
        with rasterio.open(tiff_path) as dataset:
            self.image = torch.tensor(dataset.read(1), device=device).unsqueeze(0)
        
        tile_count_x = self.image.shape[2] // tile_size
        tile_count_y = self.image.shape[1] // tile_size

        self.image = self.image[:,:tile_count_x*tile_size,:tile_count_y*tile_size]

        self.image -= self.image.mean()
        self.image /= self.image.std()

        self.tile_averages = torch.nn.functional.avg_pool2d(self.image, tile_size)

        self.vmin = 0.0
        self.vmax = 0.0

        for tile_y in range(tile_count_y):
            for tile_x in range(tile_count_x):
                tile = self.get_tile(tile_x, tile_y)
                tile -= self.tile_averages[:,tile_y,tile_x]

                self.vmin = min(self.vmin, tile.min().item())
                self.vmax = max(self.vmax, tile.max().item())

        

    def get_tile(self, tile_x: int, tile_y: int) -> torch.Tensor:
        x0 = tile_x * self.tile_size
        x1 = tile_x * self.tile_size + self.tile_size
        y0 = tile_y * self.tile_size
        y1 = tile_y * self.tile_size + self.tile_size
        return self.image[:,y0:y1,x0:x1]

    def get_conditions(self, tile_x: int, tile_y: int) -> torch.Tensor:
        return self.tile_averages[:,tile_y-1:tile_y+2,tile_x-1:tile_x+2].flatten()

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param batch_size: number of patches to return
        :param resolution: size of patches to return
        :return: tuple with patches and conditions ([batch_size, 1, tile_size, tile_size], [batch_size, 9])
        """

        tiles_x = torch.randint(1, self.image.shape[2] // self.tile_size - 1, [batch_size])
        tiles_y = torch.randint(1, self.image.shape[1] // self.tile_size - 1, [batch_size])

        patches = torch.empty([batch_size, 1, self.tile_size, self.tile_size], device=self.image.device)
        conditions = torch.empty([batch_size, 9], device=self.image.device)  

        for i, (tile_x, tile_y) in enumerate(zip(tiles_x, tiles_y)):
            patches[i,:,:,:] = self.get_tile(tile_x, tile_y)
            conditions[i,:] = self.get_conditions(tile_x, tile_y)

        return patches, conditions

if __name__ == "__main__":
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    height_map = RealHeightMap(tile_size=128, device=device)

    patch, conditions = height_map.get_batch(4)

    print(patch.mean(dim=(1, 2, 3)))
    print(conditions[:,4])

    print(height_map.vmin)
    print(height_map.vmax)
