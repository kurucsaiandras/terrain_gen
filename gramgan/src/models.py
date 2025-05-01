import torch
import torch.nn as nn
import torch.nn.functional as F

class Noise(nn.Module):
   
    def __init__(self, features: int, resolution: int, device = None):
        super().__init__()

        self.image = torch.randn(features, resolution, resolution, device=device)


    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return sample_bilinear(self.image, coords)

def sample_bilinear(image: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    :param image: image to interpolate from [features, height, width]
    :param coords: coordinates at which to sample the image [*, features, 2]
    :return: interpolated image values at coords [*, features]
    """

    (_, height, width) = image.size()

    x = coords[...,0] - 0.5
    y = coords[...,1] - 0.5

    x0 = x.floor().to(torch.int32)
    x1 = x0 + 1
    y0 = y.floor().to(torch.int32)
    y1 = y0 + 1

    xw = x - x0
    yw = y - y0
    
    x0 = x0 % width
    x1 = x1 % width
    y0 = y0 % height
    y1 = y1 % height
    
    # all needed image values
    feature_indices = torch.arange(image.shape[0], device=image.device)
    feature_indices = feature_indices.view(*(1,) * (coords.dim() - 2) + (-1,))
    
    i00 = image[feature_indices, y0, x0]
    i01 = image[feature_indices, y0, x1]
    i10 = image[feature_indices, y1, x0]
    i11 = image[feature_indices, y1, x1]

    # interpolate x    
    i0 = torch.lerp(i00, i01, xw)
    i1 = torch.lerp(i10, i11, xw)
    
    # interpolate y
    i = torch.lerp(i0, i1, yw)

    return i

class NoiseTransforms(nn.Module):
    def __init__(self, noise_features: int):
        super().__init__()
        self.noise_features = noise_features
        
        angles = 2.0 * torch.pi * torch.rand(noise_features)

        rotations = torch.empty(noise_features, 2, 2)
        rotations[:,0,0] = torch.cos(angles)
        rotations[:,0,1] = torch.sin(angles)
        rotations[:,1,0] = -torch.sin(angles)
        rotations[:,1,1] = torch.cos(angles)
        
        shears = torch.empty(noise_features, 2, 2)
        shears[:,0,0] = 1.0
        shears[:,0,1] = 0.0
        shears[:,1,0] = torch.randn(noise_features)
        shears[:,1,1] = 1.0

        transforms = torch.matmul(rotations, shears)

        for i in range(noise_features):
            transforms[i,:,:] *= 0.25 * 2.0 ** (0.5 * i)

        self.transformations = nn.Parameter(transforms)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        :param coords: coordinates to transform [*, 2]
        :return: transformed coordinates ([*, noise_features, 2])
        """
        return torch.einsum('...n,Mmn->...Mm', coords, self.transformations)

class NoiseScales(nn.Module):

    def __init__(self, noise_features: int):
        super().__init__()
        scales_init = torch.tensor([2.0 ** i for i in range(noise_features)])
        scales_init = scales_init.view([noise_features, 1])
        self.scales = nn.Parameter(scales_init)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return coords.unsqueeze(-2) * self.scales


class GeneratorLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int, noise_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear_noise = nn.Linear(noise_features, out_features, bias=False)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x) + self.linear_noise(noise))

class Generator(nn.Module):
    
    def __init__(self, hidden_features: int, noise_features: int, out_features: int):
        super().__init__()
        self.input = nn.Parameter(torch.ones(hidden_features))
        self.noise_transforms = NoiseTransforms(noise_features)
        # self.noise_transforms = NoiseScales(noise_features)
        self.noise_features_per_layer = noise_features // 4
        self.noise_layers = nn.ModuleList(
            [GeneratorLayer(hidden_features, hidden_features, self.noise_features_per_layer) for _ in range(4)],
        ) 
        self.tail = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.LeakyReLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, noise: Noise, coords: torch.Tensor) -> torch.Tensor:
        """
        :param noise: an instance of the Noise class
        :param coords: coordinates of sampled points [..., 2]
        :return: height [..., out_features] 
        """
        
        noise_coords = self.noise_transforms(coords)
        noise_values = noise(noise_coords)

        x = self.input

        for i, layer in enumerate(self.noise_layers):
            noise_layer_values = noise_values[...,i*self.noise_features_per_layer:(i+1)*self.noise_features_per_layer]
            x = layer(x, noise_layer_values)

        return self.tail(x)


class Discriminator(nn.Module):
    def __init__(self, image_features: int):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(image_features, 32, 3, padding=1),
            nn.Conv2d( 32,  64, 3, padding=1),
            nn.Conv2d( 64, 128, 3, padding=1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.Conv2d(512, 512, 3, padding=1),
        ])

        self.tail = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        extracted_features = []

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x)
            extracted_features.append(x)
            x = F.avg_pool2d(x, 2)

        x = F.adaptive_avg_pool2d(x, output_size=1)

        return self.tail(x), extracted_features
