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
            transforms[i,:,:] *= 2.0 ** (i - 4.0)

        self.transformations = nn.Parameter(transforms)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        :param coords: coordinates to transform [*, 2]
        :return: transformed coordinates ([*, noise_features, 2])
        """
        return torch.einsum('...n,Mmn->...Mm', coords, self.transformations)


class GeneratorLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int, noise_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear_noise = nn.Linear(noise_features, out_features, bias=False)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        :param x: activation from previous layer [batch_size, ..., hidden_size]
        :param noise: noise values [batch_size, ..., noise_features]
        :return: activation of this layer []
        """

        return self.activation(self.linear(x) + self.linear_noise(noise))

class LayerConditionMap(nn.Module):
    def __init__(
        self,
        hidden_features: int,
    ):
        super().__init__()
        self.hidden_features = hidden_features    
        self.map = nn.Sequential(
            nn.Linear(64, 2*hidden_features),
            nn.ReLU(),
            nn.Linear(2*hidden_features, 2*hidden_features),
        ) 

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.map(condition)
        return x[:,:self.hidden_features], x[:,self.hidden_features:]


class Generator(nn.Module):
    
    def __init__(
        self,
        hidden_features: int,
        noise_features: int,
        out_features: int,
        num_hidden_layers: int,
    ):
        super().__init__()

        self.hidden_features = hidden_features
        self.noise_features = noise_features

        self.input = torch.nn.Parameter(torch.ones(hidden_features))
        self.noise_transforms = NoiseTransforms(noise_features)
        self.noise_features_per_layer = noise_features // num_hidden_layers
        
        self.condition_head = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.layer_condition_maps = nn.ModuleList([
            LayerConditionMap(hidden_features) for _ in range(num_hidden_layers)
        ])

        self.noise_layers = nn.ModuleList([
            GeneratorLayer(hidden_features, hidden_features, self.noise_features_per_layer) for _ in range(num_hidden_layers)
        ]) 
        self.tail = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.LeakyReLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, conditions: torch.Tensor, noise: Noise, coords: torch.Tensor) -> torch.Tensor:
        """
        :param noise: an instance of the Noise class
        :param conditions: condition vectors [batch_size, 9]
        :param coords: coordinates of sampled points [batch_size, ..., 2]
        :return: values [batch_size, ..., out_features]
        """
        noise_coords: torch.Tensor = self.noise_transforms(coords) # [batch_size, ..., noise_features, 2]
        noise_values: torch.Tensor = noise(noise_coords) #[batch_size, ..., noise_features]

        x = self.input

        z = self.condition_head(conditions)
        layer_conditions = [map(z) for map in self.layer_condition_maps]
        
        batch_size = coords.shape[0]
        weights_bias_shape = [batch_size] + [1] * (coords.dim() - 2) + [self.hidden_features]

        k = self.noise_features_per_layer
        for i, (layer, (weights, biases)) in enumerate(zip(self.noise_layers, layer_conditions)):
            noise_layer_values = noise_values[...,i*k:(i+1)*k] # [batch_size, ..., k]

            weights = weights.view(weights_bias_shape)
            biases = biases.view(weights_bias_shape)

            x = weights * layer(x, noise_layer_values) + biases

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

        self.linear_features = nn.Linear(512, 256 + 128)
        self.linear_conditions = nn.Linear(9, 128)

        self.tail = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, conditions: torch.Tensor, image: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        extracted_features = []

        x = image
        # [1, 128, 128] -> [512, 1, 1]
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x)
            extracted_features.append(x)
            x = F.avg_pool2d(x, 2)

        x = x.flatten(start_dim=1)
        x = self.linear_features(x)
        
        y = conditions
        y = self.linear_conditions(y)

        z = torch.cat([x, y], dim=-1)

        return self.tail(z), extracted_features
