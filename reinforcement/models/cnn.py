from typing import Union, Tuple
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from numpy import ndarray, floor, ceil

ConvLayer = Tuple[nn.Conv2d, nn.LeakyReLU, nn.MaxPool2d]


class ImageEnvironment2D(nn.Module):
    def __init__(
        self,
        input_dim: Tuple[int, int, int] = (1, 84, 84),
        action_space: int = 16,
        conv_layers: int = 4,
        lr: float = 1e-3,
        optimizer: Optimizer = None,
    ):
        super(ImageEnvironment2D, self).__init__()

        self.input_dim = input_dim

        layers = [2 ** i for i in range(2, conv_layers + 2)]
        layers.insert(0, input_dim[0])

        layers_list = [
            self._build_conv_layer(inp, out) for inp, out in zip(layers[:-1], layers[1:])
        ]

        self.cnn = nn.Sequential(
            *[layer for cnn_layer in layers_list for layer in cnn_layer]
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._calculate_flatten_dim(input_dim, conv_layers), 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, action_space)
        )

        # Initializing Conv2D layers using xavier
        self.apply(self._init_xavier_leaky)
        self.optimizer = (
            optimizer if optimizer
            else torch.optim.Adam(self.parameters(), lr=lr, eps=1e-4, weight_decay=0.001)
        )

    @staticmethod
    def _build_conv_layer(input_dim: int, output_dim: int) -> ConvLayer:
        return (
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2)
        )

    @staticmethod
    def _calculate_flatten_dim(input_dim: Tuple[int], layers: int) -> int:
        pooling = 2**layers
        output_filters = 2**(layers + 1)
        return int(
            floor(input_dim[1] / pooling)
            * floor(input_dim[2] / pooling)
            * output_filters
        )

    @staticmethod
    def _normalize_input(x: Tensor) -> Tensor:
        return x.div(255)

    @staticmethod
    def _init_xavier_leaky(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(
                m.weight.data,
                gain=nn.init.calculate_gain('leaky_relu', 0.01)
            )

    @staticmethod
    def _to_tensor(x: Union[ndarray, Tensor]) -> Tensor:
        if isinstance(x, ndarray):
            x = torch.from_numpy(x.copy())
        return x

    def forward(self, x: Union[ndarray, Tensor]) -> Tensor:
        x = self._to_tensor(x)

        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        if x.shape[1:] != self.input_dim:
            raise ValueError(f'Input image must be {self.input_dim}')

        max_value = x.max()
        if max_value > 255:
            raise ValueError('Image values cannot be greater than 255')
        elif max_value > 1:
            x = self._normalize_input(x)

        out = self.cnn(x)
        out = self.dense(out)
        return out

    def predict(self, x: Union[ndarray, Tensor]) -> Tensor:
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
        return out

    def fit(
        self,
        inputs: Union[ndarray, Tensor],
        targets: Union[ndarray, Tensor],
        epochs: int,
        batch_size: int = 64,
    ):
        criterion = nn.MSELoss()

        inputs = self._to_tensor(inputs)
        targets = self._to_tensor(targets)

        for epoch in range(epochs):
            for batch in range(ceil(inputs.shape[0] / batch_size).astype(int)):
                input_var = torch.autograd.Variable(inputs[batch*batch_size:(batch+1)*batch_size])
                target_var = torch.autograd.Variable(targets[batch*batch_size:(batch+1)*batch_size])
                output = self.forward(input_var)

                # Backward pass
                loss = criterion(output, target_var)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def load_model(self, path: Union[Path, str]):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def save_model(self, path: Union[Path, str]):
        torch.save(self.state_dict(), path)