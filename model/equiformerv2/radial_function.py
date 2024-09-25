"""Code."""
import torch
import torch.nn as nn


class RadialFunction(nn.Module):
    """Define Class RadialFunction."""

    '''
        Contruct a radial function (linear layers + layer normalization + SiLU) given a list of channels
    '''

    def __init__(self, channels_list):
        """Run __init__ method."""
        # code.
        super().__init__()
        modules = []
        input_channels = channels_list[0]
        for i in range(len(channels_list)):
            if i == 0:
                continue

            modules.append(nn.Linear(input_channels, channels_list[i], bias=True))
            input_channels = channels_list[i]

            if i == len(channels_list) - 1:
                break

            modules.append(nn.LayerNorm(channels_list[i]))
            modules.append(torch.nn.SiLU())

        self.net = nn.Sequential(*modules)

    def forward(self, inputs):
        """Run forward method."""
        # code.
        return self.net(inputs)
