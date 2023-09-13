"""
Code adapted from: https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

batch_size, feature_size = None, None


class SineLayer(nn.Module):
    """
    See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of
    omega_0.

    If is_first=True, omega_0 is a frequency factor which simply multiplies the
    activations before the nonlinearity. Different signals may require different omega_0
    in the first layer - this is a hyperparameter.

    If is_first=False, then the weights will be divided by omega_0 so as to keep the
    magnitude of activations constant, but boost gradients to the weight matrix (see
    supplement Sec. 1.5)
    """

    def __init__(
        self,
        n_input_dims: int,
        n_output_dims: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: int = 30,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.n_input_dims = n_input_dims
        self.linear = nn.Linear(n_input_dims, n_output_dims, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.n_input_dims, 1 / self.n_input_dims
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.n_input_dims) / self.omega_0,
                    np.sqrt(6 / self.n_input_dims) / self.omega_0,
                )

    def forward(self, coords: TensorType["batch_size", "feature_size"]):
        return torch.sin(self.omega_0 * self.linear(coords))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(
        self,
        n_input_dims: int,
        hidden_features: int,
        hidden_layers: int,
        n_output_dims: int,
        outermost_linear: bool = False,
        first_omega_0: int = 30,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                n_input_dims, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, n_output_dims)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    n_output_dims,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords: TensorType["batch_size", "feature_size"]):
        # coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output

    def forward_with_activations(
        self, coords: TensorType["batch_size", "feature_size"], retain_grad=False
    ):
        """Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!"""
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations["input"] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations[
                    "_".join((str(layer.__class__), "%d" % activation_count))
                ] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations["_".join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
