"""Low-rank adapters for the two frozen backbones.

Ref: Sec. 2.2 -- "Low-rank adapters of rank ``r`` have jointly adapted two
pretrained backbones ... Adaptation has been limited to the adaptors, rendering
the typical backbones static."

Both an adapted linear projection and an adapted convolution are provided,
because the front end's backbones are convolutional while the response head's
projections are linear. Each adapter is a two-factor parameterisation of the
update, ``dW = (alpha / r) B A``, added to a frozen base; the rank is the only
tunable width, so an adaptation costs ``r * (in + out)`` parameters per wrapped
operator.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(frozen=True)
class LoRAConfig:
    """Rank, scaling and dropout of the adapters."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0

    @property
    def scaling(self) -> float:
        return self.alpha / float(self.rank)


class LoRALinear(nn.Module):
    """A frozen linear projection with a trainable low-rank update."""

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.config = config
        self.base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, config.rank, bias=False)
        self.up = nn.Linear(config.rank, base.out_features, bias=False)
        self.dropout: nn.Module = (
            nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
        )
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        frozen: Tensor = self.base(inputs)
        adapted: Tensor = self.up(self.down(self.dropout(inputs))) * self.config.scaling
        return frozen + adapted

    def adapter_parameters(self) -> int:
        return int(self.down.weight.numel() + self.up.weight.numel())


class LoRAConv2d(nn.Module):
    """A frozen convolution with a trainable rank-``r`` 1x1 update.

    A rank-r factorisation of a 1x1 convolution is the convolutional analogue of
    the low-rank weight update: the frozen operator keeps its receptive field and
    the adapter adds a low-rank channel mixing that the forward pass really uses.
    """

    def __init__(self, base: nn.Conv2d, config: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.config = config
        self.base.requires_grad_(False)
        self.down = nn.Conv2d(base.in_channels, config.rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(config.rank, base.out_channels, kernel_size=1, bias=False)
        self.dropout: nn.Module = (
            nn.Dropout2d(config.dropout) if config.dropout > 0.0 else nn.Identity()
        )
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        frozen: Tensor = self.base(inputs)
        adapted: Tensor = self.up(self.down(self.dropout(inputs))) * self.config.scaling
        return frozen + adapted

    def adapter_parameters(self) -> int:
        return int(self.down.weight.numel() + self.up.weight.numel())


def adapt_conv2d(module: nn.Module, config: LoRAConfig) -> nn.Module:
    """Replace every 3x3 convolution of a module with an adapted one, in place."""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and child.kernel_size == (3, 3):
            setattr(module, name, LoRAConv2d(child, config))
        else:
            adapt_conv2d(child, config)
    return module


def count_adapter_parameters(module: nn.Module) -> int:
    """Trainable parameters of a module; for an adapted backbone these are the adapters."""
    return int(
        sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    )


def freeze_parameters(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module
