import torch.nn as nn
import torch
from horizon_plugin_pytorch.quantization import QuantStub

class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale: Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))
        self.scale_quant_stub = QuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass through the Scale module.

        Args:
            x (torch.Tensor): Input tensor to be scaled.

        Returns:
            torch.Tensor: Scaled output tensor.
        """
        scale = self.scale_quant_stub(self.scale)
        return x * scale