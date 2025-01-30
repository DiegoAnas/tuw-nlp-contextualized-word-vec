import torch
from torch import nn


class Maxout(nn.Module):
    def __init__(self, layer_input_dim, layer_output_dim, pool_size=2):
        """
        Implements a Maxout layer.
        
        Args:
            layer_input_dim (int): Number of input features.
            layer_output_dim (int): Number of output features.
            pool_size (int): Number of linear transformations to take the max over.
        """
        super().__init__()
        self.input_dim = layer_input_dim
        self.output_dim = layer_output_dim
        self.pool_size = pool_size
        self.fc = nn.Linear(self.input_dim, self.output_dim * self.pool_size)

    def forward(self, x):
        """
        Forward pass for the Maxout layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, output_dim).
        """
        assert self.pool_size > 1, "Pool size for Maxout must be greater than 1."
        # Apply the linear transformation
        affine_output = self.fc(x)
        # Reshape for max pooling
        shape = list(x.size())
        shape[-1] = self.output_dim
        shape.append(self.pool_size)
        # Max pooling over the last dimension
        maxed_output = torch.max(affine_output.view(*shape), dim=-1)[0]
        return maxed_output


# Test Maxout functionality
if __name__ == "__main__":
    # Test Maxout
    batch_size, input_dim, output_dim, pool_size = 32, 128, 64, 2
    x = torch.randn(batch_size, input_dim)
    maxout_layer = Maxout(input_dim, output_dim, pool_size)
    output = maxout_layer(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    assert output.shape == (batch_size, output_dim), "Maxout output shape mismatch!"
