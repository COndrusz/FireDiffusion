"""
Christopher Ondrusz
GitHub: acse_cro23
"""
import torch.nn as nn


class FullyConnectedAutoencoder(nn.Module):
    """
    A fully connected autoencoder neural network for image compression and
    reconstruction.
    The autoencoder compresses the input image into a lower-dimensional
    representation using fully connected layers and then reconstructs the
    image from this representation.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    def __init__(self):
        super(FullyConnectedAutoencoder, self).__init__()

        input_size = 128 * 128

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, input_size)
        )

    def forward(self, x):
        """
        Defines the forward pass of the fully connected autoencoder. The input
        image is first flattened, then passed through the encoder to obtain a
        compressed representation, and finally passed through the decoder to
        reconstruct the image.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor of shape (batch_size, 1, 128, 128) representing
            grayscale images.

        Returns:
        --------
        torch.Tensor
            The reconstructed image tensor of shape (batch_size, 1, 128, 128).
        """
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 1, 128, 128)
        return x
