"""
Christopher Ondrusz
GitHub: acse_cro23
"""
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    A convolutional autoencoder neural network for image compression and
    reconstruction.
    The autoencoder consists of an encoder that compresses the input image
    into a lower-dimensional representation and a decoder that reconstructs
    the image from this representation.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3,
                      stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2,
                               padding=1, output_padding=1)
        )

    def forward(self, x):
        """
        Defines the forward pass of the autoencoder, where the input image
        is passed through the encoder to obtain a compressed representation
        and then through the decoder to reconstruct the image.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor of shape (batch_size, 1, height, width)
            representing grayscale images.

        Returns:
        --------
        torch.Tensor
            The reconstructed image tensor of the same shape as the input.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x
