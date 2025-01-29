"""
Christopher Ondrusz
GitHub: acse_cro23
"""
import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    """
    A convolutional block that applies two convolutional layers with batch
    normalization and activation functions.

    Parameters:
    -----------
    in_channels : int
        The number of input channels for the convolutional layers.
    out_channels : int
        The number of output channels for the convolutional layers.

    Returns:
    --------
    torch.Tensor
        The output tensor after applying the convolutional block, with the
        same height and width as the input.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        """
        Forward pass through the convolutional block.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        --------
        torch.Tensor
            The output tensor of shape (batch_size, out_channels,
                                        height, width).
        """
        return self.conv(x)


class Bottleneck(nn.Module):
    """
    A bottleneck layer with a convolutional block and multi-head attention,
    followed by a 1x1 convolution to reduce the number of channels.

    Parameters:
    -----------
    embed_dim : int, optional, default=128
        The dimensionality of the embeddings used in the multi-head attention.
    num_heads : int, optional, default=8
        The number of heads in the multi-head attention mechanism.

    Returns:
    --------
    torch.Tensor
        The output tensor after the bottleneck, with 128 channels and the
        same spatial dimensions as the input.
    """
    def __init__(self, embed_dim=128, num_heads=8):
        super(Bottleneck, self).__init__()
        self.conv_block = ConvBlock(64, embed_dim)  # Embed_dim is 128
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                               num_heads=num_heads)
        self.conv1x1 = nn.Conv2d(embed_dim, 128, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the bottleneck layer.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor of shape (batch_size, 64, height, width).

        Returns:
        --------
        torch.Tensor
            The output tensor of shape (batch_size, 128, height, width).
        """
        x = self.conv_block(x)
        batch_size, channels, height, width = x.size()
        x_flat = x.view(batch_size, channels, -1).permute(2, 0, 1)
        attn_output, _ = self.attention(x_flat, x_flat, x_flat)
        attn_output = attn_output.permute(1, 2, 0).view(batch_size,
                                                        channels,
                                                        height, width)
        x = self.conv1x1(attn_output)
        return x


class Encoder(nn.Module):
    """
    The encoder part of a U-Net, which reduces the spatial dimensions
    of the input through convolutional blocks and max-pooling layers.

    Returns:
    --------
    tuple :
        A tuple containing:
        - The downsampled output tensor after the final pooling layer.
        - A tuple of feature maps from each convolutional block,
        to be used in the decoder for skip connections.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.enc_conv1 = ConvBlock(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc_conv2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc_conv3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor of shape (batch_size, 1, height, width).

        Returns:
        --------
        tuple :
            A tuple containing:
            - The output tensor after the final pooling layer, of shape
            (batch_size, 64, height/8, width/8).
            - A tuple of tensors containing the output of each convolutional
            block before pooling, to be used in the decoder for skip
            connections.
        """
        e1 = self.enc_conv1(x)
        p1 = self.pool1(e1)
        e2 = self.enc_conv2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc_conv3(p2)
        p3 = self.pool3(e3)
        return p3, (e1, e2, e3)


class Decoder(nn.Module):
    """
    The decoder part of a U-Net, which upsamples the input using transposed
    convolutions and refines the features using convolutional blocks.

    Returns:
    --------
    torch.Tensor
        The final output tensor after the decoder, with 1 channel and the
        same spatial dimensions as the original input.
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.up_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv3 = ConvBlock(128, 64)
        self.up_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv2 = ConvBlock(64, 32)
        self.up_conv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec_conv1 = ConvBlock(32, 16)
        self.final = nn.Conv2d(16, 1, kernel_size=1)
        self.gelu = nn.GELU()

    def forward(self, x, enc_features):
        """
        Forward pass through the decoder.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor from the bottleneck layer of shape
            (batch_size, 128, height/8, width/8).
        enc_features : tuple
            A tuple of feature maps from the encoder to be concatenated
            with the upsampled features at each step.

        Returns:
        --------
        torch.Tensor
            The output tensor of shape (batch_size, 1, height, width)
            after applying the final GELU activation.
        """
        e1, e2, e3 = enc_features
        u3 = self.up_conv3(x)
        c3 = torch.cat((u3, e3), dim=1)
        d3 = self.dec_conv3(c3)
        u2 = self.up_conv2(d3)
        c2 = torch.cat((u2, e2), dim=1)
        d2 = self.dec_conv2(c2)
        u1 = self.up_conv1(d2)
        c1 = torch.cat((u1, e1), dim=1)
        d1 = self.dec_conv1(c1)
        out = self.final(d1)
        return self.gelu(out)


class UNet(nn.Module):
    """
    A U-Net model combining an encoder, a bottleneck with an attention
    mechanism, and a decoder to produce a final output with the same
    spatial dimensions as the input.

    Returns:
    --------
    torch.Tensor
        The final output tensor after passing through the U-Net, with
        1 channel and the same spatial dimensions as the input.
    """
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = Encoder()
        self.bottleneck = Bottleneck()
        self.decoder = Decoder()

    def forward(self, x):
        """
        Forward pass through the U-Net model.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor of shape (batch_size, 1, height, width).

        Returns:
        --------
        torch.Tensor
            The output tensor of shape (batch_size, 1, height, width)
            after passing through the entire U-Net.
        """
        encoder_output, encoder_features = self.encoder(x)
        bottleneck_output = self.bottleneck(encoder_output)
        output = self.decoder(bottleneck_output, encoder_features)
        return output
