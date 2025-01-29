"""
Christopher Ondrusz
GitHub: acse_cro23
"""
import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    """
    A convolutional block consisting of two convolutional layers, each
    followed by a ReLU activation. The block preserves the spatial dimensions
    of the input.

    Parameters:
    -----------
    in_channels : int
        The number of input channels for the convolutional layers.
    out_channels : int
        The number of output channels for the convolutional layers.

    Returns:
    --------
    None
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.enc_conv1 = ConvBlock(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc_conv2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc_conv3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc_conv1(x)
        p1 = self.pool1(e1)
        e2 = self.enc_conv2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc_conv3(p2)
        p3 = self.pool3(e3)
        return p3, (e1, e2, e3)


class Bottleneck(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8):
        super(Bottleneck, self).__init__()
        self.conv_block = ConvBlock(64, embed_dim)  # Embed_dim is 128
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                               num_heads=num_heads)
        self.conv1x1 = nn.Conv2d(embed_dim, 128, kernel_size=1)

    def forward(self, x):
        x = self.conv_block(x)
        # Reshape for MultiheadAttention:
        # [Batch, Channels, Height, Width] -> [Batch, Height*Width, Channels]
        batch_size, channels, height, width = x.size()
        x_flat = x.view(batch_size, channels, -1).permute(2, 0, 1)
        # Apply MultiheadAttention
        attn_output, _ = self.attention(x_flat, x_flat, x_flat)
        # Reshape back to original dimensions
        attn_output = attn_output.permute(1, 2, 0).view(batch_size,
                                                        channels,
                                                        height, width)
        # Apply a 1x1 convolution to integrate the attention output
        x = self.conv1x1(attn_output)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv3 = ConvBlock(128, 64)
        self.up_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv2 = ConvBlock(64, 32)
        self.up_conv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec_conv1 = ConvBlock(32, 16)
        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x, enc_features):
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
        return out


class PredictorUNet(nn.Module):
    def __init__(self):
        super(PredictorUNet, self).__init__()
        self.encoder = Encoder()
        self.bottleneck = Bottleneck()
        self.decoder = Decoder()

    def forward(self, x):
        encoder_output, encoder_features = self.encoder(x)
        bottleneck_output = self.bottleneck(encoder_output)
        output = self.decoder(bottleneck_output, encoder_features)
        return output
