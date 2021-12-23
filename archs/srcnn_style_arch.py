import math
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY

cfg = {'A': [64, 128, 128, 128, 128, 128]}


def make_layers(cfg, scale, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(negative_slope=0.1, inplace=True)]
            else:
                layers += [conv2d, nn.LeakyReLU(negative_slope=0.1, inplace=True)]
            in_channels = v
    # upsample
    if (scale & (scale - 1)) == 0:  # scale = 2^n
        for _ in range(int(math.log(scale, 2))):
            layers += [nn.Conv2d(in_channels, 4 * in_channels, 3, 1, 1)]
            layers += [nn.PixelShuffle(2), nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    elif scale == 3:
        layers += [nn.Conv2d(in_channels, 9 * in_channels, 3, 1, 1)]
        layers += [nn.PixelShuffle(3), nn.LeakyReLU(negative_slope=0.1, inplace=True)]

    out_channels = 3
    layers += [nn.Conv2d(in_channels, in_channels, 3, 1, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    layers += [nn.Conv2d(in_channels, out_channels, 3, 1, 1)]
    return nn.Sequential(*layers)


class SRCNNStyle(nn.Module):

    def __init__(self, features, init_weights=True):
        super(SRCNNStyle, self).__init__()
        self.features = features
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


@ARCH_REGISTRY.register()
def srcnn_style_net(scale, **kwargs):
    """srcnn_style 9-layer model (configuration "A")

    Args:
        scale (int): Upsampling factor. Support x2, x3 and x4.
            Default: 4.
    """
    model = SRCNNStyle(make_layers(cfg['A'], scale, **kwargs))
    return model
