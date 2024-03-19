from torch import nn


def normalized_conv1d(*args, **kwargs):
    return nn.utils.parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))


def normalized_conv_trans1d(*args, **kwargs):
    return nn.utils.parametrizations.weight_norm(nn.ConvTranspose1d(*args, **kwargs))
