import torch
import torch.nn as nn
from collections import OrderedDict
import sys
sys.path.append('../../')

from util.spconv_utils import spconv, replace_feature


class ResidualBlock(spconv.modules.SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output = replace_feature(output, self.i_branch(identity).features + output.features)

        return output


class VGGBlock(spconv.modules.SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)


def build_blocks(inPlane, outPlane, block_fns, norm_fn, indice_key_id):
    blocks = {}
    for i, block_fn in enumerate(block_fns):
        blocks[f'block{i}'] = block_fn(inPlane, outPlane, norm_fn, indice_key=f'subm{indice_key_id}')
        inPlane = outPlane
    blocks = OrderedDict(blocks)
    blocks = nn.Sequential(blocks)
    return blocks


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1, downsample_padding=0):
        super().__init__()

        self.nPlanes = nPlanes

        block_fns = [block] * block_reps
        self.blocks = build_blocks(nPlanes[0], nPlanes[0], block_fns, norm_fn, indice_key_id)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, padding=downsample_padding,
                                    bias=False, indice_key=f'spconv{indice_key_id}')
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id+1,
                            downsample_padding=downsample_padding)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2,
                                           bias=False, indice_key=f'spconv{indice_key_id}')
            )

            block_fns_tail = [block] * block_reps
            self.blocks_tail = build_blocks(nPlanes[0] * 2, nPlanes[0], block_fns_tail, norm_fn, indice_key_id)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decode = self.conv(output)
            output_decode = self.u(output_decode)
            output_decode = self.deconv(output_decode)
            decode_features = output_decode.features

            output = replace_feature(output, torch.cat((identity.features, decode_features), dim=1))

            output = self.blocks_tail(output)

        return output


class UNet(nn.Module):
    def __init__(self, input_c, m, nPlanes, block_reps, block, norm_fn, model_cfg={}):
        super().__init__()

        assert m == nPlanes[0], f'm: {m} nPlanes[0]: {nPlanes[0]}'

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )
        self.unet = UBlock(nPlanes, norm_fn, block_reps, block, indice_key_id=1,
                           downsample_padding=model_cfg.get('downsample_padding', 0))
        self.output_layer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )

    def forward(self, input):
        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)

        return output
