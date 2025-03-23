#!/usr/bin/env python3
'''
Methods for AP-GeM
'''
from __future__ import annotations

import os
import sys
import random
import warnings
import math
from math import ceil
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageOps
from PIL.Image import BICUBIC as PIL_BICUBIC, PERSPECTIVE as PIL_PERSPECTIVE, \
    BILINEAR as PIL_BILINEAR, FLIP_LEFT_RIGHT as PIL_FLIP_LEFT_RIGHT #pylint: disable=E0611

import torch
import torch.nn as nn
from torch import ones as t_ones, cat as t_cat, matmul as t_matmul, stack as t_stack, \
    mean as t_mean, sign as t_sign #pylint: disable=E0611
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F

try:
    from torch.nn.functional import grab_img, update_img_and_labels, aff_mul, aff_rotate, \
        aff_translate, is_pil_image, DummyImg, rand_log_uniform
except ImportError:
    warnings.warn("Failed to import helper functions from torch.nn.functional. Some methods and "
                  "classes will not work.")

import torchvision.transforms as tvf
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, \
    adjust_saturation, adjust_hue

################################### dirtorch.nets #######################################

class DummyWatcher:
    '''
    Dummy class for sys.modules['utils.watcher']
    '''
    class AverageMeter:
        '''
        Dummy subclass
        '''
    #
    class Watch:
        '''
        Dummy subclass
        '''

def create_model(arch, *args, pretrained='', delete_fc=False, **kwargs):
    ''' Create an empty network for RMAC.

    arch : str type; name of the function to call
    args : list type; mandatory arguments
    kwargs : dict type; optional arguments
    '''
    # creating model
    # if arch not in model_names:
    #     raise NameError("unknown model architecture '%s'\nSelect one in %s" % (
    #                      arch, ','.join(model_names)))
    print(f'Arch type: {arch}')
    model = globals()[arch](*args, **kwargs)

    model.preprocess = dict(
        mean=model.rgb_means,
        std=model.rgb_stds,
        input_size=max(model.input_size)
    )

    if os.path.isfile(pretrained or ''):
        sys.modules['utils.watcher'] = DummyWatcher
        weights = torch.load(pretrained, map_location=lambda storage, loc: storage)['state_dict']
        load_pretrained_weights(model, weights, delete_fc=delete_fc)

    elif pretrained:
        assert hasattr(model, 'load_pretrained_weights'), \
            f'Model {arch} must be initialized with a valid model file (not {pretrained})'
        model.load_pretrained_weights(pretrained)

    return model

def load_pretrained_weights(net, state_dict, delete_fc=False):
    """ Load the pretrained weights (chop the last FC layer if needed)
        If layers are missing or of  wrong shape, will not load them.
    """

    new_dict = OrderedDict()
    for k, v in list(state_dict.items()):
        if k.startswith('module.'):
            k = k.replace('module.', '')
        new_dict[k] = v

    # Add missing weights from the network itself
    d = net.state_dict()
    for k, v in list(d.items()):
        if k not in new_dict:
            if not k.endswith('num_batches_tracked'):
                print(f"Loading weights for {type(net).__name__}: Missing layer {k}")
            new_dict[k] = v
        elif v.shape != new_dict[k].shape:
            print(f"Loading weights for {type(net).__name__}: Bad shape for layer {k}, skipping")
            new_dict[k] = v

    net.load_state_dict(new_dict)

    # Remove the FC layer if size doesn't match
    if delete_fc:
        fc = net.fc_name
        del new_dict[fc+'.weight']
        del new_dict[fc+'.bias']

# ################################# dirtorch.nets.backbones.resnet ######################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    '''
    Basic nn.Module
    '''
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    #
    def forward(self, x):
        '''
        Pass through network
        '''
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    '''
    Standard bottleneck block
    input  = inplanes * H * W
    middle =   planes * H/stride * W/stride
    output = 4*planes * H/stride * W/stride
    '''
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    #
    def forward(self, x):
        '''
        Pass through network
        '''
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def reset_weights(net):
    '''
    Reset weights for all modules in net
    '''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class ResNet(nn.Module):
    """
    A standard ResNet.
    """
    def __init__(self, block, layers, fc_out, model_name, self_similarity_radius=None,
                 self_similarity_version=2):
        nn.Module.__init__(self)
        self.model_name = model_name
        # default values for a network pre-trained on imagenet
        self.rgb_means = [0.485, 0.456, 0.406]
        self.rgb_stds  = [0.229, 0.224, 0.225]
        self.input_size = (3, 224, 224)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
            self_similarity_radius=self_similarity_radius,
            self_similarity_version=self_similarity_version)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            self_similarity_radius=self_similarity_radius,
            self_similarity_version=self_similarity_version)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
            self_similarity_radius=self_similarity_radius,
            self_similarity_version=self_similarity_version)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
            self_similarity_radius=self_similarity_radius,
            self_similarity_version=self_similarity_version)
        #
        reset_weights(self)
        #
        self.fc = None
        self.fc_out = fc_out
        if self.fc_out > 0:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512 * block.expansion, fc_out)
            self.fc_name = 'fc'
    #
    def _make_layer(self, block, planes, blocks, stride=1, self_similarity_radius=None,
                    self_similarity_version=1): #pylint: disable=W0613
        '''
        Helper to make layers
        '''
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes=planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        if self_similarity_radius:
            raise NotImplementedError('Imports missing here, please try avoid using this! '
                                        'Code has been commented out.')
        # if self_similarity_radius:
        #     if self_similarity_version == 1:
        #         from . self_sim import SelfSimilarity1
        #         layers.append(SelfSimilarity1(self_similarity_radius, self.inplanes))
        #     else:
        #         from . self_sim import SelfSimilarity2
        #         layers.append(SelfSimilarity2(self_similarity_radius, self.inplanes))
        return nn.Sequential(*layers)

    def forward(self, x, out_layer=0):
        '''
        Pass through network
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if out_layer==-1:
            return x, self.layer4(x)
        x = self.layer4(x)
        if self.fc_out > 0:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x

    # def load_pretrained_weights(self, pretrain_code):
    #     '''
    #     Load pretrained weights from pytorch.org
    #     '''
    #     if pretrain_code == 'imagenet':
    #         model_urls = {
    #             'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    #             'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    #             'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    #             'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    #         }
    #     else:
    #         raise NameError(f"Unknown pretraining code '{pretrain_code}'")

    #     print(f"Loading ImageNet pretrained weights for {pretrain_code}")
    #     assert self.model_name in model_urls, f"Unknown model '{self.model_name}'"

    #     model_dir='dirtorch/data/models/classification/'
    #     try:
    #         os.makedirs(model_dir)
    #     except OSError:
    #         pass

    #     state_dict = model_zoo.load_url(model_urls[self.model_name], model_dir=model_dir)

    #     from . import load_pretrained_weights as lpw
    #     lpw(self, state_dict)

# def resnet18(out_dim=2048):
#     """Constructs a ResNet-18 model.
#     """
#     net = ResNet(BasicBlock, [2, 2, 2, 2], out_dim, 'resnet18')
#     return net

# def resnet50(out_dim=2048):
#     """Constructs a ResNet-50 model.
#     """
#     net = ResNet(Bottleneck, [3, 4, 6, 3], out_dim, 'resnet50')
#     return net

def resnet101(out_dim=2048):
    """Constructs a ResNet-101 model.
    """
    net = ResNet(Bottleneck, [3, 4, 23, 3], out_dim, 'resnet101')
    return net

# def resnet152(out_dim=2048):
#     """Constructs a ResNet-152 model.
#     """
#     net = ResNet(Bottleneck, [3, 8, 36, 3], out_dim, 'resnet152')
#     return net

# ########################### dirtorch.nets.layers.pooling ################################

class GeneralizedMeanPooling(Module):
    r"""
    Applies a 2D power-average adaptive pooling over an input signal composed of several input
    planes.

    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`

        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    """
    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps
    #
    def forward(self, x):
        '''
        Pass through network
        '''
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)
    #
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.p) + ', ' \
            + 'output_size=' + str(self.output_size) + ')'

class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """
    Same, but norm is trainable
    """
    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = Parameter(t_ones(1) * norm)


# ############################ dirtorch.nets.rmac_resnet #####################################

def l2_normalize(x, axis=-1):
    '''
    Perform l-2 normalization
    '''
    x = F.normalize(x, p=2, dim=axis)
    return x

class ResNet_RMAC(ResNet): #pylint: disable=C0103
    """
    ResNet for RMAC (without ROI pooling)
    """
    def __init__(self, block, layers, model_name, out_dim=2048, norm_features=False,
                       pooling='gem', gemp=3, center_bias=0,
                       dropout_p=None, without_fc=False, **kwargs):
        ResNet.__init__(self, block, layers, 0, model_name, **kwargs)
        self.norm_features = norm_features
        self.without_fc = without_fc
        self.pooling = pooling
        self.center_bias = center_bias

        if pooling == 'max':
            self.adpool = nn.AdaptiveMaxPool2d(output_size=1)
        elif pooling == 'avg':
            self.adpool = nn.AdaptiveAvgPool2d(output_size=1)
        elif pooling.startswith('gem'):
            self.adpool = GeneralizedMeanPoolingP(norm=gemp)
        else:
            raise ValueError(pooling)

        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.fc = nn.Linear(512 * block.expansion, out_dim)
        self.fc_name = 'fc'
        self.feat_dim = out_dim
        self.detach = False

    def forward(self, x): #pylint: disable=W0221
        '''
        Pass through network
        '''

        x = ResNet.forward(self, x)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.detach:
            # stop the back-propagation here, if needed
            x = Variable(x.detach())
            x = self.id(x)  # fake transformation

        if self.center_bias > 0:
            b = self.center_bias
            bias = 1 + torch.FloatTensor([[[[0,0,0,0],[0,b,b,0],[0,b,b,0],[0,0,0,0]]]])\
                            .to(x.device)
            bias = torch.nn.functional.interpolate(bias, size=x.shape[-2:], mode='bilinear',
                                                   align_corners=True)
            x = x*bias

        # global pooling
        x = self.adpool(x)

        if self.norm_features:
            x = l2_normalize(x, axis=1)

        x.squeeze_()
        if not self.without_fc:
            x = self.fc(x)

        x = l2_normalize(x, axis=-1)
        return x

# ###################### dirtorch.nets.rmac_resnet_fpn ##########################

class ResNet_RMAC_FPN(ResNet): #pylint: disable=C0103
    """
    ResNet for RMAC (without ROI pooling)
    """
    def __init__(self, block, layers, model_name, out_dim=None, norm_features=False,
                       pooling='gem', gemp=3, center_bias=0, mode=1,
                       dropout_p=None, without_fc=False, **kwargs):
        ResNet.__init__(self, block, layers, 0, model_name, **kwargs)
        self.norm_features = norm_features
        self.without_fc = without_fc
        self.pooling = pooling
        self.center_bias = center_bias
        self.mode = mode

        dim1 = 256 * block.expansion
        dim2 = 512 * block.expansion
        if out_dim is None:
            out_dim = dim1 + dim2
        #FPN
        if self.mode == 1:
            self.conv1x5 = nn.Conv2d(dim2, dim1, kernel_size=1, stride=1, bias=False)
            self.conv3c4 = nn.Conv2d(dim1, dim1, kernel_size=3, stride=1, padding=1, bias=False)
            self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        if pooling == 'max':
            self.adpool = nn.AdaptiveMaxPool2d(output_size=1)
        elif pooling == 'avg':
            self.adpool = nn.AdaptiveAvgPool2d(output_size=1)
        elif pooling == 'gem':
            self.adpoolx5 = GeneralizedMeanPoolingP(norm=gemp)
            self.adpoolc4 = GeneralizedMeanPoolingP(norm=gemp)

        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.fc = nn.Linear(768 * block.expansion, out_dim)
        self.fc_name = 'fc'
        self.feat_dim = out_dim
        self.detach = False

    def forward(self, x): #pylint: disable=W0221
        '''
        Pass through network
        '''
        x4, x5 = ResNet.forward(self, x, -1)

        # FPN
        if self.mode == 1:
            c5 = F.interpolate(x5, size=x4.shape[-2:], mode='nearest')

            c5 = self.conv1x5(c5)
            c5 = self.relu(c5)
            x4 = x4 + c5
            x4 = self.conv3c4(x4)
            x4 = self.relu(x4)

        if self.dropout is not None:
            x5 = self.dropout(x5)
            x4 = self.dropout(x4)

        if self.detach:
            # stop the back-propagation here, if needed
            x5 = Variable(x5.detach())
            x5 = self.id(x5)  # fake transformation
            x4 = Variable(x4.detach())
            x4 = self.id(x4)  # fake transformation

        # global pooling
        x5 = self.adpoolx5(x5)
        x4 = self.adpoolc4(x4)

        x = t_cat((x4, x5), 1)

        if self.norm_features:
            x = l2_normalize(x, axis=1)

        x.squeeze_()
        if not self.without_fc:
            x = self.fc(x)

        x = l2_normalize(x, axis=-1)
        return x




# def resnet18_fpn_rmac(backbone=ResNet_RMAC_FPN, **kwargs):
#     kwargs.pop('scales', None)
#     return backbone(BasicBlock, [2, 2, 2, 2], 'resnet18', **kwargs)

# def resnet50_fpn_rmac(backbone=ResNet_RMAC_FPN, **kwargs):
#     kwargs.pop('scales', None)
#     return backbone(Bottleneck, [3, 4, 6, 3], 'resnet50', **kwargs)

def resnet101_fpn_rmac(backbone=ResNet_RMAC_FPN, **kwargs):
    '''
    TODO
    '''
    kwargs.pop('scales', None)
    return backbone(Bottleneck, [3, 4, 23, 3], 'resnet101', **kwargs)

# def resnet101_fpn0_rmac(backbone=ResNet_RMAC_FPN, **kwargs):
#     kwargs.pop('scales', None)
#     return backbone(Bottleneck, [3, 4, 23, 3], 'resnet101', mode=0, **kwargs)

# def resnet152_fpn_rmac(backbone=ResNet_RMAC_FPN, **kwargs):
#     kwargs.pop('scales', None)
#     return backbone(Bottleneck, [3, 8, 36, 3], 'resnet152', **kwargs)

# ######################### dirtorch.nets.rmac_resnext ####################################

class ResNext_RMAC(nn.Module): #pylint: disable=C0103
    """
    ResNet for RMAC (without ROI pooling)
    """
    def __init__(self, backbone, out_dim=2048, norm_features=False,
                       pooling='gem', gemp=3, center_bias=0,
                       dropout_p=None, without_fc=False, **kwargs):
        super(ResNext_RMAC, self).__init__()
        self.backbone = backbone
        self.norm_features = norm_features
        self.without_fc = without_fc
        self.pooling = pooling
        self.center_bias = center_bias

        if pooling == 'max':
            self.adpool = nn.AdaptiveMaxPool2d(output_size=1)
        elif pooling == 'avg':
            self.adpool = nn.AdaptiveAvgPool2d(output_size=1)
        elif pooling == 'gem':
            self.adpool = GeneralizedMeanPoolingP(norm=gemp)

        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(2048, out_dim)
        self.fc_name = 'last_linear'
        self.feat_dim = out_dim
        self.detach = False

    def forward(self, x):
        '''
        Pass through network
        '''
        x = ResNet.forward(self, x)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.detach:
            # stop the back-propagation here, if needed
            x = Variable(x.detach())
            x = self.id(x)  # fake transformation

        if self.center_bias > 0:
            b = self.center_bias
            bias = 1 + torch.FloatTensor([[[[0,0,0,0],[0,b,b,0],[0,b,b,0],[0,0,0,0]]]])\
                        .to(x.device)
            bias = torch.nn.functional.interpolate(bias, size=x.shape[-2:], mode='bilinear',
                                                   align_corners=True)
            x = x*bias

        # global pooling
        x = self.adpool(x)

        if self.norm_features:
            x = l2_normalize(x, axis=1)

        x.squeeze_()
        if not self.without_fc:
            x = self.fc(x)

        x = l2_normalize(x, axis=-1)
        return x




# def resnet18_rmac(backbone=ResNet_RMAC, **kwargs):
#     kwargs.pop('scales', None)
#     return backbone(BasicBlock, [2, 2, 2, 2], 'resnet18', **kwargs)

# def resnet50_rmac(backbone=ResNet_RMAC, **kwargs):
#     kwargs.pop('scales', None)
#     return backbone(Bottleneck, [3, 4, 6, 3], 'resnet50', **kwargs)

def resnet101_rmac(backbone=ResNet_RMAC, **kwargs):
    '''
    TODO
    '''
    kwargs.pop('scales', None)
    return backbone(Bottleneck, [3, 4, 23, 3], 'resnet101', **kwargs)

# def resnet152_rmac(backbone=ResNet_RMAC, **kwargs):
#     kwargs.pop('scales', None)
#     return backbone(Bottleneck, [3, 8, 36, 3], 'resnet152', **kwargs)

###################################### dirtorch.utils.common ###################################

def typename(x):
    '''
    TODO
    '''
    return type(x).__module__

def tonumpy(x):
    '''
    TODO
    '''
    if typename(x) == torch.__name__:
        return x.cpu().numpy()
    else:
        return x

def matmul(a, b):
    '''
    TODO
    '''
    if typename(a) == np.__name__:
        b = tonumpy(b)
        scores = np.dot(a, b.T)
    elif typename(b) == torch.__name__:
        scores = t_matmul(a, b.t()).cpu().numpy()
    else:
        raise TypeError("matrices must be either numpy or torch type")
    return scores

def pool(x, pooling='mean', gemp=3):
    '''
    TODO
    '''
    if len(x) == 1:
        return x[0]
    x = t_stack(x, dim=0)
    if pooling == 'mean':
        return t_mean(x, dim=0)
    elif pooling == 'gem':
        def sympow(x, p, eps=1e-6):
            s = t_sign(x)
            return (x*s).clamp(min=eps).pow(p) * s
        x = sympow(x, gemp)
        x = t_mean(x, dim=0)
        return sympow(x, 1/gemp)
    else:
        raise ValueError("Bad pooling mode: "+str(pooling))

def load_checkpoint(filename, iscuda=False):
    '''
    TODO
    '''
    if not filename:
        return None
    assert os.path.isfile(filename), f"=> no checkpoint found at '{filename}'"
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage, weights_only=False)
    print(f"=> loading checkpoint '{filename}'", end='')
    for key in ['epoch', 'iter', 'current_iter']:
        if key in checkpoint:
            print(f" ({key} {checkpoint[key]:d})", end='')
    print()

    new_dict = OrderedDict()
    for k, v in list(checkpoint['state_dict'].items()):
        if k.startswith('module.'):
            k = k[7:]
        new_dict[k] = v
    checkpoint['state_dict'] = new_dict

    if iscuda and 'optimizer' in checkpoint:
        try:
            for state in checkpoint['optimizer']['state'].values():
                for k, v in state.items():
                    if iscuda and torch.is_tensor(v):
                        state[k] = v.cuda()
        except RuntimeError as e:
            print("RuntimeError:", e, f"(machine {os.environ['HOSTNAME']},",
                  f"GPU {os.environ['CUDA_VISIBLE_DEVICES']})", file=sys.stderr)
            sys.exit(1)

    return checkpoint

def switch_model_to_cuda(model, iscuda=True, checkpoint=None):
    '''
    TODO
    '''
    if iscuda:
        if checkpoint:
            checkpoint['state_dict'] = {'module.' + k: v \
                                        for k, v in checkpoint['state_dict'].items()}
        try:
            model = torch.nn.DataParallel(model)

            # copy attributes automatically
            for var in dir(model.module):
                if var.startswith('_'):
                    continue
                val = getattr(model.module, var)
                if isinstance(val, (bool, int, float, str, dict)) or \
                   (callable(val) and var.startswith('get_')):
                    setattr(model, var, val)

            model.cuda()
            model.isasync = True
        except RuntimeError as e:
            print("RuntimeError:", e, f"(machine {os.environ['HOSTNAME']},",
                  f"GPU {os.environ['CUDA_VISIBLE_DEVICES']})", file=sys.stderr)
            sys.exit(1)

    model.iscuda = iscuda
    return model

def transform(pca, x, whitenp=0.5, whitenv=None, whitenm=1.0, use_sklearn=True):
    '''
    TODO
    '''
    if use_sklearn:
        # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/base.py#L99
        if pca.mean_ is not None:
            x = x - pca.mean_
        x_transformed = np.dot(x, pca.components_[:whitenv].T)
        if pca.whiten:
            x_transformed /= whitenm * np.power(pca.explained_variance_[:whitenv], whitenp)
    else:
        x = x - pca['means']
        x_transformed = np.dot(x, pca['W'])
    return x_transformed

def whiten_features(x, pca, l2norm=True, whitenp=0.5, whitenv=None, whitenm=1.0, use_sklearn=True):
    '''
    TODO
    '''
    res = transform(pca, x, whitenp=whitenp, whitenv=whitenv, whitenm=whitenm,
                    use_sklearn=use_sklearn)
    if l2norm:
        res = res / np.expand_dims(np.linalg.norm(res, axis=1), axis=1)
    return res

###################################### dirtorch.utils.convenient ###################################

def mkdir(d):
    '''
    TODO
    '''
    try:
        os.makedirs(d)
    except OSError:
        pass

####################################### dirtorch.utils.funcs #######################################

def sigmoid(x, a=1, b=0):
    '''
    TODO
    '''
    return 1 / (1 + np.exp(a * (b - x)))


def sigmoid_range(x, at5, at95):
    """
    create sigmoid function like that: 
            sigmoid(at5)  = 0.05
            sigmoid(at95) = 0.95
    and returns sigmoid(x) 
    """
    a = 6 / (at95 - at5)
    b = at5 + 3 / a
    return sigmoid(x, a, b)

###################################### dirtorch.utils.transforms ###################################

def create(cmd_line, to_tensor=False, **kwargs):
    '''
    Create a sequence of transformations.

    cmd_line: (str)
        Comma-separated list of transformations.
        Ex: "Rotate(10), Scale(256)"

    to_tensor: (bool)
        Whether to add the "ToTensor(), Normalize(mean, std)"
        automatically to the end of the transformation string

    kwargs: (dict)
        dictionary of global variables.
    '''
    if to_tensor:
        if not cmd_line:
            cmd_line = "ToTensor(), Normalize(mean=mean, std=std)"
        elif to_tensor and 'ToTensor' not in cmd_line:
            cmd_line += ", ToTensor(), Normalize(mean=mean, std=std)"

    assert isinstance(cmd_line, str)

    cmd_line = f"tvf.Compose([{cmd_line}])"
    try:
        return eval(cmd_line, globals(), kwargs) #pylint: disable=W0123
    except Exception as e:
        raise SyntaxError("Cannot interpret this transform list: "
                          f"{cmd_line}\nReason: {e}") from e

class Identity (object):
    """ Identity transform. It does nothing!
    """
    def __call__(self, inp):
        return inp


class Pad(object):
    """ Pads the shortest side of the image to a given size

    If size is shorter than the shortest image, then the image will be untouched
    """

    def __init__(self, size, color=(127,127,127)):
        self.size = size
        assert len(color) == 3
        if not all(isinstance(c,int) for c in color):
            color = tuple([int(255*c) for c in color])
        self.color = color

    def __call__(self, inp):
        img = grab_img(inp)
        w, h = img.size
        if w >= h:
            newh = max(h,self.size)
            neww = w
        else:
            newh = h
            neww = max(w,self.size)

        if (neww,newh) != img.size:
            img2 = Image.new('RGB', (neww,newh), self.color)
            img2.paste(img, ((neww-w)//2,(newh-h)//2) )
            img = img2

        return update_img_and_labels(inp, img, aff=(1,0,0,0,1,0))

class PadSquare (object):
    """ Pads the image to a square size

    The dimension of the output image will be equal to size x size

    If size is None, then the image will be padded to the largest dimension

    If size is smaller than the original image size, the image will be cropped
    """

    def __init__(self, size=None, color=(127,127,127)):
        self.size = size
        assert len(color) == 3
        if not all(isinstance(c,int) for c in color):
            color = tuple([int(255*c) for c in color])
        self.color = color

    def __call__(self, inp):
        img = grab_img(inp)
        w, h = img.size
        s = self.size or max(w, h)


        if (s,s) != img.size:
            img2 = Image.new('RGB', (s,s), self.color)
            img2.paste(img, ((s-w)//2,(s-h)//2) )
            img = img2

        return update_img_and_labels(inp, img, aff=(1,0,0,0,1,0))

class RandomBorder (object):
    """
    Expands the image with a random size border
    """

    def __init__(self, min_size, max_size, color=(127,127,127)):
        assert isinstance(min_size, int) and min_size >= 0
        assert isinstance(max_size, int) and min_size <= max_size
        self.min_size = min_size
        self.max_size = max_size
        assert len(color) == 3
        if not all(isinstance(c,int) for c in color):
            color = tuple([int(255*c) for c in color])
        self.color = color

    def __call__(self, inp):
        img = grab_img(inp)

        bh = random.randint(self.min_size, self.max_size)
        bw = random.randint(self.min_size, self.max_size)

        img = ImageOps.expand(img, border=(bw,bh,bw,bh), fill=self.color)

        return update_img_and_labels(inp, img, aff=(1,0,0,0,1,0))


class Scale (object):
    """
    Rescale the input PIL.Image to a given size.
    Same as torchvision.Scale

    The smallest dimension of the resulting image will be = size.

    if largest == True: same behaviour for the largest dimension.

    if not can_upscale: don't upscale
    if not can_downscale: don't downscale
    """
    def __init__(self, size, interpolation=PIL_BILINEAR, largest=False, can_upscale=True,
                 can_downscale=True):
        assert isinstance(size, (float,int)) or (len(size) == 2)
        self.size = size
        if isinstance(self.size, float):
            assert 0 < self.size <= 4, 'bad float self.size, cannot be outside of range ]0,4]'
        self.interpolation = interpolation
        self.largest = largest
        self.can_upscale = can_upscale
        self.can_downscale = can_downscale

    def get_params(self, imsize):
        '''
        TODO
        '''
        w,h = imsize
        def is_smaller(a,b,largest):
            return (a>=b) if largest else (a<=b)
        if isinstance(self.size, int):
            if (is_smaller(w, h, self.largest) and w == self.size) \
                or (is_smaller(h, w, self.largest) and h == self.size):
                ow, oh = w, h
            elif is_smaller(w, h, self.largest):
                ow = self.size
                oh = int(0.5 + self.size * h / w)
            else:
                oh = self.size
                ow = int(0.5 + self.size * w / h)

        elif isinstance(self.size, float):
            ow, oh = int(0.5 + self.size*w), int(0.5 + self.size*h)

        else: # tuple of ints
            ow, oh = self.size
        return ow, oh

    def __call__(self, inp):
        img = grab_img(inp)
        w, h = img.size

        size2 = ow,oh = self.get_params(img.size)

        if size2 != img.size:
            a1, a2 = img.size, size2
            if (self.can_upscale and min(a1) < min(a2)) \
                or (self.can_downscale and min(a1) > min(a2)):
                img = img.resize(size2, self.interpolation)

        return update_img_and_labels(inp, img, aff=(ow/w,0,0,0,oh/h,0))



class RandomScale (Scale):
    """Rescale the input PIL.Image to a random size.

    Args:
        min_size (int): min size of the smaller edge of the picture.
        max_size (int): max size of the smaller edge of the picture.

        ar (float or tuple):
            max change of aspect ratio (width/height).

        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, min_size, max_size, ar=1, can_upscale=False, can_downscale=True,
                 interpolation=PIL_BILINEAR, largest=False):
        Scale.__init__(self, 0, can_upscale=can_upscale, can_downscale=can_downscale,
                       interpolation=interpolation, largest=largest)
        assert isinstance(min_size, int) and min_size >= 1
        assert isinstance(max_size, int) and min_size <= max_size
        self.min_size = min_size
        self.max_size = max_size
        if type(ar) in (float,int):
            ar = (min(1/ar,ar),max(1/ar,ar))
        assert 0.2 < ar[0] <= ar[1] < 5
        self.ar = ar
        self.largest = largest

    def get_params(self, imsize):
        w,h = imsize
        if self.can_upscale:
            self.max_size = self.max_size
        else:
            self.max_size = min(self.max_size, w, h)
        size = max(min(int(0.5 + rand_log_uniform(self.min_size,self.max_size)),
                       self.max_size), self.min_size)
        ar = rand_log_uniform(*self.ar) # change of aspect ratio

        if not self.largest:
            if w < h : # image is taller
                ow = size
                oh = int(0.5 + size * h / w / ar)
                if oh < self.min_size:
                    ow,oh = int(0.5 + ow*float(self.min_size)/oh),self.min_size
            else: # image is wider
                oh = size
                ow = int(0.5 + size * w / h * ar)
                if ow < self.min_size:
                    ow,oh = self.min_size,int(0.5 + oh*float(self.min_size)/ow)
            assert ow >= self.min_size
            assert oh >= self.min_size
        else: # if self.largest
            if w > h: # image is wider
                ow = size
                oh = int(0.5 + size * h / w / ar)
            else: # image is taller
                oh = size
                ow = int(0.5 + size * w / h * ar)
            assert ow <= self.max_size
            assert oh <= self.max_size

        return ow, oh


class RandomCrop (object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        '''
        TODO
        '''
        w, h = img.size
        th, tw = output_size
        assert h >= th and w >= tw, f"Image of {w:d}x{h:d} is too small for crop {tw:d}x{th:d}"

        y = np.random.randint(0, h - th) if h > th else 0
        x = np.random.randint(0, w - tw) if w > tw else 0
        return x, y, tw, th

    def __call__(self, inp):
        img = grab_img(inp)

        padl = padt = 0
        if self.padding > 0:
            if is_pil_image(img):
                img = ImageOps.expand(img, border=self.padding, fill=0)
            else:
                assert isinstance(img, DummyImg)
                img = img.expand(border=self.padding)
            if isinstance(self.padding, int):
                padl = padt = self.padding
            else:
                padl, padt = self.padding[0:2]

        i, j, tw, th = self.get_params(img, self.size)
        img = img.crop((i, j, i+tw, j+th))

        return update_img_and_labels(inp, img, aff=(1,0,padl-i,0,1,padt-j))



class CenterCrop (RandomCrop):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        y = int(0.5 +((h - th) / 2.))
        x = int(0.5 +((w - tw) / 2.))
        return x, y, tw, th



class CropToBbox(object):
    """ Crop the image according to the bounding box.

    margin (float):
        ensure a margin around the bbox equal to (margin * min(bbWidth,bbHeight))

    min_size (int):
        result cannot be smaller than this size
    """
    def __init__(self, margin=0.5, min_size=0):
        self.margin = margin
        self.min_size = min_size

    def __call__(self, inp):
        img = grab_img(inp)
        w, h = img.size

        assert min(w,h) >= self.min_size

        x0,y0,x1,y1 = inp['bbox']
        assert x0 < x1 and y0 < y1
        bbw, bbh = x1-x0, y1-y0
        margin = int(0.5 + self.margin * min(bbw, bbh))

        i = max(0, x0 - margin)
        j = max(0, y0 - margin)
        w = min(w, x1 + margin) - i
        h = min(h, y1 + margin) - j

        if w < self.min_size:
            i = max(0, i-(self.min_size-w)//2)
            w = self.min_size
        if h < self.min_size:
            j = max(0, j-(self.min_size-h)//2)
            h = self.min_size

        img = img.crop((i,j,i+w,j+h))

        return update_img_and_labels(inp, img, aff=(1,0,-i,0,1,-j))

class RandomRotation(object):
    """Rescale the input PIL.Image to a random size.

    Args:
        degrees (float):
            rotation angle.

        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, degrees, interpolation=PIL_BILINEAR):
        self.degrees = degrees
        self.interpolation = interpolation

    def __call__(self, inp):
        img = grab_img(inp)
        w, h = img.size

        angle = np.random.uniform(-self.degrees, self.degrees)

        img = img.rotate(angle, resample=self.interpolation)
        w2, h2 = img.size

        aff = aff_translate(-w/2,-h/2)
        aff = aff_mul(aff, aff_rotate(-angle * np.pi/180))
        aff = aff_mul(aff, aff_translate(w2/2,h2/2))
        return update_img_and_labels(inp, img, aff=aff)


class RandomFlip (object):
    """Randomly flip the image.
    """
    def __call__(self, inp):
        img = grab_img(inp)
        w, _ = img.size

        flip = np.random.rand() < 0.5
        if flip:
            img = img.transpose(PIL_FLIP_LEFT_RIGHT)

        return update_img_and_labels(inp, img, aff=(-1,0,w-1,0,1,0))

class RandomTilting(object):
    """Apply a random tilting (left, right, up, down) to the input PIL.Image

    Args:
        maginitude (float):
            maximum magnitude of the random skew (value between 0 and 1)
        directions (string):
            tilting directions allowed (all, left, right, up, down)
            examples: "all", "left,right", "up-down-right"
    """

    def __init__(self, magnitude, directions='all'):
        self.magnitude = magnitude
        self.directions = directions.lower().replace(',',' ').replace('-',' ')

    def __call__(self, inp):
        img = grab_img(inp)
        w, h = img.size

        x1,y1,x2,y2 = 0,0,h,w
        original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]

        max_skew_amount = max(w, h)
        max_skew_amount = int(ceil(max_skew_amount * self.magnitude))
        skew_amount = random.randint(1, max_skew_amount)

        if self.directions == 'all':
            choices = [0,1,2,3]
        else:
            dirs = ['left', 'right', 'up', 'down']
            choices = []
            for d in self.directions.split():
                try:
                    choices.append(dirs.index(d))
                except Exception as e:
                    raise ValueError(f'Tilting direction {d} not recognized') from e

        skew_direction = random.choice(choices)

        if skew_direction == 0:
            # Left Tilt
            new_plane = [(y1, x1 - skew_amount),  # Top Left
                         (y2, x1),                # Top Right
                         (y2, x2),                # Bottom Right
                         (y1, x2 + skew_amount)]  # Bottom Left
        elif skew_direction == 1:
            # Right Tilt
            new_plane = [(y1, x1),                # Top Left
                         (y2, x1 - skew_amount),  # Top Right
                         (y2, x2 + skew_amount),  # Bottom Right
                         (y1, x2)]                # Bottom Left
        elif skew_direction == 2:
            # Forward Tilt
            new_plane = [(y1 - skew_amount, x1),  # Top Left
                         (y2 + skew_amount, x1),  # Top Right
                         (y2, x2),                # Bottom Right
                         (y1, x2)]                # Bottom Left
        elif skew_direction == 3:
            # Backward Tilt
            new_plane = [(y1, x1),                # Top Left
                         (y2, x1),                # Top Right
                         (y2 + skew_amount, x2),  # Bottom Right
                         (y1 - skew_amount, x2)]  # Bottom Left

        # To calculate the coefficients required by PIL for the perspective skew,
        # see the following Stack Overflow discussion: https://goo.gl/sSgJdj
        matrix = []

        for p1, p2 in zip(new_plane, original_plane):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        big_a = np.matrix(matrix, dtype=np.float)
        big_b = np.array(original_plane).reshape(8)

        homography = np.dot(np.linalg.pinv(big_a), big_b)
        homography = tuple(np.array(homography).reshape(8))

        img =  img.transform(img.size, PIL_PERSPECTIVE, homography, resample=PIL_BICUBIC)

        homography = np.linalg.pinv(np.array(homography+(1,)).reshape(3,3),
                                    dtype=np.float32).ravel()[:8]
        return update_img_and_labels(inp, img, persp=tuple(homography))



class StillTransform (object):
    """
    Takes and return an image, without changing its shape or geometry.
    """
    def _transform(self, image):
        raise NotImplementedError()

    def __call__(self, inp):
        image = grab_img(inp)

        # transform the image (size should not change)
        image = self._transform(image)

        return update_img_and_labels(inp, image, aff=(1,0,0,0,1,0))



class ColorJitter (StillTransform):
    """
    Randomly change the brightness, contrast and saturation of an image.
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
        Transform which randomly adjusts brightness, contrast and
        saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(tvf.Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(tvf.Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(tvf.Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(tvf.Lambda(lambda img: adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform_result = tvf.Compose(transforms)
        return transform_result

    def _transform(self, image):
        transform_func = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        return transform_func(image)


class RandomErasing (StillTransform):
    """
    Class that performs Random Erasing, an augmentation technique described
    in `https://arxiv.org/abs/1708.04896 <https://arxiv.org/abs/1708.04896>`_
    by Zhong et al. To quote the authors, random erasing:

    "*... randomly selects a rectangle region in an image, and erases its
    pixels with random values.*"

    The size of the random rectangle is controlled using the
    :attr:`area` parameter. This area is random in its
    width and height.

    Args:
        area: The percentage area of the image to occlude.
    """
    def __init__(self, area):
        self.area = area

    def _transform(self, image):
        """
        Adds a random noise rectangle to a random area of the passed image,
        returning the original image with this rectangle superimposed.

        :param image: The image to add a random noise rectangle to.
        :type image: PIL.Image
        :return: The image with the superimposed random rectangle as type
         image PIL.Image
        """
        w, h = image.size
        w_occlusion_max = int(w * self.area)
        h_occlusion_max = int(h * self.area)
        w_occlusion_min = int(w * self.area/2)
        h_occlusion_min = int(h * self.area/2)
        if not (w_occlusion_min < w_occlusion_max and h_occlusion_min < h_occlusion_max):
            return image
        w_occlusion = np.random.randint(w_occlusion_min, w_occlusion_max)
        h_occlusion = np.random.randint(h_occlusion_min, h_occlusion_max)
        if len(image.getbands()) == 1:
            rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion,
                                                    h_occlusion) * 255))
        else:
            rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion,
                                                    h_occlusion, len(image.getbands())) * 255))
        #
        assert w > w_occlusion and h > h_occlusion
        random_position_x = np.random.randint(0, w - w_occlusion)
        random_position_y = np.random.randint(0, h - h_occlusion)
        image = image.copy() # don't modify the original
        image.paste(rectangle, (random_position_x, random_position_y))
        return image


class ToTensor(StillTransform, tvf.ToTensor):
    '''
    TODO
    '''
    def _transform(self, image):
        return tvf.ToTensor.__call__(self, image)

class Normalize(StillTransform, tvf.Normalize):
    '''
    TODO
    '''
    def _transform(self, image):
        return tvf.Normalize.__call__(self, image)

class BBoxToPixelLabel (object):
    """
    Convert a bbox into per-pixel label
    """
    def __init__(self, nclass, downsize, mode):
        self.nclass = nclass
        self.downsize = downsize
        self.mode = mode
        self.nbin = 5
        self.log_scale = 1.5
        self.ref_scale = 8.0

    def __call__(self, inp):
        assert isinstance(inp, dict)

        w, h = inp['img'].size
        ds = self.downsize
        assert w % ds == 0
        assert h % ds == 0

        x0,y0,x1,y1 = inp['bbox']
        inp['bbox'] = np.int64(inp['bbox'])

        ll = x0/ds
        rr = (x1-1)/ds
        tt = y0/ds
        bb = (y1-1)/ds
        l = max(0, int(ll))
        r = min(w//ds, 1+int(rr))
        t = max(0, int(tt))
        b = min(h//ds, 1+int(bb))
        inp['bbox_downscaled'] = np.array((l,t,r,b), dtype=np.int64)
        #
        big_w, big_h = w//ds, h//ds
        res = np.zeros((big_h,big_w), dtype=np.int64)
        res[:] = self.nclass # last bin is null class
        res[t:b, l:r] = inp['label']
        inp['pix_label'] = res
        #
        if self.mode == 'hough':
            # compute hough parameters
            def hough_topos(left, pos, right):
                return np.floor( self.nbin * (pos - left) / (right - left) )
            def tolog(size):
                size = max(size,1e-8) # make it positive
                return np.round( np.log(size / self.ref_scale) \
                                / np.log(self.log_scale) + (self.nbin-1)/2 )
            # for each pixel, find its x and y position
            yc,xc = np.mgrid[0:big_h, 0:big_w]
            res = -np.ones((4, big_h, big_w), dtype=np.int64)
            res[0] = hough_topos(ll, xc, rr)
            res[1] = hough_topos(tt, yc, bb)
            res[2] = tolog(rr - ll)
            res[3] = tolog(bb - tt)
            res = np.clip(res, 0, self.nbin-1)
            inp['pix_bbox_hough'] = res
        #
        elif self.mode == 'regr':
            def regr_toppos(left, pos, right):
                return (pos - left) / (right - left)
            def tolog(size):
                size = max(size,1) # make it positive
                return np.log(size / self.ref_scale) / np.log(self.log_scale)
            # for each pixel, find its x and y position
            yc,xc = np.float64(np.mgrid[0:big_h, 0:big_w]) + 0.5
            res = -np.ones((4, big_h, big_w), dtype=np.float32)
            res[0] = regr_toppos(ll, xc, rr)
            res[1] = regr_toppos(tt, yc, bb)
            res[2] = tolog(rr - ll)
            res[3] = tolog(bb - tt)
            inp['pix_bbox_regr'] = res
        #
        else:
            raise NotImplementedError()

        return inp
