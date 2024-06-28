#! /usr/bin/env python3
'''
Modules for torch.nn
'''
import copy
from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor, zeros, log, neg, clamp, mean, sigmoid, sum as tsum # pylint: disable=E0611
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

class SplitLinear(nn.Module):
    '''
    Acts similar to nn.Linear, but not fully-connected along the layer.
    '''
    def __init__(self, structure: Tuple[Tuple[int,int]], *args, device=None, dtype=None, **kwargs
                 ) -> None:
        super().__init__(*args, **kwargs)
        factory_kwargs = {'device': device, 'dtype': dtype}
        #
        self.structure  = structure
        self.layer      = [nn.Linear(in_features=in_num, out_features=out_num, bias=True, \
                                     **factory_kwargs) for in_num, out_num in self.structure]
        #
        self.weight     = nn.ParameterList([i.weight for i in self.layer])
        self.bias       = nn.ParameterList([i.bias for i in self.layer])
        #
        self.in_dims    = [i[0] for i in self.structure]
        self.out_dims   = [i[1] for i in self.structure]
        self.in_size    = int(np.sum(self.in_dims))
        self.out_size   = int(np.sum(self.out_dims))
        self.splits     = len(self.structure)
        #
        self.in_inds   = [0] + [np.sum(self.in_dims[0:i+1]) for i in range(len(self.in_dims)-1)]
        self.out_inds  = [0] + [np.sum(self.out_dims[0:i+1]) for i in range(len(self.out_dims)-1)]

    def forward(self, x: Tensor) -> Tensor:
        '''
        Pass x (Tensor) through layer.
        '''
        y = zeros([x.shape[0], self.out_size]).float()
        for c, (in_dim, out_dim, in_ind, out_ind) in \
            enumerate(zip(self.in_dims, self.out_dims, self.in_inds, self.out_inds)):
            y[:, out_ind : out_ind + out_dim] = self.layer[c](x[:, in_ind : in_ind + in_dim])
        return y

    def extra_repr(self) -> str:
        '''
        String representation
        '''
        return f'in_size={self.in_size}, out_size={self.out_size}, splits={self.splits}'

class CustomMLPNN(nn.Module):
    '''
    Provide a customized version of Basic_MLP_NN, with SplitLinear input.
    '''
    def __init__(self, *args,
                 input_structure: Tuple[Tuple[int,int]] = ((4096, 256),),
                 hidden_structure: Tuple[Tuple[int,int,float]] = ((256, 256, 0.0),)*4,
                 output_structure: Tuple[int, int, float] = (256, 1, 0.0),
                 add_sigmoid: bool = True, add_desc: str = '', **kwargs):
        #
        super(CustomMLPNN, self).__init__(*args, **kwargs)
        #
        self.input_structure    = input_structure
        self.hidden_structure   = hidden_structure
        self.output_structure   = output_structure
        self.add_sigmoid        = add_sigmoid
        self.add_desc           = add_desc
        self.scaler             = StandardScaler()
        #
        self.first_layer        = SplitLinear(structure=self.input_structure)
        self.hidden_layers      = nn.ModuleList([nn.Sequential(
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(in_features=in_num, out_features=out_num)
                                    )
                                        for in_num, out_num, dropout in self.hidden_structure
                                    ])
        #
        self.output_layer       = nn.Sequential(
                                    nn.ReLU(),
                                    nn.Dropout(p=self.output_structure[2]),
                                    nn.Linear(in_features=self.output_structure[0],
                                              out_features=self.output_structure[1]),
                                        *({nn.Sigmoid()} if self.add_sigmoid else {})
                                    )
        #
        self.desc = {'version': "Custom v0", 'input_structure': input_structure,
                         'hidden_structure': hidden_structure, 'output_structure': output_structure, 
                         'add_sigmoid': add_sigmoid, 'add_desc': add_desc}

    def get_scaler(self) -> StandardScaler:
        '''
        get scaler
        '''
        return self.scaler

    def set_scaler(self, scaler: StandardScaler):
        '''
        set scaler
        '''
        self.scaler = copy.deepcopy(scaler)

    def get_descriptor(self) -> dict:
        '''
        get descriptor
        '''
        return self.desc

    def forward(self, x: Tensor) -> torch.Tensor:
        '''
        Pass x (Tensor) through layer.
        '''
        y = self.first_layer(x)
        for layer in self.hidden_layers:
            y = layer(y)
        return self.output_layer(y)

class BasicMLPNN(nn.Module):
    '''
    Basic Multi-Layer Perceptron Neural Network
    '''
    def __init__(self, *args, input_ftrs: int = 4096, n_classes: int = 2, layer_size: int = 256,
                 num_layers: int = 4, dropout: float = 0.0, add_desc: str = '', **kwargs):
        #
        super(BasicMLPNN, self).__init__(*args, **kwargs)
        #
        self.input_ftrs = input_ftrs
        self.n_classes  = n_classes
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.dropout    = dropout
        self.add_desc   = add_desc
        self.scaler     = StandardScaler()
        #
        self.base_model = nn.Sequential(nn.Linear(in_features=self.input_ftrs,
                                                  out_features=self.layer_size))
        self.hidden = nn.ModuleList()
        #
        if self.num_layers > 1:
            for _ in range(self.num_layers-1):
                self.hidden.append(nn.Sequential(   nn.ReLU(),
                                                    nn.Dropout(p=self.dropout),
                                                    nn.Linear(in_features=self.layer_size,
                                                              out_features=self.layer_size)
                                                    ))
        self.output = nn.Sequential(nn.ReLU(),
                                    nn.Dropout(p=self.dropout),
                                    nn.Linear(in_features=self.layer_size,
                                              out_features=self.n_classes))
        #
        self.sigm = nn.Sigmoid()

    def get_scaler(self) -> StandardScaler:
        '''
        scaler getter
        '''
        return self.scaler

    def set_scaler(self, scaler: StandardScaler):
        '''
        scaler setter
        '''
        self.scaler = copy.deepcopy(scaler)

    def get_descriptor(self) -> dict:
        '''
        get unique descriptor
        '''
        if not self.add_desc:
            return {'version': "v0", 'input_ftrs': self.input_ftrs, 'n_classes': self.n_classes,
                        'layer_size': self.layer_size, 'num_layers': self.num_layers,
                        'dropout': self.dropout}
        return {'version': "v0", 'input_ftrs': self.input_ftrs, 'n_classes': self.n_classes,
                    'layer_size': self.layer_size, 'num_layers': self.num_layers, 
                    'dropout': self.dropout, 'add_desc': self.add_desc}

    def forward(self, x) -> torch.Tensor:
        '''
        pass x (Tensor) through network.
        '''
        y = self.base_model(x)
        for layer in self.hidden:
            y = layer(y)
        y = self.sigm(self.output(y))
        return y

class ClassWeightedBCELoss(nn.Module):
    '''
    Loss with balanced weights to satisfy sum(weights) == 1, weight[0] = loss_scale * weight[1]
    '''
    def __init__(self, loss_scale: float, *args, **kwargs):
        super(ClassWeightedBCELoss, self).__init__(*args, **kwargs)
        self.loss_scale = loss_scale
        self.weights    = [self.loss_scale/(self.loss_scale+1), 1/(self.loss_scale+1)]

    def forward(self, pred: Tensor, actual: Tensor) -> Tensor:
        '''
        pass pred (Tensor) through loss; aiming for actual
        '''
        pred = clamp(pred, min=1e-7, max=1-1e-7)
        loss = self.weights[1] * (actual * log(pred)) + \
               self.weights[0] * ((1 - actual) * log(1 - pred))
        return neg(mean(loss))

class WeightedMSELoss(nn.Module):
    '''
    Mean-Squared-Error Loss with weight
    '''
    def __init__(self, *args, weight_fp: float = 9.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_fp = weight_fp

    def forward(self, pred: Tensor, actual: Tensor):
        '''
        pass pred (Tensor) through loss; aiming for actual
        '''
        diff = actual - pred
        squared_loss = mean(diff**2)
        false_positives = clamp(diff, max=0)
        false_positives_loss = mean(false_positives**2) * self.weight_fp
        return squared_loss + false_positives_loss

class GTAwareWeightedMSELoss(nn.Module):
    '''
    Ground-Truth Aware, Mean-Squared-Error Loss with weight
    '''
    def __init__(self, *args, weight_fp=9.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_fp = weight_fp

    def forward(self, pred: Tensor, actual: Tensor, gt_error: Tensor, tolerance: Tensor): #pylint: disable=W0613
        '''
        pass pred (Tensor) through loss; aiming for actual
        '''
        diff            = actual - pred
        squared_loss    = mean(diff**2)
        bool_positives  = actual < 0.5
        bool_false      = pred >= 0.5
        # Create a term that unevenly penalises false positives depending on their severity:
        bool_false_positives = bool_false & bool_positives
        false_positives_loss = mean(bool_false_positives.float() * gt_error)
        # Create a term that evenly reduces the penality for all true positives:
        bool_true_positives = (~bool_false) & bool_positives
        true_positives_loss = mean(bool_true_positives.float() * squared_loss * -0.5)
        # print(squared_loss, false_positives_loss, not_true_positives_loss)
        # Stack:
        return squared_loss + false_positives_loss + true_positives_loss

class GTAwareWeightedMSELoss2(nn.Module):
    '''
    Ground-Truth Aware, Mean-Squared-Error Loss with weight
    '''
    def __init__(self, *args, weight_fp=9.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_fp = weight_fp

    def forward(self, pred: Tensor, actual: Tensor, gt_error: Tensor, tolerance: Tensor): #pylint: disable=W0613
        '''
        pass pred (Tensor) through loss; aiming for actual
        '''
        diff            = actual - pred
        squared_loss    = mean(diff**2)
        bool_positives  = actual < 0.5
        bool_false      = pred >= 0.5
        # Create a term that unevenly penalises false positives depending on their severity:
        bool_false_positives = bool_false & bool_positives
        false_positives_loss = mean(bool_false_positives.float() * gt_error)
        # Stack:
        return squared_loss + false_positives_loss

class GTAwareWeightedMSELoss3(nn.Module):
    '''
    Ground-Truth Aware, Mean-Squared-Error Loss with weight
    '''
    def __init__(self, *args, weight_fp=9.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_fp = weight_fp

    def forward(self, pred: Tensor, actual: Tensor, gt_error: Tensor, tolerance: Tensor): #pylint: disable=W0613
        '''
        pass pred (Tensor) through loss; aiming for actual
        '''
        diff            = actual - pred
        squared_loss    = mean(diff**2)
        #torch.sigmoid((a - b) * 100)  -> Smooth approximation of b < a
        bool_positives  = sigmoid((0.5 - actual) * 100) #actual < 0.5
        bool_false      = sigmoid((pred - 0.5) * 100) #pred > 0.5
        # Create a term that unevenly penalises false positives depending on their severity:
        bool_false_positives = bool_false * bool_positives
        false_positives_loss = mean(bool_false_positives.float() * gt_error)
        # Stack:
        return squared_loss + false_positives_loss

class GTAwareWeightedMSELoss4(nn.Module):
    '''
    Ground-Truth Aware, Mean-Squared-Error Loss with weight
    '''
    def __init__(self, *args, weight_fp=9.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_fp = weight_fp

    def forward(self, pred: Tensor, actual: Tensor, gt_error: Tensor, tolerance: Tensor): #pylint: disable=W0613
        '''
        pass pred (Tensor) through loss; aiming for actual
        '''
        diff            = actual - pred
        squared_loss    = mean(diff**2)
        #torch.sigmoid((a - b) * 100)  -> Smooth approximation of b < a
        bool_positives  = sigmoid((0.5 - actual) * 100) #actual < 0.5
        bool_false      = sigmoid((pred - 0.5) * 100) #pred > 0.5
        # Create a term that unevenly penalises false positives depending on their severity:
        bool_false_positives = bool_false * bool_positives
        false_positives_loss = mean(bool_false_positives.float() * (gt_error**2))
        # Stack:
        return squared_loss + false_positives_loss

class GTAwareWeightedMSELoss5(nn.Module):
    '''
    Ground-Truth Aware, Mean-Squared-Error Loss with weight
    '''
    def __init__(self, *args, weight_fp=9.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_fp = weight_fp

    def forward(self, pred: Tensor, actual: Tensor, gt_error: Tensor, tolerance: Tensor): #pylint: disable=W0613
        '''
        pass pred (Tensor) through loss; aiming for actual
        '''
        return tsum(gt_error * pred)

class GTAwareWeightedMSELoss6(nn.Module):
    '''
    Ground-Truth Aware, Mean-Squared-Error Loss with weight
    '''
    def __init__(self, *args, weight_fp=9.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_fp = weight_fp

    def forward(self, pred: Tensor, actual: Tensor, gt_error: Tensor, tolerance: Tensor): #pylint: disable=W0613
        '''
        pass pred (Tensor) through loss; aiming for actual
        '''
        return tsum(clamp(gt_error - tolerance, min=0) * pred)

class LeakyReLUMSE(nn.Module):
    '''
    Leaky Rectified Linear Unit Mean-Squared-Error Loss
    '''
    def __init__(self, *args, neg_slope=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.neg_slope = neg_slope

    def forward(self, pred: Tensor, actual: Tensor):
        '''
        pass pred (Tensor) through loss; aiming for actual
        '''
        return 1000*mean(F.leaky_relu(actual-pred, negative_slope=self.neg_slope)**2)

class BiasWeightedMSELoss(nn.Module):
    '''
    Mean-Squared-Error Loss with weight, but biased. Output takes mean.
    '''
    def forward(self, pred: Tensor, actual: Tensor):
        '''
        pass pred (Tensor) through loss; aiming for actual
        '''
        diff = actual - pred
        squared_loss = diff**2
        fp_mask = (pred < 0.5) & (actual < 0.5)
        false_positives_loss = fp_mask * diff
        return mean(squared_loss.float() + false_positives_loss.float())

class BiasWeightedMSELoss2(nn.Module):
    '''
    Mean-Squared-Error Loss with weight, but biased. Output takes sum.
    '''
    def forward(self, pred: Tensor, actual: Tensor):
        '''
        pass pred (Tensor) through loss; aiming for actual
        '''
        diff = actual - pred
        squared_loss = diff**2
        fp_mask = (pred < 0.5) & (actual < 0.5)
        false_positives_loss = fp_mask * diff
        return tsum(squared_loss.float() + false_positives_loss.float()) # sum instead of mean
