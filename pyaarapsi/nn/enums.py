#! /usr/bin/env python3
'''
A collection of enumerations.
'''
from enum import Enum, unique
from functools import partial

from typing import Tuple

import numpy as np
from torch import nn

from pyaarapsi.nn.general_helpers import throw
from pyaarapsi.nn.modules import BasicMLPNN, CustomMLPNN, ClassWeightedBCELoss, WeightedMSELoss, \
                        BiasWeightedMSELoss, BiasWeightedMSELoss2, LeakyReLUMSE, \
                        GTAwareWeightedMSELoss, GTAwareWeightedMSELoss2, GTAwareWeightedMSELoss3, \
                        GTAwareWeightedMSELoss4, GTAwareWeightedMSELoss5, GTAwareWeightedMSELoss6

@unique
class TrainOrTest(Enum):
    '''
    Instead of True/False, for clarity.
    '''
    TRAIN   = 0
    TEST    = 1
    #
    @staticmethod
    class Exception(Exception):
        '''
        Bad usage
        '''

@unique
class AblationVersion(Enum):
    '''
    For Experiment 2: which type of ablation calculation to perform.
    '''
    ORIGINAL    = 0
    MEDIAN      = 1
    ADJ_MEDIAN  = 2
    #
    @staticmethod
    class Exception(Exception):
        '''
        Bad usage
        '''

@unique
class SampleMode(Enum):
    '''
    Which method for separating train and check data
    '''
    FRONT   = 0
    RANDOM  = 1
    #
    @staticmethod
    class Exception(Exception):
        '''
        Bad usage
        '''

@unique
class TrainData(Enum):
    '''
    Which data to use to train a model.
    '''
    OFFICE_SVM = "Office SVM"
    CAMPUS_SVM = "Campus SVM"
    BOTH_SVM   = "Office+Campus SVM"
    #
    @staticmethod
    class Exception(Exception):
        '''
        Bad usage
        '''

@unique
class ApplyModel(Enum):
    '''
    Controls TrainData
    '''
    USE_OFFICE      = 0 # Regardless of environment, TrainData.OFFICE_SVM
    USE_CAMPUS      = 1 # Regardless of environment, TrainData.CAMPUS_SVM
    USE_FUSED       = 2 # Regardless of environment, TrainData.BOTH_SVM
    MATCH_TO_TRAIN  = 3 # For Office use TrainData.OFFICE_SVM, for Campus use TrainData.CAMPUS_SVM
    #
    @staticmethod
    class Exception(Exception):
        '''
        Bad usage
        '''

@unique
class ScalerUsage(Enum):
    '''
    When generating datasets, which method for scaling and normalizing.
    '''
    NONE            = 0
    NORM1           = 1
    STANDARD        = 2
    STANDARD_SHARED = 3
    #
    @staticmethod
    class Exception(Exception):
        '''
        Bad usage
        '''

@unique
class ModelClass(Enum):
    '''
    When generating a model, which model class to use.
    '''
    BASIC           = BasicMLPNN
    CUSTOM          = CustomMLPNN
    #
    @staticmethod
    class Exception(Exception):
        '''
        Bad usage
        '''

@unique
class LossType(Enum):
    '''
    When generating a model, which loss to use.
    '''
    NONE                            = partial(lambda *args, **kwargs: throw(*args, **kwargs)) #pylint: disable=W0108
    L1LOSS                          = nn.L1Loss
    NLLLOSS                         = nn.NLLLoss
    POISSONNLLLOSS                  = nn.PoissonNLLLoss
    KLDIVLOSS                       = nn.KLDivLoss
    MSELOSS                         = nn.MSELoss
    BCELOSS                         = nn.BCELoss
    BCEWITHLOGITSLOSS               = nn.BCEWithLogitsLoss
    HINGEEMBEDDINGLOSS              = nn.HingeEmbeddingLoss
    MULTILABELMARGINLOSS            = nn.MultiLabelMarginLoss
    SMOOTHL1LOSS                    = nn.SmoothL1Loss
    HUBERLOSS                       = nn.HuberLoss
    SOFTMARGINLOSS                  = nn.SoftMarginLoss
    CROSSENTROPYLOSS                = nn.CrossEntropyLoss
    MULTILABELSOFTMARGINLOSS        = nn.MultiLabelSoftMarginLoss
    MULTIMARGINLOSS                 = nn.MultiMarginLoss
    CLASSWEIGHTEDBCELOSS1           = partial(lambda: ClassWeightedBCELoss(1))
    CLASSWEIGHTEDBCELOSS2           = partial(lambda: ClassWeightedBCELoss(2))
    CLASSWEIGHTEDBCELOSS3           = partial(lambda: ClassWeightedBCELoss(3))
    CLASSWEIGHTEDBCELOSS10          = partial(lambda: ClassWeightedBCELoss(10))
    CLASSWEIGHTEDBCELOSS40          = partial(lambda: ClassWeightedBCELoss(40))
    WEIGHTED_MSE_LOSS               = WeightedMSELoss
    WEIGHTED_MSE_LOSS2              = partial(lambda: WeightedMSELoss(weight_fp=2))
    WEIGHTED_MSE_LOSS5              = partial(lambda: WeightedMSELoss(weight_fp=5))
    WEIGHTED_MSE_LOSS34             = partial(lambda: WeightedMSELoss(weight_fp=34))
    BIAS_WEIGHTED_MSE_LOSS_MEAN     = BiasWeightedMSELoss
    BIAS_WEIGHTED_MSE_LOSS_SUM      = BiasWeightedMSELoss2
    LEAKY_MSE                       = LeakyReLUMSE
    GT_AWARE_WEIGHTED_MSE_LOSS      = GTAwareWeightedMSELoss
    GT_AWARE_WEIGHTED_MSE_LOSS2     = GTAwareWeightedMSELoss2
    GT_AWARE_WEIGHTED_MSE_LOSS3     = GTAwareWeightedMSELoss3
    GT_AWARE_WEIGHTED_MSE_LOSS4     = GTAwareWeightedMSELoss4
    GT_AWARE_WEIGHTED_MSE_LOSS5     = GTAwareWeightedMSELoss5
    GT_AWARE_WEIGHTED_MSE_LOSS6     = GTAwareWeightedMSELoss6
    # These don't work as their inputs aren't compatible
    # GAUSSIANNLLLOSS               = nn.GaussianNLLLoss
    # COSINEEMBEDDINGLOSS           = nn.CosineEmbeddingLoss
    # MARGINRANKINGLOSS             = nn.MarginRankingLoss
    # TRIPLETMARGINLOSS             = nn.TripletMarginLoss
    # TRIPLETMARGINWITHDISTANCELOSS = nn.TripletMarginWithDistanceLoss
    # CTCLOSS                       = nn.CTCLoss
    #
    @staticmethod
    class Exception(Exception):
        '''
        Bad usage
        '''

@unique
class GenMode(Enum):
    '''
    When generating datasets for neural network training, which method for generating features.
    '''
    SIMPLE_COMPONENTS                   = (0,  48,  4, False)
    COMPLEX_COMPONENTS                  = (1,  43,  4, False)
    NORM_SIMPLE_COMPONENTS              = (2,  48,  4, False)
    HIST_SIMPLE_COMPONENTS              = (4,  48,  4, False)
    LONG_HIST_SIMPLE_COMPONENTS         = (5,  48,  8, False)
    HIST_NORM_SIMPLE_COMPONENTS         = (6,  48,  4, False)
    LONG_HIST_NORM_SIMPLE_COMPONENTS    = (7,  48,  8, False)
    TINY_HIST_NORM_SIMPLE_COMPONENTS    = (8,  48,  4,  True)
    LONG_HIST2_NORM_SIMPLE_COMPONENTS   = (9,  48,  8, False)
    LONG_HIST3_NORM_SIMPLE_COMPONENTS   = (10, 48, 12, False)
    LONG_HIST4_NORM_SIMPLE_COMPONENTS   = (11, 48,  8, False)
    LONG_HIST5_NORM_SIMPLE_COMPONENTS   = (12, 48,  8, False)
    LONG_HIST6_NORM_SIMPLE_COMPONENTS   = (13, 48,  4, False)
    TINY2_HIST_NORM_SIMPLE_COMPONENTS   = (14, 48,  4, False)
    TINY3_HIST_NORM_SIMPLE_COMPONENTS   = (15, 48,  4, False)
    MATCH_INDS                          = (16,  0,  0,  True)
    TEST1                               = (17, 48,  1,  True)
    TEST2                               = (18, 48,  4,  True)

    def __init__(self, _, subcomponent_size, num_subcomponent_groups, uses_query_length):
        self.subcomponent_size       = subcomponent_size
        self.num_subcomponent_groups = num_subcomponent_groups
        self.uses_query_length       = uses_query_length

    def get_num_features(self, query_length: int) -> int:
        '''
        Get length of neural network input
        '''
        return (self.num_subcomponent_groups * self.subcomponent_size) \
                + (query_length if self.uses_query_length else 0)

    def get_structure(self, query_length: int) -> Tuple[Tuple[int, int]]:
        '''
        Get neural network structure (all layer shapes)
        '''
        return tuple([(self.subcomponent_size, np.min([self.subcomponent_size, 10])) \
                      for i in range(self.num_subcomponent_groups)] + \
                        ([(query_length, query_length)] if self.uses_query_length else []))

    def get_structure_output_size(self, query_length: int) -> int:
        '''
        Get length of neural network output
        '''
        return (np.min([self.subcomponent_size, 10]) * self.num_subcomponent_groups) \
                + (query_length if self.uses_query_length else 0)
    #
    @staticmethod
    class Exception(Exception):
        '''
        Bad usage
        '''
