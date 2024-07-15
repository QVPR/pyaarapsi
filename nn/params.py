#! /usr/bin/env python3
'''
Parameters.
'''
import copy
import warnings
from typing import List, Tuple
import torch
import numpy as np
from matplotlib import image as mpl_image

from pyaarapsi.core.classes.defaultdict import DefaultDict
from pyaarapsi.vpr.classes.vprdescriptor import VPRDescriptor
from pyaarapsi.vpr.classes.data.svmparams import SVMToleranceMode
from pyaarapsi.vpr.vpr_dataset_tool import VPRDatasetProcessor
from pyaarapsi.vpr import config

from pyaarapsi.nn.enums import SampleMode, ScalerUsage, ApplyModel, ModelClass, GenMode, LossType, \
    AblationVersion
from pyaarapsi.nn.param_helpers import get_model_params, ParamHolder
from pyaarapsi.nn.vpr_helpers import make_vpr_dataset_params_subset, make_svm_dataset_params_subset
from pyaarapsi.vpr.classes.dimensions import ImageDimensions
from pyaarapsi.vpr.classes.data.rosbagdatafilter import RosbagDataFilter

# How to configure pyaarapsi's config:
# >>> import pyaarapsi.vpr_simple.config as config
# >>> from pathlib import Path
# >>> config.make_config(data_path=Path("D:/CADDRA/aarapsiprivate/datasets/aarapsi_robot_pack"),
#                        workspace_path_1=Path("D:/CADDRA/aarapsiprivate/"))
# Or:
# >>> config.make_config(data_path=Path("F:/CADDRA/aarapsiprivate/datasets/aarapsi_robot_pack"),
#                        workspace_path_1=Path("F:/CADDRA/aarapsiprivate/"))

_COMBOS  =  {   'Office':
                {   'SVM':      {'ref': 'sim_cw_3', 'qry': 'sim_cw_5'},
                    'Normal':   {'ref': 's5_ccw_1_c', 'qry': 's6_ccw_1_c'},
                    'Adverse':  {'ref': 's4_cw_1_c',  'qry': 'lab_adv_2_fixed_c'},
                    'tolerance': 0.5
                },
                'Campus': 
                {   'SVM':      {'ref': 'll_cw_1',  'qry': 'll_cw_3'},
                    'Normal':   {'ref': 'run3_fix_c', 'qry': 'run4_fix_c'},
                    'Adverse':  {'ref': 'run1_fix_c', 'qry': 'run5_ccw_c'},
                    'tolerance': 1.0
                }
            }
#pylint: disable=C0103,C3002
class DFVPR(ParamHolder):
    '''
    All of these parameters must impact data generation
    (For any usage)
    All variables starting with an underscore are ignored.
    ---
    These variables are used across the experiment and testing jupyter notebooks.
    '''
    ODOM_TOPIC: str                     = "/odom/true"
    ENC_TOPIC: str                      = "/jackal_velocity_controller/odom"
    IMG_TOPIC: str                      = "/ros_indigosdk_occam/image0/compressed"
    ODOM_TOPICS: List[str]              = (ODOM_TOPIC, ENC_TOPIC)
    IMG_TOPICS: List[str]               = (IMG_TOPIC,)
    IMG_DIMS: List[int]                 = ImageDimensions(width=64,height=64)
    SAMPLE_RATE: int                    = 10 * 1000
    SVM_FACTORS: DefaultDict            = DefaultDict(default=["grad", "va"])
    SVM_TOL_MODE: SVMToleranceMode      = SVMToleranceMode.DISTANCE
    COMBOS: dict                        = copy.deepcopy(_COMBOS)
    TRAIN_REF_IMAGE_FILTERS: Tuple[RosbagDataFilter]   = ()
    TRAIN_QRY_IMAGE_FILTERS: Tuple[RosbagDataFilter]   = ()
    TEST_REF_IMAGE_FILTERS: Tuple[RosbagDataFilter]    = ()
    TEST_QRY_IMAGE_FILTERS: Tuple[RosbagDataFilter]    = ()
    TRAIN_REF_FEATURE_FILTERS: Tuple[RosbagDataFilter] = ()
    TRAIN_QRY_FEATURE_FILTERS: Tuple[RosbagDataFilter] = ()
    TEST_REF_FEATURE_FILTERS: Tuple[RosbagDataFilter]  = ()
    TEST_QRY_FEATURE_FILTERS: Tuple[RosbagDataFilter]  = ()

class General(ParamHolder):
    '''
    None of these parameters are allowed to impact data generation
    (For any usage)
    All variables starting with an underscore are ignored.
    ---
    These variables are used across the experiment and testing jupyter notebooks.
    '''
    VPRDP_VERBOSE           = False
    # Stash a VPRDatasetProcessor here for collective use, to minimise memory consumption:
    _KWARGS_VERBOSE         = {"use_tqdm": True, "printer": None}
    _KWARGS_SILENT          = {"use_tqdm": False, "printer": lambda *args, **kwargs: None}
    VPR_DP                  = (lambda kw=_KWARGS_VERBOSE if VPRDP_VERBOSE
                                                            else _KWARGS_SILENT:
                                VPRDatasetProcessor(dataset_params=None, try_gen=True, cuda=True,
                                                    autosave=True, ros=False, root=None,
                                                    **kw))()
    NN_IM_SCRIPT_VERBOSE    = True
    FORCE_GENERATE          = False
    SKIP_SAVE               = False
    DEVICE                  = torch.device('cpu')
    PRED_TYPES              = ['vpr', 'nvp', 'nvr', 'svm', 'prd']
    MODE_NAMES              = {'vpr': 'VPR', 'nvp': '$N_P$', 'nvr': '$N_R$', 'prd': 'NN (Ours)',
                               'svm': 'SVM', 'gt': 'Ground Truth'}
    HUE_ORDER               = (lambda mn=MODE_NAMES, pt=PRED_TYPES:
                                [mn[i] for i in pt])()
    SET_ORDER               = ['Office\nNormal','Office\nAdverse',
                               'Campus\nNormal','Campus\nAdverse']
    _COLORS                 = ['#C5304B', '#F29E4C', '#EFEA5A', '#83E377', '#16DB93', '#048BA8']
    PALETTE                 = (lambda mn=MODE_NAMES, pt=PRED_TYPES, clrs=_COLORS:
                                {mn[k]: c for k,c in zip(pt + ['gt'], clrs)})()
    DIR_AP                  = config.get_wsp_path(1)
    DIR_NN_REP              = DIR_AP + "/Paper_1_Scripts/NN_Replacement"
    DIR_NN                  = DIR_NN_REP + "/networks"
    DIR_NN_DS               = DIR_NN_REP + "/datasets_training"
    DIR_EXP_DS              = DIR_NN_REP + "/datasets_experiments"
    DIR_MEDIA               = DIR_NN_REP + "/media"
    PATH_EXP2_KEY_RESULTS   = DIR_MEDIA + "/exp2_key_results%s"
    PATH_EXP2_EXT_RESULTS   = DIR_MEDIA + "/exp2_extended_key_results_vertical%s"
    PATH_EXP2_EXAMPLE       = DIR_MEDIA + "/exp2_along_path_example%s"
    BGIMG1                  = mpl_image.imread(DIR_AP + '/media/outdoors_2a.jpg')
    BGIMG2                  = mpl_image.imread(DIR_AP + '/media/carpet_2.jpg')

    # Ensure these do not get stored.
    ParamHolder.IGNORED_VARIABLES.extend(["VPR_DP", "BGIMG1", "BGIMG2", "DEVICE"])
    try:
        VPR_DP.init_all_containers()
    except:#pylint: disable=W0702
        warnings.warn("[params] Unable to initialize generation for GEN_IP.")

class DFGeneral(ParamHolder):
    '''
    All of these parameters must impact data generation
    (For any usage)
    All variables starting with an underscore are ignored.
    ---
    These variables are used across the experiment and testing jupyter notebooks.
    '''
    DESCRIPTOR_TYPES        = [VPRDescriptor.APGEM, VPRDescriptor.NETVLAD, VPRDescriptor.SALAD]
    VPR                     = DFVPR()
    SVM_SUBSETS             = {} # Populate this below:
    TRAIN_REF_SUBSETS       = {} # ^^
    TRAIN_QRY_SUBSETS       = {} # ^^
    TEST_REF_SUBSETS        = {} # ^^
    TEST_QRY_SUBSETS        = {} # ^^

for descriptor_type in DFGeneral.DESCRIPTOR_TYPES:
    DFGeneral.TRAIN_REF_SUBSETS[descriptor_type.name] = make_vpr_dataset_params_subset(
        img_topics=DFVPR.IMG_TOPICS, odom_topics=DFVPR.ODOM_TOPICS, \
        vpr_descriptors=(descriptor_type,), img_dims=DFVPR.IMG_DIMS, sample_rate=DFVPR.SAMPLE_RATE,
        image_filters=DFVPR.TRAIN_REF_IMAGE_FILTERS,
        feature_filters=DFVPR.TRAIN_REF_FEATURE_FILTERS)
    DFGeneral.TRAIN_QRY_SUBSETS[descriptor_type.name] = make_vpr_dataset_params_subset(
        img_topics=DFVPR.IMG_TOPICS, odom_topics=DFVPR.ODOM_TOPICS, \
        vpr_descriptors=(descriptor_type,), img_dims=DFVPR.IMG_DIMS, sample_rate=DFVPR.SAMPLE_RATE,
        image_filters=DFVPR.TRAIN_QRY_IMAGE_FILTERS,
        feature_filters=DFVPR.TRAIN_QRY_FEATURE_FILTERS)
    DFGeneral.TEST_REF_SUBSETS[descriptor_type.name] = make_vpr_dataset_params_subset(
        img_topics=DFVPR.IMG_TOPICS, odom_topics=DFVPR.ODOM_TOPICS, \
        vpr_descriptors=(descriptor_type,), img_dims=DFVPR.IMG_DIMS, sample_rate=DFVPR.SAMPLE_RATE,
        image_filters=DFVPR.TEST_REF_IMAGE_FILTERS,
        feature_filters=DFVPR.TEST_REF_FEATURE_FILTERS)
    DFGeneral.TEST_QRY_SUBSETS[descriptor_type.name] = make_vpr_dataset_params_subset(
        img_topics=DFVPR.IMG_TOPICS, odom_topics=DFVPR.ODOM_TOPICS, \
        vpr_descriptors=(descriptor_type,), img_dims=DFVPR.IMG_DIMS, sample_rate=DFVPR.SAMPLE_RATE,
        image_filters=DFVPR.TEST_QRY_IMAGE_FILTERS,
        feature_filters=DFVPR.TEST_QRY_FEATURE_FILTERS)
    DFGeneral.SVM_SUBSETS[descriptor_type.name] = make_svm_dataset_params_subset(
        tol_mode=DFVPR.SVM_TOL_MODE,
        ref_subset=copy.deepcopy(DFGeneral.TRAIN_REF_SUBSETS[descriptor_type.name]),
        qry_subset=copy.deepcopy(DFGeneral.TRAIN_QRY_SUBSETS[descriptor_type.name]))

class DFNNTrain(ParamHolder):
    '''
    All of these parameters must impact NN generation
    (For any usage)
    All variables starting with an underscore are ignored.
    ---
    These variables are used to generate or train a NN model.
    '''
    BATCH_SIZE              = 8         # pass x features through at once (typically powers of 2)
    LEARNING_RATE           = 0.00001   # on each pass, how much change is made to each weight
    MAX_EPOCH               = DefaultDict(default=-1, **{
                                          VPRDescriptor.APGEM.name: 50,
                                          VPRDescriptor.NETVLAD.name: 50,
                                          VPRDescriptor.SALAD.name: 100,
                                          })
    # MAX_EPOCH               = DefaultDict(default=100)
    TRAIN_CHECK_RATIO       = 0.8
    SAMPLE_MODE             = SampleMode.RANDOM
    GENERATE_MODE           = GenMode.NORM_SIMPLE_COMPONENTS
    QUERY_LENGTH            = 1         # Depth of history for statistical feature generation
    # These variables are absorbed into MODEL_PARAMS and are only for ModelClass.BASIC:
    _LAYER_SIZE             = 128       # number of nodes in each layer
    _NUM_LAYERS             = 4         # number of layers
    _DROPOUT                = 0.1       # percentage of ...
    _NUM_CLASSES            = 1         # binary
    _NUM_FEATURES           = GENERATE_MODE.get_num_features(query_length=QUERY_LENGTH)
    TRAIN_THRESHOLD         = DefaultDict(default=0.5)
    CONTINUOUS_MODEL        = False
    USE_FAKE_GOOD           = False
    APPLY_SCALERS           = ScalerUsage.STANDARD_SHARED # use NONE, NORM1 at own risk.
    APPLY_MODEL             = ApplyModel.USE_FUSED
    LOSS_TYPE               = DefaultDict(default=LossType.NONE, **{
                                          VPRDescriptor.APGEM.name: LossType.WEIGHTED_MSE_LOSS5,
                                          VPRDescriptor.NETVLAD.name: LossType.WEIGHTED_MSE_LOSS2,
                                          VPRDescriptor.SALAD.name: LossType.WEIGHTED_MSE_LOSS34,
                                          })
    VPR                     = DFVPR()
    QRY_SUBSETS             = copy.deepcopy(DFGeneral.TRAIN_QRY_SUBSETS)
    REF_SUBSETS             = copy.deepcopy(DFGeneral.TRAIN_REF_SUBSETS)
    MODEL_CLASS             = ModelClass.BASIC
    MODEL_PARAMS            = get_model_params(model_class=MODEL_CLASS, num_features=_NUM_FEATURES,
        num_classes=_NUM_CLASSES, layer_size=_LAYER_SIZE, num_layers=_NUM_LAYERS, dropout=_DROPOUT,
        generate_mode=GENERATE_MODE, query_length=QUERY_LENGTH)

class DFNNTest(ParamHolder):
    '''
    All of these parameters must impact data generated by NN
    (For any usage)
    All variables starting with an underscore are ignored.
    ---
    These variables are used when applying or testing a NN model.
    '''
    TEST_THRESHOLD          = DefaultDict(default=0.5)
    QRY_SUBSETS             = copy.deepcopy(DFGeneral.TEST_QRY_SUBSETS)
    REF_SUBSETS             = copy.deepcopy(DFGeneral.TEST_REF_SUBSETS)

class NNGeneral(ParamHolder):
    '''
    None of these parameters are allowed to impact NN generation
    (For any usage)
    All variables starting with an underscore are ignored.
    ---
    These variables are used whenever necessary as part of NN operations.
    '''
    VERBOSE                 = True
    FORCE_GENERATE          = False
    SKIP_SAVE               = False
    NUM_WORKERS             = 3 # num additional threads / CPU cores for data loader

class DFExperiment1(ParamHolder):
    '''
    All of these parameters must impact experiment dataset generation
    (For Experiment 1)
    All variables starting with an underscore are ignored.
    ---
    These variables are used in experiment 1.
    '''
    TRUNCATE_LENGTH         = 5
    SLICE_LENGTHS           = [5.0, 10.0, 25.0, 50.0]
    NUM_ITERATIONS          = 50
    ASSESS_TOLERANCE        = 0.50
    ROBOT_TOLERANCE         = -0.1
    GENERAL                 = DFGeneral()
    NN_TRAIN                = DFNNTrain()
    NN_TEST                 = DFNNTest()

class Experiment1(ParamHolder):
    '''
    None of these parameters are allowed to impact experiment dataset generation
    (For Experiment 1)
    All variables starting with an underscore are ignored.
    ---
    These variables are used in experiment 1.
    '''
    VERBOSE                 = True
    FORCE_GENERATE          = False
    SKIP_SAVE               = False

class DFExperiment2(ParamHolder):
    '''
    All of these parameters must impact experiment dataset generation
    (For Experiment 2)
    All variables starting with an underscore are ignored.
    ---
    These variables are used in experiment 2.
    '''
    HISTORY_LENGTH          = 1.5
    START_OFFSET            = np.max([10-HISTORY_LENGTH, 0])
    VERSION_ENUM            = AblationVersion.ORIGINAL
    GENERAL                 = DFGeneral()
    NN_TRAIN                = DFNNTrain()
    NN_TEST                 = DFNNTest()

class Experiment2(ParamHolder):
    '''
    None of these parameters are allowed to impact experiment dataset generation
    (For Experiment 2)
    All variables starting with an underscore are ignored.
    ---
    These variables are used in experiment 2.
    '''
    VERBOSE                 = True
    FORCE_GENERATE          = False
    SKIP_SAVE               = False
#pylint: enable=C0103, C3002
