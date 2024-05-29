#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import copy
import logging
import os
from contextlib import suppress
import gc
import datetime
from pathlib import Path
from ..vpr_simple import config
ROSPKG_ROOT = config.prep_rospkg_root()
from ..core.enum_tools import enum_get
from ..core.helper_tools import formatException, vis_dict
try:
    from ..core.ros_tools import process_bag
except:
    logging.warn('Could not access ros_tools; generating features from rosbags will fail. This is typically due to a missing or incorrect ROS installation. \nError code: \n%s' % formatException())
from ..core.roslogger import LogType, roslogger
from ..core.file_system_tools import scan_directory
from .vpr_helpers import filter_dataset, FeatureType, correct_filters, make_dataset_dictionary, getFeat
from ..vpr_classes.netvlad import NetVLAD_Container
from ..vpr_classes.hybridnet import HybridNet_Container
from ..vpr_classes.salad import SALAD_Container
from ..vpr_classes.apgem import APGEM_Container

from typing import Optional, Union, overload, Callable, List

class VPRProcessorBase:
    def __init__(self, cuda: bool = False, ros: bool = False, printer: Optional[Callable] = None,
                 init_netvlad: bool = False, init_hybridnet: bool = False, init_salad: bool = False,
                 init_apgem: bool = False):
        
        self.dataset: dict                  = {}
        self.dataset_params                 = {}
        self.printer: Optional[Callable]    = printer
        self.ros                            = ros
        self.cuda                           = cuda

        self.borrowed_nns                               = {'netvlad': False, 'hybridnet': False, 'salad': False, 'apgem': False}
        self.netvlad: Optional[NetVLAD_Container]       = None
        self.hybridnet: Optional[HybridNet_Container]   = None
        self.salad: Optional[SALAD_Container]           = None
        self.apgem: Optional[APGEM_Container]           = None
        self.init_netvlad                               = init_netvlad
        self.init_hybridnet                             = init_hybridnet
        self.init_salad                                 = init_salad
        self.init_apgem                                 = init_apgem

    def get(self) -> dict:
        '''
        Get whole dataset

        Inputs:
        - None
        Returns:
            dict type; whole dataset dictionary
        '''
        return self.dataset

    def get_data(self) -> dict:
        '''
        Get dataset data

        Inputs:
        - None
        Returns:
            dict type; dataset data dictionary
        '''
        if not self.dataset:
            return {}
        return self.dataset['dataset']

    def get_params(self) -> dict:
        '''
        Get dataset parameters

        Inputs:
        - None
        Returns:
            dict type; dataset parameter dictionary
        '''
        if not self.dataset:
            return {}
        return self.dataset['params']
    
    def show(self, printer = print) -> str:
        '''
        Print dataset contents

        Inputs:
        - printer:    method type {default: print}; will display contents through provided print method
        Returns:
            str type; representation of dataset dictionary contents
        '''
        return vis_dict(self.dataset, printer=printer)

    def print(self, text: str, logtype: LogType = LogType.INFO, throttle: float = 0) -> None:
        '''
        Diagnostics and debugging print handler

        Inputs:
        - text:     str type; contents to print
        - logtype:  LogType type; logging channel to write to
        - throttle: float type {default: 0}; rate to limit publishing contents at if repeatedly executed
        Returns:
            None
        '''
        text = '[VPRDatasetProcessor] ' + text
        if self.printer is None:
            roslogger(text, logtype, throttle=throttle, ros=self.ros)
        else:
            self.printer(text, logtype, throttle=throttle, ros=self.ros)

    def prep_netvlad(self, cuda: Optional[bool] = None, ready_up: bool = True) -> bool:
        '''
        Prepare netvlad for use

        Inputs:
        - cuda:     bool type {default: None}; if not None, overrides initialisation cuda variable for netvlad loading
        - ready_up: bool type {default: True}; whether or not to automatically ready the model
        Returns:
            bool type; True (on success, else Exception)
        '''
        if not isinstance(self.netvlad, NetVLAD_Container):
            if cuda is None:
                cuda = self.cuda
            self.netvlad        = NetVLAD_Container(cuda=cuda, ngpus=int(self.cuda), logger=self.print)
            self.init_netvlad   = True
        if ready_up:
            self.netvlad.ready_up()
        return True

    def prep_hybridnet(self, cuda: Optional[bool] = None, ready_up: bool = True) -> bool:
        '''
        Prepare hybridnet for use

        Inputs:
        - cuda:     bool type {default: None}; if not None, overrides initialisation cuda variable for hybridnet loading
        - ready_up: bool type {default: True}; whether or not to automatically ready the model
        Returns:
            bool type; True (on success, else Exception)
        '''
        if not isinstance(self.hybridnet, HybridNet_Container):
            if cuda is None:
                cuda = self.cuda
            self.hybridnet      = HybridNet_Container(cuda=cuda, logger=self.print)
            self.init_hybridnet = True
        if ready_up:
            self.hybridnet.ready_up()
        return True

    def prep_salad(self, cuda: Optional[bool] = None, ready_up: bool = True) -> bool:
        '''
        Prepare salad for use

        Inputs:
        - cuda:     bool type {default: None}; if not None, overrides initialisation cuda variable for salad loading
        - ready_up: bool type {default: True}; whether or not to automatically ready the model
        Returns:
            bool type; True (on success, else Exception)
        '''
        if not isinstance(self.salad, SALAD_Container):
            if cuda is None:
                cuda = self.cuda
            self.salad      = SALAD_Container(cuda=cuda, logger=self.print)
            self.init_salad = True
        if ready_up:
            self.salad.ready_up()
        return True

    def prep_apgem(self, cuda: Optional[bool] = None, ready_up: bool = True) -> bool:
        '''
        Prepare apgem for use

        Inputs:
        - cuda:     bool type {default: None}; if not None, overrides initialisation cuda variable for apgem loading
        - ready_up: bool type {default: True}; whether or not to automatically ready the model
        Returns:
            bool type; True (on success, else Exception)
        '''
        if not isinstance(self.apgem, SALAD_Container):
            if cuda is None:
                cuda = self.cuda
            self.apgem      = APGEM_Container(cuda=cuda, logger=self.print)
            self.init_apgem = True
        if ready_up:
            self.apgem.ready_up()
        return True
    
    def init_nns(self, netvlad: bool = True, hybridnet: bool = True, salad: bool = True, apgem: bool = True) -> None:
        '''
        Flag VPRDatasetProcessor to initialize specific feature extraction containers

        Inputs:
            processor:              VPRDatasetProcessor type
            netvlad:                bool type {default: True}; whether or not to initialize netvlad container
            hybridnet:              bool type {default: True}; whether or not to initialize hybridnet container
            salad:                  bool type {default: True}; whether or not to initialize salad container
            apgem:                  bool type {default: True}; whether or not to initialize apgem container
        Returns:
            None
        '''
        if netvlad:
            self.init_netvlad = True
        if hybridnet:
            self.init_hybridnet = True
        if salad:
            self.init_salad = True
        if apgem:
            self.init_apgem = True

    def pass_nns(self, processor: VPRDatasetProcessor, try_load_if_missing: bool = True, 
                 netvlad: bool = True, hybridnet: bool = True, salad: bool = True,
                 apgem: bool = True) -> bool:
        '''
        Overwrite this VPRDatasetProcessor's instances of each container with another's instantiations
        Will check and, if initialised, destroy existing container

        Inputs:
            processor:              VPRDatasetProcessor type
            try_load_if_missing:    bool type {default: True}; whether or not to trigger a prep if detects uninitialised container instance
            netvlad:                bool type {default: True}; whether or not to overwrite netvlad instance
            hybridnet:              bool type {default: True}; whether or not to overwrite hybridnet instance
            salad:                  bool type {default: True}; whether or not to overwrite salad instance
            apgem:                  bool type {default: True}; whether or not to overwrite apgem instance
        Returns:
            bool type; True (on success, else Exception)
        '''
        assert isinstance(processor, VPRDatasetProcessor)
        if netvlad:
            if processor.netvlad is None:
                if try_load_if_missing:
                    processor.prep_netvlad()
                else:
                    raise Exception('Passing requires a NetVLAD_Container, or pass argument try_load_if_missing=True.')
            if isinstance(self.netvlad, NetVLAD_Container):
                try:
                    self.netvlad.destroy()
                except:
                    self.print('Failed to destroy existing netvlad instance, with error: ' + str(formatException()))
            self.netvlad = processor.netvlad
            self.init_netvlad = True
            self.borrowed_nns['netvlad'] = True
            self.print('Passed NetVLAD.', LogType.DEBUG)
        if hybridnet:
            if processor.hybridnet is None:
                if try_load_if_missing:
                    processor.prep_hybridnet()
                else:
                    raise Exception('Passing requires a HybridNet_Container, or pass argument try_load_if_missing=True.')
            if isinstance(self.hybridnet, HybridNet_Container):
                try:
                    self.hybridnet.destroy()
                except:
                    self.print('Failed to destroy existing hybridnet instance, with error: ' + str(formatException()))
            self.hybridnet = processor.hybridnet
            self.init_hybridnet = True
            self.borrowed_nns['hybridnet'] = True
            self.print('Passed HybridNet.', LogType.DEBUG)
        if salad:
            if processor.salad is None:
                if try_load_if_missing:
                    processor.prep_salad()
                else:
                    raise Exception('Passing requires a SALAD_Container, or pass argument try_load_if_missing=True.')
            if isinstance(self.salad, SALAD_Container):
                try:
                    self.salad.destroy()
                except:
                    self.print('Failed to destroy existing salad instance, with error: ' + str(formatException()))
            self.salad = processor.salad
            self.init_salad = True
            self.borrowed_nns['salad'] = True
            self.print('Passed SALAD.', LogType.DEBUG)
        if apgem:
            if processor.apgem is None:
                if try_load_if_missing:
                    processor.prep_apgem()
                else:
                    raise Exception('Passing requires a APGEM_Container, or pass argument try_load_if_missing=True.')
            if isinstance(self.apgem, APGEM_Container):
                try:
                    self.apgem.destroy()
                except:
                    self.print('Failed to destroy existing apgem instance, with error: ' + str(formatException()))
            self.apgem = processor.apgem
            self.init_apgem = True
            self.borrowed_nns['apgem'] = True
            self.print('Passed APGEM.', LogType.DEBUG)
        return True

    def check_netvlad(self, ft_types: list) -> bool:
        '''
        Check if NetVLAD is initialised: if not initialised but needed (ft_types contains FeatureType.NETVLAD) will attempt to initialise.
        Delays loading of NetVLAD model until required by system.

        Inputs:
            ft_types:  list type; list of FeatureType enumerations. If it contains FeatureType.NETVLAD and NetVLAD is not loaded, this method will attempt to initialise NetVLAD
        Returns:
            bool type; True on successful update, False if no change
        '''
        if (FeatureType.NETVLAD in ft_types) and self.init_netvlad: # If needed and we've been asked to initialise NetVLAD:
            if self.netvlad is None: # if it currently doesn't exist,
                self.prep_netvlad(self.cuda, True)
            elif not self.netvlad.is_ready(): # if it exists but isn't ready,
                self.netvlad.ready_up()
            else:
                return False
            return True
        return False
    
    def check_hybridnet(self, ft_types: list) -> bool:
        '''
        Check if HybridNet is initialised: if not initialised but needed (ft_types contains FeatureType.HYBRIDNET) will attempt to initialise.
        Delays loading of HybridNet model until required by system.

        Inputs:
            ft_types:  list type; list of FeatureType enumerations. If it contains FeatureType.HYBRIDNET and HybridNet is not loaded, this method will attempt to initialise HybridNet
        Returns:
            bool type; True on successful update, False if no change
        '''
        if (FeatureType.HYBRIDNET in ft_types) and self.init_hybridnet: # If needed and we've been asked to initialise HybridNet:
            if self.hybridnet is None: # if it currently doesn't exist,
                self.prep_hybridnet(self.cuda, True)
            elif not self.hybridnet.is_ready(): # if it exists but isn't ready,
                self.hybridnet.ready_up()
            else:
                return False
            return True
        return False
    
    def check_salad(self, ft_types: list) -> bool:
        '''
        Check if SALAD is initialised: if not initialised but needed (ft_types contains FeatureType.SALAD) will attempt to initialise.
        Delays loading of SALAD model until required by system.

        Inputs:
            ft_types:  list type; list of FeatureType enumerations. If it contains FeatureType.SALAD and SALAD is not loaded, this method will attempt to initialise SALAD
        Returns:
            bool type; True on successful update, False if no change
        '''
        if (FeatureType.SALAD in ft_types) and self.init_salad: # If needed and we've been asked to initialise SALAD:
            if self.salad is None: # if it currently doesn't exist,
                self.prep_salad(self.cuda, True)
            elif not self.salad.is_ready(): # if it exists but isn't ready,
                self.salad.ready_up()
            else:
                return False
            return True
        return False
    
    def check_apgem(self, ft_types: list) -> bool:
        '''
        Check if APGEM is initialised: if not initialised but needed (ft_types contains FeatureType.APGEM) will attempt to initialise.
        Delays loading of APGEM model until required by system.

        Inputs:
            ft_types:  list type; list of FeatureType enumerations. If it contains FeatureType.APGEM and APGEM is not loaded, this method will attempt to initialise APGEM
        Returns:
            bool type; True on successful update, False if no change
        '''
        if (FeatureType.APGEM in ft_types) and self.init_apgem: # If needed and we've been asked to initialise APGEM:
            if self.apgem is None: # if it currently doesn't exist,
                self.prep_apgem(self.cuda, True)
            elif not self.apgem.is_ready(): # if it exists but isn't ready,
                self.apgem.ready_up()
            else:
                return False
            return True
        return False
    
    @overload
    def getFeat(self, img: np.ndarray, fttype_in: FeatureType, dims: list, use_tqdm: bool = False) -> np.ndarray: ...
    
    @overload
    def getFeat(self, img: List[np.ndarray], fttype_in: FeatureType, dims: list, use_tqdm: bool = False) -> List[np.ndarray]: ...
        
    def getFeat(self, img: Union[np.ndarray, List[np.ndarray]], fttype_in: FeatureType, dims: list, use_tqdm: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
        '''
        Feature Extraction Helper

        Inputs:
        - img:          np.ndarray type (or list of np.ndarray); Image array (can be RGB or greyscale; if greyscale, will be stacked to RGB equivalent dimensions for neural network input.
        - fttype_in:    FeatureType type; Type of feature to extract from each image.
        - dims:         list type; list of int types (two-positive-integer list). Dimensions to reduce input images to (height x width).
        - use_tqdm:     bool type {default: False}; whether or not to display extraction/loading statuses using tqdm
        Returns:
            np.ndarray type (or list of np.ndarray); flattened features from image

        '''
        # Get features from img, using VPRDatasetProcessor's set image dimensions.
        # Specify type via fttype_in= (from FeatureType enum; list of FeatureType elements is also handled)
        # Specify dimensions with dims= (two-element positive integer tuple)
        # Returns feature array, as a flattened array

        if not (dims[0] > 0 and dims[1] > 0):
            raise Exception("[getFeat] image dimensions are invalid")
        if not isinstance(fttype_in, list):
            fttypes = [fttype_in]
        else:
            fttypes = fttype_in
        if any([fttype.name == FeatureType.HYBRIDNET.name for fttype in fttypes]):
            if self.init_hybridnet:
                self.check_hybridnet(fttypes)
            else:
                raise Exception("[getFeat] FeatureType.HYBRIDNET provided but VPRDatasetProcessor not initialised with init_hybridnet=True")
        if any([fttype.name == FeatureType.NETVLAD.name for fttype in fttypes]):
            if self.init_netvlad:
                self.check_netvlad(fttypes)
            else:
                raise Exception("[getFeat] FeatureType.NETVLAD provided but VPRDatasetProcessor not initialised with init_netvlad=True")
        if any([fttype.name == FeatureType.SALAD.name for fttype in fttypes]):
            if self.init_salad:
                self.check_salad(fttypes)
            else:
                raise Exception("[getFeat] FeatureType.SALAD provided but VPRDatasetProcessor not initialised with init_salad=True")
        if any([fttype.name == FeatureType.APGEM.name for fttype in fttypes]):
            if self.init_apgem:
                self.check_apgem(fttypes)
            else:
                raise Exception("[getFeat] FeatureType.APGEM provided but VPRDatasetProcessor not initialised with init_apgem=True")
        try:
            feats = getFeat(img, fttypes, dims, use_tqdm=use_tqdm, nn_hybrid=self.hybridnet, nn_netvlad=self.netvlad, nn_salad=self.salad, nn_apgem=self.apgem)
            if isinstance(feats, list):
                return [np.array(i, dtype=np.float32) for i in feats]
            return np.array(feats, dtype=np.float32)
        except Exception as e:
            raise Exception("[getFeat] Feature vector could not be constructed.\nCode: %s" % (e))

class VPRDatasetProcessor(VPRProcessorBase):
    def __init__(self, dataset_params: Optional[dict] = None, try_gen: bool = True, 
                 init_netvlad: bool = False, init_hybridnet: bool = False, init_salad: bool = False,
                 init_apgem: bool = False,
                 cuda: bool = False, use_tqdm: bool = False, 
                 autosave: bool = False, ros: bool = True, root: Optional[str] = None, printer=None):
        '''
        Initialisation

        Inputs:
        - dataset_params:   dict type with following keys (str type) and values (detailed below):
                            - npz_dbp:        str type; directory relative to root to find compressed data sets
                            - bag_dbp:        str type; directory relative to root to find rosbags
                            - bag_name:       str type; name of rosbag to find
                            - sample_rate:    float type; rate to sample messages in rosbag
                            - odom_topic:     str type; name of ROS topic to extract nav_msgs/Odometry messages from. If a list is provided, each will be stored as a column in the dataset
                            - img_topics:     list type; list of str types. names of ROS topics to extract sensor_msgs/CompressedImage messages from.
                            - img_dims:       list type; list of int types (two-positive-integer list). Dimensions to reduce input images to (height x width).
                            - ft_types:       list type; list of str types (names from FeatureType). Features to extract from each image; will become accessible keys in the dataset.
                            - filters:        str type; json string for designating filters to be applied. See pyaarapsi.vpr_simple.vpr_helpers.filter_dataset()
        - try_gen:          bool type {default: True}; whether or not to attempt generation if load fails
        - init_netvlad:     bool type {default: False}; whether or not to initialise netvlad (loads model)
        - init_hybridnet:   bool type {default: False}; whether or not to initialise hybridnet (loads model)
        - init_salad:       bool type {default: False}; whether or not to initialise salad (loads model)
        - init_apgem:       bool type {default: False}; whether or not to initialise apgem (loads model)
        - cuda:             bool type {default: False}; whether or not to use CUDA for feature type/GPU acceleration
        - use_tqdm:         bool type {default: False}; whether or not to display extraction/loading statuses using tqdm
        - autosave:         bool type {default: False}; whether or not to automatically save any generated datasets
        - ros:              bool type {default: True}; whether or not to use rospy logging (requires operation within ROS node scope)
        - root:             str type {default: None}; base root inserted in front of npz_dbp, bag_dbp, and svm_dbp
        - printer:          method {default: None}; if provided, overrides logging and will pass inputs to specified method on print
        Returns:
            self
        '''
        
        super().__init__(cuda=cuda, ros=ros, printer=printer,
                         init_netvlad=init_netvlad, init_hybridnet=init_hybridnet, init_salad=init_salad, init_apgem=init_apgem)

        self.use_tqdm       = use_tqdm
        self.autosave       = autosave

        if root is None:
            self.root       = ROSPKG_ROOT
        else:
            self.root       = root

        # Declare attributes:
        self.npz_dbp: str                               = ''
        self.bag_dbp: str                               = ''

        if not (dataset_params is None): # If parameters have been provided:
            self.print("Loading model from parameters...", LogType.DEBUG)
            name = self.load_dataset(dataset_params, try_gen=try_gen)

            if not self.dataset:
                raise Exception("Dataset load failed.")
            self.print("Dataset Ready (loaded: %s)." % str(name))

        else: # None-case needed for SVM training
            self.print("Ready; no dataset loaded.")

    def unload(self):
        try:
            del self.dataset
        except:
            pass
        self.dataset = {}

    def generate_dataset(self, npz_dbp: str, bag_dbp: str, bag_name: str, sample_rate: float, 
                         odom_topic: str, img_topics: list, img_dims: list, ft_types: list, 
                         filters: Union[str, dict] = {}, store: bool = True) -> dict:
        '''
        Generate new datasets from parameters
        Inputs are specified such that, other than store, a correct dataset parameters dictionary can be dereferenced to autofill inputs

        Inputs:
        - npz_dbp:        str type; directory relative to root to find compressed data sets
        - bag_dbp:        str type; directory relative to root to find rosbags
        - bag_name:       str type; name of rosbag to find
        - sample_rate:    float type; rate to sample messages in rosbag
        - odom_topic:     str type; name of ROS topic to extract nav_msgs/Odometry messages from. If a list is provided, each will be stored as a column in the dataset
        - img_topics:     list type; list of str types. names of ROS topics to extract sensor_msgs/CompressedImage messages from.
        - img_dims:       list type; list of int types (two-positive-integer list). Dimensions to reduce input images to (height x width).
        - ft_types:       list type; list of str types (names from FeatureType). Features to extract from each image; will become accessible keys in the dataset.
        - filters:        str type; json string for designating filters to be applied. See pyaarapsi.vpr_simple.vpr_helpers.filter_dataset()
        - store:          bool type {default: True}; whether or not to store in VPRDatasetProcessor
        Returns:
            dict type; Generated dataset dictionary
        '''
        self.dataset_params = make_dataset_dictionary(bag_name=bag_name, npz_dbp=npz_dbp, bag_dbp=bag_dbp, 
                                                      odom_topic=odom_topic, img_topics=img_topics, sample_rate=sample_rate,
                                                      ft_types=ft_types, img_dims=img_dims, filters=filters)
        self.check_netvlad(ft_types)
        self.check_hybridnet(ft_types)

        # store for access in saving operation:
        if store:
            self.npz_dbp        = npz_dbp
            self.bag_dbp        = bag_dbp

        # generate:
        rosbag_dict         = process_bag(self.root +  '/' + bag_dbp + '/' + bag_name, sample_rate, odom_topic, img_topics, 
                                          printer=self.print, use_tqdm=self.use_tqdm)
        gc.collect()
        self.print('[generate_dataset] Performing feature extraction...')

        feat_vect_dict_raw  = {ft_type: np.stack([self.getFeat(list(rosbag_dict[i]), # Extract features ...
                                                    enum_get(ft_type, FeatureType), # convert string to enum
                                                    img_dims, use_tqdm=self.use_tqdm)
                                        for i in img_topics], axis=1) # for each image topic provided,
                                    for ft_type in ft_types} # for each feature type provided
        [rosbag_dict.pop(i,None) for i in img_topics] # reduce memory overhead

        # Flatten arrays if only one image topic provided:
        feature_vector_dict = {i: feat_vect_dict_raw[i][:,0] \
                               if feat_vect_dict_raw[i].shape[1] == 1 else feat_vect_dict_raw[i] \
                                for i in feat_vect_dict_raw}
        
        del feat_vect_dict_raw
        gc.collect()

        self.print('[generate_dataset] Done.')

        # Create dataset dictionary and add feature vectors
        params_dict         = dict( bag_name=bag_name, npz_dbp=npz_dbp, bag_dbp=bag_dbp, odom_topic=odom_topic, img_topics=img_topics, \
                                    sample_rate=sample_rate, ft_types=ft_types, img_dims=img_dims, filters=filters)
        dataset_dict        = dict( time=rosbag_dict.pop('t'), \
                                    px=rosbag_dict.pop('px'), py=rosbag_dict.pop('py'), pw=rosbag_dict.pop('pw'), \
                                    vx=rosbag_dict.pop('vx'), vy=rosbag_dict.pop('vy'), vw=rosbag_dict.pop('vw') )
        dataset_dict.update(feature_vector_dict)
        dataset_raw         = dict(params=params_dict, dataset=dataset_dict)
        dataset             = filter_dataset(dataset_raw, _printer=lambda msg: self.print(msg, LogType.DEBUG))

        if store:
            if hasattr(self, 'dataset'):
                del self.dataset
            self.dataset        = copy.deepcopy(dataset)

            if self.autosave:
                self.save_dataset()

        return dataset

    def save_dataset(self, name: Optional[str] = None, allow_overwrite: bool = False) -> VPRDatasetProcessor:
        '''
        Save loaded dataset to file system

        Inputs:
        - Name: str type {default: None}; if None, a unique name will be generated of the format dataset_%Y%m%d_X.
        Returns:
            self
        '''
        
        if not self.dataset:
            raise Exception("Dataset not loaded in system. Either call 'generate_dataset' or 'load_dataset' before using this method.")
        dir = self.root + '/' + self.npz_dbp
        Path(dir).mkdir(parents=False, exist_ok=True)
        Path(dir+"/params").mkdir(parents=False, exist_ok=True)
        
        # Ensure file name is of correct format, generate if not provided
        file_list, _, _, = scan_directory(dir, short_files=True)
        overwriting = False
        if (not name is None):
            if not (name.startswith('dataset')):
                name = "dataset_" + name
            if (name in file_list):
                if (not allow_overwrite):
                    raise Exception("Dataset with name %s already exists in directory." % name)
                else:
                    overwriting = True
        else:
            name = datetime.datetime.today().strftime("dataset_%Y%m%d")

        self.print("[save_dataset] Splitting dataset into files for feature types: %s" % self.dataset['params']['ft_types'], LogType.DEBUG)
        for ft_type in self.dataset['params']['ft_types']:
            # Generate unique name:
            file_name = name
            if not overwriting:
                count = 0
                while file_name in file_list:
                    file_name = name + "_%d" % count
                    count += 1
            file_list       = file_list + [file_name]
            full_file_path  = dir + "/" + file_name
            full_param_path = dir + "/params/" + file_name

            sub_data                = copy.deepcopy({key: self.dataset['dataset'][key] for key in ['time', 'px', 'py', 'pw', 'vx', 'vy', 'vw', ft_type]})
            sub_params              = copy.deepcopy(self.dataset['params'])
            sub_params['ft_types']  = [ft_type]
            sub_dataset             = dict(params=sub_params, dataset=sub_data)

            if not overwriting:
                file_ = self._check(params=sub_params)
                if file_:
                    self.print("[save_dataset] File exists with identical parameters (%s); skipping save." % file_, LogType.DEBUG)
                    continue
            
            if overwriting:
                with suppress(FileNotFoundError): os.remove(full_file_path)
                with suppress(FileNotFoundError): os.remove(full_param_path)
            np.savez(full_file_path, **sub_dataset)
            np.savez(full_param_path, params=sub_dataset['params']) # save whole dictionary to preserve key object types
            if overwriting:
                self.print("[save_dataset] Overwrite complete.\n\t  file: %s\n\tparams: %s." % (full_file_path, full_param_path))
            else:
                self.print("[save_dataset] Save complete.\n\t  file: %s\n\tparams: %s." % (full_file_path, full_param_path))
            del sub_dataset
        return self

    def extend_dataset(self, new_ft_type: FeatureType, try_gen: bool = False, save: bool = False) -> bool:
        '''
        Add an additional feature type into the loaded data set

        Inputs:
        - new_ft_type: FeatureType type; new feature to load
        - try_gen:     bool type {default: True}; whether or not to attempt generation if load fails
        - save:        bool type {default: False}; whether or not to automatically save any generated datasets
        Returns:
            bool type; True if successful extension.
        '''
        if not self.dataset:
            self.print("[extend_dataset] Load failed; no dataset loaded to extend.", LogType.DEBUG)
            return False
        if isinstance(new_ft_type, FeatureType):
            str_ft_type = new_ft_type.name
        else:
            str_ft_type = new_ft_type
        new_params = copy.deepcopy(self.dataset['params'])
        new_params['ft_types'] = [str_ft_type]
        self.print("[extend_dataset] Loading dataset to extend existing...", LogType.DEBUG)
        new_dataset = self.open_dataset(new_params, try_gen=try_gen)
        if new_dataset is None:
            self.print("[extend_dataset] Load failed.", LogType.DEBUG)
            return False
        self.print("[extend_dataset] Load success. Extending.", LogType.DEBUG)
        self.dataset['dataset'][str_ft_type] = copy.deepcopy(new_dataset['dataset'][str_ft_type])
        self.dataset['params']['ft_types'].append(str_ft_type)
        if save:
            self.save_dataset()
        return True

    def open_dataset(self, dataset_params: dict, try_gen: bool = False) -> Union[dict, None]:
        '''
        Open a dataset by searching for a param dictionary match (return as a variable, do not store in VPRDatasetProcessor)

        Inputs:
        - dataset_params:   dict type with following keys (str type) and values (detailed below):
                            - npz_dbp:        str type; directory relative to root to find compressed data sets
                            - bag_dbp:        str type; directory relative to root to find rosbags
                            - bag_name:       str type; name of rosbag to find
                            - sample_rate:    float type; rate to sample messages in rosbag
                            - odom_topic:     str type; name of ROS topic to extract nav_msgs/Odometry messages from. If a list is provided, each will be stored as a column in the dataset
                            - img_topics:     list type; list of str types. names of ROS topics to extract sensor_msgs/CompressedImage messages from.
                            - img_dims:       list type; list of int types (two-positive-integer list). Dimensions to reduce input images to (height x width).
                            - ft_types:       list type; list of str types (names from FeatureType). Features to extract from each image; will become accessible keys in the dataset.
                            - filters:        str type; json string for designating filters to be applied. See pyaarapsi.vpr_simple.vpr_helpers.filter_dataset()
        - try_gen:          bool type {default: False}; whether or not to attempt generation if load fails
        Returns:
            dict type; opened dataset dictionary if successful, else None.
        '''

        # load via search for param match but don't load into the processor's self.dataset
        if not self.npz_dbp:
            self.npz_dbp = dataset_params['npz_dbp']
        else:
            assert self.npz_dbp == dataset_params['npz_dbp'], 'Must preserve npz_dbp'
        if not self.bag_dbp:
            self.bag_dbp = dataset_params['bag_dbp']
        else:
            assert self.bag_dbp == dataset_params['bag_dbp'], 'Must preserve bag_dbp'
        assert len(dataset_params['ft_types']) == 1, 'Can only open one dataset'

        self.print("[open_dataset] Loading dataset.", LogType.DEBUG)
        datasets = self._get_datasets()
        for name in datasets:
            if datasets[name]['params'] == dataset_params:
                try:
                    self.print("[open_dataset] Loading ...", LogType.DEBUG)
                    return self._load(name, store=False)
                except:
                    self._fix(name)
        if try_gen:
            try:
                self.print("[open_dataset] Generating ...", LogType.DEBUG)
                return self.generate_dataset(**dataset_params, store=False)
            except:
                self.print(formatException())
        
        self.print("[open_dataset] Load, generation failed.", LogType.DEBUG)
        return None
    
    def get_bag_path(self):
        assert not (self.dataset_params is None)
        return self.root +  '/' + str(self.dataset_params['bag_dbp']) + '/' + str(self.dataset_params['bag_name'])
    
    def get_dataset_params(self):
        assert not (self.dataset_params is None)
        return copy.deepcopy(self.dataset_params)

    def load_dataset(self, dataset_params: dict, try_gen: bool = False) -> str:
        '''
        Load a dataset by searching for a param dictionary match into VPRDatasetProcessor

        Inputs:
        - dataset_params:   dict type with following keys (str type) and values (detailed below):
                            - npz_dbp:        str type; directory relative to root to find compressed data sets
                            - bag_dbp:        str type; directory relative to root to find rosbags
                            - bag_name:       str type; name of rosbag to find
                            - sample_rate:    float type; rate to sample messages in rosbag
                            - odom_topic:     str type; name of ROS topic to extract nav_msgs/Odometry messages from. If a list is provided, each will be stored as a column in the dataset
                            - img_topics:     list type; list of str types. names of ROS topics to extract sensor_msgs/CompressedImage messages from.
                            - img_dims:       list type; list of int types (two-positive-integer list). Dimensions to reduce input images to (height x width).
                            - ft_types:       list type; list of str types (names from FeatureType). Features to extract from each image; will become accessible keys in the dataset.
                            - filters:        str type; json string for designating filters to be applied. See pyaarapsi.vpr_simple.vpr_helpers.filter_dataset()
        - try_gen:          bool type {default: False}; whether or not to attempt generation if load fails
        Returns:
            str type; loaded dataset dictionary file name if successful, 'NEW GENERATION' if a file is generated, else ''.
        '''
        self.dataset_params = copy.deepcopy(dataset_params)
        orig_filters = copy.deepcopy(self.dataset_params['filters'])
        self.dataset_params.pop('filters')
        self.dataset_params['filters'], filters_corrected = correct_filters(orig_filters)
        if filters_corrected: self.print('[load_dataset] Trivial filters provided, reduced from: <%s, %s> to: <%s, %s>' \
                                         % (str(orig_filters), str(type(orig_filters)), \
                                            str(self.dataset_params['filters']), str(type(self.dataset_params['filters']))), LogType.WARN)
        self.npz_dbp = self.dataset_params['npz_dbp']
        self.bag_dbp = self.dataset_params['bag_dbp']

        self.print("[load_dataset] Loading dataset.")
        datasets = self._get_datasets()
        sub_params = copy.deepcopy(self.dataset_params)
        sub_params['ft_types'] = [self.dataset_params['ft_types'][0]]
        if (self.dataset_params['filters'] == {}):
            _name = self._filter_load_helper(datasets=datasets, sub_params=copy.deepcopy(sub_params), save=try_gen)
        else:
            _name = self._normal_load_helper(datasets=datasets, sub_params=copy.deepcopy(sub_params))
        if not _name == '':
            if len(self.dataset_params['ft_types']) > 1:
                for ft_type in self.dataset_params['ft_types'][1:]:
                    if not self.extend_dataset(ft_type, try_gen=try_gen, save=try_gen):
                        return ''
            return _name
        else: 
            if try_gen:
                self.print('[load_dataset] Generating dataset with params: %s' % (str(self.dataset_params)), LogType.DEBUG)
                self.generate_dataset(**self.dataset_params)
                return 'NEW GENERATION'
            return ''
        
    def _filter_load_helper(self, datasets: dict, sub_params: dict, save: bool = True):
        # Attempt to load, as-is:
        name_from_filter_load = self._normal_load_helper(datasets=datasets, sub_params=sub_params)
        if not (name_from_filter_load == ''): return name_from_filter_load
        # Couldn't find a match: try to find an unfiltered version:
        stored_filters = sub_params.pop('filters')
        sub_params['filters'] = {}
        name_from_filterless_load = self._normal_load_helper(datasets=datasets, sub_params=sub_params)
        if (name_from_filterless_load == ''): return name_from_filterless_load
        # Success - now apply filters:
        self.dataset['filters'] = copy.deepcopy(stored_filters)
        self.dataset = filter_dataset(self.dataset, _printer=lambda msg: self.print(msg, LogType.DEBUG))
        if save:
            self.save_dataset()
        return 'NEW GENERATION'
        
    def _normal_load_helper(self, datasets: dict, sub_params: dict):
        for name in datasets:
            if datasets[name]['params'] == sub_params:
                try:
                    self._load(name)
                    return name
                except:
                    self._fix(name)
        return ''
    
    def swap(self, dataset_params: dict, generate: bool = False, allow_false: bool = True) -> bool:
        '''
        Swap out the currently loaded dataset by searching for a param dictionary match into VPRDatasetProcessor.
        Attempts to simply extend the existing dataset if possible.

        Inputs:
        - dataset_params:   dict type with following keys (str type) and values (detailed below):
                            - npz_dbp:        str type; directory relative to root to find compressed data sets
                            - bag_dbp:        str type; directory relative to root to find rosbags
                            - bag_name:       str type; name of rosbag to find
                            - sample_rate:    float type; rate to sample messages in rosbag
                            - odom_topic:     str type; name of ROS topic to extract nav_msgs/Odometry messages from. If a list is provided, each will be stored as a column in the dataset
                            - img_topics:     list type; list of str types. names of ROS topics to extract sensor_msgs/CompressedImage messages from.
                            - img_dims:       list type; list of int types (two-positive-integer list). Dimensions to reduce input images to (height x width).
                            - ft_types:       list type; list of str types (names from FeatureType). Features to extract from each image; will become accessible keys in the dataset.
                            - filters:        str type; json string for designating filters to be applied. See pyaarapsi.vpr_simple.vpr_helpers.filter_dataset()
        - generate:         bool type {default: False}; whether or not to attempt generation if load fails
        - allow_false:      bool type {default: True}; if False, triggers an exception if the swap operation fails.
        Returns:
            bool type; True if successful, else False.
        '''
        # Check if we can just extend the current dataset:
        try:
            if not self.dataset:
                raise Exception('No dataset presently loaded.')
            self.print('[swap] Attempting to extend dataset...')
            dataset_params_test = copy.deepcopy(dataset_params)
            dataset_params_test['ft_types'] = copy.copy(self.dataset['params']['ft_types'])
            if dataset_params_test == self.dataset['params']:
                self.print("[swap] Candidate for extension. Feature types: %s [against: %s]" % (str(dataset_params['ft_types']), str(self.dataset['params']['ft_types'])), LogType.DEBUG)
                for ft_type in dataset_params['ft_types']:
                    if not (ft_type in self.dataset['params']['ft_types']):
                        return self.extend_dataset(ft_type, try_gen=generate, save=generate)
        except:
            self.print("[swap] Dataset corrupted; forcing a fresh dataset load. Error details: " + formatException(), LogType.DEBUG)
        self.print('[swap] Extension cancelled, attempting to load dataset...')
        # Can't extend, let's load in the replacement:
        if self.load_dataset(dataset_params, try_gen=generate):
            return True
        if not allow_false:
            raise Exception('Dataset failed to load.')
        return False
    
    def destroy(self) -> None:
        '''
        Class destructor

        Inputs:
        - None
        Returns:
            None
        '''
        if self.init_hybridnet and (not self.borrowed_nns['hybridnet']):
            try:
                if not self.hybridnet is None:
                    self.hybridnet.destroy()
                del self.hybridnet
            except:
                pass
        if self.init_netvlad and (not self.borrowed_nns['netvlad']):
            try:
                if not self.netvlad is None:
                    self.netvlad.destroy()
                del self.netvlad
            except:
                pass
        if self.init_salad and (not self.borrowed_nns['salad']):
            try:
                if not self.salad is None:
                    self.salad.destroy()
                del self.salad
            except:
                pass
        if self.init_apgem and (not self.borrowed_nns['apgem']):
            try:
                if not self.apgem is None:
                    self.apgem.destroy()
                del self.apgem
            except:
                pass
        try:
            del self.dataset
        except:
            pass
        try:
            del self.dataset_params
        except:
            pass

        del self.init_netvlad
        del self.init_hybridnet
        del self.init_salad
        del self.init_apgem
        del self.npz_dbp
        del self.bag_dbp
        del self.root
        del self.borrowed_nns
        del self.cuda
        del self.ros
        del self.use_tqdm
        del self.autosave
        del self.printer
        
        try:
            del self
        except:
            pass

    #### Private methods:
    def _check(self, params: Optional[dict] = None) -> str:
        '''
        Helper function to check if params already exist in saved npz_dbp library

        Inputs:
        - params: dict type {default: None}; parameter dictionary to compare against in search (if None, uses loaded dataset parameter dictionary)
        Returns:
            str type; '' if no matching parameters, otherwise file name of matching parameters (from npz_dbp/params)
        '''
        datasets = self._get_datasets()
        if params is None:
            if not self.dataset:
                return ""
            params = self.dataset['params']
        for name in datasets:
            if datasets[name]['params'] == params:
                return name
        return ""
    
    def _get_datasets(self) -> dict:
        '''
        Helper function to iterate over parameter dictionaries and compile a list

        Inputs:
        - None
        Returns:
            dict type; keys correspond to file names in npz_dbp/params, values are file contents (loaded parameter dictionaries)
        '''
        datasets = {}
        try:
            entry_list = os.scandir(self.root + '/' + self.npz_dbp + "/params/")
        except FileNotFoundError:
            raise Exception("Directory invalid.")
        for entry in entry_list:
            if entry.is_file() and entry.name.startswith('dataset'):
                raw_npz = dict(np.load(entry.path, allow_pickle=True))
                datasets[os.path.splitext(entry.name)[0]] = dict(params=raw_npz['params'].item())
        return datasets
    
    def _fix(self, dataset_name: str) -> list:
        '''
        Helper function to remove broken/erroneous files

        Inputs:
        - dataset_name: str type; name of file to search and remove (from npz_dbp)
        Returns:
            list type; contains two booleans: 1) whether a data set was purged, 2) whether a param file was purged 
        '''
        if not dataset_name.endswith('.npz'):
            dataset_name = dataset_name + '.npz'
        _purges = [False, False]
        self.print("[_fix] Bad dataset state detected, performing cleanup...", LogType.DEBUG)
        try:
            os.remove(self.root + '/' + self.npz_dbp + '/' + dataset_name)
            self.print("[_fix] Purged: %s" % (self.npz_dbp + '/' + dataset_name), LogType.DEBUG)
            _purges[0] = True
        except:
            pass
        try:
            os.remove(self.root + '/' + self.npz_dbp + '/params/' + dataset_name)
            self.print("[_fix] Purged: %s" % (self.npz_dbp + '/params/' + dataset_name), LogType.DEBUG)
            _purges[1] = True
        except:
            pass
        return _purges

    def _load(self, dataset_name: str, store: bool = True) -> dict:
        '''
        Helper function to load a dataset from a file name

        Inputs:
        - dataset_name: str type; name of file to load (from npz_dbp directory)
        - store         bool type {default: True}; whether or not to store in VPRDatasetProcessor
        Returns:
            dict type; loaded dataset dictionary
        '''
        if not dataset_name.endswith('.npz'):
            dataset_name = dataset_name + '.npz'
        full_file_path  = self.root + '/' + self.npz_dbp + '/' + dataset_name

        self.print('[_load] Attempting to load: %s' % full_file_path, LogType.DEBUG)
        raw_dataset     = np.load(full_file_path, allow_pickle=True)
        # when loading objects inside dicts from .npz files, must extract with .item() each object:
        dataset         = dict(dataset=raw_dataset['dataset'].item(), params=raw_dataset['params'].item())
        if store:
            if hasattr(self, 'dataset'):
                del self.dataset
            self.dataset = copy.deepcopy(dataset)
        return dataset