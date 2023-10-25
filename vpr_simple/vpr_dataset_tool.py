#!/usr/bin/env python3

import numpy as np
import copy
import logging
import os
try:
    import rospkg
    ROSPKG_ROOT = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__)))
except:
    logging.warn('Could not access rospkg; ensure you specify root if using this tool. This is typically due to a missing or incorrect ROS installation.')
import datetime
from pathlib import Path
from ..core.enum_tools import enum_name, enum_get
try:
    from ..core.ros_tools import process_bag
except:
    logging.warn('Could not access ros_tools; generating features from rosbags will fail. This is typically due to a missing or incorrect ROS installation.')
from ..core.roslogger import LogType, roslogger
from ..core.helper_tools import formatException, vis_dict
from ..core.file_system_tools import scan_directory
from .vpr_helpers import *
from ..vpr_classes.netvlad import NetVLAD_Container
from ..vpr_classes.hybridnet import HybridNet_Container

class VPRDatasetProcessor:
    def __init__(self, dataset_params: dict, try_gen: bool = True, init_netvlad: bool = False, 
                 init_hybridnet: bool = False, cuda: bool = False, use_tqdm: bool = False, 
                 autosave: bool = False, ros: bool = True, root: str = None, printer=None):
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
        - init_netvlad:     bool type {default: False}; whether or not to initialise netvlad (loads model into GPU)
        - init_hybridnet:   bool type {default: False}; whether or not to initialise hybridnet (loads model into GPU)
        - cuda:             bool type {default: False}; whether or not to use CUDA for feature type/GPU acceleration
        - use_tqdm:         bool type {default: False}; whether or not to display extraction/loading statuses using tqdm
        - autosave:         bool type {default: False}; whether or not to automatically save any generated datasets
        - ros:              bool type {default: True}; whether or not to use rospy logging (requires operation within ROS node scope)
        - root:             str type {default: None}; base root inserted in front of npz_dbp, bag_dbp, and svm_dbp
        - printer:          method {default: None}; if provided, overrides logging and will pass inputs to specified method on print
        Returns:
            self
        '''
        self.dataset_ready  = False
        self.cuda           = cuda
        self.use_tqdm       = use_tqdm
        self.autosave       = autosave
        self.init_netvlad   = init_netvlad
        self.init_hybridnet = init_hybridnet
        self.printer        = printer
        self.borrowed_nns   = [False, False]

        if root is None:
            self.root       = ROSPKG_ROOT
        else:
            self.root       = root

        self.ros            = ros
        self.netvlad        = None
        self.hybridnet      = None

        self.netvlad_rdy    = False
        self.hybridnet_rdy  = False

        if not (dataset_params is None): # If parameters have been provided:
            self.print("Loading model from parameters...")
            name = self.load_dataset(dataset_params, try_gen=try_gen)

            if not self.dataset_ready:
                raise Exception("Dataset load failed.")
            self.print("Dataset Ready (loaded: %s)." % str(name))

        else: # None-case needed for SVM training
            self.print("Ready; no dataset loaded.")

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
        return self.dataset['dataset']

    def get_params(self) -> dict:
        '''
        Get dataset parameters

        Inputs:
        - None
        Returns:
            dict type; dataset parameter dictionary
        '''
        return self.dataset['params']
    
    def show(self, printer = print) -> dict:
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

    def prep_netvlad(self, cuda: bool = None, load: bool = True, prep: bool = True) -> bool:
        '''
        Prepare netvlad for use

        Inputs:
        - cuda:   bool type {default: None}; if not None, overrides initialisation cuda variable for netvlad loading
        - load:   bool type {default: True}; whether or not to automatically trigger netvlad.load()
        - prep:   bool type {default: True}; whether or not to automatically trigger netvlad.prep()
        Returns:
            bool type; True (on success, else Exception)
        '''
        if not (cuda is None):
            cuda = self.cuda
        self.netvlad        = NetVLAD_Container(cuda=cuda, ngpus=int(self.cuda), logger=self.print, load=load, prep=prep)
        self.init_netvlad   = True
        return True

    def prep_hybridnet(self, cuda: bool = None, load: bool = True) -> bool:
        '''
        Prepare hybridnet for use

        Inputs:
        - cuda:   bool type {default: None}; if not None, overrides initialisation cuda variable for hybridnet loading
        - load:   bool type {default: True}; whether or not to automatically trigger hybridnet.load()
        Returns:
            bool type; True (on success, else Exception)
        '''
        if not (cuda is None):
            cuda = self.cuda
        self.hybridnet  = HybridNet_Container(cuda=cuda, logger=self.print, load=load)
        self.init_hybridnet   = True
        return True

    def pass_nns(self, processor: object, netvlad: bool = True, hybridnet: bool = True) -> bool:
        '''
        Overwrite this VPRDatasetProcessor's instances of netvlad and hybridnet with another's instantiations
        Will check and, if initialised, destroy existing netvlad/hybridnet

        Inputs:
            processor:  VPRDatasetProcessor type
            netvlad:    bool type {default: True}; whether or not to overwrite netvlad instance
            hybridnet:  bool type {default: True}; whether or not to overwrite hybridnet instance
        Returns:
            bool type; True (on success, else Exception)
        '''
        if netvlad:
            if isinstance(self.netvlad, NetVLAD_Container):
                self.netvlad.destroy()
            self.netvlad = processor.netvlad
            self.init_netvlad = True
            self.borrowed_nns[0] = True
        if hybridnet:
            if isinstance(self.hybridnet, HybridNet_Container):
                self.hybridnet.destroy()
            self.hybridnet = processor.hybridnet
            self.init_hybridnet = True
            self.borrowed_nns[1] = True
        return True

    def check_netvlad(self, ft_types: list) -> bool:
        '''
        Check if NetVLAD is initialised: if not initialised but needed (ft_types contains FeatureType.NETVLAD) will attempt to initialise.
        Delays loading of NetVLAD model until required by system.

        Inputs:
            ft_types:  list type; list of FeatureType enumerations. If it contains FeatureType.NETVLAD and NetVLAD is not loaded, this method will attempt to initialise NetVLAD
        Returns:
            bool type; True (on success, else Exception)
        '''
        if (FeatureType.NETVLAD in ft_types) and (self.netvlad_rdy == False) and self.init_netvlad: # If needed, initialise NetVLAD
            self.netvlad        = NetVLAD_Container(cuda=self.cuda, ngpus=int(self.cuda), logger=self.print)
            self.netvlad_rdy    = True
            return True
        return False
    
    def check_hybridnet(self, ft_types: list) -> bool:
        '''
        Check if HybridNet is initialised: if not initialised but needed (ft_types contains FeatureType.HYBRIDNET) will attempt to initialise.
        Delays loading of HybridNet model until required by system.

        Inputs:
            ft_types:  list type; list of FeatureType enumerations. If it contains FeatureType.HYBRIDNET and HybridNet is not loaded, this method will attempt to initialise HybridNet
        Returns:
            bool type; True (on success, else Exception)
        '''
        if (FeatureType.HYBRIDNET in ft_types) and (self.hybridnet_rdy == False) and self.init_hybridnet: # If needed, initialise HybridNet
            self.hybridnet      = HybridNet_Container(cuda=self.cuda, logger=self.print)
            self.hybridnet_rdy  = True
            return True
        return False

    def generate_dataset(self, npz_dbp: str, bag_dbp: str, bag_name: str, sample_rate: float, 
                         odom_topic: str, img_topics: list, img_dims: list, ft_types: list, 
                         filters: str = '{}', store: bool = True) -> dict:
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

        self.check_netvlad(ft_types)
        self.check_hybridnet(ft_types)

        # store for access in saving operation:
        if store:
            self.dataset_ready  = False
            self.npz_dbp        = npz_dbp
            self.bag_dbp        = bag_dbp

        # generate:
        rosbag_dict         = process_bag(self.root +  '/' + bag_dbp + '/' + bag_name, sample_rate, odom_topic, img_topics, printer=self.print, use_tqdm=self.use_tqdm)
        self.print('[generate_dataset] Performing feature extraction...')
        feature_vector_dict = {ft_type: self.getFeat(list(rosbag_dict[img_topics[0]]), enum_get(ft_type, FeatureType), img_dims, use_tqdm=self.use_tqdm) for ft_type in ft_types}
        self.print('[generate_dataset] Done.')

        # Create dataset dictionary and add feature vectors
        params_dict         = dict( bag_name=bag_name, npz_dbp=npz_dbp, bag_dbp=bag_dbp, odom_topic=odom_topic, img_topics=img_topics, \
                                    sample_rate=sample_rate, ft_types=ft_types, img_dims=img_dims, filters=filters)
        dataset_dict        = dict( time=rosbag_dict['t'], \
                                    px=rosbag_dict['px'], py=rosbag_dict['py'], pw=rosbag_dict['pw'], \
                                    vx=rosbag_dict['vx'], vy=rosbag_dict['vy'], vw=rosbag_dict['vw'] )
        dataset_dict.update(feature_vector_dict)
        dataset_raw         = dict(params=params_dict, dataset=dataset_dict)
        dataset             = filter_dataset(dataset_raw)

        if store:
            if hasattr(self, 'dataset'):
                del self.dataset
            self.dataset        = copy.deepcopy(dataset)
            self.dataset_ready  = True

            if self.autosave:
                self.save_dataset()

        return dataset

    def save_dataset(self, name: str = None) -> object:
        '''
        Save loaded dataset to file system

        Inputs:
        - Name: str type {default: None}; if None, a unique name will be generated of the format dataset_%Y%m%d_X.
        Returns:
            self
        '''
        if not self.dataset_ready:
            raise Exception("Dataset not loaded in system. Either call 'generate_dataset' or 'load_dataset' before using this method.")
        dir = self.root + '/' + self.npz_dbp
        Path(dir).mkdir(parents=False, exist_ok=True)
        Path(dir+"/params").mkdir(parents=False, exist_ok=True)
        
        # Ensure file name is of correct format, generate if not provided
        file_list, _, _, = scan_directory(dir, short_files=True)
        if (not name is None):
            if not (name.startswith('dataset')):
                name = "dataset_" + name
            if (name in file_list):
                raise Exception("Dataset with name %s already exists in directory." % name)
        else:
            name = datetime.datetime.today().strftime("dataset_%Y%m%d")

        self.print("[save_dataset] Splitting dataset into files for feature types: %s" % self.dataset['params']['ft_types'])
        for ft_type in self.dataset['params']['ft_types']:
            # Generate unique name:
            file_name = name
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

            file_ = self._check(params=sub_params)
            if file_:
                self.print("[save_dataset] File exists with identical parameters (%s); skipping save." % file_)
                continue
            
            np.savez(full_file_path, **sub_dataset)
            np.savez(full_param_path, params=sub_dataset['params']) # save whole dictionary to preserve key object types
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
        if isinstance(new_ft_type, FeatureType):
            new_ft_type = enum_name(new_ft_type)
        new_params = copy.deepcopy(self.dataset['params'])
        new_params['ft_types'] = [new_ft_type]
        self.print("[extend_dataset] Loading dataset to extend existing...", LogType.DEBUG)
        new_dataset = self.open_dataset(new_params, try_gen=try_gen)
        if new_dataset is None:
            self.print("[extend_dataset] Load failed.", LogType.DEBUG)
            return False
        self.print("[extend_dataset] Load success. Extending.", LogType.DEBUG)
        self.dataset['dataset'][new_ft_type] = copy.deepcopy(new_dataset['dataset'][new_ft_type])
        self.dataset['params']['ft_types'].append(new_ft_type)
        if save:
            self.save_dataset()
        return True

    def open_dataset(self, dataset_params: dict, try_gen: bool = False) -> dict:
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
        assert len(dataset_params['ft_types']) == 1, 'Can only open one dataset'
        assert self.npz_dbp == dataset_params['npz_dbp'], 'Must preserve npz_dbp'
        assert self.bag_dbp == dataset_params['bag_dbp'], 'Must preserve bag_dbp'

        self.print("[open_dataset] Loading dataset.", LogType.DEBUG)
        datasets = self._get_datasets()
        for name in datasets:
            if datasets[name]['params'] == dataset_params:
                try:
                    self.print("[open_dataset] Loading ...", LogType.DEBUG)
                    return self._load(name, store=False)
                except:
                    self.print(formatException())
        if try_gen:
            try:
                self.print("[open_dataset] Generating ...", LogType.DEBUG)
                return self.generate_dataset(**dataset_params, store=False)
            except:
                self.print(formatException())
        
        self.print("[open_dataset] Load, generation failed.", LogType.DEBUG)
        return None

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
        self.npz_dbp = dataset_params['npz_dbp']
        self.bag_dbp = dataset_params['bag_dbp']
        self.dataset_ready = False
        self.print("[load_dataset] Loading dataset.")
        datasets = self._get_datasets()
        sub_params = copy.deepcopy(dataset_params)
        sub_params['ft_types'] = [dataset_params['ft_types'][0]]
        for name in datasets:
            if datasets[name]['params'] == sub_params:
                try:
                    self._load(name)
                    break
                except:
                    self._fix(name)
        if self.dataset_ready:
            if len(dataset_params['ft_types']) > 1:
                for ft_type in dataset_params['ft_types'][1:]:
                    if not self.extend_dataset(ft_type, try_gen=try_gen, save=try_gen):
                        return ''
            return name
        else: 
            if try_gen:
                self.print('Generating dataset with params: %s' % (str(dataset_params)), LogType.DEBUG)
                self.generate_dataset(**dataset_params)
                return 'NEW GENERATION'
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
        
    def getFeat(self, img, fttype_in: FeatureType, dims: list, use_tqdm: bool = False) -> np.ndarray:
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
    # Get features from img, using VPRImageProcessor's set image dimensions.
    # Specify type via fttype_in= (from FeatureType enum; list of FeatureType elements is also handled)
    # Specify dimensions with dims= (two-element positive integer tuple)
    # Returns feature array, as a flattened array

        if not (dims[0] > 0 and dims[1] > 0):
            raise Exception("[getFeat] image dimensions are invalid")
        if not isinstance(fttype_in, list):
            fttypes = [fttype_in]
        else:
            fttypes = fttype_in
        if not all([isinstance(fttype, FeatureType) for fttype in fttypes]):
            raise Exception("[getFeat] fttype_in provided contains elements that are not of type FeatureType")
        if any([fttype == FeatureType.HYBRIDNET for fttype in fttypes]):
            if self.init_hybridnet:
                if not self.hybridnet_rdy:
                    self.check_hybridnet(fttypes)
            else:
                raise Exception("[getFeat] FeatureType.HYBRIDNET provided but VPRImageProcessor not initialised with init_hybridnet=True")
        if any([fttype == FeatureType.NETVLAD for fttype in fttypes]):
            if self.init_netvlad:
                if not self.netvlad_rdy:
                    self.check_netvlad(fttypes)
            else:
                raise Exception("[getFeat] FeatureType.NETVLAD provided but VPRImageProcessor not initialised with init_netvlad=True")
        try:
            feats = getFeat(img, fttypes, dims, use_tqdm=use_tqdm, nn_hybrid=self.hybridnet, nn_netvlad=self.netvlad)
            if isinstance(feats, list):
                return [np.array(i, dtype=np.float32) for i in feats]
            return np.array(feats, dtype=np.float32)
        except Exception as e:
            raise Exception("[getFeat] Feature vector could not be constructed.\nCode: %s" % (e))
    
    def destroy(self) -> None:
        '''
        Class destructor

        Inputs:
        - None
        Returns:
            None
        '''
        if self.init_hybridnet and (not self.borrowed_nns[1]):
            try:
                self.hybridnet.destroy()
                del self.hybridnet
            except:
                pass
        if self.init_netvlad and (not self.borrowed_nns[0]):
            try:
                self.netvlad.destroy()
                del self.netvlad
            except:
                pass
        del self.dataset_ready
        del self.cuda
        del self.use_tqdm
        del self.autosave
        del self.init_netvlad
        del self.init_hybridnet
        try:
            del self.dataset
        except:
            pass
        try:
            del self
        except:
            pass

    #### Private methods:
    def _check(self, params: dict = None) -> str:
        '''
        Helper function to check if params already exist in saved npz_dbp library

        Inputs:
        - params: dict type {default: None}; parameter dictionary to compare against in search (if None, uses loaded dataset parameter dictionary)
        Returns:
            str type; '' if no matching parameters, otherwise file name of matching parameters (from npz_dbp/params)
        '''
        datasets = self._get_datasets()
        if params is None:
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
            self.dataset_ready = True
        return dataset