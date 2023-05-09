#!/usr/bin/env python3

import numpy as np
import copy
import rospkg
import os
import datetime
from pathlib import Path
from ..core.enum_tools import enum_name, enum_get
from ..core.ros_tools import process_bag, LogType, roslogger
from ..core.helper_tools import formatException
from ..core.file_system_tools import scan_directory
from ..vpr_classes import NetVLAD_Container, HybridNet_Container
from .vpr_helpers import *

class VPRDatasetProcessor: # main ROS class
    def __init__(self, dataset_params: dict, try_gen=True, init_netvlad=False, init_hybridnet=False, cuda=False, use_tqdm=False, autosave=False, ros=True):
        '''
        Initialisation

        Inputs:
            dataset_params: dict type with following keys:
                                - bag_name      [string]
                                - npz_dbp       [string]
                                - bag_dbp       [string]
                                - odom_topic    [string]
                                - img_topics    [string list]
                                - sample_rate   [float]
                                - ft_types      [string list]
                                - img_dims      [int list]
                                - filters       [string]
            try_gen:        bool type {default: True}; whether or not to attempt generation if load fails
            init_netvlad:   bool type {default: False}; whether or not to initialise netvlad (loads model into GPU)
            init_hybridnet: bool type {default: False}; whether or not to initialise hybridnet (loads model into GPU)
            cuda:           bool type {default: False}; whether or not to use CUDA for feature type/GPU acceleration
            use_tqdm:       bool type {default: False}; whether or not to display extraction/loading statuses using tqdm
            autosave:       bool type {default: False}; whether or not to automatically save any generated datasets
            printer:        method type {default: print}; what function to use to display logging messages
        Returns:
            self
        '''
        self.dataset_ready  = False
        self.cuda           = cuda
        self.use_tqdm       = use_tqdm
        self.autosave       = autosave
        self.init_netvlad   = init_netvlad
        self.init_hybridnet = init_hybridnet

        self.ros            = ros
        self.netvlad        = None
        self.hybridnet      = None

        if self.init_netvlad: # If needed, initialise NetVLAD
            self.netvlad    = NetVLAD_Container(cuda=self.cuda, ngpus=int(self.cuda), logger=self.print)

        if self.init_hybridnet: # If needed, initialise HybridNet
            self.hybridnet  = HybridNet_Container(cuda=self.cuda, logger=self.print)

        if not (dataset_params is None): # If parameters have been provided:
            self.print("[VPRDatasetProcessor] Loading model from parameters...")
            self.load_dataset(dataset_params, try_gen=try_gen)

            if not self.dataset_ready:
                raise Exception("Dataset load failed.")
            self.print("[VPRDatasetProcessor] Dataset Ready.")

        else: # None-case needed for SVM training
            self.print("[VPRDatasetProcessor] Ready; no dataset loaded.")

    def print(self, text, logtype=LogType.INFO, throttle=0):
        roslogger(text, logtype, throttle=throttle, ros=self.ros)

    def prep_netvlad(self, cuda=None, load=True, prep=True):
        '''
        Prepare netvlad for use

        Inputs:
            cuda:   bool type {default: None}; if not None, overrides initialisation cuda variable for netvlad loading
            load:   bool type {default: True}; whether or not to automatically trigger netvlad.load()
            prep:   bool type {default: True}; whether or not to automatically trigger netvlad.prep()
        Returns:
            None
        '''
        if not (cuda is None):
            cuda = self.cuda
        self.netvlad        = NetVLAD_Container(cuda=cuda, ngpus=int(self.cuda), logger=lambda x: self.print(x, LogType.DEBUG), load=load, prep=prep)
        self.init_netvlad   = True

    def prep_hybridnet(self, cuda=None, load=True):
        '''
        Prepare hybridnet for use

        Inputs:
            cuda:   bool type {default: None}; if not None, overrides initialisation cuda variable for hybridnet loading
            load:   bool type {default: True}; whether or not to automatically trigger hybridnet.load()
        Returns:
            None
        '''
        if not (cuda is None):
            cuda = self.cuda
        self.hybridnet  = HybridNet_Container(cuda=cuda, logger=lambda x: self.print(x, LogType.DEBUG), load=load)
        self.init_hybridnet   = True

    def pass_nns(self, processor, netvlad=True, hybridnet=True):
        '''
        Overwrite this VPRDatasetProcessor's instances of netvlad and hybridnet with another's instantiations
        Will check and, if initialised, destroy existing netvlad/hybridnet

        Inputs:
            processor:  VPRDatasetProcessor type
            netvlad:    bool type {default: True}; whether or not to overwrite netvlad instance
            hybridnet:  bool type {default: True}; whether or not to overwrite hybridnet instance
        Returns:
            None
        '''
        if (not self.netvlad is None) and netvlad:
            if isinstance(self.netvlad, NetVLAD_Container):
                self.netvlad.destroy()
            self.netvlad = processor.netvlad
            self.init_netvlad = True
        if (not self.hybridnet is None) and hybridnet:
            if isinstance(self.hybridnet, HybridNet_Container):
                self.hybridnet.destroy()
            self.hybridnet = processor.hybridnet
            self.init_hybridnet = True

    def generate_dataset(self, npz_dbp, bag_dbp, bag_name, sample_rate, odom_topic, img_topics, img_dims, ft_types, filters={}, store=True):
        '''
        Generate new datasets from parameters
        Inputs are specified such that, other than store, a correct dataset parameters dictionary can be dereferenced to autofill inputs

        Inputs:
            npz_dbp:        string type
            bag_dbp:        string type
            bag_name:       string type
            sample_rate:    float type
            odom_topic:     string type
            img_topics:     string list type
            img_dims:       int list type
            ft_types:       string list type
            filters:        string type
            store:          bool type; whether or not to overwrite this VPRDatasetProcessor's currently loaded dataset
        Returns:
            Generated dataset dictionary
        '''

        # store for access in saving operation:
        if store:
            self.dataset_ready  = False
            self.npz_dbp        = npz_dbp
            self.bag_dbp        = bag_dbp

        # generate:
        rosbag_dict         = process_bag(rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) +  '/' + bag_dbp + '/' + bag_name, sample_rate, odom_topic, img_topics, printer=self.print, use_tqdm=self.use_tqdm)
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
        dataset             = dict(params=params_dict, dataset=dataset_dict)

        if store:
            if hasattr(self, 'dataset'):
                del self.dataset
            self.dataset        = copy.deepcopy(dataset)
            self.dataset_ready  = True

            if self.autosave:
                self.save_dataset()

        return dataset

    def save_dataset(self, name=None):
        '''
        
        '''
        if not self.dataset_ready:
            raise Exception("Dataset not loaded in system. Either call 'generate_dataset' or 'load_dataset' before using this method.")
        dir = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/' + self.npz_dbp
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

            if self._check(params=sub_params):
                self.print("[save_dataset] File exists with identical parameters, skipping.")
                continue
            
            np.savez(full_file_path, **sub_dataset)
            np.savez(full_param_path, params=sub_dataset['params']) # save whole dictionary to preserve key object types
            self.print("[save_dataset] Save complete.\n\tfile: %s\n\tparams:%s." % (full_file_path, full_param_path))
            del sub_dataset
        return self

    def extend_dataset(self, new_ft_type, try_gen=False, save=False):
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

    def open_dataset(self, dataset_params, try_gen=False):
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

    def load_dataset(self, dataset_params, try_gen=False):
    # load via search for param match
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
                        return False
            return True
        else: 
            if try_gen:
                self.generate_dataset(**dataset_params)
                return True
            return False
    
    def swap(self, dataset_params, generate=False, allow_false=True):
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
        
    def getFeat(self, imgs, fttype_in, dims, use_tqdm=False):
    # Get features from imgs, using VPRImageProcessor's set image dimensions.
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
        if any([fttype == FeatureType.HYBRIDNET for fttype in fttypes]) and not self.init_hybridnet:
            raise Exception("[getFeat] FeatureType.HYBRIDNET provided but VPRImageProcessor not initialised with init_hybridnet=True")
        if any([fttype == FeatureType.NETVLAD for fttype in fttypes]) and not self.init_netvlad:
            raise Exception("[getFeat] FeatureType.NETVLAD provided but VPRImageProcessor not initialised with init_netvlad=True")
        try:
            return getFeat(imgs, fttypes, dims, use_tqdm=use_tqdm, nn_hybrid=self.hybridnet, nn_netvlad=self.netvlad)
        except Exception as e:
            raise Exception("[getFeat] Feature vector could not be constructed.\nCode: %s" % (e))
        
    #### Private methods:
    def _check(self, params=None):
        datasets = self._get_datasets()
        if params is None:
            params = self.dataset['params']
        for name in datasets:
            if datasets[name]['params'] == params:
                return name
        return ""
    
    def _get_datasets(self):
        datasets = {}
        try:
            entry_list = os.scandir(rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/' + self.npz_dbp + "/params/")
        except FileNotFoundError:
            raise Exception("Directory invalid.")
        for entry in entry_list:
            if entry.is_file() and entry.name.startswith('dataset'):
                raw_npz = dict(np.load(entry.path, allow_pickle=True))
                datasets[os.path.splitext(entry.name)[0]] = dict(params=raw_npz['params'].item())
        return datasets
    
    def _fix(self, dataset_name):
        if not dataset_name.endswith('.npz'):
            dataset_name = dataset_name + '.npz'
        self.print("[_fix] Bad dataset state detected, performing cleanup...", LogType.DEBUG)
        try:
            os.remove(rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/' + self.npz_dbp + '/' + dataset_name)
            self.print("[_fix] Purged: %s" % (self.npz_dbp + '/' + dataset_name), LogType.DEBUG)
        except:
            pass
        try:
            os.remove(rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/' + self.npz_dbp + '/params/' + dataset_name)
            self.print("[_fix] Purged: %s" % (self.npz_dbp + '/params/' + dataset_name), LogType.DEBUG)
        except:
            pass

    def _load(self, dataset_name, store=True):
    # when loading objects inside dicts from .npz files, must extract with .item() each object
        if not dataset_name.endswith('.npz'):
            dataset_name = dataset_name + '.npz'
        full_file_path  = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/' + self.npz_dbp + '/' + dataset_name

        self.print('[_load] Attempting to load: %s' % full_file_path, LogType.DEBUG)
        raw_dataset     = np.load(full_file_path, allow_pickle=True)
        dataset         = dict(dataset=raw_dataset['dataset'].item(), params=raw_dataset['params'].item())
        if store:
            if hasattr(self, 'dataset'):
                del self.dataset
            self.dataset = copy.deepcopy(dataset)
            self.dataset_ready = True
        return dataset
    
    def destroy(self):
        if self.init_hybridnet:
            self.hybridnet.destroy()
            del self.hybridnet
        if self.init_netvlad:
            self.netvlad.destroy()
            del self.netvlad
        del self.dataset_ready
        del self.cuda
        del self.use_tqdm
        del self.autosave
        del self.init_netvlad
        del self.init_hybridnet
        del self.print
        del self.dataset
