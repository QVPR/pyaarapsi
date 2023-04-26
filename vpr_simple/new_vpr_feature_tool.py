#!/usr/bin/env python3

import numpy as np
import copy
import os
import cv2
import sys
import csv
import datetime
from enum import Enum
from tqdm import tqdm
from pathlib import Path
from ..core.enum_tools import enum_name
from ..core.ros_tools import LogType, roslogger, process_bag
from ..vpr_classes import NetVLAD_Container, HybridNet_Container
from ..core.helper_tools import formatException
from ..core.file_system_tools import scan_directory
from .imageprocessor_helpers import *

class VPRImageProcessor: # main ROS class
    def __init__(self, bag_dbp=None, npz_dbp=None, dataset=None, try_gen=True, ros=False, init_netvlad=False, init_hybridnet=False, cuda=False, use_tqdm=False, autosave=False):
        self.npz_dbp        = npz_dbp
        self.bag_dbp        = bag_dbp
        self.dataset_ready  = False
        self.ros            = ros
        self.cuda           = cuda
        self.use_tqdm       = use_tqdm
        self.autosave       = autosave
        self.init_netvlad   = init_netvlad
        self.init_hybridnet = init_hybridnet

        if self.init_netvlad:
            self.netvlad    = NetVLAD_Container(cuda=self.cuda, ngpus=int(self.cuda), logger=lambda x: self.print(x, LogType.DEBUG))

        if self.init_hybridnet:
            self.hybridnet  = HybridNet_Container(cuda=self.cuda, logger=lambda x: self.print(x, LogType.DEBUG))

        if (not dataset is None) and (not self.npz_dbp is None):
            if isinstance(dataset, str):
                self.load_dataset(dataset)
            elif isinstance(dataset, dict):
                self.load_dataset_params(dataset)
                if try_gen and (not self.dataset_ready):
                    self.generate_dataset(**dataset)
            else:
                raise Exception("Dataset type not supported. Valid types: str, dict")
            if not self.dataset_ready:
                raise Exception("Dataset load failed.")
            self.print("[VPRImageProcessor] Dataset Ready.", LogType.INFO)
        else:
            self.print("[VPRImageProcessor] Initialised, no dataset loaded.", LogType.INFO)

    def generate_dataset(self, bag_name, npz_dbp, bag_dbp, sample_rate, odom_topic, img_topics, img_dims, ft_types, filters={}, install=True):
        # store for access in saving operation:
        self.dataset_ready  = False
        self.npz_dbp        = npz_dbp
        self.bag_dbp        = bag_dbp

        # generate:
        rosbag_dict         = process_bag(bag_dbp + '/' + bag_name, sample_rate, odom_topic, img_topics, printer=lambda x: roslogger(x, LogType.INFO, ros=self.ros), use_tqdm=self.use_tqdm)
        roslogger('Performing feature extraction...', LogType.INFO, ros=self.ros)
        feature_vector_dict = {enum_name(ft_type): self.getFeat(list(rosbag_dict[img_topics[0]]), ft_type, img_dims, use_tqdm=self.use_tqdm) for ft_type in ft_types}
        roslogger('Done.', LogType.INFO, ros=self.ros)
        params_dict         = dict( bag_name=bag_name, npz_dbp=npz_dbp, bag_dbp=bag_dbp, \
                                    odom_topic=odom_topic, img_topics=img_topics, \
                                    sample_rate=sample_rate, ft_types=ft_types, img_dims=img_dims, filters=filters)
        
        dataset_dict        = { 'time': rosbag_dict['t'], \
                                'odom': dict( position=dict( x=rosbag_dict['px'], y=rosbag_dict['py'], yaw=rosbag_dict['pw'] ),
                                              velocity=dict( x=rosbag_dict['vx'], y=rosbag_dict['vy'], yaw=rosbag_dict['vw'] ))}
        dataset_dict.update(feature_vector_dict)
        del self.dataset    
        dataset             = dict(params=params_dict, dataset=dataset_dict)

        if install:
            self.dataset        = dataset
            self.dataset_ready  = True

        if self.autosave:
            self.save_dataset(check_exists=True)

        return dataset

    def save_dataset(self, dir=None, name=None, check_exists=False):
        if dir is None:
            dir = self.npz_dbp
        if not self.dataset_ready:
            raise Exception("Dataset not loaded in system. Either call 'generate_dataset' or 'load_dataset(_params)' before using this method.")
        Path(dir).mkdir(parents=False, exist_ok=True)
        Path(dir+"/params").mkdir(parents=False, exist_ok=True)
        if check_exists:
            existing_file = self._check(dir)
            if existing_file:
                self.print("[save_dataset] File exists with identical parameters: %s", LogType.INFO)
                return self
        
        # Ensure file name is of correct format, generate if not provided
        file_list, _, _, = scan_directory(dir, short_files=True)
        if (not name is None):
            if not (name.startswith('dataset')):
                name = "dataset_" + name
            if (name in file_list):
                raise Exception("Dataset with name %s already exists in directory." % name)
        else:
            name = datetime.datetime.today().strftime("dataset_%Y%m%d")

        self.print("[save_dataset] Splitting dataset into files for feature types: %s" % enum_name(self.dataset['params']['ft_types']), LogType.INFO)
        for ft_type in self.dataset['params']['ft_types']:
            # Generate unique name:
            file_name = name
            count = 0
            while file_name in file_list:
                file_name = name + "_%d" % count
                count += 1
            file_list = file_list + [file_name]
            
            full_file_path = dir + "/" + file_name
            full_param_path = dir + "/params/" + file_name

            sub_data                = copy.deepcopy({key: self.dataset['dataset'][key] for key in ['time', 'odom', enum_name(ft_type)]})
            sub_params              = copy.deepcopy(self.dataset['params'])
            sub_params['ft_types']  = [ft_type]
            sub_dataset             = dict(params=sub_params, dataset=sub_data)
            np.savez(full_file_path, **sub_dataset)
            np.savez(full_param_path, params=sub_dataset['params']) # save whole dictionary to preserve key object types
            self.print("[save_dataset] Save complete.\n\tfile: %s\n\tparams:%s." % (full_file_path, full_param_path), LogType.INFO)
            del sub_dataset
        return self

    def extend_dataset(self, new_ft_type, save=False):
        new_params = self.dataset['params']
        new_params['ft_types'] = new_ft_type
        if not self.load_dataset_params(new_params):
            new_dataset = self.generate_dataset(**new_params, load=False)
        self.dataset[enum_name(new_ft_type)] = copy.deepcopy(new_dataset[enum_name(new_ft_type)])
        if save:
            self.save_dataset(check_exists=True)

    def load_dataset_by_parts(self, dataset_params, dir=None):
    # load via search for param match
        self.print("[load_dataset_by_parts] Loading dataset.", LogType.DEBUG)
        if dir is None:
            dir = self.npz_dbp
        datasets = self._get_datasets(dir)
        self.dataset = {}
        for name in datasets:
            if datasets[name]['params'] == dataset_params:
                try:
                    self._load(name)
                    return True
                except:
                    self._fix(name)
        return False

    def load_dataset_params(self, dataset_params, dir=None):
    # load via search for param match
        self.print("[load_dataset_params] Loading dataset.", LogType.DEBUG)
        if dir is None:
            dir = self.npz_dbp
        datasets = self._get_datasets(dir)
        self.dataset = {}
        for name in datasets:
            if datasets[name]['params'] == dataset_params:
                try:
                    self._load(name)
                    return True
                except:
                    self._fix(name)
        return False

    def load_dataset(self, dataset_name, dir=None):
    # load via string matching name of dataset file
        self.print("[load_dataset] Loading %s" % (dataset_name), LogType.DEBUG)
        if dir is None:
            dir = self.npz_dbp
        datasets = self._get_datasets(dir)
        self.dataset = {}
        for name in datasets:
            if name == dataset_name:
                try:
                    self._load(name)
                    True
                except:
                    self._fix(name)
        return False
    
    def swap(self, dataset_params, generate=False, allow_false=True):
        datasets = self._get_datasets(self.npz_dbp)
        for name in datasets:
            if datasets[name]['params'] == dataset_params:
                try:
                    self._load(name)
                    return True
                except:
                    self._fix(name)
        if generate:
            self.generate_dataset(**dataset_params)
            return True
        if not allow_false:
            raise Exception('Dataset failed to load.')
        return False

    def print(self, text, logtype):
        roslogger(text, logtype, self.ros)
        
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
        if any([fttype == FeatureType.NONE for fttype in fttypes]):
            raise Exception("[getFeat] fttype_in provided contains at least one FeatureType.NONE")
        if any([fttype == FeatureType.HYBRIDNET for fttype in fttypes]) and not self.init_hybridnet:
            raise Exception("[getFeat] FeatureType.HYBRIDNET provided but VPRImageProcessor not initialised with init_hybridnet=True")
        if any([fttype == FeatureType.NETVLAD for fttype in fttypes]) and not self.init_netvlad:
            raise Exception("[getFeat] FeatureType.NETVLAD provided but VPRImageProcessor not initialised with init_netvlad=True")
        try:
            return getFeat(imgs, fttypes, dims, use_tqdm=use_tqdm, nn_hybrid=self.hybridnet, nn_netvlad=self.netvlad)
        except Exception as e:
            raise Exception("[getFeat] Feature vector could not be constructed.\nCode: %s" % (e))
        
    #### Private methods:
    def _check(self, dir):
        datasets = self._get_datasets(dir)
        for name in datasets:
            if datasets[name]['params'] == self.dataset['params']:
                return name
        return ""
    
    def _get_datasets(self, dir=None):
        if dir is None:
            dir = self.npz_dbp
        datasets = {}
        try:
            entry_list = os.scandir(dir+"/params/")
        except FileNotFoundError:
            self.print("Error: directory invalid.", LogType.ERROR)
            return datasets
        for entry in entry_list:
            if entry.is_file() and entry.name.startswith('dataset'):
                loaded_dataset = dict(np.load(entry.path, allow_pickle=True))
                datasets[os.path.splitext(entry.name)[0]] = loaded_dataset
        return datasets
    
    def _fix(self, dataset_name):
        if not dataset_name.endswith('.npz'):
            dataset_name = dataset_name + '.npz'
        self.print("Bad dataset state detected, performing cleanup...", LogType.WARN)
        try:
            os.remove(self.npz_dbp + '/' + dataset_name)
            self.print("Purged: %s" % (self.npz_dbp + '/' + dataset_name), LogType.WARN)
        except:
            pass
        try:
            os.remove(self.npz_dbp + '/params/' + dataset_name)
            self.print("Purged: %s" % (self.npz_dbp + '/params/' + dataset_name), LogType.WARN)
        except:
            pass

    def _load(self, dataset_name, install=True):
    # when loading objects inside dicts from .npz files, must extract with .item() each object
        if not dataset_name.endswith('.npz'):
            dataset_name = dataset_name + '.npz'
        
        raw_dataset = np.load(self.npz_dbp + "/" + dataset_name, allow_pickle=True)
        if install:
            del self.dataset
            self.dataset = dict(dataset=raw_dataset['dataset'].item(), params=raw_dataset['params'].item())
            self.dataset_ready = True
            return self.dataset
        return raw_dataset
    
    def destroy(self):
        if self.init_hybridnet:
            self.hybridnet.destroy()
            del self.hybridnet
        if self.init_netvlad:
            self.netvlad.destroy()
            del self.netvlad
