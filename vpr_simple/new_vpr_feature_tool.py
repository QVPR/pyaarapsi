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
from ..core.ros_tools import LogType, roslogger
from ..vpr_classes import NetVLAD_Container, HybridNet_Container
from ..core.helper_tools import formatException
from ..core.file_system_tools import scan_directory
from .imageprocessor_helpers import *

class VPRImageProcessor: # main ROS class
    def __init__(self, data_dir, data=None, try_gen=True, ros=False, init_netvlad=False, init_hybridnet=False, cuda=False):
        self.data_dir       = data_dir
        self.data_ready     = False
        self.ros            = ros
        self.cuda           = cuda
        self.init_netvlad   = init_netvlad
        self.init_hybridnet = init_hybridnet

        if self.init_netvlad:
            self.netvlad    = NetVLAD_Container(cuda=self.cuda, ngpus=int(self.cuda), logger=lambda x: self.print(x, LogType.DEBUG))

        if self.init_hybridnet:
            self.hybridnet  = HybridNet_Container(cuda=self.cuda, logger=lambda x: self.print(x, LogType.DEBUG))

        if not (data is None):
            if isinstance(data, str):
                self.load_data(data)
            elif isinstance(data, dict):
                self.load_data_params(data)
                if try_gen and (not self.data_ready):
                    self.generate_data(**data)
            else:
                raise Exception("Data type not supported. Valid types: str, dict")
            if not self.data_ready:
                raise Exception("Data load failed.")
            self.print("[VPRImageProcessor] Data Ready.", LogType.INFO)

    def save_data(self, dir=None, name=None, check_exists=False):
        if dir is None:
            dir = self.data_dir
        if not self.data_ready:
            raise Exception("Data not loaded in system. Either call 'generate_data' or 'load_data(_params)' before using this method.")
        Path(dir).mkdir(parents=False, exist_ok=True)
        if check_exists:
            existing_file = self._check(dir)
            if existing_file:
                self.print("[save_data] File exists with identical parameters: %s", LogType.INFO)
                return self
        
        # Ensure file name is of correct format, generate if not provided
        file_list, _, _, = scan_directory(dir, short_files=True)
        if (not name is None):
            if not (name.startswith('dataset')):
                name = "dataset_" + name
            if (name in file_list):
                raise Exception("Data with name %s already exists in directory." % name)
        else:
            name = datetime.datetime.today().strftime("dataset_%Y%m%d")

        # Check file_name won't overwrite existing datasets
        file_name = name
        count = 0
        while file_name in file_list:
            file_name = name + "_%d" % count
            count += 1
        
        separator = ""
        if not (dir[-1] == "/"):
            separator = "/"
        full_file_path = dir + separator + file_name
        np.savez(full_file_path, **self.data)
        self.print("[save_data] Saved file to %s" % full_file_path, LogType.INFO)
        return self

    def load_data_params(self, data_params, dir=None):
    # load via search for param match
        self.print("[load_data_params] Loading dataset.", LogType.DEBUG)
        self.data_ready = False
        if dir is None:
            dir = self.data_dir
        datasets = self._get_datasets(dir)
        self.data = {}
        for name in datasets:
            if datasets[name]['params'] == data_params:
                self._load(datasets[name])
                break
        return self

    def load_data(self, dataset_name, dir=None):
    # load via string matching name of dataset file
        self.print("[load_data] Loading %s" % (dataset_name), LogType.DEBUG)
        self.data_ready = False
        if dir is None:
            dir = self.data_dir
        datasets = self._get_datasets(dir)
        self.data = {}
        for name in datasets:
            if name == dataset_name:
                self._load(datasets[name])
                break
        return self
    
    def swap(self, data_params, generate=False):
        datasets = self._get_datasets(self.data_dir)
        for name in datasets:
            if datasets[name]['params'] == data_params:
                self._load(datasets[name])
                return True
        if generate:
            self.generate_data(**data_params)
            return True
        return False

    def print(self, text, logtype):
        roslogger(text, logtype, self.ros)
        
    def getFeat(self, im, fttype_in, dims, use_tqdm=False):
    # Get features from im, using VPRImageProcessor's set image dimensions.
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
            getFeat(im, fttypes, dims, use_tqdm=use_tqdm, nn_hybrid=self.hybridnet, nn_netvlad=self.netvlad)
        except Exception as e:
            raise Exception("[getFeat] Feature vector could not be constructed.\nCode: %s" % (e))

    def generate_data(self, database_path, qry, ref, img_dims, folder, ft_type, save=True):
        # store for access in saving operation:
        self.database_path      = database_path
        self.cal_qry_dataset    = qry
        self.cal_ref_dataset    = ref 
        self.img_dims           = img_dims
        self.folder             = folder
        self.feat_type          = ft_type
        self.data_ready         = False

        # generate:
        self._load()

        params_dict        = dict(ref=self.cal_ref_dataset, qry=self.cal_qry_dataset, \
                                    img_dims=self.img_dims, folder=self.folder, \
                                    database_path=self.database_path, ft_type=self.feat_type)
        data_dict          = dict(svm=self.svm_model, scaler=self.scaler, rstd=self.rstd, rmean=self.rmean, factors=[self.factor1_cal, self.factor2_cal])
        del self.data
        self.data          = dict(params=params_dict, data=data_dict)
        self.data_ready    = True

        self.data_ready         = True

        if save:
            self.save_data(check_exists=True)

        return self
        
    def _check(self, dir):
        datasets = self._get_datasets(dir)
        for name in datasets:
            if datasets[name]['params'] == self.data['params']:
                return datasets[name]
        return ""
    
    def _get_datasets(self, dir=None):
        if dir is None:
            dir = self.data_dir
        datasets = {}
        try:
            entry_list = os.scandir(dir)
        except FileNotFoundError:
            self.print("Error: directory invalid.", LogType.ERROR)
            return datasets
        for entry in entry_list:
            if entry.is_file() and entry.name.startswith('dataset'):
                datasets[os.path.splitext(entry.name)[0]] = np.load(entry.path, allow_pickle=True)
        return datasets

    def _load(self, raw_model):
    # when loading objects inside dicts from .npz files, must extract with .item() each object
        del self.model
        self.model = dict(model=raw_model['model'].item(), params=raw_model['params'].item())
        self.model_ready = True
    
    def destroy(self):
        if self.init_hybridnet:
            self.hybridnet.destroy()
            del self.hybridnet
        if self.init_netvlad:
            self.netvlad.destroy()
            del self.netvlad
        del self.init_netvlad
        del self.init_hybridnet
