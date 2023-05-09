#!/usr/bin/env python3

import numpy as np
import copy
import rospkg
import os
import datetime
from pathlib import Path
from ..core.enum_tools import enum_name, enum_get
from ..core.ros_tools import process_bag
from ..vpr_classes import NetVLAD_Container, HybridNet_Container
from ..core.file_system_tools import scan_directory
from .vpr_helpers import *

class VPRDatasetProcessor: # main ROS class
    def __init__(self, dataset_params: dict, try_gen=True, init_netvlad=False, init_hybridnet=False, cuda=False, use_tqdm=False, autosave=False, printer=print):
        self.dataset_ready  = False
        self.cuda           = cuda
        self.use_tqdm       = use_tqdm
        self.autosave       = autosave
        self.init_netvlad   = init_netvlad
        self.init_hybridnet = init_hybridnet

        self.print          = printer
        self.netvlad        = None
        self.hybridnet      = None

        if self.init_netvlad:
            self.netvlad    = NetVLAD_Container(cuda=self.cuda, ngpus=int(self.cuda), logger=self.print)

        if self.init_hybridnet:
            self.hybridnet  = HybridNet_Container(cuda=self.cuda, logger=self.print)

        if not (dataset_params is None): # None-case needed for SVM training
            self.print("[VPRDatasetProcessor] Loading model from parameters...")
            self.load_dataset(dataset_params, try_gen=try_gen)

            if not self.dataset_ready:
                raise Exception("Dataset load failed.")
            self.print("[VPRDatasetProcessor] Dataset Ready.")
        else:
            self.print("[VPRDatasetProcessor] Ready; no dataset loaded.")

    def prep_netvlad(self, cuda=None, load=True, prep=True):
        if not (cuda is None):
            self.cuda = cuda
        self.netvlad        = NetVLAD_Container(cuda=self.cuda, ngpus=int(self.cuda), logger=self.print, load=load, prep=prep)
        self.init_netvlad   = True

    def prep_hybridnet(self, cuda=None):
        if not (cuda is None):
            self.cuda = cuda
        self.hybridnet  = HybridNet_Container(cuda=self.cuda, logger=self.print)
        self.init_hybridnet   = True

    def pass_nns(self, vpr_image_proc, netvlad=True, hybridnet=True):
        if (not self.netvlad is None) and netvlad:
            self.netvlad.destroy()
            self.netvlad = vpr_image_proc.netvlad
            self.init_netvlad = True
        if (not self.hybridnet is None) and hybridnet:
            self.hybridnet.destroy()
            self.hybridnet = vpr_image_proc.hybridnet
            self.init_hybridnet = True

    def generate_dataset(self, npz_dbp, bag_dbp, bag_name, sample_rate, odom_topic, img_topics, img_dims, ft_types, filters={}):
        # store for access in saving operation:
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

        del self.dataset   
        self.dataset        = dataset
        self.dataset_ready  = True

        if self.autosave:
            self.save_dataset()

        return dataset

    def save_dataset(self, name=None):
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
        new_params = self.dataset['params']
        new_params['ft_types'] = new_ft_type
        if not self.load_dataset(new_params):
            if try_gen:
                new_dataset = self.generate_dataset(**new_params, load=False)
            else:
                return False
        self.dataset[new_ft_type] = copy.deepcopy(new_dataset[new_ft_type])
        if save:
            self.save_dataset()
        return True

    def load_dataset(self, dataset_params, try_gen=False):
    # load via search for param match
        self.npz_dbp = dataset_params['npz_dbp']
        self.bag_dbp = dataset_params['bag_dbp']
        self.dataset_ready = False
        self.print("[load_dataset] Loading dataset.")
        datasets = self._get_datasets()
        self.dataset = {}
        sub_params = copy.deepcopy(dataset_params)
        sub_params['ft_types'] = [dataset_params['ft_types'][0]]

        for name in datasets:
            if dict(datasets[name]['params']) == sub_params:
                self._load(name)
                break
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
        dataset_params_test = copy.deepcopy(dataset_params)
        dataset_params_test['ft_types'] = self.dataset['params']['ft_types']
        if dataset_params_test == self.dataset['params']:
            for ft_type in dataset_params['ft_types']:
                if not (ft_type in self.dataset['params']['ft_types']):
                    if not self.extend_dataset(ft_type, try_gen=generate, save=generate):
                        if not allow_false:
                            raise Exception('Dataset failed to load.')
                        return False
            return True
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
        self.print("[_fix] Bad dataset state detected, performing cleanup...")
        try:
            os.remove(rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/' + self.npz_dbp + '/' + dataset_name)
            self.print("[_fix] Purged: %s" % (self.npz_dbp + '/' + dataset_name))
        except:
            pass
        try:
            os.remove(rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/' + self.npz_dbp + '/params/' + dataset_name)
            self.print("[_fix] Purged: %s" % (self.npz_dbp + '/params/' + dataset_name))
        except:
            pass

    def _load(self, dataset_name,):
    # when loading objects inside dicts from .npz files, must extract with .item() each object
        if not dataset_name.endswith('.npz'):
            dataset_name = dataset_name + '.npz'
        full_file_path = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/' + self.npz_dbp + '/' + dataset_name
        self.print('[_load] Attempting to load: %s' % full_file_path)
        raw_dataset = np.load(full_file_path, allow_pickle=True)

        del self.dataset
        self.dataset = dict(dataset=raw_dataset['dataset'].item(), params=raw_dataset['params'].item())
        self.dataset_ready = True
        return self.dataset
    
    def _extend(self, dataset_name, ft_type):
    # when loading objects inside dicts from .npz files, must extract with .item() each object
        if not dataset_name.endswith('.npz'):
            dataset_name = dataset_name + '.npz'
        if isinstance(ft_type, FeatureType):
            ft_type = enum_name(ft_type)
        
        raw_dataset = np.load(rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/' + self.npz_dbp + "/" + dataset_name, allow_pickle=True)['dataset'].item()
        self.dataset['dataset'][ft_type] = raw_dataset[ft_type]
        return self.dataset
    
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
