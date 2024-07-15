#!/usr/bin/env python3
'''
VPRDatasetProcessor Class; also has abstract VPRProcessorBase
'''
from __future__ import annotations

import copy
import logging
import os
from enum import Enum
from contextlib import suppress
from abc import ABC
import gc
import datetime
from pathlib import Path
from typing import Optional, Union, overload, Callable, List, Literal, Dict, Tuple#, Any, Type

import numpy as np

from pyaarapsi.vpr import config
from pyaarapsi.core.helper_tools import format_exception, vis_dict, \
    check_if_ndarray_with_ndim_or_more
try:
    from pyaarapsi.core.ros_tools import process_bag
except ImportError:
    logging.warning("Could not access ros_tools; generating features from rosbags will fail. "
                    "This is typically due to a missing or incorrect ROS installation. "
                    "\nError code: \n%s", format_exception())
from pyaarapsi.core.roslogger import LogType, roslogger
from pyaarapsi.core.file_system_tools import scan_directory
from pyaarapsi.vpr.vpr_helpers import filter_image_dataset, filter_feature_dataset, get_feat
from pyaarapsi.vpr.classes.vprdescriptor import VPRDescriptor
from pyaarapsi.vpr.classes.dimensions import ImageDimensions
from pyaarapsi.vpr.classes.data.rosbagdata import RosbagData
from pyaarapsi.vpr.classes.data.rosbagparams import RosbagParams
from pyaarapsi.vpr.classes.data.rosbagdataset import RosbagDataset

ROSPKG_ROOT = config.prep_rospkg_root()

class VPRDatasetProcessorException(Exception):
    '''
    Base class of VPRDatasetProcessor failure.
    '''

class DatasetLoadSaveError(VPRDatasetProcessorException):
    '''
    All errors with dataset loading, saving.
    '''

class FeatureExtractorException(VPRDatasetProcessorException):
    '''
    Base class of feature extraction failure.
    '''

class FeatureExtractorNotReady(FeatureExtractorException):
    '''
    Typically raised when a feature extractor has been called, but not loaded
    '''

class VPRProcessorBase(ABC):
    '''
    Abstract base class of VPRDatasetProcessor
    '''
    def __init__(self, cuda: bool = False, ros: bool = False, printer: Optional[Callable] = None,
                 init_containers: Optional[List[str]] = None):
        if init_containers is None:
            init_containers = []
        self.dataset: RosbagDataset = RosbagDataset()
        self.printer: Optional[Callable] = printer
        self.ros: bool = ros
        self.cuda: bool = cuda
        self.initialized_containers = {descriptor.name: False \
                             for descriptor in VPRDescriptor.descriptors_with_container()}
        self.init_containers(init_containers=init_containers)
        self.borrowed_containers = {descriptor.name: False \
                             for descriptor in VPRDescriptor.descriptors_with_container()}
        self.held_containers = {descriptor.name: None \
                             for descriptor in VPRDescriptor.descriptors_with_container()}
    #
    def get(self, _safe: bool = True) -> RosbagDataset:
        '''
        Get whole dataset

        Inputs:
        - None
        Returns:
            RosbagFeatureDataset type; whole dataset dictionary
        '''
        if _safe:
            return copy.deepcopy(self.dataset)
        return self.dataset
    #
    def get_data(self, _safe: bool = True) -> RosbagData:
        '''
        Get dataset data

        Inputs:
        - None
        Returns:
            RosbagFeatureData type; dataset data dictionary
        '''
        if _safe:
            return copy.deepcopy(self.dataset.get_data())
        return self.dataset.get_data()
    #
    def get_params(self, _safe: bool = True) -> RosbagParams:
        '''
        Get dataset parameters

        Inputs:
        - None
        Returns:
            RosbagParams type; dataset parameter dictionary
        '''
        if _safe:
            return copy.deepcopy(self.dataset.get_params())
        return self.dataset.get_params()
    #
    def show(self, printer = print) -> str:
        '''
        Print dataset contents

        Inputs:
        - printer: method type {default: print}; used for verbose printing
        Returns:
            str type; representation of dataset dictionary contents
        '''
        return vis_dict(self.dataset.to_dict(), printer=printer)
    #
    def print(self, text: str, logtype: LogType = LogType.INFO, throttle: float = 0) -> bool:
        '''
        Diagnostics and debugging print handler

        Inputs:
        - text:     str type; contents to print
        - logtype:  LogType type; logging channel to write to
        - throttle: float type {default: 0}; rate to limit publishing contents for repeated prints
        Returns:
            bool type; True if print succeeded
        '''
        text = '[VPRDatasetProcessor] ' + text
        if self.printer is None:
            return roslogger(text, logtype, throttle=throttle, ros=self.ros)
        self.printer(text, logtype, throttle=throttle, ros=self.ros)
        return True
    #
    def prep_descriptor(self, descriptor: VPRDescriptor, cuda: Optional[bool] = None, \
                        ready_up: bool = True, overwrite: bool = False) -> VPRProcessorBase:
        '''
        Prepare a VPR descriptor for use

        Inputs:
        - descriptor:   VPRDescriptor type; which descriptor to prepare
        - cuda:         bool type {default: None}; if not None, overrides initialisation cuda
                            variable for loading the descriptor
        - ready_up:     bool type {default: True}; whether or not to automatically ready the model
        Returns:
            Self
        '''
        if not descriptor.name in self.held_containers:
            raise VPRDescriptor.Exception(f"Descriptor {descriptor.name} does "
                                                    "not require preparation.")
        if (self.held_containers[descriptor.name] is not None) and overwrite:
            self.held_containers[descriptor.name] = None
        if self.held_containers[descriptor.name] is None:
            if cuda is None:
                cuda = self.cuda
            self.held_containers[descriptor.name] = descriptor.get_container_class()()
            self.initialized_containers[descriptor.name] = True
        if ready_up:
            self.held_containers[descriptor.name].ready_up()
        return self
    #
    def check_descriptor(self, descriptor: VPRDescriptor, descriptor_types: List[VPRDescriptor]
                         ) -> bool:
        '''
        Check if a descriptor is initialised: if not initialised but needed (i.e., ft_types
        contains descriptor.name), this function will attempt to initialize it.
        Delays loading of descriptor's container until required by system.

        Inputs:
            ft_types:  list type; list of VPRDescriptor enumeration names. If it contains
                        descriptor and descriptor is not loaded, this method will attempt to
                        initialise descriptor.
        Returns:
            bool type; True on successful update, False if no change
        '''
        # If needed and we've been asked to initialize the descriptor:
        if (descriptor in descriptor_types) and self.initialized_containers[descriptor.name]:
            return self.prep_descriptor(descriptor=descriptor, cuda=None, ready_up=True,
                                        overwrite=False)
        return False
    #
    def init_all_containers(self) -> VPRProcessorBase:
        '''
        Flag VPRDatasetProcessor to initialize all feature extraction containers

        Returns:
            Self
        '''
        self.initialized_containers = {container_name: True \
                                       for container_name in self.initialized_containers}
    #
    def init_containers(self, init_containers: List[str]) -> VPRProcessorBase:
        '''
        Flag VPRDatasetProcessor to initialize specific feature extraction containers

        Inputs:
            init_containers:  List[str] type; names of VPRDescriptor containers to initialize
        Returns:
            Self
        '''
        for container_name in init_containers:
            if container_name not in self.initialized_containers:
                raise KeyError(f"'{container_name}' was provided, but does not exist as an "
                               "initializable container.")
            self.initialized_containers[container_name] = True
        return self
    #
    def pass_containers(self, processor: VPRDatasetProcessor, containers: List[VPRDescriptor],
                 try_load_if_missing: bool = True) -> bool:
        '''
        Overwrite this VPRDatasetProcessor's instances of each container with another's
        instantiations. Will check and, if initialised, destroy existing container

        Inputs:
            processor:            VPRDatasetProcessor type
            containers:           List[VPRDescriptor] type; which descriptors to overwrite
            try_load_if_missing:  bool type {default: True}; whether to trigger a prep if detects
                                    uninitialised container instance
        Returns:
            Self
        '''
        assert isinstance(processor, VPRDatasetProcessor)
        for descriptor in containers:
            assert descriptor.name in self.initialized_containers, \
                "containers has non-initializable entry."
            if processor.held_containers[descriptor.name] is None:
                if try_load_if_missing:
                    processor.prep_descriptor(descriptor=descriptor, cuda=None, ready_up=False,
                                              overwrite=False)
                else:
                    raise FeatureExtractorNotReady('Passing requires processor to have container '
                            'ready, or pass argument try_load_if_missing=True. Container matching '
                            f'descriptor: {descriptor.name}')
                if self.held_containers[descriptor.name] is not None:
                    try:
                        del self.held_containers[descriptor.name]
                        self.held_containers.pop(descriptor.name)
                    except (NameError, KeyError):
                        self.print('Failed to destroy existing instance, with error: '
                               + str(format_exception()))
                self.held_containers[descriptor.name] = processor.held_containers[descriptor.name]
                self.initialized_containers[descriptor.name] = True
                self.borrowed_containers[descriptor.name] = True
                self.print(f'Passed container for {descriptor.get_descriptor_name()}.',
                           LogType.DEBUG)
        return self
    #
    @overload
    def get_feat(self, img: np.ndarray, descriptor: VPRDescriptor, dims: ImageDimensions, \
                 use_tqdm: bool = False) -> np.ndarray: ...
    #
    @overload
    def get_feat(self, img: List[np.ndarray], descriptor: VPRDescriptor, dims: ImageDimensions, \
                 use_tqdm: bool = False) -> List[np.ndarray]: ...
    #
    def get_feat(self, img: Union[np.ndarray, List[np.ndarray]], descriptor: VPRDescriptor, \
                 dims: ImageDimensions, use_tqdm: bool = False
                 ) -> Union[np.ndarray, List[np.ndarray]]:
        '''
        Feature Extraction Helper

        Inputs:
        - img:          np.ndarray type (or list of np.ndarray); Image array (can be RGB or 
                            greyscale; if greyscale, will be stacked to RGB equivalent dimensions
                            for neural network input.
        - descriptor:   VPRDescriptor type; Type of features to extract from each image.
        - dims:         Dimensions type; Dimensions to reduce input images to
        - use_tqdm:     bool type {default: False}; whether or not to display extraction/loading
                            statuses using tqdm
        Returns:
            np.ndarray type (or list of np.ndarray); flattened features from image

        '''
        if check_if_ndarray_with_ndim_or_more(img, 4):
            try:
                if img.ndim == 5:
                    img = [img[i,0] for i in range(img.shape[0])]
                else:
                    img = [img[i] for i in range(img.shape[0])]
            except Exception as e:
                raise FeatureExtractorException("Can only process large dimensional np.ndarray " \
                    "if shape in format (item, height, width, channels) for ndim==4, or (item, " \
                    f"1, height, width, channels) for ndim==5. Got shape: {str(img.shape)}") from e
        if not isinstance(descriptor, list):
            descriptors = [descriptor]
        else:
            descriptors = descriptor
        for descrip in descriptors:
            if descrip.requires_init:
                if self.initialized_containers[descrip.name]:
                    self.check_descriptor(descriptor=descrip, descriptor_types=descriptors)
                else:
                    raise FeatureExtractorNotReady(f"[get_feat] {descrip} provided but "
                                                    "VPRDatasetProcessor container not initialised")
        try:
            feats = get_feat(im=img, descriptors=descriptors, dims=dims, use_tqdm=use_tqdm,
                             containers=self.held_containers)
            if isinstance(feats, list):
                return [np.array(i, dtype=np.float32) for i in feats]
            return np.array(feats, dtype=np.float32)
        except Exception as e:
            raise FeatureExtractorException("[get_feat] Feature vector could not be constructed."
                                            f"\nCode: {e}") from e
    #
    def __del__(self):
        del self.dataset
        del self.printer
        del self.ros
        del self.cuda
        self.initialized_containers.clear()
        del self.initialized_containers
        self.borrowed_containers.clear()
        del self.borrowed_containers
        self.held_containers.clear()
        del self.held_containers

class VPRDatasetProcessor(VPRProcessorBase):
    '''
    All-in-one rosbag extractor, npz saver, and VPR feature extractor class.
    '''
    def __init__(self, dataset_params: Optional[RosbagParams] = None, try_gen: bool = True,
                 init_containers: Optional[List[str]] = None,
                 cuda: bool = False, use_tqdm: bool = False,
                 autosave: bool = False, ros: bool = True, root: Optional[str] = None,
                 printer: Callable = None):
        '''
        Inputs:
        - dataset_params:   RosbagParams type (default: None); params to define a unique extraction
        - try_gen:          bool type {default: True}; whether or not to attempt generation if
                                load fails
        - init_containers:  List[str] type; names of VPRDescriptor entries to initialize
                                containers for
        - cuda:             bool type {default: False}; whether or not to use CUDA for feature
                                type/GPU acceleration
        - use_tqdm:         bool type {default: False}; whether or not to display
                                extraction/loading statuses using tqdm
        - autosave:         bool type {default: False}; whether or not to automatically save any
                                generated datasets
        - ros:              bool type {default: True}; whether or not to use rospy logging
                                (requires operation within ROS node scope)
        - root:             str type {default: None}; root path to data storage
        - printer:          method {default: None}; if provided, overrides logging and will pass
                                inputs to specified method on print
        Returns:
            self
        '''
        super().__init__(cuda=cuda, ros=ros, printer=printer,
                         init_containers=init_containers)
        self.use_tqdm       = use_tqdm
        self.autosave       = autosave
        self.root           = ROSPKG_ROOT if root is None else root
        # Declare attributes:
        self.upd_dbp        = "/data/unprocessed_sets"
        self.npz_dbp        = "/data/compressed_sets"
        self.bag_dbp        = "/data/rosbags"
        # If parameters have been provided:
        if not dataset_params is None:
            self.print("Loading model from parameters...", LogType.DEBUG)
            name = self.load_dataset(dataset_params=dataset_params, try_gen=try_gen)
            if not self.dataset.is_populated():
                raise DatasetLoadSaveError("Dataset load failed.")
            self.print(f"Dataset Ready (loaded: {str(name)}).")
        else: # None-case needed for SVM training
            self.print("Ready; no dataset loaded.")
    #
    def unload(self) -> VPRDatasetProcessor:
        '''
        Unload any existing dataset
        Returns:
            Self
        '''
        self.dataset.unload()
        return self
    #
    def _adjust_upd_params_for_saving(self, params: RosbagParams) -> RosbagParams:
        '''
        Erase attributes in a RosbagParams attribute that do not impact unprocessed dataset
        generation, and would otherwise unnecessarily restrict parameter searches.
        '''
        return params.but_with(attr_changes={"vpr_descriptors": (), "img_dims": (), \
                                                       "feature_filters": ()})
    #
    def _adjust_upd_dataset_for_saving(self, dataset: RosbagDataset) -> RosbagDataset:
        '''
        Erase attributes in a RosbagDataset's RosbagParams attribute that do not impact
        unprocessed dataset generation, and would otherwise unnecessarily restrict parameter
        searches to match a dataset.
        '''
        return RosbagDataset().populate( \
            params=self._adjust_upd_params_for_saving(params=dataset.params),
            data=dataset.data)
    #
    def load_upd_dataset(self, dataset_params: RosbagParams, save: bool=True
                         ) -> Tuple[str, RosbagDataset]:
        '''
        Load an unprocessed dataset by searching for a param dictionary match into
        VPRDatasetProcessor.

        Inputs:
        - dataset_params:   RosbagParams type; params to define a unique extraction
        Returns:
            Tuple[str, RosbagDataset] type; name of, and actual, loaded or generated
                RosbagDataset for unprocessed data
        '''
        params_for_upd = self._adjust_upd_params_for_saving(params=dataset_params)
        self.print("[load_upd_dataset] Loading dataset.")
        upd_datasets = self.get_all_saved_upd_dataset_params()
        if len(dataset_params.image_filters) == 0:
            upd_name, loaded_upd_dataset = self._normal_load_upd_helper(datasets=upd_datasets, \
                                                                params=params_for_upd)
        else:
            upd_name, loaded_upd_dataset = self._filter_load_upd_helper(datasets=upd_datasets, \
                                                                params=params_for_upd, save=save)
        if upd_name != '':
            return upd_name, loaded_upd_dataset
        else:
            self.print("[load_dataset] Generating unprocessed dataset with params: "
                        f"{str(dataset_params.to_dict())}", LogType.DEBUG)
            loaded_upd_dataset = self.generate_upd_dataset(dataset_params=params_for_upd, \
                                                           save=save)
            return 'NEW GENERATION', loaded_upd_dataset
    #
    def _filter_load_upd_helper(self, datasets: Dict[str, RosbagParams], params: RosbagParams, \
                            save: bool = True) -> Tuple[str, RosbagDataset]:
        '''
        Helper function for filter loading and processing.
        '''
        params_for_upd = self._adjust_upd_params_for_saving(params=params)
        # Attempt to load, as-is:
        upd_name, loaded_upd_dataset = \
            self._normal_load_upd_helper(datasets=datasets, params=params_for_upd)
        if upd_name != '':
            return upd_name, loaded_upd_dataset # load success, early finish
        # Couldn't find a match: try to find an unfiltered version:
        filterless_upd_params = params_for_upd.but_with(attr_changes={"image_filters": ()})
        filterless_upd_name, filterless_loaded_upd_dataset \
            = self._normal_load_upd_helper(datasets=datasets, params=filterless_upd_params)
        if filterless_upd_name == '':
            return '', RosbagDataset() # load failed, leave (and maybe go to generation)
        # Success - now apply filters:
        filtered_upd_dataset = filter_image_dataset(dataset=RosbagDataset().populate(
                            params=params_for_upd,
                            data=filterless_loaded_upd_dataset.data
                            ), _printer=lambda msg: self.print(msg, LogType.DEBUG))
        if save:
            self.save_upd_dataset(dataset=filtered_upd_dataset)
        return 'NEW GENERATION', filtered_upd_dataset
    #
    def _normal_load_upd_helper(self, datasets: Dict[str, RosbagParams], params: RosbagParams \
                                ) -> Tuple[str, RosbagDataset]:
        params_for_upd = self._adjust_upd_params_for_saving(params=params)
        for key, value in datasets.items():
            if value == params_for_upd:
                try:
                    load_result = self._load_upd(dataset_name=key)
                    return key, load_result
                except IOError:
                    self._fix_unprocessed(key)
        return '', RosbagDataset()
    #
    def generate_upd_dataset(self, dataset_params: RosbagParams, save: bool = True
                             ) -> RosbagDataset:
        '''
        Generate new unprocessed datasets from parameters
        Parameter 'feature_filters' is ignored.

        Inputs:
        - dataset_params: RosbagParams type; params to define a unique extraction
        - store:          bool type {default: True}; whether or not to store in VPRDatasetProcessor
        Returns:
            dict type; Generated dataset dictionary
        '''
        try:
            params_for_upd = self._adjust_upd_params_for_saving(params=dataset_params)
            filterless_params_for_upd = params_for_upd.but_with(attr_changes={"image_filters": ()})
            bag_path = self.root +  '/' + self.bag_dbp + '/' + params_for_upd.bag_name
            # generate:
            rosbag_image_dataset = process_bag(bag_path=bag_path, \
                                            dataset_params=filterless_params_for_upd, \
                                            printer=self.print, use_tqdm=self.use_tqdm)
            if save:
                self.save_upd_dataset(dataset=rosbag_image_dataset)
            filtered_image_dataset = filter_image_dataset(RosbagDataset().populate(
                params=params_for_upd, data=rosbag_image_dataset.get_data()
                ), _printer=lambda msg: self.print(msg, LogType.DEBUG))
            if save:
                self.save_upd_dataset(dataset=filtered_image_dataset)
            return filtered_image_dataset
        except Exception as e:
            raise DatasetLoadSaveError("Failed to generate unprocessed data") from e
    #
    def generate_dataset(self, dataset_params: RosbagParams, store: bool = True) -> dict:
        '''
        Generate new datasets from parameters

        Inputs:
        - dataset_params: RosbagParams type; params to define a unique extraction
        - store:          bool type {default: True}; whether or not to store in VPRDatasetProcessor
        Returns:
            dict type; Generated dataset dictionary
        '''
        try:
            _, filtered_image_dataset = self.load_upd_dataset(dataset_params=dataset_params, \
                                                              save=self.autosave)
            self.print('[generate_dataset] Performing feature extraction...')
            # feature extraction:
            features_data   = {descriptor.name:
                                np.stack([
                                    self.get_feat(
                                        filtered_image_dataset.data.data[i], descriptor,
                                        dataset_params.img_dims, use_tqdm=self.use_tqdm
                                    ) for i in dataset_params.img_topics
                                ], axis=1) # stack each img_topic
                                for descriptor in dataset_params.vpr_descriptors}
            gc.collect() # ease memory load
            self.print('[generate_dataset] Done.')
            # Create dataset dictionary and add feature vectors
            rosbag_feature_data = RosbagData().populate(
                positions   = copy.deepcopy(filtered_image_dataset.data.positions),
                velocities  = copy.deepcopy(filtered_image_dataset.data.velocities),
                times       = copy.deepcopy(filtered_image_dataset.data.times),
                data        = features_data
            )
            if self.autosave and len(filtered_image_dataset.params.feature_filters) > 0:
                pre_filtered_dataset = RosbagDataset().populate(
                                        params=filtered_image_dataset.params.but_with(\
                                            attr_changes={"feature_filters":()}),
                                        data=rosbag_feature_data)
                self.save_dataset(dataset=pre_filtered_dataset)
            filtered_feature_dataset = filter_feature_dataset(RosbagDataset().populate(
                                        params=filtered_image_dataset.params,
                                        data=rosbag_feature_data
                                        ), _printer=lambda msg: self.print(msg, LogType.DEBUG))
            if store:
                self.dataset.populate(  params=filtered_feature_dataset.params, \
                                        data=filtered_feature_dataset.data)
                if self.autosave:
                    self.save_dataset()
                return self.dataset
            dataset = RosbagDataset().populate(params=filtered_feature_dataset.params, \
                                                data=filtered_feature_dataset.data)
            if self.autosave:
                self.save_dataset(dataset=dataset)
        except Exception as e:
            raise DatasetLoadSaveError("Failed to generate processed data") from e
    #
    def make_saveable_file_name(self, save_dir: str, name: Optional[str] = None, \
                             allow_overwrite: bool = False) -> Tuple[bool, str, List[str]]:
        '''
        Make a file name that is unique
        '''
        Path(save_dir).mkdir(parents=False, exist_ok=True)
        Path(save_dir+"/params").mkdir(parents=False, exist_ok=True)
        # Ensure file name is of correct format, generate if not provided
        file_list, _, _, = scan_directory(save_dir, short_files=True)
        overwriting = False
        if not name is None:
            if not name.startswith('dataset'):
                name = "dataset_" + name
            if name in file_list:
                if not allow_overwrite:
                    raise DatasetLoadSaveError(f"Dataset with name >{name}< already exists "
                                               "in directory.")
                else:
                    overwriting = True
        else:
            name = datetime.datetime.today().strftime("dataset_%Y%m%d")
        return overwriting, name, file_list
    #
    def save_upd_dataset(self, dataset: RosbagDataset, name: Optional[str] = None,
                     allow_overwrite: bool = False) -> VPRDatasetProcessor:
        '''
        Save unprocessed dataset to file system

        Inputs:
        - dataset:          RosbagDataset type; unprocessed dataset to save
        - name:             str type {default: None}; if None, a unique name will be generated of
                                the format dataset_%Y%m%d_X.
        - allow_overwrite:  bool type (default: False); whether to permit overwriting an existing
                                file with specified name
        Returns:
            self
        '''
        assert isinstance(dataset, RosbagDataset)
        if not dataset.is_populated():
            raise DatasetLoadSaveError("Dataset not populated.")
        save_dir = self.root + '/' + self.upd_dbp
        # Generate unique name:
        overwriting, name, file_list = self.make_saveable_file_name(save_dir=save_dir, name=name, \
                                                           allow_overwrite=allow_overwrite)
        file_name = name
        if not overwriting:
            count = 0
            while file_name in file_list:
                file_name = name + f"_{count:d}"
                count += 1
        file_list       = file_list + [file_name]
        full_file_path  = save_dir + "/" + file_name
        full_param_path = save_dir + "/params/" + file_name

        if not overwriting:
            file_ = self._check(params=dataset.params)
            if file_:
                self.print(f"[save_dataset] File exists with identical parameters ({file_}); "
                            "skipping save.", LogType.DEBUG)
                return self
        if overwriting:
            with suppress(FileNotFoundError):
                os.remove(full_file_path)
            with suppress(FileNotFoundError):
                os.remove(full_param_path)
        np.savez(full_file_path, params=dataset.params.save_ready(),
                                    data=dataset.data.save_ready())
        np.savez(full_param_path, params=dataset.params.save_ready())
        if overwriting:
            self.print(f"[save_upd_dataset] Overwrite complete.\n\t  file: {full_file_path}"
                        f"\n\tparams: {full_param_path}.")
        else:
            self.print(f"[save_upd_dataset] Save complete.\n\t  file: {full_file_path}"
                        f"\n\tparams: {full_param_path}.")
        return self
    #
    def save_dataset(self, dataset: Optional[RosbagDataset] = None, name: Optional[str] = None,
                     allow_overwrite: bool = False) -> VPRDatasetProcessor:
        '''
        Save loaded dataset to file system

        Inputs:
        - dataset:          RosbagDataset type (default: None); dataset to save. If none, uses
                                self.dataset
        - name:             str type {default: None}; if None, a unique name will be generated of
                                the format dataset_%Y%m%d_X.
        - allow_overwrite:  bool type (default: False); whether to permit overwriting an existing
                                file with specified name
        Returns:
            self
        '''
        if dataset is None:
            dataset = self.dataset
        assert isinstance(dataset, RosbagDataset)
        if not dataset.is_populated():
            raise DatasetLoadSaveError("Dataset not loaded in system. Either call "
                                       "'generate_dataset' or 'load_dataset' before using this "
                                       "method.")
        save_dir = self.root + '/' + self.npz_dbp
        overwriting, name, file_list = self.make_saveable_file_name(save_dir=save_dir, name=name, \
                                                           allow_overwrite=allow_overwrite)
        self.print("[save_dataset] Splitting dataset into files for different descriptors: "
                   f"{dataset.get_params().vpr_descriptors}", LogType.DEBUG)
        for descriptor in dataset.get_params().vpr_descriptors:
            # Generate unique name:
            file_name = name
            if not overwriting:
                count = 0
                while file_name in file_list:
                    file_name = name + f"_{count:d}"
                    count += 1
            file_list       = file_list + [file_name]
            full_file_path  = save_dir + "/" + file_name
            full_param_path = save_dir + "/params/" + file_name

            sub_dataset = dataset.to_singular_descriptor(descriptor)

            if not overwriting:
                file_ = self._check(params=sub_dataset.params)
                if file_:
                    self.print(f"[save_dataset] File exists with identical parameters ({file_}); "
                               "skipping save.", LogType.DEBUG)
                    continue
            if overwriting:
                with suppress(FileNotFoundError):
                    os.remove(full_file_path)
                with suppress(FileNotFoundError):
                    os.remove(full_param_path)
            np.savez(full_file_path, params=sub_dataset.params.save_ready(),
                                        data=sub_dataset.data.save_ready())
            np.savez(full_param_path, params=sub_dataset.params.save_ready())
            if overwriting:
                self.print(f"[save_dataset] Overwrite complete.\n\t  file: {full_file_path}"
                           f"\n\tparams: {full_param_path}.")
            else:
                self.print(f"[save_dataset] Save complete.\n\t  file: {full_file_path}"
                           f"\n\tparams: {full_param_path}.")
            del sub_dataset
        return self
    #
    def extend_dataset(self, new_descriptor: VPRDescriptor, try_gen: bool = False,
                       save: bool = False) -> bool:
        '''
        Add an additional feature type into the loaded data set

        Inputs:
        - new_descriptor:   VPRDescriptor type; new descriptor to load
        - try_gen:          bool type {default: True}; whether or not to attempt generation if load
                                fails.
        - save:             bool type {default: False}; whether or not to automatically save any
                                generated datasets.
        Returns:
            bool type; True if successful extension.
        '''
        if not self.dataset.is_populated():
            self.print("[extend_dataset] Load failed; no dataset loaded to extend.", LogType.DEBUG)
            return False
        new_descriptor = \
            new_descriptor.name if isinstance(new_descriptor, Enum) else new_descriptor
        new_params = copy.deepcopy(self.dataset.get_params())
        new_params.vpr_descriptors = (new_descriptor,)
        self.print("[extend_dataset] Loading dataset to extend existing...", LogType.DEBUG)
        new_dataset = self.open_dataset(new_params, try_gen=try_gen)
        if new_dataset is None:
            self.print("[extend_dataset] Load failed.", LogType.DEBUG)
            return False
        self.print("[extend_dataset] Load success. Extending.", LogType.DEBUG)
        self.dataset.data.data[new_descriptor] = \
            copy.deepcopy(new_dataset.data.data[new_descriptor])
        self.dataset.params.vpr_descriptors = \
            (*self.dataset.params.vpr_descriptors, new_descriptor)
        if save:
            self.save_dataset()
        return True
    #
    def open_dataset(self, dataset_params: RosbagParams, try_gen: bool = False
                     ) -> Union[RosbagDataset, None]:
        '''
        Open a dataset by searching for a param dictionary match (return as a variable, do not
        store in VPRDatasetProcessor)

        Inputs:
        - dataset_params:   RosbagParams type; params to define a unique extraction
        - try_gen:          bool type {default: False}; whether or not to attempt generation if
                                load fails
        Returns:
            dict type; opened dataset dictionary if successful, else None.
        '''
        # load via search for param match but don't load into the processor's self.dataset
        assert len(dataset_params['ft_types']) == 1, 'Can only open one dataset'

        self.print("[open_dataset] Loading dataset.", LogType.DEBUG)
        datasets = self.get_all_saved_dataset_params()
        for key, value in datasets.items():
            if value == dataset_params:
                try:
                    self.print("[open_dataset] Loading ...", LogType.DEBUG)
                    return self._load_npz(key, store=False)
                except IOError:
                    self._fix_processed(key)
        if try_gen:
            try:
                self.print("[open_dataset] Generating ...", LogType.DEBUG)
                return self.generate_dataset(**dataset_params, store=False)
            except DatasetLoadSaveError:
                self.print(format_exception())
        self.print("[open_dataset] Load, generation failed.", LogType.DEBUG)
        return None
    #
    def get_bag_path(self) -> str:
        '''
        Return full file path to bag
        '''
        assert self.dataset.is_populated()
        return self.root +  '/' + self.bag_dbp + '/' + self.dataset.params.bag_name + ".bag"
    #
    def load_dataset(self, dataset_params: RosbagParams, try_gen: bool = False) -> str:
        '''
        Load a dataset by searching for a param dictionary match into VPRDatasetProcessor

        Inputs:
        - dataset_params:   RosbagParams type; params to define a unique extraction
        - try_gen:          bool type {default: False}; whether or not to attempt generation if
                                load fails
        Returns:
            str type; loaded dataset dictionary file name if successful, 'NEW GENERATION' if a
                        file is generated, else ''.
        '''

        self.print("[load_dataset] Loading dataset.")
        datasets = self.get_all_saved_dataset_params()
        sub_params = dataset_params.to_descriptor(dataset_params.vpr_descriptors[0])
        if len(dataset_params.feature_filters) == 0:
            _name = self._normal_load_helper(datasets=datasets, params=sub_params)
        else:
            _name = self._filter_load_helper(datasets=datasets, params=sub_params, \
                                             save=try_gen)
        if not _name == '':
            if len(dataset_params.vpr_descriptors) > 1:
                for descriptor in dataset_params.vpr_descriptors[1:]:
                    if not self.extend_dataset(descriptor, try_gen=try_gen, save=try_gen):
                        return ''
            return _name
        else:
            if try_gen:
                self.print("[load_dataset] Generating dataset with params: "
                           f"{str(dataset_params.to_dict())}", LogType.DEBUG)
                self.generate_dataset(dataset_params=dataset_params, store=True)
                return 'NEW GENERATION'
            return ''

    def _filter_load_helper(self, datasets: Dict[str, RosbagParams], params: RosbagParams, \
                            save: bool = True) -> str:
        '''
        Helper function for filter loading and processing.
        '''
        # Attempt to load, as-is:
        name_from_filter_load = self._normal_load_helper(datasets=datasets, params=params,
                                                         store=True)
        if name_from_filter_load != '':
            return name_from_filter_load # load success, early finish
        # Couldn't find a match: try to find an unfiltered version:
        filterless_params = params.but_with(attr_changes={"feature_filters": ()})
        filterless_dataset = self._normal_load_helper(datasets=datasets, params=filterless_params,
                                                   store=False)
        if not filterless_dataset.is_populated():
            return '' # load failed, leave (and maybe go to generation)
        # Success - now apply filters:
        filtered_dataset = filter_feature_dataset(dataset=RosbagDataset().populate(
                            params=params,
                            data=filterless_dataset.data
                            ), _printer=lambda msg: self.print(msg, LogType.DEBUG))
        self.dataset.populate(params=filtered_dataset.params, data=filtered_dataset.data)
        if save:
            self.save_dataset()
        return 'NEW GENERATION'
    #
    @overload
    def _normal_load_helper(self, datasets: Dict[str, RosbagParams], params: RosbagParams,
                            store: Literal[True] = True) -> str:
        ...
    #
    @overload
    def _normal_load_helper(self, datasets: Dict[str, RosbagParams], params: RosbagParams,
                            store: Literal[False] = True) -> RosbagDataset:
        ...
    #
    def _normal_load_helper(self, datasets: Dict[str, RosbagParams], params: RosbagParams,
                            store: bool = True) -> Union[str, RosbagDataset]:
        for key, value in datasets.items():
            if value == params:
                try:
                    load_result = self._load_npz(dataset_name=key, store=store)
                    return key if store else load_result
                except IOError:
                    self._fix_processed(key)
        return '' if store else RosbagDataset()
    #
    def swap(self, dataset_params: RosbagParams, generate: bool = False, allow_false: bool = True
             ) -> bool:
        '''
        Swap out the currently loaded dataset by searching for a param dictionary match into
        VPRDatasetProcessor.
        Attempts to simply extend the existing dataset if possible.

        Inputs:
        - dataset_params:   RosbagParams type; params to define a unique extraction
        - generate:         bool type {default: False}; whether or not to attempt generation if
                                load fails
        - allow_false:      bool type {default: True}; if False, triggers an exception if the swap
                                operation fails.
        Returns:
            bool type; True if successful, else False.
        '''
        if not self.dataset.is_populated():
            raise DatasetLoadSaveError('No dataset presently loaded.')
        self.print('[swap] Attempting to extend dataset...')
        # Check if we can just extend the current dataset:
        if dataset_params.to_descriptor(self.dataset.params.vpr_descriptors) \
            == self.dataset.params:
            self.print("[swap] Candidate for extension. Feature types: "
                        f"{str(dataset_params.vpr_descriptors)} [against: "
                        f"{str(self.dataset.params.vpr_descriptors)}]", LogType.DEBUG)
            for descriptor in dataset_params.vpr_descriptors:
                if not descriptor in self.dataset.params.vpr_descriptors:
                    return self.extend_dataset(descriptor, try_gen=generate, save=generate)
        self.print('[swap] Extension failed, attempting to load dataset...')
        # Can't extend, let's load in the replacement:
        if self.load_dataset(dataset_params, try_gen=generate):
            return True
        if not allow_false:
            raise DatasetLoadSaveError('Dataset failed to load.')
        return False
    #
    def __del__(self) -> None:
        '''
        Class destructor

        Inputs:
        - None
        Returns:
            None
        '''
        super(VPRDatasetProcessor, self).__del__()
        del self.use_tqdm
        del self.autosave
        del self.root
        del self.npz_dbp
        del self.bag_dbp
    #
    def _check(self, params: Optional[RosbagParams] = None) -> str:
        '''
        Helper function to check if params already exist in saved npz_dbp library

        Inputs:
        - params: dict type {default: None}; parameter dictionary to compare against in search
                    (if None, uses loaded dataset parameter dictionary)
        Returns:
            str type; "" if no matching parameters, otherwise file name of matching parameters
                    (from npz_dbp/params)
        '''
        datasets = self.get_all_saved_dataset_params()
        if params is None:
            if not self.dataset.is_populated():
                return ""
            params = self.dataset.get_params()
        for key, value in datasets.items():
            if value == params:
                return key
        return ""
    #
    def get_all_saved_dataset_params(self) -> Dict[str, RosbagParams]:
        '''
        Helper function to iterate over parameter dictionaries and compile a list
        For: processed datasets

        Inputs:
        - None
        Returns:
            dict type; keys correspond to file names in npz_dbp/params, values are file contents
                (loaded parameter dictionaries)
        '''
        param_choices = {}
        try:
            entry_list = os.scandir(self.root + '/' + self.npz_dbp + "/params/")
        except FileNotFoundError as e:
            raise FileNotFoundError("Directory invalid.") from e
        for entry in entry_list:
            if entry.is_file() and entry.name.startswith('dataset'):
                raw_npz = dict(np.load(entry.path, allow_pickle=True))
                param_choices[os.path.splitext(entry.name)[0]] = \
                    RosbagParams.from_save_ready(raw_npz['params'].item())
        return param_choices
    #
    def get_all_saved_upd_dataset_params(self) -> Dict[str, RosbagParams]:
        '''
        Helper function to iterate over parameter dictionaries and compile a list
        For: unprocessed datasets

        Inputs:
        - None
        Returns:
            dict type; keys correspond to file names in upd_dbp/params, values are file contents
                (loaded parameter dictionaries)
        '''
        param_choices = {}
        try:
            entry_list = os.scandir(self.root + '/' + self.upd_dbp + "/params/")
        except FileNotFoundError as e:
            raise FileNotFoundError("Directory invalid.") from e
        for entry in entry_list:
            if entry.is_file() and entry.name.startswith('dataset'):
                raw_npz = dict(np.load(entry.path, allow_pickle=True))
                param_choices[os.path.splitext(entry.name)[0]] = \
                    RosbagParams.from_save_ready(raw_npz['params'].item())
        return param_choices
    #
    def _fix(self, dataset_name: str, path: str) -> list:
        '''
        Helper function to remove broken/erroneous files

        Inputs:
        - dataset_name: str type; name of file to search and remove
        - path: str type; path for where to search
        Returns:
            list type; contains two booleans: 1) whether a data set was purged, 2) whether a param
                        file was purged 
        '''
        if not dataset_name.endswith('.npz'):
            dataset_name = dataset_name + '.npz'
        _purges = [False, False]
        self.print("[_fix_unprocessed] Bad dataset state detected, performing cleanup...", \
                   LogType.DEBUG)
        try:
            os.remove(path + dataset_name)
            self.print(f"[_fix_unprocessed] Purged: {path + dataset_name}", \
                       LogType.DEBUG)
            _purges[0] = True
        except FileNotFoundError:
            pass
        try:
            os.remove(path + 'params/' + dataset_name)
            self.print(f"[_fix_unprocessed] Purged: {path + 'params/' + dataset_name}", \
                       LogType.DEBUG)
            _purges[1] = True
        except FileNotFoundError:
            pass
        return _purges
    #
    def _fix_unprocessed(self, dataset_name: str) -> list:
        '''
        Helper function to remove broken/erroneous unprocessed files

        Inputs:
        - dataset_name: str type; name of file to search and remove (from upd_dbp)
        Returns:
            list type; contains two booleans: 1) whether a data set was purged, 2) whether a param
                        file was purged 
        '''
        return self._fix(dataset_name=dataset_name, path=self.root + '/' + self.upd_dbp + '/')
    #
    def _fix_processed(self, dataset_name: str) -> list:
        '''
        Helper function to remove broken/erroneous processed files

        Inputs:
        - dataset_name: str type; name of file to search and remove (from npz_dbp)
        Returns:
            list type; contains two booleans: 1) whether a data set was purged, 2) whether a param
                        file was purged 
        '''
        return self._fix(dataset_name=dataset_name, path=self.root + '/' + self.npz_dbp + '/')
    #
    def _load_npz(self, dataset_name: str, store: bool = True) -> RosbagDataset:
        '''
        Helper function to load a dataset from a file name

        Inputs:
        - dataset_name: str type; name of file to load (from npz_dbp directory)
        - store         bool type {default: True}; whether or not to store in VPRDatasetProcessor
        Returns:
            RosbagDataset type; loaded dataset
        '''
        return self._load(dataset_name=dataset_name, path=self.root + '/' + self.npz_dbp + '/', \
                          store=store)
    #
    def _load_upd(self, dataset_name: str) -> RosbagDataset:
        '''
        Helper function to load a dataset from a file name

        Inputs:
        - dataset_name: str type; name of file to load (from upd_dbp directory)
        Returns:
            RosbagDataset type; loaded dataset
        '''
        return self._load(dataset_name=dataset_name, path=self.root + '/' + self.upd_dbp + '/', \
                          store=False)
    #
    def _load(self, dataset_name: str, path: str, store: bool = True) -> RosbagDataset:
        '''
        Helper function to load a dataset from a file name

        Inputs:
        - dataset_name: str type; name of file to load
        - path:         str type; directory to load from
        - store         bool type {default: True}; whether or not to store in VPRDatasetProcessor
        Returns:
            RosbagDataset type; loaded dataset
        '''
        if not dataset_name.endswith('.npz'):
            dataset_name = dataset_name + '.npz'
        full_file_path  = path + dataset_name
        self.print(f'[_load] Attempting to load: {full_file_path}', LogType.DEBUG)
        raw_dataset = np.load(full_file_path, allow_pickle=True)
        # when loading objects inside dicts from .npz files, must extract with .item() each object:
        if store:
            self.dataset.populate(
                params=RosbagParams.from_save_ready(raw_dataset['params'].item()),
                data=RosbagData.from_save_ready(raw_dataset['data'].item())
            )
            return self.dataset
        return RosbagDataset().populate(
                params=RosbagParams.from_save_ready(raw_dataset['params'].item()),
                data=RosbagData.from_save_ready(raw_dataset['data'].item())
            )
