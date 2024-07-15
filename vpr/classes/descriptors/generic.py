#!/usr/bin/env python3
'''
Abstract base class for VPR containers
'''
from abc import ABC, abstractmethod

from torch.utils.data import Dataset

class PlaceDatasetError(Exception):
    '''
    Error in processing
    '''

class DescriptorContainerError(Exception):
    '''
    Error in processing
    '''

class GenericPlaceDataset(Dataset, ABC):
    '''
    Abstract base class for PlaceDatasets
    '''
    @abstractmethod
    def __getitem__(self, index):
        '''
        Get items
        '''
        raise NotImplementedError("")
    #
    @abstractmethod
    def __len__(self):
        '''
        Length of dataset
        '''
        raise NotImplementedError("")
    #
    @abstractmethod
    def __del__(self):
        '''
        Clean-up
        '''
        raise NotImplementedError("")

class DescriptorContainer(ABC):
    '''
    Abstract base class for VPR containers 
    '''
    @abstractmethod
    def is_ready(self):
        '''
        Whether model can be used right now
        '''
        raise NotImplementedError()
    #
    @abstractmethod
    def ready_up(self):
        '''
        Prepare for feature extraction
        '''
        raise NotImplementedError()
    #
    @abstractmethod
    def load(self):
        '''
        Load model data
        '''
        raise NotImplementedError()
    #
    @abstractmethod
    def __del__(self):
        '''
        Destroy all data, objects
        '''
        raise NotImplementedError()
    #
    @abstractmethod
    def prep(self):
        '''
        Any prepwork, such as passing a default object through the container
        '''
        raise NotImplementedError()
    #
    @abstractmethod
    def get_feat(self, dataset_input, dims=None, use_tqdm=True, save_dir=None):
        '''
        Perform feature extraction
        '''
        raise NotImplementedError()
