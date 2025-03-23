#! /usr/bin/env python3
'''
Define a selection of class objects
'''
from __future__ import annotations

import copy
from typing import Optional
import numpy as np
# pylint: disable=E0611
from torch import from_numpy, tensor
# pylint: enable=E0611
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler

class BasicDataset(Dataset):
    '''
    Dataset class to be used with DataLoader
    '''
    def __init__(self, data: list, labels: list, gt: list, tol: list,
                 scale_data: bool = True, provide_gt: bool = False,
                 scaler: Optional[StandardScaler] = None):
        #
        assert isinstance(data, list) and isinstance(labels, list) and isinstance(gt, list)\
            and isinstance(tol, list)
        self.raw_data   = copy.deepcopy(data)
        self.raw_labels = copy.deepcopy(labels)
        self.raw_gt     = copy.deepcopy(gt)
        self.raw_tol    = copy.deepcopy(tol)
        assert len(self.raw_data) == len(self.raw_labels) == len(self.raw_gt) == len(self.raw_tol)
        # Init variables to empty lists:
        self.data       = []
        self.labels     = []
        self.gt         = []
        self.tol        = []
        self.scaled_raw_data = None
        #
        self.provide_gt = provide_gt
        #
        self.scale_data = scale_data
        self.fitted     = not scaler is None
        self.scaler     = StandardScaler() if not self.fitted else copy.deepcopy(scaler)
        self.seed       = None
        #
        self.update()

    def get_dataset_vars(self) -> dict:
        '''
        For load/saving BasicDataset.
        '''
        return {"provide_gt": self.provide_gt, "scale_data": self.scale_data, \
                "fitted": self.fitted, "seed": self.seed}

    def set_dataset_vars(self, dataset_vars: dict) -> None:
        '''
        For load/saving BasicDataset.
        '''
        self.provide_gt = dataset_vars['provide_gt']
        self.scale_data = dataset_vars['scale_data']
        self.fitted     = dataset_vars['fitted']
        self.seed       = dataset_vars['seed']

    def set_seed(self, seed: int) -> BasicDataset:
        '''
        Set random generator seed; used in shuffling.
        '''
        self.seed = seed
        return self

    def get_seed(self) -> int:
        '''
        Get random generator seed.
        '''
        return self.seed

    def fuse(self, dataset: BasicDataset) -> BasicDataset:
        '''
        Extend existing dataset by another.
        '''
        self.raw_data.extend(dataset.get_raw_data())
        self.raw_labels.extend(dataset.get_raw_labels())
        self.raw_gt.extend(dataset.get_raw_gt())
        self.raw_tol.extend(dataset.get_raw_tol())
        assert len(self.raw_data) == len(self.raw_labels) == len(self.raw_gt) == len(self.raw_tol)
        #
        self.update()
        #
        return self

    def update(self) -> BasicDataset:
        '''
        Re-calculate dataset lists.
        '''
        self.data       = []
        self.labels     = []
        self.gt         = []
        self.tol        = []
        self.length     = len(self.raw_data)
        #
        if not self.fitted:
            self.fit_scaler()
        else:
            self.use_scaler()
        #
        for i in range(self.length):
            self.data.append(from_numpy(self.scaled_raw_data[i]).float())
            self.labels.append(tensor(self.raw_labels[i]).float())
            self.gt.append(tensor(self.raw_gt[i]).float())
            self.tol.append(tensor(self.raw_tol[i]).float())
        assert len(self.data) == len(self.labels) == len(self.gt) == len(self.tol)
        #
        return self

    def shuffle(self) -> BasicDataset:
        '''
        Use seed to shuffle dataset.
        '''
        inds = list(range(self.length))
        np.random.seed(self.seed)
        np.random.shuffle(inds)
        self.data   = [self.data[i]     for i in inds]
        self.labels = [self.labels[i]   for i in inds]
        self.gt     = [self.gt[i]       for i in inds]
        self.tol    = [self.tol[i]      for i in inds]
        assert len(self.data) == len(self.labels) == len(self.gt) == len(self.tol)
        return self

    def fit_scaler(self) -> BasicDataset:
        '''
        Try-fit a scaler; regardless, generate self.scaled_raw_data
        '''
        if self.scale_data:
            self.scaled_raw_data = self.scaler.fit_transform(self.raw_data)
            self.fitted = True
        else:
            self.scaled_raw_data = copy.deepcopy(self.raw_data)
        return self

    def use_scaler(self) -> BasicDataset:
        '''
        Apply a scaler; regardless, generate self.scaled_raw_data
        '''
        if self.scale_data:
            self.scaled_raw_data = self.scaler.transform(self.raw_data)
        else:
            self.scaled_raw_data = copy.deepcopy(self.raw_data)
        return self

    def pass_scaler(self, scaler: StandardScaler) -> BasicDataset:
        '''
        Override scaler attribute.
        '''
        self.scaler = copy.deepcopy(scaler)
        self.fitted = True
        self.update()
        return self

    def get_scaler(self) -> StandardScaler:
        '''
        Return scaler attribute
        '''
        return self.scaler

    def get_raw_data(self) -> list:
        '''
        Return raw_data attribute
        '''
        return self.raw_data

    def get_raw_labels(self) -> list:
        '''
        Return raw_labels attribute
        '''
        return self.raw_labels

    def get_raw_gt(self) -> list:
        '''
        Return raw_gt attribute
        '''
        return self.raw_gt

    def get_raw_tol(self) -> list:
        '''
        Return raw_tol attribute
        '''
        return self.raw_tol

    def get_data(self) -> list:
        '''
        Return data attribute
        '''
        return self.data

    def get_labels(self) -> list:
        '''
        Return labels attribute
        '''
        return self.labels

    def get_gt(self) -> list:
        '''
        Return gt attribute
        '''
        return self.gt

    def get_tol(self) -> list:
        '''
        Return tol attribute
        '''
        return self.tol

    def __getitem__(self, item) -> tuple:
        if not self.provide_gt:
            return self.data[item], self.labels[item]
        return self.data[item], self.labels[item], self.gt[item], self.tol[item]

    def __len__(self) -> int:
        return self.length
