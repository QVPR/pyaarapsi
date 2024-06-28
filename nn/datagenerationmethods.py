#! /usr/bin/env python3
'''
Abstract base class for generating datasets for neural networks
'''
from abc import ABC, abstractmethod
from enum import Enum
from itertools import chain as ch
from typing import Optional, Tuple, Union

import numpy as np

from pyaarapsi.nn.general_helpers import get_rand_seed
from pyaarapsi.nn.enums import SampleMode

class DataGenerationMethods(ABC):
    '''
    A class to allow abstraction of data generation
    '''
    @staticmethod
    def construct_train_check_data_lists(   fd_list: list, train_inds: list, check_inds: list,
                                            shuffle: bool = True, rand_seed: Optional[int] = None
                                            ) -> Tuple[list, list, int]:
        '''
        Helper to split training data into train, check
        '''
        train_data_list = list(ch.from_iterable([[fdl[i] for i in inds] \
            for fdl, inds in zip(fd_list, train_inds)]))
        check_data_list = list(ch.from_iterable([[fdl[i] for i in inds] \
            for fdl, inds in zip(fd_list, check_inds)]))
        if shuffle:
            if rand_seed is None:
                rand_seed = get_rand_seed()
            np.random.seed(rand_seed)
            # If we use shuffle in DataLoader, then SampleMode doesn't know \
            # where the "front" is - so, we shuffle last:
            np.random.shuffle(train_data_list)
            np.random.shuffle(check_data_list)
        return train_data_list, check_data_list, rand_seed

    @staticmethod
    def split_train_check_data( sample_mode: Union[SampleMode,str], fd_list: list,
                                train_check_ratio: float, shuffle: bool = True
                                ) -> Tuple[list, list, dict]:
        '''
        Split training data into train, check
        '''
        sample_mode = sample_mode.name if isinstance(sample_mode, Enum) else sample_mode
        #
        dl_lens = [len(i) for i in fd_list]
        dl_nums = [int(i * train_check_ratio) for i in dl_lens]
        #
        if sample_mode == SampleMode.FRONT.name:
            train_inds = [np.arange(0,num).tolist() for num in dl_nums]
            check_inds = [np.arange(num,length).tolist() \
                        for num, length in zip(dl_nums, dl_lens)]
        elif sample_mode == SampleMode.RANDOM.name:
            dl_arng = [np.arange(i) for i in dl_lens]
            train_inds = [np.random.choice(arng, num, replace=False).tolist() \
                        for arng, num in zip(dl_arng, dl_nums)]
            check_inds = [np.delete(arng, inds).tolist() \
                        for arng, inds in zip(dl_arng, train_inds)]
        else:
            raise SampleMode.Exception(f"Unknown sample mode: {sample_mode}.")
        #
        train_data_list, check_data_list, rand_seed = \
            DataGenerationMethods.construct_train_check_data_lists(
            fd_list=fd_list, train_inds=train_inds, check_inds=check_inds, shuffle=shuffle)
        #
        recovery_info = {"train_inds": train_inds, "check_inds": check_inds,
                        "rand_seed": rand_seed, "shuffle": shuffle}
        return train_data_list, check_data_list, recovery_info

    @staticmethod
    @abstractmethod
    def generate_dataloader_from_npz(*args, **kwargs):
        '''
        TODO
        '''
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def rebuild_training_data(*args, **kwargs):
        '''
        TODO
        '''
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def prepare_training_data(*args, **kwargs):
        '''
        TODO
        '''
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def test_nn_using_npz(*args, **kwargs):
        '''
        TODO
        '''
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def test_nn_using_mvect(*args, **kwargs):
        '''
        TODO
        '''
        raise NotImplementedError()
