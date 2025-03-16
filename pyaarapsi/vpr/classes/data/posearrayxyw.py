#!/usr/bin/env python3
'''
Class to define an array of x, y, and yaw pose information
'''
from __future__ import annotations

import copy
from typing import Tuple, Union
from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray

from pyaarapsi.vpr.classes.data.abstractdata import AbstractData

class PoseArrayXYW(AbstractData):
    '''
    Class to handle packing x, y, and yaw data
    '''
    def __init__(self, x: Union[list, NDArray], y: Union[list, NDArray], w: Union[list, NDArray],
                 labels: Tuple[str]) -> Self:
        np_x = np.array(x)
        np_y = np.array(y)
        np_w = np.array(w)
        assert np_x.shape == np_y.shape == np_w.shape
        assert len(labels) == np_x.shape[1]
        self.labels = copy.deepcopy(labels)
        self._x = copy.deepcopy(np_x)
        self._y = copy.deepcopy(np_y)
        self._w = copy.deepcopy(np_w)
        self.xyw = np.stack([np_x,np_y,np_w], axis=2)
        self.shape = self.xyw.shape
    #
    def get(self, label: str, preserve_shape: bool = False) -> NDArray:
        '''
        Return components for a specific label
        '''
        assert label in self.labels, \
            f"Must provide a valid label, from user-defined list: {self.labels}"
        index_of_label = self.labels.index(label)
        if preserve_shape:
            return self[:, index_of_label:index_of_label+1, :]
        return self[:, index_of_label, :]
    #
    def to_dict(self) -> dict:
        '''
        Convert to dictionary
        '''
        return {"x": self._x, "y": self._y, "w": self._w, "labels": self.labels}
    #
    def __getitem__(self, item) -> tuple:
        return self.xyw[item]
    #
    def __len__(self):
        return len(self.xyw)
    #
    def x(self) -> NDArray:
        '''
        Getter for _x
        '''
        return self._x
    #
    def y(self) -> NDArray:
        '''
        Getter for _y
        '''
        return self._y
    #
    def w(self) -> NDArray:
        '''
        Getter for _w
        '''
        return self._w
    #
    def __del__(self):
        del self.labels
        del self._x
        del self._y
        del self._w
        del self.xyw
        del self.shape
    #
    def __repr__(self):
        return f"PoseArrayXYW(shape={self.shape},labels={self.labels})"
    #
    def save_ready(self) -> dict:
        '''
        Convert class instance into a dictionary, where all keys are class attribute names, and
        all values are ready for pickling.
        '''
        try:
            return self.to_dict()
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to encode.") from e
    #
    @staticmethod
    def from_save_ready(save_ready_dict: dict) -> Self:
        '''
        Convert the output of save_ready() back into a class instance.
        '''
        try:
            return PoseArrayXYW(x=save_ready_dict['x'], y=save_ready_dict['y'], \
                                w=save_ready_dict['w'], labels=save_ready_dict['labels'])
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to decode.") from e
