#!/usr/bin/env python3
'''
Data structure classes for interfacing with ROS and VPR
'''
from __future__ import annotations
import copy
from typing import Dict
from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray

from pyaarapsi.core.argparse_tools import assert_iterable_instances, assert_instance
from pyaarapsi.core.enum_tools import enum_get
from pyaarapsi.vpr.classes.vprdescriptor import VPRDescriptor
from pyaarapsi.vpr.classes.data.posearrayxyw import PoseArrayXYW
from pyaarapsi.vpr.classes.data.abstractdata import AbstractData
from pyaarapsi.vpr.classes.data.datatypes import DataTypes

class RosbagData(AbstractData):
    '''
    VPR Data
    '''
    def __init__(self) -> Self:
        self.positions: PoseArrayXYW = None
        self.velocities: PoseArrayXYW = None
        self.times: NDArray = None
        self.data: Dict[str, NDArray] = None
        self.data_type: DataTypes = None
        self.populated: bool = False
    #
    def populate(self, positions: PoseArrayXYW, velocities: PoseArrayXYW, times: NDArray,
                    data: Dict[str, NDArray], data_type: DataTypes) -> Self:
        '''
        Populate with data contents
        '''
        self.positions = copy.deepcopy(assert_instance(positions, PoseArrayXYW))
        self.velocities = copy.deepcopy(assert_instance(velocities, PoseArrayXYW))
        self.times = copy.deepcopy(assert_instance(times, np.ndarray))
        self.data = copy.deepcopy(assert_instance(data, dict))
        assert_iterable_instances(list(self.data.keys()), str, empty_ok=False)
        assert_iterable_instances(list(self.data.values()), np.ndarray, empty_ok=False)
        self.data_type = copy.deepcopy(assert_instance(data_type, DataTypes))
        self.populated = True
        return self
    #
    def to_singular_descriptor(self, descriptor: VPRDescriptor) -> Self:
        '''
        Extract a singular descriptor from an extended feature dataset
        '''
        assert descriptor.name in self.data, f"Descriptor ({descriptor.name}) not present"
        return RosbagData().populate(positions=self.positions, velocities=self.velocities,
                    times=self.times, data={descriptor.name: self.data[descriptor.name]},
                    data_type=self.data_type)
    #
    def is_populated(self) -> bool:
        '''
        Whether contents exist
        '''
        return self.populated
    #
    def to_dict(self) -> dict:
        '''
        Convert to dictionary
        '''
        if not self.populated:
            return {"positions": None, "velocities": None, "times": None, "data": None, \
                    "data_type": None, "populated": False}
        return {"positions": self.positions.save_ready(), \
                "velocities": self.velocities.save_ready(), \
                "times": self.times, "data": self.data, "data_type": self.data_type, \
                "populated": self.populated}
    #
    def populate_from(self, data: RosbagData) -> Self:
        '''
        Populate with another data source. Protects pointers to existing instances
        '''
        self.populate(positions=data.positions, velocities=data.velocities, times=data.times, \
                      data=data.data, data_type=data.data_type)
        return self
    #
    def save_ready(self) -> dict:
        '''
        Convert class instance into a dictionary, where all keys are class attribute names, and
        all values are ready for pickling.
        '''
        try:
            as_dict = self.to_dict()
            as_dict["data_type"] = as_dict["data_type"].name
            return as_dict
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to encode.") from e
    #
    @staticmethod
    def from_save_ready(save_ready_dict: dict) -> Self:
        '''
        Convert the output of save_ready() back into a class instance.
        '''
        try:
            return RosbagData().populate(positions=PoseArrayXYW.from_save_ready(\
                                                            save_ready_dict["positions"]),
                                     velocities=PoseArrayXYW.from_save_ready(\
                                                            save_ready_dict["velocities"]),
                                     times=save_ready_dict["times"],
                                     data=save_ready_dict["data"],
                                     data_type=enum_get(value=save_ready_dict["data_type"], \
                                                enumtype=DataTypes, wrap=False, allow_fail=False))
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to decode.") from e
    #
    def __repr__(self) -> str:
        if not self.is_populated():
            return "RosbagData(populated=False)"
        data_str = ','.join([f"{i}:{self.data[i].shape}" for i in self.data])
        return f"RosbagData(positions={self.positions},velocities={self.velocities}," \
                f"times={self.times.shape},data={{{data_str}}},data_type={self.data_type.name})"
    #
    def __del__(self):
        del self.positions
        del self.velocities
        del self.times
        del self.data
        del self.data_type
        del self.populated
