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
from pyaarapsi.vpr.classes.vprdescriptor import VPRDescriptor
from pyaarapsi.vpr.classes.data.posearrayxyw import PoseArrayXYW
from pyaarapsi.vpr.classes.data.abstractdata import AbstractData

class RosbagData(AbstractData):
    '''
    VPR Data
    '''
    def __init__(self) -> RosbagData:
        self.positions: PoseArrayXYW = None
        self.velocities: PoseArrayXYW = None
        self.times: NDArray = None
        self.data: Dict[str, NDArray] = None
        self.populated: bool = False
    #
    def populate(self, positions: PoseArrayXYW, velocities: PoseArrayXYW, times: NDArray,
                 data: Dict[str, NDArray]) -> RosbagData:
        '''
        Populate with data contents
        '''
        self.positions = copy.deepcopy(assert_instance(positions, PoseArrayXYW))
        self.velocities = copy.deepcopy(assert_instance(velocities, PoseArrayXYW))
        self.times = copy.deepcopy(assert_instance(times, np.ndarray))
        self.data = copy.deepcopy(assert_instance(data, dict))
        assert_iterable_instances(list(self.data.keys()), str, empty_ok=False)
        assert_iterable_instances(list(self.data.values()), np.ndarray, empty_ok=False)
        self.populated = True
        return self
    #
    def to_singular_descriptor(self, descriptor: VPRDescriptor) -> Self:
        '''
        Extract a singular descriptor from an extended feature dataset
        '''
        assert descriptor.name in self.data, "Descriptor not present"
        return RosbagData().populate(positions=self.positions, velocities=self.velocities,
                    times=self.times, data={descriptor.name: self.data[descriptor.name]})
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
                    "populated": False}
        return {"positions": self.positions.save_ready(), \
                "velocities": self.velocities.save_ready(), \
                "times": self.times, "data": self.data, "populated": self.populated}
    #
    def populate_from(self, data: RosbagData) -> Self:
        '''
        Populate with another data source. Protects pointers to existing instances
        '''
        self.populate(positions=data.positions, velocities=data.velocities, times=data.times, \
                      data=data.data)
        return self
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
            return RosbagData().populate(positions=PoseArrayXYW.from_save_ready(\
                                                            save_ready_dict["positions"]),
                                     velocities=PoseArrayXYW.from_save_ready(\
                                                            save_ready_dict["velocities"]),
                                     times=save_ready_dict["times"],
                                     data=save_ready_dict["data"])
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to decode.") from e
    #
    def __repr__(self) -> str:
        if not self.is_populated():
            return "RosbagData(populated=False)"
        data_str = ','.join([f"{i}:{self.data[i].shape}" for i in self.data])
        return f"RosbagData(positions={self.positions},velocities={self.velocities}," \
                f"self.times={self.times.shape},self.data={{{data_str}}})"
    #
    def __del__(self):
        del self.positions
        del self.velocities
        del self.times
        del self.data
        del self.populated
