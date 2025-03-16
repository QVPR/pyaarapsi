#!/usr/bin/env python3
'''
Class definition for RosbagDataFilter
'''
from __future__ import annotations
import copy
from abc import abstractmethod
from typing import Optional, Any, List, Tuple
from typing_extensions import Self

import numpy as np

from pyaarapsi.core.helper_tools import perforate, m2m_dist
from pyaarapsi.vpr.classes.data.rosbagdata import RosbagData
from pyaarapsi.vpr.classes.data.filterlessrosbagparams import FilterlessRosbagParams
from pyaarapsi.vpr.classes.data.posearrayxyw import PoseArrayXYW
from pyaarapsi.vpr.classes.data.abstractdata import AbstractData
from pyaarapsi.pathing.basic import calc_path_stats

class RosbagDataFilter(AbstractData):
    '''
    Define a transformation
    '''
    @abstractmethod
    def apply(self, data: RosbagData, params: FilterlessRosbagParams) -> RosbagData:
        '''
        Apply filter to dataset
        '''
        raise NotImplementedError()
    #
    @abstractmethod
    def __repr__(self):
        raise NotImplementedError()
    #
    @abstractmethod
    def __del__(self):
        raise NotImplementedError()
    #
    def filter_by_indices(self, data: RosbagData, indices: Any) -> RosbagData:
        '''
        Generate RosbagData using indices/slice of an input
        '''
        return RosbagData().populate(
            positions =PoseArrayXYW(data.positions.x()[indices],
                                    data.positions.y()[indices],
                                    data.positions.w()[indices],
                                    data.positions.labels),
            velocities=PoseArrayXYW(data.velocities.x()[indices],
                                    data.velocities.y()[indices],
                                    data.velocities.w()[indices],
                                    data.velocities.labels),
            times = data.times[indices],
            data = {key: value[indices] for key, value in data.data.items()},
            data_type=data.data_type)

class DistanceFilter(RosbagDataFilter):
    '''
    Filter odometry to sample points every n metres
    '''
    def __init__(self, distance: float, odometry_topic: str) -> Self:
        self.distance = distance
        self.odometry_topic = odometry_topic
    #
    def apply(self, data: RosbagData, params: FilterlessRosbagParams) -> RosbagData:
        '''
        Apply filter to dataset
        '''
        assert data.is_populated()
        odometry_index = params.odom_topics.index(self.odometry_topic)
        xy_sum, xy_len = calc_path_stats(data.positions.xyw[:, odometry_index, 0:2])
        filt_indices = [np.argmin(np.abs(xy_sum-(self.distance*i))) \
                                    for i in np.arange(int((1/self.distance) * xy_len))]
        return self.filter_by_indices(data=data, indices=filt_indices)
    #
    def __repr__(self):
        return f"DistanceFilter (distance: {self.distance}, odometry_topic: {self.odometry_topic})"
    #
    def save_ready(self) -> dict:
        '''
        Convert class instance into a dictionary, where all keys are class attribute names, and
        all values are ready for pickling.
        '''
        try:
            return {"type": "DistanceFilter", "data": \
                {"distance": self.distance, "odometry_topic": self.odometry_topic}}
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to encode.") from e
    #
    @staticmethod
    def from_save_ready(save_ready_dict: dict) -> Self:
        '''
        Convert the output of save_ready() back into a class instance.
        '''
        try:
            return DistanceFilter(**save_ready_dict["data"])
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to decode.") from e
    #
    def __del__(self):
        del self.distance
        del self.odometry_topic

class PerforateFilter(RosbagDataFilter):
    '''
    Perforate data
    '''
    def __init__(self, randomness: Optional[float] = None, num_holes: Optional[int] = None,
                 hole_damage: Optional[float] = None, offset: Optional[float] = None,
                 _override: int = 0):
        self.randomness = randomness
        self.num_holes = num_holes
        self.hole_damage = hole_damage
        self.offset = offset
        self._override = _override
    #
    def apply(self, data: RosbagData, params: FilterlessRosbagParams) -> RosbagData:
        '''
        Apply filter to dataset
        '''
        assert data.is_populated()
        filt_indices = perforate(_len=len(data.times), num_holes=self.num_holes,
                                 randomness=self.randomness, hole_damage=self.hole_damage,
                                 offset=self.offset, _override=self._override)
        return self.filter_by_indices(data=data, indices=filt_indices)
    #
    def __repr__(self):
        return f"PerforateFilter (randomness: {self.randomness}, num_holes: {self.num_holes}, " \
                f"hole_damage: {self.hole_damage}, offset: {self.offset}, " \
                f"_override: {self._override})"
    #
    def save_ready(self) -> dict:
        '''
        Convert class instance into a dictionary, where all keys are class attribute names, and
        all values are ready for pickling.
        '''
        try:
            return {"type": "PerforateFilter", "data": \
                {"randomness": self.randomness, "num_holes": self.num_holes,
                "hole_damage": self.hole_damage, "offset": self.offset,
                "_override": self._override}}
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to encode.") from e
    #
    @staticmethod
    def from_save_ready(save_ready_dict: dict) -> Self:
        '''
        Convert the output of save_ready() back into a class instance.
        '''
        try:
            return PerforateFilter(**save_ready_dict["data"])
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to decode.") from e
    #
    def __del__(self):
        del self.randomness
        del self.num_holes
        del self.hole_damage
        del self.offset
        del self._override

class ForwardFilter(RosbagDataFilter):
    '''
    Forward-only data
    '''
    def __init__(self, odometry_topic: str):
        self.odometry_topic = odometry_topic
    #
    def apply(self, data: RosbagData, params: FilterlessRosbagParams) -> RosbagData:
        '''
        Apply filter to dataset
        '''
        assert data.is_populated()
        odometry_index = params.odom_topics.index(self.odometry_topic)
        filt_indices = [True if i >= 0 else False \
                        for i in data.velocities.x()[:,odometry_index]]
        return self.filter_by_indices(data=data, indices=filt_indices)
    #
    def __repr__(self):
        return f"ForwardFilter (odometry_topic: {self.odometry_topic})"
    #
    def save_ready(self) -> dict:
        '''
        Convert class instance into a dictionary, where all keys are class attribute names, and
        all values are ready for pickling.
        '''
        try:
            return {"type": "ForwardFilter", "data": \
                {"odometry_topic": self.odometry_topic}}
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to encode.") from e
    #
    @staticmethod
    def from_save_ready(save_ready_dict: dict) -> Self:
        '''
        Convert the output of save_ready() back into a class instance.
        '''
        try:
            return ForwardFilter(**save_ready_dict["data"])
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to decode.") from e
    #
    def __del__(self):
        del self.odometry_topic

class CropLoopFilter(RosbagDataFilter):
    '''
    Crop data to a loop
    '''
    def __init__(self, odometry_topic: str):
        self.odometry_topic = odometry_topic
    #
    def apply(self, data: RosbagData, params: FilterlessRosbagParams) -> RosbagData:
        '''
        Apply filter to dataset
        '''
        assert data.is_populated()
        odometry_index = params.odom_topics.index(self.odometry_topic)
        xy = copy.deepcopy(data.positions.xyw[:,odometry_index,0:2])
        xy_start = copy.deepcopy(xy[0:1,:])
        xy[0:int(data.positions.shape[0] / 2),:] = 1000
        filt_indices = slice(0, np.argmin(m2m_dist(xy_start, xy)))
        return self.filter_by_indices(data=data, indices=filt_indices)
    #
    def __repr__(self):
        return f"CropLoopFilter (odometry_topic: {self.odometry_topic})"
    #
    def save_ready(self) -> dict:
        '''
        Convert class instance into a dictionary, where all keys are class attribute names, and
        all values are ready for pickling.
        '''
        try:
            return {"type": "CropLoopFilter", "data": \
                {"odometry_topic": self.odometry_topic}}
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to encode.") from e
    #
    @staticmethod
    def from_save_ready(save_ready_dict: dict) -> Self:
        '''
        Convert the output of save_ready() back into a class instance.
        '''
        try:
            return CropLoopFilter(**save_ready_dict["data"])
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to decode.") from e
    #
    def __del__(self):
        del self.odometry_topic

class CropBoundsFilter(RosbagDataFilter):
    '''
    Crop data between bounds
    '''
    def __init__(self, start: Optional[int] = None, end: Optional[int] = None):
        self.start = start
        self.end = end
    #
    def apply(self, data: RosbagData, params: FilterlessRosbagParams) -> RosbagData:
        '''
        Apply filter to dataset
        '''
        assert data.is_populated()
        filt_indices = slice(self.start, self.end)
        return self.filter_by_indices(data=data, indices=filt_indices)
    #
    def __repr__(self):
        return f"CropBoundsFilter (start: {self.start}, end: {self.end})"
    #
    def save_ready(self) -> dict:
        '''
        Convert class instance into a dictionary, where all keys are class attribute names, and
        all values are ready for pickling.
        '''
        try:
            return {"type": "CropBoundsFilter", "data": \
                {"start": self.start, "end": self.end}}
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to encode.") from e
    #
    @staticmethod
    def from_save_ready(save_ready_dict: dict) -> Self:
        '''
        Convert the output of save_ready() back into a class instance.
        '''
        try:
            return CropBoundsFilter(**save_ready_dict["data"])
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to decode.") from e
    #
    def __del__(self):
        del self.start
        del self.end

class DeleteSegmentsFilter(RosbagDataFilter):
    '''
    Forward-only data
    '''
    def __init__(self, segments: List[Tuple[Optional[int], Optional[int]]]):
        self.segments = segments
    #
    def apply(self, data: RosbagData, params: FilterlessRosbagParams) -> RosbagData:
        '''
        Apply filter to dataset
        '''
        assert data.is_populated()
        filt_indices = np.ones(len(data.times), dtype=bool)
        for _start, _end in self.segments:
            filt_indices[_start:_end] = False
        return self.filter_by_indices(data=data, indices=filt_indices)
    #
    def __repr__(self):
        return f"DeleteSegmentsFilter (segments: {self.segments})"
    #
    def save_ready(self) -> dict:
        '''
        Convert class instance into a dictionary, where all keys are class attribute names, and
        all values are ready for pickling.
        '''
        try:
            return {"type": "DeleteSegmentsFilter", "data": \
                {"segments": self.segments}}
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to encode.") from e
    #
    @staticmethod
    def from_save_ready(save_ready_dict: dict) -> Self:
        '''
        Convert the output of save_ready() back into a class instance.
        '''
        try:
            return DeleteSegmentsFilter(**save_ready_dict["data"])
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to decode.") from e
    #
    def __del__(self):
        del self.segments
