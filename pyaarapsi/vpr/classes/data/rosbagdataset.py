#!/usr/bin/env python3
'''
Class definition for RosbagDataset
'''
from __future__ import annotations

from enum import Enum
from typing import Union, Optional
from typing_extensions import Self

from numpy.typing import NDArray

from pyaarapsi.core.argparse_tools import assert_instance
from pyaarapsi.vpr.classes.vprdescriptor import VPRDescriptor
from pyaarapsi.vpr.classes.data.rosbagdata import RosbagData
from pyaarapsi.vpr.classes.data.rosbagparams import RosbagParams
from pyaarapsi.vpr.classes.data.abstractdata import AbstractData

class RosbagDataset(AbstractData):
    '''
    Holder for dataset parameters and data
    '''
    def __init__(self) -> RosbagDataset:
        self.params = RosbagParams()
        self.data = RosbagData()
        self.populated = False
    #
    def populate(self, params: RosbagParams, data: RosbagData) -> RosbagDataset:
        '''
        Provide params and data
        '''
        assert_instance(params, RosbagParams)
        assert_instance(data, RosbagData)
        assert params.is_populated() and data.is_populated() # avoid type error
        self.params.populate_from(params)
        self.data.populate_from(data)
        self.populated = True
        return self
    #
    def populate_from(self, dataset: RosbagDataset) -> Self:
        '''
        Populate with another dataset source. Protects pointers to existing instances
        '''
        self.populate(params=dataset.params, data=dataset.data)
        return self
    #
    def to_singular_descriptor(self, descriptor: VPRDescriptor) -> RosbagDataset:
        '''
        Extract a singular descriptor from an extended feature dataset
        '''
        assert descriptor in self.params.vpr_descriptors, "Descriptor not present"
        return RosbagDataset().populate(
            params=self.params.to_descriptor(descriptor=descriptor),
            data=self.data.to_singular_descriptor(descriptor=descriptor)
        )
    #
    def get_params(self) -> RosbagParams:
        '''
        Params getter
        '''
        return self.params
    #
    def get_data(self) -> RosbagData:
        '''
        Data getter
        '''
        return self.data
    #
    def is_populated(self) -> bool:
        '''
        Whether contents exist
        '''
        return self.populated
    #
    def pxyw_of(self, topic_name: Optional[str] = None) -> NDArray:
        '''
        Return position xyw data
        topic_name is only optional for data with only one odom_topics.
        '''
        if not self.is_populated():
            raise self.NotPopulatedException("Dataset not populated.")
        try:
            if topic_name is not None:
                topic_index = self.params.odom_topics.index(topic_name)
            else:
                assert len(self.params.odom_topics) == 1, \
                    "topic_name is only optional for data with only one odom_topics!"
                topic_index = 0
            return self.data.positions.xyw[:,topic_index]
        except Exception as e:
            raise self.DataAccessError(f"Failed to access ({topic_name}) of stored positions.") \
                from e
    #
    def vxyw_of(self, topic_name: Optional[str] = None) -> NDArray:
        '''
        Return velocity xyw data
        topic_name is only optional for data with only one odom_topics.
        '''
        if not self.is_populated():
            raise self.NotPopulatedException("Dataset not populated.")
        try:
            if topic_name is not None:
                topic_index = self.params.odom_topics.index(topic_name)
            else:
                assert len(self.params.odom_topics) == 1, \
                    "topic_name is only optional for data with only one odom_topics!"
                topic_index = 0
            return self.data.velocities.xyw[:,topic_index]
        except Exception as e:
            raise self.DataAccessError(f"Failed to access ({topic_name}) of stored velocities.") \
                from e
    #
    def data_of(self, descriptor_key: Optional[Union[VPRDescriptor, str]] = None, \
                topic_name: Optional[str] = None) -> NDArray:
        '''
        Return data corresponding to a descriptor and respective topic
        descriptor_key is only optional for data with only one key.
        topic_name is only optional for data with only one img_topic.
        '''
        if not self.is_populated():
            raise self.NotPopulatedException("Dataset not populated.")
        try:
            if descriptor_key is not None:
                descriptor_key_name = descriptor_key.name if isinstance(descriptor_key, Enum) \
                                            else descriptor_key
            else:
                keys_tuple = tuple(self.data.data.keys())
                assert len(keys_tuple) == 1, \
                    "descriptor_key is only optional for data with only one key!"
                descriptor_key_name = keys_tuple[0]
            if topic_name is not None:
                topic_index = self.params.img_topics.index(topic_name)
            else:
                assert len(self.params.img_topics) == 1, \
                    "topic_name is only optional for data with only one img_topic!"
                topic_index = 0
            return self.data.data[descriptor_key_name][:,topic_index]
        except Exception as e:
            raise self.DataAccessError(f"Failed to access ({descriptor_key_name},{topic_name}) "
                                       "of stored data.") from e
    #
    def to_dict(self) -> dict:
        '''
        Convert to dictionary
        '''
        return {'params': self.params.to_dict(), 'data': self.data.to_dict()}
    #
    def save_ready(self) -> dict:
        '''
        Convert class instance into a dictionary, where all keys are class attribute names, and
        all values are ready for pickling.
        '''
        try:
            return {'params': self.params.save_ready(), 'data': self.data.save_ready()}
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to encode.") from e
    #
    @staticmethod
    def from_save_ready(save_ready_dict: dict) -> Self:
        '''
        Convert the output of save_ready() back into a class instance.
        '''
        try:
            return RosbagDataset().populate(
                        params=RosbagParams.from_save_ready(save_ready_dict['params']),
                        data=RosbagData.from_save_ready(save_ready_dict['data']))
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to decode.") from e
    #
    def unload(self) -> dict:
        '''
        Clear out stored parameters and data
        '''
        del self.params
        del self.data
        self.params = RosbagParams()
        self.data = RosbagData()
        self.populated = False
        return self
    #
    def __repr__(self) -> str:
        if not self.is_populated():
            return "RosbagDataset(populated=False)"
        return f"RosbagDataset(data={{{self.data}}},params={{{self.params}}})"
    #
    def __del__(self):
        del self.params
        del self.data
        del self.populated
    #
    @classmethod
    class NotPopulatedException(Exception):
        '''
        Not populated
        '''
    #
    @classmethod
    class DataAccessError(Exception):
        '''
        Unable to access data
        '''
