#!/usr/bin/env python3
'''
Class to define abstract class of unique rosbag extraction parameters
'''
from __future__ import annotations

import copy
from typing import Optional, Tuple, Union, Any, Dict
from typing_extensions import Self

from pyaarapsi.core.enum_tools import enum_get
from pyaarapsi.core.argparse_tools import assert_iterable_instances, assert_instance
from pyaarapsi.vpr.classes.vprdescriptor import VPRDescriptor
from pyaarapsi.vpr.classes.dimensions import ImageDimensions
from pyaarapsi.vpr.classes.data.abstractdata import AbstractData

class FilterlessRosbagParams(AbstractData):
    '''
    VPR Dataset Parameters
    '''
    #
    def __init__(self) -> Self:
        self.bag_name: Optional[str] = None
        self.img_topics: Optional[Tuple[str]] = None
        self.odom_topics: Optional[Tuple[str]] = None
        self.vpr_descriptors: Optional[Tuple[VPRDescriptor]] = None
        self.img_dims: Optional[ImageDimensions] = None
        self.sample_rate: Optional[int] = None
        self.populated: bool = False
    #
    def populate_base(self, bag_name: str, img_topics: Tuple[str], odom_topics: Tuple[str],
                        vpr_descriptors: Tuple[VPRDescriptor], img_dims: ImageDimensions,
                        sample_rate: int) -> Self:
        '''
        Inputs:
        - bag_name: str type; name of rosbag to read.
        - img_topics: Tuple[str] type; names of rosbag image topics to process.
        - odom_topics: Tuple[str] type; names of rosbag odometry topics to process.
        - vpr_descriptors: Tuple[VPRDescriptor] type; types of VPR descriptors to process
        - img_dims: ImageDimensions type; dimensions to resize images to once loaded from rosbag.
        - sample_rate: int type; the sample rate in milli-Hertz to process a rosbag at.
        '''
        self.bag_name = copy.deepcopy(assert_instance(bag_name, str))
        self.img_topics = copy.deepcopy(assert_iterable_instances(img_topics, \
                                            str, empty_ok=False, iter_type=tuple))
        self.odom_topics = copy.deepcopy(assert_iterable_instances(odom_topics, \
                                            str, empty_ok=False, iter_type=tuple))
        self.vpr_descriptors = copy.deepcopy(assert_iterable_instances(vpr_descriptors, \
                                            VPRDescriptor, empty_ok=False, iter_type=tuple))
        self.img_dims = copy.deepcopy(assert_instance(img_dims, ImageDimensions))
        self.sample_rate = copy.deepcopy(assert_instance(sample_rate, int))
        self.populated = True
        return self
    #
    def but_with(self, attr_changes: Dict[str, Any]) -> Self:
        '''
        Reconstruct an FilterlessRosbagParams instance, but overwrite an attribute value
        '''
        reconstructed_params_dict = copy.deepcopy(self.to_dict())
        reconstructed_params_dict.pop("populated")
        for key, value in attr_changes.items():
            reconstructed_params_dict[key] = value
        return self.__class__().populate_base(**reconstructed_params_dict)
    #
    def to_descriptor(self, descriptor: Union[VPRDescriptor, Tuple[VPRDescriptor]]
                      ) -> Self:
        '''
        Extract a singular descriptor from an extended feature dataset
        '''
        singular_dataset = copy.deepcopy(self)
        singular_dataset.vpr_descriptors = descriptor if isinstance(descriptor, tuple) \
                                                        else (descriptor,)
        return singular_dataset
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
        return {"bag_name": self.bag_name, "img_topics": self.img_topics, \
                "odom_topics": self.odom_topics, "vpr_descriptors": self.vpr_descriptors, \
                "img_dims": self.img_dims, "sample_rate": self.sample_rate, \
                "populated": self.populated}
    #
    def save_ready(self) -> dict:
        '''
        Convert class instance into a dictionary, where all keys are class attribute names, and
        all values are ready for pickling.
        '''
        try:
            as_dict = self.to_dict()
            as_dict["vpr_descriptors"] = [i.name for i in as_dict["vpr_descriptors"]]
            as_dict["img_dims"] = as_dict["img_dims"].save_ready()
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
            return FilterlessRosbagParams().populate_base(
                        bag_name = save_ready_dict["bag_name"],
                        img_topics = tuple(save_ready_dict["img_topics"]),
                        odom_topics = tuple(save_ready_dict["odom_topics"]),
                        vpr_descriptors = tuple(enum_get(save_ready_dict["vpr_descriptors"],
                                                        VPRDescriptor)),
                        img_dims = ImageDimensions.from_save_ready(save_ready_dict["img_dims"]),
                        sample_rate=save_ready_dict["sample_rate"]
            )
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to decode.") from e
    #
    def __repr__(self) -> str:
        if not self.is_populated():
            return "FilterlessRosbagParams(populated=False)"
        return f"FilterlessRosbagParams(bag_name={self.bag_name},num_topics=" \
                f"{len(self.img_topics)+len(self.odom_topics)},vpr_descriptors=" \
                f"{self.vpr_descriptors},img_dims={self.img_dims}," \
                f"sample_rate={self.sample_rate/1000}Hz)"
    #
    def __eq__(self, other_params: Self) -> bool:
        '''
        Compare parameters
        '''
        if self.bag_name != other_params.bag_name:
            return False
        # Order does not matter for these three:
        if set(self.img_topics) != set(other_params.img_topics):
            return False
        if set(self.odom_topics) != set(other_params.odom_topics):
            return False
        if set(i.name for i in self.vpr_descriptors) \
            != set(i.name for i in other_params.vpr_descriptors):
            return False
        if self.img_dims.for_cv() != other_params.img_dims.for_cv():
            return False
        if self.sample_rate != other_params.sample_rate:
            return False
        # All parameters passed
        return True
    #
    def __del__(self):
        del self.bag_name
        del self.img_topics
        del self.odom_topics
        del self.vpr_descriptors
        del self.img_dims
        del self.sample_rate
        del self.populated
