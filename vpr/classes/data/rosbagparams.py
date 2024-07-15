#!/usr/bin/env python3
'''
Class to define unique rosbag extraction parameters
'''
from __future__ import annotations

import copy
from typing import Optional, Tuple, Any, Dict
from typing_extensions import Self

from pyaarapsi.vpr.classes.vprdescriptor import VPRDescriptor
from pyaarapsi.vpr.classes.dimensions import ImageDimensions
from pyaarapsi.vpr.classes.data.rosbagdatafilter import RosbagDataFilter, FilterType
from pyaarapsi.vpr.classes.data.filterlessrosbagparams import FilterlessRosbagParams
from pyaarapsi.vpr.classes.data.abstractdata import AbstractData
from pyaarapsi.core.argparse_tools import assert_iterable_instances
from pyaarapsi.core.enum_tools import enum_get

class RosbagParams(FilterlessRosbagParams):
    '''
    With support for filters
    '''
    def __init__(self) -> Self:
        super(RosbagParams, self).__init__()
        self.image_filters: Optional[Tuple[RosbagDataFilter]] = None
        self.feature_filters: Optional[Tuple[RosbagDataFilter]] = None
    #
    def populate(self, bag_name: str, img_topics: Tuple[str], odom_topics: Tuple[str],
                    vpr_descriptors: Tuple[VPRDescriptor], img_dims: ImageDimensions,
                    sample_rate: int, image_filters: Tuple[RosbagDataFilter],
                    feature_filters: Tuple[RosbagDataFilter]) -> Self: #
        '''
        Inputs:
        - bag_name: str type; name of rosbag to read.
        - img_topics: Tuple[str] type; names of rosbag image topics to process.
        - odom_topics: Tuple[str] type; names of rosbag odometry topics to process.
        - vpr_descriptors: Tuple[VPRDescriptor] type; types of VPR descriptors to process
        - img_dims: ImageDimensions type; dimensions to resize images to once loaded from rosbag.
        - sample_rate: int type; the sample rate in milli-Hertz to process a rosbag at.
        - image_filters: Tuple[RosbagDataFilter] type; filters to apply to images
        - feature_filters: Tuple[RosbagDataFilter] type; filters to apply to features
        '''
        super(RosbagParams, self).populate_base(bag_name=bag_name, img_topics=img_topics,
                                        odom_topics=odom_topics, vpr_descriptors=vpr_descriptors,
                                        img_dims=img_dims, sample_rate=sample_rate)
        self.populated = False
        assert_iterable_instances(image_filters, RosbagDataFilter, empty_ok=True, iter_type=tuple)
        assert_iterable_instances(feature_filters, RosbagDataFilter, empty_ok=True, iter_type=tuple)
        self.image_filters = copy.deepcopy(image_filters)
        self.feature_filters = copy.deepcopy(feature_filters)
        self.populated = True
        return self
    #
    def populate_from(self, params: RosbagParams) -> Self:
        '''
        Populate with another params source. Protects pointers to existing instances
        '''
        self.populate(bag_name=params.bag_name, img_topics=params.img_topics, \
            odom_topics=params.odom_topics, vpr_descriptors=params.vpr_descriptors, \
            img_dims=params.img_dims, sample_rate=params.sample_rate, \
            image_filters=params.image_filters, feature_filters=params.feature_filters)
        return self
    #
    def but_with(self, attr_changes: Dict[str, Any]) -> Self:
        '''
        Reconstruct a RosbagParams instance, but overwrite an attribute value
        '''
        reconstructed_params_dict = copy.deepcopy(self.to_dict())
        reconstructed_params_dict.pop("populated")
        for key, value in attr_changes.items():
            reconstructed_params_dict[key] = value
        return self.__class__().populate(**reconstructed_params_dict)
    #
    def to_dict(self) -> dict:
        '''
        Convert to dictionary
        '''
        base_dict = super(RosbagParams, self).to_dict()
        base_dict.update(image_filters=self.image_filters, feature_filters=self.feature_filters)
        return base_dict
    #
    def save_ready(self) -> dict:
        '''
        Convert class instance into a dictionary, where all keys are class attribute names, and
        all values are ready for pickling.
        '''
        try:
            as_dict = super(RosbagParams, self).save_ready()
            as_dict["image_filters"] = [i.save_ready() for i in as_dict["image_filters"]]
            as_dict["feature_filters"] = [i.save_ready() for i in as_dict["feature_filters"]]
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
            return RosbagParams().populate(
                    bag_name = save_ready_dict["bag_name"],
                    img_topics = tuple(save_ready_dict["img_topics"]),
                    odom_topics = tuple(save_ready_dict["odom_topics"]),
                    vpr_descriptors = tuple(enum_get(save_ready_dict["vpr_descriptors"],
                                                     VPRDescriptor)),
                    img_dims = ImageDimensions.from_save_ready(save_ready_dict["img_dims"]),
                    sample_rate=save_ready_dict["sample_rate"],
                    image_filters=tuple(FilterType.make_filter(i) \
                                    for i in save_ready_dict["image_filters"]),
                    feature_filters=tuple(FilterType.make_filter(i) \
                                    for i in save_ready_dict["feature_filters"])
            )
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to decode.") from e
    #
    def __repr__(self) -> str:
        if not self.is_populated():
            return "RosbagParams(populated=False)"
        return f"RosbagParams(bag_name={self.bag_name},num_topics=" \
                f"{len(self.img_topics)+len(self.odom_topics)},vpr_descriptors=" \
                f"{self.vpr_descriptors},img_dims={self.img_dims}," \
                f"sample_rate={self.sample_rate/1000}Hz,num_image_filters=" \
                f"{len(self.image_filters)},num_feature_filters={len(self.feature_filters)})"
    #
    def __eq__(self, other_params: RosbagParams) -> bool:
        '''
        Compare parameters
        '''
        if not super(RosbagParams, self).__eq__(other_params):
            return False
        # Order matters for these two:
        if tuple(i.save_ready() for i in self.image_filters) \
            != tuple(i.save_ready() for i in other_params.image_filters):
            return False
        if tuple(i.save_ready() for i in self.feature_filters) \
            != tuple(i.save_ready() for i in other_params.feature_filters):
            return False
        # All parameters passed
        return True
    #
    def __del__(self):
        super(RosbagParams, self).__del__()
        del self.image_filters
        del self.feature_filters
        