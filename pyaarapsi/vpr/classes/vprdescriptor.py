#!/usr/bin/env python3
'''
Definition of VPRDescriptor enumeration
'''
from __future__ import annotations

from enum import Enum, unique
from typing import Union, List

from pyaarapsi.vpr.classes.descriptors.generic import DescriptorContainer
from pyaarapsi.vpr.classes.descriptors.netvlad import NetVLADContainer
from pyaarapsi.vpr.classes.descriptors.hybridnet import HybridNetContainer
from pyaarapsi.vpr.classes.descriptors.salad import SALADContainer
from pyaarapsi.vpr.classes.descriptors.apgem import APGEMContainer
from pyaarapsi.vpr.classes.dimensions import ImageDimensions

@unique
class VPRDescriptor(Enum):
    '''
    VPR Descriptors
    '''
    UNPROCESSED         = (1, "Unprocessed", None,               None, True,  False)
    SAD                 = (1, "SAD",         None,               None, True,  False)
    NORM                = (2, "Norm",        None,               None, True,  False)
    ROLLNORM            = (3, "RollNorm",    None,               None, True,  False)
    PATCHNORM           = (4, "PatchNorm",   None,               None, True,  False)
    NETVLAD             = (5, "NetVLAD",     NetVLADContainer,   4096, False, True )
    HYBRIDNET           = (6, "HybridNet",   HybridNetContainer, 4096, True,  True )
    SALAD               = (7, "SALAD",       SALADContainer,     8192, False, True )
    APGEM               = (8, "AP-GeM",      APGEMContainer,     2048, False, True )
    #
    def __init__(self, _, descriptor_name: str, container_class: Union[DescriptorContainer, None],
                 feature_length: int, is_spatially_related: bool, requires_init: bool
                 ) -> VPRDescriptor:
        self.descriptor_name = descriptor_name
        self.container_class = container_class
        self.feature_length = feature_length
        self.is_spatially_related = is_spatially_related
        self.requires_init = requires_init
    #
    def get_descriptor_name(self) -> str:
        '''
        Return visualization-friendly name
        '''
        return self.descriptor_name
    #
    def has_container_class(self) -> bool:
        '''
        Check if has a valid container class
        '''
        return self.container_class is not None
    #
    def get_container_class(self) -> Union[DescriptorContainer, None]:
        '''
        Return feature extraction container class
        '''
        return self.container_class
    #
    def has_feature_length(self) -> bool:
        '''
        Check if has a valid fixed feature length
        '''
        return self.feature_length is not None
    #
    def get_feature_length(self) -> Union[int, None]:
        '''
        Return fixed feature length
        '''
        return self.feature_length
    #
    def __repr__(self) -> str:
        return f"{self.__class__.__name__:s}.{self.name:s}"
    #
    @classmethod
    def _prepare_descriptor_lists(cls: VPRDescriptor) -> None:
        VPRDescriptor.Variables.DESCRIPTORS_WITH_CONTAINER = []
        VPRDescriptor.Variables.CONTAINERLESS_DESCRIPTORS = []
        for descriptor in cls:
            if descriptor.requires_init:
                VPRDescriptor.Variables.DESCRIPTORS_WITH_CONTAINER.append(descriptor)
            else:
                VPRDescriptor.Variables.CONTAINERLESS_DESCRIPTORS.append(descriptor)
    #
    @classmethod
    def descriptors_with_container(cls: VPRDescriptor) -> List[VPRDescriptor]:
        '''
        Get a list of VPRDescriptors that have a container requiring initialization
        '''
        if not VPRDescriptor.Variables.CONTAINERS_PROCESSED:
            VPRDescriptor._prepare_descriptor_lists()
        return list(VPRDescriptor.Variables.DESCRIPTORS_WITH_CONTAINER)
    #
    @classmethod
    def containerless_descriptors(cls: VPRDescriptor) -> List[VPRDescriptor]:
        '''
        Get a list of VPRDescriptors that do not have a container requiring initialization
        '''
        if not VPRDescriptor.Variables.CONTAINERS_PROCESSED:
            VPRDescriptor._prepare_descriptor_lists()
        return list(VPRDescriptor.Variables.CONTAINERLESS_DESCRIPTORS)
    #
    @staticmethod
    def calculate_feature_length(descriptor_type: VPRDescriptor, img_dims: ImageDimensions) -> int:
        '''
        Get length of vector per VPRDescriptor when used to generate features
        Although measurable after-the-fact, useful for allocating ahead of time.
        '''
        if descriptor_type.has_feature_length():
            return descriptor_type.get_feature_length()
        else:
            return img_dims.width * img_dims.height
    #
    @staticmethod
    class Variables:
        '''
        Holder for variables; avoids passing into VPRDescriptor's __init__
        '''
        CONTAINERS_PROCESSED = False
        CONTAINERLESS_DESCRIPTORS: List[VPRDescriptor] = []
        DESCRIPTORS_WITH_CONTAINER: List[VPRDescriptor] = []
    #
    @staticmethod
    class Exception(Exception):
        '''
        Bad usage.
        '''
