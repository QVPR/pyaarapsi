#!/usr/bin/env python3
'''
Class definition for SVM Parameters
'''
from __future__ import annotations

import copy
from enum import Enum, unique
from typing import Optional, Any, Tuple, Dict
from typing_extensions import Self

import numpy as np

from pyaarapsi.core.argparse_tools import assert_instance, assert_iterable_instances
from pyaarapsi.vpr.classes.data.rosbagparams import RosbagParams
from pyaarapsi.vpr.classes.data.abstractdata import AbstractData
from pyaarapsi.core.enum_tools import enum_get

@unique
class SVMToleranceMode(Enum):
    '''
    How to determine whether a match is in-tolerance
    '''
    DISTANCE            = 0
    FRAME               = 1
    TRACK_DISTANCE      = 2

    @staticmethod
    class Exception(Exception):
        '''
        Bad usage
        '''

class SVMParams(AbstractData):
    '''
    Uniquely define an SVM
    '''
    def __init__(self) -> Self:
        self.qry_params: Optional[RosbagParams] = None
        self.ref_params: Optional[RosbagParams] = None
        self.tol_mode: Optional[SVMToleranceMode] = None
        self.tol_thresh: Optional[float] = None
        self.factors: Optional[Tuple[str]] = None
        self.populated = False
    #
    def populate(self, ref_params: RosbagParams, qry_params: RosbagParams, \
                 tol_mode: SVMToleranceMode, tol_thresh: float, factors: Tuple[str]) -> Self: #
        '''
        Inputs:
        - ref_params:   RosbagParams type; unique parameters to define the reference rosbag
        - qry_params:   RosbagParams type; unique parameters to define the query rosbag
        - tol_mode:     SVMToleranceMode type; how to measure whether a match is in-tolerance
        - factors:      Tuple[str] type; which factors to extract
        Returns:
        - SVMParams (self)
        '''
        self.ref_params = assert_instance(ref_params, RosbagParams)
        assert self.ref_params.is_populated(), "ref_params is not populated!"
        self.qry_params = assert_instance(qry_params, RosbagParams)
        assert self.qry_params.is_populated(), "qry_params is not populated!"
        assert len(self.ref_params.vpr_descriptors) == 1, "SVM can only process one VPR descriptor"
        assert len(self.qry_params.vpr_descriptors) == 1, "SVM can only process one VPR descriptor"
        assert len(self.ref_params.img_topics) == 1, "SVM can only process one image topic"
        assert len(self.qry_params.img_topics) == 1, "SVM can only process one image topic"
        assert len(self.ref_params.odom_topics) == 1, "SVM can only process one odometry topic"
        assert len(self.qry_params.odom_topics) == 1, "SVM can only process one odometry topic"
        assert self.ref_params.vpr_descriptors == self.qry_params.vpr_descriptors, \
            "Reference and query VPR descriptors must be identical"
        self.tol_mode = assert_instance(tol_mode, SVMToleranceMode)
        self.tol_thresh = assert_instance(tol_thresh, float)
        self.factors = SVMParams.process_factors(factors_in=factors)
        self.populated = True
        return self
    #
    def populate_from(self, params: SVMParams) -> Self:
        '''
        Populate with another param set. Protects pointers to existing instances
        '''
        self.populate(ref_params=params.ref_params, qry_params=params.qry_params, \
                        tol_mode=params.tol_mode, tol_thresh=params.tol_thresh, \
                        factors=params.factors)
        return self
    #
    @staticmethod
    def process_factors(factors_in: Any) -> tuple:
        '''
        Ensure alphabetised, sorted, and a tuple.
        '''
        return tuple(np.sort(list(set(assert_iterable_instances(iterable_in=factors_in, \
                                            type_in=str, empty_ok=False, iter_type=tuple)))))
    #
    def num_factors(self) -> int:
        '''
        Helper to count number of factors
        '''
        assert self.is_populated(), "Params are not populated!"
        return len(self.factors)
    #
    def but_with(self, attr_changes: Dict[str, Any]) -> Self:
        '''
        Reconstruct a SVMParams instance, but overwrite an attribute value
        '''
        reconstructed_params_dict = copy.deepcopy(self.to_dict())
        reconstructed_params_dict.pop("populated")
        for key, value in attr_changes.items():
            reconstructed_params_dict[key] = value
        return self.__class__().populate(**reconstructed_params_dict)
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
        base_dict = {"ref_params": self.ref_params, "qry_params": self.qry_params, \
                        "tol_mode": self.tol_mode, "tol_thresh": self.tol_thresh, \
                        "populated": self.populated}
        return base_dict
    #
    def save_ready(self) -> dict:
        '''
        Convert class instance into a dictionary, where all keys are class attribute names, and
        all values are ready for pickling.
        '''
        try:
            as_dict = self.to_dict()
            as_dict["ref_params"] = as_dict["ref_params"].save_ready()
            as_dict["qry_params"] = as_dict["qry_params"].save_ready()
            as_dict["tol_mode"] = as_dict["tol_mode"].name
            # factors, tol_thresh are unchanged
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
            return SVMParams().populate( \
                ref_params=RosbagParams.from_save_ready( \
                    save_ready_dict=save_ready_dict["ref_params"]),
                qry_params=RosbagParams.from_save_ready( \
                    save_ready_dict=save_ready_dict["qry_params"]),
                tol_mode=enum_get(value=save_ready_dict["tol_mode"], enumtype=SVMToleranceMode, \
                                wrap=False), \
                tol_thresh=save_ready_dict["tol_thresh"], \
                factors=SVMParams.process_factors(save_ready_dict["factors"]) \
            )
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to decode.") from e
    #
    def __repr__(self) -> str:
        if not self.is_populated():
            return "SVMParams(populated=False)"
        return f"SVMParams(ref_params={self.ref_params},qry_params={self.qry_params}," \
                f"tol_mode={self.tol_mode},factors={self.factors})"
    #
    def __eq__(self, other_params: SVMParams) -> bool:
        '''
        Compare parameters
        '''
        if self.ref_params != other_params.ref_params:
            return False
        if self.qry_params != other_params.qry_params:
            return False
        if self.tol_mode.name != other_params.tol_mode.name:
            return False
        if self.tol_thresh != other_params.tol_thresh:
            return False
        if self.factors != other_params.factors:
            return False
        # All parameters passed
        return True
    #
    def __del__(self):
        del self.ref_params
        del self.qry_params
        del self.tol_mode
        del self.tol_thresh
        del self.factors
        del self.populated
        