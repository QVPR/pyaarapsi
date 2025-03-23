#!/usr/bin/env python3
'''
FilterType (and generative FilterTypeE) enumeration
'''
from __future__ import annotations
import os
import copy
import importlib
import warnings
from contextlib import suppress
from enum import Enum, unique
import json
from typing import Type, Dict, Optional
from typing_extensions import Self

from pyaarapsi.core.argparse_tools import assert_subclass, assert_instance
from pyaarapsi.vpr.classes.data.abstractdata import AbstractData
from pyaarapsi.vpr.classes.data.rosbagdatafilter import RosbagDataFilter

import pyaarapsi.vpr.classes.data as _root

DEFAULT_FILTERS: Dict[str, str] = { \
                    "DistanceFilter": "pyaarapsi.vpr.classes.data.rosbagdatafilter", \
                    "PerforateFilter": "pyaarapsi.vpr.classes.data.rosbagdatafilter", \
                    "ForwardFilter": "pyaarapsi.vpr.classes.data.rosbagdatafilter", \
                    "CropLoopFilter": "pyaarapsi.vpr.classes.data.rosbagdatafilter", \
                    "CropBoundsFilter": "pyaarapsi.vpr.classes.data.rosbagdatafilter", \
                    "DeleteSegmentsFilter": "pyaarapsi.vpr.classes.data.rosbagdatafilter" \
                }

CUSTOM_FILTERS: Dict[str, str] = {}
CUSTOM_FILTER_FILE_NAME = 'custom_filters.json'
CUSTOM_FILTER_FILE_PATH = _root.__path__[0] + '/' + CUSTOM_FILTER_FILE_NAME

def does_custom_filter_file_exist() -> bool:
    '''
    Check if config file exists
    '''
    return os.path.exists(CUSTOM_FILTER_FILE_PATH)

def make_custom_filter_file(overwrite: bool = True) -> bool:
    '''
    Try make custom filter file
    '''
    if overwrite:
        with suppress(OSError):
            os.remove(path=CUSTOM_FILTER_FILE_PATH)
    if not does_custom_filter_file_exist():
        with open(CUSTOM_FILTER_FILE_PATH, 'w', encoding="utf-8") as fp:
            json.dump(CUSTOM_FILTERS, fp)
    return does_custom_filter_file_exist()

def clear_custom_filters():
    '''
    Reset all custom filters
    '''
    CUSTOM_FILTERS.clear()
    with open(CUSTOM_FILTER_FILE_PATH, 'w', encoding="utf-8") as fp:
        json.dump({}, fp)

def print_custom_filter_file_error_help():
    '''
    Helper method to print instructions
    '''
    print(f"\tUnable to read {CUSTOM_FILTER_FILE_NAME}. Please ensure you have generated a " \
            "custom filter file.")
    print("\tTo generate a custom filter file, execute:")
    print("\t>>> from pyaarapsi.vpr.classes.data import filtertype")
    print("\t>>> filtertype.make_custom_filter_file()")

def get_custom_filters() -> Dict[str, str]:
    '''
    Load latest list of custom filters
    '''
    CUSTOM_FILTERS.clear()
    if not does_custom_filter_file_exist():
        print_custom_filter_file_error_help()
    else:
        with open(CUSTOM_FILTER_FILE_PATH, encoding="utf-8") as fp:
            CUSTOM_FILTERS.update(json.load(fp))
    return CUSTOM_FILTERS

def add_custom_filter(module_path: str, class_name: str, overwrite: bool = False) -> bool:
    '''
    Add a new custom filter
    '''
    if class_name in DEFAULT_FILTERS:
        message = f"[add_custom_filter] A filter with name \"{class_name}\" already exists " \
                        "in the default filter list."
        if overwrite:
            warnings.warn(message)
        else:
            raise ValueError(message)
    if class_name in CUSTOM_FILTERS:
        message = f"[add_custom_filter] A filter with name \"{class_name}\" already exists " \
                        "in the custom filter list."
        if overwrite:
            warnings.warn(message)
        else:
            raise ValueError(message)
    try:
        imported_module = importlib.import_module(module_path)
    except Exception as e:
        raise ValueError(f"Module \"{module_path}\" does not exist, or is not importable.") from e
    try:
        imported_class = getattr(imported_module, class_name)
    except Exception as e:
        raise ValueError(f"Module \"{module_path}\" exists, but does not have the importable " \
                         f"class \"{class_name}\".") from e
    try:
        assert_subclass(type_in=imported_class, baseclass_in=RosbagDataFilter)
    except Exception as e:
        raise ValueError(f"Class \"{class_name}\" of module \"{module_path}\" exists, but is " \
                            f"not a subclass of RosbagDataFilter (type: {type(imported_class)})." \
                                ) from e
    get_custom_filters() # also internally updates CUSTOM_FILTERS
    CUSTOM_FILTERS[class_name] = module_path
    return make_custom_filter_file(overwrite=True)

def build_filters():
    '''
    Build all filters
    '''
    make_custom_filter_file(overwrite=False)
    all_filters: Dict[str, str] = copy.deepcopy(DEFAULT_FILTERS)
    all_filters.update(copy.deepcopy(get_custom_filters()))
    built_filters = {key.upper(): (c, getattr(importlib.import_module(value), key)) \
                        for c, (key, value) in enumerate(all_filters.items())}
    return built_filters

def prep_filters():
    '''
    Prepare all filters to be built
    '''
    make_custom_filter_file(overwrite=False)
    all_filters: Dict[str, str] = copy.deepcopy(DEFAULT_FILTERS)
    all_filters.update(copy.deepcopy(get_custom_filters()))
    prepped_filters = {key.upper(): (c, key, value) \
                        for c, (key, value) in enumerate(all_filters.items())}
    return prepped_filters

@unique
class FilterTypeE(Enum):
    '''
    Filter types for RosbagParams. All classes extend RosbagDataFilter
    '''
    #
    # def __init__(self, _, cls: Type[RosbagDataFilter]):
    def __init__(self, _, cls_name, cls_path):
        # self.cls = assert_subclass(cls, RosbagDataFilter)
        self.cls_name = assert_instance(cls_name, str)
        self.cls_path = assert_instance(cls_path, str)
        self.cls: Optional[RosbagDataFilter] = None
    #
    def get_cls_name(self) -> str:
        '''
        Return class name
        '''
        return self.cls_name
    #
    def get_cls(self) -> Type[RosbagDataFilter]:
        '''
        Return class instance
        '''
        if self.cls is None:
            self.cls = getattr(importlib.import_module(self.cls_path), self.cls_name)
        return self.cls
    #
    @classmethod
    def find_filter_type(cls, cls_name: str) -> Self:
        '''
        Find matching entry
        '''
        for i in cls:
            if i.cls_name == cls_name:
                return i
        raise FilterTypeE.Exception("Failed to find a match")
    #
    @classmethod
    def find_cls(cls, cls_name: str) -> Type[RosbagDataFilter]:
        '''
        Find matching entry's class instance
        '''
        return cls.find_filter_type(cls_name).get_cls()
    #
    @classmethod
    def make_filter(cls, save_ready_dict: dict) -> RosbagDataFilter:
        '''
        Build a filter from saved data
        '''
        try:
            return cls.find_cls(cls_name=save_ready_dict["type"])(**save_ready_dict["data"])
        except (KeyError, AbstractData.LoadSaveProtocolError) as e:
            raise cls.Exception("Corrupt save_ready_dict, only has keys: " \
                                       f"{list(save_ready_dict.keys())}") from e
    #
    @classmethod
    class Exception(Exception):
        """
        Bad usage.
        """

FilterType = FilterTypeE("FilterType", prep_filters()) #pylint: disable=E1121
