#!/usr/bin/env python3
'''
Definition of helper dimension classes
'''
from __future__ import annotations
from typing import Tuple
from typing_extensions import Self
from pyaarapsi.vpr.classes.data.abstractdata import AbstractData

class ImageDimensions(AbstractData):
    '''
    Dimensions (width, height)
    '''
    def __init__(self, width: int = -1, height: int = -1) -> Self:
        self.populated = not (width == -1 and height == -1)
        if self.populated:
            assert isinstance(width, int) and width > 0
            assert isinstance(height, int) and height > 0
        self.width = width
        self.height = height
    #
    def populate(self, width: int, height: int) -> Self:
        '''
        Populate
        '''
        assert isinstance(width, int) and width > 0
        assert isinstance(height, int) and height > 0
        self.width = width
        self.height = height
        self.populated = True
    #
    def for_cv(self) -> Tuple[int, int]:
        '''
        Convert to order for cv2 operations
        The result will have width columns and height rows
        '''
        assert self.is_populated(), "ImageDimensions are not populated!"
        return (self.width, self.height)
    #
    def for_np(self) -> Tuple[int, int]:
        '''
        Convert to order for numpy operations
        The result will have width columns and height rows
        '''
        assert self.is_populated(), "ImageDimensions are not populated!"
        return (self.height, self.width)
    #
    def is_populated(self) -> bool:
        '''
        Check if contains data
        '''
        return self.populated
    #
    def save_ready(self) -> dict:
        '''
        Convert class instance into a dictionary, where all keys are class attribute names, and
        all values are ready for pickling.
        '''
        try:
            return {"width": self.width, "height": self.height}
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to encode.") from e
    #
    @staticmethod
    def from_save_ready(save_ready_dict: dict) -> Self:
        '''
        Convert the output of save_ready() back into a class instance.
        '''
        try:
            return ImageDimensions(width=save_ready_dict["width"], \
                                   height=save_ready_dict["height"])
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to decode.") from e
    #
    @staticmethod
    def from_cv(cv_shape) -> Self:
        '''
        Convert to ImageDimensions from a cv2 shape
        '''
        return ImageDimensions(cv_shape[0], cv_shape[1])
    #
    @staticmethod
    def from_np(np_shape) -> Self:
        '''
        Convert to ImageDimensions from a numpy shape
        '''
        return ImageDimensions(np_shape[1], np_shape[0])
    #
    def __repr__(self) -> str:
        if not self.is_populated():
            return "ImageDimensions(populated=False)"
        return f"ImageDimensions(w={self.width:d},h={self.height:d})"
    #
    def __eq__(self, other_dims: ImageDimensions) -> bool:
        return (self.width == other_dims.width) and (self.height == other_dims.height)
    #
    def __del__(self):
        del self.width
        del self.height
        del self.populated
