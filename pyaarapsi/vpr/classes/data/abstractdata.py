#!/usr/bin/env python3
'''
Abstract class to define protocol methods for loading and saving
'''
from __future__ import annotations
from abc import ABC, abstractmethod
from typing_extensions import Self

class AbstractData(ABC):
    '''
    Abstract base class with load/save protocol methods
    '''
    @abstractmethod
    def save_ready(self) -> dict:
        '''
        Convert class instance into a dictionary, where all keys are class attribute names, and
        all values are ready for pickling.
        '''
        raise AbstractData.LoadSaveProtocolError()
    #
    @staticmethod
    @abstractmethod
    def from_save_ready(save_ready_dict: dict) -> Self:
        '''
        Convert the output of save_ready() back into a class instance.
        '''
        raise AbstractData.LoadSaveProtocolError()
    #
    @classmethod
    class LoadSaveProtocolError(Exception):
        '''
        Errors in AbstractData class
        '''
