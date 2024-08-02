#!/usr/bin/env python3
'''
DataTypes enumeration
'''
from enum import Enum, unique

@unique
class DataTypes(Enum):
    '''
    Enum to define the type of data in self.data
    '''
    DEFAULT     = 0
    UNPROCESSED = 1
    PROCESSED   = 2
    #
    @classmethod
    class Exception(Exception):
        '''
        Bad usage
        '''
