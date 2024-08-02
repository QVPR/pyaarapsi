#!/usr/bin/env python3
'''
This file provides a collections of functions for enumerations
'''
from enum import Enum
from typing import Type, List, Any, Union, Tuple, TypeVar, overload, Optional

def enum_contains(value: Any, enumtype: Type[Enum], wrap: bool = False) -> Union[bool, List[bool]]:
    '''
    Return true/false if the value exists within enum
    '''
    if not isinstance(value, list):
        for i in enumtype:
            if i.value == value:
                if wrap:
                    return [True]
            return True
    else:
        newvalues = [False] * len(value)
        for c, val in enumerate(value):
            for entry in enumtype:
                if entry.value == val:
                    newvalues[c] = True
                    break
        return newvalues
    return False

EnumBaseT = TypeVar("EnumBaseT", bound=Enum)

@overload
def enum_get(value: Any, enumtype: Type[EnumBaseT], wrap: bool = False, \
                allow_fail: bool = True) -> Union[EnumBaseT,None]: ...

@overload
def enum_get(value: List[Any], enumtype: Type[EnumBaseT], wrap: bool = False, \
                allow_fail: bool = True) -> Union[List[EnumBaseT],None]: ...

def enum_get(value: Union[Any, List[Any]], enumtype: Type[EnumBaseT], wrap: bool = False, \
                allow_fail: bool = True) -> Union[EnumBaseT,List[EnumBaseT],None]:
    '''
    Return enumtype corresponding to value if it exists (or return None)
    If allow_fail=False, raises ValueError instead of None.
    '''
    if not isinstance(value, list):
        for i in enumtype:
            if i.value == value or i.name == value:
                if wrap:
                    return [i]
                return i
    else:
        for c, val in enumerate(value):
            for i in enumtype:
                if i.value == val or i.name == val:
                    value[c] = i
                    break
        return value
    if not allow_fail:
        raise ValueError(f"Could not match value {value} to enum {enumtype}.")
    return None

def enum_value_options(enumtype: Type[Enum], skip: Optional[List[Enum]] = None) -> Tuple[list, str]:
    '''
    Return lists of an enumtype's values and a cleaned string variant for printing purposes
    '''
    if skip is None:
        skip = []
    elif isinstance(skip, Enum):
        skip = [skip]
    options = []
    options_text = []
    for i in enumtype:
        if i in skip:
            continue
        options.append(i.value)
        options_text.append(i.name)
    return options, str(options_text).replace('\'', '')

def enum_value(enum_in: Union[Enum, List[Enum]], wrap: bool = False) -> Union[Any, List[Any]]:
    '''
    Convert to values
    '''
    if isinstance(enum_in, list):
        return [i.value for i in enum_in]
    if wrap:
        return [enum_in.value]
    return enum_in.value

def enum_name(enum_in: Union[Enum, List[Enum]], wrap: bool = False) -> Union[str, List[str]]:
    '''
    Convert to string name/s
    '''
    if isinstance(enum_in, list):
        return [i.name for i in enum_in]
    if wrap:
        return [str(enum_in.name)]
    return str(enum_in.name)
