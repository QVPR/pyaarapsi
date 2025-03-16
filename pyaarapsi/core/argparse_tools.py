#!/usr/bin/env python3
'''
This file provides a collections of functions to parse (or raise an exception whilst trying to
parse) an input. Functions are named check_<desired output type>
'''
import argparse as ap
import string
from enum import Enum
from typing import List, Tuple, Any, Optional, Type, Union, TypeVar, overload
from pyaarapsi.core.enum_tools import enum_value_options


InstanceT = TypeVar("InstanceT")
TypeT = TypeVar("TypeT")

@overload
def assert_iterable_instances(iterable_in: List[InstanceT], type_in: Type,
                                empty_ok: bool = False, iter_type: Optional[Type] = None \
                                ) -> List[InstanceT]:
    ...

@overload
def assert_iterable_instances(iterable_in: Tuple[InstanceT], type_in: Type,
                                empty_ok: bool = False, iter_type: Optional[Type] = None \
                                ) -> Tuple[InstanceT]:
    ...

def assert_iterable_instances(iterable_in: Union[List[InstanceT], Tuple[InstanceT]], type_in: Type,
                                empty_ok: bool = False, iter_type: Optional[Type] = None \
                                ) -> Union[List[InstanceT], Tuple[InstanceT]]:
    '''
    Perform assertion on an iterable.
    Raises AssertionError on failure
    '''
    if iter_type is not None:
        assert_instance(instance_in=iterable_in, type_in=iter_type)
    if not empty_ok:
        assert len(iterable_in) > 0, "Iterable must have non-zero length."
    for i in iterable_in:
        assert_instance(instance_in=i, type_in=type_in)
    return iterable_in

def assert_instance(instance_in: InstanceT, type_in: Type) -> InstanceT: #, reload_ok: bool = True
    '''
    Perform assertion on an instance.
    Raises AssertionError on failure
    '''
    # if reload_ok:
    #     try:
    #         assert (instance_in.__class__ == type_in.__class__) \
    #             and (instance_in.__module__ == type_in.__module__), \
    #             f"Incorrect type, got: {instance_in.__module__}:{instance_in.__class__} " \
    #             f"(wanted: {type_in.__module__}:{type_in.__class__})"
    #     except AttributeError:
    #         return assert_instance(instance_in=instance_in, type_in=type_in, reload_ok=False)
    # else:
    assert isinstance(instance_in, type_in), \
            f"Incorrect type, got: {type(instance_in)} (wanted: {type_in})"
    return instance_in

def assert_subclass(type_in: TypeT, baseclass_in: Type) -> TypeT: #, reload_ok: bool = True
    '''
    Perform assertion on a type.
    Raises AssertionError on failure
    '''
    # if reload_ok:
    #     try:
    #         assert (instance_in.__class__ == type_in.__class__) \
    #             and (instance_in.__module__ == type_in.__module__), \
    #             f"Incorrect type, got: {instance_in.__module__}:{instance_in.__class__} " \
    #             f"(wanted: {type_in.__module__}:{type_in.__class__})"
    #     except AttributeError:
    #         return assert_instance(instance_in=instance_in, type_in=type_in, reload_ok=False)
    # else:
    assert issubclass(type_in, baseclass_in), \
            f"Incorrect type, got type with bases: {type_in.__bases__} (wanted: {baseclass_in})"
    return type_in

def check_int(value: Any) -> int:
    '''
    Parse an input value as an int.

    Inputs:
    - any
    Returns:
    - int
    '''
    error_text = f"{str(value)} is an invalid integer value."
    try:
        ivalue = int(value)
    except Exception as e:
        raise ap.ArgumentTypeError(error_text) from e
    return ivalue

def check_positive_int(value: Any) -> int:
    '''
    Parse an input value as a positive (>= 0) int.
    https://stackoverflow.com/questions/14117415/in-python-using-argparse-allow-
        only-positive-integers
    Inputs:
    - any
    Returns:
    - int
    '''
    error_text = f"{str(value)} is an invalid positive integer value."
    try:
        ivalue = int(value)
    except Exception as e:
        raise ap.ArgumentTypeError(error_text) from e
    if ivalue <= 0:
        raise ap.ArgumentTypeError(error_text)
    return ivalue

def check_float(value: Any) -> float:
    '''
    Parse an input as a float.

    Inputs:
    - any
    Returns:
    - float
    '''
    error_text = f"{str(value)} is an invalid float value."
    try:
        ivalue = float(value)
    except Exception as e:
        raise ap.ArgumentTypeError(error_text) from e
    return ivalue

def check_positive_float(value: Any) -> float:
    '''
    Parse an input as a positive (> 0) float.

    Inputs:
    - any
    Returns:
    - float
    '''
    error_text = f"{str(value)} is an invalid positive float value."
    try:
        ivalue = float(value)
    except Exception as e:
        raise ap.ArgumentTypeError(error_text) from e
    if ivalue <= 0:
        raise ap.ArgumentTypeError(error_text)
    return ivalue

def check_positive_or_zero_float (value: Any) -> float:
    '''
    Parse an input as a positive or zero (>= 0) float.

    Inputs:
    - any
    Returns:
    - float
    '''
    error_text = f"{str(value)} is an invalid positive or zero float value."
    try:
        ivalue = float(value)
    except Exception as e:
        raise ap.ArgumentTypeError(error_text) from e
    if ivalue < 0:
        raise ap.ArgumentTypeError(error_text)
    return ivalue

def check_bounded_float(value: Any, _min: float, _max: float, equality: str) -> float:
    '''
    Parse an input as a float bounded between two numbers
    Choose the bounding function from either:
        'lower': min <= val <  max
        'upper': min <  val <= max
        'both':  min <= val <= max
        'none':  min <  val <  max

    Inputs:
    - value:    any
    - _min:     float
    - _max:     float
    - equality: str of either 'lower', 'upper', 'both', or 'none
    Returns:
    - bounded float

    '''
    error_text = f"{str(value)} is either an invalid float value."
    error_text_bounds = "%s is not between the required bounds of {%s}."
    try:
        if _max < _min:
            raise ap.ArgumentTypeError('Invalid bounds; _max must be greater than _min')
        ivalue = float(value)
        if equality == 'lower':
            if not ((_min <= ivalue) and (ivalue < _max)):
                bounding_str = f"{str(_min)} <= val < {str(_max)}"
                raise ap.ArgumentTypeError(error_text_bounds % (str(value), bounding_str))
        elif equality == 'upper':
            if not ((_min < ivalue) and (ivalue <= _max)):
                bounding_str = f"{str(_min)} < val <= {str(_max)}"
                raise ap.ArgumentTypeError(error_text_bounds % (str(value), bounding_str))
        elif equality == 'both':
            if not ((_min <= ivalue) and (ivalue <= _max)):
                bounding_str = f"{str(_min)} <= val <= {str(_max)}"
                raise ap.ArgumentTypeError(error_text_bounds % (str(value), bounding_str))
        elif equality == 'none':
            if not ((_min < ivalue) and (ivalue < _max)):
                bounding_str = f"{str(_min)} < val < {str(_max)}"
                raise ap.ArgumentTypeError(error_text_bounds % (str(value), bounding_str))
        else:
            raise ap.ArgumentTypeError('Invalid equality mode {str(equality)}. Valid modes: '
                                       'lower, upper, both, none.')
        return ivalue
    except Exception as e:
        raise ap.ArgumentTypeError(error_text) from e

def check_bool(value: Any) -> bool:
    '''
    Parse an input as a boolean

    Inputs:
    - any
    Returns:
    - boolean
    '''
    error_text = f"{str(value)} is an invalid boolean."
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return not value == 0
    if isinstance(value, str):
        return value.lower() == "true"
    raise ap.ArgumentTypeError(error_text)

def check_positive_two_int_tuple(value: Any) -> Tuple[int, int]:
    '''
    Parse an input as a two-element positive-integer tuple

    Inputs:
    - any
    Returns:
    - tuple; two elements (positive integers)
    '''
    error_text = f"{str(value)} is an invalid positive two-integer tuple."
    str_value = str(value) # force to string
    value_list = str_value.replace(' ', '').replace('(','').replace(')','')\
                            .replace('[','').replace(']','').split(',')
    if not len(value_list) == 2:
        raise ap.ArgumentTypeError(error_text)
    if '.' in str(str_value):
        raise ap.ArgumentTypeError(error_text)
    try:
        ivalue = (int(value_list[0]), int(value_list[1]))
    except Exception as e:
        raise ap.ArgumentTypeError(error_text) from e
    if not (ivalue[0] > 0 and ivalue[1] > 0):
        raise ap.ArgumentTypeError(error_text)
    return ivalue

def check_positive_two_int_list(value: Any) -> List[int]:
    '''
    Parse an input as a two-element positive-integer list

    Inputs:
    - any
    Returns:
    - list; two elements (positive integers)
    '''
    error_text = f"{str(value)} is an invalid positive two-integer list."
    str_value = str(value) # force to string
    value_list = str_value.replace(' ', '').replace('(','').replace(')','').replace('[','') \
                            .replace(']','').split(',')
    if not len(value_list) == 2:
        raise ap.ArgumentTypeError(error_text)
    if '.' in str(str_value):
        raise ap.ArgumentTypeError(error_text)
    try:
        ivalue = [int(value_list[0]), int(value_list[1])]
    except Exception as e:
        raise ap.ArgumentTypeError(error_text) from e
    if not (ivalue[0] > 0 and ivalue[1] > 0):
        raise ap.ArgumentTypeError(error_text)
    return ivalue

def check_float_list(value: Any, _num: Optional[int] = None) -> List[float]:
    '''
    Parse an input as a list of floats

    Inputs:
    - value: any
    - _num:  int type {default: None}; number of elements in list (if None, doesn't check)
    Returns:
    - list of floats
    '''
    str_value = str(value) # force to string
    value_list = str_value.replace(' ', '').replace('(','').replace(')','').replace('[','') \
                            .replace(']','').split(',')
    if not _num is None and not len(value_list) == _num:
        raise ap.ArgumentTypeError(f"{str(value)} is an invalid list of floats: does not contain "
                                   "the correct number of elements.")
    try:
        ivalue = [float(i) for i in value_list]
    except Exception as e:
        raise ap.ArgumentTypeError(f"{str(value)} is an invalid list of floats: could not "
                                   "parse elements to float.") from e
    return ivalue

def check_valid_ip(value: Any) -> str:
    '''
    Parse an input as an IPv4 address.
    Note; this function does not check against all possible options. It checks:
    - if the input is 'localhost'
    - if the input is a collection of integers, separated by '.' i.e. 1.1.1.1

    Inputs:
    - any
    Returns:
    - str IP address
    '''
    error_text = f"{str(value)} is an invalid ip address."
    ip_raw = str(value)
    if ip_raw == 'localhost':
        return ip_raw
    ip_slice = ip_raw.split('.')
    if not len(ip_slice) == 4:
        raise ap.ArgumentTypeError(error_text)
    for num in ip_slice:
        try:
            int(num)
        except Exception as e:
            raise ap.ArgumentTypeError(error_text) from e
    return ip_raw

def check_string(value: Any) -> Union[str, None]:
    '''
    Parse an input as a string (str)

    Inputs:
    - any
    Returns:
    - str
    '''
    str_value = str(value)
    if str_value.lower() == 'none':
        return None
    return str_value

def check_string_list(value: Any) -> List[str]:
    '''
    Parse an input as a string (str) list

    Inputs:
    - any
    Returns:
    - list of str type elements
    '''
    error_text = f"{str(value)} is an invalid string list."
    try:
        if isinstance(value, list):
            return [str(i) for i in value]
        return str(value).replace('(','').replace(')','').replace('[','').replace(']','') \
                            .translate(str.maketrans('', '', string.whitespace)).split(',')
    except Exception as e:
        raise ap.ArgumentTypeError(error_text) from e

def check_enum_list(value: Any, enum: Type[Enum], skip: Optional[List[Enum]] = None) -> List[Enum]:
    '''
    Parse an input as a list of Enums

    Inputs:
    - value: any
    - enum:  Enum type to compare against
    - skip:  Enum values to ignore; if any enum within value is within this list, parsing will fail
    Returns:
    - list of Enum type elements
    '''
    if skip is None:
        skip = []
    error_text = f"{str(value)} is an invalid (or not accepted) list of identifiers within the " \
                    f"enumeration {str(enum)}"
    if isinstance(value, enum):
        value = [value]
    try:
        if not isinstance(value, list):
            value = str(value).replace('(','').replace(')','').replace('[','').replace(']','') \
                                .translate(str.maketrans('', '', string.whitespace)).split(',')
        all_good = True
        for i in value:
            if not (isinstance(i, enum) and not i in skip):
                all_good = False
                break
        if all_good:
            return value # already enum list, done
        # get enum information to compare against:
        enum_ids, enum_names = enum_value_options(enum, skip)
        enum_ids_str = [str(i) for i in enum_ids]
        enum_names_str = enum_names.replace('(','').replace(')','').replace('[','') \
                            .replace(']','').translate(str.maketrans('', '', string.whitespace)) \
                            .split(',')

        # at this point, we should have a list of strings.
        str_list = []
        for i in value:
            if i in enum_ids_str:
                str_list.append(enum[enum_names_str[enum_ids_str.index(i)]])
            elif i in enum_names_str:
                str_list.append(enum[enum_names_str[enum_names_str.index(i)]])
        return str_list
    except Exception as e:
        raise ap.ArgumentTypeError(error_text) from e

def check_enum(value: Any, enum: Type[Enum], skip: Optional[List[Enum]] = None) -> Enum:
    '''
    Parse an input an Enum

    Inputs:
    - value: any
    - enum:  Enum type to compare against
    - skip:  Enum values to ignore; if value is within this list, parsing will fail
    Returns:
    - Enum type
    '''
    if skip is None:
        skip = []
    error_text = f"{str(value)} is an invalid (or not accepted) identifier within the " \
                 f"enumeration {str(enum)}"
    if isinstance(value, enum) and not value in skip:
        return value
    try:
        str_value = str(value)
        enum_ids, enum_names = enum_value_options(enum, skip)
        enum_ids_str = [str(i) for i in enum_ids]
        enum_names_str = enum_names.replace('(','').replace(')','').replace('[','') \
                            .replace(']','').translate(str.maketrans('', '', string.whitespace)) \
                            .split(',')
        if str_value in enum_ids_str:
            index = enum_ids_str.index(str_value)
            #print("yes - id", index)
            return enum[enum_names_str[index]]
        if str_value in enum_names_str:
            index = enum_names_str.index(str_value)
            #print("yes - name", index)
            return enum[enum_names_str[index]]
    except Exception as e:
        raise ap.ArgumentTypeError(error_text) from e
