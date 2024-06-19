#! /usr/bin/env python3
'''
General helpers; not specific to any action.
'''
import copy
from datetime import datetime
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

def get_rand_seed() -> int:
    '''
    Get a random seed using current clock
    '''
    stamp = datetime.now().timestamp()
    return int(stamp)

def fill_to_value(arr: NDArray, size: float, value: Optional[float] = 0) -> NDArray:
    '''
    Pad the end of a numpy 1D array with a constant value
    '''
    return_arr = np.ones(size) * value
    return_arr[:len(arr)] = arr
    return return_arr

def bin_search(min_val, max_val, criteria, iterations):
    '''
    Basic binary search algorithm.
    '''
    c       = 0
    smin    = min_val
    smax    = max_val
    found   = False
    while (not found) and (c < iterations):
        sval = int((smax+smin)/2.0)
        if criteria(sval):
            smin = sval
        else:
            smax = sval
        if smin == (smax-1):
            found = True
        c += 1
    if criteria(smin):
        return sval
    return smin

def inds_to_bool(inds: Union[NDArray, list], length: int) -> list:
    '''
    From an array of index, generate a boolean array for indexing.
    '''
    arr = np.array([False]*length)
    arr[inds] = True
    return arr.tolist()

def keys_insert(dict_in: dict, **kwargs):
    '''
    Insert **kwargs into dict_in
    '''
    dict_out = copy.deepcopy(dict_in)
    dict_out.update(kwargs)
    return dict_out

def subkeys_insert(dict_in: dict, dict_key: str, **kwargs):
    '''
    Insert **kwargs into dict_in[dict_key]
    '''
    assert dict_key in dict_in
    assert isinstance(dict_in[dict_key], dict)
    dict_out = copy.deepcopy(dict_in)
    dict_out[dict_key].update(kwargs)
    return dict_out

def throw(e: Exception = Exception()) -> None:
    '''
    Throw an exception. For use in comprehensions.
    '''
    raise e
