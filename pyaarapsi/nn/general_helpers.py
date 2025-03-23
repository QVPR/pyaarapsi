#! /usr/bin/env python3
'''
General helpers; not specific to any action.
'''
import copy
from datetime import datetime
from typing import Optional, Union, Iterable, Callable, Any

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

def try_get(arr_in: Iterable, index_in: int, if_fail: Any = None) -> Any:
    '''
    Attempt to retrieve a positive index from an array ('arr_in') or a Callable wrapper around an
    array; if the index provided is negative or larger than the length of the iterable, then the
    function returns the 'if_fail' value
    '''
    if index_in < 0:
        return if_fail
    try:
        return arr_in(index_in)
    except (TypeError, IndexError):
        try:
            return arr_in[index_in]
        except (TypeError, IndexError):
            return if_fail

def bin_search_edge(left_bound: int, right_bound: int, criteria: Callable, \
                    max_iter: int = 20, _debug: bool = False) -> int:
    '''
    Perform a binary search to look for an edge. Provide some compute-expensive function 'criteria',
    which over the integer search range from left_bound to right_bound returns either 1 or 0. Find
    the first value in the search range which returns a value of 1 where that value + 1 returns a
    value of 0 (falling edge detection).
    '''
    assert right_bound > left_bound
    assert right_bound - left_bound > 1
    for _ in range(max_iter):
        middle = int((right_bound + left_bound) / 2)
        if criteria(middle) == 1:
            if _debug:
                print('left', left_bound, right_bound, middle)
            left_bound = middle
            try:
                if criteria(middle+1) == 0:
                    break
            except IndexError:
                break
        else:
            if _debug:
                print('right', left_bound, right_bound, middle)
            right_bound = middle
    if _debug:
        print(f'Finished -- result: {middle} [{try_get(criteria, middle-1)}, ' \
              f'{try_get(criteria, middle)}, {try_get(criteria, middle+1)}]')
    return middle

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
