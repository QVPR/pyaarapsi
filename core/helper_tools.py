#!/usr/bin/env python3
'''
A collection of miscellaneous helper tools
'''
import select
import sys
import traceback
import pickle
import warnings
from enum import Enum, unique
# from threading import Timer as threadingTimer
from typing import Optional, Callable, TypeVar, Protocol, List, Literal, Any

from fastdist import fastdist
import numpy as np

import matplotlib
from matplotlib.backend_bases import FigureCanvasBase
import matplotlib.pyplot as plt

from numpy.typing import NDArray

@unique
class Bool(Enum):
    '''
    Bool with an optional unset value
    '''
    UNSET = -1
    FALSE = 0
    TRUE  = 1

    @staticmethod
    class Exception(Exception):
        '''
        Bad usage
        '''

class SupportsCanvas(Protocol):
    '''
    Whether it not it has a canvas attribute
    '''
    canvas: FigureCanvasBase

def check_if_ndarray_with_ndim(in_arr: Any, ndim: int) -> bool:
    '''
    Try-except wrapper to check if we have an ndarray with a specified ndim
    '''
    try:
        return in_arr.ndim == ndim
    except AttributeError:
        return False

def check_if_ndarray_with_ndim_or_more(in_arr: Any, ndim: int) -> bool:
    '''
    Try-except wrapper to check if we have an ndarray with at least a specified ndim
    '''
    try:
        return in_arr.ndim >= ndim
    except AttributeError:
        return False

def brandn(*args, mean: float = 0.5, scale: float = 6.0, fill_with_linear=True):
    '''
    Bounded normal random with smooth probability distribution between 0 and 1.
    '''
    x = mean + (np.random.randn(*args)/scale)
    input_was_array = isinstance(x, np.ndarray) # flag so we return same type
    if not input_was_array:
        x = np.array([x]) # convert scalar to array
    out_of_bounds = (x<0) | (x>1)
    if not fill_with_linear:
        while (_sum:=np.sum(out_of_bounds)) > 0:
            x[out_of_bounds] = brandn(_sum)
            out_of_bounds = (x<0) | (x>1)
    else:
        x[out_of_bounds] = np.random.rand(np.sum(out_of_bounds))
    if not input_was_array:
        x = x[0] # convert array back to scalar
    return x

def input_with_timeout(prompt: str, timeout: float) -> str:
    'https://stackoverflow.com/questions/15528939/time-limited-input'
    ready, _, _ = select.select([sys.stdin], [],[], timeout)
    if ready:
        return sys.stdin.readline().rstrip('\n') # expect stdin to be line-buffered
    print(prompt)
    return 'y'

def ask_yesnoexit(question: str, auto: Optional[float] = None) -> bool:
    """
    Helper to get yes / no / exit answer from user.
    """
    _yes = {'yes', 'y'}
    _no = {'no', 'n'}
    _exit = {'q', 'quit', 'e', 'exit'}
    done = False
    print(question)
    while not done:
        if auto is None:
            choice = input().lower()
        else:
            print(f'Download will proceed if no response within wait period ({str(auto)}s).')
            choice = input_with_timeout('Wait period elapsed, proceeding with download(s) ...',
                                        auto).lower()
        if choice in _yes:
            return True
        elif choice in _no:
            return False
        elif choice in _exit:
            sys.exit()
        else:
            print("Please respond '(y)es', '(n)o', or '(q)uit'.")
    return False

def perforate(  _len: int, \
                num_holes: Optional[int] = 3, \
                randomness: Optional[float] = 0.2, \
                hole_damage: Optional[float] = 0.5, \
                offset: Optional[float] = 0.0,
                _override: Optional[int] = 1) -> NDArray[np.uint16]:
    '''
    Generate perforated indices
    Generate indices from 0 up to _len, with randomness% removed and num_hole regions with 
        hole_damage% cut out.
    '''
    if _override:
        warnings.warn("[perforate] Override flag detected - no type checking will "
                      "be performed. Use at own risk.")
        if num_holes is None:
            num_holes = 3
        if randomness is None:
            randomness = 0.2
        if hole_damage is None:
            hole_damage = 0.5
        if offset is None:
            offset = 0.0
    else:
        assert _len > 10 and _len < 65536, 'Length must be an integer in range uint16 ' \
                                            'greater than 10 (10 < _len < 65536).'
        if num_holes is None:
            num_holes = 3
        else: assert num_holes < int(_len / 4), 'Too many holes for given length (must be ' \
                                                    'less than length / 4).'
        if randomness is None:
            randomness = 0.2
        else: assert randomness <= 0.5 and randomness >= 0, 'Random hole percentage should not ' \
                                                                'exceed 0.5.'
        if hole_damage is None:
            hole_damage = 0.5
        else: assert hole_damage <= 0.5, 'Hole damage percentage should not exceed 0.5.'
        if offset is None:
            offset = 0.0
        else: assert abs(offset) <= 1.0, 'Offset percentage cannot exceed -1 or 1 (-100% to 100%)'
    # mark random removal:
    out = np.arange(_len)
    _random_len = int(_len * randomness)
    random_damage = np.argpartition(np.random.rand(int(_len)), _random_len)[0:_random_len]
    out[random_damage] = -1
    # mark holes:
    partition_length = int((_len / num_holes))
    for i in range(int(num_holes)):
        out[(i+1)*partition_length-1 : \
            int((i+1)*(partition_length)-partition_length*hole_damage)-1 : -1] = -1
    # apply perforation:
    out = out[out != -1]
    # roll by offset:
    out = np.sort(((out + (offset*_len)).astype(int)) % _len)
    return out

def plt_pause(interval: float, fig: SupportsCanvas) -> None:
    '''
    Matplotlib Helper Function to handle drawing events
    Awesome for animating plots because it doesn't let matplotlib steal focus! :D
    '''
    backend = plt.rcParams['backend']
    if backend not in matplotlib.rcsetup.interactive_bk:
        return
    if fig.canvas.figure.stale: #type: ignore
        fig.canvas.draw()
    fig.canvas.start_event_loop(interval) #type: ignore

def roll(img: NDArray, i: int, fill: int = 0) -> NDArray:
    '''
    Custom roll, with direction consistent with np.roll. This implementation however replaces data
    with the fill value when it rolls outside of the array.
    '''
    if abs(i) > img.shape[1]:
        return np.ones(img.shape) * fill
    if i == 0:
        return img
    i = i * -1 # set direction to be consistent with np.roll
    _m = img[:,np.max([i, 0]):np.min([i+img.shape[1],img.shape[1]])]
    _e = np.ones((img.shape[0], np.max([i,0]))) * fill
    _s = np.ones((img.shape[0], img.shape[1] - np.min([i+img.shape[1],img.shape[1]]))) * fill
    img_out = np.concatenate([_s, _m, _e], axis=1)
    return np.array(img_out, dtype=img.dtype)

def m2m_dist(arr_1: NDArray, arr_2: NDArray, flatten: bool = False):
    '''
    Matrix to matrix distance calculation
    Wrapper for fastdist, to keep things simple.
    '''
    out = fastdist.matrix_to_matrix_distance(np.matrix(arr_1), np.matrix(arr_2),
                                             fastdist.euclidean, "euclidean")
    if flatten:
        out = out.flatten()
    return out

def p2p_dist_2d(xy1: NDArray, xy2: NDArray, _sqrt: bool = True):
    '''
    Point to point distance calculation
    '''
    _d = np.square(xy1[0]-xy2[0]) + np.square(xy1[1]-xy2[1])
    if _sqrt:
        return np.sqrt(_d)
    return _d

def try_load_var(path: str, var_name: str) -> object:
    '''
    Light-weight loader for save_var()
    '''
    try:
        return np.load(path+"/"+var_name+".npz", allow_pickle=True)[var_name]
    except FileNotFoundError:
        print(var_name + " missing.")
        return None

def save_var(path: str, var: object, var_name: str) -> None:
    '''
    Light-weight saver for try_load_var()
    '''
    np.savez(path+"/"+var_name, **{var_name: var})

FloatOrArrayOfFloatsT = TypeVar("FloatOrArrayOfFloatsT", float, np.ndarray)

def normalize_angle(angle: FloatOrArrayOfFloatsT) -> FloatOrArrayOfFloatsT:
    '''
    Normalize angle between [-pi, +pi]
    '''
    if isinstance(angle, np.ndarray):
        angle[angle>np.pi] = angle[angle>np.pi] - 2*np.pi
        angle[angle<-np.pi] = angle[angle<-np.pi] + 2*np.pi
        return angle
    else:
        if angle > np.pi:
            norm_angle = angle - 2*np.pi
        elif angle < -np.pi:
            norm_angle = angle + 2*np.pi
        else:
            norm_angle = angle
        return norm_angle

def angle_wrap(angle_in: float, mode: Literal['DEG', 'RAD'] = 'DEG') -> float:
    '''
    Wrap an angle after addition/subtraction to be the smallest equivalent
    Inputs:
    - angle_in: numeric
    - mode:     str type; either 'DEG' or 'RAD' for degrees or radians
    Returns:
    angle-wrapped numeric
    '''
    assert mode in ['DEG', 'RAD'], 'Mode must be either DEG or RAD.'
    if mode == 'DEG':
        return ((angle_in + 180.0) % 360) - 180
    return ((angle_in + np.pi) % (np.pi * 2)) - np.pi

def r2d(angle_in: float) -> float:
    '''
    Convert radians to degrees
    '''
    return angle_in * 180 / np.pi

def d2r(angle_in: float) -> float:
    '''
    Convert degrees to radians
    '''
    return angle_in * np.pi / 180

def np_ndarray_to_uint8_list(ndarray: NDArray) -> list:
    '''
    Convert any numpy ndarray into a list of uint8 representing byte information
    For use with transferring numpy ndarray data agnostic of dtype, shape
    '''
    byte_string = pickle.dumps(ndarray)
    uint8_list  = list(byte_string)
    return uint8_list

def uint8_list_to_np_ndarray(uint8_list: list) -> NDArray:
    '''
    Convert any list of uint8 representing byte information back into a numpy ndarray
    For use with transferring numpy ndarray data agnostic of dtype, shape
    '''
    byte_string = bytes(uint8_list)
    np_ndarray  = pickle.loads(byte_string)
    return np_ndarray

def format_exception(dump: bool = False) -> str:
    '''
    Print a terminal-friendly version of an exception
    https://www.adamsmith.haus/python/answers/how-to-retrieve-the-file,-line-number,\
        -and-type-of-an-exception-in-python
    '''
    exception_type, e, exception_traceback = sys.exc_info()
    if not exception_traceback is None:
        filename = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno
    else:
        filename = '<unknown>'
        line_number = '<unknown>'
    traceback_list = traceback.extract_tb(exception_traceback)
    if dump:
        traceback_string = str(traceback.format_exc())
    else:
        traceback_string = ""
        for c, i in enumerate(traceback_list):
            traceback_string += f"{str(i[2])} [{str(i[1])}]"
            if c < len(traceback_list) - 1:
                traceback_string += " >> "
    return f"Exception Caught.\n\tDetails: {str(exception_type)} {str(e)}\n\t" \
            f"File {str(filename)} [Line {str(line_number)}]\n\tTrace: {traceback_string}"

def get_array_statistics(arr: NDArray) -> str:
    '''
    Generate some stats about an array
    '''
    _shape  = str(np.shape(arr))
    _type   = str(type((arr.flatten())[0]))
    _min    = str(np.min(arr))
    _max    = str(np.max(arr))
    _mean   = str(np.mean(arr))
    _range  = str(np.max(arr) - np.min(arr))
    string_to_ret = f"{_shape}{_type} {_min}<{_mean}<{_max} [{_range}]"
    return string_to_ret

def combine_dictionaries(dicts: List[dict], cast: Callable = list) -> dict:
    '''
    Fuse two dictionaries together, key-wise, grouping keys by the cast operation.
    '''
    keys = []
    for d in dicts:
        keys.extend(list(d.keys()))
    # dict comprehension:
    return { k: # key
            cast(d[k] for d in dicts if k in d) # what the dict entry will be (tuple comprehension)
            for k in set(keys) # define iterations for k
            }

def get_num_decimals(num: float) -> int:
    '''
    Count number of decimals in a float
    '''
    return str(num)[::-1].find('.')

def vis_dict(item_in: dict, printer: Callable = print) -> str:
    '''
    Visualize a dictionary
    '''
    def sub_dict_struct(item_in, lvl, key):
        if lvl == 0:
            indent = ''
        else:
            indent = '\t'*lvl
        try:
            if isinstance(item_in, np.ndarray):
                _this_len = str(item_in.shape)
                item_in = item_in.flatten()
            else:
                _this_len = '(' + str(len(item_in)) + ',)' # if not iterable, will error here.
            _this_str = ""
            try:
                if isinstance(item_in, dict):
                    for sub_key in set(item_in.keys()):
                        _this_str += sub_dict_struct(item_in[sub_key], lvl + 1, sub_key)
                else:
                    _this_str += f"\t{indent}{type(item_in[0])}\n"
            except TypeError:
                _this_str = "\t{indent}[Unknown]\n"
            return f"{indent}{key} {type(item_in)} {_this_len}:\n{_this_str}"
        except TypeError:
            return f"{indent}{key} {type(item_in)}\n"
    dictionary_view = sub_dict_struct(item_in, 0, 'root')
    if not printer is None:
        printer(dictionary_view)
    return dictionary_view
