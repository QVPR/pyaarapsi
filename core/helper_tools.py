#!/usr/bin/env python3
import time
import sys
import traceback
import numpy as np
import pickle
from fastdist import fastdist
from enum import Enum
import matplotlib
from matplotlib.backend_bases import FigureCanvasBase
import matplotlib.pyplot as plt
from typing import Optional, Callable, TypeVar, Protocol

class Bool(Enum):
    UNSET = -1
    FALSE = 0
    TRUE  = 1

class SupportsCanvas(Protocol):
    canvas: FigureCanvasBase

def plt_pause(interval: float, fig: SupportsCanvas):
    '''
    Matplotlib Helper Function to handle drawing events
    Awesome for animating plots because it doesn't let matplotlib steal focus! :D
    '''
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        if fig.canvas.figure.stale: #type: ignore
            fig.canvas.draw()
        fig.canvas.start_event_loop(interval) #type: ignore
        return

def roll(img: np.ndarray, i: int, fill=0) -> np.ndarray:
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

def m2m_dist(arr_1, arr_2, flatten=False):
    out = fastdist.matrix_to_matrix_distance(np.matrix(arr_1), np.matrix(arr_2), fastdist.euclidean, "euclidean")
    if flatten:
        out = out.flatten()
    return out

def p2p_dist_2d(xy1, xy2, _sqrt=True):
    _d = np.square(xy1[0]-xy2[0]) + np.square(xy1[1]-xy2[1])
    if _sqrt:
        return np.sqrt(_d)
    return _d

def try_load_var(path: str, var_name: str) -> object:
    try:
        return np.load(path+"/"+var_name+".npz", allow_pickle=True)[var_name]
    except FileNotFoundError as e:
        print(var_name + " missing.")
        return None

def save_var(path: str, var: object, var_name: str) -> None:
    np.savez(path+"/"+var_name, **{var_name: var})

floatOrArrayOfFloats = TypeVar("floatOrArrayOfFloats", float, np.ndarray)

def normalize_angle(angle: floatOrArrayOfFloats) -> floatOrArrayOfFloats:
    # Normalize angle [-pi, +pi]
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

def angle_wrap(angle_in: float, mode: str = 'DEG') -> float:
    '''
    Wrap an angle after addition/subtraction to be the smallest equivalent
    Inputs:
    - angle_in: numeric
    - mode:     str type; either 'DEG' or 'RAD' for degrees or radians
    Returns:
    angle-wrapped numeric
    '''
    if mode == 'DEG':
        return ((angle_in + 180.0) % 360) - 180
    elif mode == 'RAD':
        return ((angle_in + np.pi) % (np.pi * 2)) - np.pi
    else:
        raise Exception('Mode must be either DEG or RAD.')
    
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

def np_ndarray_to_uint8_list(ndarray: np.ndarray) -> list:
    '''
    Convert any numpy ndarray into a list of uint8 representing byte information
    For use with transferring numpy ndarray data agnostic of dtype, shape
    '''
    byte_string = pickle.dumps(ndarray)
    uint8_list  = list(byte_string)
    return uint8_list

def uint8_list_to_np_ndarray(uint8_list: list) -> np.ndarray:
    '''
    Convert any list of uint8 representing byte information back into a numpy ndarray
    For use with transferring numpy ndarray data agnostic of dtype, shape
    '''
    byte_string = bytes(uint8_list)
    np_ndarray  = pickle.loads(byte_string)
    return np_ndarray

class Timer:
    def __init__(self, rospy_on: bool = False):
        self.points = []
        self.rospy_on = rospy_on
        self.add_bounds = False

    def add(self) -> None:
        self.points.append(time.perf_counter())
    
    def addb(self) -> None:
        self.add_bounds = True

    def calc(self, thresh: float = 0.001) -> list:
        times = []
        for i in range(len(self.points) - 1):
            this_time = abs(self.points[i+1]-self.points[i])
            if this_time < thresh:
                this_time = 0.0
            times.append(this_time)
        if self.add_bounds and len(self.points) > 0:
            times.append(abs(self.points[-1] - self.points[0]))
        return times

    def show(self, name: Optional[str] = None, thresh: float = 0.001) -> None:
        times = self.calc(thresh)
        string = str(["%8.4f" % i for i in times]).replace(' ','')
        if not (name is None):
            string = "[" + name + "] " + string
        self.print(string)
        self.clear()

    def clear(self) -> None:
        self.points[:] = []
        self.add_bounds = False

    def print(self, string: str, printer=print) -> None:
        if self.rospy_on:
            try:
                print(string)
                return
            except:
                pass
        print(string)

def formatException(dump: bool = False) -> str:
    # https://www.adamsmith.haus/python/answers/how-to-retrieve-the-file,-line-number,-and-type-of-an-exception-in-python
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
            traceback_string += "%s [%s]" % (str(i[2]), str(i[1]))
            if c < len(traceback_list) - 1:
                traceback_string += " >> "
    return "Exception Caught.\n\tDetails: %s %s\n\tFile %s [Line %s]\n\tTrace: %s" \
        % (str(exception_type), str(e), str(filename), str(line_number), traceback_string)

def getArrayDetails(arr: np.ndarray) -> str:
    _shape  = str(np.shape(arr))
    _type   = str(type((arr.flatten())[0]))
    _min    = str(np.min(arr))
    _max    = str(np.max(arr))
    _mean   = str(np.mean(arr))
    _range  = str(np.max(arr) - np.min(arr))
    string_to_ret = "%s%s %s<%s<%s [%s]" % (_shape, _type, _min, _mean, _max, _range)
    return string_to_ret

def combine_dictionaries(dicts: list, cast: Callable = list) -> dict:
    keys = []
    for d in dicts:
        keys.extend(list(d.keys()))
    # dict comprehension:
    return { k: # key
            cast(d[k] for d in dicts if k in d) # what the dict entry will be (tuple comprehension)
            for k in set(keys) # define iterations for k
            }

def get_num_decimals(num: float) -> int:
    return str(num)[::-1].find('.')

def vis_dict(input: dict, printer = print) -> str:
    def sub_dict_struct(input, lvl, key):
        if lvl == 0: indent = ''
        else: indent = '\t'*lvl
        try:
            if isinstance(input, np.ndarray):
                _this_len = str(input.shape)
                input = input.flatten()
            else: 
                _this_len = '(' + str(len(input)) + ',)' # if not iterable, will error here.
            _this_str = ""
            try:
                if isinstance(input, dict):
                    for sub_key in set(input.keys()):
                        _this_str += sub_dict_struct(input[sub_key], lvl + 1, sub_key)
                else:
                    _this_str += "\t%s%s\n" % (indent, type(input[0]))
            except:
                _this_str = "\t%s[Unknown]\n" % (indent)
            return "%s%s %s %s:\n%s" % (indent, key, type(input), _this_len, _this_str)
        except:
            return "%s%s %s\n" % (indent, key, type(input))
    dictionary_view = sub_dict_struct(input, 0, 'root')
    if not printer is None:
        printer(dictionary_view)
    return dictionary_view
