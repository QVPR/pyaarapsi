#!/usr/bin/env python3
import time
import rospy
import sys
import traceback
import numpy as np
import pickle

def try_load_var(path, var_name):
    try:
        return np.load(path+"/"+var_name+".npz", allow_pickle=True)[var_name]
    except FileNotFoundError as e:
        print(var_name + " missing.")
        return None

def save_var(path, var, var_name):
    np.savez(path+"/"+var_name, **{var_name: var})

def angle_wrap(angle_in, mode='DEG'):
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

def np_ndarray_to_uint8_list(ndarray):
    '''
    Convert any numpy ndarray into a list of uint8 representing byte information
    For use with transferring numpy ndarray data agnostic of dtype, shape
    '''
    byte_string = pickle.dumps(ndarray)
    uint8_list  = [i for i in byte_string]
    return uint8_list

def uint8_list_to_np_ndarray(uint8_list):
    '''
    Convert any list of uint8 representing byte information back into a numpy ndarray
    For use with transferring numpy ndarray data agnostic of dtype, shape
    '''
    byte_string = bytes(uint8_list)
    np_ndarray  = pickle.loads(byte_string)
    return np_ndarray

class Timer:
    def __init__(self,rospy_on=False):
        self.points = []
        self.rospy_on = rospy_on
        self.add_bounds = False

    def add(self):
        self.points.append(time.perf_counter())
    
    def addb(self):
        self.add_bounds = True

    def calc(self, thresh=0.001):
        times = []
        for i in range(len(self.points) - 1):
            this_time = abs(self.points[i+1]-self.points[i])
            if this_time < thresh:
                this_time = 0.0
            times.append(this_time)
        if self.add_bounds and len(self.points) > 0:
            times.append(abs(self.points[-1] - self.points[0]))
        return times

    def show(self, name=None, thresh=0.001):
        times = self.calc(thresh)
        string = str(["%8.4f" % i for i in times]).replace(' ','')
        if not (name is None):
            string = "[" + name + "] " + string
        self.print(string)
        self.clear()

    def clear(self):
        self.points[:] = []
        self.add_bounds = False

    def print(self, string):
        if self.rospy_on:
            try:
                rospy.loginfo(string)
                return
            except:
                pass
        print(string)

def formatException(dump=False):
    # https://www.adamsmith.haus/python/answers/how-to-retrieve-the-file,-line-number,-and-type-of-an-exception-in-python
    exception_type, e, exception_traceback = sys.exc_info()
    filename = exception_traceback.tb_frame.f_code.co_filename
    line_number = exception_traceback.tb_lineno
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

def getArrayDetails(arr):
    _shape  = str(np.shape(arr))
    _type   = str(type((arr.flatten())[0]))
    _min    = str(np.min(arr))
    _max    = str(np.max(arr))
    _mean   = str(np.mean(arr))
    _range  = str(np.max(arr) - np.min(arr))
    string_to_ret = "%s%s %s<%s<%s [%s]" % (_shape, _type, _min, _mean, _max, _range)
    return string_to_ret

def combine_dicts(dicts, cast=list):
    keys = []
    for d in dicts:
        keys.extend(list(d.keys()))
    # dict comprehension:
    return { k: # key
            cast(d[k] for d in dicts if k in d) # what the dict entry will be (tuple comprehension)
            for k in set(keys) # define iterations for k
            }

def get_num_decimals(num):
    return str(num)[::-1].find('.')

def vis_dict(input, printer=print):
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
    printer(sub_dict_struct(input, 0, 'root'))
