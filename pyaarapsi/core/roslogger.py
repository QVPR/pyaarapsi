#!/usr/bin/env python3
'''
Custom re-implementation of rospy's logging classes for smoother integration between ROS and
non-ROS environments.
'''
import os
import inspect
import logging
import pickle
from hashlib import md5
from enum import Enum
from typing import Optional, Callable
from pyaarapsi.core.enum_tools import enum_value, enum_get
from pyaarapsi.core.vars import C_I_GREEN, C_I_YELLOW, C_I_RED, C_I_BLUE, C_RESET, C_CLEAR

##############################################################################
############# FROM ROSPY.CORE ################################################
############# vvvvvvvvvvvvvvv ################################################
##############################################################################

try:
    import rospy
    from rospy.core import _client_ready
    ROSPY_ACCESSIBLE = True
except ImportError:
    _client_ready = False
    ROSPY_ACCESSIBLE = False

def is_initialized():
    '''
    Check to see if the rospy client is ready 
    '''
    return _client_ready

class LoggingThrottle(object):
    '''
    Class to handle logging, prohibits message printing beyond a specified rate
    '''
    last_logging_time_table = {}
    def __call__(self, caller_id, period):
        """Do logging specified message periodically.

        - caller_id (str): Id to identify the caller
        - logging_func (function): Function to do logging.
        - period (float): Period to do logging in second unit.
        - msg (object): Message to do logging.
        """
        now = rospy.Time.now()

        last_logging_time = self.last_logging_time_table.get(caller_id)

        if (last_logging_time is None or
              (now - last_logging_time) > rospy.Duration(period)):
            self.last_logging_time_table[caller_id] = now
            return True
        elif last_logging_time > now:
            self.last_logging_time_table = {}
            self.last_logging_time_table[caller_id] = now
            return True
        return False


_logging_throttle = LoggingThrottle()

class LoggingIdentical(object):
    '''
    Class to handle logging, prohibits identical message printing
    '''
    last_logging_msg_table = {}

    def __call__(self, caller_id, msg):
        """Do logging specified message only if distinct from last message.

        - caller_id (str): Id to identify the caller
        - msg (str): Contents of message to log
        """
        msg_hash = md5(msg.encode()).hexdigest()

        if msg_hash != self.last_logging_msg_table.get(caller_id):
            self.last_logging_msg_table[caller_id] = msg_hash
            return True
        return False

_logging_identical = LoggingIdentical()

class LoggingOnce(object):
    '''
    Class to handle logging, prohibits repeated message printing
    '''
    called_caller_ids = set()
    def __call__(self, caller_id):
        if caller_id not in self.called_caller_ids:
            self.called_caller_ids.add(caller_id)
            return True
        return False

_logging_once = LoggingOnce()

def _frame_to_caller_id(frame):
    # from rospy.core
    caller_id = (
        inspect.getabsfile(frame),
        frame.f_lineno,
        frame.f_lasti,
    )
    return pickle.dumps(caller_id)

##############################################################################
############# ^^^^^^^^^^^^^^^ ################################################
############# FROM ROSPY.CORE ################################################
##############################################################################

class LogType(Enum):
    '''
    LogType Enumeration

    For use with roslogger
    '''

    ROS_DEBUG   = 1
    DEBUG       = 1.5
    INFO        = 2
    WARN        = 4
    ERROR       = 8
    FATAL       = 16

class LogLevel(Enum):
    '''
    Logging level for warning implementation
    '''
    ROS_DEBUG   = 10
    INFO        = 20
    WARN        = 30
    ERROR       = 40
    FATAL       = 50

class LoggerConversionError(Exception):
    '''
    In case of failed conversion
    '''

def log_type_to_level(_type):
    '''
    Convert LogType to LogLevel
    '''
    try:
        if not isinstance(_type, LogType):
            _type = enum_get(_type, LogType).name
        return enum_get(_type, LogLevel)
    except Exception as e:
        raise LoggerConversionError("Failed to convert LogType to LogLevel") from e

def log_level_to_type(_level):
    '''
    Convert LogLevel to LogType
    '''
    try:
        if not isinstance(_level, LogLevel):
            _level = enum_get(_level, LogLevel).name
        return enum_get(_level, LogType)
    except Exception as e:
        raise LoggerConversionError("Failed to convert LogLevel to LogType") from e

roslog_rospy_types = {1: 'debug', 2: 'info', 4: 'warn', 8: 'error', 16: 'critical'}
roslog_colours     = {1: C_I_GREEN, 1.5: C_I_BLUE, 2: '', 4: C_I_YELLOW, 8: C_I_RED, 16: C_I_RED}

def _roslogger(logfunc: Callable, logtype: LogType, ros: bool, text: str, no_stamp: bool = True):
    if ros:
        if no_stamp:
            text = '\r' + C_CLEAR + '[' + logtype.name + '] ' + text
        logfunc(text)
        return is_initialized()
    else:
        text = roslog_colours[logtype.value] + '[' + logtype.name + '] ' + text
        if no_stamp:
            text = '\r' + C_CLEAR + text  + C_RESET
        logfunc(text)
        return True

def roslogger(text, logtype: LogType = LogType.INFO, throttle: Optional[float] = None,
              ros: bool = True, name: Optional[str] = None, no_stamp: Optional[bool] = True,
              once: bool = False, throttle_identical: bool = False,
              log_level: Optional[LogType] = None) -> bool:
    '''
    Print function helper; overrides rospy.core._base_logger
    For use with integration with ROS
        This function seeks to exploit rospy's colouring and logging scheme, but add
        functionality such that a user can add a prefix, hide the stamp, and switch 
        quickly to work outside of a ROS node.

    Inputs:
    - text:                 text string to be printed, must be pre-formatted (can't be done 
                                inside roslogger)
    - logtype:              LogType enum type {default: LogType.INFO}; Define which print type 
                                (debug, info, etc...) is requested
    - throttle:             float type (default: 0); rate to limit publishing contents at if 
                                repeatedly executed
    - ros:                  bool type {default: True}; Whether to use rospy logging or default 
                                to print
    - name:                 str type {default: None}; If provided as str, prepends a label in the
                                format [name] to the text message (will appear after log level tags)
    - no_stamp:             bool type {default: True}; Whether to remove generic rospy timestamp
    - once:                 bool type {defualt: False}; Whether to only send once
    - throttle_identical:   bool type {default: False}; Whether to only throttle if message is 
                                identical
    - log_level:            LogType enumtype {default: None}; Can be passed to override global 
                                logging level
    Returns:
    bool type; True if print succeeded (else False)
    '''
    text = str(text) # just in case someone did something silly
    if isinstance(name, str):
        text = '[' + name + '] ' + text
    if not ROSPY_ACCESSIBLE:
        ros = False
    if no_stamp is None:
        no_stamp = False
    logfunc = print
    if ros:
        rospy_logger = logging.getLogger('rosout')
        if log_level is None:
            try:
                log_level = log_level_to_type(rospy_logger.level)
            except LoggerConversionError:
                log_level = LogType.INFO
        if enum_value(logtype) in roslog_rospy_types:
            logfunc = getattr(rospy_logger, roslog_rospy_types[int(logtype.value)])
        else:
            ros = False
    if not ros:
        if log_level is None:
            try:
                log_level = enum_get(os.environ['ROSLOGGER_LOGLEVEL'], LogType)
            except KeyError:
                log_level = LogType.INFO
        # if the requested log level is below the print threshold (i.e. it shouldn't be displayed):
        if enum_value(logtype) < enum_value(log_level):
            return False
    _currentframe = inspect.currentframe()
    assert _currentframe is not None
    assert _currentframe.f_back is not None
    caller_id = _frame_to_caller_id(_currentframe.f_back)
    if once:
        if _logging_once(caller_id):
            return _roslogger(logfunc, logtype, ros, text, no_stamp=no_stamp)
    elif throttle_identical:
        throttle_elapsed = False
        if throttle is not None:
            throttle_elapsed = _logging_throttle(caller_id, throttle)
        if _logging_identical(caller_id, text) or throttle_elapsed:
            return _roslogger(logfunc, logtype, ros, text, no_stamp=no_stamp)
    elif throttle:
        if _logging_throttle(caller_id, throttle):
            return _roslogger(logfunc, logtype, ros, text, no_stamp=no_stamp)
    else:
        return _roslogger(logfunc, logtype, ros, text, no_stamp=no_stamp)
    return False

def _base_logger(msg, args, kwargs, level: str, throttle: Optional[float] = None,
                 throttle_identical: bool = False, once: bool = False):
    rospy_logger = logging.getLogger('rosout')
    name = kwargs.pop('logger_name', None)
    if name:
        rospy_logger = rospy_logger.getChild(name)
    logfunc = getattr(rospy_logger, level)

    _currentframe = inspect.currentframe()
    assert _currentframe is not None
    assert _currentframe.f_back is not None
    caller_id = _frame_to_caller_id(_currentframe.f_back.f_back)

    if once:
        if _logging_once(caller_id):
            logfunc(msg, *args, **kwargs)
            return is_initialized()
    elif throttle_identical:
        throttle_elapsed = False
        if throttle is not None:
            throttle_elapsed = _logging_throttle(caller_id, throttle)
        if _logging_identical(caller_id, msg) or throttle_elapsed:
            logfunc(msg, *args, **kwargs)
            return is_initialized()
    elif throttle:
        if _logging_throttle(caller_id, throttle):
            logfunc(msg, *args, **kwargs)
            return is_initialized()
    else:
        logfunc(msg, *args, **kwargs)
        return is_initialized()
    return False



def logdebug(msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, level='debug')

def loginfo(msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, level='info')

def logwarn(msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, level='warning')

def logerr(msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, level='error')

def logfatal(msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, level='critical')



def logdebug_throttle(period, msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, throttle=period, level='debug')

def loginfo_throttle(period, msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, throttle=period, level='info')

def logwarn_throttle(period, msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, throttle=period, level='warn')

def logerr_throttle(period, msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, throttle=period, level='error')

def logfatal_throttle(period, msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, throttle=period, level='critical')



def logdebug_throttle_identical(period, msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, throttle=period, throttle_identical=True,
                 level='debug')

def loginfo_throttle_identical(period, msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, throttle=period, throttle_identical=True,
                 level='info')

def logwarn_throttle_identical(period, msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, throttle=period, throttle_identical=True,
                 level='warn')

def logerr_throttle_identical(period, msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, throttle=period, throttle_identical=True,
                 level='error')

def logfatal_throttle_identical(period, msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, throttle=period, throttle_identical=True,
                 level='critical')



def logdebug_once(msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, once=True, level='debug')

def loginfo_once(msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, once=True, level='info')

def logwarn_once(msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, once=True, level='warn')

def logerr_once(msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, once=True, level='error')

def logfatal_once(msg, *args, **kwargs):
    '''
    TODO
    '''
    return _base_logger(msg, args, kwargs, once=True, level='critical')
