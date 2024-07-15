#!/usr/bin/env python3
'''
A base class implementation for a ROS node
'''
import argparse as ap
import sys
from typing import Type, Optional, Any
import numpy as np
import genpy
import rospy

from std_msgs.msg import Header, String
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from aarapsi_robot_pack.msg import Debug

from pyaarapsi.core.ros_tools import ROSParamServer, Heartbeat, NodeState, SubscribeListener, \
set_rospy_log_lvl, ROSPublisher, q_from_yaw
from pyaarapsi.core.enum_tools import enum_get
from pyaarapsi.core.argparse_tools import check_bool, check_enum, check_int, check_positive_float, \
    check_string, check_positive_two_int_list, check_string_list
from pyaarapsi.core.roslogger import roslogger, LogType
from pyaarapsi.vpr.vpr_helpers import SVMToleranceMode
from pyaarapsi.vpr.classes.vprdescriptor import VPRDescriptor

def base_optional_args(parser: ap.ArgumentParser, node_name: str = 'node_name', rate: float = 10.0,
                       anon: bool = False, namespace: str = '/vpr_nodes', log_level: float = 2,
                       reset: bool = True, order_id: Optional[int] = None) -> ap.ArgumentParser:
    '''
    Shared node arguments to be configured
    '''
    parser.add_argument('--node-name', '-N',  type=check_string,                     \
        default=node_name, help="Specify node name (default: %(default)s).")
    parser.add_argument('--rate_num',  '-r',  type=check_positive_float,             \
        default=rate,      help='Specify node rate (default: %(default)s).')
    parser.add_argument('--anon',      '-a',  type=check_bool,                       \
        default=anon,      help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace', '-n',  type=check_string,                     \
        default=namespace, help="Specify ROS namespace (default: %(default)s).")
    parser.add_argument('--log-level', '-V',  type=lambda x: check_enum(x, LogType), \
        default=log_level, help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--reset',     '-R',  type=check_bool,                       \
        default=reset,     help='Force reset of parameters to specified (default: %(default)s)')
    parser.add_argument('--order-id',  '-ID', type=int,                              \
        default=order_id,  help='Specify boot order of pipeline nodes (default: %(default)s).')
    return parser

class SuperROSClass:
    '''
    Super-wrapper class container for rospy

    Bundles:
    - rospy.init_node
    - A ROS_Home instance, including a ROS_Parameter_Server instance and a heartbeat publisher
    - Assigns namespace, node_name, and nodespace
    - Optionally creates a subscriber and callback for debugging/diagnostics
    - Optionally handles launch control using order_id for sequencing
    - Optionally colours the init_node message (blue)
    '''

    def __init__(self, node_name: str, namespace: str, rate_num: float, anon: bool, \
                 log_level: LogType, order_id: Optional[int] = None, throttle: float = 30, \
                 colour: bool = True, debug: bool = True, disable_signals: bool = False, \
                 reset: bool = True, hb_topic: str = '/heartbeats'):
        '''
        Initialisation

        Inputs:
        - mrc:          class type; Main ROS Class to assign parameters to
        - node_name:    str type;   Name of node, used in rospy.init_node and nodespace
        - namespace:    str type;   rospy namespace
        - rate_num:     float type; ROS node execution rate
        - anon:         bool type;  Whether to run the node as anonymous
        - log_level:    int type;   Initial rospy log level
        - order_id:     int type {default: None}; The namespace/launch_step parameter value to
                            wait for before proceeding
        - throttle:     float type {default: 30}; Wait seconds before publishing rospy.DEBUG
                            launch_step wait status
        - colour:       bool type {default: True}; Whether or not to colour the launch message
        - debug:        bool type {default: True}; Whether or not to create a subscriber and
                            callback for debugging
        - hb_topic:     str type {default: /heartbeats}; Which topic to publish heartbeat
                            messages on
        Returns:
        None
        '''
        self.namespace   = namespace
        self.node_name   = node_name
        self.nodespace   = self.namespace + '/' + self.node_name
        if not isinstance(log_level, LogType):
            _log_level = enum_get(log_level, LogType)
            assert isinstance(_log_level, LogType)
        else:
            _log_level = log_level
        if reset:
            _log_level = enum_get(rospy.get_param(self.nodespace + '/log_level', _log_level.name),
                                  LogType)
            assert not _log_level is None
        #
        rospy.init_node(node_name, anonymous=anon, log_level=np.ceil(_log_level.value).astype(int),
                        disable_signals=disable_signals)
        self.pubs = {}
        self.params = ROSParamServer(printer=self.print)
        self.hb = Heartbeat(self.node_name, self.namespace, rate_num, node_state=NodeState.INIT,
                            hb_topic=hb_topic, server=self)
        if debug:
            self._debug_sub = rospy.Subscriber(self.namespace + '/debug', Debug, self.debug_cb,
                                               queue_size=1)
        if not order_id is None:
            launch_step = check_int(rospy.get_param(self.namespace + '/launch_step', order_id))
            while launch_step < order_id:
                if rospy.is_shutdown():
                    try:
                        self.exit()
                    finally:
                        sys.exit()
                self._print(f'{str(self.node_name)} waiting in line, position {str(order_id)}.',
                            LogType.DEBUG, throttle=throttle, log_level=log_level)
                rospy.sleep(0.2)
                launch_step = check_int(rospy.get_param(self.namespace + '/launch_step', order_id))
        if colour:
            self._print(f'\033[96mStarting {self.node_name} node.\033[0m', log_level=log_level)
        else:
            self._print(f'Starting {self.node_name} node.', log_level=log_level)

    def init_params(self, rate_num: float, log_level: float, reset: bool, *args, **kwargs) -> None:
        
        self.RATE_NUM               = self.params.add(self.nodespace + "/rate",                     rate_num,                       check_positive_float,                           force=reset)
        self.LOG_LEVEL              = self.params.add(self.nodespace + "/log_level",                log_level,                      lambda x: check_enum(x, LogType),               force=reset)

    def init_vars(self, *args, **kwargs) -> None:

        self.parameters_ready = True

    def init_rospy(self, *args, **kwargs) -> None:

        self.rate_obj        = rospy.Rate(self.RATE_NUM.get())
        self.params.add_sub(self.namespace + "/params_update", self.param_callback)
        self.sublis          = SubscribeListener(printer=self.print)

    def node_ready(self, order_id: Optional[int] = None) -> None:
        self.main_ready      = True
        if not order_id is None: 
            if rospy.get_param(self.namespace + '/launch_step') == order_id:
                rospy.set_param(self.namespace + '/launch_step', order_id + 1)

    def param_helper(self, msg: String) -> None:
        pass

    def param_callback(self, msg: String) -> None:
        self.parameters_ready = False
        if self.params.exists(msg.data):
            if not self.params.update(msg.data):
                self.print("Change to parameter [%s]; bad value." % msg.data, LogType.DEBUG)
        
            else:
                self.print("Change to parameter [%s]; updated." % msg.data, LogType.DEBUG)

                if msg.data == self.LOG_LEVEL.name:
                    set_rospy_log_lvl(self.LOG_LEVEL.get())
                elif msg.data == self.RATE_NUM.name:
                    self.rate_obj = rospy.Rate(self.RATE_NUM.get())

                self.param_helper(msg)
        else:
            self.print("Change to untracked parameter [%s]; ignored." % msg.data, LogType.DEBUG)
        self.parameters_ready = True

    def debug_cb(self, msg: Debug) -> None:
        if msg.node_name == self.node_name:
            try:
                pass
            except:
                self.print("Debug operation failed.", LogType.DEBUG)

    def add_pub(self, topic: str, data_class: Type[genpy.Message], queue_size: int = 1, latch: bool = False, 
                subscriber_listener: Optional[SubscribeListener] = None) -> ROSPublisher:
        '''
        Add new ROS_Publisher

        Inputs:
        - topic:                string topic
        - data_class:           ROS data class
        - queue_size:           integer number of messages to store for publishing
        - latch:                bool True/False
        - subscriber_listener:  rospy.SubscribeListener class
        Returns:
        Generated ROS_Publisher
        '''

        self.pubs[topic] = ROSPublisher(topic, data_class, queue_size, latch, server=self, subscriber_listener=subscriber_listener)
        return self.pubs[topic]

    def set_state(self, state: NodeState) -> None:
        '''
        Set heartbeat node_state

        Inputs:
        - state:    NodeState enum type
        Returns:
        None
        '''
        self.hb.set_state(state)
    
    def _print(self, text: Any, logtype: LogType = LogType.INFO, throttle: float = 0, 
              ros: Optional[bool] = None, name: Optional[str] = None, no_stamp: Optional[bool] = None, log_level: Optional[LogType] = None) -> bool:
        if ros is None:
            ros = True
        if name is None:
            name = self.node_name
        if no_stamp is None:
            no_stamp = True
        if log_level is None:
            no_stamp = self.LOG_LEVEL.get()
        return roslogger(text, logtype, throttle=throttle, ros=ros, name=name, no_stamp=no_stamp, log_level=log_level)

    def print(self, text: Any, logtype: LogType = LogType.INFO, throttle: float = 0, 
              ros: Optional[bool] = None, name: Optional[str] = None, no_stamp: Optional[bool] = None, log_level: Optional[LogType] = None) -> bool:
        return self._print(text, logtype, throttle, ros, name, no_stamp, log_level)

    def exit(self) -> None:
        self.print("Quit received")
        sys.exit()

class BaseROSClass(SuperROSClass):
    '''
    VPR-wrapper class container for rospy

    Bundles:
    - Super_ROS_Class functionality
    - VPR Helper Functions
    '''
    def init_params(self, *args, **kwargs) -> None:
        super().init_params(*args, **kwargs)

        self.SIMULATION             = self.params.add(self.namespace + "/simulation",               False,                          check_bool,                                     force=False)

        self.FEAT_TYPE              = self.params.add(self.namespace + "/feature_type",             VPRDescriptor.SAD,              lambda x: check_enum(x, VPRDescriptor),         force=False)
        self.IMG_DIMS               = self.params.add(self.namespace + "/img_dims",                 [64,64],                        check_positive_two_int_list,                    force=False)
        self.NPZ_DBP                = self.params.add(self.namespace + "/npz_dbp",                  "",                             check_string,                                   force=False)
        self.BAG_DBP                = self.params.add(self.namespace + "/bag_dbp",                  "",                             check_string,                                   force=False)
        self.SVM_DBP                = self.params.add(self.namespace + "/svm_dbp",                  "",                             check_string,                                   force=False)
        self.IMG_TOPIC              = self.params.add(self.namespace + "/img_topic",                "",                             check_string,                                   force=False)
        self.SLAM_ODOM_TOPIC        = self.params.add(self.namespace + "/slam_odom_topic",          "",                             check_string,                                   force=False)
        self.ROBOT_ODOM_TOPIC       = self.params.add(self.namespace + "/robot_odom_topic",         "",                             check_string,                                   force=False)
        
        self.IMG_HFOV               = self.params.add(self.namespace + "/img_hfov",                 360,                            check_positive_float,                           force=False)
        
        self.REF_BAG_NAME           = self.params.add(self.namespace + "/ref/bag_name",             "",                             check_string,                                   force=False)
        self.REF_FILTERS            = self.params.add(self.namespace + "/ref/filters",              "{}",                           check_string,                                   force=False)
        self.REF_SAMPLE_RATE        = self.params.add(self.namespace + "/ref/sample_rate",          5.0,                            check_positive_float,                           force=False) # Hz

        self.SVM_QRY_BAG_NAME       = self.params.add(self.namespace + "/svm/qry/bag_name",         "",                             check_string,                                   force=False)
        self.SVM_QRY_FILTERS        = self.params.add(self.namespace + "/svm/qry/filters",          "{}",                           check_string,                                   force=False)
        self.SVM_QRY_SAMPLE_RATE    = self.params.add(self.namespace + "/svm/qry/sample_rate",      5.0,                            check_positive_float,                           force=False)

        self.SVM_REF_BAG_NAME       = self.params.add(self.namespace + "/svm/ref/bag_name",         "",                             check_string,                                   force=False)
        self.SVM_REF_FILTERS        = self.params.add(self.namespace + "/svm/ref/filters",          "{}",                           check_string,                                   force=False)
        self.SVM_REF_SAMPLE_RATE    = self.params.add(self.namespace + "/svm/ref/sample_rate",      5.0,                            check_positive_float,                           force=False)
        
        self.SVM_FACTORS            = self.params.add(self.namespace + "/svm/factors",              ['va','grad'],                  check_string_list,                              force=False)
        self.SVM_TOL_MODE           = self.params.add(self.namespace + "/svm/tolerance/mode",       SVMToleranceMode.DISTANCE,      lambda x: check_enum(x, SVMToleranceMode),      force=False)
        self.SVM_TOL_THRES          = self.params.add(self.namespace + "/svm/tolerance/threshold",  0.5,                            check_positive_float,                           force=False)
        
        self.REF_DATA_PARAMS        = [self.NPZ_DBP, self.BAG_DBP, self.REF_BAG_NAME, self.REF_FILTERS, self.REF_SAMPLE_RATE, self.IMG_TOPIC, self.SLAM_ODOM_TOPIC, self.FEAT_TYPE, self.IMG_DIMS]
        self.REF_DATA_NAMES         = [i.name for i in self.REF_DATA_PARAMS]

        self.SVM_DATA_PARAMS        = [self.FEAT_TYPE, self.IMG_DIMS, self.NPZ_DBP, self.BAG_DBP, self.SVM_DBP, self.IMG_TOPIC, self.SLAM_ODOM_TOPIC, \
                                       self.SVM_QRY_BAG_NAME, self.SVM_QRY_FILTERS, self.SVM_QRY_SAMPLE_RATE, \
                                       self.SVM_REF_BAG_NAME, self.SVM_REF_FILTERS, self.SVM_REF_SAMPLE_RATE, \
                                       self.SVM_FACTORS, self.SVM_TOL_MODE, self.SVM_TOL_THRES]
        self.SVM_DATA_NAMES         = [i.name for i in self.SVM_DATA_PARAMS]

    def init_vars(self, *args, **kwargs) -> None:
        super().init_vars(*args, **kwargs)

    def init_rospy(self, *args, **kwargs) -> None:
        super().init_rospy(*args, **kwargs)

    def debug_cb(self, msg: Debug) -> None:
        if msg.node_name == self.node_name:
            try:
                if msg.instruction == 0:
                    self.print(self.make_svm_model_params(), LogType.DEBUG)
                elif msg.instruction == 1:
                    self.print(self.make_dataset_dict(), LogType.DEBUG)
                else:
                    self.print(msg.instruction, LogType.DEBUG)
            finally:
                self.print("Debug operation failed.", LogType.DEBUG)

    def make_dataset_dict(self) -> dict:
        return dict(bag_name=self.REF_BAG_NAME.get(), npz_dbp=self.NPZ_DBP.get(), bag_dbp=self.BAG_DBP.get(), \
                    odom_topic=self.SLAM_ODOM_TOPIC.get(), img_topics=[self.IMG_TOPIC.get()], sample_rate=self.REF_SAMPLE_RATE.get(), \
                    ft_types=[self.FEAT_TYPE.get().name], img_dims=self.IMG_DIMS.get(), filters=self.REF_FILTERS.get())

    def make_svm_model_params(self) -> dict:
        qry_dict = dict(bag_name=self.SVM_QRY_BAG_NAME.get(), npz_dbp=self.NPZ_DBP.get(), bag_dbp=self.BAG_DBP.get(), \
                        odom_topic=self.SLAM_ODOM_TOPIC.get(), img_topics=[self.IMG_TOPIC.get()], sample_rate=self.SVM_QRY_SAMPLE_RATE.get(), \
                        ft_types=[self.FEAT_TYPE.get().name], img_dims=self.IMG_DIMS.get(), filters=self.SVM_QRY_FILTERS.get())
        ref_dict = dict(bag_name=self.SVM_REF_BAG_NAME.get(), npz_dbp=self.NPZ_DBP.get(), bag_dbp=self.BAG_DBP.get(), \
                        odom_topic=self.SLAM_ODOM_TOPIC.get(), img_topics=[self.IMG_TOPIC.get()], sample_rate=self.SVM_REF_SAMPLE_RATE.get(), \
                        ft_types=[self.FEAT_TYPE.get().name], img_dims=self.IMG_DIMS.get(), filters=self.SVM_REF_FILTERS.get())
        svm_dict = dict(factors=self.SVM_FACTORS.get(), tol_thres=self.SVM_TOL_THRES.get(), tol_mode=self.SVM_TOL_MODE.get().name)
        return dict(ref=ref_dict, qry=qry_dict, svm=svm_dict, npz_dbp=self.NPZ_DBP.get(), bag_dbp=self.BAG_DBP.get(), svm_dbp=self.SVM_DBP.get())

    def generate_path(self, dataset: dict, _every: int = 3) -> Path:
        '''
        Generate a nav_msgs/Path object using data set odometry
        '''
        px      = dataset['dataset']['px']
        py      = dataset['dataset']['py']
        pw      = dataset['dataset']['pw']
        time    = dataset['dataset']['time']
        new_path = Path(header=Header(stamp=rospy.Time.now(), frame_id="map"))
        new_path.poses = []
        for (c, (x, y, w, t)) in enumerate(zip(px, py, pw, time)):
            if not c % _every == 0:
                continue
            new_pose = PoseStamped(header=Header(stamp=rospy.Time.from_sec(t),
                                                 frame_id="map", seq=c))
            new_pose.pose.position = Point(x=x, y=y, z=0)
            new_pose.pose.orientation = q_from_yaw(w)
            new_path.poses.append(new_pose)
            del new_pose
        return new_path
