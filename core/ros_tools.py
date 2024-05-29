#!/usr/bin/env python3
from __future__ import annotations

import rospy
from rospy.impl.rosout import _rospy_to_logging_levels
import genpy
import rosbag
import logging
import numpy as np
import genpy
import warnings
from enum import Enum

from tqdm.auto import tqdm

from std_msgs.msg               import String
from geometry_msgs.msg          import Quaternion, Pose, PoseWithCovariance, PoseStamped, Twist, TwistStamped

from tf.transformations         import quaternion_from_euler, euler_from_quaternion
from .helper_tools              import formatException
from .roslogger                 import LogType, roslogger
from .image_transforms          import *

from typing import List, Dict, Union, Optional, Callable, TypeVar, Generic, Generator, Tuple, Type

def import_rosmsg_from_string(msg_str: str) -> Type[genpy.Message]:
    '''
    Load a rosmsg class from a string

    Inputs:
    - msg_str: str type; import path to class i.e. nav_msgs.msg.Odometry
    Returns:
    base class of genpy.Message type; imported class
    
    '''
    _module_components = msg_str.split('.')
    _class_string  = _module_components[-1]
    _module_string = '.'.join(_module_components[:-1])
    _module = __import__(_module_string, fromlist=[_class_string])
    return getattr(_module, _class_string)

def open_rosbag(bag_path: str, topics: List[str]) -> Generator[Tuple[str, genpy.Message, rospy.rostime.Time], None, None]:
    '''
    Open a rosbag in read-mode as a generator and yield topics
    This method hides away some of the mean-ness of reading rosbags and guarantees type.

    Inputs:
    - bag_path:     str type; full path to rosbag
    - topics:       list type; list of strings where each string is a topic to extract from the rosbag
    Yields:
    Tuple[str, genpy.Message, rospy.rostime.Time] (contents: ROS topic, ROS message, ROS timestamp)
    '''
    ros_bag = rosbag.Bag(bag_path, 'r')

    for i in ros_bag.read_messages(topics=topics):
        assert isinstance(i, rosbag.bag.BagMessage)
        yield i[0], i[1], i[2]

def pose_covariance_to_stamped(pose: PoseWithCovariance, frame_id='map') -> PoseStamped:
    '''
    Convert a geometry_msgs/PoseWithCovariance to geometry_msgs/PoseStamped
    
    Inputs:
    - pose:     PoseWithCovariance type
    - frame_id: str type {default: 'map'}; specify PoseStamped header frame
    Returns:
    type PoseStamped
    '''
    out                 = PoseStamped(pose=pose.pose)
    out.header.stamp    = rospy.Time.now()
    out.header.frame_id = frame_id
    return out

def pose2xyw(pose: Union[Pose, PoseStamped]) -> List[float]:
    '''
    Extract x, y, and yaw from a geometry_msgs/Pose object

    Inputs:
    - pose:     geometry_msgs/Pose[Stamped] ROS message object
    Returns:
    type list; [x, y, yaw]
    '''
    if isinstance(pose, PoseStamped):
        pose = pose.pose
    return [pose.position.x, pose.position.y, yaw_from_q(pose.orientation)]

def twist2xyw(twist: Union[Twist, TwistStamped]) -> List[float]:
    '''
    Extract dx, dy, and dyaw from a geometry_msgs/Twist object

    Inputs:
    - twist:    geometry_msgs/Twist[Stamped] ROS message object
    Returns:
    type list; [dx, dy, dyaw]
    '''
    if isinstance(twist, TwistStamped):
        twist = twist.twist
    return [twist.linear.x, twist.linear.y, twist.angular.z]

def process_bag(bag_path: str, sample_rate: float, odom_topic: Union[str, List[str]], img_topics: List[str], printer: Callable = print, use_tqdm: bool = True) -> dict:
    '''
    Open a ROS bag and extract odometry + image topics, sampling at a specified rate.
    Data is appended by row containing the processed ROS data, stored as numpy types.
    Returns type dict

    Inputs:
    - bag_path:     String for full file path for bag, i.e. /home/user/bag_file.bag
    - sample_rate:  Float for rate at which rosbag should be sampled
    - odom_topic:   string or list of strings for odom topic/s to extract, if list then the order specifies order in returned data
    - img_topics:   List of strings for image topics to extract, order specifies order in returned data
    - printer:      Method wrapper for printing (default: print)
    - use_tqdm:     Bool to enable/disable use of tqdm (default: True)
    Returns:
    dict
    '''

    if not bag_path.endswith('.bag'):
        bag_path += '.bag'

    if not isinstance(odom_topic, list):
        _odom_topics = [odom_topic]
    else:
        _odom_topics = odom_topic

    topic_list      = _odom_topics + img_topics

    # Read rosbag
    data = rip_bag(bag_path, sample_rate, topic_list, printer=printer, use_tqdm=use_tqdm)
    _len = len(data)
    printer("Converting stored messages (%s)" % (str(_len)))
    new_dict        = {key: [] for key in img_topics + ['px', 'py', 'pw', 'vx', 'vy', 'vw', 't']}

    none_rows       =   0
    if _len < 1:
        raise Exception('No usable data!')
    if use_tqdm:
        iter_obj = tqdm(data)
    else:
        iter_obj = data
    for row in iter_obj:
        if None in row:
            none_rows = none_rows + 1
            continue

        new_dict['t'].append(row[-1].header.stamp.to_sec()) # get time stamp

        if len(_odom_topics) == 1:
            new_dict['px'].append(row[1].pose.pose.position.x)
            new_dict['py'].append(row[1].pose.pose.position.y)
            new_dict['pw'].append(yaw_from_q(row[1].pose.pose.orientation))

            new_dict['vx'].append(row[1].twist.twist.linear.x)
            new_dict['vy'].append(row[1].twist.twist.linear.y)
            new_dict['vw'].append(row[1].twist.twist.angular.z)
        else:
            new_odoms = []
            for topic in _odom_topics:
                new_odoms.append(np.array(pose2xyw(row[1 + topic_list.index(topic)].pose.pose) + twist2xyw(row[1 + topic_list.index(topic)].twist.twist)))
            new_odoms = np.array(new_odoms)
            new_dict['px'].append(new_odoms[:,0])
            new_dict['py'].append(new_odoms[:,1])
            new_dict['pw'].append(new_odoms[:,2])
            new_dict['vx'].append(new_odoms[:,3])
            new_dict['vy'].append(new_odoms[:,4])
            new_dict['vw'].append(new_odoms[:,5])
            
        for topic in img_topics:
            if "/compressed" in topic:
                new_dict[topic].append(compressed2np(row[1 + topic_list.index(topic)]))
            else:
                new_dict[topic].append(raw2np(row[1 + topic_list.index(topic)]))
    
    printer("%0.2f%% of %d rows contained NoneType; these were ignored." % (100 * none_rows / _len, _len))
    return {key: np.array(new_dict[key]) for key in new_dict}

def rip_bag(bag_path: str, sample_rate: float, topics_in: List[str], timing: int = -1, printer: Callable = print, use_tqdm: bool = True) -> list:
    '''
    Open a ROS bag and store messages from particular topics, sampling at a specified rate.
    If no messages are received, list is populated with NoneType (empties are also NoneType)
    Data is appended by row containing the raw ROS messages
    First column corresponds to sample_rate * len(data)
    Returns data (type List)

    Inputs:
    - bag_path:     String for full file path for bag, i.e. /home/user/bag_file.bag
    - sample_rate:  Float for rate at which rosbag should be sampled
    - topics_in:    List of strings for topics to extract, order specifies order in returned data (time column added to the front)
    - timing:       int type; index of topics_in for topic to use as timekeeper
    - printer:      Method wrapper for printing (default: print)
    - use_tqdm:     Bool to enable/disable use of tqdm (default: True)
    Returns:
    List
    '''
    # warnings.warn('Deprecated, please use scan_bag.')
    
    topics         = [None] + topics_in
    data           = []
    logged_t       = -1
    num_topics     = len(topics)
    num_rows       = 0
    dt             = 1/sample_rate

    # Read rosbag
    printer("Ripping through rosbag, processing topics: %s" % str(topics[1:]))

    row = [None] * num_topics
    with rosbag.Bag(bag_path, 'r') as ros_bag:
        iter_obj = ros_bag.read_messages(topics=topics)
        if use_tqdm: 
           iter_obj = tqdm(iter_obj)
        for (topic, msg, timestamp) in iter_obj: # type: ignore
            row[topics.index(topic)] = msg
            _timer = row[timing] # for linting
            if not isinstance(_timer, genpy.Message) or _timer is None:
                continue
            if hasattr(_timer, 'header') and not (_timer is None):
                if _timer.header.stamp.to_sec() - logged_t > dt:
                    logged_t    = _timer.header.stamp.to_sec()
                    row[0]      = (logged_t + dt/2) - ((logged_t + dt/2) % dt)
                    data.append(row)
                    row         = [None] * num_topics
                    num_rows    = num_rows + 1
                
    return data

def scan_bag(bag_path: str, sample_rate: float, topics_in: List[str], timer_topic: Optional[str] = None, 
             printer: Callable = print, use_tqdm: bool = True) -> list:
    '''
    Prototype

    Open a ROS bag and store messages from particular topics, sampling at a specified rate.
    If no messages are received, list is populated with NoneType (empties are also NoneType)
    Data is appended by row containing the raw ROS messages
    First column corresponds to sample_rate * len(data)
    Returns data (type List)

    Inputs:
    - bag_path:     str type; Full file path for bag, i.e. /home/user/bag_file.bag
    - sample_rate:  float type; Rate at which rosbag should be sampled
    - topics_in:    List[str] type; List of strings for topics to extract, forms keys in returned data
    - timer_topic:  str type (Optional); topic from topics_in to grab timestamps from
    - printer:      Callable type (default: print); Method wrapper for printing
    - use_tqdm:     bool type (default: True); Bool to enable/disable use of tqdm
    Returns:
    List
    '''
    warnings.warn('scan_bag is in development stage, use at own risk.')
    
    data           = {k: [] for k in topics_in + ['__counter', '__timestamp']}
    logged_t       = -1
    logged_c       = -1
    dt             = 1/sample_rate

    timer_topic    = topics_in[-1] if timer_topic is None else timer_topic

    # Read rosbag
    printer("Scanning through rosbag, processing topics: %s" % str(topics_in))

    row = {k: None for k in topics_in + ['__counter', '__timestamp']}
    with rosbag.Bag(bag_path, 'r') as ros_bag:
        iter_obj = ros_bag.read_messages(topics=topics_in)
        for (topic, msg, timestamp) in tqdm(iter_obj) if use_tqdm else iter_obj:
            if logged_t == -1: logged_t = timestamp.stamp.to_sec()
            row[topic] = msg
            _timer = row[timer_topic]
            if not isinstance(_timer, genpy.Message) or _timer is None:
                continue
            logged_c            = _timer.header.stamp.to_sec()
            if timestamp.stamp.to_sec() - logged_t > dt:
                logged_t         = timestamp.header.stamp.to_sec()
                row['__timestamp'] = (logged_t + dt/2) - ((logged_t + dt/2) % dt) # round to nearest dt step
                row['__counter']   = (logged_c + dt/2) - ((logged_c + dt/2) % dt)
                for k, v in row.items():
                    data[k].append(v)
                row         = {k: None for k in topics_in + ['__counter', '__timestamp']} # reset row
                
    return data

def set_rospy_log_lvl(log_level: LogType = LogType.INFO):
    '''
    Change a ROS node's log_level after init_node has been performed.
    Source: https://answers.ros.org/question/9802/change-rospy-node-log-level-while-running/

    Inputs:
    - log_level: LogType type {default: LogType.INFO}
    Returns:
    None
    '''
    logger = logging.getLogger('rosout')
    log_level_rospy = int(log_level.value + 0.5)
    logger.setLevel(_rospy_to_logging_levels[log_level_rospy])

class NodeState(Enum):
    '''
    NodeState Enumeration

    For use with Heartbeat to provide an indication of node health
    '''
    EXIT        = 0
    FAIL        = 1
    INIT        = 2
    WAIT        = 3
    MAIN        = 4

class Heartbeat:
    '''
    Heartbeat Class

    Purpose:
    - Wrap a timer and publisher for easy implementation
    - Create a listen point for diagnosing which nodes have failed
        - If node fails, heartbeat will stop
    '''
    def __init__(self, node_name, namespace, node_rate, node_state=NodeState.INIT, hb_topic='/heartbeats', server=None, period=1):
        '''
        Initialisation

        Inputs:
        - node_name:    string name of node as per ROS
        - namespace:    string namespace to be appended to heartbeat topic
        - node_rate:    float node ROS rate
        - node_state:   NodeState node state (default: NodeState.INIT)
        - hb_topic:     string topic name for heartbeat to be published to (default: /heartbeats)
        Returns:
        None
        '''
        self.namespace  = namespace
        self.node_name  = node_name
        self.node_rate  = node_rate
        self.node_state = node_state.value
        self.hb_topic   = hb_topic
        self.server     = server

        from aarapsi_robot_pack.msg import Heartbeat as Hb

        self.hb_msg     = Hb(node_name=self.node_name, node_rate=self.node_rate, node_state=self.node_state)
        self.hb_pub     = rospy.Publisher(self.namespace + self.hb_topic, Hb, queue_size=1)
        if self.server is None:
            self.hb_timer   = rospy.Timer(rospy.Duration(secs=period), self.heartbeat_cb)
        else:
            self.hb_timer   = rospy.Timer(rospy.Duration(secs=period), self.heartbeat_server_cb)

    def set_state(self, state: NodeState):
        '''
        Set heartbeat node_state

        Inputs:
        - state:    NodeState enum type
        Returns:
        None
        '''
        self.node_state         = state.value
        self.hb_msg.node_state  = self.node_state

    def heartbeat_cb(self, event):
        '''
        Heartbeat timer callback; no server

        Triggers publish of new heartbeat message
        '''
        self.hb_msg.header.stamp = rospy.Time.now()
        self.hb_pub.publish(self.hb_msg)

    def heartbeat_server_cb(self, event):
        '''
        Heartbeat timer callback; with server

        Triggers publish of new heartbeat message
        '''
        assert self.server != None
        self.hb_msg.header.stamp = rospy.Time.now()
        self.hb_msg.topics = list(self.server.pubs.keys())
        now = rospy.Time.now().to_sec()
        self.hb_msg.periods = [round(self.server.pubs[i].last_t - now) for i in self.server.pubs.keys()]
        self.hb_pub.publish(self.hb_msg)

class SubscribeListener(rospy.SubscribeListener):
    '''
    Wrapper for in-built ROS Class to handle detections of subscribe and unsubscribe events
    '''
    def __init__(self, printer=roslogger):
        '''
        Initialisation

        Inputs:
        - printer:      Method wrapper for printing (default: roslogger)
        Returns:
        self
        '''
        super(SubscribeListener, self).__init__()
        self.printer = printer
        self.topics = {}

    def peer_subscribe(self, topic_name, topic_publish, peer_publish):
        '''
        Overwrite base class method for when a subscribe action is detected

        Inputs:
        - topic_name:       str type; topic name
        - topic_publish:    method; unknown purpose
        - peer_publish:     method; unknown purpose
        Returns:
        - None
        '''
        self.printer("[SubscribeListener] Subscribed: %s" % topic_name, LogType.DEBUG, ros=True)
        if topic_name in self.topics.keys():
            if not (self.topics[topic_name]['sub'] is None):
                self.topics[topic_name]['sub'](topic_name)

    def peer_unsubscribe(self, topic_name, num_peers):
        '''
        Overwrite base class method for when an unsubscribe action is detected

        Inputs:
        - topic_name:       str type; topic name
        - num_peers:        int type; number of new subscribers
        Returns:
        - None
        '''
        self.printer("[SubscribeListener] Unsubscribed: %s" % topic_name, LogType.DEBUG, ros=True)
        if topic_name in self.topics.keys():
            if not (self.topics[topic_name]['unsub'] is None):
                self.topics[topic_name]['unsub'](topic_name)

    def add_operation(self, topic_name, method_sub=None, method_unsub=None):
        '''
        Hacky method because I don't understand peer_subscribe and peer_unsubscribe...
        Purpose: add new methods to be called by peer_subscribe and peer_unsubscribe

        Inputs:
        - topic_name:       str type; topic name to add methods for
        - method_sub:       method; function to be triggered by subscribe actions
        - method_unsub:     method; function to be triggered by unsubscribe actions
        Returns:
        - None
        '''
        self.topics[topic_name] = {'sub': method_sub, 'unsub': method_unsub}
        self.printer('New listener added. All current listeners: \n%s' % str(self.topics), LogType.DEBUG, ros=True)

def yaw_from_q(q):
    '''
    Convert geometry_msgs/Quaternion into a float yaw

    Inputs:
    - q: geometry_msgs/Quaternion
    Returns:
    - float, yaw equivalent
    '''
    return euler_from_quaternion([float(q.x), float(q.y), float(q.z), float(q.w)])[2]

def q_from_yaw(yaw):
    '''
    Convert a float yaw into a geometry_msgs/Quaternion

    Inputs:
    - yaw: float value
    Returns:
    - geometry_msgs/Quaternion, quaternion equivalent
    '''
    q = quaternion_from_euler(0, 0, yaw) # roll, pitch, yaw
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

def q_from_rpy(r,p,y, mode='RAD'):
    '''
    Convert a float roll, pitch, and yaw into a geometry_msgs/Quaternion

    Inputs:
    - r: float type; roll
    - p: float type; pitch
    - w: float type; yaw
    Returns:
    - geometry_msgs/Quaternion, quaternion equivalent
    '''
    if not mode == 'RAD':
        r = r * 180 / np.pi
        p = p * 180 / np.pi
        y = y * 180 / np.pi
    q = quaternion_from_euler(r, p, y) # roll, pitch, yaw
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


ROS_ParamType = TypeVar("ROS_ParamType")
class ROS_Param(Generic[ROS_ParamType]):
    '''
    ROS Parameter Class

    Purpose:
    - Wrapper class that handles dynamic updates to ROS parameter values
    '''

    updates_queued      = [] # list of parameters pending an update
    updates_possible    = [] # list of parameter names to check against (in case a parameter update was triggered against a parameter not in scope)
    param_objects       = []

    def __init__(self, name, value: ROS_ParamType, evaluation, force=False, server: Optional["ROS_Param_Server"] = None, printer=roslogger):
        '''
        Initialisation

        Inputs:
        - name:         string name of parameter to be used on parameter server
        - value:        default value of parameter if unset on parameter server
        - evaluation:   handle to method to check value type
        - force:        bool to force update of value on parameter server with input value (defaults to False)
        - server:       ROS_Param_Server reference (defaults to None)
        - printer:      Method wrapper for printing (default: roslogger)
        Returns:
        None
        '''

        if value is None and force:
            raise Exception("Can't force a NoneType value!")

        self.name       = name
        self.server     = server
        self.updates_possible.append(self.name)
        self.param_objects.append(self)

        if self.server is None:
            self.get = self._get_no_server
        else:
            self.get = self._get_server

        self.evaluation                     = evaluation
        self.value: Optional[ROS_ParamType] = None
        self.old                            = None
        self.printer                        = printer
        

        if (not rospy.has_param(self.name)) or (force):
            # If either the parameter doesn't exist, or we want to focus the
            # parameter server to have the inputted value:
            self.set(value)
        else:
            self.value  = self.evaluation(rospy.get_param(self.name))

    def revert(self):
        '''
        Undo latest value change if possible

        Inputs:
        - None
        Returns:
        - None
        '''
        if self.old is None:
            raise Exception('No old value to revert to.')
        self.set_param(self.old)
        self.value = self.old
        self.old = None

    def update(self):
        '''
        Update to latest parameter value

        Inputs:
        - None
        Returns:
        - None
        '''
        try:
            return self.set(rospy.get_param(self.name))
        except:
            self.printer(formatException())
            return False

    def _get_server(self) -> ROS_ParamType:
        '''
        Return value of variable

        Inputs:
        None
        Returns:
        - parsed value of variable
        '''
        assert self.server != None
        if self.server.autochecker:
            assert self.value != None
            return self.value
        else:
            return self._get_no_server()

    def _get_no_server(self) -> ROS_ParamType:
        '''
        Retrieve value of variable if no ROS_Param_Server exists
        Runs a check to see if a change has been made to parameter server value

        Inputs:
        None
        Returns:
        - parsed value of variable
        '''
        assert self.value != None
        if self.name in self.updates_queued:
            self.updates_queued.remove(self.name)
            try:
                self.value = self.evaluation(rospy.get_param(self.name))
            except:
                self.set(self.value)
        return self.value

    def set(self, value):
        '''
        Set value of variable
        Updates parameter server

        Inputs:
        - value to be parsed to parameter server
        Returns:
        None
        '''
        if value is None:
            raise Exception("Cannot set %s to NoneType." % self.name)
        try:
            check_value = self.evaluation(value)
            if not (check_value == self.value):
                self.old    = self.value
                self.value  = check_value
                self.set_param(self.value)
            return True
        except:
            self.set_param(self.value)
            return False
        
    def set_param(self, value):
        if issubclass(type(value), Enum):
            rospy.set_param(self.name, value.value)
        else:
            rospy.set_param(self.name, value)

class ROS_Param_Server:
    '''
    ROS Parameter Server Class
    Purpose:
    - Wrapper class that for ROS_Param that manages and handles dynamic updates to ROS_Param instances
    '''
    def __init__(self, printer: Callable = roslogger):
        '''
        Initialisation

        Inputs:
        - printer:      Method wrapper for printing (default: roslogger)
        Returns:
        None
        '''
        self.updates_possible               = []
        self.params: Dict[str, ROS_Param]   = {}
        self.autochecker                    = False
        self.param_sub                      = None
        self.printer                        = printer

        self.connection_timer               = rospy.Timer(rospy.Duration(5), self._check_server)

    def _check_server(self, event) -> bool:
        if self.param_sub is None:
            self.autochecker = False
        elif self.param_sub.get_num_connections() > 0:
            self.autochecker = True
        else:
            self.autochecker = False
        return self.autochecker

    def add_sub(self, topic, cb, queue_size=100):
        self.param_sub = rospy.Subscriber(topic, String, cb, queue_size=queue_size)

    def add(self, name: str, value, evaluation: Callable, force: bool = False) -> ROS_Param:
        '''
        Add new ROS_Param

        Inputs:
        - name:         string name of parameter to be used on parameter server
        - value:        default value of parameter if unset on parameter server
        - evaluation:   handle to method to check value type
        - force:        bool to force update of value on parameter server with input value
        Returns:
        Generated ROS_Param
        '''
        self.params[name] = ROS_Param[type(value)](name, value, evaluation, force, server=self, printer=self.printer)
        self.updates_possible.append(name)
        return self.params[name]

    def update(self, name: str) -> bool:
        '''
        Update specified parameter / ROS_Param

        Inputs:
        - name:         string name of parameter to be used on parameter server
        Returns:
        None 
        '''
        if not name in self.params:
            return False
        try:
            current_value = str(rospy.get_param(name))
        except:
            current_value = str(None)
        update_status = self.params[name].update()
        if not update_status:
            self.printer("[ROS_Param_Server] Bad parameter server value for %s [%s]. Remaining with last safe value, %s." 
                      % (str(name), current_value, str(self.params[name].value)), LogType.ERROR)
        return update_status

    def exists(self, name: str):
        '''
        Check if specified parameter / ROS_Param exists on server

        Inputs:
        - name:         string name of parameter to be used on parameter server
        Returns:
        bool True or False if exists 

        '''
        if name in self.updates_possible:
            return True
        return False
    
class ROS_Publisher:
    '''
    ROS Publisher Container Class
    Purpose:
    - Wrapper class for ROS Publishers
    '''
    def __init__(self, topic: str, data_class, queue_size: int = 1, latch: bool = False, server = None, subscriber_listener: Optional[SubscribeListener] = None):
        '''
        Initialisation

        Inputs:
        - topic:        string topic
        - data_class:   ROS data class
        - queue_size:   integer number of messages to store for publishing
        - latch:        bool True/False
        Returns:
        None
        '''
        self.topic      = topic
        self.data_class = data_class
        self.queue_size = queue_size
        self.latch      = latch
        self.server     = server
        self.last_t     = -1
        self.period     = -1
        self.sublist    = subscriber_listener

        self.publisher  = rospy.Publisher(self.topic, self.data_class, queue_size=self.queue_size, latch=self.latch, subscriber_listener=subscriber_listener)
    
    def publish(self, msg):
        '''
        Helper for publishing

        Inputs:
        - msg:  ROS data class
        Returns:
        bool True/False for success / fail on publishing

        '''
        try:
            self.publisher.publish(msg)
            now = rospy.Time.now().to_sec()
            if not self.last_t == -1:
                self.period = now - self.last_t
            self.last_t = now
            return True
        except:
            return False