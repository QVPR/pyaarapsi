#!/usr/bin/env python3
import rospy
import rosbag
import logging
import sys
import numpy as np
from enum import Enum
import cv2
from cv_bridge import CvBridge
from tqdm.auto import tqdm

from geometry_msgs.msg          import Quaternion
from sensor_msgs.msg            import Image, CompressedImage

from aarapsi_robot_pack.msg     import Debug # Our custom msg structures
from aarapsi_robot_pack.msg     import Heartbeat as Hb

from tf.transformations         import quaternion_from_euler, euler_from_quaternion
from .helper_tools              import formatException, vis_dict
from .enum_tools                import enum_name
from .argparse_tools            import check_enum, check_positive_two_int_list, check_string, check_positive_float, check_positive_int, check_string_list
from ..vpr_simple.vpr_helpers   import FeatureType, SVM_Tolerance_Mode

class SubscribeListener(rospy.SubscribeListener):
    '''
    Wrapper for in-built ROS Class to handle detections of subscribe and unsubscribe events
    '''
    def __init__(self):
        '''
        Initialisation

        Inputs:
        - None
        Returns:
        self
        '''
        super(SubscribeListener, self).__init__()
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
        roslogger("[SubscribeListener] Subscribed: %s" % topic_name, LogType.DEBUG, ros=True)
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
        roslogger("[SubscribeListener] Unsubscribed: %s" % topic_name, LogType.DEBUG, ros=True)
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
        roslogger('New listener added. All current listeners: \n%s' % str(self.topics), LogType.DEBUG, ros=True)

def process_bag(bag_path, sample_rate, odom_topic, img_topics, printer=print, use_tqdm=True):
    '''
    Open a ROS bag and extract odometry + image topics, sampling at a specified rate.
    Data is appended by row containing the processed ROS data, stored as numpy types.
    Returns type dict

    Inputs:
    - bag_path:     String for full file path for bag, i.e. /home/user/bag_file.bag
    - sample_rate:  Float for rate at which rosbag should be sampled
    - odom_topic:   string for odom topic to extract
    - img_topics:   List of strings for image topics to extract, order specifies order in returned data
    - printer:      Method wrapper for printing (default: print)
    - use_tqdm:     Bool to enable/disable use of tqdm (default: True)
    Returns:
    dict
    '''

    if not bag_path.endswith('.bag'):
        bag_path += '.bag'

    topic_list = [odom_topic] + img_topics
    # Read rosbag
    data = rip_bag(bag_path, sample_rate, topic_list, printer=printer, use_tqdm=use_tqdm)

    printer("Converting stored messages (%s)" % (str(len(data))))
    bridge          = CvBridge()
    new_dict        = {key: [] for key in img_topics + ['px', 'py', 'pw', 'vx', 'vy', 'vw', 't']}

    compress_func   = lambda msg: bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
    raw_img_func    = lambda msg: bridge.imgmsg_to_cv2(msg, "passthrough")
    none_rows       =   0
    if len(data) < 1:
        raise Exception('No usable data!')
    if use_tqdm:
        iter_obj = tqdm(data)
    else:
        iter_obj = data
    for row in iter_obj:
        if None in row:
            none_rows = none_rows + 1
            continue
        new_dict['t'].append(row[1].header.stamp.to_sec())
        new_dict['px'].append(row[1].pose.pose.position.x)
        new_dict['py'].append(row[1].pose.pose.position.y)
        new_dict['pw'].append(yaw_from_q(row[1].pose.pose.orientation))

        new_dict['vx'].append(row[1].twist.twist.linear.x)
        new_dict['vy'].append(row[1].twist.twist.linear.y)
        new_dict['vw'].append(row[1].twist.twist.angular.z)

        for topic in img_topics:
            if "/compressed" in topic:
                new_dict[topic].append(compress_func(row[1 + topic_list.index(topic)]))
            else:
                new_dict[topic].append(raw_img_func(row[1 + topic_list.index(topic)]))
    printer("%0.2f%% of %d rows contained NoneType; these were ignored." % (100 * none_rows / len(data), len(data)))
    return {key: np.array(new_dict[key]) for key in new_dict}

def rip_bag(bag_path, sample_rate, topics_in, printer=print, use_tqdm=True):
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
    - printer:      Method wrapper for printing (default: print)
    - use_tqdm:     Bool to enable/disable use of tqdm (default: True)
    Returns:
    List
    '''
    topics         = [None] + topics_in
    data           = []
    logged_t       = -1
    num_topics     = len(topics)
    num_rows       = 0

    # Read rosbag
    printer("Ripping through rosbag, processing topics: %s" % str(topics[1:]))

    row = [None] * num_topics
    with rosbag.Bag(bag_path, 'r') as ros_bag:
        if use_tqdm: 
            iter_obj = tqdm(ros_bag.read_messages(topics=topics))
        else: 
            iter_obj = ros_bag.read_messages(topics=topics)
        for topic, msg, timestamp in iter_obj:
            row[topics.index(topic)] = msg
            if logged_t == -1:
                logged_t    = timestamp.to_sec()
            elif timestamp.to_sec() - logged_t > 1/sample_rate:
                row[0]      = sample_rate * num_rows
                data.append(row)
                row         = [None] * num_topics
                num_rows    = num_rows + 1
                logged_t    = timestamp.to_sec()
                
    return data

def set_rospy_log_lvl(log_level):
    '''
    Change a ROS node's log_level after init_node has been performed.
    Source: https://answers.ros.org/question/9802/change-rospy-node-log-level-while-running/

    Inputs:
    - log_level: rospy log_level, either an integer or rospy enum (i.e. to set to debug, use either 1 or rospy.DEBUG)
    Returns:
    None
    '''
    logger = logging.getLogger('rosout')
    logger.setLevel(rospy.impl.rosout._rospy_to_logging_levels[log_level])

def imgmsgtrans(msg, transform, bridge=None):
    '''
    Transform ROS image data

    Inputs:
    - msg:          sensor_msgs/(Compressed)Image
    - transform:    handle to function to be applied
    - bridge:       CvBridge object, or none (function will initialise)
    Returns:
    - transformed image of type input msg
    '''
    if bridge is None:
        bridge = CvBridge()

    if isinstance(msg, CompressedImage):
        img         = bridge.compressed_imgmsg_to_cv2(msg, "passthrough")
        img_trans   = transform(img)
        msg_trans   = bridge.cv2_to_compressed_imgmsg(img_trans, "jpeg")
    elif isinstance(msg, Image):
        img         = bridge.imgmsg_to_cv2(msg, "passthrough")
        img_trans   = transform(img)
        msg_trans   = bridge.cv2_to_imgmsg(img_trans, "bgr8")
    else:
        raise Exception("Type not CompressedImage or Image.")
    return msg_trans

def msg2img(msg, bridge=None):
    '''
    Convert ROS msg to cv2 image

    Inputs:
    - msg:      sensor_msgs/(Compressed)Image
    - bridge:   CvBridge object, or none (function will initialise)
    Returns:
    - converted image as cv2 array
    '''
    if bridge is None:
        bridge = CvBridge()

    if isinstance(msg, CompressedImage):
        img         = bridge.compressed_imgmsg_to_cv2(msg, "passthrough")
    elif isinstance(msg, Image):
        img         = bridge.imgmsg_to_cv2(msg, "passthrough")
    else:
        raise Exception("Type not CompressedImage or Image.")
    return img

def img2msg(img, mode, bridge=None):
    '''
    Convert cv2 img to ROS msg

    Inputs:
    - img:      cv2 image array
    - mode:     string, either 'Image' or 'CompressedImage'
    - bridge:   CvBridge object, or none (function will initialise)
    Returns:
    - sensor_msgs/(Compressed)Image
    '''
    if bridge is None:
        bridge = CvBridge()

    if mode == 'CompressedImage':
        msg = bridge.cv2_to_compressed_imgmsg(img, "jpeg")
    elif mode == 'Image':
        msg = bridge.cv2_to_imgmsg(img, "bgr8")
    else:
        raise Exception("Mode not 'CompressedImage' or 'Image'.")
    return msg

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
        self.hb_msg.header.stamp = rospy.Time.now()
        self.hb_msg.topics = list(self.server.pubs.keys())
        now = rospy.Time.now().to_sec()
        self.hb_msg.periods = [round(self.server.pubs[i].last_t - now) for i in self.server.pubs.keys()]
        self.hb_pub.publish(self.hb_msg)

class LogType(Enum):
    '''
    LogType Enumeration

    For use with roslogger
    '''

    DEBUG       = "[DEBUG]"
    INFO        = "[INFO]"
    WARN        = "[WARN]"
    ERROR       = "[!ERROR!]"
    FATAL       = "[!!FATAL!!]"

def roslogger(text, logtype=LogType.INFO, throttle=0, ros=True, name=None, no_stamp=True):
    '''
    Print function helper
    For use with integration with ROS
        This function seeks to exploit rospy's colouring and logging scheme, but add
        functionality such that a user can add a prefix, hide the stamp, and switch 
        quickly to work outside of a ROS node.

    Inputs:
    - text:     text string to be printed, must be pre-formatted (can't be done inside roslogger)
    - logtype:  LogType enum to control which print type (debug, info, etc...)
    - ros:      bool that swaps between rospy logging and print
    - throttle: number of seconds of pause between each message (rospy logging only)
    - no_stamp: bool to enable overwriting of generic rospy timestamp
    - obj:      Object with any of the following attributes: 
                    obj.logros (overrides ros)
                    obj.logname (overrides name)
                    obj.logstamp (overrides no_stamp)

    Returns:
    None
    '''

    text = str(text) # just in case someone did something silly

    if isinstance(name, str):
        text = '[' + name + '] ' + text

    try:
        if ros: # if used inside of a running ROS node
            if no_stamp:
                text = ('\b' * 21) + text + (' ' * np.max([21 - len(text), 0]))
            if logtype == LogType.DEBUG:
                rospy.logdebug_throttle(throttle, text)
            elif logtype == LogType.INFO:
                rospy.loginfo_throttle(throttle, text)
            elif logtype == LogType.WARN:
                rospy.logwarn_throttle(throttle, text)
            elif logtype == LogType.ERROR:
                rospy.logerr_throttle(throttle, text)
            elif logtype == LogType.FATAL:
                rospy.logfatal_throttle(throttle, text)
        else:
            raise Exception
    except:
        print(logtype.value + " " + text)

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

class ROS_Param:
    '''
    ROS Parameter Class

    Purpose:
    - Wrapper class that handles dynamic updates to ROS parameter values
    '''

    updates_queued      = [] # list of parameters pending an update
    updates_possible    = [] # list of parameter names to check against (in case a parameter update was triggered against a parameter not in scope)
    param_objects       = []

    def __init__(self, name, value, evaluation, force=False, server=None):
        '''
        Initialisation

        Inputs:
        - name:         string name of parameter to be used on parameter server
        - value:        default value of parameter if unset on parameter server
        - evaluation:   handle to method to check value type
        - force:        bool to force update of value on parameter server with input value (defaults to False)
        - server:       ROS_Param_Server reference (defaults to None)
        Returns:
        None
        '''

        if value is None and force:
            raise Exception("Can't force a NoneType value!")

        self.name       = name
        self.server     = server

        if self.server is None:
            self.updates_possible.append(self.name)
            self.param_objects.append(self)
            self.get = self._get_no_server
        else:
            self.get = self._get_server

        self.evaluation = evaluation
        self.value      = None
        self.old        = None

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
            roslogger(formatException())
            return False

    def _get_server(self):
        '''
        Return value of variable

        Inputs:
        None
        Returns:
        - parsed value of variable
        '''
        return self.value

    def _get_no_server(self):
        '''
        Retrieve value of variable if no ROS_Param_Server exists
        Runs a check to see if a change has been made to parameter server value

        Inputs:
        None
        Returns:
        - parsed value of variable
        '''
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
            rospy.set_param(self.name, enum_name(value))
        else:
            rospy.set_param(self.name, value)

class ROS_Param_Server:
    '''
    ROS Parameter Server Class
    Purpose:
    - Wrapper class that for ROS_Param that manages and handles dynamic updates to ROS_Param instances
    '''
    def __init__(self):
        '''
        Initialisation

        Inputs:
        None
        Returns:
        None
        '''
        self.updates_possible   = []
        self.params             = {}

    def add(self, name, value, evaluation, force=False):
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
        self.params[name] = ROS_Param(name, value, evaluation, force, server=self)
        self.updates_possible.append(name)
        return self.params[name]

    def update(self, name):
        '''
        Update specified parameter / ROS_Param

        Inputs:
        - name:         string name of parameter to be used on parameter server
        Returns:
        None 
        '''
        try:
            current_value = str(rospy.get_param(name))
        except:
            current_value = str(None)
        update_status = self.params[name].update()
        if not update_status:
            roslogger("[ROS_Param_Server] Bad parameter server value for %s [%s]. Remaining with last safe value, %s." 
                      % (str(name), current_value, str(self.params[name].value)), LogType.ERROR)
        return update_status

    def exists(self, name):
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
    def __init__(self, topic, data_class, queue_size=1, latch=False, server=None, subscriber_listener=None):
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

class Base_ROS_Class:
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

    def __init__(self, node_name, namespace, rate_num, anon, log_level, \
                 order_id=None, throttle=30, colour=True, debug=True, disable_signals=False, hb_topic='/heartbeats'):
        '''
        Initialisation

        Inputs:
        - mrc:          class type; Main ROS Class to assign parameters to
        - node_name:    str type;   Name of node, used in rospy.init_node and nodespace
        - namespace:    str type;   rospy namespace
        - rate_num:     float type; ROS node execution rate
        - anon:         bool type;  Whether to run the node as anonymous
        - log_level:    int type;   Initial rospy log level
        - order_id:     int type {default: None};        The namespace/launch_step parameter value to wait for before proceeding
        - throttle:     float type {default: 30};        Wait seconds before publishing rospy.DEBUG launch_step wait status
        - colour:       bool type {default: True};       Whether or not to colour the launch message
        - debug:        bool type {default: True};       Whether or not to create a subscriber and callback for debugging
        - hb_topic:     str type {default: /heartbeats}; Which topic to publish heartbeat messages on
        Returns:
        - bool, True on success (False on Exception)
        '''
        rospy.init_node(node_name, anonymous=anon, log_level=log_level, disable_signals=disable_signals)
        self.namespace   = namespace
        self.node_name   = node_name
        self.nodespace   = self.namespace + '/' + self.node_name

        self.pubs       = {}
        self.params     = ROS_Param_Server()

        self.hb          = Heartbeat(self.node_name, self.namespace, rate_num, node_state=NodeState.INIT, hb_topic=hb_topic, server=self)

        if debug:
            self._debug_sub  = rospy.Subscriber(self.namespace + '/debug', Debug, self.debug_cb, queue_size=1)

        if not order_id is None:
            launch_step = rospy.get_param(self.namespace + '/launch_step')
            while (launch_step < order_id):
                if rospy.is_shutdown():
                    try:
                        self.exit()
                    except:
                        sys.exit()
                roslogger('%s waiting in line, position %s.' % (str(self.node_name), str(order_id)), LogType.DEBUG, throttle=throttle, ros=True)
                rospy.sleep(0.2)
                launch_step = rospy.get_param(self.namespace + '/launch_step')
        if colour:
            roslogger('\033[96mStarting %s node.\033[0m' % (self.node_name), ros=True, no_stamp=True)
        else:
            roslogger('Starting %s node.' % (self.node_name), ros=True, no_stamp=True)
        return True
    
    def init_params(self, rate_num, log_level, reset):
        self.FEAT_TYPE              = self.params.add(self.namespace + "/feature_type",        None,                   lambda x: check_enum(x, FeatureType),           force=False)
        self.IMG_DIMS               = self.params.add(self.namespace + "/img_dims",            None,                   check_positive_two_int_list,                    force=False)
        self.NPZ_DBP                = self.params.add(self.namespace + "/npz_dbp",             None,                   check_string,                                   force=False)
        self.BAG_DBP                = self.params.add(self.namespace + "/bag_dbp",             None,                   check_string,                                   force=False)
        self.SVM_DBP                = self.params.add(self.namespace + "/svm_dbp",             None,                   check_string,                                   force=False)
        self.IMG_TOPIC              = self.params.add(self.namespace + "/img_topic",           None,                   check_string,                                   force=False)
        self.ODOM_TOPIC             = self.params.add(self.namespace + "/odom_topic",          None,                   check_string,                                   force=False)
        
        self.PATH_BAG               = self.params.add(self.namespace + "/path/bag_name",       None,                   check_string,                                   force=False)
        self.PATH_ODOM              = self.params.add(self.namespace + "/path/odom_topic",     None,                   check_string,                                   force=False)
        self.PATH_IMG               = self.params.add(self.namespace + "/path/img_topic",      None,                   check_string,                                   force=False)

        self.REF_BAG_NAME           = self.params.add(self.namespace + "/ref/bag_name",        None,                   check_string,                                   force=False)
        self.REF_FILTERS            = self.params.add(self.namespace + "/ref/filters",         None,                   check_string,                                   force=False)
        self.REF_SAMPLE_RATE        = self.params.add(self.namespace + "/ref/sample_rate",     None,                   check_positive_float,                           force=False) # Hz
        
        self.RATE_NUM               = self.params.add(self.nodespace + "/rate",                rate_num,               check_positive_float,                           force=reset)
        self.LOG_LEVEL              = self.params.add(self.nodespace + "/log_level",           log_level,              check_positive_int,                             force=reset)

        self.REF_DATA_PARAMS        = [self.NPZ_DBP, self.BAG_DBP, self.REF_BAG_NAME, self.REF_FILTERS, self.REF_SAMPLE_RATE, self.IMG_TOPIC, self.ODOM_TOPIC, self.FEAT_TYPE, self.IMG_DIMS]
        self.REF_DATA_NAMES         = [i.name for i in self.REF_DATA_PARAMS]

        self.SVM_QRY_BAG_NAME       = self.params.add(self.namespace + "/svm/qry/bag_name",        None,             check_string,                                 force=False)
        self.SVM_QRY_FILTERS        = self.params.add(self.namespace + "/svm/qry/filters",         None,             check_string,                                 force=False)
        self.SVM_QRY_SAMPLE_RATE    = self.params.add(self.namespace + "/svm/qry/sample_rate",     None,             check_positive_float,                         force=False)

        self.SVM_REF_BAG_NAME       = self.params.add(self.namespace + "/svm/ref/bag_name",        None,             check_string,                                 force=False)
        self.SVM_REF_FILTERS        = self.params.add(self.namespace + "/svm/ref/filters",         None,             check_string,                                 force=False)
        self.SVM_REF_SAMPLE_RATE    = self.params.add(self.namespace + "/svm/ref/sample_rate",     None,             check_positive_float,                         force=False)
        
        self.SVM_FACTORS            = self.params.add(self.namespace + "/svm/factors",             None,             check_string_list,                            force=False)
        self.SVM_TOL_MODE           = self.params.add(self.namespace + "/svm/tolerance/mode",      None,             lambda x: check_enum(x, SVM_Tolerance_Mode),  force=False)
        self.SVM_TOL_THRES          = self.params.add(self.namespace + "/svm/tolerance/threshold", None,             check_positive_float,                         force=False)
        

        self.SVM_DATA_PARAMS        = [self.FEAT_TYPE, self.IMG_DIMS, self.NPZ_DBP, self.BAG_DBP, self.SVM_DBP, self.IMG_TOPIC, self.ODOM_TOPIC, \
                                       self.SVM_QRY_BAG_NAME, self.SVM_QRY_FILTERS, self.SVM_QRY_SAMPLE_RATE, \
                                       self.SVM_REF_BAG_NAME, self.SVM_REF_FILTERS, self.SVM_REF_SAMPLE_RATE, \
                                       self.SVM_FACTORS, self.SVM_TOL_MODE, self.SVM_TOL_THRES]
        self.SVM_DATA_NAMES         = [i.name for i in self.SVM_DATA_PARAMS]

    def make_dataset_dict(self, path=False):
        if path:
            bag_name    = self.PATH_BAG.get()
            odom_topic  = self.PATH_ODOM.get()
            img_topics  = [self.PATH_IMG.get()]
        else:
            bag_name = self.REF_BAG_NAME.get()
            odom_topic  = self.ODOM_TOPIC.get()
            img_topics  = [self.IMG_TOPIC.get()]
        return dict(bag_name=bag_name, npz_dbp=self.NPZ_DBP.get(), bag_dbp=self.BAG_DBP.get(), \
                    odom_topic=odom_topic, img_topics=img_topics, sample_rate=self.REF_SAMPLE_RATE.get(), \
                    ft_types=enum_name(self.FEAT_TYPE.get(),wrap=True), img_dims=self.IMG_DIMS.get(), filters='{}')
    
    def make_svm_model_params(self):
        qry_dict = dict(bag_name=self.SVM_QRY_BAG_NAME.get(), npz_dbp=self.NPZ_DBP.get(), bag_dbp=self.BAG_DBP.get(), \
                        odom_topic=self.ODOM_TOPIC.get(), img_topics=[self.IMG_TOPIC.get()], sample_rate=self.SVM_REF_SAMPLE_RATE.get(), \
                        ft_types=enum_name(self.FEAT_TYPE.get(),wrap=True), img_dims=self.IMG_DIMS.get(), filters='{}')
        ref_dict = dict(bag_name=self.SVM_REF_BAG_NAME.get(), npz_dbp=self.NPZ_DBP.get(), bag_dbp=self.BAG_DBP.get(), \
                        odom_topic=self.ODOM_TOPIC.get(), img_topics=[self.IMG_TOPIC.get()], sample_rate=self.SVM_REF_SAMPLE_RATE.get(), \
                        ft_types=enum_name(self.FEAT_TYPE.get(),wrap=True), img_dims=self.IMG_DIMS.get(), filters='{}')
        svm_dict = dict(factors=self.SVM_FACTORS.get(), tol_thres=self.SVM_TOL_THRES.get(), tol_mode=enum_name(self.SVM_TOL_MODE.get()))
        return dict(ref=ref_dict, qry=qry_dict, svm=svm_dict, npz_dbp=self.NPZ_DBP.get(), bag_dbp=self.BAG_DBP.get(), svm_dbp=self.SVM_DBP.get())
    
    def debug_cb(self, msg):
        if msg.node_name == self.node_name:
            try:
                if msg.instruction == 0:
                    self.print(self.make_svm_model_params(), LogType.DEBUG)
                elif msg.instruction == 1:
                    self.print(self.make_dataset_dict(), LogType.DEBUG)
                elif msg.instruction == 2:
                    self.print(vis_dict(self.ip.dataset), LogType.DEBUG)
                else:
                    self.print(msg.instruction, LogType.DEBUG)
            except:
                self.print("Debug operation failed.", LogType.DEBUG)

    def add_pub(self, topic, data_class, queue_size=1, latch=False, subscriber_listener=None):
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

        self.pubs[topic] = ROS_Publisher(topic, data_class, queue_size, latch, server=self, subscriber_listener=subscriber_listener)
        return self.pubs[topic]

    def set_state(self, state: NodeState):
        '''
        Set heartbeat node_state

        Inputs:
        - state:    NodeState enum type
        Returns:
        None
        '''
        self.hb.set_state(state)

    def print(self, text, logtype=LogType.INFO, throttle=0, ros=None, name=None, no_stamp=None):
        if ros is None:
            ros = True
        if name is None:
            name = self.node_name
        if no_stamp is None:
            no_stamp = True
        roslogger(text, logtype, throttle=throttle, ros=ros, name=name, no_stamp=no_stamp)

    def exit(self):
        self.print("Quit received")
        sys.exit()

    


    