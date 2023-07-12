#!/usr/bin/env python3
import rospy
import rosbag
import logging
import numpy as np
from enum import Enum

import cv2
from cv_bridge import CvBridge
from cv_bridge.boost.cv_bridge_boost import cvtColor2

from tqdm.auto import tqdm

from std_msgs.msg               import String
from geometry_msgs.msg          import Quaternion, Pose, PoseWithCovariance, PoseStamped
from sensor_msgs.msg            import Image, CompressedImage

from tf.transformations         import quaternion_from_euler, euler_from_quaternion
from .helper_tools              import formatException
from .enum_tools                import enum_name
from .roslogger                 import LogType, roslogger

def compressed2np(msg: CompressedImage, encoding: str = "passthrough") -> np.ndarray:
    buf     = np.ndarray(shape=(1, len(msg.data)), dtype=np.uint8, buffer=msg.data)
    img_in  = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if encoding == "passthrough":
        img_out = img_in
    else:
        img_out = cvtColor2(img_in, "bgr8", encoding)
    return img_out

def np2compressed(img_in: np.ndarray, encoding: str = "jpg", add_stamp: bool = True):
    msg_out = CompressedImage()
    msg_out.format = encoding
    msg_out.data = np.array(cv2.imencode('.' + encoding, img_in)[1]).tostring()
    if add_stamp:
        msg_out.header.stamp = rospy.Time.now()
    return msg_out

def raw2np(msg: Image, bridge: CvBridge) -> np.ndarray:
    return bridge.imgmsg_to_cv2(msg)

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

def pose2xyw(pose: Pose, stamped=False):
    '''
    Extract x, y, and yaw from a geometry_msgs/Pose object

    Inputs:
    - pose:     geometry_msgs/Pose[Stamped] ROS message object
    - stamped:  bool type {default: False}; if true, extracts Pose from PoseStamped
    Returns:
    type list; [x, y, yaw]
    '''
    if stamped:
        pose = pose.pose
    return [pose.position.x, pose.position.y, yaw_from_q(pose.orientation)]

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

class ROS_Param:
    '''
    ROS Parameter Class

    Purpose:
    - Wrapper class that handles dynamic updates to ROS parameter values
    '''

    updates_queued      = [] # list of parameters pending an update
    updates_possible    = [] # list of parameter names to check against (in case a parameter update was triggered against a parameter not in scope)
    param_objects       = []

    def __init__(self, name, value, evaluation, force=False, server=None, printer=roslogger):
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

        self.evaluation = evaluation
        self.value      = None
        self.old        = None
        self.printer    = printer
        

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

    def _get_server(self):
        '''
        Return value of variable

        Inputs:
        None
        Returns:
        - parsed value of variable
        '''
        if self.server.autochecker:
            return self.value
        else:
            return self._get_no_server()

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
    def __init__(self, printer=roslogger):
        '''
        Initialisation

        Inputs:
        - printer:      Method wrapper for printing (default: roslogger)
        Returns:
        None
        '''
        self.updates_possible   = []
        self.params             = {}
        self.autochecker        = False
        self.param_sub          = None
        self.printer            = printer

        self.connection_timer   = rospy.Timer(rospy.Duration(5), self._check_server)

    def _check_server(self, event):
        if self.param_sub is None:
            self.autochecker = False
        elif self.param_sub.get_num_connections() > 0:
            self.autochecker = True
        else:
            self.autochecker = False
        return self.autochecker

    def add_sub(self, topic, cb, queue_size=100):
        self.param_sub = rospy.Subscriber(topic, String, cb, queue_size=queue_size)

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
        self.params[name] = ROS_Param(name, value, evaluation, force, server=self, printer=self.printer)
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
            self.printer("[ROS_Param_Server] Bad parameter server value for %s [%s]. Remaining with last safe value, %s." 
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