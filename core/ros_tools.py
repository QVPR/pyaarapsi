#!/usr/bin/env python3
import rospy
from enum import Enum
from cv_bridge import CvBridge

from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Image, CompressedImage
from aarapsi_robot_pack.msg import Heartbeat as Hb

from tf.transformations import quaternion_from_euler, euler_from_quaternion

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
    def __init__(self, node_name, namespace, node_state: NodeState, node_rate, hb_topic='/heartbeats'):
        '''
        Initialisation

        Inputs:
        - node_name:    string name of node as per ROS
        - namespace:    string namespace to be appended to heartbeat topic
        - node_state:   int node state
        - node_rate:    float node ROS rate
        - hb_topic:     string topic name for heartbeat to be published to
        Returns:
        None
        '''
        self.namespace  = namespace
        self.node_name  = node_name
        self.node_rate  = node_rate
        self.node_state = node_state.value
        self.hb_topic   = hb_topic

        self.hb_msg     = Hb(node_name=self.node_name, node_rate=self.node_rate, node_state=self.node_state)

        self.hb_pub     = rospy.Publisher(self.namespace + self.hb_topic, Hb, queue_size=1)
        self.hb_timer   = rospy.Timer(rospy.Duration(secs=1), self.heartbeat_cb)

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
        Heartbeat timer callback

        Triggers publish of new heartbeat message
        '''
        self.hb_msg.header.stamp = rospy.Time.now()
        self.hb_pub.publish(self.hb_msg)

class LogType(Enum):
    '''
    Enum type to use with roslogger
    '''

    DEBUG       = "[DEBUG]"
    INFO        = "[INFO]"
    WARN        = "[WARN]"
    ERROR       = "[!ERROR!]"
    FATAL       = "[!!FATAL!!]"

def roslogger(text, logtype, ros=False, throttle=0, no_stamp=False):
    '''
    Print function helper
    For use with integration with ROS 

    Inputs:
    - text:     text string to be printed, must be pre-formatted (can't be done inside roslogger)
    - logtype:  LogType enum to control which print type (debug, info, etc...)
    - ros:      bool that swaps between rospy logging and print
    - throttle: number of seconds of pause between each message (rospy logging only)
    - no_stamp: bool to enable overwriting of generic rospy timestamp

    Returns:
    None
    '''

    try:
        if ros: # if used inside of a running ROS node
            if no_stamp:
                go_back = '\b' * 21
            else:
                go_back = ''
            if logtype == LogType.DEBUG:
                rospy.logdebug_throttle(throttle, go_back + text)
            elif logtype == LogType.INFO:
                rospy.loginfo_throttle(throttle, go_back + text)
            elif logtype == LogType.WARN:
                rospy.logwarn_throttle(throttle, go_back + text)
            elif logtype == LogType.ERROR:
                rospy.logerr_throttle(throttle, go_back + text)
            elif logtype == LogType.FATAL:
                rospy.logfatal_throttle(throttle, go_back + text)
        else:
            raise Exception
    except:
        print(logtype.value + " " + str(text))

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

    def __init__(self, name, value, evaluation, force=False, namespace=None):
        '''
        Initialisation

        Inputs:
        - name:         string name of parameter to be used on parameter server
        - value:        default value of parameter if unset on parameter server
        - evaluation:   handle to method to check value type
        - force:        bool to force update of value on parameter server with input value
        - namespace:    string name of namespace to be appended to name
        Returns:
        None
        '''
        if namespace is None:
            self.name = name
        else:
            self.name = namespace + "/" + name
        self.updates_possible.append(self.name)
        self.evaluation = evaluation
        self.value = None
        if rospy.has_param(self.name) and (not force):
            try:
                check_value = self.evaluation(rospy.get_param(self.name, value))
                self.value = check_value
            except:
                pass
        else:
            self.set(value)

    def get(self):
        '''
        Retrieve value of variable
        Runs a check to see if a change has been made to parameter server value

        Inputs:
        None
        Returns:
        - parsed value of variable
        '''
        if self.name in self.updates_queued:
            self.updates_queued.remove(self.name)
            try:
                check_value = self.evaluation(rospy.get_param(self.name, self.value))
                self.value = check_value
            except:
                pass
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
        rospy.set_param(self.name, value)
        self.value = self.evaluation(value)