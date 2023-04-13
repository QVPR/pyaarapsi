#!/usr/bin/env python3
import rospy
from enum import Enum
from geometry_msgs.msg import Quaternion
from tf.transformations import quaternion_from_euler, euler_from_quaternion

# For logging
class LogType(Enum):
    DEBUG       = "[DEBUG]"
    INFO        = "[INFO]"
    WARN        = "[WARN]"
    ERROR       = "[!ERROR!]"
    FATAL       = "[!!FATAL!!]"

def roslogger(text, logtype, ros=False, throttle=0, no_stamp=False):
# Print function helper
# For use with integration with ROS
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
    return euler_from_quaternion([float(q.x), float(q.y), float(q.z), float(q.w)])[2]

def q_from_yaw(yaw):
    q = quaternion_from_euler(0, 0, yaw) # roll, pitch, yaw
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

class ROS_Param:

    updates_queued = []
    updates_possible = []

    def __init__(self, name, value, evaluation, force=False, namespace=None):
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
        if self.name in self.updates_queued:
            self.updates_queued.remove(self.name)
            try:
                check_value = self.evaluation(rospy.get_param(self.name, self.value))
                self.value = check_value
            except:
                pass
        return self.value

    def set(self, value):
        rospy.set_param(self.name, value)
        self.value = self.evaluation(value)