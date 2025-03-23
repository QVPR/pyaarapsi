#!/usr/bin/env python3
from enum import Enum
import rospy

from aarapsi_robot_pack.msg import Heartbeat as Hb

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
    def __init__(self, node_name, namespace, node_rate, node_state=NodeState.INIT,
                 hb_topic='/heartbeats', server=None, period=1):
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
        self.namespace = namespace
        self.node_name = node_name
        self.node_rate = node_rate
        self.node_state = node_state.value
        self.hb_topic = hb_topic
        self.server = server
        self.hb_msg = Hb(node_name=self.node_name, node_rate=self.node_rate,
                         node_state=self.node_state)
        self.hb_pub = rospy.Publisher(self.namespace + self.hb_topic, Hb, queue_size=1)
        if self.server is None:
            self.hb_timer = rospy.Timer(rospy.Duration(secs=period), self.heartbeat_cb)
        else:
            self.hb_timer = rospy.Timer(rospy.Duration(secs=period), self.heartbeat_server_cb)

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
        assert self.server is not None
        self.hb_msg.header.stamp = rospy.Time.now()
        self.hb_msg.topics = list(self.server.pubs.keys())
        now = rospy.Time.now().to_sec()
        self.hb_msg.periods = [round(self.server.pubs[i].last_t - now) \
                                    for i in self.server.pubs.keys()]
        self.hb_pub.publish(self.hb_msg)
