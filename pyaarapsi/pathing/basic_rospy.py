#! /usr/bin/env python3
'''
Basic pathing helper functions, with rospy addition.
'''
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import rospy
from nav_msgs.msg import Path
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import PoseStamped, Point, Vector3
from visualization_msgs.msg import MarkerArray, Marker

from pyaarapsi.core.helper_tools import angle_wrap
from pyaarapsi.core.ros_tools import q_from_yaw, q_from_rpy, ROSPublisher

def make_path_speeds(path_xyws: NDArray[np.float32], path_indices: List[int]
                     ) -> Tuple[Path, MarkerArray]:
    '''
    TODO
    '''
    path            = Path(header=Header(stamp=rospy.Time.now(), frame_id="map"))
    path.poses      = []
    speeds          = MarkerArray()
    speeds.markers  = []
    direction       = np.sign(np.sum(path_xyws[:,2]))
    for i in path_indices:
        new_pose                        = PoseStamped(header=Header(stamp=rospy.Time.now(),
                                                                    frame_id="map"))
        new_pose.pose.position          = Point(x=path_xyws[i,0], y=path_xyws[i,1], z=0.0)
        new_pose.pose.orientation       = q_from_yaw(path_xyws[i,2])
        #
        new_speed                       = Marker(header=Header(stamp=rospy.Time.now(),
                                                               frame_id='map'))
        new_speed.type                  = new_speed.ARROW
        new_speed.action                = new_speed.ADD
        new_speed.id                    = i
        new_speed.color                 = ColorRGBA(r=0.859, g=0.094, b=0.220, a=0.5)
        new_speed.scale                 = Vector3(x=path_xyws[i,3], y=0.05, z=0.05)
        new_speed.pose.position         = Point(x=path_xyws[i,0], y=path_xyws[i,1], z=0.0)
        new_speed.pose.orientation      = q_from_yaw(path_xyws[i,2] - direction * np.pi/2)
        #
        path.poses.append(new_pose)
        speeds.markers.append(new_speed)
    return path, speeds

def make_zones(path_xyws: NDArray[np.float32], zone_indices: List[int]) -> MarkerArray:
    '''
    TODO
    '''
    zones           = MarkerArray()
    zones.markers   = []
    for c, i in enumerate(zone_indices):
        k                           = i % path_xyws.shape[0]
        new_zone                    = Marker(header=Header(stamp=rospy.Time.now(),
                                                           frame_id='map'))
        new_zone.type               = new_zone.CUBE
        new_zone.action             = new_zone.ADD
        new_zone.id                 = c
        new_zone.color              = ColorRGBA(r=1.000, g=0.616, b=0.000, a=0.5)
        new_zone.scale              = Vector3(x=0.05, y=1.0, z=1.0)
        new_zone.pose.position      = Point(x=path_xyws[k,0], y=path_xyws[k,1], z=0.0)
        new_zone.pose.orientation   = q_from_yaw(path_xyws[k,2])

        zones.markers.append(new_zone)
    return zones

def publish_reversible_xyw_pose(pose_xyw: list, pub: Union[rospy.Publisher, ROSPublisher],
                                frame_id='map', reverse: bool = False) -> None:
    '''
    TODO
    '''
    pose = [*pose_xyw] # decouple from input
    if reverse:
        pose[2] = angle_wrap(pose[2] + np.pi, 'RAD')
    publish_xyw_pose(pose, pub, frame_id)

def publish_xyw_pose(pose_xyw: list, pub: Union[rospy.Publisher, ROSPublisher],
                     frame_id='map') -> None:
    '''
    TODO
    '''
    # Update visualisation of current goal/target pose
    goal                    = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id=frame_id))
    goal.pose.position      = Point(x=pose_xyw[0], y=pose_xyw[1], z=0.0)
    goal.pose.orientation   = q_from_yaw(pose_xyw[2])
    pub.publish(goal)

def publish_xyzrpy_pose(xyzrpy: list, pub: Union[rospy.Publisher, ROSPublisher],
                        frame_id='map') -> None:
    '''
    TODO
    '''
    # Update visualisation of current goal/target pose
    goal                    = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id=frame_id))
    goal.pose.position      = Point(x=xyzrpy[0], y=xyzrpy[1], z=xyzrpy[2])
    goal.pose.orientation   = q_from_rpy(r=xyzrpy[3],p=xyzrpy[4],y=xyzrpy[5])
    pub.publish(goal)
