import numpy as np
import rospy
from nav_msgs.msg                   import Path
from std_msgs.msg                   import Header, ColorRGBA
from geometry_msgs.msg              import PoseStamped, Point, Vector3
from visualization_msgs.msg         import MarkerArray, Marker

from pyaarapsi.core.ros_tools       import q_from_yaw, q_from_rpy, ROS_Publisher
from pyaarapsi.core.helper_tools    import angle_wrap, normalize_angle, m2m_dist
from .enums                         import Lookahead_Mode
from typing import Union

def make_speed_array(w_interp):
    # Generate speed profile based on curvature of track:
    points_diff     = np.abs(angle_wrap(np.roll(w_interp, 1, 0) - np.roll(w_interp, -1, 0), mode='RAD'))
    path_density    = np.mean(np.sum([np.roll(points_diff, i, 0) for i in [-1,0,1]], axis=0))
    k = int(2 / path_density)
    if k % 2 == 1:
        k = k + 1
    points_smooth   = np.sum([np.roll(points_diff, i, 0) for i in np.arange(k + 1)-int(k/2)], axis=0)
    s_interp        = (1 - ((points_smooth - np.min(points_smooth)) / (np.max(points_smooth) - np.min(points_smooth)))) **2
    s_interp[s_interp<np.mean(s_interp)/2] = np.mean(s_interp)/2

    return s_interp

def calc_path_stats(path_xyws):
    path_dists      = np.sqrt( \
                            np.square(path_xyws[:,0] - np.roll(path_xyws[:,0], 1)) + \
                            np.square(path_xyws[:,1] - np.roll(path_xyws[:,1], 1)) \
                        )[1:]
    path_sum        = [0]
    for i in path_dists:
        path_sum.append(np.sum([path_sum[-1], i]))
    path_len        = path_sum[-1]

    return path_sum, path_len

def calc_zone_stats(path_len, len_guess, num_guess):
    

    if path_len / len_guess > num_guess:
        num_true    = int(num_guess)
    else:
        num_true    = int(path_len / len_guess)
    len_true        = path_len / num_true

    return len_true, num_true

def make_path_speeds(path_xyws, path_indices):
    path            = Path(header=Header(stamp=rospy.Time.now(), frame_id="map"))
    path.poses      = []
    speeds          = MarkerArray()
    speeds.markers  = []
    direction       = np.sign(np.sum(path_xyws[:,2]))
    for i in path_indices:
        new_pose                        = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id="map"))
        new_pose.pose.position          = Point(x=path_xyws[i,0], y=path_xyws[i,1], z=0.0)
        new_pose.pose.orientation       = q_from_yaw(path_xyws[i,2])

        new_speed                       = Marker(header=Header(stamp=rospy.Time.now(), frame_id='map'))
        new_speed.type                  = new_speed.ARROW
        new_speed.action                = new_speed.ADD
        new_speed.id                    = i
        new_speed.color                 = ColorRGBA(r=0.859, g=0.094, b=0.220, a=0.5)
        new_speed.scale                 = Vector3(x=path_xyws[i,3], y=0.05, z=0.05)
        new_speed.pose.position         = Point(x=path_xyws[i,0], y=path_xyws[i,1], z=0.0)
        new_speed.pose.orientation      = q_from_yaw(path_xyws[i,2] - direction * np.pi/2)

        path.poses.append(new_pose)
        speeds.markers.append(new_speed)
    return path, speeds

def make_zones(path_xyws, zone_indices):
    zones           = MarkerArray()
    zones.markers   = []
    for c, i in enumerate(zone_indices):
        k                           = i % path_xyws.shape[0]
        new_zone                    = Marker(header=Header(stamp=rospy.Time.now(), frame_id='map'))
        new_zone.type               = new_zone.CUBE
        new_zone.action             = new_zone.ADD
        new_zone.id                 = c
        new_zone.color              = ColorRGBA(r=1.000, g=0.616, b=0.000, a=0.5)
        new_zone.scale              = Vector3(x=0.05, y=1.0, z=1.0)
        new_zone.pose.position      = Point(x=path_xyws[k,0], y=path_xyws[k,1], z=0.0)
        new_zone.pose.orientation   = q_from_yaw(path_xyws[k,2])

        zones.markers.append(new_zone)
    return zones

def calc_path_errors(ego, current_ind, path_xyws):
    _dx     = ego[0] - path_xyws[current_ind, 0]
    _dy     = ego[1] - path_xyws[current_ind, 1]
    _dw     = np.arctan2(_dy, _dx)
    _dr     = np.sqrt(np.square(_dx) + np.square(_dy))
    lin_err = abs(_dr * np.sin(path_xyws[current_ind, 2] - _dw))
    ang_err = abs(angle_wrap(path_xyws[current_ind, 2] - ego[2], 'RAD'))
    return lin_err, ang_err

def global2local(ego, x, y):

    Tx  = x - ego[0]
    Ty  = y - ego[1]
    R   = np.sqrt(np.power(Tx, 2) + np.power(Ty, 2))
    A   = np.arctan2(Ty, Tx) - ego[2]

    return list(np.multiply(np.cos(A), R)), list(np.multiply(np.sin(A), R))

def calc_yaw_error(ego, goal) -> float:
    # Heading error (angular):
    rel_x, rel_y    = global2local(ego, np.array([goal[0]]), np.array([goal[1]])) # Convert to local coordinates
    error_yaw       = normalize_angle(np.arctan2(rel_y[0], rel_x[0]))
    return error_yaw

def calc_y_error(ego, goal) -> float:
    # Heading error (angular):
    rel_x, rel_y    = global2local(ego, np.array([goal[0]]), np.array([goal[1]])) # Convert to local coordinates
    return rel_y[0]

def calc_nearest_zone(zone_indices, current_ind, _len):
    return zone_indices[np.argmin(m2m_dist(current_ind, np.transpose(np.matrix(zone_indices))))] % _len
        
def calc_current_zone(ind: int, num_zones: int, zone_indices: list):
    # The closest zone boundary that is 'behind' the closest index (in the direction of the path):
    zone        = np.max(np.arange(num_zones)[np.array(zone_indices[0:-1]) <= ind] + 1)
    return zone
    
def calc_current_ind(ego, path_xyws):
    # Closest index based on provided ego:
    ind         = np.argmin(m2m_dist(path_xyws[:,0:2], ego[0:2], True), axis=0)
    return ind
    
def calc_target(current_ind: int, lookahead: float, lookahead_mode: Lookahead_Mode, path_xyws: np.ndarray):
    adj_lookahead = np.max([lookahead * (path_xyws[current_ind, 3]/np.max(path_xyws[:, 3].flatten())), 0.3])
    if lookahead_mode == Lookahead_Mode.INDEX:
        target_ind  = (current_ind + int(np.round(adj_lookahead))) % path_xyws.shape[0]
    elif lookahead_mode == Lookahead_Mode.DISTANCE:
        target_ind  = current_ind
        # find first index at least lookahead-distance-away from current index:
        dist        = np.sqrt(np.sum(np.square(path_xyws[target_ind, 0:2] - path_xyws[current_ind, 0:2])))
        while dist < adj_lookahead:
            target_ind  = (target_ind + 1) % path_xyws.shape[0] 
            dist        = np.sqrt(np.sum(np.square(path_xyws[target_ind, 0:2] - path_xyws[current_ind, 0:2])))
        adj_lookahead = dist
    else:
        raise Exception('Unknown lookahead_mode: %s' % str(lookahead_mode))
    return target_ind, adj_lookahead

def publish_xyw_pose(pose_xyw: list, pub: Union[rospy.Publisher, ROS_Publisher], frame_id='map') -> None:
    # Update visualisation of current goal/target pose
    goal                    = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id=frame_id))
    goal.pose.position      = Point(x=pose_xyw[0], y=pose_xyw[1], z=0.0)
    goal.pose.orientation   = q_from_yaw(pose_xyw[2])
    pub.publish(goal)

def publish_xyzrpy_pose(xyzrpy: list, pub: Union[rospy.Publisher, ROS_Publisher], frame_id='map') -> None:
    # Update visualisation of current goal/target pose
    goal                    = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id=frame_id))
    goal.pose.position      = Point(x=xyzrpy[0], y=xyzrpy[1], z=xyzrpy[2])
    goal.pose.orientation   = q_from_rpy(r=xyzrpy[3],p=xyzrpy[4],y=xyzrpy[5])
    pub.publish(goal)


                