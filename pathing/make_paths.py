import numpy as np
import rospy
from nav_msgs.msg                   import Path
from std_msgs.msg                   import Header, ColorRGBA
from geometry_msgs.msg              import PoseStamped, Point, Vector3
from visualization_msgs.msg         import MarkerArray, Marker

from pyaarapsi.core.ros_tools       import q_from_yaw
from pyaarapsi.core.helper_tools    import angle_wrap

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
    path       = Path(header=Header(stamp=rospy.Time.now(), frame_id="map"))
    speeds     = MarkerArray()
    direction  = np.sign(np.sum(path_xyws[:,2]))
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
    zones = MarkerArray()
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
                