#! /usr/bin/env python3
'''
Basic pathing helper functions
'''
import warnings
from typing                         import Union, List, Tuple
import numpy as np
from numpy.typing                   import NDArray

try:
    #pylint: disable=W0611
    from pyaarapsi.pathing.basic_rospy import make_path_speeds, make_zones, \
        publish_reversible_xyw_pose, publish_xyw_pose, publish_xyzrpy_pose
    #pylint: enable=W0611
except ImportError:
    warnings.warn("Failed to import ROS-related packages and message structures: some functions"
                  "will error.")

from pyaarapsi.core.helper_tools    import angle_wrap, normalize_angle, m2m_dist
from pyaarapsi.pathing.enums        import Lookahead_Mode

def make_speed_array(w_interp):
    '''
    TODO
    '''
    # Generate speed profile based on curvature of track:
    points_diff     = np.abs(angle_wrap(np.roll(w_interp, 1, 0) - np.roll(w_interp, -1, 0),
                                        mode='RAD'))
    path_density    = np.mean(np.sum([np.roll(points_diff, i, 0) for i in [-1,0,1]], axis=0))
    k = int(2 / path_density)
    if k % 2 == 1:
        k = k + 1
    points_smooth   = np.sum([np.roll(points_diff, i, 0) for i in np.arange(k + 1)-int(k/2)],
                                axis=0)
    s_interp        = (1 - ((points_smooth - np.min(points_smooth)) \
                            / (np.max(points_smooth) - np.min(points_smooth)))) ** 2
    s_interp[s_interp<np.mean(s_interp)/2] = np.mean(s_interp)/2

    return s_interp

def calc_path_stats(path_xyws: NDArray[np.float32]) -> Tuple[NDArray[np.float32], float]:
    '''
    TODO
    '''
    path_dists      = np.sqrt( \
                            np.square(path_xyws[:,0] - np.roll(path_xyws[:,0], 1)) + \
                            np.square(path_xyws[:,1] - np.roll(path_xyws[:,1], 1)) \
                        )[1:]
    path_sum        = [0.0]
    for i in path_dists:
        path_sum.append(np.sum([path_sum[-1], i]))

    path_len        = path_sum[-1]
    path_sum        = np.array(path_sum)
    return path_sum, path_len

def calc_zone_stats(path_len: float, len_guess: float, num_guess: Union[float, int]
                    ) -> Tuple[float, int]:
    '''
    TODO
    '''
    if path_len / len_guess > num_guess:
        num_true    = int(num_guess)
    else:
        num_true    = int(path_len / len_guess)
    len_true        = path_len / num_true

    return len_true, num_true



def calc_path_errors(ego, current_ind: int, path_xyws: NDArray[np.float32]
                     ) -> Tuple[float, float]:
    '''
    TODO
    '''
    _dx     = ego[0] - path_xyws[current_ind, 0]
    _dy     = ego[1] - path_xyws[current_ind, 1]
    _dw     = np.arctan2(_dy, _dx)
    _dr     = np.sqrt(np.square(_dx) + np.square(_dy))
    lin_err = abs(_dr * np.sin(path_xyws[current_ind, 2] - _dw))
    ang_err = abs(angle_wrap(path_xyws[current_ind, 2] - ego[2], 'RAD'))
    return lin_err, ang_err

def global2local(ego, x: Union[float, NDArray[np.float32]], y: Union[float, NDArray[np.float32]]
                 ) -> Tuple[List[float], List[float]]:
    '''
    TODO
    '''
    tx  = x - ego[0]
    ty  = y - ego[1]
    arr_r   = np.sqrt(np.power(tx, 2) + np.power(ty, 2))
    arr_a   = np.arctan2(ty, tx) - ego[2]
    return list(np.multiply(np.cos(arr_a), arr_r)), list(np.multiply(np.sin(arr_a), arr_r))

def calc_yaw_error(ego, goal) -> float:
    '''
    Heading error (angular):
    '''
    # Convert to local coordinates
    rel_x, rel_y    = global2local(ego, np.array([goal[0]]), np.array([goal[1]]))
    error_yaw       = normalize_angle(np.arctan2(rel_y[0], rel_x[0]))
    return error_yaw

def calc_y_error(ego, goal) -> float:
    '''
    Heading error, linear
    '''
    # Convert to local coordinates
    return global2local(ego, np.array([goal[0]]), np.array([goal[1]]))[1][0]

def calc_nearest_zone(zone_indices: List[int], current_ind: int, _len: int) -> int:
    '''
    TODO
    '''
    return zone_indices[np.argmin(m2m_dist(current_ind,
                                           np.transpose(np.matrix(zone_indices))))] % _len

def calc_current_zone(ind: int, num_zones: int, zone_indices: list) -> int:
    '''
    The closest zone boundary that is 'behind' the closest index (in the direction of the path)
    '''
    return np.max(np.arange(num_zones)[np.array(zone_indices[0:-1]) <= ind] + 1)

def calc_current_ind(ego, path_xyws: NDArray[np.float32]) -> int:
    '''
    Closest index based on provided ego
    '''
    return np.argmin(m2m_dist(path_xyws[:,0:2], ego[0:2], True), axis=0)

class BadLookaheadMode(Exception):
    '''
    Bad lookahead.
    '''

def calc_target(current_ind: int, lookahead: float, lookahead_mode: Lookahead_Mode,
                path_xyws: NDArray[np.float32], path_sum: NDArray[np.float32],
                reverse: bool = False):
    '''
    TODO
    '''
    adj_lookahead = np.max([lookahead * (path_xyws[current_ind, 3] \
                                         / np.max(path_xyws[:, 3].flatten())), 0.4])
    if reverse:
        adj_lookahead *= -1
    if lookahead_mode == Lookahead_Mode.INDEX:
        target_ind  = (current_ind + int(np.round(adj_lookahead))) % path_xyws.shape[0]
    elif lookahead_mode == Lookahead_Mode.DISTANCE:
        target_ind    = int(np.argmin(np.abs(path_sum - ((path_sum[current_ind] + adj_lookahead)
                                                         % path_sum[-1]))))
        dist = (path_sum[target_ind] - path_sum[current_ind]) % path_sum[-1]
        while dist < np.abs(adj_lookahead):
            target_ind = int((target_ind + np.sign(adj_lookahead)) % path_sum.shape[0])
            dist = (path_sum[target_ind] - path_sum[current_ind]) % path_sum[-1]
        adj_lookahead = dist
    else:
        raise BadLookaheadMode(f'Unknown lookahead_mode: {str(lookahead_mode)}')
    return target_ind, adj_lookahead
                