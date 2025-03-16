#!/usr/bin/env python3
'''
Tools for transforms
'''
from typing import Optional, Tuple, Literal, overload, Union
import numpy as np
from numpy.typing import NDArray

def rotation_matrix_to_euler_angles(rot_mat: NDArray) -> NDArray:
    '''
    Convert a 3x3 Rotation Matrix to roll, pitch, yaw Euler angles
    https://learnopencv.com/rotation-matrix-to-euler-angles/
    Inputs:
    - R: np.ndarray type; 3x3 Rotation Matrix
    Returns:
    np.ndarray type; [roll, pitch, yaw]
    '''
    sy = np.sqrt(rot_mat[0,0] * rot_mat[0,0] +  rot_mat[1,0] * rot_mat[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2( rot_mat[2,1] , rot_mat[2,2])
        y = np.arctan2(-rot_mat[2,0], sy)
        z = np.arctan2( rot_mat[1,0], rot_mat[0,0])
    else:
        x = np.arctan2(-rot_mat[1,2], rot_mat[1,1])
        y = np.arctan2(-rot_mat[2,0], sy)
        z = 0
    return np.array([x, y, z])

def rotation_matrix_2d(y: float, radians=True):
    '''
    Create a 2D rotation matrix
    Inputs:
    - y:        float type; yaw angle
    - radians:  bool type {default: True}; set unit of y to radians
    Returns:
    - np.ndarray type (2x2) rotation matrix
    '''
    if not radians:
        y = y * np.pi / 180
    c, s = np.cos(y), np.sin(y)
    rot_mat = np.array([[c,-s],[s,c]])
    return rot_mat

def rotation_matrix_xy(y: float, radians=True):
    '''
    Create a 3D rotation matrix for pitch (x-y plane).

    Inputs:
    - y:        float type; yaw angle
    - radians:  bool type {default: True}; set unit of y to radians
    Returns:
    - np.ndarray type (3x3) rotation matrix
    '''
    if not radians:
        y = y * np.pi / 180
    cy, sy = np.cos(y), np.sin(y)
    rot_z_mat = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return rot_z_mat

def rotation_matrix_xz(p: float, radians=True):
    '''
    Create a 3D rotation matrix for pitch (x-z plane).

    Inputs:
    - p:        float type; pitch angle
    - radians:  bool type {default: True}; set unit of p to radians
    Returns:
    - np.ndarray type (3x3) rotation matrix
    '''
    if not radians:
        p = p * np.pi / 180
    cp, sp = np.cos(p), np.sin(p)
    rot_y_mat = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
    return rot_y_mat

def rotation_matrix_yz(r: float, radians=True):
    '''
    Create a 3D rotation matrix for roll (y-z plane).

    Inputs:
    - r:        float type; roll angle
    - radians:  bool type {default: True}; set unit of r to radians
    Returns:
    - np.ndarray type (3x3) rotation matrix
    '''
    if not radians:
        r = r * np.pi / 180
    cr, sr = np.cos(r), np.sin(r)
    rot_x_mat = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return rot_x_mat

def rotation_matrix_3d(r: float, p: float, y: float, order='rpy', radians=True):
    '''
    Create a 3D rotation matrix by providing roll, pitch, yaw and order of application.

    Inputs:
    - r:        float type; roll angle
    - p:        float type; pitch angle
    - y:        float type; yaw angle
    - order:    str type {default: 'rpy'}; order of application (some combination of 'rpy')
    - radians:  bool type {default: True}; set unit of r,p,y to radians
    Returns:
    - np.ndarray type (3x3) rotation matrix
    '''
    assert sum([i in order for i in 'rpy']) == 3 and len(order) == 3, \
        'Order must be some combination of "rpy"'
    individual_rot_mats = { 'r': rotation_matrix_yz(r, radians=radians),
                            'p': rotation_matrix_xz(p, radians=radians),
                            'y': rotation_matrix_xy(y, radians=radians) }
    rot_mat = np.matmul(individual_rot_mats[order[2]], \
                        np.matmul(individual_rot_mats[order[1]], individual_rot_mats[order[0]]))
    return rot_mat

def homogeneous_transform(r_ang: float, p_ang: float, y_ang: float, \
                          x_lin: float, y_lin: float, z_lin: float, order='rpy', radians=True):
    '''
    Input rotation angles (roll, pitch, yaw) and linear translations (x, y, z) to construct a
        4x4 homogeneous transform matrix
    Inputs:
    - r:        float type; roll angle
    - p:        float type; pitch angle
    - y:        float type; yaw angle
    - x:        float type; x-axis linear translation
    - y:        float type; y-axis linear translation
    - z:        float type; z-axis linear translation
    - order:    str type {default: 'rpy'}; order of application (some combination of 'rpy')
    - radians:  bool type {default: True}; set unit of r,p,y to radians
    Returns:
    - np.ndarray type (4x4) homogeneous transform matrix
    '''
    rot_mat = rotation_matrix_3d(r_ang, p_ang, y_ang, order=order, radians=radians)
    h_transform = np.eye(4)
    h_transform[0:3,0:3] = rot_mat
    h_transform[0:3,3] = np.array([x_lin, y_lin, z_lin])
    return h_transform

@overload
def apply_homogeneous_transform(h_transform: NDArray, x_data: list, y_data: list, \
                                z_data: Optional[list] = None, cast_to_list: Literal[True] = True
                                ) -> Tuple[list, list, list]:
    ...

@overload
def apply_homogeneous_transform(h_transform: NDArray, x_data: list, y_data: list, \
                                z_data: Optional[list] = None, cast_to_list: Literal[False] = True
                                ) -> Tuple[NDArray, NDArray, NDArray]:
    ...

def apply_homogeneous_transform(h_transform: NDArray, x_data: list, y_data: list, \
                                z_data: Optional[list] = None, cast_to_list: bool = True
                                ) -> Tuple[Union[NDArray, list], Union[NDArray, list], \
                                           Union[NDArray, list]]:
    '''
    Input a homogeneous transform and lists of X, Y, Z values for points to transform

    Inputs:
    - H: np.ndarray type; homogeneous transform matrix (4x4)
    - X: list type; x coordinate list of points array
    - Y: list type; y coordinate list of points array
    - Z: list type; z coordinate list of points array
    - cast_to_list: bool type; whether to return the result arrays as lists rather than np.ndarrays
    Returns:
    - (X, Y, Z); transformed point coordinate lists
    '''
    x_data_in = np.array(x_data) if isinstance(x_data, list) else x_data
    y_data_in = np.array(y_data) if isinstance(y_data, list) else y_data
    if z_data is None:
        z_data_in = np.array([0] * len(x_data))
    else:
        z_data_in = np.array(z_data) if isinstance(z_data, list) else z_data
    assert isinstance(x_data_in, np.ndarray) and isinstance(y_data_in, np.ndarray) \
        and isinstance(z_data_in, np.ndarray)
    assert h_transform.shape == (4,4), \
        'Homogeneous transform dimensions incorrect; Criteria: H.shape == (4,4)'
    assert x_data_in.size == y_data_in.size == z_data_in.size, \
        'Number of elements in X must match that of in Y and Z'
    flat_xyz = np.stack([x_data_in.flatten(), y_data_in.flatten(), z_data_in.flatten(), \
                         np.ones(len(z_data_in)).flatten()])
    transformed_xyz = np.matmul(h_transform, flat_xyz)
    if cast_to_list:
        return list(transformed_xyz[0,:]), list(transformed_xyz[1,:]), list(transformed_xyz[2,:])
    return transformed_xyz[0,:], transformed_xyz[1,:], transformed_xyz[2,:]

class TransformBuilder(object):
    '''
    Tool to construct a homogeneous transformation matrix
    '''
    def __init__(self):
        self._transform  = np.eye(4)
        self._components = []
    #
    def translate(self, x: float = 0, y: float = 0, z: float = 0):
        '''
        Add a translation to the homogeneous transform matrix
        '''
        new_transform = homogeneous_transform(0,0,0,x,y,z)
        self._components.append(new_transform)
        self._transform = np.matmul(new_transform, self._transform)
        return self
    #
    def rotate(self, r: float = 0, p: float = 0, y: float = 0, order: str = 'rpy', radians=True):
        '''
        Add a rotation to the homogeneous transform matrix
        '''
        new_transform   = homogeneous_transform(r,p,y,0,0,0,order=order,radians=radians)
        self._transform = np.matmul(new_transform, self._transform)
        self._components.append(new_transform)
        return self
    #
    def get(self):
        '''
        Get transform
        '''
        return self._transform
    #
    def get_components(self):
        '''
        Get transform components
        '''
        return self._components
