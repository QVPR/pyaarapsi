#!/usr/bin/env python3
import numpy as np

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
    R = np.array([[c,-s],[s,c]])
    return R

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
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz

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
    Ry = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
    return Ry

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
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rx

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
    assert sum([i in order for i in 'rpy']) == 3 and len(order) == 3, 'Order must be some combination of "rpy"'
    Rs_ = { 'r': rotation_matrix_yz(r, radians=radians), 
            'p': rotation_matrix_xz(p, radians=radians), 
            'y': rotation_matrix_xy(y, radians=radians) }
    R = np.matmul(Rs_[order[2]], np.matmul(Rs_[order[1]], Rs_[order[0]]))
    return R

def homogeneous_transform(r: float, p: float, y: float, X: float, Y: float, Z: float, order='rpy', radians=True):
    '''
    Input rotation angles (roll, pitch, yaw) and linear translations (x, y, z) to construct a 4x4 homogeneous transform matrix

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
    R = rotation_matrix_3d(r, p, y, order=order, radians=radians)
    H = np.eye(4)
    H[0:3,0:3] = R
    H[0:3,3] = np.array([X,Y,Z])
    return H

def apply_homogeneous_transform(H: np.ndarray, X: list, Y: list, Z: list = None, cast_to_list = True):
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
    if Z is None:
        Z = [0] * len(X)
    assert H.shape == (4,4), 'Homogeneous transform dimensions incorrect; Criteria: H.shape == (4,4)'
    assert len(X) == len(Y) == len(Z), 'Number of elements in X must match that of in Y and Z'
    xyz_ = np.stack([np.array(X).flatten(),np.array(Y).flatten(), np.array(Z).flatten(), np.ones(len(X)).flatten()])
    XYZ_ = np.matmul(H, xyz_)
    if cast_to_list:
        return list(XYZ_[0,:]), list(XYZ_[1,:]), list(XYZ_[2,:])
    return XYZ_[0,:], XYZ_[1,:], XYZ_[2,:]
