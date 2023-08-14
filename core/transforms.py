#!/usr/bin/env python3
import numpy as np

def rotation_matrix_2d(y: float, radians=True):
    if not radians:
        y = y * np.pi / 180
    c, s = np.cos(y), np.sin(y)
    R = np.array([[c,-s],[s,c]])
    return R

def rotation_matrix_xy(y: float, radians=True):
    if not radians:
        y = y * np.pi / 180
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz

def rotation_matrix_xz(p: float, radians=True):
    if not radians:
        p = p * np.pi / 180
    cp, sp = np.cos(p), np.sin(p)
    Ry = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
    return Ry

def rotation_matrix_yz(r: float, radians=True):
    if not radians:
        r = r * np.pi / 180
    cr, sr = np.cos(r), np.sin(r)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rx

def rotation_matrix_3d(r: float, p: float, y: float, radians=True):
    Rx = rotation_matrix_yz(r, radians=radians)
    Ry = rotation_matrix_xz(p, radians=radians)
    Rz = rotation_matrix_xy(y, radians=radians)
    R = Rx * Ry * Rz
    return R

def homogeneous_transform(r: float, p: float, y: float, X: float, Y: float, Z: float, radians=True):
    R = rotation_matrix_3d(r, p, y, radians=radians)
    H = np.eye(4)
    H[0:3,0:3] = R
    H[0:3,3] = np.array([X,Y,Z])
    return H

def apply_homogeneous_transform(H: np.ndarray, X: list, Y: list, Z: list = None, cast_to_list =- True):
    if Z is None:
        Z = [0] * len(X)
    assert len(X) == len(Y) == len(Z), 'Number of elements in X must match that of in Y and Z'
    xyz_ = np.stack([np.array(X).flatten(),np.array(Y).flatten(), np.array(Z).flatten(), np.ones(len(X)).flatten()])
    XYZ_ = np.matmul(H, xyz_)
    if cast_to_list:
        return list(XYZ_[0,:]), list(XYZ_[1,:]), list(XYZ_[2,:])
    return XYZ_[0,:], XYZ_[1,:], XYZ_[2,:]
