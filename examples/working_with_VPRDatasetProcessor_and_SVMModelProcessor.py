#! /usr/bin/env python3

import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')
import copy
from tqdm.auto import tqdm

import rospkg
import rospy
import copy
import cv2
import os
import sys
import torch

from pyaarapsi.vpr_simple.vpr_helpers       import FeatureType
from pyaarapsi.vpr_simple.vpr_dataset_tool  import VPRDatasetProcessor
from pyaarapsi.vpr_simple.svm_model_tool    import SVMModelProcessor
from pyaarapsi.core.helper_tools            import m2m_dist, angle_wrap, angle_wrap, vis_dict
from pyaarapsi.core.ros_tools               import rip_bag, compressed2np, pose2xyw
from pyaarapsi.core.enum_tools              import enum_name
from pyaarapsi.core.transforms              import apply_homogeneous_transform, homogeneous_transform
from pyaarapsi.pathing.basic                import calc_path_stats
from pyaarapsi.vpred.vpred_tools            import find_vpr_performance_metrics
from pyaarapsi.core.vars                    import C_RESET, C_I_BLUE

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

root_path   = rospkg.RosPack().get_path("aarapsi_robot_pack")
video_path  = root_path + "/data/videos/"
npz_dbp     = "/data/compressed_sets"
svm_dbp     = "/cfg/svm_models"
bag_dbp     = "/data/rosbags"
odom_topic  = "/odom/true"
enc_topic   = "/jackal_velocity_controller/odom"
img_topics  = ["/ros_indigosdk_occam/image0/compressed"]
ft_type     = FeatureType.RAW
img_dims    = [64,64]
filters     = ''#'{"distance": 0.1}'
factors     = ['va', 'grad']
tol_mode    = 'DISTANCE'
sample_rate = 5 # Hz
tol_thresh  = 0.5 # m

# How to use VPRDatasetProcessor to generate data from a rosbag:
ref_dict        = dict( bag_name='s4_ccw_1', npz_dbp=npz_dbp, bag_dbp=bag_dbp, \
                            odom_topic=odom_topic, img_topics=img_topics, sample_rate=sample_rate, \
                            ft_types=[enum_name(ft_type)], img_dims=img_dims, filters=filters)
qry_dict        = dict( bag_name='s5_ccw_1', npz_dbp=npz_dbp, bag_dbp=bag_dbp, \
                            odom_topic=odom_topic, img_topics=img_topics, sample_rate=sample_rate, \
                            ft_types=[enum_name(ft_type)], img_dims=img_dims, filters=filters)
ref_ip          = VPRDatasetProcessor(ref_dict, try_gen=True, cuda=True, use_tqdm=True, autosave=True, ros=False, init_netvlad=True, printer=None)
qry_ip          = VPRDatasetProcessor(None, try_gen=True, cuda=True, use_tqdm=True, autosave=True, ros=False, init_netvlad=False, printer=None)
qry_ip.pass_nns(ref_ip, netvlad=True, hybridnet=False)
qry_ip.load_dataset(qry_dict, try_gen=True)

# How to use SVMModelProcessor to generate an SVM:
svm_svm_dict    = dict(factors=factors, tol_thres=tol_thresh, tol_mode=tol_mode)
svm_dict        = dict(ref=ref_dict, qry=qry_dict, svm=svm_svm_dict, npz_dbp=npz_dbp, bag_dbp=bag_dbp, svm_dbp=svm_dbp)

svm             = SVMModelProcessor(ros=False, root=None, load_field=True, printer=None)
svm.pass_nns(ref_ip, netvlad=True, hybridnet=False)
svm.load_model(svm_dict, try_gen=True, gen_datasets=True, save_datasets=True)

def feature2image(img, dims, dstack=True):
    _min      = np.min(img)
    _max      = np.max(img)
    _img_norm = (img - _min) / (_max - _min)
    _img_uint = np.array(_img_norm * 255, dtype=np.uint8)
    _img_dims = np.reshape(_img_uint, dims)
    if dstack: return np.dstack((_img_dims,)*3)
    return _img_dims

def make_video(_video_path: str, _ip_dict: dict, nnip=None, netvlad=False, hybridnet=False, fps=40):
    if nnip is None:
        _ip = VPRDatasetProcessor(_ip_dict, try_gen=True, cuda=True, use_tqdm=True, autosave=True, ros=False, init_netvlad=netvlad, init_hybridnet=hybridnet, printer=None)
    else:
        _ip = VPRDatasetProcessor(None, try_gen=True, cuda=True, use_tqdm=True, autosave=True, ros=False, init_netvlad=False, printer=None)
        _ip.pass_nns(nnip, netvlad=netvlad, hybridnet=hybridnet)
        _ip.load_dataset(_ip_dict, try_gen=True)

    file_path = _video_path + _ip.dataset['params']['bag_name'] + ('_%dx%d_RAW.avi' % tuple(_ip.dataset['params']['img_dims']))
    if os.path.isfile(file_path): os.remove(file_path)
    vid_writer = cv2.VideoWriter(file_path, 0, fps, _ip.dataset['params']['img_dims'])
    for i in range(len(_ip.dataset['dataset']['time'])):
        img = feature2image(_ip.dataset['dataset']['RAW'][i], _ip.dataset['params']['img_dims'])
        vid_writer.write(img)
    vid_writer.release()

for bag_name in ['sim_cw_3', 'sim_cw_5', 's5_ccw_1', 's6_ccw_1', 's4_cw_1', 'lab_adv_2', 'll_cw_1', 'll_cw_3', 'run3_fix', 'run4_fix', 'run1_fix', 'run5_ccw']:
    print(C_I_BLUE + 'Generating video for ' + bag_name + "..." + C_RESET)
    ip_dict = dict( bag_name=bag_name, npz_dbp=npz_dbp, bag_dbp=bag_dbp, \
                    odom_topic=[odom_topic, enc_topic], img_topics=img_topics, sample_rate=sample_rate, \
                    ft_types=[enum_name(ft_type)], img_dims=img_dims, filters=filters)
    make_video(video_path, ip_dict)

#vis_dict(qry_ip.dataset)

