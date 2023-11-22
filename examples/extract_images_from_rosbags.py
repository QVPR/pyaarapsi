#! /usr/bin/env python3

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import rospkg
import sys
import cv2

from pyaarapsi.vpr_simple.vpr_helpers       import FeatureType
from pyaarapsi.vpr_simple.vpr_dataset_tool  import VPRDatasetProcessor
from pyaarapsi.core.enum_tools              import enum_name
from pyaarapsi.core.vars                    import C_RESET, C_I_BLUE
from pyaarapsi.core.ros_tools               import process_bag

print('Initialisation...')
root_path   = rospkg.RosPack().get_path("aarapsi_robot_pack")
video_path  = root_path + "/data/videos/"
npz_dbp     = "/data/compressed_sets"
svm_dbp     = "/cfg/svm_models"
bag_dbp     = "/data/rosbags"
odom_topic  = "/odom/true"
enc_topic   = "/jackal_velocity_controller/odom"
img_topics  = ["/ros_indigosdk_occam/image0/compressed"]
ft_type     = FeatureType.RAW
img_dims    = [720,480]
filters     = ''#'{"distance": 0.1}'
factors     = ['va', 'grad']
tol_mode    = 'DISTANCE'
sample_rate = 0.1 # Hz
tol_thresh  = 0.5 # m

def feature2image(img, dims, dstack=True):
    _min      = np.min(img)
    _max      = np.max(img)
    _img_norm = (img - _min) / (_max - _min)
    _img_uint = np.array(_img_norm * 255, dtype=np.uint8)
    _img_dims = np.reshape(_img_uint, dims)
    if dstack: return np.dstack((_img_dims,)*3)
    return _img_dims

print('Starting...')
for bag_name in ['sim_cw_3', 'sim_cw_5', 's5_ccw_1', 's6_ccw_1', 's4_cw_1', 'lab_adv_2', 'll_cw_1', 'll_cw_3', 'run3_fix', 'run4_fix', 'run1_fix', 'run5_ccw']:
    print(C_I_BLUE + 'Getting data from ' + bag_name + "..." + C_RESET)
    rosbag_dict = process_bag(root_path +  '/' + bag_dbp + '/' + bag_name, sample_rate, odom_topic, img_topics, printer=print, use_tqdm=True)
    for c, i in enumerate(list(rosbag_dict[img_topics[0]])):
        cv2.imwrite('/home/claxton/Desktop/frames/' + bag_name + ('_%d'%c) + '.png', i[:,:,-1::-1])


print('Done!')
    