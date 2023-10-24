#! /usr/bin/env python3

import rospkg
import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')
import copy
from tqdm.auto import tqdm

from pyaarapsi.vpr_simple.vpr_helpers       import FeatureType
from pyaarapsi.vpr_simple.vpr_dataset_tool  import VPRDatasetProcessor
from pyaarapsi.vpr_simple.svm_model_tool    import SVMModelProcessor
from pyaarapsi.core.helper_tools            import m2m_dist, angle_wrap, angle_wrap
from pyaarapsi.core.ros_tools               import rip_bag, compressed2np, pose2xyw
from pyaarapsi.core.enum_tools              import enum_name
from pyaarapsi.core.transforms              import apply_homogeneous_transform, homogeneous_transform
from pyaarapsi.pathing.basic                import calc_path_stats
from pyaarapsi.vpred.vpred_tools            import find_vpr_performance_metrics


root_path   = rospkg.RosPack().get_path("aarapsi_robot_pack")
npz_dbp     = "/data/compressed_sets"
svm_dbp     = "/cfg/svm_models"
bag_dbp     = "/data/rosbags"
odom_topic  = "/odom/true"
enc_topic   = "/jackal_velocity_controller/odom"
img_topics  = ["/ros_indigosdk_occam/image0/compressed"]
ft_type     = FeatureType.NORM
img_dims    = [128,128]
filters     = ''#'{"distance": 0.1}'
factors     = ['va', 'grad']
tol_mode    = 'DISTANCE'
sample_rate = 5 # Hz
tol_thresh  = 0.5 # m

ref_dict        = dict( bag_name='s4_ccw_1', npz_dbp=npz_dbp, bag_dbp=bag_dbp, \
                            odom_topic=odom_topic, img_topics=img_topics, sample_rate=sample_rate, \
                            ft_types=[enum_name(ft_type)], img_dims=img_dims, filters=filters)
qry_dict        = dict( bag_name='s5_ccw_1', npz_dbp=npz_dbp, bag_dbp=bag_dbp, \
                            odom_topic=odom_topic, img_topics=img_topics, sample_rate=sample_rate, \
                            ft_types=[enum_name(ft_type)], img_dims=img_dims, filters=filters)
ref_ip          = VPRDatasetProcessor(ref_dict, try_gen=True, cuda=True, use_tqdm=True, autosave=True, ros=False, init_netvlad=True, printer=None)
qry_ip          = VPRDatasetProcessor(None, try_gen=True, cuda=True, use_tqdm=True, autosave=True, ros=False, init_netvlad=True, printer=None)
qry_ip.pass_nns(ref_ip, netvlad=True, hybridnet=False)
qry_ip.load_dataset(qry_dict, try_gen=True)

svm_svm_dict    = dict(factors=factors, tol_thres=tol_thresh, tol_mode=tol_mode)
svm_dict        = dict(ref=ref_dict, qry=qry_dict, svm=svm_svm_dict, npz_dbp=npz_dbp, bag_dbp=bag_dbp, svm_dbp=svm_dbp)

svm             = SVMModelProcessor(ros=False, root=None, load_field=True, printer=None)
svm.pass_nns(ref_ip, netvlad=True, hybridnet=False)
svm.load_model(svm_dict, try_gen=True, gen_datasets=True, save_datasets=True)

print(svm.field)