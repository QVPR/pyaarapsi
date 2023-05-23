#!/usr/bin/env python3

import numpy as np
import os
from pathlib import Path
import datetime

import rospkg

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastdist import fastdist
#from sklearnex import patch_sklearn # Package for speeding up sklearn (must be run on GPU; TODO)
#patch_sklearn()

from sklearn import svm
from sklearn.preprocessing import StandardScaler

from ..core.file_system_tools import scan_directory
from ..core.ros_tools         import roslogger, LogType
from .vpr_dataset_tool        import VPRDatasetProcessor
from ..vpred import *

class SVMModelProcessor: # main ROS class
    def __init__(self, ros=False):

        self.model_ready    = False
        self.ros            = ros

        # for making new models (prep but don't load anything yet)
        self.cal_qry_ip     = VPRDatasetProcessor(None, ros=self.ros)
        self.cal_ref_ip     = VPRDatasetProcessor(None, ros=self.ros)

        self.print("[SVMModelProcessor] Processor Ready.")

    def print(self, text, logtype=LogType.INFO, throttle=0):
        roslogger(text, logtype, throttle=throttle, ros=self.ros)
            
    def generate_model(self, ref, qry, bag_dbp, npz_dbp, svm_dbp, save=True):
        assert ref['img_dims'] == qry['img_dims'], "Reference and query metadata must be the same."
        assert ref['ft_types'] == qry['ft_types'], "Reference and query metadata must be the same."
        # store for access in saving operation:
        self.bag_dbp            = bag_dbp
        self.npz_dbp            = npz_dbp
        self.svm_dbp            = svm_dbp
        self.cal_qry_params     = qry
        self.cal_ref_params     = ref 
        self.img_dims           = ref['img_dims']
        self.feat_type          = ref['ft_types'][0]
        self.model_ready        = False

        # generate:
        load_statuses = self._load_cal_data() # [qry, ref]
        if all(load_statuses):
            self._calibrate()
            self._train()
            self._make()
            self.model_ready    = True
            if save:
                self.save_model()
        return load_statuses

    def save_model(self, name=None):
        if not self.model_ready:
            raise Exception("Model not loaded in system. Either call 'generate_model' or 'load_model' before using this method.")
        dir = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/' + self.svm_dbp
        Path(dir).mkdir(parents=False, exist_ok=True)
        Path(dir+"/params").mkdir(parents=False, exist_ok=True)
        if self._check():
            self.print("[save_model] File exists with identical parameters: %s")
            return self
        
        # Ensure file name is of correct format, generate if not provided
        file_list, _, _, = scan_directory(dir, short_files=True)
        if (not name is None):
            if not (name.startswith('svmmodel')):
                name = "svmmodel_" + name
            if (name in file_list):
                raise Exception("Model with name %s already exists in directory." % name)
        else:
            name = datetime.datetime.today().strftime("svmmodel_%Y%m%d")

        # Check file_name won't overwrite existing models
        file_name = name
        count = 0
        while file_name in file_list:
            file_name = name + "_%d" % count
            count += 1
        
        full_file_path = dir + "/" + file_name
        full_param_path = dir + "/params/" + file_name
        np.savez(full_file_path, **self.model)
        np.savez(full_param_path, params=self.model['params']) # save whole dictionary to preserve key object types
        self.print("[save_model] Saved file, params to %s, %s" % (full_file_path, full_param_path))
        return self

    def load_model(self, model_params):
    # load via search for param match
        self.svm_dbp = model_params['svm_dbp']
        self.print("[load_model] Loading model.")
        self.model_ready = False
        models = self._get_models()
        self.model = {}
        for name in models:
            if models[name]['params'] == model_params:
                try:
                    self._load(name)
                    return True
                except:
                    self._fix(name)
        return False
    
    def swap(self, model_params, generate=False, allow_false=True):
        models = self._get_models()
        for name in models:
            if models[name]['params'] == model_params:
                try:
                    self._load(name)
                    return True
                except:
                    self._fix(name)
        if generate:
            self.generate_model(**model_params)
            return True
        if not allow_false:
            raise Exception('Model failed to load.')
        return False
    
    def predict(self, dvc):
        if not self.model_ready:
            raise Exception("Model not loaded in system. Either call 'generate_model' or 'load_model' before using this method.")
        sequence        = (dvc - self.model['model']['rmean']) / self.model['model']['rstd'] # normalise using parameters from the reference set
        factor1_qry     = find_va_factor(np.c_[sequence])[0]
        factor2_qry     = find_grad_factor(np.c_[sequence])[0]
        # rt for realtime; still don't know what 'X' and 'y' mean! TODO
        Xrt             = np.c_[factor1_qry, factor2_qry]      # put the two factors into a 2-column vector
        Xrt_scaled      = self.model['model']['scaler'].transform(Xrt)  # perform scaling using same parameters as calibration set
        y_zvalues_rt    = self.model['model']['svm'].decision_function(Xrt_scaled)[0] # 'z' value; not probability but "related"...
        y_pred_rt       = self.model['model']['svm'].predict(Xrt_scaled)[0] # Make the prediction: predict whether this match is good or bad
        prob            = self.model['model']['svm'].predict_proba(Xrt_scaled)[:,1] # get probability of prediction
        return (y_pred_rt, y_zvalues_rt, [factor1_qry, factor2_qry], prob)
    
    def generate_svm_mat(self, lims=None, array_dim=500):
        # Generate decision function matrix:
        if lims is None:
            x_lim       = [0, self.model['model']['factors'][0].max()]
            y_lim       = [0, self.model['model']['factors'][1].max()]
        else:
            x_lim       = lims['x']
            y_lim       = lims['y']
        f1          = np.linspace(x_lim[0], x_lim[1], array_dim)
        f2          = np.linspace(y_lim[0], y_lim[1], array_dim)
        F1, F2      = np.meshgrid(f1, f2)
        Fscaled     = self.model['model']['scaler'].transform(np.vstack([F1.ravel(), F2.ravel()]).T)
        y_zvalues_t = self.model['model']['svm'].decision_function(Fscaled).reshape([array_dim, array_dim])

        fig, ax = plt.subplots()
        ax.imshow(y_zvalues_t, origin='lower',extent=[0, f1[-1], 0, f2[-1]], aspect='auto')
        z_contour = ax.contour(F1, F2, y_zvalues_t, levels=[0], colors=['red','blue','green'])
        p_contour = ax.contour(F1, F2, y_zvalues_t, levels=[0.75])
        ax.clabel(p_contour, inline=True, fontsize=8)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_box_aspect(1)
        ax.axis('off')
        fig.canvas.draw()

        # extract matplotlib canvas as an rgb image:
        img_np_raw_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_np_raw = img_np_raw_flat.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close('all') # close matplotlib
        #plt.show()

        return (img_np_raw, (x_lim, y_lim))
    
    #### Private methods:
    def _check(self):
        models = self._get_models()
        for name in models:
            if models[name]['params'] == self.model['params']:
                return name
        return ""

    def _load_cal_data(self):
        # Process calibration data (only needs to be done once)
        self.print("Loading calibration query dataset...", LogType.DEBUG)
        load_cal_qry = self.cal_qry_ip.load_dataset(self.cal_qry_params)
        self.print("Loading calibration reference dataset...", LogType.DEBUG)
        load_cal_ref = self.cal_ref_ip.load_dataset(self.cal_ref_params)
        return (load_cal_qry, load_cal_ref)

    def _create_similarity_matrix(self, ft_ref_arr, ft_qry_arr):
        # query features must be a 2D array of feature vectors: np.array([[X,X...],[X,X...]])
        # reference features must be a 2D array of feature vectors
        Srefref     = fastdist.matrix_to_matrix_distance(ft_ref_arr, ft_ref_arr, fastdist.euclidean, "euclidean")
        Srefqry     = fastdist.matrix_to_matrix_distance(ft_ref_arr, ft_qry_arr, fastdist.euclidean, "euclidean")
        self.rmean  = Srefref.mean()
        self.rstd   = Srefref.std()
        self.Scal   = (Srefqry - self.rmean)/self.rstd

    def _calibrate(self):
        self.print("Calibrating datasets...")
        # Goals: 
        # 1. Reshape calref to match length of calqry
        # 2. Reorder calref to match 1:1 indices with calqry
        calqry_xy = np.transpose(np.stack((self.cal_qry_ip.dataset['dataset']['px'], self.cal_qry_ip.dataset['dataset']['py'])))
        calref_xy = np.transpose(np.stack((self.cal_ref_ip.dataset['dataset']['px'], self.cal_ref_ip.dataset['dataset']['py'])))
        match_mat = np.sum((calqry_xy[:,np.newaxis] - calref_xy)**2, 2)
        match_min = np.argmin(match_mat, 1) # should have the same number of rows as calqry (but as a vector)
        calref_xy = calref_xy[match_min, :]

        self.features_calqry  = np.array(self.cal_qry_ip.dataset['dataset'][self.feat_type])
        self.features_calref  = np.array(self.cal_ref_ip.dataset['dataset'][self.feat_type])
        self.features_calref  = self.features_calref[match_min, :]
        self.actual_match_cal = np.arange(len(self.features_calqry))
        self._create_similarity_matrix(self.features_calref, self.features_calqry)

    def _train(self):
        self.print("Performing training...")
        # We define the acceptable tolerance for a 'correct' match as +/- one image frame:
        self.tolerance      = 10

        # Extract factors that describe the "sharpness" of distance vectors
        self.factor1_cal    = find_va_factor(self.Scal)
        self.factor2_cal    = find_grad_factor(self.Scal)

        # Form input vector
        self.Xcal           = np.c_[self.factor1_cal, self.factor2_cal]
        self.scaler         = StandardScaler()
        self.Xcal_scaled    = self.scaler.fit_transform(self.Xcal)
        
        # Form desired output vector
        self.y_cal          = find_y(self.Scal, self.actual_match_cal, self.tolerance)

        # Define and train the Support Vector Machine
        self.svm_model      = svm.SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', probability=True)
        self.svm_model.fit(self.Xcal_scaled, self.y_cal)

        # Make predictions on calibration set to assess performance
        self.y_pred_cal     = self.svm_model.predict(self.Xcal_scaled)
        self.y_zvalues_cal  = self.svm_model.decision_function(self.Xcal_scaled)

        [precision, recall, num_tp, num_fp, num_tn, num_fn] = \
            find_prediction_performance_metrics(self.y_pred_cal, self.y_cal, verbose=False)
        self.print('Performance of prediction on calibration set:\nTP={0}, TN={1}, FP={2}, FN={3}\nprecision={4:3.1f}% recall={5:3.1f}%\n' \
                   .format(num_tp,num_tn,num_fp,num_fn,precision*100,recall*100))

    def _get_models(self):
        models = {}
        entry_list = os.scandir(rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/' + self.svm_dbp + "/params/")
        for entry in entry_list:
            if entry.is_file() and entry.name.startswith('svmmodel'):
                raw_npz = dict(np.load(entry.path, allow_pickle=True))
                models[os.path.splitext(entry.name)[0]] = dict(params=raw_npz['params'].item())
        return models

    def _make(self):
        params_dict         = dict(ref=self.cal_ref_params, qry=self.cal_qry_params, \
                                    npz_dbp=self.npz_dbp, bag_dbp=self.bag_dbp, svm_dbp=self.svm_dbp)
        model_dict          = dict(svm=self.svm_model, scaler=self.scaler, rstd=self.rstd, rmean=self.rmean, factors=[self.factor1_cal, self.factor2_cal])
        del self.model
        self.model          = dict(params=params_dict, model=model_dict)
        self.model_ready    = True
    
    def _fix(self, model_name):
        if not model_name.endswith('.npz'):
            model_name = model_name + '.npz'
        self.print("Bad dataset state detected, performing cleanup...", LogType.DEBUG)
        try:
            os.remove(rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/' + self.svm_dbp + '/' + model_name)
            self.print("Purged: %s" % (self.svm_dbp + '/' + model_name), LogType.DEBUG)
        except:
            pass
        try:
            os.remove(rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) +  '/' + self.svm_dbp + '/params/' + model_name)
            self.print("Purged: %s" % (self.svm_dbp + '/params/' + model_name), LogType.DEBUG)
        except:
            pass

    def _load(self, model_name):
    # when loading objects inside dicts from .npz files, must extract with .item() each object
        if not model_name.endswith('.npz'):
            model_name = model_name + '.npz'
        raw_model = np.load(rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/' + self.svm_dbp + '/' + model_name, allow_pickle=True)
        self.model_ready = False
        del self.model
        self.model = dict(model=raw_model['model'].item(), params=raw_model['params'].item())
        self.model_ready = True

