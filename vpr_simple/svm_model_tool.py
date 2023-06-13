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
from ..vpred                  import find_factors, find_prediction_performance_metrics

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
            
    def generate_model(self, ref, qry, svm, bag_dbp, npz_dbp, svm_dbp, save=True):
        assert ref['img_dims'] == qry['img_dims'], "Reference and query metadata must be the same."
        assert ref['ft_types'] == qry['ft_types'], "Reference and query metadata must be the same."
        # store for access in saving operation:
        self.bag_dbp            = bag_dbp
        self.npz_dbp            = npz_dbp
        self.svm_dbp            = svm_dbp
        self.qry_params         = qry
        self.ref_params         = ref 
        self.svm_params         = svm
        self.img_dims           = ref['img_dims']
        self.feat_type          = ref['ft_types'][0]
        self.model_ready        = False

        # generate:
        load_statuses = self._load_training_data() # [qry, ref]
        if all(load_statuses):
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
        file_ = self._check()
        if file_:
            self.print("[save_model] File exists with identical parameters (%s); skipping save." % file_)
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

    def load_model(self, model_params, try_gen=False):
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
        if try_gen:
            self.generate_model(**model_params)
            return True
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
    
    def predict(self, dvc, mInd):
        if not self.model_ready:
            raise Exception("Model not loaded in system. Either call 'generate_model' or 'load_model' before using this method.")
        factor1_qry, factor2_qry = find_factors(self.model['params']['svm']['factors'], dvc, self.ref_odom[:, 0:2], mInd)

        X          = np.c_[factor1_qry, factor2_qry]                           # put the two factors into a 2-column vector
        X_scaled   = self.model['model']['scaler'].transform(X)                # perform scaling using same parameters as calibration set
        zvalues    = self.model['model']['svm'].decision_function(X_scaled)[0] # 'z' value; not probability but "related"...
        pred       = self.model['model']['svm'].predict(X_scaled)[0]           # Make the prediction: predict whether this match is good or bad
        prob       = self.model['model']['svm'].predict_proba(X_scaled)[:,1]   # get probability of prediction
        return (pred, zvalues, [factor1_qry[0], factor2_qry[0]], prob)
    
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

        # generate matplotlib contour of decision boundary:
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

    def _load_training_data(self):
        # Process calibration data (only needs to be done once)
        self.print("Loading calibration query dataset...", LogType.DEBUG)
        load_qry = self.cal_qry_ip.load_dataset(self.qry_params)
        self.print("Loading calibration reference dataset...", LogType.DEBUG)
        load_ref = self.cal_ref_ip.load_dataset(self.ref_params)
        return (load_qry, load_ref)

    def _train(self):
        self.print("Performing training...")

        # Define SVM:
        self.svm_model      = svm.SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', probability=True)
        self.scaler         = StandardScaler()

        # Generate details:
        self.S_train = fastdist.matrix_to_matrix_distance(  
            self.cal_ref_ip.dataset['dataset'][self.feat_type], \
            self.cal_qry_ip.dataset['dataset'][self.feat_type], \
            fastdist.euclidean, "euclidean")
        
        self.ref_odom       = np.stack([self.cal_ref_ip.dataset['dataset']['px'], self.cal_ref_ip.dataset['dataset']['py']], 1)
        self.qry_odom       = np.stack([self.cal_qry_ip.dataset['dataset']['px'], self.cal_qry_ip.dataset['dataset']['py']], 1)

        euc_dists_train = fastdist.matrix_to_matrix_distance(
            self.ref_odom[:, 0:2],
            self.qry_odom[:, 0:2],
            fastdist.euclidean, "euclidean")
        
        true_inds_train     = np.argmin(euc_dists_train, axis=0)
        match_inds_train    = np.argmin(self.S_train, axis=0)
        error_inds_train    = np.min(np.array([-1 * abs(match_inds_train - true_inds_train) + len(match_inds_train), 
                                               abs(match_inds_train - true_inds_train)]), axis=0)
        # error_dist_train    = np.sqrt( \
        #                             np.square(self.ref_odom[match_inds_train,0] - self.ref_odom[true_inds_train,0]) + \
        #                             np.square(self.ref_odom[match_inds_train,1] - self.ref_odom[true_inds_train,1]) \
        #                             )
        
        self.y_train        = error_inds_train <= 10 # +-10 frames

        # Extract factors:
        self.factor1_train, self.factor2_train = find_factors(self.svm_params['factors'], self.S_train, self.ref_odom[:, 0:2], np.argmin(self.S_train, axis=0))

        # Form input vector
        self.X_train        = np.c_[self.factor1_train, self.factor2_train]
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)

        # Define and train the Support Vector Machine
        self.svm_model.fit(self.X_train_scaled, self.y_train)

        # Make predictions on calibration set to assess performance
        self.pred_train     = self.svm_model.predict(self.X_train_scaled)
        self.zvalues_train  = self.svm_model.decision_function(self.X_train_scaled)
        self.prob_train     = self.svm_model.predict_proba(self.X_train_scaled)[:,1]

        [precision, recall, num_tp, num_fp, num_tn, num_fn] = \
            find_prediction_performance_metrics(self.pred_train, self.y_train, verbose=False)
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
        params_dict         = dict(ref=self.ref_params, qry=self.qry_params, svm=self.svm_params,\
                                    npz_dbp=self.npz_dbp, bag_dbp=self.bag_dbp, svm_dbp=self.svm_dbp)
        model_dict          = dict(svm=self.svm_model, scaler=self.scaler, factors=[self.factor1_train, self.factor2_train])
        try:
            del self.model
        except:
            pass
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
    
    def _load_ips(self):
        self.print("Loading calibration query dataset...", LogType.DEBUG)
        self.cal_qry_ip.load_dataset(self.model['params']['qry'])
        self.print("Loading calibration reference dataset...", LogType.DEBUG)
        self.cal_ref_ip.load_dataset(self.model['params']['ref'])

        self.ref_odom    = np.stack([self.cal_ref_ip.dataset['dataset']['px'], self.cal_ref_ip.dataset['dataset']['py']], 1)
        self.qry_odom    = np.stack([self.cal_qry_ip.dataset['dataset']['px'], self.cal_qry_ip.dataset['dataset']['py']], 1)

    def _load(self, model_name):
    # when loading objects inside dicts from .npz files, must extract with .item() each object
        if not model_name.endswith('.npz'):
            model_name = model_name + '.npz'
        raw_model = np.load(rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/' + self.svm_dbp + '/' + model_name, allow_pickle=True)
        self.model_ready = False
        del self.model
        self.model       = dict(model=raw_model['model'].item(), params=raw_model['params'].item())
        self._load_ips()
        self.model_ready = True

