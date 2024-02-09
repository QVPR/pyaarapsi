#!/usr/bin/env python3

import numpy as np
import os
from pathlib import Path
from PIL import Image
import piexif
import json
import datetime
import cv2
import rospkg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#from sklearnex import patch_sklearn # Package for speeding up sklearn (must be run on GPU; TODO)
#patch_sklearn()
from sklearn import svm
from sklearn.preprocessing import StandardScaler

from ..core.file_system_tools   import scan_directory
from ..core.ros_tools           import roslogger, LogType
from ..core.enum_tools          import enum_get
from ..core.helper_tools        import formatException, m2m_dist
from .vpr_dataset_tool          import VPRDatasetProcessor
from .vpr_helpers               import SVM_Tolerance_Mode, FeatureType
from ..vpred.vpred_tools        import find_prediction_performance_metrics
from ..vpred.vpred_factors      import find_factors
from ..vpred.robotvpr           import RobotVPR, RobotRun

from typing import Optional

class SVMModelProcessor:
    def __init__(self, ros=False, root=None, load_field=False, printer=None, use_tqdm: bool = False, cuda: bool = False, 
                 qry_ip: Optional[VPRDatasetProcessor] = None, ref_ip: Optional[VPRDatasetProcessor] = None):

        self.model_ready    = False
        self.do_field       = load_field
        self.ros            = ros
        self.printer        = printer

        if root is None:
            self.root       = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__)))
        else:
            self.root       = root

        # for making new models (prep but don't load anything yet)
        if qry_ip is None:
            self.cal_qry_ip = VPRDatasetProcessor(None, ros=self.ros, root=root, printer=self.printer, use_tqdm=use_tqdm, cuda=cuda)
            self.qry_loaded = bool(self.cal_qry_ip.dataset)
        else:
            self.cal_qry_ip = qry_ip
            self.qry_loaded = False

        if ref_ip is None:
            self.cal_ref_ip = VPRDatasetProcessor(None, ros=self.ros, root=root, printer=self.printer, use_tqdm=use_tqdm, cuda=cuda)
            self.ref_loaded = bool(self.cal_ref_ip.dataset)
        else:
            self.cal_ref_ip = ref_ip
            self.ref_loaded = False

        self._vars_ready    = False

        self.print("[SVMModelProcessor] Processor Ready.")

    def pass_nns(self, processor, *args, **kwargs):
        self.cal_qry_ip.pass_nns(processor=processor, *args, **kwargs)
        self.cal_ref_ip.pass_nns(processor=processor, *args, **kwargs)

    def print(self, text: str, logtype: LogType = LogType.INFO, throttle: float = 0) -> None:
        text = '[SVMModelProcessor] ' + text
        if self.printer is None:
            roslogger(text, logtype, throttle=throttle, ros=self.ros)
        else:
            self.printer(text, logtype, throttle=throttle, ros=self.ros)

    def load_training_data(self, ref, qry, svm, bag_dbp, npz_dbp, svm_dbp, try_gen=False, save_datasets=False):
        assert ref['img_dims'] == qry['img_dims'], "Reference and query metadata must be the same."
        assert ref['ft_types'] == qry['ft_types'], "Reference and query metadata must be the same."

        # store for access in saving operation:
        self.bag_dbp                = bag_dbp
        self.npz_dbp                = npz_dbp
        self.svm_dbp                = svm_dbp
        self.qry_params             = qry
        self.ref_params             = ref 
        self.svm_params             = svm
        self.svm_params['factors']  = list(np.sort(self.svm_params['factors']))
        self.img_dims               = ref['img_dims']
        self.feat_type              = ref['ft_types'][0]

        return self._load_training_data(try_gen=try_gen, save_datasets=save_datasets) # [qry, ref]

    def generate_model(self, ref, qry, svm, bag_dbp, npz_dbp, svm_dbp, save=True, try_gen=False, save_datasets=False):
        self.model_ready            = False
        load_statuses = self.load_training_data(ref=ref, qry=qry, svm=svm, bag_dbp=bag_dbp, npz_dbp=npz_dbp, svm_dbp=svm_dbp, 
                                                try_gen=try_gen, save_datasets=save_datasets)
        if all(load_statuses):
            self._train()
            self._make()
            self.model_ready    = True
            if save:
                self.save_model(name=None)
        return load_statuses

    def save_model(self, name=None):
        if not self.model_ready:
            raise Exception("Model not loaded in system. Either call 'generate_model' or 'load_model' before using this method.")
        dir = self.root + '/' + self.svm_dbp
        Path(dir).mkdir(parents=False, exist_ok=True)
        Path(dir+"/params").mkdir(parents=False, exist_ok=True)
        Path(dir+"/fields").mkdir(parents=False, exist_ok=True)
        file_ = self._check()
        if file_:
            self.print("[save_model] File exists with identical parameters (%s); skipping save." % file_)
            return self
        
        # Ensure file name is of correct format, generate if not provided
        file_list, _, _, = scan_directory(path=dir, short_files=True)
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
        full_field_path = dir + "/fields/" + file_name
        np.savez(full_file_path, **self.model)
        np.savez(full_param_path, params=self.model['params']) # save whole dictionary to preserve key object types
        self._save_field(path=full_field_path)

        self.print("[save_model] Model %s saved." % file_name)
        self.print("[save_model] Saved model, params, field to %s, %s, %s" % (full_file_path, full_param_path, full_field_path), LogType.DEBUG)
        self.print("[save_model] Parameters: \n%s" % str(self.model['params']), LogType.DEBUG)
        return self
    
    def load_model(self, model_params: dict, try_gen: bool = False, gen_datasets: bool = False, save_datasets: bool = False) -> str:
    # load via search for param match
        self.svm_dbp = model_params['svm_dbp']
        self.print("[load_model] Loading model.")
        self.model_ready = False
        models = self._get_models()
        self.model = {}
        self.field = None
        for name in models:
            if models[name]['params'] == model_params:
                try:
                    self._load(model_name=name)
                    return name
                except:
                    self.print("Load failed, performing cleanup. Code: \n%s" % formatException())
                    self._fix(model_name=name)
        if try_gen:
            self.print('[load_model] Generating model with params: %s' % (str(model_params)), LogType.DEBUG)
            self.generate_model(**model_params, try_gen=gen_datasets, save_datasets=save_datasets, save=True)
            return 'NEW GENERATION'
        return ''
    
    def swap(self, model_params, generate=False, allow_false=True):
        models = self._get_models()
        for name in models:
            if models[name]['params'] == model_params:
                try:
                    self._load(model_name=name)
                    return True
                except:
                    self.print("Load failed, performing cleanup. Code: \n%s" % formatException())
                    self._fix(model_name=name)
        if generate:
            self.generate_model(**model_params, try_gen=True, save_datasets=True, save=True)
            return True
        if not allow_false:
            raise Exception('Model failed to load.')
        return False
    
    def predict(self, dvc, mInd, rXY, init_pos=np.array([0,0])):
        if not self.model_ready:
            raise Exception("Model not loaded in system. Either call 'generate_model' or 'load_model' before using this method.")
        
        # Extract factors:
        _factors_out = find_factors(factors_in=self.svm_params['factors'], _S=dvc, 
                                          rXY=rXY, mInd=mInd, init_pos=init_pos, return_as_dict=True)

        X = np.c_[[_factors_out[i] for i in self.svm_params['factors']]]
        if X.shape[1] == 1:
            X      = np.transpose(X)                                 # put the factors into a column vector
        X_scaled   = self.scaler.transform(X=X)                 # perform scaling using same parameters as calibration set
        zvalues    = self.svm_model.decision_function(X=X_scaled)[0]  # 'z' value; not probability but "related"...
        pred       = self.svm_model.predict(X=X_scaled)[0]            # Make the prediction: predict whether this match is good or bad
        prob       = self.svm_model.predict_proba(X=X_scaled)[:,1]      # get probability of prediction
        return (pred, zvalues, X, prob)
    
    def predict_quality(self, *args, **kwargs):
        '''
        In the style of HC's RobotMonitor
        '''
        return self.pred_train
    
    def predict_from_robotvpr(self, _rvpr: RobotVPR, _ft_type: FeatureType, do_print: bool = False):
        ref: RobotRun = _rvpr.ref
        qry: RobotRun = _rvpr.qry

        ref_dict = {'px': ref.x, 'py': ref.y, _ft_type.name: ref.features}
        qry_dict = {'px': qry.x, 'py': qry.y, _ft_type.name: qry.features}

        return self.predict_from_datasets(ref=ref_dict, qry=qry_dict, do_print=do_print)
    
    def predict_from_datasets(self, ref, qry, do_print=False):
        if isinstance(ref, VPRDatasetProcessor):
            ref = ref.dataset['dataset']
        elif 'dataset' in ref.keys():
            ref = ref['dataset']

        if isinstance(qry, VPRDatasetProcessor):
            qry = qry.dataset['dataset']
        elif 'dataset' in qry.keys():
            qry = qry['dataset']

        # Generate similarity matrix for reference to query:
        S_test = m2m_dist(arr_1=ref[self.feat_type], arr_2=qry[self.feat_type])

        # Extract factors:
        _factors_out = find_factors(factors_in=self.svm_params['factors'], _S=S_test, 
                                    rXY=np.stack(arrays=[ref['px'], ref['py']], axis=1), 
                                    mInd=np.argmin(a=S_test, axis=0), return_as_dict=True)

        factors_test = np.array([_factors_out[i] for i in self.svm_params['factors']])

        # Form input vector
        X_test = np.transpose(np.array(factors_test))

        # Generate data transforming scaler:
        X_test_scaled = self.scaler.transform(X=X_test)

        # Make predictions on calibration set to assess performance
        pred_test = self.svm_model.predict(X=X_test_scaled)
        
        # Generate reference and query numpy array; columns are x,y:
        test_ref_xy = np.stack([ref['px'], ref['py']], 1)
        test_qry_xy = np.stack([qry['px'], qry['py']], 1)

        # Generate similarity matrix for reference to query **Positions**:
        euc_dists_test = self._calc_euc_dists(ref_xy=test_ref_xy, qry_xy=test_qry_xy)
        
        # Generate minima indices:
        true_inds_test, match_inds_test = self._calc_inds(_euc_dists=euc_dists_test, _s=S_test)

        error_dist_test, error_inds_test, minimum_in_tol = self._calc_errors(
            ref_xy=test_ref_xy, match_inds=match_inds_test, true_inds=true_inds_test, euc_dists=euc_dists_test)
        
        y_test, _ = self._calc_y(error_dist=error_dist_test, error_inds=error_inds_test, 
                                 minimum_in_tol=minimum_in_tol)

        [precision, recall, num_tp, num_fp, num_tn, num_fn] = \
            find_prediction_performance_metrics(predicted_in_tolerance=pred_test, 
                                                actually_in_tolerance=y_test, verbose=False)
        if do_print:
            self.print('Performance of prediction on calibration set:\nTP={0}, TN={1}, FP={2}, FN={3}\nprecision={4:3.1f}% recall={5:3.1f}%\n' \
                    .format(num_tp,num_tn,num_fp,num_fn,precision*100,recall*100))
        performance_metrics = {'precision': precision, 'recall': recall, 'num_tp': num_tp, 'num_fp': num_fp, 'num_tn': num_tn, 'num_fn': num_fn}
        
        return pred_test, performance_metrics
    
    def generate_svm_mat(self, array_dim=500):
        # Generate decision function matrix:
        x_lim       = [0, self.model['model']['factors'][0].max()]
        y_lim       = [0, self.model['model']['factors'][1].max()]
            
        f1          = np.linspace(x_lim[0], x_lim[1], array_dim)
        f2          = np.linspace(y_lim[0], y_lim[1], array_dim)
        F1, F2      = np.meshgrid(f1, f2)
        Fscaled     = self.scaler.transform(X=np.vstack([F1.ravel(), F2.ravel()]).T)
        zvalues_t   = self.svm_model.decision_function(X=Fscaled).reshape([array_dim, array_dim])

        # generate matplotlib contour of decision boundary:
        fig, ax = plt.subplots()
        ax.imshow(zvalues_t, origin='lower',extent=[0, f1[-1], 0, f2[-1]], aspect='auto')
        contour_levels = ax.contour(F1, F2, zvalues_t, levels=[0, 0.75], colors=['red','blue','green'])
        ax.clabel(contour_levels, inline=True, fontsize=8)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_box_aspect(1)
        ax.axis('off')
        fig.canvas.draw()

        # extract matplotlib canvas as an rgb image:
        img_np_raw_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) #type: ignore
        img_np_raw = img_np_raw_flat.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close('all') # close matplotlib

        # extract only plot region; ditch padded borders; resize to 1000x1000
        img_np              = np.flip(img_np_raw, axis=2) # to bgr format, for ROS
        indices_cols        = np.arange(img_np.shape[1])[np.sum(np.sum(img_np,2),0) != 255*3*img_np.shape[0]]
        indices_rows        = np.arange(img_np.shape[0])[np.sum(np.sum(img_np,2),1) != 255*3*img_np.shape[1]]
        img_np_crop         = img_np[np.min(indices_rows) : np.max(indices_rows)+1, np.min(indices_cols) : np.max(indices_cols)+1]
        final_image         = np.array(cv2.resize(img_np_crop, (array_dim, array_dim), interpolation = cv2.INTER_AREA), dtype=np.uint8)
        
        return {'image': final_image, 'x_lim': x_lim, 'y_lim': y_lim}

    def get_performance_metrics(self):
        return self.performance_metrics
        
    def generate_missing_variables(self, try_gen=False, save_datasets=False):
        '''
        Generate all training meta variables: assumes training has been performed!
        '''
        if self._vars_ready:
            return
        
        if not (self.qry_loaded and self.ref_loaded):
            self.load_training_data(ref=self.ref_params, qry=self.qry_params, svm=self.svm_params, bag_dbp=self.bag_dbp, 
                                    npz_dbp=self.npz_dbp, svm_dbp=self.svm_dbp, try_gen=try_gen, save_datasets=save_datasets)
            if not (self.qry_loaded and self.ref_loaded):
                raise Exception("Could not load datasets. Did you permit generation?")
            
        # Generate supporting variables; matrices and corresponding indices:
        self._generate_helper_variables()

        # Generate data transforming scaler:
        self.X_train_scaled = self.scaler.transform(X=self.X_train)
        
        # Classify input data:
        self._classify()

        # Make predictions on calibration set to assess performance
        self.pred_train     = self.svm_model.predict(X=self.X_train_scaled)
        self.zvalues_train  = self.svm_model.decision_function(X=self.X_train_scaled)
        self.prob_train     = self.svm_model.predict_proba(X=self.X_train_scaled)[:,1]

        self._vars_ready    = True

    #### Private methods:
    def _check(self):
        models = self._get_models()
        for name in models:
            if models[name]['params'] == self.model['params']:
                return name
        return ""

    def _load_training_data(self, try_gen=False, save_datasets=False):
        # Process calibration data (only needs to be done once)
        self.cal_qry_ip.autosave = save_datasets
        self.cal_ref_ip.autosave = save_datasets
        self.print("Loading calibration query dataset...", LogType.DEBUG)
        self.qry_loaded = self.cal_qry_ip.load_dataset(dataset_params=self.qry_params, try_gen=try_gen)
        self.print("Loading calibration reference dataset...", LogType.DEBUG)
        self.ref_loaded = self.cal_ref_ip.load_dataset(dataset_params=self.ref_params, try_gen=try_gen)
        return (self.qry_loaded, self.ref_loaded)
    
    def _generate_helper_variables(self):

        # Generate similarity matrix for reference to query **Features**:
        self.S_train = m2m_dist(arr_1=self.cal_ref_ip.dataset['dataset'][self.feat_type], arr_2=self.cal_qry_ip.dataset['dataset'][self.feat_type])
        
        # Generate reference and query numpy array; columns are x,y:
        self.cal_ref_xy       = np.stack(arrays=[self.cal_ref_ip.dataset['dataset']['px'], self.cal_ref_ip.dataset['dataset']['py']], axis=1)
        self.cal_qry_xy       = np.stack(arrays=[self.cal_qry_ip.dataset['dataset']['px'], self.cal_qry_ip.dataset['dataset']['py']], axis=1)

        # Generate similarity matrix for reference to query **Positions**:
        self.euc_dists_train = self._calc_euc_dists(ref_xy=self.cal_ref_xy, qry_xy=self.cal_qry_xy)
        
        # Generate minima indices:
        self.true_inds_train, self.match_inds_train = self._calc_inds(_euc_dists=self.euc_dists_train, _s=self.S_train)

        # Extract factors:
        _factors_out = find_factors(factors_in=self.svm_params['factors'], _S=self.S_train, 
                                          rXY=self.cal_ref_xy, mInd=self.match_inds_train, return_as_dict=True)
        self.factors_train = np.array([_factors_out[i] for i in self.svm_params['factors']])

        # Form input vector
        self.X_train        = np.transpose(np.array(self.factors_train))

    def _calc_euc_dists(self, ref_xy, qry_xy):
        return m2m_dist(arr_1=ref_xy, arr_2=qry_xy, flatten=False)

    def _calc_inds(self, _euc_dists, _s):
        true_inds     = np.argmin(a=_euc_dists,  axis=0) # From position we find true matches
        match_inds    = np.argmin(a=_s,          axis=0) # From features we find VPR matches
        return true_inds, match_inds
    
    def _calc_errors(self, ref_xy, match_inds, true_inds, euc_dists):
        error_dist    = np.sqrt( \
                                    np.square(ref_xy[match_inds,0] - ref_xy[true_inds,0]) + \
                                    np.square(ref_xy[match_inds,1] - ref_xy[true_inds,1]) \
                                    )
        error_inds   = np.min(np.array([-1 * np.abs(match_inds - true_inds) + len(match_inds), 
                                                np.abs(match_inds - true_inds)]), axis=0)
        minimum_in_tol      = euc_dists[match_inds, np.arange(stop=match_inds.shape[0])]
        return error_dist, error_inds, minimum_in_tol

    def _calc_y(self, error_dist, error_inds, minimum_in_tol):
        tol_mode     = enum_get(self.svm_params['tol_mode'], SVM_Tolerance_Mode)
        if tol_mode == SVM_Tolerance_Mode.DISTANCE:
            _y        = (error_dist <= self.svm_params['tol_thres'])
        elif tol_mode == SVM_Tolerance_Mode.TRACK_DISTANCE:
            _y        = ((error_dist <= self.svm_params['tol_thres']).astype(int) + \
                                   (minimum_in_tol <= self.svm_params['tol_thres']).astype(int)) == 2

        elif tol_mode == SVM_Tolerance_Mode.FRAME:
            self.print('[_train] Frame mode has been selected; no accounting for off-path training.', LogType.WARN)
            _y        = error_inds <= self.svm_params['tol_thres']
        else:
            raise Exception("Unknown tolerance mode (%s, %s)" % (str(tol_mode), str(self.svm_params['tol_mode'])))
        return _y, tol_mode
    
    def _classify(self):
        # Classify input data:
        error_dist_train, error_inds_train, minimum_in_tol = self._calc_errors(
            ref_xy=self.cal_ref_xy, match_inds=self.match_inds_train, 
            true_inds=self.true_inds_train, euc_dists=self.euc_dists_train)
        
        self.y_train, tol_mode = self._calc_y(error_dist=error_dist_train, error_inds=error_inds_train, 
                                              minimum_in_tol=minimum_in_tol)
        
        # Check input data is appropriately classed:
        _classes, _class_counts = np.unique(self.y_train, return_counts=True)
        classes = {_class: _count for _class, _count in zip(tuple(_classes), tuple(_class_counts))}
        if len(classes.keys()) < 2:
            self.print('Bad class state! Could not define two classes based on parameters provided. Classes: %s' % str(classes), LogType.ERROR)
            if tol_mode == SVM_Tolerance_Mode.DISTANCE:
                self.print('Minimum Distance: %.2fm, Minimum Error: %.2fm' % (np.min(minimum_in_tol), np.min(error_dist_train)), LogType.DEBUG)
            elif tol_mode == SVM_Tolerance_Mode.FRAME:
                self.print('Minimum Error: %.2fi' % (np.min(error_inds_train)), LogType.DEBUG)

    def _train(self):
        self.print("Performing training...")
        assert (not self.cal_ref_ip.dataset is None) and (not self.cal_qry_ip.dataset is None)

        # Generate supporting variables; matrices and corresponding indices:
        self._generate_helper_variables()
        
        # Classify input data:
        self._classify()

        # Generate transforming scaler:
        self.scaler         = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X=self.X_train, y=self.y_train)
        
        # Define and train the Support Vector Machine
        self.svm_model      = svm.SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', probability=True)
        if 'weights' in self.svm_params.keys():
            self.svm_model.fit(X=self.X_train_scaled, y=self.y_train, sample_weight=self.svm_params['weights'])
        else:
            self.svm_model.fit(X=self.X_train_scaled, y=self.y_train)

        # Make predictions on calibration set to assess performance
        self.pred_train     = self.svm_model.predict(X=self.X_train_scaled)
        self.zvalues_train  = self.svm_model.decision_function(X=self.X_train_scaled)
        self.prob_train     = self.svm_model.predict_proba(X=self.X_train_scaled)[:,1]

        self._vars_ready    = True

        [precision, recall, num_tp, num_fp, num_tn, num_fn] = \
            find_prediction_performance_metrics(predicted_in_tolerance=self.pred_train, 
                                                actually_in_tolerance=self.y_train, verbose=False)
        self.print('Performance of prediction on calibration set:\nTP={0}, TN={1}, FP={2}, FN={3}\nprecision={4:3.1f}% recall={5:3.1f}%\n' \
                   .format(num_tp,num_tn,num_fp,num_fn,precision*100,recall*100))
        
        self.performance_metrics = {'precision': precision, 'recall': recall, 'num_tp': num_tp, 'num_fp': num_fp, 'num_tn': num_tn, 'num_fn': num_fn}
        
    def _get_models(self):
        models      = {}
        entry_list  = os.scandir(self.root + '/' + self.svm_dbp + "/params/")
        for entry in entry_list:
            if entry.is_file() and entry.name.startswith('svmmodel'):
                raw_npz = dict(np.load(entry.path, allow_pickle=True))
                models[os.path.splitext(entry.name)[0]] = dict(params=raw_npz['params'].item())
        return models

    def _make(self):
        params_dict         = dict(ref=self.ref_params, qry=self.qry_params, svm=self.svm_params,\
                                    npz_dbp=self.npz_dbp, bag_dbp=self.bag_dbp, svm_dbp=self.svm_dbp)
        model_dict          = dict(svm=self.svm_model, scaler=self.scaler, factors=list(self.factors_train))

        try:
            del self.model
            del self.field
        except:
            pass
        self.model          = dict(params=params_dict, model=model_dict, perf=self.performance_metrics)
        if len(self.factors_train) == 2:
            self.field      = self.generate_svm_mat(array_dim=500)
        else:
            self.field      = None
        self.model_ready    = True

    def _unfurl_model(self):
        self.ref_params             = self.model['params']['ref']
        self.qry_params             = self.model['params']['qry']
        self.svm_params             = self.model['params']['svm']
        self.npz_dbp                = self.model['params']['npz_dbp']
        self.bag_dbp                = self.model['params']['bag_dbp']
        self.svm_dbp                = self.model['params']['svm_dbp']
        self.svm_model              = self.model['model']['svm']
        self.scaler                 = self.model['model']['scaler']
        self.factors_train          = np.array(self.model['model']['factors'])
        self.performance_metrics    = self.model['perf']
    
    def _fix(self, model_name):
        if not model_name.endswith('.npz'):
            model_name = model_name + '.npz'
        field_name = model_name[:-3] + 'png'
        self.print("Bad dataset state detected, performing cleanup...", LogType.DEBUG)
        try:
            os.remove(self.root + '/' + self.svm_dbp + '/' + model_name)
            self.print("Purged: %s" % (self.svm_dbp + '/' + model_name), LogType.DEBUG)
        except:
            pass
        try:
            os.remove(self.root +  '/' + self.svm_dbp + '/params/' + model_name)
            self.print("Purged: %s" % (self.svm_dbp + '/params/' + model_name), LogType.DEBUG)
        except:
            pass
        try:
            os.remove(self.root +  '/' + self.svm_dbp + '/fields/' + field_name)
            self.print("Purged: %s" % (self.svm_dbp + '/fields/' + field_name), LogType.DEBUG)
        except:
            pass

    def _load(self, model_name):
    # when loading objects inside dicts from .npz files, must extract with .item() each object
        if not model_name.endswith('.npz'):
            model_name = model_name + '.npz'
        raw_model = np.load(self.root + '/' + self.svm_dbp + '/' + model_name, allow_pickle=True)
        self.model_ready = False
        del self.model
        self.model       = dict(model=raw_model['model'].item(), params=raw_model['params'].item(), perf=raw_model['perf'].item())
        self._unfurl_model()
        if self.do_field:
            self._load_field(path=self.root +  '/' + self.svm_dbp + '/fields/' + model_name)
        self.model_ready = True
        self._vars_ready = False
    
    def _save_field(self, path: str) -> bool:
        if self.field is None:
            self.print('[_save_field] Field is none, cannot proceed!', LogType.DEBUG)
            return False
        
        if path.endswith('.npz'):
            field_name = path[0:-3] + 'png'
        elif not path.endswith('.png'):
            field_name = path + '.png'
        else:
            field_name = path

        # Generate exif metadata:
        exif_ifd = {piexif.ExifIFD.UserComment: json.dumps({'x_lim': self.field['x_lim'], 'y_lim': self.field['y_lim']}).encode()}
        exif_dat = piexif.dump({"Exif": exif_ifd})

        # Save image:
        im = Image.fromarray(self.field['image']) 
        im.save(field_name, exif=exif_dat)
        return True

    def _load_field(self, path: str):
        del self.field
        self.field_ready = False
        if path.endswith('.npz'):
            field_name = path[0:-3] + 'png'
        elif not path.endswith('.png'):
            field_name = path + '.png'
        else:
            field_name = path

        image = Image.open(field_name)
        _field = {'image': np.array(image)}
        _field.update(json.loads(image.getexif()[37510].decode())) # may need to be _getexif
        self.field = _field
        self.field_ready = True

        return True
    
class SVMFieldLoader(SVMModelProcessor):
    def __init__(self, model_params, ros=False):
        self.field          = None
        self.field_ready    = False
        self.ros            = ros
        self.load_field(model_params=model_params)

    def load_field(self, model_params):
    # load via search for param match
        self.svm_dbp = model_params['svm_dbp']
        self.print("[load_field] Loading model.")
        self.field_ready    = False
        models              = self._get_models() # because we're looking at the model params
        self.field          = None
        for name in models:
            if models[name]['params'] == model_params:
                try:
                    self._load_field(path=name)
                    return True
                except:
                    self.print("Load failed, performing cleanup. Code: \n%s" % formatException())
                    self._fix(name)
        return False

