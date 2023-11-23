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
from fastdist import fastdist
#from sklearnex import patch_sklearn # Package for speeding up sklearn (must be run on GPU; TODO)
#patch_sklearn()
from sklearn import svm
from sklearn.preprocessing import StandardScaler

from ..core.file_system_tools   import scan_directory
from ..core.ros_tools           import roslogger, LogType
from ..core.enum_tools          import enum_get
from ..core.helper_tools        import formatException
from .vpr_dataset_tool          import VPRDatasetProcessor
from .vpr_helpers               import SVM_Tolerance_Mode
from ..vpred.vpred_tools        import find_prediction_performance_metrics
from ..vpred.vpred_factors      import find_factors

class SVMModelProcessor:
    def __init__(self, ros=False, root=None, load_field=False, printer=None):

        self.model_ready    = False
        self.do_field       = load_field
        self.ros            = ros
        self.printer        = printer

        if root is None:
            self.root       = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__)))
        else:
            self.root       = root

        # for making new models (prep but don't load anything yet)
        self.cal_qry_ip     = VPRDatasetProcessor(None, ros=self.ros, root=root, printer=self.printer)
        self.cal_ref_ip     = VPRDatasetProcessor(None, ros=self.ros, root=root, printer=self.printer)

        self.print("[SVMModelProcessor] Processor Ready.")

    def pass_nns(self, processor, netvlad=True, hybridnet=True):
        self.cal_qry_ip.pass_nns(processor, netvlad, hybridnet)
        self.cal_ref_ip.pass_nns(processor, netvlad, hybridnet)

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
        self.bag_dbp            = bag_dbp
        self.npz_dbp            = npz_dbp
        self.svm_dbp            = svm_dbp
        self.qry_params         = qry
        self.ref_params         = ref 
        self.svm_params         = svm
        self.img_dims           = ref['img_dims']
        self.feat_type          = ref['ft_types'][0]

        # load:
        return self._load_training_data(try_gen=try_gen, save_datasets=save_datasets) # [qry, ref]

    def generate_model(self, ref, qry, svm, bag_dbp, npz_dbp, svm_dbp, save=True, try_gen=False, save_datasets=False):
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
        load_statuses = self._load_training_data(try_gen=try_gen, save_datasets=save_datasets) # [qry, ref]
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
        dir = self.root + '/' + self.svm_dbp
        Path(dir).mkdir(parents=False, exist_ok=True)
        Path(dir+"/params").mkdir(parents=False, exist_ok=True)
        Path(dir+"/fields").mkdir(parents=False, exist_ok=True)
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
        full_field_path = dir + "/fields/" + file_name
        np.savez(full_file_path, **self.model)
        np.savez(full_param_path, params=self.model['params']) # save whole dictionary to preserve key object types
        self._save_field(full_field_path)

        self.print("[save_model] Model %s saved." % file_name)
        self.print("[save_model] Saved model, params, field to %s, %s, %s" % (full_file_path, full_param_path, full_field_path), LogType.DEBUG)
        self.print("[save_model] Parameters: \n%s" % str(self.model['params']), LogType.DEBUG)
        return self
    
    def _save_field(self, path: str):
        if path.endswith('.npz'):
            field_name = path[0:-3] + 'png'
        elif not path.endswith('.png'):
            field_name = path + '.png'
        else:
            field_name = path
        assert not self.field is None

        # Generate exif metadata:
        exif_ifd = {piexif.ExifIFD.UserComment: json.dumps({'x_lim': self.field['x_lim'], 'y_lim': self.field['y_lim']}).encode()}
        exif_dat = piexif.dump({"Exif": exif_ifd})

        # Save image:
        im = Image.fromarray(self.field['image']) 
        im.save(field_name, exif=exif_dat)

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
                    self._load(name)
                    return name
                except:
                    self.print("Load failed, performing cleanup. Code: \n%s" % formatException())
                    self._fix(name)
        if try_gen:
            self.print('[load_model] Generating model with params: %s' % (str(model_params)), LogType.DEBUG)
            self.generate_model(**model_params, try_gen=gen_datasets, save_datasets=save_datasets)
            return 'NEW GENERATION'
        return ''
    
    def swap(self, model_params, generate=False, allow_false=True):
        models = self._get_models()
        for name in models:
            if models[name]['params'] == model_params:
                try:
                    self._load(name)
                    return True
                except:
                    self.print("Load failed, performing cleanup. Code: \n%s" % formatException())
                    self._fix(name)
        if generate:
            self.generate_model(**model_params)
            return True
        if not allow_false:
            raise Exception('Model failed to load.')
        return False
    
    def predict(self, dvc, mInd, rXY, init_pos=np.array([0,0])):
        if not self.model_ready:
            raise Exception("Model not loaded in system. Either call 'generate_model' or 'load_model' before using this method.")
        factor1_qry, factor2_qry = find_factors(self.model['params']['svm']['factors'], dvc, rXY, mInd, init_pos=init_pos)

        X          = np.c_[factor1_qry, factor2_qry]                           # put the two factors into a 2-column vector
        X_scaled   = self.model['model']['scaler'].transform(X)                # perform scaling using same parameters as calibration set
        zvalues    = self.model['model']['svm'].decision_function(X_scaled)[0] # 'z' value; not probability but "related"...
        pred       = self.model['model']['svm'].predict(X_scaled)[0]           # Make the prediction: predict whether this match is good or bad
        prob       = self.model['model']['svm'].predict_proba(X_scaled)[:,1]   # get probability of prediction
        return (pred, zvalues, [factor1_qry[0], factor2_qry[0]], prob)
    
    def generate_svm_mat(self, array_dim=500):
        # Generate decision function matrix:
        x_lim       = [0, self.model['model']['factors'][0].max()]
        y_lim       = [0, self.model['model']['factors'][1].max()]
            
        f1          = np.linspace(x_lim[0], x_lim[1], array_dim)
        f2          = np.linspace(y_lim[0], y_lim[1], array_dim)
        F1, F2      = np.meshgrid(f1, f2)
        Fscaled     = self.model['model']['scaler'].transform(np.vstack([F1.ravel(), F2.ravel()]).T)
        zvalues_t   = self.model['model']['svm'].decision_function(Fscaled).reshape([array_dim, array_dim])

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
        img_np_crop         = img_np[min(indices_rows) : max(indices_rows)+1, min(indices_cols) : max(indices_cols)+1]
        final_image         = np.array(cv2.resize(img_np_crop, (array_dim, array_dim), interpolation = cv2.INTER_AREA), dtype=np.uint8)
        
        return {'image': final_image, 'x_lim': x_lim, 'y_lim': y_lim}
    
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
        load_qry = self.cal_qry_ip.load_dataset(self.qry_params, try_gen=try_gen)
        self.print("Loading calibration reference dataset...", LogType.DEBUG)
        load_ref = self.cal_ref_ip.load_dataset(self.ref_params, try_gen=try_gen)
        return (load_qry, load_ref)
    
    def _classify(self, ref_xy, qry_xy, S_train):
        # Use ground truth to label classes of SVM training data

        # Generate similarity matrix for reference to query **Positions**:
        euc_dists_train = fastdist.matrix_to_matrix_distance(
            ref_xy,
            qry_xy,
            fastdist.euclidean, "euclidean")
        
        true_inds_train     = np.argmin(euc_dists_train, axis=0) # From position we find true matches
        match_inds_train    = np.argmin(S_train,    axis=0)      # From features we find VPR matches
        
        tol_mode = enum_get(self.svm_params['tol_mode'], SVM_Tolerance_Mode)
        if tol_mode == SVM_Tolerance_Mode.DISTANCE:
            error_dist_train    = np.sqrt( \
                                        np.square(ref_xy[match_inds_train,0] - ref_xy[true_inds_train,0]) + \
                                        np.square(ref_xy[match_inds_train,1] - ref_xy[true_inds_train,1]) \
                                        )
            y_train             = error_dist_train <= self.svm_params['tol_thres']

        elif tol_mode == SVM_Tolerance_Mode.FRAME:
            error_inds_train    = np.min(np.array([-1 * abs(match_inds_train - true_inds_train) + len(match_inds_train), 
                                                abs(match_inds_train - true_inds_train)]), axis=0)
        
            y_train             = error_inds_train <= self.svm_params['tol_thres']
        else:
            raise Exception("Unknown tolerance mode (%s, %s)" % (str(tol_mode), str(self.svm_params['tol_mode'])))

        return y_train

    def _train(self):
        self.print("Performing training...")
        assert (not self.cal_ref_ip.dataset is None) and (not self.cal_qry_ip.dataset is None)

        # Generate similarity matrix for reference to query **Features**::
        self.S_train = fastdist.matrix_to_matrix_distance(  
            self.cal_ref_ip.dataset['dataset'][self.feat_type], \
            self.cal_qry_ip.dataset['dataset'][self.feat_type], \
            fastdist.euclidean, "euclidean")
        
        # Generate reference and query numpy array; columns are x,y
        ref_xy       = np.stack([self.cal_ref_ip.dataset['dataset']['px'], self.cal_ref_ip.dataset['dataset']['py']], 1)
        qry_xy       = np.stack([self.cal_qry_ip.dataset['dataset']['px'], self.cal_qry_ip.dataset['dataset']['py']], 1)

        # Extract factors:
        self.factor1_train, self.factor2_train = find_factors(self.svm_params['factors'], self.S_train, ref_xy, np.argmin(self.S_train, axis=0))

        # Form input vector
        self.X_train        = np.c_[self.factor1_train, self.factor2_train]
        self.scaler         = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)

        # Create class labels for training data:
        self.y_train        = self._classify(ref_xy, qry_xy, self.S_train)

        # Define and train the Support Vector Machine
        self.svm_model      = svm.SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', probability=True)
        classes = {unique: count for unique, count in np.unique(self.y_train, return_counts=True)}
        if len(classes.keys()) < 2:
            self.print('Bad class state! Could not define two classes based on parameters provided. Classes: %s' % str(classes), LogType.ERROR)
        
        self.svm_model.fit(self.X_train_scaled, self.y_train)

        # Make predictions on calibration set to assess performance
        self.pred_train     = self.svm_model.predict(self.X_train_scaled)
        self.zvalues_train  = self.svm_model.decision_function(self.X_train_scaled)
        self.prob_train     = self.svm_model.predict_proba(self.X_train_scaled)[:,1]

        [precision, recall, num_tp, num_fp, num_tn, num_fn] = \
            find_prediction_performance_metrics(self.pred_train, self.y_train, verbose=False)
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
        model_dict          = dict(svm=self.svm_model, scaler=self.scaler, factors=[self.factor1_train, self.factor2_train])

        try:
            del self.model
            del self.field
        except:
            pass
        self.model          = dict(params=params_dict, model=model_dict, perf=self.performance_metrics)
        self.field          = self.generate_svm_mat()
        self.model_ready    = True
    
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
        if self.do_field:
            self._load_field(self.root +  '/' + self.svm_dbp + '/fields/' + model_name)
        self.model_ready = True

class SVMFieldLoader(SVMModelProcessor):
    def __init__(self, model_params, ros=False):
        self.field          = None
        self.field_ready    = False
        self.ros            = ros
        self.load_field(model_params)

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
                    self._load_field(name)
                    return True
                except:
                    self.print("Load failed, performing cleanup. Code: \n%s" % formatException())
                    self._fix(name)
        return False

