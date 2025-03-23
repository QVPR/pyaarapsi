#!/usr/bin/env python3
'''
SVM Model Tool
'''
import os
import copy
from pathlib import Path
import json
import datetime
from typing import Optional, Callable, Tuple, Dict
from typing_extensions import Self

from cv2 import resize as cv_resize, INTER_AREA as cv_INTER_AREA #pylint: disable=E0611
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import piexif

import matplotlib
import matplotlib.pyplot as plt

#from sklearnex import patch_sklearn # Package for speeding up sklearn (must be run on GPU; TODO)
#patch_sklearn()
from sklearn import svm
from sklearn.preprocessing import StandardScaler

from pyaarapsi.core.file_system_tools import scan_directory
from pyaarapsi.core.roslogger import roslogger, LogType
from pyaarapsi.core.helper_tools import format_exception
from pyaarapsi.vpr.vpr_dataset_tool import VPRDatasetProcessor
from pyaarapsi.vpr import config
from pyaarapsi.vpr.pred.vpred_factors import find_factors
from pyaarapsi.vpr.classes.data.svmparams import SVMParams
from pyaarapsi.vpr.classes.data.svmmodel import SVMModel, SVMStatsAndVariables
from pyaarapsi.vpr.classes.data.rosbagdataset import RosbagDataset

matplotlib.use('Agg')
ROSPKG_ROOT = config.prep_rospkg_root()

class SVMLoadSaveError(Exception):
    '''
    Load save failure
    '''

class SVMDataNotLoadedError(Exception):
    '''
    Action performed that requires data; data is not loaded
    '''

class SVMModelProcessor:
    '''
    Class to handle generation and load/saving of an SVM
    '''
    def __init__(self, ros: bool = False, root: Optional[str] = None, \
                 printer: Optional[Callable] = None, use_tqdm: bool = False, cuda: bool = False,
                 vprdp: Optional[VPRDatasetProcessor] = None) -> Self:
        self.model: SVMModel = SVMModel()
        self.ros: bool = ros
        self.printer: Optional[Callable] = printer
        self.use_tqdm: bool = use_tqdm
        self.cuda: bool = cuda
        self.root: str = ROSPKG_ROOT if root is None else root
        self.vprdp: VPRDatasetProcessor = vprdp if vprdp is not None else \
            VPRDatasetProcessor(None, ros=self.ros, root=self.root, printer=self.printer, \
                                use_tqdm=self.use_tqdm, cuda=self.cuda)
        self.svm_dbp: str = "/cfg/svm_models"
        self.print("[SVMModelProcessor] Processor Ready.")
    #
    def pass_containers(self, processor, *args, **kwargs) -> Self:
        '''
        Pass containers into network
        '''
        self.vprdp.pass_containers(processor=processor, *args, **kwargs)
        return self
    #
    def print(self, text: str, logtype: LogType = LogType.INFO, throttle: float = 0) -> bool:
        '''
        Print helper
        Returns:
        - bool type; True if successfully printed
        '''
        text = '[SVMModelProcessor] ' + text
        if self.printer is None:
            return roslogger(text, logtype, throttle=throttle, ros=self.ros)
        self.printer(text, logtype, throttle=throttle, ros=self.ros)
        return True
    #
    def load_training_data(self, model_params: SVMParams, try_gen: bool = False,
                           save_datasets: bool = False) -> Tuple[RosbagDataset, RosbagDataset]:
        '''
        Load training data
        Inputs:
        - model_params: SVMParams type; unique definition of an SVM
        - try_gen: bool type; whether to permit generation of training datasets
        - save_datasets: bool type; whether to permit saving generated training datasets
        Returns:
        - Tuple[RosbagDataset, RosbagDataset]: Query and Reference datasets
        '''
        assert model_params.ref_params.img_dims.for_cv() \
            == model_params.qry_params.img_dims.for_cv(), \
                "Reference and query metadata must be the same."
        assert model_params.ref_params.vpr_descriptors == model_params.qry_params.vpr_descriptors, \
            "Reference and query metadata must be the same."
        self.vprdp.autosave = save_datasets
        # Query:
        self.print("Loading training query dataset...", LogType.DEBUG)
        self.vprdp.load_dataset(dataset_params=model_params.qry_params, try_gen=try_gen)
        assert self.vprdp.dataset.is_populated(), "Dataset load failure"
        qry_dataset = copy.deepcopy(self.vprdp.dataset)
        # Reference:
        self.print("Loading training reference dataset...", LogType.DEBUG)
        self.vprdp.load_dataset(dataset_params=model_params.ref_params, try_gen=try_gen)
        assert self.vprdp.dataset.is_populated(), "Dataset load failure"
        ref_dataset = copy.deepcopy(self.vprdp.dataset)
        self.vprdp.unload()
        return qry_dataset, ref_dataset

    def generate_model(self, model_params: SVMParams, save_model: bool = True, \
                        try_gen: bool = True, save_datasets: bool = True, store: bool = True \
                        ) -> SVMModel:
        '''
        Generate an SVM model
        Inputs:
        - model_params: SVMParams type; params which uniquely define an SVM model
        - save_model: bool type; whether to save the model to disk
        - try_gen: bool type; whether to attempt to generate missing training datasets
        - save_datasets: bool type; whether to save any generated training datasets to disk
        - store: bool type; whether to save a generated model into this instance
        Returns:
        - SVMModel type; generated model.
        '''
        qry_dataset, ref_dataset = self.load_training_data(model_params=model_params,
                                                try_gen=try_gen, save_datasets=save_datasets)
        return self.generate_model_from_datasets(ref_dataset=ref_dataset, qry_dataset=qry_dataset, \
                                    model_params=model_params, save_model=save_model, store=store)
    #
    def generate_model_from_datasets(self, ref_dataset: RosbagDataset, \
                                        qry_dataset: RosbagDataset, model_params: SVMParams, \
                                        save_model: bool = True, store: bool = True) -> SVMModel:
        '''
        Generate an SVM model from pre-generated RosbagDatasets
        Inputs:
        - ref_dataset: RosbagDataset type; reference dataset
        - qry_dataset: RosbagDataset type; query dataset
        - model_params: SVMParams type; params which uniquely define an SVM model
        - save_model: bool type; whether to save the model to disk
        - store: bool type; whether to save a generated model into this instance
        Returns:
        - SVMModel type; generated model.
        '''
        scaler = StandardScaler() # data transforming scaler
        model = svm.SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', \
                            probability=True) # Define and train the Support Vector Machine
        self.print("Performing training...")
        stats = SVMStatsAndVariables().populate(scaler=scaler, model=model, \
                    qry_dataset=qry_dataset, ref_dataset=ref_dataset, model_params=model_params, \
                    printer=lambda text: self.print(text, LogType.DEBUG))
        self.print("Performance of prediction on calibration set:\n" \
                    f"TP={stats.performance_metrics['num_tp']}, " \
                    f"TN={stats.performance_metrics['num_tn']}, " \
                    f"FP={stats.performance_metrics['num_fp']}, " \
                    f"FN={stats.performance_metrics['num_fn']}\n" \
                    f"precision={stats.performance_metrics['precision']*100:3.1f}%, " \
                    f"recall={stats.performance_metrics['recall']*100:3.1f}%")
        svm_model = SVMModel().populate(params=model_params, scaler=scaler, model=model, \
                                        stats=stats)
        if save_model:
            self.save_model(model=svm_model, name=None)
        if store:
            self.model.populate_from(svm_model)
        return svm_model
    #
    def predict_from_datasets(self, ref_dataset: RosbagDataset, qry_dataset: RosbagDataset, \
                              model_params: SVMParams, save_model: bool = True
                              ) -> SVMStatsAndVariables:
        '''
        Predict from pre-generated RosbagDatasets; optionally, save the generated model to disk
        for future use (not returned by this method; use 'generate_model_from_datasets' directly).
        Inputs:
        - ref_dataset: RosbagDataset type; reference dataset
        - qry_dataset: RosbagDataset type; query dataset
        - model_params: SVMParams type; params which uniquely define an SVM model
        - save_model: bool type; whether to save the model to disk
        Returns:
        - SVMStatsAndVariables type; generated model statistics.
        '''
        return self.generate_model_from_datasets(ref_dataset=ref_dataset, qry_dataset=qry_dataset, \
                            model_params=model_params, save_model=save_model, store=False).stats
    #
    def make_saveable_file_name(self, name: Optional[str] = None) -> Tuple[str, str]:
        '''
        Make a file name that is unique
        '''
        svm_dir = self.root + '/' + self.svm_dbp
        Path(svm_dir).mkdir(parents=False, exist_ok=True)
        Path(svm_dir+"/params").mkdir(parents=False, exist_ok=True)
        Path(svm_dir+"/fields").mkdir(parents=False, exist_ok=True)
        existing_file_name = self._check()
        if existing_file_name:
            self.print("[make_saveable_file_name] File exists with identical parameters " \
                       f"({existing_file_name}); skipping save.")
            return self
        # Ensure file name is of correct format, generate if not provided
        file_list, _, _, = scan_directory(path=svm_dir, short_files=True)
        if name is not None:
            if not name.startswith('svmmodel'):
                name = "svmmodel_" + name
            if name in file_list:
                raise SVMLoadSaveError(f"Model with name {name} already exists in directory.")
        else:
            name = datetime.datetime.today().strftime("svmmodel_%Y%m%d")
        # Check file_name won't overwrite existing models
        file_name = name
        count = 0
        while file_name in file_list:
            file_name = name + f"_{count:d}"
            count += 1
        return svm_dir, file_name
    #
    def _save_model(self, model: SVMModel, svm_dir: str, file_name: str) -> Self:
        '''
        Save model helper, for injecting additional files
        '''
        full_file_path = svm_dir + "/" + file_name
        full_param_path = svm_dir + "/params/" + file_name
        # save whole dictionary to preserve key object types:
        np.savez(full_file_path, model=model.save_ready())
        self.print(f"[save_model] Saved model to {full_file_path}", LogType.DEBUG)
        np.savez(full_param_path, params=model.params.save_ready())
        self.print(f"[save_model] Saved params to {full_param_path}", LogType.DEBUG)
    #
    def save_model(self, model: Optional[SVMModel] = None, name: Optional[str] = None) -> Self:
        '''
        Save SVM model
        '''
        if model is None:
            model = self.model
        if not model.is_populated():
            raise SVMDataNotLoadedError("Model not loaded in system. Either call 'generate_model' "
                                        "or 'load_model' before using this method.")
        svm_dir, file_name = self.make_saveable_file_name(name=name)
        self._save_model(model=model, svm_dir=svm_dir, file_name=file_name)
        self.print(f"[save_model] Model {file_name} saved.")
        self.print(f"[save_model] Parameters: \n{str(model['params'])}", LogType.DEBUG)
        return self
    #
    def load_model(self, model_params: dict, try_gen: bool = False, gen_datasets: bool = False, \
                   save_datasets: bool = False) -> str:
        '''
        Load SVM model from disk
        '''
        self.print("[load_model] Loading model.")
        models = self.get_all_saved_model_params()
        for key, value in models.items():
            if value['params'] == model_params:
                try:
                    self._load(model_name=key)
                    return key
                except SVMLoadSaveError:
                    self.print(f"Load failed, performing cleanup. Code: \n{format_exception()}")
                    self._fix(model_name=key)
        if try_gen:
            self.print(f'[load_model] Generating model with params: {str(model_params)}', \
                       LogType.DEBUG)
            self.generate_model(**model_params, try_gen=gen_datasets, save_datasets=save_datasets, \
                                save=True)
            return 'NEW GENERATION'
        return ''
    #
    def predict(self, sim_matrix, match_ind, ref_xy, init_pos=np.array([0,0])):
        '''
        Make a prediction from the SVM
        '''
        if not self.model.is_populated():
            raise SVMLoadSaveError("Model not loaded in system. Either call 'generate_model' or "
                                    "'load_model' before using this method.")
        # Extract factors:
        factors_out = find_factors(factors_in=self.model.params.factors, sim_matrix=sim_matrix,
                        ref_xy=ref_xy, match_ind=match_ind, init_pos=init_pos, return_as_dict=True)

        x_column = np.c_[[factors_out[i] for i in self.model.params.factors]]
        if x_column.shape[1] == 1:
            x_column = np.transpose(x_column)
        x_scaled = self.model.scaler.transform(X=x_column)
        zvalues = self.model.model.decision_function(X=x_scaled)[0]
        pred = self.model.model.predict(X=x_scaled)[0]
        prob = self.model.model.predict_proba(X=x_scaled)[:,1]
        return (pred, zvalues, x_column, prob)
    #
    def predict_quality(self) -> NDArray:
        '''
        In the style of HC's RobotMonitor
        '''
        return self.model.stats.pred_state
    #
    def get_performance_metrics(self):
        '''
        Getter for performance metrics
        '''
        assert self.model.is_populated()
        return self.model.stats.performance_metrics
    #
    #### Private methods:
    def _check(self):
        models = self.get_all_saved_model_params()
        for key, value in models.items():
            if value == self.model.params:
                return key
        return ""
    #
    def get_all_saved_model_params(self) -> Dict[str, SVMParams]:
        '''
        Retrieve all model params that have been saved to disk
        Returns:
        - dict[str, SVMParams]; keys are paths to each SVMParams object
        '''
        models      = {}
        entry_list  = os.scandir(self.root + '/' + self.svm_dbp + "/params/")
        for entry in entry_list:
            if entry.is_file() and entry.name.startswith('svmmodel'):
                raw_npz = dict(np.load(entry.path, allow_pickle=True))
                models[os.path.splitext(entry.name)[0]] = dict(params=raw_npz['params'].item())
        return models
    #
    def _fix(self, model_name: str) -> Self:
        '''
        Purge corrupted files
        '''
        if not model_name.endswith('.npz'):
            model_name = model_name + '.npz'
        field_name = model_name[:-3] + 'png'
        self.print("Bad dataset state detected, performing cleanup...", LogType.DEBUG)
        try:
            os.remove(self.root + '/' + self.svm_dbp + '/' + model_name)
            self.print(f"Purged: {self.svm_dbp + '/' + model_name}", LogType.DEBUG)
        except OSError:
            pass
        try:
            os.remove(self.root +  '/' + self.svm_dbp + '/params/' + model_name)
            self.print(f"Purged: {self.svm_dbp + '/params/' + model_name}", LogType.DEBUG)
        except OSError:
            pass
        try:
            os.remove(self.root +  '/' + self.svm_dbp + '/fields/' + field_name)
            self.print(f"Purged: {self.svm_dbp + '/fields/' + field_name}", LogType.DEBUG)
        except OSError:
            pass
        return self
    #
    def _load(self, model_name: str) -> SVMModel:
        '''
        Helper for loading
        '''
        # when loading objects inside dicts from .npz files, must extract with .item() each object
        try:
            if not model_name.endswith('.npz'):
                model_name = model_name + '.npz'
            raw_model = np.load(self.root + '/' + self.svm_dbp + '/' + model_name, \
                                allow_pickle=True)
            del self.model
            self.model = SVMModel.from_save_ready(save_ready_dict=raw_model['model'].item())
        except Exception as e:
            raise SVMLoadSaveError("Failed to load model") from e
        return self.model

class ExtendedSVMModelProcessor(SVMModelProcessor):
    '''
    Base class to handle processing of a field object
    '''
    def __init__(self, *args, do_field: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.field: dict = {"image": None, "x_lim": None, "y_lim": None}
        self.field_ready: bool = False
        self.do_field: bool = do_field
        if self.do_field:
            self.load_field()
    #
    def generate_field(self, array_dim: int = 500) -> Self:
        '''
        Generate decision function matrix
        '''
        assert self.model.is_populated(), "Model is not populated."
        assert self.model.params.num_factors() == 2, \
            "Cannot generate a field unless there are only two factors in use."
        x_lim = [self.model.stats.factors_out[0].min(), self.model.stats.factors_out[0].max()]
        y_lim = [self.model.stats.factors_out[1].min(), self.model.stats.factors_out[1].max()]
        factor_1 = np.linspace(x_lim[0], x_lim[1], array_dim)
        factor_2 = np.linspace(y_lim[0], y_lim[1], array_dim)
        factor_1_mesh, factor_2_mesh = np.meshgrid(factor_1, factor_2)
        factors_scaled = self.model.scaler.transform(X=np.vstack([factor_1_mesh.ravel(), \
                                                            factor_2_mesh.ravel()]).T)
        z_function = self.model.model.decision_function(X=factors_scaled) \
                                        .reshape([array_dim, array_dim])
        # generate matplotlib contour of decision boundary:
        fig, ax = plt.subplots()
        ax.imshow(z_function, origin='lower', aspect='auto', \
                    extent=[0, factor_1[-1], 0, factor_2[-1]])
        contour_levels = ax.contour(factor_1_mesh, factor_2_mesh, z_function, levels=[0, 0.75], \
                                    colors=['red', 'blue', 'green'])
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
        img_np = np.flip(img_np_raw, axis=2) # to bgr format, for ROS
        indices_cols = np.arange(img_np.shape[1]) \
                            [np.sum(np.sum(img_np,2),0) != 255*3*img_np.shape[0]]
        indices_rows = np.arange(img_np.shape[0]) \
                            [np.sum(np.sum(img_np,2),1) != 255*3*img_np.shape[1]]
        img_np_crop = img_np[np.min(indices_rows) : np.max(indices_rows)+1, \
                             np.min(indices_cols) : np.max(indices_cols)+1]
        final_image = np.array(cv_resize(img_np_crop, (array_dim, array_dim), \
                                         interpolation = cv_INTER_AREA), dtype=np.uint8)
        self.field.update(image=final_image, x_lim=x_lim, y_lim=y_lim)
        return self
    #
    def load_field_helper(self, path: str) -> bool:
        '''
        Helper to handle the actual file access
        '''
        try:
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
        except Exception as e:
            raise SVMLoadSaveError("Failed to load field") from e
    #
    def load_field(self) -> bool:
        '''
        Load from disk via search for param match
        '''
        self.print("[load_field] Loading model.")
        self.field_ready = False
        models = self.get_all_saved_model_params() # because we're looking at the model params
        self.field = None
        for key, value in models.items():
            if value['params'] == self.model.params:
                try:
                    self.load_field_helper(path=key)
                    return True
                except SVMLoadSaveError:
                    self.print(f"Load failed, performing cleanup. Code: \n{format_exception()}")
                    self._fix(key)
        return False
    #
    def save_field(self, svm_dir: str, file_name: str) -> str:
        '''
        Save field to disk
        '''
        full_field_path: str = svm_dir + '/fields/' + file_name + ".png"
        if self.field is None:
            raise SVMDataNotLoadedError('[save_field] Field is none, cannot proceed!')
        try:
            # Generate exif metadata:
            exif_ifd = {piexif.ExifIFD.UserComment: \
                            json.dumps({'x_lim': self.field['x_lim'],
                                        'y_lim': self.field['y_lim']}).encode()}
            exif_dat = piexif.dump({"Exif": exif_ifd})
            # Save image:
            im = Image.fromarray(self.field['image'])
            im.save(full_field_path, exif=exif_dat)
            return full_field_path
        except Exception as e:
            raise SVMLoadSaveError(f"Failed to save field (path: {full_field_path})") from e
    #
    def _load(self, model_name: str) -> Self:
        '''
        Helper for loading
        '''
        super(ExtendedSVMModelProcessor, self)._load(model_name=model_name)
        try:
            if self.do_field:
                if not self.load_field():
                    raise SVMLoadSaveError("Load failure.")
        except Exception as e:
            raise SVMLoadSaveError("Failed to load field.") from e
        return self
    #
    def _save_model(self, model: SVMModel, svm_dir: str, file_name: str) -> Self:
        '''
        Save model helper, with field injection
        '''
        super(ExtendedSVMModelProcessor, self)._save_model(model=model, svm_dir=svm_dir, \
                                                           file_name=file_name)
        full_field_path = self.save_field(svm_dir=svm_dir, file_name=file_name)
        self.print(f"[save_model] Saved field to {full_field_path}", LogType.DEBUG)
        return self
