#!/usr/bin/env python3
'''
Class to define an SVM Model
'''
from __future__ import annotations
# import copy
from typing import Callable
from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pyaarapsi.core.helper_tools import m2m_dist
from pyaarapsi.core.argparse_tools import assert_instance
from pyaarapsi.vpr.classes.data.svmparams import SVMParams, SVMToleranceMode
from pyaarapsi.vpr.classes.data.rosbagdataset import RosbagDataset
from pyaarapsi.vpr.classes.data.abstractdata import AbstractData
from pyaarapsi.vpr.pred.vpred_factors import find_factors
from pyaarapsi.vpr.pred.vpred_tools import find_prediction_performance_metrics

class SVMStatsAndVariables(AbstractData):
    '''
    A holder for statistics and variables from training
    '''
    def __init__(self) -> Self:
        self.feature_sim_matrix: NDArray = None
        self.metric_sim_matrix: NDArray = None
        self.match_inds: NDArray = None
        self.truth_inds: NDArray = None
        self.factors_out: NDArray = None
        self.error_dist: NDArray = None
        self.error_inds: NDArray = None
        self.minimum_in_tol: NDArray = None
        self.true_state: NDArray = None
        self.factors_transformed: NDArray = None
        self.pred_state: NDArray = None
        self.z_function: NDArray = None
        self.prob_state: NDArray = None
        self.performance_metrics: dict = None
        self.populated =  False
    #
    def populate(self, scaler: StandardScaler, model: SVC, qry_dataset: RosbagDataset, \
                    ref_dataset: RosbagDataset, model_params: SVMParams, printer: Callable \
                    ) -> Self:
        '''
        Populate
        '''
        # Generate feature and metric matrices and indices:
        self.feature_sim_matrix = m2m_dist(arr_1=ref_dataset.data_of(None, None), \
                              arr_2=qry_dataset.data_of(None, None), flatten=False)
        ref_xy = ref_dataset.pxyw_of(None)[:,0:2]
        qry_xy = qry_dataset.pxyw_of(None)[:,0:2]
        self.metric_sim_matrix = m2m_dist(arr_1=ref_xy, arr_2=qry_xy, flatten=False)
        # From features we find VPR matches:
        self.match_inds = np.argmin(a=self.feature_sim_matrix, axis=0)
        # From metric data we find true matches:
        self.truth_inds = np.argmin(a=self.metric_sim_matrix, axis=0)
        # Extract factors:
        self.factors_out = np.transpose(np.array(find_factors(factors_in=model_params.factors, \
                                            sim_matrix=self.feature_sim_matrix, ref_xy=ref_xy, \
                                            match_ind=self.match_inds, return_as_dict=False)))
        self.error_dist = np.sqrt(np.square(ref_xy[self.match_inds,0] - ref_xy[self.truth_inds,0]) \
                                + np.square(ref_xy[self.match_inds,1] - ref_xy[self.truth_inds,1]))
        self.error_inds = np.min(np.array([-1 * np.abs(self.match_inds - self.truth_inds) \
                                           + len(self.match_inds), \
                                            np.abs(self.match_inds - self.truth_inds)]), axis=0)
        self.minimum_in_tol = self.metric_sim_matrix[self.match_inds, \
                                                     np.arange(stop=self.match_inds.shape[0])]
        # Generate true state:
        if model_params.tol_mode == SVMToleranceMode.DISTANCE:
            self.true_state = self.error_dist <= model_params.tol_thresh
        elif model_params.tol_mode == SVMToleranceMode.TRACK_DISTANCE:
            self.true_state = ((self.error_dist <= model_params.tol_thresh).astype(int) + \
                            (self.minimum_in_tol <= model_params.tol_thresh).astype(int)) == 2
        elif model_params.tol_mode == SVMToleranceMode.FRAME:
            printer('Frame mode has been selected; no accounting for off-path training.')
            self.true_state = self.error_inds <= model_params.tol_thresh
        else:
            raise SVMToleranceMode.Exception("Unknown tolerance mode " \
                                             f"({str(model_params.tol_mode)})")
        # Check input data is appropriately classed:
        _classes, _class_counts = np.unique(self.true_state, return_counts=True)
        classes = {_class: _count for _class, _count in zip(tuple(_classes), tuple(_class_counts))}
        if len(classes.keys()) < 2:
            if model_params.tol_mode == SVMToleranceMode.DISTANCE:
                additional_text = f'\n\tMinimum Distance: {np.min(self.minimum_in_tol):.2f}m' \
                                    f'\n\tMinimum Error: {np.min(self.error_dist):.2f}m'
            elif model_params.tol_mode == SVMToleranceMode.FRAME:
                additional_text = f'\n\tMinimum Error: {np.min(self.error_inds):.2f}i'
            else:
                additional_text = "\n\tNONE"
            raise self.TrainingFailure('Bad class state! Could not define two classes based on ' \
                                       f'parameters provided. Classes: {str(classes)}.\nDebug:' \
                                        f'{additional_text}')
        # Fit scaler and model:
        self.factors_transformed = scaler.fit_transform(X=self.factors_out, y=self.true_state)
        model.fit(X=self.factors_transformed, y=self.true_state)
        # Make predictions on calibration set to assess performance
        self.pred_state = model.predict(X=self.factors_transformed)
        self.z_function = model.decision_function(X=self.factors_transformed)
        self.prob_state = model.predict_proba(X=self.factors_transformed)[:,1]
        # Assess:
        [precision, recall, num_tp, num_fp, num_tn, num_fn] = \
            find_prediction_performance_metrics(predicted_in_tolerance=self.pred_state,
                                            actually_in_tolerance=self.true_state, verbose=False)
        self.performance_metrics = {'precision': precision, 'recall': recall, 'num_tp': num_tp, \
                                    'num_fp': num_fp, 'num_tn': num_tn, 'num_fn': num_fn}
        self.populated = True
        return self
    #
    def to_dict(self) -> dict:
        '''
        Convert to dictionary
        '''
        return {"feature_sim_matrix": self.feature_sim_matrix, \
                "metric_sim_matrix": self.metric_sim_matrix, "match_inds": self.match_inds, \
                "truth_inds": self.truth_inds, "factors_out": self.factors_out, \
                "error_dist": self.error_dist, "error_inds": self.error_inds, \
                "minimum_in_tol": self.minimum_in_tol, "true_state": self.true_state, \
                "factors_transformed": self.factors_transformed, "pred_state": self.pred_state, \
                "z_function": self.z_function, "prob_state": self.prob_state, \
                "performance_metrics": self.performance_metrics, "populated": self.populated}
    #
    def save_ready(self) -> dict:
        '''
        Convert class instance into a dictionary, where all keys are class attribute names, and
        all values are ready for pickling.
        '''
        try:
            return self.to_dict()
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to encode.") from e
    #
    @staticmethod
    def from_save_ready(save_ready_dict: dict) -> Self:
        '''
        Convert the output of save_ready() back into a class instance.
        '''
        try:
            stats = SVMStatsAndVariables()
            for key, value in save_ready_dict.items():
                setattr(stats, key, value)
            return stats
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to decode.") from e
    #
    def is_populated(self) -> bool:
        '''
        Check if populated
        '''
        return self.populated
    #
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(populated={self.populated})"
    #
    def __del__(self):
        del self.feature_sim_matrix
        del self.metric_sim_matrix
        del self.match_inds
        del self.truth_inds
        del self.factors_out
        del self.error_dist
        del self.error_inds
        del self.minimum_in_tol
        del self.true_state
        del self.factors_transformed
        del self.pred_state
        del self.z_function
        del self.prob_state
        del self.performance_metrics
        del self.populated
    #
    @classmethod
    class TrainingFailure(Exception):
        '''
        Training Failure
        '''

class SVMModel(AbstractData):
    '''
    An SVM model
    '''
    def __init__(self):
        self.params: SVMParams = SVMParams()
        self.scaler: StandardScaler = StandardScaler()
        self.model: SVC = None
        self.stats: SVMStatsAndVariables = None
        self.populated: bool = False
    #
    def populate(self, params: SVMParams, scaler: StandardScaler, model: SVC, \
                 stats: SVMStatsAndVariables) -> Self:
        '''
        Inputs:
        - params: SVMParams type; uniquely defining parameters instance
        '''
        self.params.populate_from(assert_instance(params, SVMParams))
        self.scaler = assert_instance(scaler, StandardScaler)
        self.model = assert_instance(model, SVC)
        self.stats = assert_instance(stats, SVMStatsAndVariables)
        assert self.stats.is_populated(), "stats is not populated."
        self.populated = True
        return self
    #
    def populate_from(self, model: SVMModel) -> Self:
        '''
        Populate with another model. Protects pointers to existing instances
        '''
        self.populate(params=model.params, scaler=model.scaler, model=model.model, \
                      stats=model.stats)
        return self
    #
    def is_populated(self) -> bool:
        '''
        Check if populated
        '''
        return self.populated
    #
    def to_dict(self) -> dict:
        '''
        Convert to dictionary
        '''
        base_dict = {"params": self.params, "scaler": self.scaler, "model": self.model, \
                     "stats": self.stats, "populated": self.populated}
        return base_dict
    #
    def save_ready(self) -> dict:
        '''
        Convert class instance into a dictionary, where all keys are class attribute names, and
        all values are ready for pickling.
        '''
        try:
            as_dict = self.to_dict()
            as_dict["params"] = as_dict["params"].save_ready()
            as_dict["stats"] = as_dict["stats"].save_ready()
            return as_dict
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to encode.") from e
    #
    @staticmethod
    def from_save_ready(save_ready_dict: dict) -> Self:
        '''
        Convert the output of save_ready() back into a class instance.
        '''
        try:
            return SVMModel().populate( \
                params=SVMParams.from_save_ready(save_ready_dict=save_ready_dict["params"]),
                scaler=save_ready_dict["scaler"],
                model=save_ready_dict["model"],
                stats=SVMStatsAndVariables.from_save_ready( \
                    save_ready_dict=save_ready_dict["stats"]),
            )
        except Exception as e:
            raise AbstractData.LoadSaveProtocolError("Failed to decode.") from e
    #
    def __repr__(self) -> str:
        if not self.is_populated():
            return f"{self.__class__.__name__}(populated={self.populated})"
        return f"{self.__class__.__name__}(params={self.params})"
    #
    def __del__(self):
        '''
        Clean-up
        '''
        del self.params
        del self.scaler
        del self.model
        del self.stats
        del self.populated
