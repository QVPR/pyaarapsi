# robotmonitor.py
'''
RobotMonitor classes
'''
import copy
from abc import ABC, abstractmethod
from typing import List, Union, Optional
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pyaarapsi.vpr.pred.vpred_factors import find_factors, find_va_factor, find_grad_factor
from pyaarapsi.vpr.pred.vpred_tools import find_prediction_performance_metrics
from pyaarapsi.vpr.pred.robotvpr import RobotVPR

class RobotMonitor(ABC):
    '''
    Base Class for Robot Monitor objects
    '''
    model: svm.SVC
    factors: List[str]
    factor_1: Union[NDArray, list]
    factor_2: Union[NDArray, list]
    scaler: StandardScaler
    training_y: Union[NDArray, list]

    @abstractmethod
    def form_x(self, robot_vpr: RobotVPR):
        '''
        todo
        '''

    def predict_quality(self, robot_vpr: RobotVPR):
        '''
        todo
        '''
        return self.model.predict(self.form_x(robot_vpr))

    def probability(self, robot_vpr: RobotVPR):
        '''
        todo
        '''
        return self.model.predict_proba(self.form_x(robot_vpr))[:,1]

    def decision_values(self, robot_vpr: RobotVPR):
        '''
        todo
        '''
        return self.model.decision_function(self.form_x(robot_vpr))

    def assess_prediction(self, robot_vpr: RobotVPR, verbose: bool=False):
        '''
        todo
        '''
        y_pred = self.predict_quality(robot_vpr)
        [precision, recall, num_tp, num_fp, num_tn, num_fn] = \
            find_prediction_performance_metrics(y_pred, robot_vpr.y, verbose=verbose)
        return {'precision': precision, 'recall': recall, 'num_tp': num_tp, 'num_fp': num_fp,
                'num_tn': num_tn, 'num_fn': num_fn}

    def generate_pred_prob_z(self, robot_vpr: RobotVPR):
        '''
        todo
        '''
        x_scaled = self.form_x(robot_vpr)
        y_pred = self.model.predict(x_scaled)
        y_pred_prob = self.model.predict_proba(x_scaled)[:,1]
        y_zvalues = self.model.decision_function(x_scaled)
        return y_pred, y_pred_prob, y_zvalues

    def generate_z_function(self, robot_vpr: RobotVPR=None):
        '''
        todo
        '''
        if len(self.factors) > 2:
            print('Error: robotmonitor.py plotZ: Cannot plot Z with >2 factors (yet)')
            return
        z_size = 200
        if robot_vpr is None:
            factor_1 = self.factor_1
            factor_2 = self.factor_2
        else:
            x = self.scaler.inverse_transform(self.form_x(robot_vpr))
            factor_1 = x[:,0]
            factor_2 = x[:,1]
        factor_1 = np.linspace(factor_1.min(), factor_1.max(), z_size)
        factor_2 = np.linspace(factor_2.min(), factor_2.max(), z_size)
        factor_1_mesh, factor_2_mesh = np.meshgrid(factor_1,factor_2)
        factors_scaled = self.scaler.transform(np.c_[factor_1_mesh.ravel(), factor_2_mesh.ravel()])
        z_function = self.model.decision_function(factors_scaled).reshape([z_size, z_size])
        extent=[factor_1[0], factor_1[-1], factor_2[0], factor_2[-1]]
        return z_function, factor_1_mesh, factor_2_mesh, extent, factor_1, factor_2

    def plot_z_function(self, robot_vpr: RobotVPR=None, show_points=True, ax=None, fig=None, \
                        basic=False):
        '''
        todo
        '''
        if len(self.factors) > 2:
            print('[robotmonitor.plot_z_function]: Cannot plot Z with >2 factors (yet)')
            return

        # Plot decision function and boundary:
        z_function, factor_1_mesh, factor_2_mesh, extent, factor_1, factor_2 \
            = self.generate_z_function(robot_vpr)
        if ax is None:
            fig,ax=plt.subplots()
        if basic is True:
            ax.imshow(z_function >= 0, origin='lower', extent=extent, aspect='auto', \
                      cmap = 'gray')
        else:
            ax.imshow(z_function, origin='lower', extent=extent, aspect='auto')
            ax.contour(factor_1_mesh, factor_2_mesh, z_function, levels=[0])
        ax.set_xlabel(self.factors[0])
        ax.set_ylabel(self.factors[1])
        ax.set_title('Z')
        # Plot points:
        if show_points:
            if robot_vpr is None:
                y = self.training_y
            else:
                y = robot_vpr.y
            ax.scatter(factor_1[ y], factor_2[ y], color='g', marker='.', label='good')
            ax.scatter(factor_1[~y], factor_2[~y], color='r', marker='.', label='bad')
            ax.legend()
        return fig,ax

    def plot_p_function(self, robot_vpr: RobotVPR=None, show_points=True, ax=None, fig=None, \
                        basic=False, levels: Optional[List[float]] = None):
        '''
        todo
        '''
        if levels is None:
            levels = [0.9]
        z_size=200
        if robot_vpr is None:
            factor_1 = self.factor_1
            factor_2 = self.factor_2
        else:
            x = self.scaler.inverse_transform(self.form_x(robot_vpr))
            factor1 = x[:,0]
            factor2 = x[:,1]
        f1 = np.linspace(factor1.min(), factor1.max(), z_size)
        f2 = np.linspace(factor2.min(), factor2.max(), z_size)
        factor_1_mesh, factor_2_mesh = np.meshgrid(f1,f2)
        factors_scaled=self.scaler.transform(np.c_[factor_1_mesh.ravel(), factor_2_mesh.ravel()])
        p_function = self.model.predict_proba(factors_scaled)[:,1].reshape([z_size,z_size])
        extent=[f1[0], f1[-1], f2[0], f2[-1]]
        if ax is None:
            fig, ax = plt.subplots()
        if basic:
            ax.imshow(p_function >= 0, origin='lower', extent=extent, aspect='auto', cmap='gray')
        else:
            ax.imshow(p_function, origin='lower', extent=extent, aspect='auto')
            ax.contour(factor_1_mesh, factor_2_mesh, p_function, levels=levels)
        ax.set_xlabel(self.factors[0])
        ax.set_ylabel(self.factors[1])
        ax.set_title('P')
        # Plot points:
        if show_points:
            if robot_vpr is None:
                y = self.training_y
            else:
                y = robot_vpr.y
            ax.scatter(factor_1[ y], factor_2[ y], color='g', marker='.', label='good')
            ax.scatter(factor_1[~y], factor_2[~y], color='r', marker='.', label='bad')
            ax.legend()
        return fig,ax

class RobotMonitor2D(RobotMonitor):
    '''
    Robot Monitor using a single SVM with two factors
    '''
    def __init__(self, robot_vpr: RobotVPR, factors_in=None, sample_weight=None):
        '''
        factors_in defaults to grad, va
        '''
        print(f"SVM Factors: {factors_in}")
        if factors_in is None:
            factors_in = ["grad", "va"]
            print('Using OG factors for some reason')
        factors_in = list(np.sort(factors_in)) # alphabetize order
        assert len(np.unique(factors_in)) == 2, \
            "If _factors_in is specified (not None), user must provide two unique factors."
        factors_calc = find_factors(factors_in=factors_in, sim_matrix=robot_vpr.S, \
                                    ref_xy=robot_vpr.ref.xy, match_ind=robot_vpr.best_match, \
                                    cutoff=2, init_pos=([0,0]), _all=False, dists=None, \
                                    norm=False, return_as_dict=True)
        self.factor1        = factors_calc[factors_in[0]]
        self.factor2        = factors_calc[factors_in[1]]
        self.factor_names   = factors_in

        self.x_cal           = np.c_[self.factor1,self.factor2]
        self.scaler         = StandardScaler()
        self.x_cal_scaled    = self.scaler.fit_transform(self.x_cal, robot_vpr.y)

        self.model = svm.SVC(kernel='rbf',
                             C=1,gamma='scale',
                             class_weight='balanced',
                             probability=True)

        self.model.fit(X=self.x_cal_scaled, y=robot_vpr.y, sample_weight=sample_weight)

        # Save the training inputs in case they are needed later:
        self.training_y=robot_vpr.y
        self.training_tolerance=robot_vpr.tolerance
        self.training_s=robot_vpr.S
        self.robot_vpr = robot_vpr

        self.performance    = self.assess_prediction(robot_vpr)

    def set_factor_names(self, factor1: str, factor2: str) -> None:
        '''
        Change factor names from the default (VA ratio, and, Average Gradient)
        '''
        self.factor_names = [factor1, factor2]

    def form_x_realtime(self, vector, ref_xy, match_ind):
        '''
        vector: S
        rXY: for the set you are predicting on, the ground truth reference set XY array
        mInd: for the set you are predicting on, the best match indices
        '''
        _factors_calc = find_factors(factors_in=self.factor_names, sim_matrix=vector, ref_xy=ref_xy,
                                     match_ind=match_ind, cutoff=2, init_pos=([0,0]), _all=False,
                                     dists=None, norm=False, return_as_dict=True)
        factor1 = _factors_calc[self.factor_names[0]]
        factor2 = _factors_calc[self.factor_names[1]]
        return self.scaler.transform(np.c_[factor1,factor2])

    def form_x(self, robot_vpr: RobotVPR):
        return self.form_x_realtime(vector=robot_vpr.S, ref_xy=robot_vpr.ref.xy, \
                                   match_ind=robot_vpr.best_match)

class RobotMonitor3D(RobotMonitor):
    '''
    Robot Monitor using a single SVM with three factors
    '''
    def __init__(self, robot_vpr: RobotVPR):
        '''
        todo
        '''
        self.factor1 = find_va_factor(robot_vpr.S)
        self.factor2 = find_grad_factor(robot_vpr.S)
        self.factor3 = robot_vpr.best_match_S
        self.factors = ['VA ratio','Average Gradient','Best Match Distance']
        self.x_cal = np.c_[self.factor1,self.factor2,self.factor3]
        self.scaler = StandardScaler()
        self.x_cal_scaled = self.scaler.fit_transform(self.x_cal)
        self.model = svm.SVC(kernel='rbf',
                             C=1,gamma='scale',
                             class_weight='balanced',
                             probability=True)
        self.model.fit(self.x_cal_scaled,robot_vpr.y)
        # Save the training inputs in case they are needed later:
        self.training_y = robot_vpr.y
        self.training_tolerance = robot_vpr.tolerance
        self.training_s = robot_vpr.S

    def form_x(self, robot_vpr: RobotVPR):
        '''
        todo
        '''
        x = np.c_[find_va_factor(robot_vpr.S), find_grad_factor(robot_vpr.S), \
                  robot_vpr.best_match_S]
        return self.scaler.transform(x)

class DoubleRobotMonitor(RobotMonitor):
    '''
    Robot Monitor using two cascaded SVMs
    '''
    def __init__(self, vpr: RobotVPR, mon: RobotMonitor):
        '''
        todo
        '''
        self.factor1 = mon.decision_values(vpr)
        self.factor2 = vpr.best_match_S
        self.first_svm = copy.copy(mon)
        self.y = vpr.y
        self.factors=['zvalues','best match distances']
        x_cal = np.c_[self.factor1,self.factor2]
        self.scaler = StandardScaler()
        x_cal_scaled = self.scaler.fit_transform(x_cal)
        self.model = svm.SVC(kernel='rbf', C=1,gamma='scale', class_weight='balanced',
                             probability=True)
        self.model.fit(x_cal_scaled,self.y)
        # Save the training inputs in case they are needed later:
        self.training_y = vpr.y
        self.training_tolerance = vpr.tolerance
        self.training_s = vpr.S

    def form_x(self, robot_vpr: RobotVPR):
        '''
        todo
        '''
        zvals = self.first_svm.decision_values(robot_vpr)
        bms = robot_vpr.best_match_S
        x = np.c_[zvals,bms]
        return self.scaler.transform(x)

    def plot_z_function(self, robot_vpr: RobotVPR=None, show_points=True, ax=None, fig=None, \
                        basic=False):
        '''
        todo
        '''
        fig,ax=super().plot_z_function(robot_vpr,show_points,ax,fig,basic)
        ax.axvline(0,ls='dashed',color='blue')
        return fig,ax
