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
from pyaarapsi.vpred.vpred_factors import find_factors, find_va_factor, find_grad_factor
from pyaarapsi.vpred.vpred_tools import find_prediction_performance_metrics
from pyaarapsi.vpred.robotvpr import RobotVPR

class RobotMonitor(ABC):
    '''
    Base Class for Robot Monitor objects
    '''
    model: svm.SVC
    factors: List[str]
    factor1: Union[NDArray, list]
    factor2: Union[NDArray, list]
    scaler: StandardScaler
    training_y: Union[NDArray, list]

    @abstractmethod
    def formX(self, vpr):
        '''
        todo
        '''

    def predict_quality(self,vpr):
        '''
        todo
        '''
        return self.model.predict(self.formX(vpr))

    def probability(self,vpr):
        '''
        todo
        '''
        return self.model.predict_proba(self.formX(vpr))[:,1]

    def decision_values(self,vpr):
        '''
        todo
        '''
        return self.model.decision_function(self.formX(vpr))

    def assess_prediction(self,vpr, verbose=False):
        '''
        todo
        '''
        y_pred=self.predict_quality(vpr)
        [precision, recall, num_tp, num_fp, num_tn, num_fn] = \
            find_prediction_performance_metrics(y_pred,vpr.y,verbose=verbose)
        return {'precision': precision, 'recall': recall, 'num_tp': num_tp, 'num_fp': num_fp,
                'num_tn': num_tn, 'num_fn': num_fn}

    def generate_pred_prob_z(self,vpr):
        '''
        todo
        '''
        X_scaled = self.formX(vpr)
        y_pred = self.model.predict(X_scaled)
        y_pred_prob = self.model.predict_proba(X_scaled)[:,1]
        y_zvalues = self.model.decision_function(X_scaled)
        return y_pred, y_pred_prob, y_zvalues

    def generate_Z(self,vpr=None):
        '''
        todo
        '''
        if len(self.factors) > 2:
            print('Error: robotmonitor.py plotZ: Cannot plot Z with >2 factors (yet)')
            return
        ZSIZE=200
        if vpr == None:
            factor1=self.factor1
            factor2=self.factor2
        else:
            X=self.scaler.inverse_transform(self.formX(vpr))
            factor1=X[:,0]
            factor2=X[:,1]
        f1=np.linspace(factor1.min(),factor1.max(),ZSIZE)
        f2=np.linspace(factor2.min(),factor2.max(),ZSIZE)
        F1,F2=np.meshgrid(f1,f2)
        Fscaled=self.scaler.transform(np.c_[F1.ravel(), F2.ravel()])
        Z=self.model.decision_function(Fscaled).reshape([ZSIZE,ZSIZE])
        extent=[f1[0],f1[-1],f2[0],f2[-1]]
        return Z,F1,F2,extent,factor1,factor2

    def plotZ(self,vpr=None,show_points=True,ax=None,fig=None,basic=False):
        '''
        todo
        '''
        if len(self.factors) > 2:
            print('Error: robotmonitor.py plotZ: Cannot plot Z with >2 factors (yet)')
            return

        # Plot decision function and boundary:
        Z,F1,F2,extent,factor1,factor2=self.generate_Z(vpr)
        if ax == None:
            fig,ax=plt.subplots()
        if basic == True:
            ax.imshow(Z>=0,origin='lower',extent=extent,aspect='auto',cmap='gray')
        else:
            ax.imshow(Z,origin='lower',extent=extent,aspect='auto')
            ax.contour(F1,F2,Z,levels=[0])
        ax.set_xlabel(self.factors[0])
        ax.set_ylabel(self.factors[1])
        ax.set_title('Z')

        # Plot points:
        if show_points:
            if vpr == None:
                y=self.training_y
            else:
                y=vpr.y
            ax.scatter(factor1[ y],factor2[ y],color='g',marker='.',label='good')
            ax.scatter(factor1[~y],factor2[~y],color='r',marker='.',label='bad')
            ax.legend()
        return fig,ax

    def plotP(self, vpr=None, show_points=True, ax=None, fig=None, basic=False,
              levels: Optional[List[float]] = None):
        '''
        todo
        '''
        if levels is None:
            levels = [0.9]
        ZSIZE=200
        if vpr == None:
            factor1=self.factor1
            factor2=self.factor2
        else:
            X=self.scaler.inverse_transform(self.formX(vpr))
            factor1=X[:,0]
            factor2=X[:,1]
        f1=np.linspace(factor1.min(),factor1.max(),ZSIZE)
        f2=np.linspace(factor2.min(),factor2.max(),ZSIZE)    
        F1,F2=np.meshgrid(f1,f2)
        Fscaled=self.scaler.transform(np.c_[F1.ravel(), F2.ravel()])
        P=self.model.predict_proba(Fscaled)[:,1].reshape([ZSIZE,ZSIZE])
        extent=[f1[0],f1[-1],f2[0],f2[-1]]
        
        if ax == None:
            fig,ax=plt.subplots()
        if basic == True:
            ax.imshow(P>=0,origin='lower',extent=extent,aspect='auto',cmap='gray')
        else:
            ax.imshow(P,origin='lower',extent=extent,aspect='auto')
            ax.contour(F1,F2,P,levels=levels)
        ax.set_xlabel(self.factors[0])
        ax.set_ylabel(self.factors[1])
        ax.set_title('P')

        # Plot points:
        if show_points:
            if vpr == None:
                y=self.training_y
            else:
                y=vpr.y
            ax.scatter(factor1[ y],factor2[ y],color='g',marker='.',label='good')
            ax.scatter(factor1[~y],factor2[~y],color='r',marker='.',label='bad')
            ax.legend()
        return fig,ax

class RobotMonitor2D(RobotMonitor):
    '''
    Robot Monitor using a single SVM with two factors
    '''
    def __init__(self, rvpr: RobotVPR, factors_in=None, sample_weight=None):
        '''
        factors_in defaults to grad, va
        '''
        print("SVM Factors: {}".format(factors_in))
        if factors_in is None:
            factors_in = ["grad", "va"]
            print('Using OG factors for some reason')
        factors_in = list(np.sort(factors_in)) # alphabetize order
        assert len(np.unique(factors_in)) == 2, \
            "If _factors_in is specified (not None), user must provide two unique factors."
        factors_calc = find_factors(factors_in=factors_in, _S=rvpr.S, rXY=rvpr.ref.xy,
                        match_ind=rvpr.best_match, cutoff=2, init_pos=([0,0]), _all=False,
                        dists=None, norm=False, return_as_dict=True)
        self.factor1        = factors_calc[factors_in[0]]
        self.factor2        = factors_calc[factors_in[1]]
        self.factor_names   = factors_in

        self.Xcal           = np.c_[self.factor1,self.factor2]
        self.scaler         = StandardScaler()
        self.Xcal_scaled    = self.scaler.fit_transform(self.Xcal, rvpr.y)

        self.model = svm.SVC(kernel='rbf',
                             C=1,gamma='scale',
                             class_weight='balanced',
                             probability=True)

        self.model.fit(X=self.Xcal_scaled, y=rvpr.y, sample_weight=sample_weight)

        # Save the training inputs in case they are needed later:
        self.training_y=rvpr.y
        self.training_tolerance=rvpr.tolerance
        self.training_S=rvpr.S
        self.rvpr = rvpr

        self.performance    = self.assess_prediction(rvpr)

    def set_factor_names(self, factor1: str, factor2: str) -> None:
        '''
        Change factor names from the default (VA ratio, and, Average Gradient)
        '''
        self.factor_names = [factor1, factor2]

    def formX_realtime(self, vector, rXY, mInd):
        '''
        vector: S
        rXY: for the set you are predicting on, the ground truth reference set XY array
        mInd: for the set you are predicting on, the best match indices
        '''
        _factors_calc = find_factors(factors_in=self.factor_names, _S=vector, rXY=rXY,
                                     match_ind=mInd, cutoff=2, init_pos=([0,0]), _all=False,
                                     dists=None, norm=False, return_as_dict=True)
        factor1 = _factors_calc[self.factor_names[0]]
        factor2 = _factors_calc[self.factor_names[1]]
        return self.scaler.transform(np.c_[factor1,factor2])

    def formX(self, vpr: RobotVPR):
        return self.formX_realtime(vector=vpr.S, rXY=vpr.ref.xy, mInd=vpr.best_match)

class RobotMonitor3D(RobotMonitor):
    '''
    Robot Monitor using a single SVM with three factors
    '''
    def __init__(self,vpr):
        '''
        todo
        '''
        self.factor1 = find_va_factor(vpr.S)
        self.factor2 = find_grad_factor(vpr.S)
        self.factor3 = vpr.best_match_S
        self.factors = ['VA ratio','Average Gradient','Best Match Distance']
        self.Xcal = np.c_[self.factor1,self.factor2,self.factor3]
        self.scaler = StandardScaler()
        self.Xcal_scaled = self.scaler.fit_transform(self.Xcal)
        self.model = svm.SVC(kernel='rbf',
                             C=1,gamma='scale',
                             class_weight='balanced',
                             probability=True)
        
        self.model.fit(self.Xcal_scaled,vpr.y)
        # Save the training inputs in case they are needed later:
        self.training_y=vpr.y
        self.training_tolerance=vpr.tolerance
        self.training_S=vpr.S
        return

    def formX(self,vpr):
        '''
        todo
        '''
        X = np.c_[find_va_factor(vpr.S),find_grad_factor(vpr.S),vpr.best_match_S]
        return self.scaler.transform(X)

class DoubleRobotMonitor(RobotMonitor):
    '''
    Robot Monitor using two cascaded SVMs
    '''
    def __init__(self, vpr: RobotVPR, mon: RobotMonitor):
        '''
        todo
        '''
        self.factor1=mon.decision_values(vpr)
        self.factor2=vpr.best_match_S
        self.first_SVM=copy.copy(mon)
        self.y=vpr.y
        self.factors=['zvalues','best match distances']
        Xcal = np.c_[self.factor1,self.factor2]
        self.scaler = StandardScaler()
        Xcal_scaled = self.scaler.fit_transform(Xcal)
        self.model = svm.SVC(kernel='rbf', C=1,gamma='scale', class_weight='balanced',
                             probability=True)
        self.model.fit(Xcal_scaled,self.y)
        # Save the training inputs in case they are needed later:
        self.training_y=vpr.y
        self.training_tolerance=vpr.tolerance
        self.training_S=vpr.S

    def formX(self,vpr):
        '''
        todo
        '''
        zvals=self.first_SVM.decision_values(vpr)
        bms=vpr.best_match_S
        X=np.c_[zvals,bms]
        return self.scaler.transform(X)

    def plotZ(self,vpr=None,show_points=True,ax=None,fig=None,basic=False):
        '''
        todo
        '''
        fig,ax=super().plotZ(vpr,show_points,ax,fig,basic)
        ax.axvline(0,ls='dashed',color='blue')
        return fig,ax
