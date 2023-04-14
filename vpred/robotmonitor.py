# robotmonitor.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from copy import copy
from .vpred_factors import *
from .robotvpr import *

class RobotMonitor(object):

    def __init__(self,vpr):
        # TODO: Not sure if this is needed
        pass

    def predict_quality(self,vpr):
        return self.model.predict(self.formX(vpr))

    def probability(self,vpr):
        return self.model.predict_proba(self.formX(vpr))[:,1]

    def decision_values(self,vpr):
        return self.model.decision_function(self.formX(vpr))      

    def assess_prediction(self,vpr):
        y_pred=self.predict_quality(vpr)
        find_prediction_performance_metrics(y_pred,vpr.y,verbose=True);
        return

    def generate_pred_prob_z(self,vpr):
        X_scaled = self.formX(vpr)
        y_pred = self.model.predict(X_scaled)
        y_pred_prob = self.model.predict_proba(X_scaled)[:,1]
        y_zvalues = self.model.decision_function(X_scaled)
        return y_pred, y_pred_prob, y_zvalues

    def generate_Z(self,vpr=None):

        if len(self.factors) > 2:
            print('Error: robotmonitor.py plotZ: Cannot plot Z with >2 factors (yet)');
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
    
    def plotZ(self,vpr=None,show_points=True):
        
        if len(self.factors) > 2:
            print('Error: robotmonitor.py plotZ: Cannot plot Z with >2 factors (yet)');
            return

        # Plot decision function and boundary:
        Z,F1,F2,extent,factor1,factor2=self.generate_Z(vpr)
        fig,ax=plt.subplots();
        ax.imshow(Z,origin='lower',extent=extent,aspect='auto');
        ax.contour(F1,F2,Z,levels=[0])
        ax.set_xlabel(self.factors[0])
        ax.set_ylabel(self.factors[1]);
        ax.set_title('Z');

        # Plot points:
        if show_points:
            if vpr == None:
                y=self.training_y
            else:
                y=vpr.y
            ax.scatter(factor1[ y],factor2[ y],color='g',marker='.',label='good');
            ax.scatter(factor1[~y],factor2[~y],color='r',marker='.',label='bad');

            # Use this to modify to plot TP/FP/TN/FN in different colours:
            # tp,fp,tn,fn=find_each_prediction_metric(self.predict_quality(vpr),vpr.y)
            # ax.scatter(factor1[tp],factor2[tp],color='green',marker='.',label='TP');
            # ax.scatter(factor1[fp],factor2[fp],color='red',marker='.',label='FP');
            # ax.scatter(factor1[fn],factor2[fn],color='blue',marker='.',label='FN');
            # ax.scatter(factor1[tn],factor2[tn],color='orange',marker='.',label='TN');
            
            ax.legend();
    
        return fig,ax

class RobotMonitor2D(RobotMonitor):
    
    def __init__(self,vpr):
        self.factor1 = find_va_factor(vpr.S)
        self.factor2 = find_grad_factor(vpr.S)
        self.factors = ['VA ratio','Average Gradient']
        self.Xcal = np.c_[self.factor1,self.factor2]
        self.scaler = StandardScaler()
        self.Xcal_scaled = self.scaler.fit_transform(self.Xcal)
        self.model = svm.SVC(kernel='rbf',
                             C=1,gamma='scale',
                             class_weight='balanced',
                             probability=True)

        self.model.fit(self.Xcal_scaled,vpr.y);

        # Save the training inputs in case they are needed later:
        self.training_y=vpr.y;
        self.training_tolerance=vpr.tolerance;
        self.training_S=vpr.S;
        return
    
    def formX(self,vpr):
        X = np.c_[find_va_factor(vpr.S),find_grad_factor(vpr.S)]
        return self.scaler.transform(X)
        

class RobotMonitor3D(RobotMonitor):
    
    def __init__(self,vpr):
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
        
        self.model.fit(self.Xcal_scaled,vpr.y);
        
        # Save the training inputs in case they are needed later:
        self.training_y=vpr.y;
        self.training_tolerance=vpr.tolerance;
        self.training_S=vpr.S;
        return
    
    def formX(self,vpr):
        X = np.c_[find_va_factor(vpr.S),find_grad_factor(vpr.S),vpr.best_match_S]
        return self.scaler.transform(X)
    
class DoubleRobotMonitor(RobotMonitor):
    
    def __init__(self, vpr, mon):
        self.factor1=mon.decision_values(vpr)
        self.factor2=vpr.best_match_S
        self.first_SVM=copy(mon)
        self.y=vpr.y
        self.factors=['zvalues','best match distances']
        Xcal = np.c_[self.factor1,self.factor2]
        self.scaler = StandardScaler()
        Xcal_scaled = self.scaler.fit_transform(Xcal)
        self.model = svm.SVC(kernel='rbf',
                        C=1,gamma='scale', 
                        class_weight='balanced', 
                        probability=True)
        self.model.fit(Xcal_scaled,self.y);
        
        # Save the training inputs in case they are needed later:
        self.training_y=vpr.y;
        self.training_tolerance=vpr.tolerance;
        self.training_S=vpr.S;
    
    def formX(self,vpr):
        zvals=self.first_SVM.decision_values(vpr)
        bms=vpr.best_match_S
        X=np.c_[zvals,bms]
        return self.scaler.transform(X)