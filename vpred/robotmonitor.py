# robotmonitor.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from copy import copy
from .robotvpr import *

class RobotMonitor2D:
    
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
    
    def generate_Z(self):
        SIZE=200
        f1=np.linspace(0,self.factor1.max(),SIZE)
        f2=np.linspace(0,self.factor2.max(),SIZE)    
        F1,F2=np.meshgrid(f1,f2)
        Fscaled=self.scaler.transform(np.c_[F1.ravel(), F2.ravel()])
        self.Z=self.model.decision_function(Fscaled).reshape([SIZE,SIZE])
        self.extent=[0,f1[-1],0,f2[-1]]
        self.F1=F1
        self.F2=F2
        self.Fscaled=Fscaled
        return self.Z,self.extent
    
    def plotZ(self,show_points=False):
        fig,ax=plt.subplots()
        self.generate_Z();
        ax.imshow(self.Z,extent=self.extent,origin='lower',aspect='auto');
        ax.contour(self.F1,self.F2,self.Z,levels=[0]);
        if show_points:
            ax.scatter(self.factor1,self.factor2,color=np.where(self.training_y,'g','r'),marker='.')
        ax.set_xlabel(self.factors[0])
        ax.set_ylabel(self.factors[1]);
        ax.set_title('Z');
        return fig,ax
    
    def plot_testZ(self,vpr):
        ZSIZE=200
        X=self.scaler.inverse_transform(self.formX(vpr))
        factor1=X[:,0]
        factor2=X[:,1]
        f1=np.linspace(factor1.min(),factor1.max(),ZSIZE)
        f2=np.linspace(factor2.min(),factor2.max(),ZSIZE)    
        F1,F2=np.meshgrid(f1,f2)
        Fscaled=self.scaler.transform(np.c_[F1.ravel(), F2.ravel()])
        Z=self.model.decision_function(Fscaled).reshape([ZSIZE,ZSIZE])
        extent=[f1[0],f1[-1],f2[0],f2[-1]]
        fig,ax=plt.subplots();
        ax.imshow(Z,origin='lower',extent=extent,aspect='auto');
        ax.contour(F1,F2,Z,levels=[0])
        ax.scatter(factor1,factor2,color=np.where(vpr.y,'g','r'),marker='.');
        return fig,ax
    
    
class RobotMonitor3D:
    
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
    
class DoubleRobotMonitor:
    
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
    
    def formX(self,vpr):
        zvals=self.first_SVM.decision_values(vpr)
        bms=vpr.best_match_S
        X=np.c_[zvals,bms]
        return self.scaler.transform(X)
    
    def predict_quality(self,vpr):
        return self.model.predict(self.formX(vpr))
    
    def decision_values(self,vpr):
        return self.model.decision_function(self.formX(vpr))
    
    def probability(self,vpr):
        return self.model.predict_proba(self.formX(vpr))[:,1]
    
    def assess_prediction(self,vpr):
        y_pred=self.predict_quality(vpr)
        find_prediction_performance_metrics(y_pred,vpr.y,verbose=True);
        return
    
    def plotZ(self,vpr):
        ZSIZE=200
        factor1=self.first_SVM.decision_values(vpr)
        factor2=vpr.best_match_S
        f1=np.linspace(factor1.min(),factor1.max(),ZSIZE)
        f2=np.linspace(factor2.min(),factor2.max(),ZSIZE)    
        F1,F2=np.meshgrid(f1,f2)
        Fscaled=self.scaler.transform(np.c_[F1.ravel(), F2.ravel()])
        Z=self.model.decision_function(Fscaled).reshape([ZSIZE,ZSIZE])
        extent=[f1[0],f1[-1],f2[0],f2[-1]]

        fig,ax=plt.subplots();
        ax.imshow(Z,origin='lower',extent=extent,aspect='auto');
        ax.contour(F1,F2,Z,levels=[0])
        ax.scatter(factor1,factor2,color=np.where(vpr.y,'g','r'),marker='.');
        ax.axvline(0,ls='dashed',color='blue')