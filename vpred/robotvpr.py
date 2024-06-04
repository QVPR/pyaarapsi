#!/usr/bin/env python3
'''
RobotVPR class
'''
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from pyaarapsi.vpred.vpred_tools import find_each_prediction_metric, find_best_match, \
    create_normalised_similarity_matrix, create_similarity_matrix, find_best_match_distances, \
        find_frame_error, find_vpr_performance_metrics
from pyaarapsi.vpred.gradseq_tools import create_gradient_matrix, find_best_match_G, \
    find_best_match_distances_G, find_consensus
from pyaarapsi.vpred.robotrun import RobotRun
from pyaarapsi.core.helper_tools import angle_wrap, m2m_dist

def compute_distance_in_m(a,b):
    '''
    Compute distance in m between two sets of [x,y] coordinates
    '''
    return np.diag(m2m_dist(a,b))

def plot_PR(r,p,ax=None,fig=None):
    '''
    TODO
    '''
    if ax is None:
        fig,ax=plt.subplots()
    ax.plot(r,p)
    ax.set_title('PR Curve')
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_ylim(0,1.05)
    ax.set_xlim(0,1)
    return fig,ax

class RobotVPR:
    '''
    Class for processing VPR runs, consisting of a query run against a reference.
    A query can be added after initialisation.
    '''
    gt_match: NDArray
    frame_error: NDArray
    y: NDArray
    yaw_error: NDArray
    description: str
    G: NDArray
    G_match: NDArray
    best_match_G: NDArray
    consensus: NDArray
    Sw: NDArray
    def __init__(self,reference: RobotRun, query=None,norm=True):
        """Constructs the attributes of a RobotVPR object.
        
        Parameters
        ----------
            reference : RobotRun
                reference set
            query : RobotRun (optional)
                query set
        """
        self.ref=reference
        self.num_refs=reference.imgnum
        self.norm = norm
        if query is not None:
            self.set_query(query)
        self.tolerance = 1 # default tolerance is one frame
        self.units = 'frames'

    def set_query(self,query: RobotRun):
        '''
        Add a query run as a RobotRun object
        Once added, the VPR parameters will be computed, including distance matrix and best match
        TODO: rework when matches do not exist for each query
        '''
        self.qry = query
        self.num_qrys = query.imgnum
        if self.norm:
            self.S, self.Srefmean, self.Srefstd  = \
                create_normalised_similarity_matrix(self.ref.features,self.qry.features)
        else:
            self.S = create_similarity_matrix(self.ref.features,self.qry.features)
        self.best_match = find_best_match(self.S)
        self.best_match_S = find_best_match_distances(self.S)
        self.ALL_TRUE = np.full(self.num_qrys,True,dtype='bool')
        if self.ref.has_odom_data and self.qry.has_odom_data:
            self.gt_distances = m2m_dist(self.ref.xy, self.qry.xy)
            self.gt_match = self.gt_distances.argmin(axis=0)
            self.frame_error = abs(self.best_match-self.gt_match)
            self.min_error = self.gt_distances.min(axis=0)
            self.abs_error = self.gt_distances[self.best_match, np.arange(self.num_qrys)]
            self.ref_error = np.diag(m2m_dist(self.ref.xy[self.gt_match],self.ref.xy[self.best_match]))
        self.match_exists = self.ALL_TRUE

    def set_actual_frame_match(self, ref_frames=None):
        '''
        TODO
        '''
        if ref_frames == None:
            self.gt_match = np.arange(self.num_qrys)
        elif len(ref_frames) == self.num_qrys:
            self.gt_match = ref_frames  # default is a one-to-one match between reference and query images
        else:
            print('ERROR: set_actual_frame_match: number of frames must match number of queries')
            return
        self.frame_error=find_frame_error(self.S,self.gt_match)
    
    def add_query(self,feature_vector):
        '''
        TODO
        '''
        return
    
    def show_S(self):
        '''
        Plot distance matrix
        '''
        fig,ax=plt.subplots()
        ax.imshow(self.S)
        ax.set_xlabel('query')
        ax.set_ylabel('reference')
        ax.set_title('Distance Matrix')
        return fig,ax
    
    def distance_vector(self, qry_num):
        '''
        Return a single vector for a query in the distance matrix
        '''
        return self.S[:,qry_num]

    def find_y(self):
        '''
        Return a boolean vector representing whether or not each query is in tolerance (True), or not (False)
        '''
        if self.units == 'frames':
            self.y = (self.frame_error <= self.tolerance)
        elif self.units == 'm':
            self.y = (self.ref_error <= self.tolerance)
        elif self.units == 'degrees':
            # difference between the ground_truth yaw of the query image, and the yaw of the best matching
            # reference image
            self.yaw_error = angle_wrap(self.qry.yaw - self.ref.yaw[self.best_match])
            self.y = (abs(self.yaw_error) <= self.tolerance)
        else:
            print('Error: find_y: tolerance units needs to be in frames, m or degrees but is {0}'.format(self.units))
        return
    
    def set_tolerance(self, tolerance, units, verbose=True):
        '''
        Set tolerance in meters or frames
        '''
        #TODO: add error checking here for
        # - units in 'frames' or 'm'
        # - if frames, check tolerance is an integer
        self.tolerance=tolerance
        self.units=units
        # Check---
        #if self.units == 'm':
        #    self.match_exists=(self.min_error<=self.tolerance)
        #--------
        if verbose:
            print('tolerance={0} {1}'.format(self.tolerance, self.units))

    def set_description(self, description: str):
        '''
        setter
        '''
        self.description = description

    def assess_performance(self, tolerance=1, units='frames', match_found=None, verbose=True):
        '''
        Compute performance parameters of VPR technique given tolerance in frames or units
        
        Note the match_found vector can be used where true negatives exist in the data.
        Here, match_found=None is not referring to no matches found, but simply not specifying matches found
        '''
        self.set_tolerance(tolerance,units,verbose=verbose)
        self.find_y()
        if match_found is None:
            match_found=self.ALL_TRUE
        return find_vpr_performance_metrics(match_found,self.y,self.match_exists,verbose=verbose)

    def plot_PRcurve(self,ax=None):
        '''
        Plot precision-recall curve of VPR technique
        '''
        p,r,_,_,_,_,_=self.compute_PRcurve_metrics()
        return plot_PR(r,p,ax)

    def plot_cl_vs_bl_PRcurve(self,y_pred,ax=None,fig=None):
        '''
        Plot Precision-Recall curves for baseline VPR technique (using minimum distance as a threshold)
        vs the closed-loop system (which retains only good points, then applies the distance threshold)
        '''
        if ax is None:
            fig, ax = plt.subplots()
        p,r,_,_,_,_,_=self.compute_PRcurve_metrics()
        ax.plot(r,p,label='baseline')
        p,r,_,_,_,_,_=self.compute_cl_PRcurve_metrics(y_pred)
        ax.plot(r,p,label='closed-loop')
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1.05)
        ax.set_title('PR Curve, Tolerance={0}'.format(self.tolerance))
        ax.legend()
        return fig,ax

    def compute_cl_PRcurve_metrics(self,y_pred):
        '''
        Compute precision, recall and numbers of TP/FP/TN/FN for closed-loop system as the
        minimum distance threshold is swept over 2000 points
        '''
        num_pts = 2000                # number of points in PR-curve
            
        bm_distances = find_best_match_distances(self.S)
        assert len(bm_distances) == len(y_pred)
        
        d_sweep = np.linspace(bm_distances.min(), bm_distances.max(), num_pts)
        p=np.full_like(d_sweep,np.nan)
        r=np.full_like(d_sweep,np.nan)
        tp=np.full_like(d_sweep,np.nan)
        fp=np.full_like(d_sweep,np.nan)
        tn=np.full_like(d_sweep,np.nan)
        fn=np.full_like(d_sweep,np.nan)

        for i, threshold in enumerate(d_sweep):
            match_found = bm_distances <= threshold
            (p[i], r[i], tp[i], fp[i], tn[i], fn[i]) = find_vpr_performance_metrics(
                match_found & y_pred,self.y,self.match_exists,verbose=False)
        return p,r,tp,fp,tn,fn,d_sweep

    def compute_PRcurve_metrics(self):
        '''
        Returns precision, recall and numbers of TP/FP/TN/FN for baseline VPR technique as the
        minimum distance threshold is swept through all values
        '''
        return self.compute_cl_PRcurve_metrics(self.ALL_TRUE)

    def plot_runs(self):
        '''
        TODO
        '''
        if self.ref.has_odom_data and self.qry.has_odom_data:
            plt.plot(self.ref.x,self.ref.y,label='ref')
            plt.plot(self.qry.x,self.qry.y,label='qry')
            plt.legend()
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.title('Qry and Ref Runs')
        else:
            print('Info: plot_runs: Cannot plot as odometry data is missing.')
        return

    def plot_ypred(self,y_pred,ax=None,fig=None):
        '''
        Plot the route showing TP and FP prediction locations
        '''
        if ax is None:
            fig,ax=plt.subplots()
        if self.qry.has_odom_data:
            tp, fp, _, _ = find_each_prediction_metric(y_pred,self.y)
            ax.scatter(self.qry.x,    self.qry.y,    marker='.',color='lightgray')
            ax.scatter(self.qry.x[tp],self.qry.y[tp],marker='.',color='g',label='TP')
            ax.scatter(self.qry.x[fp],self.qry.y[fp],marker='x',color='r',label='FP')
            ax.legend()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('TP and FP locations along query route')
        else:
            print('Info: plot_ypred: query run has no odometry data, cannot plot route')
        return fig,ax
    
    def find_gap_distances(self,y_pred,verbose=False):
        '''
        Return a vector containing xy-distances between predicted good points, taken along
        the query route
        '''
        pred_idx=np.where(y_pred)[0]
        gaps=np.zeros(len(pred_idx)-1)
        for j in range(len(gaps)):
            gaps[j]=np.linalg.norm(self.qry.xy[pred_idx[j+1]]-self.qry.xy[pred_idx[j]])
        if verbose:
            print('Max gap    = {0:3.2f}m'.format(gaps.max()))
            print('Mean gap   = {0:3.2f}m'.format(gaps.mean()))
            print('Median gap = {0:3.2f}m'.format(np.median(gaps)))
            print('80th,90th,95th,99th percentiles = {0} m'.format(np.percentile(gaps,[80,90,95,99])))
        return gaps
    
    def plot_error_along_route(self,ypred):
        '''
        todo
        '''
        tp,fp,tn,fn=find_each_prediction_metric(ypred,self.y)
        fig,ax=plt.subplots(figsize=(20,4))
        qrys=np.arange(self.num_qrys)
        if self.units == 'm':
            error = self.ref_error
        elif self.units == 'frames':
            error = self.frame_error
        elif self.units == 'degrees':
            error = self.yaw_error
        else:
            print('Error: robotvpr.py: plot_error_along_route: units needs to be '
                  'either m, frames or degrees')
            return None, None
        ax.scatter(qrys[fn],error[fn],marker='.',color='lightblue',label='FN (removed)')
        ax.scatter(qrys[tn],error[tn],marker='.',color='lightgray',label='TN (removed)')
        ax.scatter(qrys[tp],error[tp],marker='.',color='g',label='TP (retained)')
        ax.scatter(qrys[fp],error[fp],marker='.',color='r',label='FP (retained)')
        ax.axhline(self.tolerance,ls='dashed',label='Tolerance')
        ax.set_xlabel('query number')
        ax.set_ylabel('error ({0})'.format(self.units))
        ax.set_title('Error along the query route - showing retained (predicted good), '
                     'and removed points')
        ax.legend()
        return fig,ax
    
    def setup_G(self,tol=1):
        '''
        todo
        '''
        self.G = create_gradient_matrix(self.S)
        self.G_match = find_best_match_G(self.G)
        self.best_match_G = find_best_match_distances_G(self.G)
        self.consensus = find_consensus(self.S,self.G,tolerance=tol)
        return
    
    def create_weighted_matrix(self,pred,w=0.99):
        '''
        todo
        '''
        self.Sw = self.S.copy()
        qrys = np.arange(self.num_qrys, dtype='uint')
        d_new = self.best_match_S - w * (self.best_match_S - self.best_match_S.min())
        # TODO: vectorise
        for q in qrys:
            if pred[q]:
                self.Sw[self.best_match[q], q] = d_new[q]
        return self.Sw

    def __repr__(self):  # Return a string containing a printable representation of an object.
        return f"{self.__class__.__name__}(ref={self.ref.description},qry={self.qry.description})"

class RobotVPR_fromS(RobotVPR):
    '''
    todo
    '''
    def __init__(self,S,actual_match,description=""):
        """
        Constructs the attributes of a RobotVPR object, based on a distance matrix
        """
        # super(RobotVPR_fromS, self).__init__()
        self.S=S
        self.gt_match=actual_match
        self.num_refs=S.shape[0]
        self.num_qrys=S.shape[1]
        self.tolerance = 1 # default tolerance is one frame
        self.units = 'frames'
        self.description=description
        self.best_match = find_best_match(self.S)
        self.best_match_S = find_best_match_distances(self.S)
        self.ALL_TRUE = np.full(self.num_qrys,True,dtype='bool')
        self.frame_error = abs(self.best_match-self.gt_match)     # number of frames between VPR match and actual match
        # TODO: rework when matches do not exist for each query
        self.match_exists = self.ALL_TRUE
        self.assess_performance()

    def plot_ypred(self,y_pred,ax=None,fig=None):
        raise NotImplementedError("plot_ypred")

    def __repr__(self):  # Return a string containing a printable representation of an object.
        return f"{self.__class__.__name__}(description={self.description},"\
                "num_refs={self.num_refs},num_qrys={self.num_qrys})"