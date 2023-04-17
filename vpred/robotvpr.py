#robotvpr.py

import numpy as np
import matplotlib.pyplot as plt
from .vpred_tools import *
from .robotrun import *
from scipy.spatial.distance import cdist

def compute_distance_in_m(a,b):
    '''
    Compute distance in m between two sets of [x,y] coordinates
    '''
    return np.diag(cdist(a,b))

def plot_PR(r,p):
    fig,ax=plt.subplots()
    ax.plot(r,p);
    ax.set_title('PR Curve');
    ax.set_xlabel('recall');
    ax.set_ylabel('precision');
    ax.set_ylim([0,1.05])
    ax.set_xlim([0,1])
    return fig,ax

class RobotVPR:
    '''
    Class for processing VPR runs, consisting of a query run against a reference.
    A query can be added after initialisation.
    '''
    
    def __init__(self,reference,query=None):
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
        if query != None:
            self.set_query(query)
        self.tolerance = 1 # default tolerance is one frame
        self.units = 'frames'
    
    def set_query(self,query):
        '''
        Add a query run as a RobotRun object
        Once added, the VPR parameters will be computed, including distance matrix and best match
        '''
        self.qry = query
        self.num_qrys = query.imgnum
        self.S, self.Srefmean, self.Srefstd  = create_normalised_similarity_matrix(self.ref.features,self.qry.features)
        self.best_match = find_best_match(self.S)
        self.best_match_S = find_best_match_distances(self.S)
        self.ALL_TRUE = np.full(self.num_qrys,True,dtype='bool')
        
        if self.ref.GEO_TAGGED and self.qry.GEO_TAGGED:
            self.gt_distances = cdist(self.ref.xy, self.qry.xy)
            self.gt_match = self.gt_distances.argmin(axis=0)          # index of the closest reference for each query
            self.frame_error = abs(self.best_match-self.gt_match)     # number of frames between VPR match and actual match
            #self.abs_error = compute_distance_in_m(self.ref.xy[gt_match],self.qry.xy)              # distance in m between matching points
            self.min_error = self.gt_distances.min(axis=0)
            self.abs_error = self.gt_distances[self.best_match, np.arange(self.num_qrys)]
            self.ref_error = np.diag(cdist(self.ref.xy[self.gt_match],self.ref.xy[self.best_match])) # distance in m along reference route only
        
        # TODO: rework when matches do not exist for each query
        self.match_exists = self.ALL_TRUE
        
            
    def set_actual_frame_match(self, ref_frames=None):
        if ref_frames == None:
            self.gt_match = np.arange(self.num_qrys)
        elif len(ref_frames) == self.num_qrys:
            self.gt_match = ref_frames  # default is a one-to-one match between reference and query images
        else:
            print('ERROR: set_actual_frame_match: number of frames must match number of queries')
            return
        self.frame_error=find_frame_error(self.S,self.gt_match)
    
    def add_query(self,feature_vector):
        #TODO
        return
    
    def show_S(self):
        '''
        Plot distance matrix
        '''
        fig,ax=plt.subplots()
        ax.imshow(self.S);
        ax.set_xlabel('query');
        ax.set_ylabel('reference');
        ax.set_title('Distance Matrix');
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
        else:
            print('Error: find_y: tolerance units needs to be in frame or units but is {0}'.format(self.units))
    
    def set_tolerance(self, tolerance, units):
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
        print('tolerance={0} {1}'.format(self.tolerance, self.units))
        
    def set_description(self,description):
        self.description = description
    
    def assess_performance(self, tolerance=1, units='frames', match_found=None):
        '''
        Compute performance parameters of VPR technique given tolerance in frames or units
        
        Note the match_found vector can be used where true negatives exist in the data.
        Here, match_found=None is not referring to no matches found, but simply not specifying matches found
        '''
        self.set_tolerance(tolerance,units)            
        print('tolerance={0} {1}'.format(tolerance, units))
        self.find_y()
        if match_found is None:
            match_found=self.ALL_TRUE
        return find_vpr_performance_metrics(match_found,self.y,self.match_exists,verbose=True);
    
    def plot_PRcurve(self):
        '''
        Plot precision-recall curve of VPR technique
        '''
        p,r,_,_,_,_,_=self.compute_PRcurve_metrics()
        return plot_PR(r,p);
    
    def plot_cl_vs_bl_PRcurve(self,y_pred,ax=None,fig=None):        
        '''
        Plot Precision-Recall curves for baseline VPR technique (using minimum distance as a threshold)
        vs the closed-loop system (which retains only good points, then applies the distance threshold)
        '''
        if ax == None:
            fig,ax=plt.subplots()
        p,r,_,_,_,_,_=self.compute_PRcurve_metrics()
        ax.plot(r,p,label='baseline');
        p,r,_,_,_,_,_=self.compute_cl_PRcurve_metrics(y_pred)
        ax.plot(r,p,label='closed-loop');
        ax.set_xlabel('recall');
        ax.set_ylabel('precision');
        ax.set_xlim([0,1]);
        ax.set_ylim([0,1.05]);
        ax.set_title('PR Curve, Tolerance={0}'.format(self.tolerance));
        ax.legend();
        return fig,ax
    
    def compute_cl_PRcurve_metrics(self,y_pred):
        '''
        Compute precision, recall and numbers of TP/FP/TN/FN for closed-loop system as the minimum distance threshold
        is swept over 2000 points
        '''
        NUM_PTS = 2000                # number of points in PR-curve
            
        bm_distances = find_best_match_distances(self.S)
        
        d_sweep = np.linspace(bm_distances.min(), bm_distances.max(), NUM_PTS)
        p=np.full_like(d_sweep,np.nan)
        r=np.full_like(d_sweep,np.nan)
        tp=np.full_like(d_sweep,np.nan)
        fp=np.full_like(d_sweep,np.nan)
        tn=np.full_like(d_sweep,np.nan)
        fn=np.full_like(d_sweep,np.nan)

        for i, threshold in enumerate(d_sweep):
            match_found = (bm_distances <= threshold)
            [p[i], r[i], tp[i], fp[i], tn[i], fn[i]] = find_vpr_performance_metrics(match_found & y_pred,self.y,self.match_exists,verbose=False)
        return p,r,tp,fp,tn,fn,d_sweep       
    
    def compute_PRcurve_metrics(self):
        '''
        Returns precision, recall and numbers of TP/FP/TN/FN for baseline VPR technique as the minimum distance
        threshold is swept through all values
        '''
        return(self.compute_cl_PRcurve_metrics(self.ALL_TRUE))
        
    def plot_runs(self):
        if self.ref.GEO_TAGGED and self.qry.GEO_TAGGED:
            plt.plot(self.ref.x,self.ref.y,label='ref');
            plt.plot(self.qry.x,self.qry.y,label='qry');
            plt.legend();
            plt.xlabel('x'); plt.ylabel('y');
            plt.title('Qry and Ref Runs');
        else:
            print('Info: plot_runs: Cannot plot as ref and query are not both GEOTAGGED')
        return
        
    def plot_ypred(self,y_pred):
        '''
        Plot the route showing TP and FP prediction locations
        '''
        if self.qry.GEO_TAGGED:
            tp,fp,tn,fn=find_each_prediction_metric(y_pred,self.y)
            plt.scatter(self.qry.x,self.qry.y,color='lightgray',marker='.');
            plt.scatter(self.qry.x[tp],self.qry.y[tp],color='g',marker='.',label='TP');
            plt.scatter(self.qry.x[fp],self.qry.y[fp],color='r',marker='x',label='FP');
            plt.legend();
            plt.xlabel('x'); plt.ylabel('y');
            plt.title('TP and FP locations along query route');
        else:
            print('Info: plot_ypred: query run is not GEOTAGGED, cannot plot route')
        return
    
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
        tp,fp,tn,fn=find_each_prediction_metric(ypred,self.y)
        fig,ax=plt.subplots(figsize=(20,4))
        qrys=np.arange(self.num_qrys)
        if self.units == 'm':
            error = self.ref_error
        elif self.units == 'frames':
            error = self.frame_error
        else:
            print('Error: robotvpr.py: plot_error_along_route: units needs to be either m or frames');
            return
        ax.scatter(qrys[fn],error[fn],marker='.',color='lightblue',label='FN (removed)')
        ax.scatter(qrys[tn],error[tn],marker='.',color='lightgray',label='TN (removed)')
        ax.scatter(qrys[tp],error[tp],marker='.',color='g',label='TP (retained)');
        ax.scatter(qrys[fp],error[fp],marker='.',color='r',label='FP (retained)');
        ax.axhline(self.tolerance,ls='dashed',label='Tolerance')
        ax.set_xlabel('query number');
        ax.set_ylabel('localisation error (m)');
        ax.set_title('Error along the query route - showing retained (predicted good), and removed points');
        ax.legend();
        return fig,ax;
    
    def __repr__(self):  # Return a string containing a printable representation of an object.
        return f"{self.__class__.__name__}(ref={self.ref.folder},qry={self.qry.folder})"
    

class RobotVPR_fromS(RobotVPR):
    
    def __init__(self,S,actual_match,description=""):
        """
        Constructs the attributes of a RobotVPR object, based on a distance matrix
        """    
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
        
        #if self.ref.GEO_TAGGED and self.qry.GEO_TAGGED:
            #self.gt_distances = cdist(self.ref.xy, self.qry.xy)
            #self.gt_match = self.gt_distances.argmin(axis=0)          # index of the closest reference for each query
        self.frame_error = abs(self.best_match-self.gt_match)     # number of frames between VPR match and actual match
            #self.abs_error = compute_distance_in_m(self.ref.xy[gt_match],self.qry.xy)              # distance in m between matching points
            #self.min_error = self.gt_distances.min(axis=0)
            #self.abs_error = self.gt_distances[self.best_match, np.arange(self.num_qrys)]
            #self.ref_error = np.diag(cdist(self.ref.xy[self.gt_match],self.ref.xy[self.best_match])) # distance in m along reference route only
        
        # TODO: rework when matches do not exist for each query
        self.match_exists = self.ALL_TRUE
        pass
    
    def __repr__(self):  # Return a string containing a printable representation of an object.
        return f"{self.__class__.__name__}(description={self.description},num_refs={self.num_refs},num_qrys={self.num_qrys})"