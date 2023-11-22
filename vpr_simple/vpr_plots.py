import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from scipy.spatial.distance import cdist
from pyaarapsi.core.helper_tools import uint8_list_to_np_ndarray
import warnings

#   _____       _____  _       _   
#  |  __ \     |  __ \| |     | |  
#  | |__) |   _| |__) | | ___ | |_ 
#  |  ___/ | | |  ___/| |/ _ \| __|
#  | |   | |_| | |    | | (_) | |_ 
#  |_|    \__, |_|    |_|\___/ \__|
#          __/ |                   
#         |___/                    

##################################################################
#### Sliding Similarity Matrix Figure: do and update

def doMtrxFig(axes, odom_in):
    plt.sca(axes)
    mtrx_image = np.zeros((len(odom_in['position']['x']), len(odom_in['position']['x'])))
    mtrx_handle = axes.imshow(mtrx_image)
    axes.set(xlabel='Query Frame', ylabel='Reference Frame')

    return {'img': mtrx_image, 'handle': mtrx_handle}

def updateMtrxFig(mInd, tInd, dvc, odom_in, fig_handles):
    img_new = np.delete(fig_handles['img'], 0, 1) # delete first column (oldest query)
    fig_handles['img'] = np.concatenate((img_new, np.array(dvc)), 1)
    fig_handles['handle'].set_data(fig_handles['img']) # TODO: access directly.
    fig_handles['handle'].autoscale() # https://stackoverflow.com/questions/10970492/matplotlib-no-effect-of-set-data-in-imshow-for-the-plot

##################################################################
#### Distance Vector Figure: do and update

def doDVecFig(axes, odom_in):
# Set up distance vector figure
# https://www.geeksforgeeks.org/data-visualization-using-matplotlib/
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html

    plt.sca(axes) # distance vector
    dist_vector = plt.plot([], [], 'k-')[0] # distance vector
    lowest_dist = plt.plot([], [], 'ro', markersize=7)[0] # matched image (lowest distance)
    actual_dist = plt.plot([], [], 'mo', markersize=7)[0] # true image (correct match)
    axes.set(xlabel='Index', ylabel='Distance')
    axes.legend(["Image Distances", "Selected", "True"])
    axes.set_xlim(0, len(odom_in['position']['x']))
    axes.set_ylim(0, 1.2)

    return {'axes': axes, 'dis': dist_vector, 'low': lowest_dist, 'act': actual_dist}

def updateDVecFig(mInd, tInd, dvc, odom_in, fig_handles):
# Update DVec figure with new data (match->mInd, true->tInd)
# update (overwrite) visualisation with new data:

    # overwrite with new distance vector / image distance:
    max_val = np.max(dvc[:])
    fig_handles['dis'].set_xdata(range(len(dvc)))
    fig_handles['dis'].set_ydata(dvc/max_val)
    # overwrite with new lowest match:
    fig_handles['low'].set_xdata(mInd)
    fig_handles['low'].set_ydata(dvc[mInd]/max_val)
    # overwrite with new truth value:
    fig_handles['act'].set_xdata(tInd)
    fig_handles['act'].set_ydata(dvc[tInd]/max_val)

##################################################################
#### Odometry Figure: do and update

def doOdomFig(axes, odom_in):
# Set up odometry figure

    plt.sca(axes)
    ref_plotted = plt.plot(odom_in['position']['x'], odom_in['position']['y'], 'b-')[0]
    mat_plotted = plt.plot([], [], 'r+', markersize=6)[0] # Match values: init as empty
    tru_plotted = plt.plot([], [], 'gx', markersize=4)[0] # True value: init as empty

    axes.set(xlabel='X-Axis', ylabel='Y-Axis')
    axes.legend(["Reference", "Match", "True"])
    axes.set_aspect('equal')

    return {'axes': axes, 'ref': ref_plotted, 'mat': mat_plotted, 'tru': tru_plotted}

def updateOdomFig(mInd, tInd, dvc, odom_in, fig_handles):
# Update odometryfigure with new data (match->mInd, true->tInd)
# Use old handles (reference, match, true)
    # Only display last 'queries_keep' number of points
    num_queries = len(list(fig_handles['tru'].get_xdata()))
    queries_keep = 10
    start_ind = num_queries - queries_keep + 1
    if num_queries < queries_keep:
        start_ind = 0
        
    ## odometry plot:
    # Append new value for "match" (what it matched the image to)
    fig_handles['mat'].set_xdata(np.append(fig_handles['mat'].get_xdata()[start_ind:num_queries], odom_in['position']['x'][mInd]))
    fig_handles['mat'].set_ydata(np.append(fig_handles['mat'].get_ydata()[start_ind:num_queries], odom_in['position']['y'][mInd]))
    # Append new value for "true" (what it should be from the robot odom)
    fig_handles['tru'].set_xdata(np.append(fig_handles['tru'].get_xdata()[start_ind:num_queries], odom_in['position']['x'][tInd]))
    fig_handles['tru'].set_ydata(np.append(fig_handles['tru'].get_ydata()[start_ind:num_queries], odom_in['position']['y'][tInd]))

#   ____        _        _     
#  |  _ \      | |      | |    
#  | |_) | ___ | | _____| |__  
#  |  _ < / _ \| |/ / _ \ '_ \ 
#  | |_) | (_) |   <  __/ | | |
#  |____/ \___/|_|\_\___|_| |_|
    
def disable_toolbar(fig, interact=False):
    # hide toolbar and disable plot interaction
    fig.toolbar_location = None
    if not interact:
        fig.toolbar.active_drag = None
        fig.toolbar.active_scroll = None
        fig.toolbar.active_tap = None
    return fig

##################################################################
#### Contour Figure: do and update

def doCntrFigBokeh():
# Set up contour figure

    _fig            = figure(title="SVM Contour", width=500, height=500, \
                                x_axis_label = 'VA Factor', y_axis_label = 'Grad Factor', \
                                x_range = (0, 1), y_range = (0, 1))

    _fig            = disable_toolbar(_fig)

    img_rand        = np.array(np.ones((1000,1000,4))*255, dtype=np.uint8)
    img_uint32      = img_rand.view(dtype=np.uint32).reshape(img_rand.shape[:-1])
    img_ds          = ColumnDataSource(data=dict(image=[img_uint32], x=[0], y=[0], dw=[10], dh=[10])) #CDS must contain columns, hence []
    img_plotted     = _fig.image_rgba(image='image', x='x', y='y', dw='dw', dh='dh', source=img_ds)
    
    # Generate legend entries:
    _fig.circle(x=[-100], y=[-100], fill_color="green",  size=8, alpha=1, legend_label="True Positive")
    _fig.circle(x=[-100], y=[-100], fill_color="red",    size=8, alpha=1, legend_label="True Negative")
    _fig.circle(x=[-100], y=[-100], fill_color="blue",   size=8, alpha=1, legend_label="False Positive")
    _fig.circle(x=[-100], y=[-100], fill_color="orange", size=8, alpha=1, legend_label="False Negative")
    
    data_plotted = _fig.circle(x=[], y=[], fill_color=[],  size=8, alpha=0.4)

    _fig.legend.orientation='horizontal'
    _fig.legend.border_line_alpha=0
    _fig.legend.background_fill_alpha=0

    return {'fig': _fig, 'img': img_plotted, 'data': data_plotted, 'xlims': [0,10], 'ylims': [0,10], 'fttype': ''}

def updateCntrFigBokeh(doc_frame, svm_field_msg, state, update_contour):

    xlims = [svm_field_msg.data.x_min, svm_field_msg.data.x_max, svm_field_msg.data.x_max - svm_field_msg.data.x_min]
    ylims = [svm_field_msg.data.y_min, svm_field_msg.data.y_max, svm_field_msg.data.y_max - svm_field_msg.data.y_min]

    colours = ['red', 'orange', 'blue', 'green']
    color = colours[2*int(state.svm_class) + int(state.gt_class)]
    
    to_stream = dict(x=[state.svm_factors[0]], y=[state.svm_factors[1]], fill_color=[color])
    doc_frame.fig_cntr_handles['data'].data_source.stream(to_stream, rollover = 100)

    if not update_contour:
        return False
    
    cv_msg_img = uint8_list_to_np_ndarray(svm_field_msg.image)

    # process image from three layer (rgb) into four layer (rgba) uint8:
    img_rgba = np.array(np.dstack((np.flipud(np.flip(cv_msg_img,2)), np.ones((1000,1000))*255)), dtype=np.uint8)
    # collapse into uint32:
    img_uint32 = img_rgba.view(dtype=np.uint32).reshape(img_rgba.shape[:-1])

    doc_frame.fig_cntr_handles['img'].data_source.data = dict(x=[xlims[0]], y=[ylims[0]], dw=[xlims[2]], \
                                                         dh=[ylims[2]], image=[img_uint32.copy()])
    
    # clear old data:
    doc_frame.fig_cntr_handles['data'].data_source.data = dict(x=[], y=[], fill_color=[])

    doc_frame.fig_cntr_handles['fig'].title.text         = svm_field_msg.data.title
    doc_frame.fig_cntr_handles['fig'].xaxis.axis_label   = svm_field_msg.data.xlab
    doc_frame.fig_cntr_handles['fig'].yaxis.axis_label   = svm_field_msg.data.ylab
    doc_frame.fig_cntr_handles['fig'].x_range.update(start=xlims[0], end=xlims[2], bounds=(xlims[0], xlims[2]))
    doc_frame.fig_cntr_handles['fig'].y_range.update(start=ylims[0], end=ylims[2], bounds=(ylims[0], ylims[2]))

    return True

##################################################################
#### Linear & Angular Vector Figure: do and update

def doXYWVFigBokeh(num_points):
# Set up distance vector figure

    _fig        = figure(title="Linear & Angular Vectors", width=500, height=250, \
                            x_axis_label = 'Index', y_axis_label = 'Value', \
                            x_range = (0, num_points))#, y_range = (0, 1.2))
    _fig        = disable_toolbar(_fig)
    rw_plotted  = _fig.circle(x=[], y=[], color="black", size=3, legend_label="Yaw Vector") # distance vector

    _fig.legend.location=(0, 140)
    _fig.legend.orientation='horizontal'
    _fig.legend.border_line_alpha=0
    _fig.legend.background_fill_alpha=0

    return {'fig': _fig, 'rw': rw_plotted, 'rwc': 1}

def updateXYWVFigBokeh(doc_frame, mInd, px, py, pw):
# Update DVec figure with new data (match->mInd, true->tInd)
# Use old handles (mat, tru) and crunched distance vector (dvc)
    new_yaw_data = dict(x=[doc_frame.fig_xywv_handles['rwc']], y=[np.round(pw[mInd],3)])
    doc_frame.fig_xywv_handles['rwc'] = (doc_frame.fig_xywv_handles['rwc'] + 1) % doc_frame.num_points
    doc_frame.fig_xywv_handles['rw'].data_source.stream(new_yaw_data, rollover=doc_frame.num_points)

##################################################################
#### Distance Vector Figure: do and update

def doDVecFigBokeh(num_points):
# Set up distance vector figure

    _fig        = figure(title="Distance Vector", width=500, height=250, \
                            x_axis_label = 'Index', y_axis_label = 'Distance', \
                            x_range = (0, num_points), y_range = (0, 1.2))
    _fig        = disable_toolbar(_fig)
    spd_plotted = _fig.line([],   [], color="orange",           legend_label="Spatial Separation") # Distance from match
    dvc_plotted = _fig.line([],   [], color="black",            legend_label="Distance Vector") # distance vector
    mat_plotted = _fig.circle([], [], color="red",     size=7,  legend_label="Selected") # matched image (lowest distance)
    tru_plotted = _fig.circle([], [], color="magenta", size=7,  legend_label="True") # true image (correct match)

    _fig.legend.location=(0, 140)
    _fig.legend.orientation='horizontal'
    _fig.legend.border_line_alpha=0
    _fig.legend.background_fill_alpha=0

    return {'fig': _fig, 'spd': spd_plotted, 'dvc': dvc_plotted, 'mat': mat_plotted, 'tru': tru_plotted}

def updateDVecFigBokeh(doc_frame, mInd, tInd, dvc, px, py):
# Update DVec figure with new data (match->mInd, true->tInd)
# Use old handles (mat, tru) and crunched distance vector (dvc)
    spd = cdist(np.transpose(np.matrix([px,py])), \
        np.matrix([px[mInd], py[mInd]]))
    spd_max = np.max(spd[:])
    dvc_max = np.max(dvc[:])
    doc_frame.fig_dvec_handles['spd'].data_source.data = {'x': list(range(len(spd-1))), 'y': spd/spd_max}
    doc_frame.fig_dvec_handles['dvc'].data_source.data = {'x': list(range(len(dvc-1))), 'y': dvc/dvc_max}
    doc_frame.fig_dvec_handles['mat'].data_source.data = {'x': [mInd], 'y': dvc[mInd]/dvc_max}
    doc_frame.fig_dvec_handles['tru'].data_source.data = {'x': [tInd], 'y': dvc[tInd]/dvc_max}

##################################################################
#### Filtered Distance Vector Figure: do and update

def doFDVCFigBokeh(num_points):
# Set up distance vector figure

    _fig        = figure(title="Filtered Distance Vector", width=500, height=250, \
                            x_axis_label = 'Index', y_axis_label = 'Distance', \
                            x_range = (0, num_points), y_range = (0, 1.2))
    _fig        = disable_toolbar(_fig)
    dvc_plotted = _fig.line([],   [], color="black",            legend_label="Filtered Distance Vector") # distance vector
    mat_plotted = _fig.circle([], [], color="red",     size=7,  legend_label="Selected") # matched image (lowest distance)
    tru_plotted = _fig.circle([], [], color="magenta", size=7,  legend_label="True") # true image (correct match)

    _fig.legend.location=(0, 140)
    _fig.legend.orientation='horizontal'
    _fig.legend.border_line_alpha=0
    _fig.legend.background_fill_alpha=0

    return {'fig': _fig, 'dvc': dvc_plotted, 'mat': mat_plotted, 'tru': tru_plotted}

def updateFDVCFigBokeh(doc_frame, mInd, tInd, dvc, px, py):
# Update DVec figure with new data (match->mInd, true->tInd)
# Use old handles (mat, tru) and crunched distance vector (dvc)
    spd = cdist(np.transpose(np.matrix([px,py])), \
        np.matrix([px[mInd], py[mInd]]))
    spd_max = np.max(spd[:])
    dvc_max = np.max(dvc[:])
    spd_norm = np.array(spd).flatten()/spd_max
    dvc_norm = np.array(dvc).flatten()/dvc_max
    spd_x_dvc = (0.5*spd_norm**2 + 0.5*dvc_norm)
    doc_frame.fig_fdvc_handles['dvc'].data_source.data = {'x': list(range(len(dvc-1))), 'y': spd_x_dvc}
    doc_frame.fig_fdvc_handles['mat'].data_source.data = {'x': [mInd], 'y': [spd_x_dvc[mInd]]}
    doc_frame.fig_fdvc_handles['tru'].data_source.data = {'x': [tInd], 'y': [spd_x_dvc[tInd]]}


##################################################################
#### Odometry Figure: do and update

def doOdomFigBokeh(px, py):
# Set up odometry figure

    xlims = (np.min(px), np.max(px))
    ylims = (np.min(py), np.max(py * 1.1))
    xrang = xlims[1] - xlims[0]
    yrang = ylims[1] - ylims[0]

    _fig        = figure(title="Odometries", width=500, height=250, \
                            x_axis_label = 'X-Axis', y_axis_label = 'Y-Axis', \
                            x_range = (xlims[0] - 0.1 * xrang, xlims[1] + 0.1 * xrang), \
                            y_range = (ylims[0] - 0.1 * yrang, ylims[1] + 0.1 * yrang), \
                            match_aspect = True, aspect_ratio = "auto")
    _fig        = disable_toolbar(_fig)

    # Make legend glyphs
    _fig.line(x=[xlims[1]*2], y=[ylims[1]*2], color="blue", line_dash='dotted', legend_label="Path")
    _fig.cross(x=[xlims[1]*2], y=[ylims[1]*2], color="red", legend_label="Match", size=14)
    _fig.plus( x=[xlims[1]*2], y=[ylims[1]*2], color="magenta", legend_label="True", size=4)
    
    ref_plotted = _fig.line(   x=px,  y=py, color="blue", alpha=0.5, line_dash='dotted')
    var_plotted = _fig.circle( x=[],  y=[], color="grey",    size=[], alpha=0.1)
    seg_plotted = _fig.segment(x0=[], y0=[], x1=[], y1=[], line_color="black", line_width=1, alpha=[])
    mat_plotted = _fig.cross(  x=[],  y=[], color="red",     size=12, alpha=[])
    tru_plotted = _fig.plus(   x=[],  y=[], color="magenta", size=8, alpha=1.0)

    _fig.legend.location= (120, 145)
    _fig.legend.orientation='horizontal'
    _fig.legend.border_line_alpha=0
    _fig.legend.background_fill_alpha=0

    return {'fig': _fig, 'ref': ref_plotted, 'var': var_plotted, 'seg': seg_plotted, 'mat': mat_plotted, 'tru': tru_plotted}

def updateOdomFigBokeh(doc_frame, mInd, tInd, px, py):
# Update odometryfigure with new data (match->mInd, true->tInd)
# Use old handles (reference, match, true)
    # Stream/append new value for "match" (estimate) and "true" (correct) odometry:
    separation  = float(np.min([abs(mInd-tInd), (-abs(mInd-tInd) % doc_frame.num_points)],0) / (doc_frame.num_points / 2))

    new_alpha   = np.round(0.9 * separation,3)
    new_size    = np.round(4.0 * np.sqrt(20.0 * separation),3)

    new_tru_data = dict(x=[np.round(px[tInd],3)], y=[np.round(py[tInd],3)])
    new_var_data = dict(**new_tru_data, size=[new_size])

    new_mat_data = dict(x=[np.round(px[mInd],3)], y=[np.round(py[mInd],3)], \
                        fill_alpha=[new_alpha], hatch_alpha=[new_alpha], line_alpha=[new_alpha])
    new_mod_data = dict(x0=new_mat_data['x'], y0=new_mat_data['y'], x1=new_tru_data['x'], y1=new_tru_data['y'], \
                        line_alpha=[0.05])
    
    with warnings.catch_warnings():
        # Bokeh gets upset because we have discrete data that we are streaming that is duplicate
        warnings.simplefilter("ignore")
        
        doc_frame.fig_odom_handles['tru'].data_source.stream(new_tru_data, rollover=1)
        doc_frame.fig_odom_handles['var'].data_source.stream(new_var_data, rollover=2*doc_frame.num_points)

        doc_frame.fig_odom_handles['seg'].data_source.stream(new_mod_data, rollover=doc_frame.num_points)
        doc_frame.fig_odom_handles['mat'].data_source.stream(new_mat_data, rollover=doc_frame.num_points)

##################################################################
#### SVM Metrics Figure: do and update

def doSVMMFigBokeh(num_points):
# Set up SVM Metrics Figure

    _fig    = figure(title="SVM Metrics", width=500, height=250, \
                            x_axis_label = 'Index', y_axis_label = 'Value', \
                            x_range = (0, num_points))
    #_fig    = disable_toolbar(_fig)
    f1_plotted  = _fig.circle(x=[], y=[], color="blue",  size=3, legend_label="Factor 1")
    f2_plotted  = _fig.circle(x=[], y=[], color="red",   size=3, legend_label="Factor 2")
    pr_plotted  = _fig.circle(x=[], y=[], color="green", size=3, legend_label="Probability")
    zv_plotted  = _fig.circle(x=[], y=[], color="black", size=3, legend_label="Z-Value")

    _fig.legend.location=(0, 140)
    _fig.legend.orientation='horizontal'
    _fig.legend.border_line_alpha=0
    _fig.legend.background_fill_alpha=0

    return {'fig': _fig, 'c': 1, 'f1': f1_plotted, 'f2': f2_plotted, 'pr': pr_plotted, 'zv': zv_plotted}

def updateSVMMFigBokeh(doc_frame, state):
# Update SVM Metrics Figure
    new_f1 = dict(x=[doc_frame.fig_svmm_handles['c']], y=[np.round(state.factors[0], 3)])
    new_f2 = dict(x=[doc_frame.fig_svmm_handles['c']], y=[np.round(state.factors[1], 3)])
    new_pr = dict(x=[doc_frame.fig_svmm_handles['c']], y=[np.round(state.prob,       3)])
    new_zv = dict(x=[doc_frame.fig_svmm_handles['c']], y=[np.round(state.mState,     3)])
    doc_frame.fig_svmm_handles['c'] = (doc_frame.fig_svmm_handles['c'] + 1) % doc_frame.num_points

    doc_frame.fig_svmm_handles['f1'].data_source.stream(new_f1, rollover=doc_frame.num_points)
    doc_frame.fig_svmm_handles['f2'].data_source.stream(new_f2, rollover=doc_frame.num_points)
    doc_frame.fig_svmm_handles['pr'].data_source.stream(new_pr, rollover=doc_frame.num_points)
    doc_frame.fig_svmm_handles['zv'].data_source.stream(new_zv, rollover=doc_frame.num_points)
