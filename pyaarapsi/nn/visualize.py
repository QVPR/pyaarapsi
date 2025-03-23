#! /usr/bin/env python3
'''
Visualization methods. Plotting, printing, and statistics
'''
from typing import Tuple

import numpy as np

import scipy
from matplotlib.patches import FancyArrow
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_confusion_bars(ax, bar_heights, bar_steps, bar_colors, bar_labels, ylabel, yrot,
                        ypad, loc):
    '''
    todo
    '''
    ax.bar(   x=bar_steps, color=bar_colors, height=bar_heights,
                tick_label=bar_labels)
    ax.set_ylabel(ylabel, rotation=yrot, labelpad=ypad, loc=loc)
    for x, y in zip(bar_steps, bar_heights):
        ax.text(x, y + 0.1, f"{y:6.2f}%", fontsize=8, color='b',
            verticalalignment='bottom', horizontalalignment='center', rotation=45)

def make_split_axes_y_linlog(fig, axes, lims, plotter, logplotter=None, subplot=111, size=1):
    '''
    Plot on a split linear-logarithmic y axis
    '''
    # Configure base axis as linear:
    axes.set_yscale('linear')
    axes.set_ylim((lims[0], lims[1]))
    # Generate, attach, and configure a secondary logarithmic axis:
    axes_divider = make_axes_locatable(axes)
    axeslog = axes_divider.append_axes("top", size=size, pad=0, sharex=axes)
    axeslog.set_yscale('log')
    axeslog.set_ylim((lims[1]+0.001*lims[1], lims[2]))  # add a miniscule amount to the start to
                                                        # prevent duplicated axis labels
    # Plot the data in both axes:
    plotter(axes)
    logplotter = plotter(axeslog) if logplotter is None else logplotter(axeslog)
    # Hide middle bar:
    axes.spines['top'].set_visible(False)
    axeslog.spines['bottom'].set_linestyle((0,(0.1,4)))
    axeslog.spines['bottom'].set_linewidth(2)
    axeslog.spines['bottom'].set_color('r')
    axeslog.xaxis.set_visible(False)
    # Create an invisible frame to provide overarching anchor positions for axis labels:
    axes.set_ylabel('')
    axes.set_xlabel('')
    axeslog.set_ylabel('')
    axeslog.set_xlabel('')
    axesi = fig.add_subplot(subplot, frameon=False)
    axesi.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    return axeslog, axesi

def adj_make_split_axes_y_linlog(axes, lims, plotter, size=1, swap_order: bool = False):
    '''
    Adjusted plot on a split linear-logarithmic y axis
    '''
    # Configure base axis as linear:
    axes.set_yscale('linear')
    axes.set_ylim((lims[0], lims[1]))
    # Generate, attach, and configure a secondary logarithmic axis:
    axes_divider = make_axes_locatable(axes)
    axeslog = axes_divider.append_axes("top", size=size, pad=0, sharex=axes)
    axeslog.set_yscale('log')
    axeslog.set_ylim((lims[1]+0.001*lims[1], lims[2]))  # add a miniscule amount to the start to
                                                        # prevent duplicated axis labels
    # Plot the data in both axes:
    if swap_order:
        plotter(axeslog)
        plotter(axes)
    else:
        plotter(axes)
        plotter(axeslog)
    # Hide middle bar:
    axes.spines['top'].set_visible(False)
    axeslog.spines['bottom'].set_visible(False)
    axeslog.xaxis.set_visible(False)
    # Create an invisible frame to provide overarching anchor positions for axis labels:
    axes.set_ylabel('')
    axes.set_xlabel('')
    axeslog.set_ylabel('')
    axeslog.set_xlabel('')
    axeslog.get_legend().set_visible(False)
    axes.get_legend().set_visible(False)
    return axeslog

def make_split_axes_x_linlog(fig, axes, lims, plotter):
    '''
    Plot on a split linear-logarithmic x axis
    '''
    # Configure base axis as linear:
    axes.set_xscale('linear')
    axes.set_xlim((lims[0], lims[1]))
    # Generate, attach, and configure a secondary logarithmic axis:
    axes_divider = make_axes_locatable(axes)
    axeslog = axes_divider.append_axes("right", size=1, pad=0, sharey=axes)
    axeslog.set_xscale('log')
    axeslog.set_xlim((lims[1]+0.001*lims[1], lims[2]))  # add a miniscule amount to the start to
                                                        # prevent duplicated axis labels
    # Plot the data in both axes:
    plotter(axes)
    plotter(axeslog)
    # Hide middle bar:
    axes.spines['right'].set_visible(False)
    axeslog.spines['left'].set_linestyle((0,(0.1,4)))
    axeslog.spines['left'].set_linewidth(2)
    axeslog.spines['left'].set_color('r')
    axeslog.yaxis.set_visible(False)
    # Create an invisible frame to provide overarching anchor positions for axis labels:
    axes.set_ylabel('')
    axes.set_xlabel('')
    axeslog.set_ylabel('')
    axeslog.set_xlabel('')
    axesi = fig.add_subplot(111, frameon=False)
    axesi.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    return axeslog, axesi

def get_confusion_stats(label: list, pred: list) -> Tuple[float, float, float, float]:
    '''
    return tp, tn, fp, fn
    '''
    lab     = np.array(label).astype(bool)
    pred    = np.array(pred).astype(bool)
    length  = len(pred)
    #pylint: disable=C0121
    tp      = ((pred == True ).astype(int) + (lab == True ).astype(int)) == 2
    tn      = ((pred == False).astype(int) + (lab == False).astype(int)) == 2
    fp      = ((pred == True ).astype(int) + (lab == False).astype(int)) == 2
    fn      = ((pred == False).astype(int) + (lab == True ).astype(int)) == 2
    #pylint: enable=C0121
    tp_perc = np.round(100 * (np.sum(tp) / length), 2)
    tn_perc = np.round(100 * (np.sum(tn) / length), 2)
    fp_perc = np.round(100 * (np.sum(fp) / length), 2)
    fn_perc = np.round(100 * (np.sum(fn) / length), 2)
    return tp_perc, tn_perc, fp_perc, fn_perc

def get_acc_and_confusion_stats(label: list, pred: list) \
    -> Tuple[float, float, float, float, float]:
    '''
    return accuracy, tp, tn, fp, fn
    '''
    acc = np.round(100 * np.mean(np.array(pred) == np.array(label)), 2)
    return acc, *get_confusion_stats(label=label, pred=pred)

def stat(data) -> None:
    '''
    Print mean, min, Q1, median, Q3, max
    '''
    print(np.mean(data), *np.percentile(data, [0,25,50,75,100]))

def mean_confidence_interval(data, confidence=0.95):
    '''
    https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    '''
    a       = 1.0 * np.array(data)
    n       = np.count_nonzero(~np.isnan(a))
    m, se   = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
    h       = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def make_legend_arrow(*args, **kwargs) -> FancyArrow: #pylint: disable=W0613
    '''
    Args: legend, orig_handle, xdescent, ydescent, width, height, fontsize
    '''
    return FancyArrow(4, 2, 6, 0, width=1, head_width=2.5, head_length=3,
                      length_includes_head=True, overhang=0)
