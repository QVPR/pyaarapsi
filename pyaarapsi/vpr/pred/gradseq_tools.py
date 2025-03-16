#!/usr/bin/env python3
'''
Tools for gradient sequence integrity monitor
'''
import numpy as np
import scipy.signal
from pyaarapsi.vpr.pred.vpred_tools import extract_similarity_vector, find_best_match
from pyaarapsi.vpr.pred.vpred_factors import find_all_grad_factors

def corner_mean_pad(a):
    '''
    Pad array left and top
    '''
    a_padded=np.full((a.shape[0]+2, a.shape[1]+2), a.mean(), dtype='float')
    a_padded[2:,2:]=a
    return a_padded

def mean_pad(a):
    '''
    Pad array all sides
    '''
    a_padded=np.full((a.shape[0]+2, a.shape[1]+2), a.mean(), dtype='float')
    a_padded[1:-1,1:-1]=a
    return a_padded

def create_gradient_matrix(s_in, kernel = np.eye(3)):
    '''
    Create gradient matrix from similarity matrix
    '''
    grad = np.zeros_like(s_in)
    num_qrys=s_in.shape[1]
    for q in range(num_qrys):
        s = extract_similarity_vector(s_in, q)
        grad[:,q] = find_all_grad_factors(s)
    grad_seq = convolve2d(corner_mean_pad(grad), kernel)
    return grad_seq

def convolve2d(image, kernel, padding=0):
    '''
    Helper for scipy's convolve2d
    TODO: replace entirely with scipy.signal.convolve2d once backwards compatibility is
        not required.
    '''
    return scipy.signal.convolve2d(image, kernel, boundary='fill', mode='valid', fillvalue=padding)

def find_best_match_grad(grad):
    '''
    argmaxes of grad
    '''
    return grad.argmax(axis=0)

def find_best_match_distances_grad(grad):
    '''
    maxes of grad
    '''
    return grad.max(axis=0)

def find_consensus(s_in, grad, tolerance=1):
    '''
    grad <-> sim matrix consensus
    '''
    return abs(find_best_match_grad(grad) - find_best_match(s_in)) <= tolerance

def create_sequence_matrix(s_in, l):
    '''
    Generate a sequence matrix from a similarity matrix
    '''
    kernel=np.eye(l)
    return convolve2d(s_in, kernel,padding=0)/l

def adjust_seq_matches(actual_match,l):
    '''
    Adjust sequence matches
    TODO: May need to adjust the reference indexes first...
    '''
    return actual_match[(l-1):]-(l-1)
