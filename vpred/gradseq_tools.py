import numpy as np
from scipy.signal import convolve2d
from .vpred_tools import extract_similarity_vector, find_best_match
from .vpred_factors import find_all_grad_factors

def corner_mean_pad(A):
    Ap=np.full((A.shape[0]+2,A.shape[1]+2),A.mean(),dtype='float')
    Ap[2:,2:]=A
    return Ap
    
def mean_pad(A):
    Ap=np.full((A.shape[0]+2,A.shape[1]+2),A.mean(),dtype='float')
    Ap[1:-1,1:-1]=A
    return Ap

def create_gradient_matrix(S,kernel=np.eye(3)):
    G=np.zeros_like(S)
    num_qrys=S.shape[1]
    for q in range(num_qrys):
        s=extract_similarity_vector(S,q)
        G[:,q]=find_all_grad_factors(s)
    Gseq = convolve2D(corner_mean_pad(G), kernel);
    return Gseq

def convolve2D(image, kernel, padding=0, strides=1):
    # TODO: replace entirely with scipy.signal.convolve2d once backwards compatibility is not required
    return convolve2d(image,kernel,boundary='fill',mode='valid',fillvalue=padding);

def find_best_match_G(G):
    return G.argmax(axis=0)

def find_best_match_distances_G(G):
    return G.max(axis=0)

def find_best_match_S(S):
    return find_best_match(S)

def find_consensus(S,G,tolerance=1):
    return abs(find_best_match_G(G) - find_best_match_S(S))<=tolerance

def find_concensus(S,G,tolerance=1):
    # Retained for backwards compatibility to deal with typo
    return find_consensus(S,G,tolerance=1)

def create_sequence_matrix(S,l):
    kernel=np.eye(l)
    return convolve2D(S,kernel,padding=0)/l

def adjust_seq_matches(actual_match,l):
    # TODO: May need to adjust the reference indexes first...
    return actual_match[(l-1):]-(l-1)
