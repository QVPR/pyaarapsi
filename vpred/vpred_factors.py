import numpy as np
#from .vpred_tools import find_nth_best_match_distances

def find_minima(sequence):
    # find the number of local minima in each sequence:
    # Returns two arrays, containing: the minima values and the minima indicies
    # Written by H.Carson 4 Aug 2021
    # TODO: vectorise

    # Initialisations
    minima = []
    minima_idx = []
    idx = 1

    #check the first value:
    if sequence[1] > sequence[0]:
        minima.append(sequence[0])
        minima_idx.append(0)
    
    for value in sequence[1:-1]: # start with the second value in the sequence

        if (sequence[idx - 1] > value) & (sequence[idx + 1] > value):
            # Conditions for minima have been met: the previous point and next point are both above the current point
            minima.append(value)
            minima_idx.append(idx)
            was_increasing = True            
        idx = idx + 1
        
    #check the last value:
    if sequence[idx - 1] > sequence[idx]:
        minima.append(sequence[idx])
        minima_idx.append(idx)

    return (np.array(minima), np.array(minima_idx))

def find_va_factor(S):
    if S.ndim == 1:
        qry_list = [0]
    else:
        qry_list=np.arange(S.shape[1])
    factors=np.zeros(len(qry_list))
    for q in qry_list:
        if S.ndim==1:
            Sv=S
        else:
            Sv=S[:,q]
        minima_values,minima_indicies = find_minima(Sv)
        minima_values.sort()
        d1 = minima_values[1]-minima_values[0]
        d2 = Sv.max()-Sv.min()
        factors[q]=(d1/d2)
    return factors

def find_grad_factor(S):
    if S.ndim==1:
        qry_list=[0]
    else:
        qry_list=np.arange(S.shape[1])
    g=np.zeros(len(qry_list))
    for q in qry_list:
        if S.ndim==1:
            Sv=S
        else:
            Sv=S[:,q]
        m0=Sv.min()
        m0_index=Sv.argmin()
        if m0_index == 0:
            g[q] = Sv[1]-Sv[0]
        elif m0_index == len(Sv)-1:
            g[q] = Sv[-2]-Sv[-1]
        else:
            g1=Sv[m0_index-1]-m0
            g2=Sv[m0_index+1]-m0
            g[q]=g1+g2
    return g

def find_all_grad_factors(s):
    g=np.zeros_like(s);
    for i in np.arange(len(s)):
        if i == 0:
            g[0]=s[1]-s[0]
        elif i < (len(s)-1):
            grad_before=s[i-1]-s[i]
            grad_after =s[i+1]-s[i]
            temp_g=(grad_before + grad_after)
            g[i]=temp_g
        elif i == (len(s)-1):
            temp_g=(s[i-1]-s[i])
            g[i]=temp_g
    return g