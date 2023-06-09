import numpy as np
import copy
from sklearn.linear_model import LinearRegression

#from .vpred_tools import find_nth_best_match_distances

# def find_minima(sequence):
#     # find the number of local minima in each sequence:
#     # Returns two arrays, containing: the minima values and the minima indicies
#     # Written by H.Carson 4 Aug 2021
#     # TODO: vectorise

#     # Initialisations
#     minima = []
#     minima_idx = []
#     idx = 1

#     #check the first value:
#     if sequence[1] > sequence[0]:
#         minima.append(sequence[0])
#         minima_idx.append(0)
    
#     for value in sequence[1:-1]: # start with the second value in the sequence

#         if (sequence[idx - 1] > value) & (sequence[idx + 1] > value):
#             # Conditions for minima have been met: the previous point and next point are both above the current point
#             minima.append(value)
#             minima_idx.append(idx)
#             was_increasing = True            
#         idx = idx + 1
        
#     #check the last value:
#     if sequence[idx - 1] > sequence[idx]:
#         minima.append(sequence[idx])
#         minima_idx.append(idx)

#     return (np.array(minima), np.array(minima_idx))

def find_minima(sequence: np.ndarray, check_ends=True):
    if not isinstance(sequence, np.ndarray):
        if isinstance(sequence, list):
            sequence = np.array(sequence)
        else:
            raise Exception("sequence must be either a np.ndarray or list object")
    if len(sequence) < 3:
        raise Exception("sequence is too short.")

    seq_l       = sequence.copy()
    seq_r       = sequence.copy()
    seq_l[:-1] -= sequence[1:]
    seq_r[1:]  -= sequence[:-1]
    minima_bool = ((sequence + seq_l > sequence) + (sequence + seq_r > sequence)) == False
    if check_ends:
        if sequence[1] > sequence[0]:
            minima_bool[0] = True
        if sequence[-1] < sequence[-2]:
            minima_bool[-1] = True
    minima_inds = np.arange(len(sequence))[minima_bool]
    minima_vals = sequence[minima_bool]
    
    return minima_vals, minima_inds

# def find_va_factor(S):
#     if S.ndim == 1:
#         qry_list = [0]
#     else:
#         qry_list=np.arange(S.shape[1])
#     factors=np.zeros(len(qry_list))
#     for q in qry_list:
#         if S.ndim==1:
#             Sv=S
#         else:
#             Sv=S[:,q]
#         minima_values,minima_indicies = find_minima(Sv)
#         minima_values.sort()
#         d1 = minima_values[1]-minima_values[0]
#         d2 = Sv.max()-Sv.min()
#         factors[q]=(d1/d2)
#     return factors

def find_va_factor(S):
    if S.ndim == 1:
        S = S[:, np.newaxis]
    qry_list = np.arange(S.shape[1])
    factors = np.zeros(qry_list[-1] + 1)
    for q in qry_list:
        Sv = S[:,q]
        minima_values,_ = find_minima(Sv)
        minima_values.sort()
        d1 = minima_values[1] - minima_values[0]
        d2 = Sv.max() - Sv.min()
        factors[q] = (d1/d2)
    return factors

# def find_grad_factor(S):
#     if S.ndim==1:
#         qry_list=[0]
#     else:
#         qry_list=np.arange(S.shape[1])
#     g=np.zeros(len(qry_list))
#     for q in qry_list:
#         if S.ndim==1:
#             Sv=S
#         else:
#             Sv=S[:,q]
#         m0=Sv.min()
#         m0_index=Sv.argmin()
#         if m0_index == 0:
#             g[q] = Sv[1]-Sv[0]
#         elif m0_index == len(Sv)-1:
#             g[q] = Sv[-2]-Sv[-1]
#         else:
#             g1=Sv[m0_index-1]-m0
#             g2=Sv[m0_index+1]-m0
#             g[q]=g1+g2
#     return g

def find_grad_factor(S):
    if S.ndim == 1:
        S = S[:, np.newaxis]
    qry_list = np.arange(S.shape[1])
    factors = np.zeros(qry_list[-1] + 1)
    for q in qry_list:
        Sv = S[:,q]
        m0 = Sv.min()
        m0_index = Sv.argmin()
        if m0_index == 0:
            factors[q] = Sv[1]-Sv[0]
        elif m0_index == len(Sv)-1:
            factors[q] = Sv[-2]-Sv[-1]
        else:
            g1 = Sv[m0_index-1]-m0
            g2 = Sv[m0_index+1]-m0
            factors[q] = g1+g2
    return factors

def find_all_grad_factors(s):
    g=np.zeros_like(s)
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

def find_area_factors(S, mInd):
    if S.ndim == 1:
        S    = S[:, np.newaxis]
        mInd = mInd[:, np.newaxis]
    
    _shape   = S.shape
    _len     = int(np.round(np.min([_shape[0] * 0.01, 10])))
    qry_list = np.arange(_shape[1])
    factor_1 = np.zeros(qry_list[-1] + 1)
    
    for q in qry_list:
        dvc         = S[:,q]
        _start      = np.max([0,         mInd[q] - _len])
        _end        = np.min([_shape[0], mInd[q] + _len])
        _max        = np.max(dvc)
        factor_1[q] = np.sum(_max - dvc[_start:_end]) / np.sum(_max - dvc)
    return factor_1

def find_peak_factors(S):
    if S.ndim == 1:
        S    = S[:, np.newaxis]
    
    _shape   = S.shape
    _len     = int(np.round(np.min([_shape[0] * 0.01, 10])))
    qry_list = np.arange(_shape[1])
    factor_1 = np.zeros(qry_list[-1] + 1)
    factor_2 = np.zeros(qry_list[-1] + 1)
    
    for q in qry_list:
        dvc         = S[:,q]
        dvc_l       = dvc.copy()
        dvc_r       = dvc.copy()
        dvc_l[:-1] -= dvc[1:]
        dvc_r[1:]  -= dvc[:-1]
        lows_bool   = ((dvc + dvc_l > dvc) + (dvc + dvc_r > dvc)) == False
        #lows_inds   = np.arange(len(dvc))[lows_bool]
        factor_1[q] = (np.sum(lows_bool))/ len(dvc)
        factor_2[q] = np.mean(dvc[lows_bool] ** 2)
        
    return factor_1, factor_2

def find_posi_factors(mInd, mXY, init_pos=np.array([0,0])):
    mInd = np.array([mInd])
    if mInd.ndim == 2:
        mInd = mInd[0]
    
    x_range  = np.max(mXY[:,0]) - np.min(mXY[:,0])
    y_range  = np.max(mXY[:,1]) - np.min(mXY[:,1])
    xy_range = np.array([x_range, y_range])
    
    _len     = len(mXY[:,0])
    _starts  = np.max(np.stack([np.arange(_len)-5, np.zeros(_len)],1),1).astype(int)
    _ends    = np.arange(_len).astype(int) + 1
    
    old_pos  = np.roll(mXY, 1, 0)
    old_pos[0, :] = init_pos
    
    factor_1 = np.sqrt(np.sum(((mXY - old_pos) / xy_range) ** 2, 1))
    factor_2 = np.array([np.mean(factor_1[_starts[i]:_ends[i]]) for i in np.arange(_len)])
    
    return factor_1, factor_2

def find_linear_factors(S, rXY, mXY, mInd, cutoff=10):
    if S.ndim == 1:
        S    = S[:, np.newaxis]
        mInd = np.array([mInd])
        
    qry_list = np.arange(S.shape[1])
    factor_1 = np.zeros(qry_list[-1] + 1)
    factor_2 = np.zeros(qry_list[-1] + 1)
    for q in qry_list:
        dvc         = S[:,q]

        euc         = np.sqrt(np.sum((rXY - mXY[q,:]) ** 2, 1))
        euc_argsort = euc.argsort()
        euc_sorted  = euc[euc_argsort]

        dvc_norm    = (dvc - np.min(dvc)) / (np.max(dvc) - np.min(dvc))
        dvc_sorted  = dvc_norm[euc_argsort]
        factor_inds = euc_sorted < cutoff
        regr        = LinearRegression(fit_intercept=False)
        regr_x      = euc_sorted[factor_inds]
        regr_y      = dvc_sorted[factor_inds]
        regr.fit(np.transpose(np.matrix(regr_x)), np.transpose(np.matrix(regr_y)))
        #regr_y_calc = (regr_x * regr.coef_[0][0])
        corr_coef   = np.corrcoef(regr_x, regr_y)[0,1]**2
        
        factor_1[q] = regr.coef_[0][0]
        factor_2[q] = corr_coef
    return factor_1, factor_2

def find_factors(factors_in, _S, mXY, rXY, mInd, cutoff=2, init_pos=np.array([0,0]), all=False):
    seq      = (_S - np.min(_S, 0)) / (np.max(_S, 0) - np.min(_S, 0))
    va       = None
    grad     = None
    lgrad    = None
    lcoef    = None
    area     = None
    dlows    = None
    mlows    = None
    dposi    = None
    dvari    = None
    
    if "va" in factors_in or all:
        va           = find_va_factor(seq)
    if "grad" in factors_in or all:
        grad         = find_grad_factor(seq)
    if ("lgrad" in factors_in) or ("lcoef" in factors_in) or all:
        lgrad, lcoef = find_linear_factors(seq, rXY, mXY, mInd, cutoff=cutoff)
    if "area" in factors_in or all:
        area = find_area_factors(seq, mInd) 
    if ("dlows" in factors_in) or ("mlows" in factors_in) or all:
        dlows, mlows = find_peak_factors(seq) 
    if ("dposi" in factors_in) or ("dvari" in factors_in) or all:
        dposi, dvari = find_posi_factors(mInd, mXY, init_pos=init_pos)
    factors_ = {"va": va,       "grad": grad,   "lgrad": lgrad, "lcoef": lcoef, \
                "area": area,                   "dlows": dlows, "mlows": mlows, \
                "dposi": dposi, "dvari": dvari}
    if all:
        return factors_
    return factors_[factors_in[0]], factors_[factors_in[1]]