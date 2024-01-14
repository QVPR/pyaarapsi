import numpy as np
import copy
from sklearn.linear_model import LinearRegression

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
    if not hasattr(mInd, '__iter__'):
        mInd = np.array([mInd])
    
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

def find_linear_factors(S, rXY, mXY, cutoff=10):
    if S.ndim == 1:
        S    = S[:, np.newaxis]
    if mXY.ndim == 1:
        mXY  = mXY[np.newaxis, :]
        
    qry_list = np.arange(S.shape[1])
    factor_1 = np.zeros(qry_list[-1] + 1)
    factor_2 = np.zeros(qry_list[-1] + 1)
    for q in qry_list:
        dvc         = S[:,q]

        assert len(dvc) == rXY.shape[0], "distance vector does not align with x,y array [%d versus %d]." % (len(dvc), rXY.shape[0])

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

def find_sort_factor(S, mInd, dists):
    if S.ndim == 1:
        S       = S[:, np.newaxis]
    qry_list    = np.arange(S.shape[1])
    factor_1    = np.zeros(qry_list[-1] + 1)
    for i,q in zip(mInd, qry_list):
        dvc         = S[:,q]
        _sorted     = np.argsort(dvc)
        _range      = np.argsort(dists[i])
        _diff       = np.sum(np.abs(_sorted-_range))
        _normed     = _diff / np.sum(_range)
        factor_1[q] = _normed
    return factor_1

def find_stat_factors(S):
    if S.ndim == 1:
        S       = S[:, np.newaxis]
    qry_list    = np.arange(S.shape[1])
    factor_1    = np.zeros(qry_list[-1] + 1)
    factor_2    = np.zeros(qry_list[-1] + 1)
    for q in qry_list:
        dvc         = S[:,q]
        _range      = np.max(dvc) - np.min(dvc)
        factor_1[q] = np.mean(dvc) / _range
        factor_2[q] = np.std(dvc) / _range
    return factor_1, factor_2

def find_posi_factors(rXY, mXY, init_pos=np.array([0,0])):
    # rXY: reference set ground truth xy array
    # mXY: history of matched points, in chronological order
    if mXY.ndim == 1:
        mXY  = mXY[np.newaxis, :]
    
    x_range  = np.max(rXY[:,0]) - np.min(rXY[:,0])
    y_range  = np.max(rXY[:,1]) - np.min(rXY[:,1])
    xy_range = np.array([x_range, y_range])
    
    _len     = len(mXY[:,0])
    _starts  = np.max(np.stack([np.arange(_len)-5, np.zeros(_len)],1),1).astype(int)
    _ends    = np.arange(_len).astype(int) + 1
    
    mXY_roll = np.roll(mXY, 1, 0) # roll mXY to align previous match to current match
    mXY_roll[0, :] = init_pos # overwrite the first row with an initial position
    
    factor_1 = np.sqrt(np.sum(((mXY - mXY_roll) / xy_range) ** 2, 1))
    factor_2 = np.array([np.mean(factor_1[_starts[i]:_ends[i]]) for i in np.arange(_len)])
    
    return factor_1, factor_2


def find_factors(factors_in, _S, rXY=None, mInd=None, cutoff=2, init_pos=np.array([0,0]), all=False, dists=None, norm=False):
    if norm:
        seq  = (_S - np.min(_S, 0)) / (np.max(_S, 0) - np.min(_S, 0))
    else:
        seq  = _S
    # Initialise:        
    va, grad, lgrad, lcoef, area, dlows, mlows, dposi, dvari, dsort, smean, sstd = (None,)*12
    if "va" in factors_in or all:
        va                          = find_va_factor(seq)
    if "grad" in factors_in or all:
        grad                        = find_grad_factor(seq)
    if ("lgrad" in factors_in) or ("lcoef" in factors_in) or all:
        assert not rXY is None
        assert not mInd is None
        lgrad, lcoef                = find_linear_factors(seq, rXY, rXY[mInd, :], cutoff=cutoff)
    if "area" in factors_in or all:
        assert not mInd is None
        area                        = find_area_factors(seq, mInd) 
    if ("dlows" in factors_in) or ("mlows" in factors_in) or all:
        dlows, mlows                = find_peak_factors(seq) 
    if ("dposi" in factors_in) or ("dvari" in factors_in) or all:
        assert not rXY is None
        assert not mInd is None
        _dposi, _dvari                = find_posi_factors(rXY, rXY[mInd, :], init_pos=init_pos)
        if _S.ndim == 1:
            dposi, dvari = _dposi[-2:-1], _dvari[-2:-1]
        else:
            _len =_S.shape[1]
            assert _dposi.shape[0] >= _len
            dposi, dvari = _dposi[-_len:], _dvari[-_len:]
    if ("dsort" in factors_in) or all:
        assert not dists is None
        assert not mInd is None
        dsort                       = find_sort_factor(seq, mInd, dists)
    if ("smean" in factors_in) or ("sstd" in factors_in) or all:
        smean, sstd                 = find_stat_factors(seq)
    factors_ = {"va": va,       "grad": grad,   "lgrad": lgrad, "lcoef": lcoef, \
                "area": area,                   "dlows": dlows, "mlows": mlows, \
                "dposi": dposi, "dvari": dvari, \
                "dsort": dsort,                 "smean": smean, "sstd": sstd}
    if all:
        return factors_
    return [factors_[i] for i in factors_in]