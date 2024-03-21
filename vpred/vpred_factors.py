import numpy as np
from numpy.typing import NDArray
import copy
from sklearn.linear_model import LinearRegression

from typing import Optional, Union

def check_right(bool_list, index, length):
    if index + 1 < length:
        if bool_list[index+1] == 1:
            return check_right(bool_list, index+1, length)
    return index

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
    minima_bool = ((sequence + seq_l >= sequence) + (sequence + seq_r >= sequence)) == False
    flat_bool   = ((seq_l == 0) + (seq_r == 0))
    i = 0
    while i < (_len:=len(sequence)):
        if flat_bool[i] == 1:
            _left = i
            _right = check_right(flat_bool, i, _len)
            points = 0
            points_possible = 2
            if _left > 0: points += 1 if sequence[_left-1] > sequence[_left] else 0
            else: points_possible -= 1
            if _right < _len - 1: points += 1 if sequence[_right+1] > sequence[_right] else 0
            else: points_possible -= 1
            if points >= points_possible:
                minima_bool[_left] = True
            i = _right + 1
        else:
            i += 1

    minima_bool[0] = 0
    minima_bool[-1] = 0
    if check_ends:
        if sequence[1] > sequence[0]:
            minima_bool[0] = True
        if sequence[-1] < sequence[-2]:
            minima_bool[-1] = True

    minima_inds = np.arange(_len)[minima_bool]
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

def find_va_grad_fusion(S):
    _va = find_va_factor(S)
    _grad = find_grad_factor(S)
    return _va * _grad, _va + _grad

def find_adj_grad_factor(S):
    if S.ndim == 1:
        S = S[:, np.newaxis]
    qry_list = np.arange(S.shape[1])
    factors = np.zeros(qry_list[-1] + 1)
    for q in qry_list:
        Sv = S[:,q]
        m0 = Sv.min()
        m0_index = Sv.argmin()
        _range = (Sv.max() - Sv.min())
        if m0_index == 0:
            factors[q] = (Sv[1]-Sv[0]) / _range
        elif m0_index == len(Sv)-1:
            factors[q] = (Sv[-2]-Sv[-1]) / _range
        else:
            g1 = Sv[m0_index-1] / _range
            g2 = Sv[m0_index+1] / _range
            factors[q] = (g1+g2) / 2
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

def find_area_factors(S, mInd: Optional[Union[float, NDArray]] = None):
    if S.ndim == 1:
        S    = S[:, np.newaxis]
    if mInd is None:
        mInd = np.argmin(S, axis=0)
    elif not hasattr(mInd, '__iter__'):
        mInd = np.array([mInd])
    assert isinstance(mInd, np.ndarray)
    
    _shape   = S.shape
    _len     = int(np.round(np.min([_shape[0] * 0.01, 10])))
    qry_list = np.arange(_shape[1])
    factor_1 = np.zeros(qry_list[-1] + 1)
    factor_2 = np.zeros(qry_list[-1] + 1)
    factor_3 = np.zeros(qry_list[-1] + 1)
    factor_4 = np.zeros(qry_list[-1] + 1)
    
    for q in qry_list:
        dvc         = S[:,q]
        _max_ind    = len(dvc) - 1
        _start      = np.max([0,         mInd[q] - _len])
        _end        = np.min([_shape[0], mInd[q] + _len])
        _max        = np.max(dvc)
        _min        = np.min(dvc)
        factor_1[q] = np.sum(_max - dvc[_start:_end]) / np.sum(_max - dvc)
        factor_2[q] = np.sum(dvc[_start:_end]) / (len(dvc)* (_max - _min))
        factor_3[q] = np.sum(_max - dvc[np.max([mInd[q]-1, 0]):np.min([mInd[q]-1, _max_ind])]) / np.sum(_max - dvc)
        factor_4[q] = np.sum(_max - dvc[np.max([mInd[q]-1, 0]):np.min([mInd[q]-1, _max_ind])]) / (len(dvc)* (_max - _min))
    return factor_1, factor_2, factor_3, factor_4

def find_under_area_factors(S, mInd: Optional[Union[float, NDArray]] = None):
    if S.ndim == 1:
        S    = S[:, np.newaxis]
    if mInd is None:
        mInd = np.argmin(S, axis=0)
    elif not hasattr(mInd, '__iter__'):
        mInd = np.array([mInd])
    assert isinstance(mInd, np.ndarray)
    
    _shape   = S.shape
    _len     = int(np.round(np.min([_shape[0] * 0.01, 10])))
    qry_list = np.arange(_shape[1])
    factor_1 = np.zeros(qry_list[-1] + 1)
    factor_2 = np.zeros(qry_list[-1] + 1)
    factor_3 = np.zeros(qry_list[-1] + 1)
    factor_4 = np.zeros(qry_list[-1] + 1)
    
    for q in qry_list:
        dvc         = S[:,q]
        _max_ind    = len(dvc) - 1
        _start      = np.max([0,         mInd[q] - _len])
        _end        = np.min([_shape[0], mInd[q] + _len])
        _max        = np.max(dvc)
        _min        = np.min(dvc)
        factor_1[q] = np.sum(dvc[_start:_end]) / np.sum(dvc)
        factor_2[q] = np.sum(dvc[_start:_end]) / (len(dvc)* (_max - _min))
        factor_3[q] = np.sum(dvc[np.max([mInd[q]-1, 0]):np.min([mInd[q]-1, _max_ind])]) / np.sum(_max - dvc)
        factor_4[q] = np.sum(dvc[np.max([mInd[q]-1, 0]):np.min([mInd[q]-1, _max_ind])]) / (len(dvc)* (_max - _min))
    return factor_1, factor_2, factor_3, factor_4

def find_sum_factor(S):
    if S.ndim == 1:
        S    = S[:, np.newaxis]

    ref_len = S.shape[0]
    _percentiles = np.percentile(S, [0,100], axis=0)
    _ranges = (_percentiles[1] - _percentiles[0])
    _sum    = np.sum(S, axis=0)

    return _sum / (_ranges * ref_len), np.min(S, axis=0) / _sum, np.mean(S, axis=0) / _sum

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
        factor_2[q] = np.mean(dvc[lows_bool])
        
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
    raise Exception("This is broken, don't use it")
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

def find_posi_factors(rXY, mXY, init_pos=np.array([0,0])):
    raise Exception("This is broken, don't use it")
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

def find_sensitivity(S):
    if S.ndim == 1:
        S = S[:, np.newaxis]
    
    ref_len = S.shape[0]
    qry_list = np.arange(S.shape[1])
    factors1 = np.zeros(qry_list[-1] + 1)
    factors2 = np.zeros(qry_list[-1] + 1)
    factors3 = np.zeros(qry_list[-1] + 1)
    factors4 = np.zeros(qry_list[-1] + 1)
    for q in qry_list:
        Sv = S[:,q]
        _min = Sv.min()
        _min_ind = Sv.argmin()
        _left = np.max([0, _min_ind - 10])
        _right = np.min([ref_len, _min_ind + 10])
        _min_bounded = Sv[_left:_right]
        _median_over_range = np.median(_min_bounded)

        _range = (Sv.max() - Sv.min())
        factors1[q] = np.abs(_median_over_range - _min) / _range
        factors2[q] = np.sum(Sv < _median_over_range) / ref_len
        factors3[q] = np.abs(np.median(Sv) - _min) / _range
        factors4[q] = np.sum(Sv < np.median(Sv)) / ref_len
    return factors1, factors2, factors3, factors4

def find_minima_variation(S):
    if S.ndim == 1:
        S = S[:, np.newaxis]
    
    ref_len = S.shape[0]
    qry_list = np.arange(S.shape[1])
    factors1 = np.zeros(qry_list[-1] + 1)
    factors2 = np.zeros(qry_list[-1] + 1)
    for q in qry_list:
        Sv = S[:,q]
        minima_values, minima_inds = find_minima(Sv)
        factors1[q] = np.std(minima_values) / (Sv.max() - Sv.min())
        factors2[q] = np.std(minima_inds) / ref_len
    return factors1, factors2

def find_minima_separation(S):
    if S.ndim == 1:
        S = S[:, np.newaxis]
    
    ref_len = S.shape[0]
    qry_list = np.arange(S.shape[1])
    factors1 = np.zeros(qry_list[-1] + 1)
    factors2 = np.zeros(qry_list[-1] + 1)
    factors3 = np.zeros(qry_list[-1] + 1)
    factors4 = np.zeros(qry_list[-1] + 1)
    for q in qry_list:
        Sv = S[:,q]
        minima_values, minima_inds = find_minima(Sv)
        minima_values_sep, minima_inds_sep = np.abs(minima_values - Sv.min()), np.abs(minima_inds - Sv.argmin())
        _range = (Sv.max() - Sv.min())
        _mat = np.multiply(minima_values_sep / _range, minima_inds_sep / ref_len)
        _euc = np.sqrt(np.square(minima_values_sep / _range) + np.square(minima_inds_sep / ref_len))
        factors1[q] = np.sum(_mat)
        factors2[q] = np.std(_mat)
        factors3[q] = np.sum(_euc)
        factors4[q] = np.std(_euc)
    return factors1, factors2, factors3, factors4

def find_match_distance(S, mInd: Optional[Union[float, NDArray]] = None):

    if S.ndim == 1:
        S = S[:, np.newaxis]
    if mInd is None:
        mInd = np.argmin(S, axis=0)
    elif not hasattr(mInd, '__iter__'):
        mInd = np.array([mInd])
    assert isinstance(mInd, np.ndarray)
    
    _percentiles = np.percentile(S, [0,100], axis=0)
    _ranges = (_percentiles[1] - _percentiles[0])
    _arange = np.arange(S.shape[1])
    _means  = np.mean(S, axis=0)
    _medis  = np.median(S, axis=0)
        
    _match_distances = S[mInd[_arange], _arange]
    return _match_distances, _match_distances / _ranges, _match_distances / _means, _match_distances / _medis
    # ['md_plain', 'rmdist', 'md_mean', 'md_medi']

def find_relative_mean_std(S):
    if S.ndim == 1:
        S = S[:, np.newaxis]
    
    _percentiles = np.percentile(S, [0,100], axis=0)
    _ranges = (_percentiles[1] - _percentiles[0])
    _stds   = np.std(S, axis=0) / _ranges
    _means  = np.mean(S, axis=0) / _ranges

    return (_means, _stds)

def find_relative_percentiles(S):
    if S.ndim == 1:
        S = S[:, np.newaxis]
    
    _percentiles = np.percentile(S, [0,25,50,75,100], axis=0)
    _ranges = (_percentiles[4] - _percentiles[0])
    _25th  = _percentiles[1] / _ranges
    _50th  = _percentiles[2] / _ranges
    _75th  = _percentiles[3] / _ranges

    return (_25th, _50th, _75th)

def find_iqr_factors(S):
    if S.ndim == 1:
        S = S[:, np.newaxis]
    
    _percentiles = np.percentile(S, [0,25,50,75,100], axis=0)
    _ranges = (_percentiles[4] - _percentiles[0])
    _25th  = _percentiles[1]
    _50th  = _percentiles[2]
    _75th  = _percentiles[3]
    
    return (np.abs(_75th - _50th) / np.abs(_25th - _50th), np.abs(_75th-_25th) / _ranges)

def find_mean_median_diff(S):
    if S.ndim == 1:
        S = S[:, np.newaxis]
    
    _percentiles = np.percentile(S, [0,50,100], axis=0)
    _ranges = (_percentiles[2] - _percentiles[0])
    _50th  = _percentiles[1]
    _mean  = np.mean(S, axis=0)
    
    return (_mean / (_mean + _50th), (_mean - _50th) / _ranges)

def find_removed_factors(S):
    if S.ndim == 1:
        S = S[:, np.newaxis]
    qry_list = np.arange(S.shape[1])
    factors1 = np.zeros(qry_list[-1] + 1)
    factors2 = np.zeros(qry_list[-1] + 1)
    for q in qry_list:
        Sv = copy.deepcopy(S[:,q])
        _min1 = Sv.min()
        _range1 = (Sv.max() - _min1)
        Sv[Sv.argmin()] = Sv.max()
        _min2 = Sv.min()
        _range2 = (Sv.max() - _min2)
        factors1[q] = np.abs(_range2 - _range1) / _range1
        factors2[q] = np.abs(_min1 - _min2) / _range1
    return factors1, factors2

def try_pop(_list: list, item, _recurse = False):
    if isinstance(item, list) and (not _recurse):
        for i in item:
            _list = try_pop(_list, i, _recurse = True)
    else:
        try:
            _list.remove(item)
        except:
            pass
    return _list

def getFactors(__factors_in, __factors_out, _factors_required, function, _all=False, **kwargs):
    if _all or any([i in _factors_required for i in __factors_in]):
        _factors_calculated = np.array(function(**kwargs))
        _factors_calculated = _factors_calculated[np.newaxis,:] if _factors_calculated.ndim == 1 else _factors_calculated
        for _key, _fac in zip(_factors_required, _factors_calculated):
            __factors_out[_key] = _fac
        __factors_in = try_pop(__factors_in, _factors_required)
    return __factors_in, __factors_out

def find_factors(factors_in, _S, rXY: NDArray, mInd: Optional[Union[float,NDArray]] = None, cutoff=2, init_pos=np.array([0,0]), _all=False, dists=None, norm=False, return_as_dict=False):
    
    _S = _S[:, np.newaxis] if _S.ndim == 1 else _S
    seq  = (_S - np.min(_S, 0)) / (np.max(_S, 0) - np.min(_S, 0)) if norm else _S
    
    if mInd is None:
        mInd = np.argmin(seq, axis=0)
    elif not hasattr(mInd, '__iter__'):
        mInd = np.array([mInd])
        
    # if (all or "dsort" in factors_in) and (dists is None):
        # dists = idk... can't remember. Probably a euclidean distance matrix, extracting out each row matching S
    
    _factors_in = list(copy.deepcopy(factors_in))
    _factors_out = {}

    _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ["va"],                                                       find_va_factor,                 S=seq,                                                _all=_all)
    _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ["grad"],                                                     find_grad_factor,               S=seq,                                                _all=_all)
    _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ["agrad"],                                                    find_adj_grad_factor,           S=seq,                                                _all=_all)
    _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ["lgrad", "lcoef"],                                           find_linear_factors,            S=seq, rXY=rXY, mXY=rXY[mInd, :], cutoff=cutoff,      _all=_all)
    _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ['area', 'area_norm', 'area_small', 'area_small_norm'],       find_area_factors,              S=seq, mInd=mInd,                                     _all=_all)
    _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ['uarea', 'uarea_norm', 'uarea_small', 'uarea_small_norm'],   find_under_area_factors,        S=seq, mInd=mInd,                                     _all=_all)
    _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ["dlows", "mlows"],                                           find_peak_factors,              S=seq,                                                _all=_all)
    # _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ["dposi", "dvari"],                                           find_posi_factors,                     rXY=rXY, mXY=rXY[mInd, :], init_pos=init_pos,  _all=_all)
    # _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ["dsort"],                                                    find_sort_factor,               S=seq, mInd=mInd, dists=dists,                        _all=_all)
    _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ["smean", "sstd"],                                            find_relative_mean_std,         S=seq,                                                _all=_all)
    _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ["s25th", "smedi", "s75th"],                                  find_relative_percentiles,      S=seq,                                                _all=_all)
    _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ["IQRskew", "rIQR"],                                          find_iqr_factors,               S=seq,                                                _all=_all)
    _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ['md_plain', 'rmdist', 'md_mean', 'md_medi'],                 find_match_distance,            S=seq, mInd=mInd,                                     _all=_all)
    _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ["minima_tmat", "minima_vmat", "minima_teuc", "minima_veuc"], find_minima_separation,         S=seq,                                                _all=_all)
    _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ["sensrange", "senssum", "sensrange_all", "senssum_all"],     find_sensitivity,               S=seq,                                                _all=_all)
    _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ["dvcsum", "dvcminsum", "dvcmeansum"],                        find_sum_factor,                S=seq,                                                _all=_all)
    _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ["minima_valvar", "minima_indvar"],                           find_minima_variation,          S=seq,                                                _all=_all)
    _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ["minchange", "rngchange"],                                   find_removed_factors,           S=seq,                                                _all=_all)
    _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ["mmperc", "mmdiff"],                                         find_mean_median_diff,          S=seq,                                                _all=_all)
    _factors_in, _factors_out = getFactors(_factors_in, _factors_out, ["vagradmult", "vagradplus"],                                 find_va_grad_fusion,            S=seq,                                                _all=_all)
    
    if len(_factors_in) > 0:
        print("[find_factors] failed to process: " + str(_factors_in) + ", continuing ...")

    if not return_as_dict:
        return [_factors_out[i] for i in _factors_out.keys()]
    return {i: _factors_out[i] for i in _factors_out.keys()}
    