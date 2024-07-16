from typing import List, Optional, Tuple

import numpy as np
from scipy import linalg
from opt_einsum import contract

def svd_truncate(tensor: np.ndarray, cutoff: float, linds: List[int], rinds: Optional[List[int]] = None, p: Optional[int] = 2, maxdim: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    description

    Parameters
    ----------
    tensor: ndarray
        Tensor to be decomposed (and truncated) by singular value decomposition
    cutoff: float
        Truncation threshold for singular values, such that the normalized discarded weight is less than the threshold
    linds: List[int]
        List of indices of the input tensor to be combined and considered as the "row index"
    rinds: List[int], Optional
        List of indices of the input tensor to be combined and considered as the "column index". Default: the list of indices forming the complement of `linds`
    p: int, Optional
        p-norm by which the discarded weights are calculated, i.e., for s_i ordered from smallest to largest, ϵ_{discard}(D) = ( \sum_{i=1}^{D} s_i^p / (\sum_i s_i^p) )^(1/p). Default: 2
    maxdim: int, Optional
        Maximum bond dimension, Default: 5000

    Returns
    -------
    U_trunc: ndarray
        Truncated U matrix, reshaped to have the initially specified left indices
    S_trunc: ndarray
        Truncated matrix of singular values
    Vh_trunc: ndarray
        Truncated Vh matrix, reshaped to have the initially specified right indices
    '''
    shp = np.array(tensor.shape)
    if rinds == None:
        rinds = [n for n in range(0, len(shp)) if n not in linds]
    ldim = np.prod(shp[linds])
    rdim = np.prod(shp[rinds])
    dest = np.concatenate((linds, rinds))
    mat = np.reshape(np.moveaxis(tensor, np.arange(len(dest)), np.argsort(dest)), (ldim, rdim))
    U, s, Vh = linalg.svd(mat, full_matrices=False, lapack_driver='gesvd')

    s2 = np.power(s, p)
    tot_wt = np.sum(s2)
    discard_wts = np.cumsum(np.flip(s2)) / tot_wt
    #print(discard_wts)
    trunc_dim = len(s)
    if cutoff > 0.0:
        trunc_dim = np.count_nonzero(discard_wts > (cutoff ** p))
    if maxdim != None:
        trunc_dim = np.minimum(trunc_dim, maxdim)

    U_trunc = np.reshape(U[:, 0:trunc_dim], np.concatenate((shp[linds], [trunc_dim])))
    Vh_trunc = np.reshape(Vh[0:trunc_dim, :], np.concatenate(([trunc_dim], shp[rinds])))
    S_trunc = np.diag(s[0:trunc_dim])

    return U_trunc, S_trunc, Vh_trunc

def reshape_qr(tensor: np.ndarray, linds: List[int], rinds: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Computes the (thin) QR decomposition of a tensor, reshaping it first into a matrix

    Parameters
    ----------
    tensor: ndarray
        Tensor to be decomposed (and truncated) by singular value decomposition
    linds: List[int]
        List of indices of the input tensor to be combined and considered as the "row index"
    rinds: List[int], Optional
        List of indices of the input tensor to be combined and considered as the "column index". Default: the list of indices forming the complement of `linds`
    
    Returns
    -------
    res_q: ndarray
        Q matrix, reshaped to have the initially specified left indices
    res_r: ndarray
        R matrix, reshaped to have the initially specified right indices
    '''
    shp = np.array(tensor.shape)
    if rinds == None:
        rinds = [n for n in range(0, len(shp)) if n not in linds]
    ldim = np.prod(shp[linds])
    rdim = np.prod(shp[rinds])
    dest = np.concatenate((linds, rinds))
    mat = np.reshape(np.moveaxis(tensor, np.arange(len(dest)), np.argsort(dest)), (ldim, rdim))
    q, r = linalg.qr(mat, mode='economic')

    trunc_dim = q.shape[1]

    res_q = np.reshape(q, np.concatenate((shp[linds], [trunc_dim])))
    res_r = np.reshape(r, np.concatenate(([trunc_dim], shp[rinds])))

    return res_q, res_r

def reshape_rq(tensor: np.ndarray, linds: List[int], rinds: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Computes the (thin) RQ decomposition of a tensor, reshaping it first into a matrix

    Parameters
    ----------
    tensor: ndarray
        Tensor to be decomposed (and truncated) by singular value decomposition
    linds: List[int]
        List of indices of the input tensor to be combined and considered as the "row index"
    rinds: List[int], Optional
        List of indices of the input tensor to be combined and considered as the "column index". Default: the list of indices forming the complement of `linds`
    
    Returns
    -------
    res_r: ndarray
        R matrix, reshaped to have the initially specified left indices
    res_q: ndarray
        Q matrix, reshaped to have the initially specified right indices
    '''
    shp = np.array(tensor.shape)
    if rinds == None:
        rinds = [n for n in range(0, len(shp)) if n not in linds]
    ldim = np.prod(shp[linds])
    rdim = np.prod(shp[rinds])
    dest = np.concatenate((linds, rinds))
    mat = np.reshape(np.moveaxis(tensor, np.arange(len(dest)), np.argsort(dest)), (ldim, rdim))
    r, q = linalg.rq(mat, mode='economic')

    trunc_dim = r.shape[1]

    res_r = np.reshape(r, np.concatenate((shp[linds], [trunc_dim])))
    res_q = np.reshape(q, np.concatenate(([trunc_dim], shp[rinds])))

    return res_r, res_q
