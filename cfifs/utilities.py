from typing import Any, List, Callable, Optional, Text, Tuple, Union

import numpy as np
from scipy import linalg
from opt_einsum import contract

def get_bond_dims(mps):
    n = len(mps)
    bds = np.zeros(n-1, dtype=int)
    bds[0] = (mps[0].shape)[1]
    for i in range(1, n-1):
        bds[i] = (mps[i].shape)[2]
    return bds

def svd_truncate(tensor: np.ndarray, cutoff: float, linds: List[int], rinds: Optional[List[int]] = None):

    shp = np.array(tensor.shape)
    if rinds == None:
        rinds = [n for n in range(0, len(shp)) if n not in linds]
    ldim = np.prod(shp[linds])
    rdim = np.prod(shp[rinds])
    dest = np.concatenate((linds, rinds))
    mat = np.reshape(np.moveaxis(tensor, np.arange(len(dest)), np.argsort(dest)), (ldim, rdim))
    U, s, Vh = linalg.svd(mat, full_matrices=False, lapack_driver='gesvd')

    s2 = np.power(s, 2)
    tot_wt = np.sum(s2)
    discard_wts = np.cumsum(np.flip(s2)) / tot_wt
    #print(discard_wts)
    trunc_dim = len(s)
    if cutoff > 0.0:
        trunc_dim = np.count_nonzero(discard_wts > (cutoff ** 2))

    U_trunc = np.reshape(U[:, 0:trunc_dim], np.concatenate((shp[linds], [trunc_dim])))
    Vh_trunc = np.reshape(Vh[0:trunc_dim, :], np.concatenate(([trunc_dim], shp[rinds])))
    S_trunc = np.diag(s[0:trunc_dim])

    return U_trunc, S_trunc, Vh_trunc

def reshape_qr(tensor: np.ndarray, linds: List[int], rinds: Optional[List[int]] = None):

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

def reshape_rq(tensor: np.ndarray, linds: List[int], rinds: Optional[List[int]] = None):

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

# def shift_ortho_center(mps, loc):
#     # Shifts the orthogonality center to the loc-th site from the left
#     ll = len(mps)

#     if loc == 0:
#         U, S, Vh = svd_truncate(mps[-1], [1], [0], 0.0)
#         mps[-1] = Vh.T
#         US = U@S

#         for i in np.arange(ll-2, 0, -1):
#             #      |       
#             #    |---|   |---|
#             #  a-|   |-b-|   |-c
#             #    |---|   |---|
#             temp = np.einsum('jab,bc->jac', mps[i], US)
#             U, S, Vh = svd_truncate(temp, [1], [0,2], 0.0)
#             mps[i] = np.moveaxis(Vh, [0, 1, 2], np.argsort([1, 0, 2]))
#             US = U@S

#         mps[0] = np.einsum('ja,ab->jb', mps[0], US)

#     elif loc == ll-1:
#         U, S, Vh = svd_truncate(mps[0], [0], [1], 0.0)
#         SVh = S@Vh
#         mps[0] = U
#         for i in range(1, ll-1):
#             #              |
#             #    |---|   |---|
#             #  a-|   |-b-|   |-c
#             #    |---|   |---|
#             temp = np.einsum('ab,jbc->jac', SVh, mps[i])
#             U, S, Vh = svd_truncate(temp, [0,1], [2], 0.0)
#             mps[i] = U
#             SVh = S@Vh
#         mps[-1] = np.einsum('ab,jb->ja', SVh, mps[-1])

#     else:
#         U, S, Vh = svd_truncate(mps[0], [0], [1], 0.0)
#         SVh = S@Vh
#         mps[0] = U
#         for i in range(1, loc):
#             #              |
#             #    |---|   |---|
#             #  a-|   |-b-|   |-c
#             #    |---|   |---|
#             temp = np.einsum('ab,jbc->jac', SVh, mps[i])
#             U, S, Vh = svd_truncate(temp, [0,1], [2], 0.0)
#             mps[i] = U
#             SVh = S@Vh
#         mps[loc] = np.einsum('ab,jbc->jac', SVh, mps[loc])

#         U, S, Vh = svd_truncate(mps[-1], [1], [0], 0.0)
#         mps[-1] = Vh.T
#         US = U@S

#         for i in np.arange(ll-2, loc, -1):
#             #      |       
#             #    |---|   |---|
#             #  a-|   |-b-|   |-c
#             #    |---|   |---|
#             temp = np.einsum('jab,bc->jac', mps[i], US)
#             U, S, Vh = svd_truncate(temp, [1], [0,2], 0.0)
#             mps[i] = np.moveaxis(Vh, [0, 1, 2], np.argsort([1, 0, 2]))
#             US = U@S

#         mps[loc] = np.einsum('jab,bc->jac', mps[loc], US)

def shift_ortho_center(mps, loc):
    # Shifts the orthogonality center to the loc-th site from the left
    ll = len(mps)

    if loc == 0:
        r, q = reshape_rq(mps[-1], [1], [0])
        mps[-1] = q.T # To follow ordering convention of indices

        for i in np.arange(ll-2, 0, -1):
            #      |       
            #    |---|   |---|
            #  a-|   |-b-|   |-c
            #    |---|   |---|
            temp = np.einsum('jab,bc->jac', mps[i], r)
            r, q = reshape_rq(temp, [1], [0,2])
            mps[i] = np.moveaxis(q, [0, 1, 2], np.argsort([1, 0, 2]))

        mps[0] = np.einsum('ja,ab->jb', mps[0], r)

    elif loc == ll-1:
        q, r = reshape_qr(mps[0], [0], [1])
        mps[0] = q
        for i in range(1, ll-1):
            #              |
            #    |---|   |---|
            #  a-|   |-b-|   |-c
            #    |---|   |---|
            temp = np.einsum('ab,jbc->jac', r, mps[i])
            q, r = reshape_qr(temp, [0,1], [2])
            mps[i] = q
        mps[-1] = np.einsum('ab,jb->ja', r, mps[-1])

    else:
        q, r = q, r(mps[0], [0], [1])
        mps[0] = q
        for i in range(1, loc):
            #              |
            #    |---|   |---|
            #  a-|   |-b-|   |-c
            #    |---|   |---|
            temp = np.einsum('ab,jbc->jac', r, mps[i])
            q, r = reshape_qr(temp, [0,1], [2])
            mps[i] = q
        mps[loc] = np.einsum('ab,jbc->jac', r, mps[loc])

        r, q = reshape_rq(mps[-1], [1], [0])
        mps[-1] = q.T

        for i in np.arange(ll-2, loc, -1):
            #      |       
            #    |---|   |---|
            #  a-|   |-b-|   |-c
            #    |---|   |---|
            temp = np.einsum('jab,bc->jac', mps[i], r)
            r, q = reshape_rq(temp, [1], [0,2])
            mps[i] = np.moveaxis(q, [0, 1, 2], np.argsort([1, 0, 2]))

        mps[loc] = np.einsum('jab,bc->jac', mps[loc], r)
        
def contract_zipup(mps, mpo, percentage):
    # Note: the latest time is always mps[0], so as the contraction proceeds, the mps will gradually become right canonical

    # At the k-th step (k=1, ..., N-1), have the MPO and MPS
    #
    #       b_(N+M-1,k-1)    ...   b_0(k-1,k-1)               : MPO
    #       j_(N+M-1)        ...   j_(k-1) ... .... j_1 j_0   : MPS
    # with orthogonality center at N-k-1

    # Written out left to right as N+M-1, N+M-2, ... 0, the MPS before contraction is assumed to be left canonical
    # We will contract the MPO from right to left, shifting the orthogonality center towards the left as we move along
    # The final MPS will be right canonical
    # A sweep returns the MPS to left canonical form

    len_o = len(mpo)

    # check that len_o <= len(mps)
    
    if len_o > 1:
        # Process tensor contraction, rightmost mps has 3 legs

        #      j
        #      |
        #    |---|
        #  a-|   |
        #    |---|
        #      |
        #      i
        #      |
        #    |---|
        #  c-|   |-d
        #    |---|

        if len_o == len(mps):
            temp = np.einsum('jai,ic->acj', mpo[-1], mps[-1])
            U, S, Vh = svd_truncate(temp, percentage/10, [0,1], [2])

            mps[-1] = np.transpose(Vh) # conform to the convention for index ordering
            US = U@S
        else:
            temp = np.einsum('jai,icd->acjd', mpo[len_o-1], mps[len_o-1])
            U, S, Vh = svd_truncate(temp, percentage/10, [0,1], [2,3])
            mps[len_o-1] = np.moveaxis(Vh, [0, 1, 2], np.argsort([1, 0, 2])) # conform to the convention for index ordering
            US = U@S

        for i in range(2, len_o):

            #      j
            #      |
            #    |---|
            #  a-|   |-b
            #    |---|
            #      |
            #      i

            #      |
            #    |---|
            #  c-|   |-d
            #    |---|
            temp = contract('jaib,icd,bde->acje', mpo[len_o-i], mps[len_o-i], US)
            U, S, Vh = svd_truncate(temp, percentage/10, [0,1], [2,3])
            mps[len_o-i] = np.moveaxis(Vh, [0, 1, 2], np.argsort([1, 0, 2]))
            US = U@S

        #      j
        #      |
        #    |---|
        #    |   |-b
        #    |---|
        #      |
        #      i
        #      |
        #    |---|
        #    |   |-d
        #    |---|
        mps[0] = contract('jib,id,bde->je', mpo[0], mps[0], US)

        # MPS is now right canonical, with orthogonality center at the leftmost site
    elif len_o == 1:
        mps[0] = np.einsum('ij,ja->ia', mpo[0], mps[0])
