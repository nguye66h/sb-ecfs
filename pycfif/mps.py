import numpy as np
from numpy import linalg
from typing import List, Optional
from opt_einsum import contract

from pycfif.utilities import svd_truncate, reshape_rq, reshape_qr

class MPS():
    '''
    Matrix product state over a finite number of sites

    Attributes
    ----------
    mps: List[np.ndarray]
        List of tensors, defining the matrix product state
    loc: int
        Site (0-indexed) of the orthogonality center, mps[loc]
    '''
    def __init__(self, mps: List[np.ndarray], loc: int):

        # Convention: indices ordered anticlockwise from top (= site index)
        self.mps = mps
        self.loc = loc

        # check that loc within bounds of mps array

    def length(self):
        '''
        Returns the length of the MPS

        Returns
        -------
        len: int
            Length of the MPS
        '''
        return len(self.mps)

    def get_bond_dims(self) -> np.ndarray:
        '''
        Returns an array of all the bond dimensions of the MPS, such that the 0th entry corresponds to the bond between sites 0 and 1, etc.

        Returns
        -------
        bds: np.ndarray
            Array of bond dimensions of the MPS
        '''

        n = len(self.mps)
        bds = np.zeros(n-1, dtype=int)
        bds[0] = (self.mps[0].shape)[1]
        for i in range(1, n-1):
            bds[i] = (self.mps[i]).shape[2]

        return bds

    #def pop(self):
    #    self.mps.pop()
    #    self.loc -= 1
    #    return
    
    def canonicalize(self, loc: int):
        '''
        Canonicalizes the MPS, with orthogonality center placed at the specified location

        Parameters
        ----------
        loc: int
            Site (0-indexed) to place the orthogonality center, i.e., mps[loc]
        '''

        ll = self.length()
        # check that loc within bounds
        
        if loc > 0:
        
            q, r = reshape_qr(self.mps[0], [0], [1])
            self.mps[0] = q
        
            for i in range(1, loc):
                #              |
                #    |---|   |---|
                #  a-|   |-b-|   |-c
                #    |---|   |---|
                temp = np.einsum('ab,jbc->jac', r, self.mps[i])
                q, r = reshape_qr(temp, [0, 1], [2])
                self.mps[i] = q

            if loc == ll-1:
                temp = np.einsum('ab,jb->ja', r, self.mps[loc])
            else:
                temp = np.einsum('ab,jbc->jac', r, self.mps[loc])
            self.mps[loc] = temp

        if loc < ll-1:
            r, q = reshape_rq(self.mps[ll-1], [1], [0])
            self.mps[ll-1] = q.T # To follow ordering convention of indices

            for i in range(ll-2, loc, -1):
                #      |
                #    |---|   |---|
                #  a-|   |-b-|   |-c
                #    |---|   |---|
                temp = np.einsum('jab,bc->jac', self.mps[i], r)
                r, q = reshape_rq(temp, [1], [0, 2])
                self.mps[i] = np.moveaxis(q, [0, 1, 2], np.argsort([1, 0, 2]))

            if loc == 0:
                temp = np.einsum('ja,ab->jb', self.mps[loc], r)
            else:
                temp = np.einsum('jab,bc->jac', self.mps[loc], r)
            self.mps[loc] = temp

        self.loc = loc

        return

    def shift_ortho_center(self, loc: int):
        '''
        For an already orthogonalized MPS, shift its orthogonality center to specified site

        Parameters
        ----------
        loc: int
            Site (0-indexed) to place the orthogonality center, i.e., mps[loc]
        '''

        ll = self.length()
        loc0 = self.loc

        if loc > ll or loc < 0:
            raise RuntimeError("Cannot shift orthogonality center beyond boundaries of MPS")

        if loc0 < loc:

            if loc0 == 0:
                q, r = reshape_qr(self.mps[loc0], [0], [1])
            else:
                q, r = reshape_qr(self.mps[loc0], [0, 1], [2])
            self.mps[loc0] = q
            for i in range(loc0+1, loc):
                #              |
                #    |---|   |---|
                #  a-|   |-b-|   |-c
                #    |---|   |---|
                temp = np.einsum('ab,jbc->jac', r, self.mps[i])
                q, r = reshape_qr(temp, [0, 1], [2])
                self.mps[i] = q
            if loc == ll-1:
                self.mps[loc] = np.einsum('ab,jb->ja', r, self.mps[loc])
            else:
                self.mps[loc] = np.einsum('ab,jbc->jac', r, self.mps[loc])

        elif loc < loc0:

            if loc0 == ll-1:
                r, q = reshape_rq(self.mps[loc0], [1], [0])
                self.mps[loc0] = q.T
            else:
                r, q = reshape_rq(self.mps[loc0], [1], [0, 2])
                self.mps[loc0] = np.moveaxis(q, [0, 1, 2], np.argsort([1, 0, 2]))
            for i in range(loc0-1, loc, -1):
                #      |       
                #    |---|   |---|
                #  a-|   |-b-|   |-c
                #    |---|   |---|
                temp = np.einsum('jab,bc->jac', self.mps[i], r)
                r, q = reshape_rq(temp, [1], [0,2])
                self.mps[i] = np.moveaxis(q, [0, 1, 2], np.argsort([1, 0, 2]))
            if loc == 0:
                self.mps[loc] = np.einsum('ja,ab->jb', self.mps[loc], r)
            else:
                self.mps[loc] = np.einsum('jab,bc->jac', self.mps[loc], r)

        self.loc = loc

        return
            
    def contract_zipup(self, mpo: List[np.ndarray], cutoff: float):
        '''
        Contract an MPO into the MPS by the zip-up algorithm (arXiv:1002.1305), truncating the bond dimensions in the process. The MPO is applied starting from the the last element of the mps, mps[-1], and the MPO can be shorter than the MPS.

        Parameters
        ----------
        mpo: List[np.ndarray] 
            List of MPOs. The length of mpo cannot exceed the length of the MPS
        cutoff: float
            Truncation threshold
        '''

        len_o = len(mpo)
        if len_o > len(self.mps):
            raise RuntimeError("Length of MPOs cannot be greater than length of MPS")
        
        if self.loc != len_o-1:
            self.shift_ortho_center(len_o - 1)

        if len_o > 1:
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
            if len_o == len(self.mps):
                temp = np.einsum('jai,ic->acj', mpo[-1], self.mps[-1])
                U, S, Vh = svd_truncate(temp, cutoff/10, [0,1], [2])

                self.mps[-1] = np.transpose(Vh) # conform to the convention for index ordering
            else:
                temp = np.einsum('jai,icd->acjd', mpo[len_o-1], self.mps[len_o-1])
                U, S, Vh = svd_truncate(temp, cutoff/10, [0,1], [2,3])
                self.mps[len_o-1] = np.moveaxis(Vh, [0, 1, 2], np.argsort([1, 0, 2])) # conform to the convention for index ordering

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
                temp = contract('jaib,icd,bde->acje', mpo[len_o-i], self.mps[len_o-i], US)
                U, S, Vh = svd_truncate(temp, cutoff/10, [0,1], [2,3])
                self.mps[len_o-i] = np.moveaxis(Vh, [0, 1, 2], np.argsort([1, 0, 2]))
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
            self.mps[0] = contract('jib,id,bde->je', mpo[0], self.mps[0], US)
            self.loc = 0
            # MPS is now right canonical, with orthogonality center at the leftmost site
            self.sweep_right(cutoff)
            
        elif len_o == 1:
            self.mps[0] = np.einsum('ij,ja->ia', mpo[0], self.mps[0])

        return

    def sweep_right(self, cutoff: float):
        '''
        Performs a rightward compression sweep (starting from mps[0])

        Parameters
        ----------
        cutoff: float
            Truncation threshold
        '''
        
        ll = self.length()
        if self.loc != 0:
            self.shift_ortho_center(0)

        U, S, Vh = svd_truncate(self.mps[0], cutoff, [0], [1])
        self.mps[0] = U
        SVh = S@Vh
        
        for i in range(1, ll-1):
            #              |
            #    |---|   |---|
            #  a-|   |-b-|   |-c
            #    |---|   |---|
            temp = np.einsum('ab,jbc->jac', SVh, self.mps[i])
            U, S, Vh = svd_truncate(temp, cutoff, [0, 1], [2])
            self.mps[i] = U
            SVh = S@Vh

        self.mps[ll-1] = np.einsum('ab,jb->ja', SVh, self.mps[ll-1])
        self.loc = ll-1

        return

    def sweep_left(self, cutoff: float):
        '''
        Performs a leftward compression sweep (ending at mps[0])

        Parameters
        ----------
        cutoff: float
            Truncation threshold        
        '''

        ll = self.length()
        if self.loc != ll-1:
            self.shift_ortho_center(ll-1)

        U, S, Vh = svd_truncate(self.mps[ll-1], cutoff, [1], [0])
        self.mps[ll-1] = Vh.T
        US = U@S
        
        for i in range(ll-2, 0, -1):
            #      |       
            #    |---|   |---|
            #  a-|   |-b-|   |-c
            #    |---|   |---|
            temp = np.einsum('jab,bc->jac', self.mps[i], US)
            U, S, Vh = reshape_rq(temp, cutoff, [1], [0,2])
            self.mps[i] = np.moveaxis(Vh, [0, 1, 2], np.argsort([1, 0, 2]))
            US = U@S

        self.mps[0] = np.einsum('ja,ab->jb', self.mps[0], US)
        self.loc = 0

        return

    def decimate_site(self, si: int):
        '''
        Sums over the specified site of the MPS and contracts the result into the next site. Assumes that the decimated site is away from the boundaries of the MPS.

        Parameters
        ----------
        si: int
            Site (0-indexed) to decimate, i.e., mps[si]
        '''
        
        ll = self.length()
        d = self.mps[0].shape[0]
        triv = np.ones(d)

        assert (si < ll-1 and si >= 0), "Decimating the last site is not supported" 
        
        self.shift_ortho_center(si)
            
        temp = contract('i,iab->ab', triv, self.mps[si])

        if si+1 == ll-1:
            self.mps[si+1] = np.einsum('ab,ib->ia', temp, self.mps[si+1])
            self.loc = ll-1
        elif si+1 < ll-1:
            self.mps[si+1] = np.einsum('ab,ibc->iac', temp, self.mps[si+1])
            self.loc = si+1

            self.shift_ortho_center(ll-1)
            
        self.mps.pop(si)
        self.loc -= 1

        return

class uMPS():
    '''
    Infinite/uniform matrix product state, assumed to have a two-site unit cell

    Attributes
    ----------
    tensors: List[np.ndarray]
        List of tensors, defining the uniform matrix product state. Must either have two elements or four elements, see `vidal_form`
    vidal_form: bool
        Boolean specifying whether the list of tensors `tensors` is in Vidal's representation or not
    '''
    def __init__(self, tensors: List[np.ndarray], vidal_form: bool):
        
        # Convention: indices ordered anticlockwise from top (= site index)
        #      0          0
        #      |          |
        #    |---|      |---|
        #  1-| A |-2  1-| B |-2
        #    |---|      |---|
        self.tensors = tensors
        self.vidal_form = vidal_form
        # If vidal_form, expect tensors = [a, s_ab, b, s_bc, c, ...]

        if vidal_form:
            assert len(tensors) == 4, "Only 2 site unit cells are supported right now"
        else:
            assert len(tensors) == 2, "Only 2 site unit cells are supported right now"

    def step_itebd_ov(self, gate: np.ndarray, cutoff: float, p: Optional[float]=1.0):
        '''
        Performs one step of iTEBD using the Orus-Vidal algorithm (arXiv:0711.3960)

        Parameters
        ----------
        gate: np.ndarray
            Nearest-neighboring two-site gate to be contracted
        cutoff: float
            Truncation threshold
        p: float, optional
            p-norm for singular value truncation

        '''

        #
        #      0      3
        #      |      |
        #   |------------|
        #   |    gate    |
        #   |------------|
        #      |      |
        #      1      2
        #

        assert self.vidal_form == True, "Convert the uMPS to Vidal's representation to use the Orus-Vidal formulation of iTEBD"
        
        # a = self.tensors[0]
        # s_ab = self.tensors[1]
        # b = self.tensors[2]
        # s_ba = self.tensors[3]

        threshold = 1e-13
        
        s_ab = self.tensors[1] * linalg.norm(self.tensors[3])
        s_ba = self.tensors[3] / linalg.norm(self.tensors[3])
        s_ba[np.abs(s_ba) < threshold] = threshold
        
        inv_s_ab = 1/s_ba
        
        c = contract('iklj,uv,kvw,wx,lxy,yz->iuzj', gate, np.diag(s_ba), self.tensors[0], np.diag(s_ab), self.tensors[2], np.diag(s_ba))

        U, s_a, Vh = svd_truncate(c, cutoff, [0, 1], [3, 2], p=p)
        s_a_norm = linalg.norm(np.diagonal(s_a))
        
        new_b = np.einsum('xiy,yz->ixz', Vh, np.diag(inv_s_ab))
        new_a = np.einsum('xy,iyz->ixz', np.diag(inv_s_ab), U)

        self.tensors[0] = new_b
        self.tensors[1] = s_ba
        self.tensors[2] = new_a
        self.tensors[3] = np.diagonal(s_a) / s_a_norm
        
        return
    
    def step_itebd_mbh(self, gate: np.ndarray, s_b: np.ndarray, cutoff: float, p: Optional[float]=2.0):
        '''
        Performs one step of iTEBD using the Hastings modification (arXiv:0903.3253) to the Orus-Vidal algorithm

        Parameters
        ----------
        gate: np.ndarray
            Nearest-neighboring two-site gate to be contracted
        cutoff: float
            Truncation threshold
        p: float, optional
            p-norm for singular value truncation
        '''
        
        #
        #      0      3
        #      |      |
        #   |------------|
        #   |    gate    |
        #   |------------|
        #      |      |
        #      1      2
        #

        assert self.vidal_form == False, "Convert the uMPS from Vidal's representation to use Hasting's modification of iTEBD"
        
        # a = self.tensors[0]
        # b = self.tensors[1]
        
        c = contract('iklj,kxy,lyz->ixzj', gate, self.tensors[0], self.tensors[1])
        theta = contract('x,ixzj->ixzj', s_b, c)

        U, s_a, Vh = svd_truncate(theta, cutoff, [0, 1], [3, 2], p=p)
        s_a_norm = linalg.norm(np.diagonal(s_a))
        
        new_b = np.moveaxis(Vh, [0, 1, 2], np.argsort([1, 0, 2]))
        new_a = contract('ixyj,jzy->ixz', c, np.conj(new_b))

        self.tensors[0] = new_b
        self.tensors[1] = new_a
        
        return np.diagonal(s_a) / s_a_norm
