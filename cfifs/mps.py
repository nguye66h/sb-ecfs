import numpy as np
from typing import List
from opt_einsum import contract

from cfifs.utilities import svd_truncate, reshape_rq, reshape_qr

class MPS():

    def __init__(self, mps: List[np.ndarray], loc: int):

        # Convention: indices ordered anticlockwise from top (= site index)
        self.mps = mps
        self.loc = loc

        # check that loc within bounds of mps array

    def length(self):
        return len(self.mps)

    def pop(self):
        self.mps.pop()
        self.loc -= 1

        return
    
    def canonicalize(self, loc: int):

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
        # Shifts the orthogonality center to mps[loc]
        ll = self.length()
        loc0 = self.loc

        if loc > ll:
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
            
    def contract_zipup(self, mpo: List[np.ndarray], percentage: float):

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
                U, S, Vh = svd_truncate(temp, percentage/10, [0,1], [2])

                self.mps[-1] = np.transpose(Vh) # conform to the convention for index ordering
            else:
                temp = np.einsum('jai,icd->acjd', mpo[len_o-1], self.mps[len_o-1])
                U, S, Vh = svd_truncate(temp, percentage/10, [0,1], [2,3])
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
                U, S, Vh = svd_truncate(temp, percentage/10, [0,1], [2,3])
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
            self.sweep_right(percentage)
            
        elif len_o == 1:
            self.mps[0] = np.einsum('ij,ja->ia', mpo[0], self.mps[0])

        return

    def sweep_right(self, percentage):

        ll = self.length()
        if self.loc != 0:
            self.shift_ortho_center(0)

        U, S, Vh = svd_truncate(self.mps[0], percentage, [0], [1])
        self.mps[0] = U
        SVh = S@Vh
        
        for i in range(1, ll-1):
            #              |
            #    |---|   |---|
            #  a-|   |-b-|   |-c
            #    |---|   |---|
            temp = np.einsum('ab,jbc->jac', SVh, self.mps[i])
            U, S, Vh = svd_truncate(temp, percentage, [0, 1], [2])
            self.mps[i] = U
            SVh = S@Vh

        self.mps[ll-1] = np.einsum('ab,jb->ja', SVh, self.mps[ll-1])
        self.loc = ll-1

        return

    def sweep_left(self, percentage):

        ll = self.length()
        if self.loc != ll-1:
            self.shift_ortho_center(ll-1)

        U, S, Vh = svd_truncate(self.mps[ll-1], percentage, [1], [0])
        self.mps[ll-1] = Vh.T
        US = U@S
        
        for i in range(ll-2, 0, -1):
            #      |       
            #    |---|   |---|
            #  a-|   |-b-|   |-c
            #    |---|   |---|
            temp = np.einsum('jab,bc->jac', self.mps[i], US)
            U, S, Vh = reshape_rq(temp, percentage, [1], [0,2])
            self.mps[i] = np.moveaxis(Vh, [0, 1, 2], np.argsort([1, 0, 2]))
            US = U@S

        self.mps[0] = np.einsum('ja,ab->jb', self.mps[0], US)
        self.loc = 0

        return

    def decimate_site(self, si):
        
        ll = self.length()
        d = self.mps[0].shape[0]
        triv = np.ones(d)

        self.shift_ortho_center(si)
            
        temp = contract('i,iab->ab', triv, self.mps[si])

        if si+1 == ll-1:
            self.mps[si+1] = np.einsum('ab,ib->ia', temp, self.mps[si+1])
            self.loc = ll-1
        elif si+1 < ll-1:
            self.mps[si+1] = np.einsum('ab,ibc->iac', temp, self.mps[si+1])
            self.loc = si+1

            self.shift_ortho_center(ll-1)
            
            # U, S, Vh = self.svd_truncate(self.mps[si+1], [0,1], [2], 0.0)
            # self.mps[si+1] = U
            # SVh = S@Vh
            # for i in range(si+2, ll-1):
            #     temp = np.einsum('ab,jbc->jac', SVh, self.mps[i])
            #     U, S, Vh = self.svd_truncate(temp, [0,1], [2], 0.0)
            #     self.mps[i] = U
            #     SVh = S@Vh
            # temp = np.einsum('ab,jb->ja', SVh, self.mps[ll-1])
            # self.mps[ll-1] = temp
        self.mps.pop(si)
        self.loc -= 1

        return
