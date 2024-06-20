#import sys
import numpy as np
from scipy import linalg
from time import time
from opt_einsum import contract
from typing import List

from cfifs.bath import Bath
from cfifs.sim_params import SimulationParams
from cfifs.baths.expbath_eta_ctc import ExpBath_Eta_CTC as eta
from cfifs.inflfn import IF_CTC as influence_functional
from cfifs.utilities import svd_truncate, reshape_qr, reshape_rq
from cfifs.mps import MPS

class TEMPO_CTC():
    
    def __init__(self, bath: Bath, sp: SimulationParams):
    
        self.N = sp.N
        self.M = sp.M
        self.dtau = (sp.t_max - 1j/(2*sp.T))/sp.N

        self.h_0 = sp.H_S
        self.dh = sp.H_S.shape[0]

        self.percentage = sp.cutoff
        self.maxdim = sp.maxdim

        self.eta = eta(bath, sp)
        self.eta_k = np.zeros(2*self.N + 2, dtype=complex)
        self.eta_kk = np.zeros((2*self.N+2, 2*self.N+2), dtype=complex)

        tt = time()
        ek = self.eta.eta_k(0)
        for i in range(0, 2*self.N+1):
            for j in range(1, i):
                self.eta_kk[i, j] = self.eta.eta_kk(i,j)

        for i in range(0, self.N+1):
            self.eta_k[i] = ek
        for i in range(self.N+1, 2*self.N+1):
            self.eta_k[i] = np.conj(ek)

        print("eta time:", time()-tt)
        
        self.ifn = influence_functional(self.eta_k, self.eta_kk)

        self.tnif = None
        self.ind_arr = np.arange(2*self.N+1,-1,-1)        


    def get_init_mps(self, opB: np.ndarray) -> MPS:

        m_arr = []

        Uprop_half = linalg.expm(-1j*self.h_0*self.dtau/2).T
        Uprop_f = Uprop_half @ Uprop_half
        
        q, r = reshape_qr(Uprop_half, [0], [1])
        rT = r.T

        m_arr.append(q)

        for _ in range(self.N-1):
            q, r = reshape_qr(Uprop_f, [0], [1])
            m_arr.append(np.einsum('sa,sb->sab', rT, q))
            rT = r.T

        Bprop = Uprop_half @ opB.T @ linalg.expm(1j*self.h_0*self.dtau.conjugate()/2).T

        q, r = reshape_qr(Bprop, [0], [1])
        m_arr.append(np.einsum('sa,sb->sab', rT, q))
        rT = r.T

        Uprop_half_b = linalg.expm(1j*self.h_0*self.dtau.conjugate()/2).T
        Uprop_b = Uprop_half_b @ Uprop_half_b
        
        for _ in range(self.N-1):            
            q, r = reshape_qr(Uprop_b, [0], [1])
            m_arr.append(np.einsum('sa,sb->sab', rT, q))
            rT = r.T

        q, r = reshape_qr(Uprop_half_b, [0], [1])
        m_arr.append(np.einsum('sa,sb->sab', rT, q))
        rT = r.T

        m_arr.append(rT)

        ll = len(m_arr)
        state = MPS(m_arr, 0)
        state.canonicalize(ll-1)

        return state

    
    def get_MPO(self,ind_arr: List[int], ind: int) -> List[np.ndarray]:

        mpo = []
        iden = np.eye(self.dh)

        delta = np.einsum('pa,ji->jpia', iden, iden)

        ## prep for tensors to add to F_MPO_k list
        F_1 = np.einsum('sz,sb->szb', iden, self.ifn.I_k(ind_arr[0], ind))
        mpo.append(F_1)

        for i in range(1, len(ind_arr)-1):
            if ind_arr[i] != ind:
                mpo.append(np.einsum('jpia,jp->jpia', delta, self.ifn.I_k(ind_arr[i],ind)))
            else:
                mpo.append(np.einsum('ji,ajb->iajb', iden, self.ifn.I_k(ind_arr[i],ind)))

        F_end = np.einsum('sz,sb->sbz', iden, self.ifn.I_k(ind_arr[-1],ind))
        mpo.append(F_end)

        return mpo
        

    def propagate(self, opA: np.ndarray, opB: np.ndarray) -> complex:

        self.ind_arr = np.arange(2*self.N+1,-1,-1)
        
        self.tnif = self.get_init_mps(opB)
        
        contract_time = 0.0
        mpo_time = 0.0
        decim_time = 0.0

        while len(self.ind_arr) > 2:

            i = len(self.ind_arr)//2
            tt = time()
            mpo = self.get_MPO(self.ind_arr, self.ind_arr[i])
            mpo_time += (time() - tt)

            tt = time()
            self.tnif.contract_zipup(mpo, self.percentage)
            contract_time += (time() - tt)

            tt = time()
            self.tnif.decimate_site(i)
                
            self.ind_arr = np.delete(self.ind_arr,i)
            decim_time += (time() - tt)
            
        res = contract('ij,kj,ki', self.tnif.mps[0], self.tnif.mps[1], opA)

        print("time to make mpo: ", mpo_time)
        print("time to do contractions: ", contract_time)
        print("time to decimate site: ", decim_time)

        return res[()]
