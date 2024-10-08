import numpy as np
from scipy import linalg
from time import time
from opt_einsum import contract
from typing import List, Tuple

from pycfif.bath import Bath
from pycfif.sim_params import SimulationParamsKBC
from pycfif.eta_kbc import Eta_KBC
from pycfif.inflfn import IF_PT as influence_functional
from pycfif.utilities import svd_truncate
from pycfif.mps import MPS

class TEMPO_PT:
    '''
    Defines a TEMPO calculation for the thermal correlation function along the Kadanoff-Baym-like contour (KBC)

    Attributes
    ----------
    N: int
        Number of timesteps along the real-time part of the contour
    M: int
        Number of timesteps along the half of the imaginary-time part of the contour
    dt: float
        Timestep along the real-time part of the contour
    db: float
        Timestep along the imaginary-time part of the contour
    h_0: np.ndarray
        System Hamiltonian
    dh: int
        Dimension of the system Hilbert space
    dl: int
        Dimension of the system Liouville space
    percentage: float
        Truncation threshold
    eta_t: ndarray
        List of eta coefficients defining the spin-boson influence functional, related to double integrals over the bath correlation function
    ifn: influence_functional
        Object storing the analytic influence functional for the given quadratic bath of bosons
    tnif: List[np.ndarray]
        Matrix product state representation of the influence functional
    '''
    def __init__ (self, bath: Bath, sp: SimulationParamsKBC, eta_obj: Eta_KBC):

        assert isinstance(sp, SimulationParamsKBC)
        assert issubclass(type(eta_obj), Eta_KBC)
        
        self.N = sp.N
        self.M = sp.M
        self.dt = sp.t_max/max(sp.N, 1)
        self.db = 1/(2*sp.M*sp.T)
        self.percentage = sp.cutoff

        self.h_0 = sp.H_S
        self.dh = sp.H_S.shape[0]
        self.dl = (self.dh)**2

        #eta_obj = eta(bath, sp)
        
        time0 = time()
        eta_t, eta_ppmm_b, eta_pmmp_b, eta_pm_b_k, mix_eta_pp_arr, mix_eta_pm_arr = self._generate_eta_list(eta_obj)
        print('time for eta:', time() - time0)
    
        self.ifn = influence_functional(eta_t,
                                        eta_ppmm_b, eta_pmmp_b,
                                        eta_pm_b_k, mix_eta_pp_arr, mix_eta_pm_arr)
        self.tnif = None


    def _generate_eta_list(self, eta_obj: Eta_KBC) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Generates the etas required to define the influence functional on the Kadanoff-Baym-like contour

        Parameters
        ----------
        eta_obj: Eta_KBC
            eta object defined on the Kadanoff-Baym-like contour

        Returns
        -------
        eta_t: ndarray
        eta_ppmm_b: ndarray
        eta_pmmp_b: ndarray
        eta_pm_b_k: ndarray
        mix_eta_pp_arr: ndarray
        mix_eta_pm_arr: ndarray
        '''
        eta_t = np.zeros((self.N+1, 4), dtype=complex)
        eta_ppmm_b = np.zeros((self.M, 2), dtype=complex)
        eta_pmmp_b = np.zeros((2*self.M-2, 2), dtype=complex)
        eta_pm_b_k = np.zeros(self.M, dtype=complex)
        mix_eta_pp_arr = np.zeros((self.N+self.M, self.M), dtype=complex)
        mix_eta_pm_arr = np.zeros((self.N+self.M, self.M), dtype=complex)

        pp = eta_obj.eta_pp_tt_k()
        pm = eta_obj.eta_pm_tt_k()

        eta_t[0, :] = [pp, np.conj(pp), pm, np.conj(pm)]


        pp = eta_obj.eta_pp_bb_k()
        eta_ppmm_b[0, :] = [pp, np.conj(pp)]

        ## k is assumed to be bigger than self.M
        for i in range(1,self.N+1):

            pp = eta_obj.eta_pp_tt_kk(i)
            pm = eta_obj.eta_pm_tt_kk(i)
            eta_t[i, :] = [pp, np.conj(pp), pm, np.conj(pm)]

            for j in range(0,self.M):
                mix_eta_pp_arr[i-1+self.M,j] = eta_obj.eta_pp_mix_kk(i,j)
                mix_eta_pm_arr[i-1+self.M,j] = eta_obj.eta_pm_mix_kk(i,j)
                
          
        eta_pmmp_b[0, :] = [0,0]
        eta_pm_b_k[0] = eta_obj.eta_pm_bb_k(0)

        for i in range(1,2*self.M-2):
            pm = eta_obj.eta_pm_bb_kk(i) #k+kp should run from 1 to 2M-3
            eta_pmmp_b[i, :] = [pm, np.conj(pm)]

        for i in range(1,self.M):
            pp = eta_obj.eta_pp_bb_kk(i) # k-kp should run from 1 to M-1
            eta_ppmm_b[i, :] = [pp, np.conj(pp)]
            eta_pm_b_k[i] = eta_obj.eta_pm_bb_k(i)

        return eta_t, eta_ppmm_b, eta_pmmp_b, eta_pm_b_k, mix_eta_pp_arr, mix_eta_pm_arr

    def _get_mpo_block(self, k: int, kp: int) -> List[np.ndarray]:
        '''
        Construct the list of matrix product operators that introduce correlations between two specified points along the Kadanoff-Baym-like contour

        Parameters
        ----------
        k: int
            Final point to correlate
        kp: int
            Initial point to correlate, should be less than or equal to k

        Returns
        -------
        mpo: List[ndarray]
            List of matrix product operators
        '''
        
        # Input arguments assume that k > kp
        # Liouvillian dimension
        dl = self.dl
        iden = np.eye(dl)
        
        mpo = []

        if k == kp:
            mpo.append(self.ifn.I_k (k,k))
            return mpo


        ### np.eye(n) is always a square matrix of size nxn
        delta = np.einsum('pa,ji->jpia', iden, iden)

        # leftmost
        b0 = np.einsum('ij,ja->jia', iden, self.ifn.I_k(k,kp))
        
        # rightmost
        bend = np.einsum('ij,ja->jai', iden, self.ifn.I_k(kp,kp))

        mpo.append(b0)


        ## gives [k-1,k-2,...,kp+1]
        for d in range(k-1, kp, -1):
            Ik = self.ifn.I_k(d, kp)
            Ik = np.einsum('jpia,ja->jaip', delta, Ik)
            #print('------IK-----',Ik.shape)
            mpo.append(Ik)

        mpo.append(bend)

        return mpo

    def _get_init_mps(self, N: int) -> MPS:
        '''
        Constructs the initial matrix product state out of a chain of b-tensors correlating the initial contour-time with all other contour-times.

        Parameters
        ----------
        N: int
            Number of steps along the real-time part of the contour
        
        Returns
        -------
        state: MPS
            Matrix product state on which the influence functional will be built
        '''
        dl = self.dl
        
        ### the arrangement of indices of MPO is anticlockwise from 12 o'clock
        ###           < o
        ###          |    ^
        ###           > - |

        m_arr = []
        delta = np.einsum('pa,ji->jpia',np.eye(dl),np.eye(dl))

        # leftmost
        b0 = self.ifn.I_k(N+self.M-1, 0)
        
        # rightmost
        bend = self.ifn.I_k(0, 0)

        m_arr.append(b0)

        ## gives [k-1,k-2,...,kp+1]
        for d in range(N+self.M-2, 0, -1):
            Ik = np.einsum('ab,ja->jab', np.eye(dl), self.ifn.I_k(d, 0))
            #print('------IK-----',Ik.shape)
            m_arr.append(Ik)

        m_arr.append(bend)

        # The mps now corresponds to the usual TEMPO tensors
        ll = len(m_arr)
        state = MPS(m_arr, 0)
        state.canonicalize(ll - 2)

        return state
        
    def get_correlator(self, opA: np.ndarray, opB: np.ndarray) -> np.ndarray:
        '''
        Computes the correlation function (either the thermal ⟨A(t) B⟩ or the symmetrized ⟨A(t-iβ/2) B⟩) over N steps on the real-time interval defined for the TEMPO calculation. Requires the influence functional to have been already constructed through `make_thermpt` or `make_sympt`.

        Parameters
        ----------
        opA: ndarray
            System observable for measurement
        opB: ndarray
            System observable for measurement
        
        Returns
        -------
        t_corr: ndarray
            List of the correlation function evaluated on the N time steps, without normalization by the partition function
        '''
        dh = self.dh
        dl = self.dl
        iden = np.eye(dl, dtype=complex)

        t_corr = np.zeros(self.N+1, dtype=complex)
        
        tt = time()
        
        expdb = linalg.expm(-self.db*self.h_0/2)
        expdt = linalg.expm(1j*self.dt*self.h_0/2)
        
        k3_t = (expdb@expdb).flatten()
        k3v_t = np.diag(k3_t * ([1.0+0.0j] * dl))
        th_imag = np.einsum('ii->i', k3v_t)

        # Measurements
        m_A = np.kron((opA@expdt).T, expdt.conj().T).T
        m_An = np.kron((expdt@opA@expdt).T, (expdt@expdt).conj().T).T

        # No measurements
        #   on the real time part of contour
        prop_re = np.kron((expdt@expdt).T, (expdt@expdt).conj().T).T
        
        print("prepare measurement components: ", time() - tt)

        prop_fin_arr = []
        endcap = np.zeros(dl, dtype=complex)
        for i in range(0, dh):
            endcap[i*(dh+1)] = complex(1)
        
        prop_fin_arr.append(endcap) # for t=t_f

        
        # eg, (iaj) = 
        #     i j
        #     | |
        #    |---|
        #    |   |-a
        #    |---|
        temp = contract('i,ia,ij->ja', prop_fin_arr[-1], self.tnif.mps[0], np.eye(dl, dtype=complex))
        prop_fin_arr.append(temp)

        for n in range(1, self.N-1):

            #     i j
            #     | |
            #    |---|
            #  a-|   |-b
            #    |---|
            temp = contract('ia,iab,ij->jb', prop_fin_arr[-1], self.tnif.mps[n], iden)
            prop_fin_arr.append(temp)
            
        print("build and store final propagations: ", time() - tt)
        
        tt = time()        
        th_imag = self.tnif.mps[self.N-1]
        for n in range(1, self.N-1):
            temp = np.einsum('ia,ij->ja', prop_fin_arr[-n], m_An)
            t_corr[n] = np.einsum('ia,ia', temp, th_imag)
            temp = np.einsum('ij,ja->ia', prop_re, th_imag)
            th_imag = contract('iab,ij,jb->ia', self.tnif.mps[self.N - n - 1], iden, temp)

        temp = np.einsum('ia,ij->ja', prop_fin_arr[1], m_An)
        t_corr[self.N-1] = np.einsum('ia,ia', temp, th_imag)

        temp = np.einsum('ij,ja->ia', prop_re, th_imag)
        th_imag = contract('ia,ij,ja->i', self.tnif.mps[0], iden, temp)    
        temp = np.einsum('i,ij->j', prop_fin_arr[0], m_A)
        t_corr[self.N] = np.einsum('i,i', temp, th_imag)
        
        print("propagate middle parts+contraction: ", time()-tt)
        
        return t_corr
    

    def make_pt(self):
        '''
        Constructs the full process tensor/influence functional on the Kadanoff-Baym contour, without contracting over indices on the pure-imaginary times on the contour.
        '''
        self.tnif = self._get_init_mps(self.N)
        
        for n in range(1, self.N+self.M):
            
            self.tnif.contract_zipup(self._get_mpo_block(self.N+self.M-1, n), self.percentage)

        return

    def make_thermpt(self, opA: np.ndarray, opB: np.ndarray):
        '''
        Constructs the partial process tensor/influence functional on the Kadanoff-Baym contour, contracting over indices on the pure-imaginary time parts of the contour. Operators are inserted to allow for the extraction of the **thermal** correlation function ⟨A(t) B⟩ over N steps on the real-time interval defined for the TEMPO calculation.

        Parameters
        ----------
        opA: ndarray
            System observable for measurement
        opB: ndarray
            System observable for measurement
        '''

        dl = self.dl
        iden = np.eye(dl, dtype=complex)

        expdb = linalg.expm(-self.db*self.h_0/2)
        expdt = linalg.expm(1j*self.dt*self.h_0/2)

        k3_t = (expdb@expdb).flatten()
        k3v_t = np.diag(k3_t * ([1.0+0.0j] * dl))
        k3_t = np.einsum('ii->i', k3v_t)
        prop_im = np.kron((expdb@expdb).T, expdb@expdb).T
        #m_C = np.kron((expdt@expdb).T, (expdb@self.opC@(expdt.conj().T))).T
        m_C = np.kron((expdt@expdb).T, (expdb@opB@(expdt.conj().T))).T

        print("initializing mps")
        self.tnif = self._get_init_mps(self.N)
        print("finished initializing mps")
        
        for n in range(1, self.N+self.M):

            tt = time()

            # TODO: don't generate the whole list of MPOs before applying; make them on the fly
            self.tnif.contract_zipup(self._get_mpo_block(self.N+self.M-1, n), self.percentage)
            # Orthogonality center at len(mps)-1

            if n == 1:
                self.tnif.mps[-1] = contract('ia,ij,j->ia', self.tnif.mps[-1], iden, k3_t)
                self.tnif.mps[-2] = contract('iab,ij,jk,kb->ia', self.tnif.mps[-2], iden, prop_im, self.tnif.mps[-1])
                self.tnif.pop()
            elif n < self.M:
                self.tnif.mps[-2] = contract('iab,ij,jk,kb->ia', self.tnif.mps[-2], iden, prop_im, self.tnif.mps[-1])
                self.tnif.pop()

            elif n == self.M:
                self.tnif.mps[-1] = np.einsum('ij,ja->ia', m_C, self.tnif.mps[-1]) 
                self.tnif.mps[-2] = np.einsum('iab,ij,jb->ia',self.tnif.mps[-2], iden, self.tnif.mps[-1])
                self.tnif.pop()

        return

    def get_therm0(self, opA: np.ndarray, opB: np.ndarray) -> Tuple[complex, complex]:
        '''
        Computes the t=0 value of the **thermal** correlation function ⟨A(0) B⟩.

        Parameters
        ----------
        opA: ndarray
            System observable for measurement
        opB: ndarray
            System observable for measurement

        Returns
        -------
        th_if: complex
            Equal time thermal correlation, without normalization by the partition function
        z_if: complex
            Normalized partition function Z_{full} / Z_{bath}
        '''

        dh = self.dh
        dl = self.dl

        iden = np.eye(dl, dtype=complex)
        endcap = np.zeros(dl, dtype=complex)
        for i in range(0, dh):
            endcap[i*(dh+1)] = complex(1)
        
        expdb = linalg.expm(-self.db*self.h_0/2)

        k3_t = (expdb@expdb).flatten()
        k3v_t = np.diag(k3_t * ([1.0+0.0j] * dl))
        k3_t = np.einsum('ii->i', k3v_t)
        prop_im = np.kron((expdb@expdb).T, expdb@expdb).T
        m_C0 = np.kron((expdb).T, (expdb@opB@opA)).T #double-check order: CA or AC
        m_z = np.kron((expdb).T, expdb).T
        
        self.tnif = self._get_init_mps(0)
        
        for n in range(1, self.M):

            self.tnif.contract_zipup(self._get_mpo_block(self.M-1, n), self.percentage)
            
            if n == 1:
                self.tnif.mps[-1] = contract('ia,ij,j->ia', self.tnif.mps[-1], iden, k3_t)
                self.tnif.mps[-2] = contract('iab,ij,jk,kb->ia', self.tnif.mps[-2], iden, prop_im, self.tnif.mps[-1])
                self.tnif.pop()
            elif n < self.M-1:
                self.tnif.mps[-2] = contract('iab,ij,jk,kb->ia', self.tnif.mps[-2], iden, prop_im, self.tnif.mps[-1])
                self.tnif.pop()
            elif n == self.M-1:
                self.tnif.mps[-2] = contract('ib,ij,jk,kb->i', self.tnif.mps[-2], iden, prop_im, self.tnif.mps[-1])
                self.tnif.pop()
            
        th_if = np.einsum('ij,j->i', m_C0, self.tnif.mps[-1])
        z_if = np.einsum('ij,j->i', m_z, self.tnif.mps[-1])

        return np.dot(th_if, endcap), np.dot(z_if, endcap)
    
    def make_sympt(self, opA: np.ndarray, opB: np.ndarray):
        '''
        Constructs the partial process tensor/influence functional on the Kadanoff-Baym contour, contracting over indices on the pure-imaginary time parts of the contour. Operators are inserted to allow for the extraction of the **symmetrized** correlation function ⟨A(t-iβ/2) B⟩ over N steps on the real-time interval defined for the TEMPO calculation.

        Parameters
        ----------
        opA: ndarray
            System observable for measurement
        opB: ndarray
            System observable for measurement
        '''

        dl = self.dl
        iden = np.eye(dl, dtype=complex)

        expdb = linalg.expm(-self.db*self.h_0/2)
        expdt = linalg.expm(1j*self.dt*self.h_0/2)

        k3_s = (expdb@opB@expdb).flatten()
        k3v_s = np.diag(k3_s * ([1.0+0.0j] * dl))
        k3_s = np.einsum('ii->i', k3v_s)
        connect = np.kron((expdb@expdt).T, (expdb@(expdt.conj().T))).T #same as m_C, with C = iden
        prop_im = np.kron((expdb@expdb).T, expdb@expdb).T
                
        print("initializing mps")
        self.tnif = self._get_init_mps(self.N)
        print("finished initializing mps")
        
        for n in range(1, self.N+self.M):

            tt = time()
            
            self.tnif.contract_zipup(self._get_mpo_block(self.N+self.M-1, n), self.percentage)
            
            if n == 1:
                self.tnif.mps[-1] = contract('ia,ij,j->ia', self.tnif.mps[-1], iden, k3_s)
                self.tnif.mps[-2] = contract('iab,ij,jk,kb->ia', self.tnif.mps[-2], iden, prop_im, self.tnif.mps[-1])
                self.tnif.pop()
            elif n < self.M:
                self.tnif.mps[-2] = contract('iab,ij,jk,kb->ia', self.tnif.mps[-2], iden, prop_im, self.tnif.mps[-1])
                self.tnif.pop()

            elif n == self.M:
                self.tnif.mps[-1] = np.einsum('ij,ja->ia', connect, self.tnif.mps[-1])
                self.tnif.mps[-2] = np.einsum('iab,ij,jb->ia',self.tnif.mps[-2], iden, self.tnif.mps[-1])
                self.tnif.pop()

        return

    def get_sym0(self, opA: np.ndarray, opB: np.ndarray) -> complex:
        '''
        Computes the t=0 value of the **symmetrized** correlation function ⟨A(-iβ/2) B⟩.

        Parameters
        ----------
        opA: ndarray
            System observable for measurement
        opB: ndarray
            System observable for measurement

        Returns
        -------
        sy_if: complex
            Symmetrized correlation at zero time, without normalization by the partition function
        '''
        dh = self.dh
        dl = self.dl
        iden = np.eye(dl, dtype=complex)
        endcap = np.zeros(dl, dtype=complex)
        for i in range(0, dh):
            endcap[i*(dh+1)] = complex(1)

        expdb = linalg.expm(-self.db*self.h_0/2)

        k_B = (expdb@opB@expdb).flatten()
        k_Bv = np.diag(k_B * ([1.0+0.0j] * dl))
        k_B = np.einsum('ii->i', k_Bv)
        m_A0 = np.kron((opA@expdb).T, expdb).T

        prop_im = np.kron((expdb@expdb).T, expdb@expdb).T
        
        
        print("initializing mps")
        self.tnif = self._get_init_mps(0)
        print("finished initializing mps")
        
        for n in range(1, self.M):

            self.tnif.contract_zipup(self._get_mpo_block(self.M-1, n), self.percentage)

            if n == 1:
                self.tnif.mps[-1] = contract('ia,ij,j->ia', self.tnif.mps[-1], iden, k_B)
                self.tnif.mps[-2] = contract('iab,ij,jk,kb->ia', self.tnif.mps[-2], iden, prop_im, self.tnif.mps[-1])
                self.tnif.pop()
            elif n < self.M-1:
                self.tnif.mps[-2] = contract('iab,ij,jk,kb->ia', self.tnif.mps[-2], iden, prop_im, self.tnif.mps[-1])
                self.tnif.pop()
            elif n == self.M-1:
                self.tnif.mps[-2] = contract('ib,ij,jk,kb->i', self.tnif.mps[-2], iden, prop_im, self.tnif.mps[-1])
                self.tnif.pop()
            
        self.tnif.mps[-1] = np.einsum('ij,j->i', m_A0, self.tnif.mps[-1])

        return np.dot(self.tnif.mps[0], endcap)
