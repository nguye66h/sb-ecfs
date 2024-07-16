import numpy as np
from typing import List
from time import time
from scipy import linalg
from scipy.linalg import expm
from opt_einsum import contract

from pycfif.bath import Bath
from pycfif.sim_params import SimulationParamsCTC
from pycfif.eta_ctc import Eta_CTC
from pycfif.inflfn import IF_CTC as influence_functional
from pycfif.utilities import svd_truncate, reshape_qr, reshape_rq
from pycfif.mps import MPS

class TEMPO_CTC():
    '''
    Defines a TEMPO calculation along the complex-time contour (CTC)

    Attributes
    ----------
    N: int
        Number of timesteps to take along one leg of the contour
    dtau: complex
        Complex timestep along the contour
    h_0: np.ndarray
        System Hamiltonian
    dh: int
        Dimension of the system Hilbert space
    percentage: float
        Truncation threshold
    maxdim: int
        Maximum bond dimension of the MPS
    ifn: influence_functional
        Object storing the analytic influence functional for the given quadratic bath of bosons
    tnif: List[np.ndarray]
        Matrix product state representation of the influence functional
    ind_arr: List[int]
        List of indices specifying the remaining active sites of the MPS IF
    '''
    
    def __init__(self, bath: Bath, sp: SimulationParamsCTC, eta_obj: Eta_CTC):

        assert isinstance(sp, SimulationParamsCTC)
        assert issubclass(type(eta_obj), Eta_CTC)
        
        self.N = int(np.ceil(np.abs(sp.t_list[-1] - 1j/(2*sp.T)) / sp.maxdtau))
        self.dtau = (sp.t_list[-1] - 1j/(2*sp.T))/self.N

        self.h_0 = sp.H_S
        self.dh = sp.H_S.shape[0]

        self.percentage = sp.cutoff
        self.maxdim = sp.maxdim

        eta_k = np.zeros(2*self.N + 2, dtype=complex)
        eta_kk = np.zeros((2*self.N+2, 2*self.N+2), dtype=complex)

        tt = time()
        ek = eta_obj.eta_k(0)
        for i in range(0, 2*self.N+1):
            for j in range(1, i):
                eta_kk[i, j] = eta_obj.eta_kk(i,j)

        for i in range(0, self.N+1):
            eta_k[i] = ek
        for i in range(self.N+1, 2*self.N+1):
            eta_k[i] = np.conj(ek)

        print("eta time:", time()-tt)
        
        self.ifn = influence_functional(eta_k, eta_kk)

        self.tnif = None
        self.ind_arr = np.arange(2*self.N+1,-1,-1)        


    def get_init_mps(self, opB: np.ndarray) -> MPS:
        '''
        Constructs the initial matrix product state from the system-only propagations and final measurement

        Parameters
        ----------
        opB: ndarray
            Observable to be measured
        
        Returns
        -------
        mps: MPS
            Matrix product state on which the influence functional will be built
        '''

        m_arr = []

        Uprop_half = expm(-1j*self.h_0*self.dtau/2).T
        Uprop_f = Uprop_half @ Uprop_half
        
        q, r = reshape_qr(Uprop_half, [0], [1])
        rT = r.T

        m_arr.append(q)

        for _ in range(self.N-1):
            q, r = reshape_qr(Uprop_f, [0], [1])
            m_arr.append(np.einsum('sa,sb->sab', rT, q))
            rT = r.T

        Bprop = Uprop_half @ opB.T @ expm(1j*self.h_0*self.dtau.conjugate()/2).T

        q, r = reshape_qr(Bprop, [0], [1])
        m_arr.append(np.einsum('sa,sb->sab', rT, q))
        rT = r.T

        Uprop_half_b = expm(1j*self.h_0*self.dtau.conjugate()/2).T
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

    
    def get_MPO(self, ind: int) -> List[np.ndarray]:
        '''
        Constructs the layer of MPOs that introduces temporal correlations between the specified site and all other remaining sites of the MPS IF

        Parameters
        ----------
        ind: int
            Site (0-indexed) of the MPS
        
        Returns
        -------
        mpo: List[np.ndarray]
            List of matrix product operators
        '''

        mpo = []
        iden = np.eye(self.dh)
        ind_arr = self.ind_arr

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
        '''
        Calculate the symmetrized correlation function between A and B for the time initially specified in the definition of the TEMPO_CTC object

        Parameters
        ----------
        opA: ndarray
            System observable at the later time
        opB: ndarray
            System observable at the initial time
        
        Returns
        -------
        res: complex
            Value of the symmetrized correlation function at the specified time
        '''
        self.ind_arr = np.arange(2*self.N+1,-1,-1)
        
        self.tnif = self.get_init_mps(opB)
        
        contract_time = 0.0
        mpo_time = 0.0
        decim_time = 0.0

        while len(self.ind_arr) > 2:

            i = len(self.ind_arr)//2
            tt = time()
            mpo = self.get_MPO(self.ind_arr[i])
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
