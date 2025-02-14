import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import eigs
from typing import Tuple
from opt_einsum import contract

from pycfif.bath import Bath
from pycfif.sim_params import SimulationParamsSS
from pycfif.eta_kbc import Eta_KBC
from pycfif.utilities import svd_truncate, reshape_qr, reshape_rq
from pycfif.mps import uMPS

class TEMPO_SS():
    '''
    Defines a TEMPO calculation for the steady state correlation function

    Attributes
    ----------
    dt: float
        Timestep along the real-time contour
    n_sim: int
        Number of timesteps over which to measure the steady state correlation function
    h_0: np.ndarray
        System Hamiltonian
    dh: int
        Dimension of the system Hilbert space
    dl: int
        Dimension of the system Liouville space
    percentage: float
        Truncation threshold
    states: ndarray
        List of eigenvalues of the system-only part of the system-bath coupling, presumed to equal 2*Sz, where Sz is the operator measuring the z-component of a spin-s degree of freedom, where s = (dh-1)/2
    o_mf: ndarray
        Representation of the (super)operator O^-_i * S^f_j, where O^-_i = S^f_i - S^b_i is the difference between forwards and backwards trajectories at time index i
    o_mb: ndarray
        Representation of the (super)operator O^-_i * S^b_j
    opA: ndarray
        System observable for measurement
    opB: ndarray
        System observable for measurement
    eta_t: ndarray
        List of eta coefficients defining the spin-boson influence functional, related to double integrals over the bath correlation function
    finf: np.ndarray
        Uniform matrix product state representation of the steady state influence functional
    '''

    def __init__(self, bath: Bath, sp: SimulationParamsSS, eta_obj: Eta_KBC):

        assert isinstance(sp, SimulationParamsSS)
        assert issubclass(type(eta_obj), Eta_KBC)
        
        self.dt = sp.dt
        self.n_sim = sp.n_sim
        self.percentage = sp.cutoff

        self.h_0 = sp.H_S
        self.dh = sp.H_S.shape[0]
        self.dl = (self.dh)**2

        # Note: *twice* the eigenvalues of Sz for spin-s
        self.states = np.linspace((self.dh-1), -(self.dh-1), self.dh)

        # log I_k = - (sf - sb) * (Reη O^- + 1j*Imη O^+)
        # log I_k = - η [(sf - sb) * sf] - η† [(sf - sb) * sb]

        # Inefficient way:
        # s_f = np.kron( np.diag(self.states), np.eye(dh) )
        # s_b = np.kron( np.eye(dh), np.diag(self.states) )
        # o_minus = s_f - s_b
        # o_plus = s_f + s_b

        # Store only diagonals, since all the operators are diagonal:
        s_f = np.repeat(self.states, self.dh) # a (dl x dl) representation of the operator S_f ⊗ I_b
        s_b = np.tile(self.states, self.dh) # I_f ⊗ S_b
        o_minus = s_f - s_b

        self.o_mf = np.outer(o_minus, s_f)
        self.o_mb = np.outer(o_minus, s_b)
        
        self.opA = sp.opA
        self.opB = sp.opB

        #eta_obj = eta(bath, sp)
        self.eta_t = np.zeros(sp.N+1, dtype=complex)
        self.eta_t[0] = eta_obj.eta_pp_tt_k()
        
        for k in range(1, sp.N+1):
            self.eta_t[k] = eta_obj.eta_pp_tt_kk(k)

        if sp.tcut != None:
            self.cut_etas(sp.tcut)
        
        self.finf = None

    def cut_etas(self, tr: float):
        '''
        Applies a smooth cutoff 
                          1
         f_k = ------------------------
                1 + exp(-(t_r - k*Δt))
        to the array of etas (i.e., eta_k *= f_k), with cutoff time tr

        Parameters
        -----------
        tr : float
            Timescale over which the cutoff starts to take effect
        '''
        for k in range(0, len(self.eta_t)):
            self.eta_t[k] = self.eta_t[k] / (1 + np.exp(-0.2*(tr - k*self.dt)))

        return

    def make_finf_ov(self):
        '''
        Makes the F_{\infty} tensor using the Orus-Vidal approach (arXiv:0711.3960)
        '''
        dl = self.dl
        
        a = np.ones((dl, 1, 1))
        b = np.ones((dl, 1, 1))
        s = np.ones(1, dtype=float)

        mps = uMPS([a, s, b, s], True)

        nc = len(self.eta_t)

        delta = np.eye(dl)
        for n in range(0, nc-1):
            e = self.eta_t[nc-1 - n]
            phi_k = -e * self.o_mf + np.conj(e) * self.o_mb
            I_k = np.exp(phi_k)
            #       i              i   y
            #       |              |   |
            #     |---|          |-------|
            #   x-|   |-y   =>   |       |
            #     |---|          |-------|
            #       |              |   |
            #       j              x   j
            # (x, y) = (α, α') in the tempo index naming convention
            gate = contract('xi,ij,xy->ixjy', I_k, delta, delta)
            mps.step_itebd_ov(gate, self.percentage)

        e = self.eta_t[0]
        # index i = x
        phi_k = -e * np.diagonal(self.o_mf) + np.conj(e) * np.diagonal(self.o_mb)
        I_k = np.exp(phi_k)
        #       i              i
        #       |              | 
        #     |---|        |-------|
        #   x-|   |   =>   |       |
        #     |---|        |-------|
        #       |            |   |
        #       j            x   j
        gate = contract('i,ix,ij->ixj', I_k, delta, delta)

        self.finf = contract('ixj,xab,b,jbc,c->iac', gate, mps.tensors[0], mps.tensors[1], mps.tensors[2], mps.tensors[3])

        print("Bond dim:", self.finf.shape[1])

        return
    
    def make_finf_mbh(self):
        '''
        Makes the F_{\infty} tensor using Hasting's modification of Orus-Vidal's iTEBD, which avoids inverting matrices.
        '''
        dl = self.dl
        
        a = np.ones((dl, 1, 1))
        b = np.ones((dl, 1, 1))
        s = np.ones(1, dtype=float)

        mps = uMPS([a, b], False)

        nc = len(self.eta_t)

        delta = np.eye(dl)
        for n in range(0, nc-1):
            e = self.eta_t[nc-1 - n]
            phi_k = -e * self.o_mf + np.conj(e) * self.o_mb
            I_k = np.exp(phi_k)
            #       i              i   y
            #       |              |   |
            #     |---|          |-------|
            #   x-|   |-y   =>   |       |
            #     |---|          |-------|
            #       |              |   |
            #       j              x   j
            # (x, y) = (α, α') in the tempo index naming convention
            gate = contract('xi,ij,xy->ixjy', I_k, delta, delta)
            s = mps.step_itebd_mbh(gate, s, self.percentage)

        e = self.eta_t[0]
        # index i = x
        phi_k = -e * np.diagonal(self.o_mf) + np.conj(e) * np.diagonal(self.o_mb)
        I_k = np.exp(phi_k)
        #       i              i
        #       |              | 
        #     |---|        |-------|
        #   x-|   |   =>   |       |
        #     |---|        |-------|
        #       |            |   |
        #       j            x   j
        gate = contract('i,ix,ij->ixj', I_k, delta, delta)

        self.finf = contract('ixj,xay,jyb->iab', gate, mps.tensors[0], mps.tensors[1])

        print("Bond dim:", self.finf.shape[1])

        return


    def get_correlator(self, op_0: np.ndarray, op_1: np.ndarray) -> np.ndarray:
        '''
        Computes the steady state correlation function.

        Parameters
        ----------
        op_0: ndarray
            dxd array corresponding to the earlier measurement, where d is the dimension of the Hilbert space of the system
        op_1: ndarray
            dxd array corresponding to the later measurement

        Returns
        -------
        res: ndarray
            Numpy array of the correlation function evaluated at multiples of Δt
        '''
        A_0 = np.kron(op_0, np.eye(self.dh, dtype=op_0.dtype))
        A_1 = np.kron(op_1, np.eye(self.dh, dtype=op_1.dtype))

        G_t = np.empty(self.n_sim+1, dtype=complex)

        evol_tens, left_state, right_state, ev = self._full_steadystate()
        evol_tens /= ev
        normalization = np.einsum('sb,sb', left_state, right_state)

        left_vec = np.einsum('sb,sc->cb', left_state, A_0)
        right_vec = np.einsum('sc,cb->sb', A_1, right_state)

        G_t[0] = np.einsum('sb,sb', left_vec, right_vec)
        
        for i in range(self.n_sim):
            left_vec = np.einsum('sb,sbta->ta', left_vec, evol_tens)
            G_t[i + 1] = np.einsum('sb,sb', left_vec, right_vec)

        return G_t / normalization
    
    def _full_steadystate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, complex]:
        '''
        Computes the left and right eigenvectors of the full propagation tensor for one timestep. The full propagation tensor is formed from the contraction of F_{\infty} with two copies of the system propagator over half of one timestep, corresponding to a symmetric Trotter splitting with local error O(Δt^3).

        Returns
        -------
        evol_tens: ndarray
            The four-legged evolution tensor
        rho_ss: ndarray
            The two-legged tensor corresponding to the maximal left eigenvector. The first and second indices respectively correspond to the system and bond spaces
        tr_ss: ndarray
            The two-legged tensor corresponding to the maximal right eigenvector, following the same index convention as rho_ss
        w: complex
            The eigenvalue of maximal modulus corresponding to the returned left and right eigenvectors
        '''
        h_s = self.h_0
        liou_s = np.kron(expm(-1j * h_s * self.dt / 2), expm(1j * h_s * self.dt / 2))
        u_s = np.einsum('ab,bc->abc', liou_s, liou_s)

        evol_tens = np.einsum('iab,sit->satb', self.finf, u_s)

        mat = evol_tens.reshape([evol_tens.shape[0] * evol_tens.shape[1], evol_tens.shape[2] * evol_tens.shape[3]]).T
        nvecs = max(100, int(np.ceil(0.1 * mat.shape[0])))

        w_l, v_l = eigs(mat, nvecs, which='LM')
        l_max = np.argmax(np.abs(w_l))
        w_r, v_r = eigs(mat.T, nvecs, which='LM')
        r_max = np.argmax(np.abs(w_r))
        #print("eigenvalue: ", w_l[l_max], ", ", w_r[r_max])
        
        res_bd = self.finf.shape[1]
        rho_ss = v_l[:, l_max].reshape([self.dl, res_bd])
        tr_ss = v_r[:, r_max].reshape([self.dl, res_bd])
        
        return evol_tens, rho_ss, tr_ss, w_l[l_max]
