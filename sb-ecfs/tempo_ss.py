import numpy as np
from scipy import integrate
#from scipy.integrate import dblquad
!!!!!! from expbath_eta_kbc import eta
from scipy.linalg import expm, norm, svd, eigh
from scipy.sparse.linalg import eigs, LinearOperator
from typing import Callable, Optional, Union
#from tqdm import tqdm
from ncon import ncon

class Simple_TEBD():

    def __init__(self, sp: SimulationParams):
        self.percentage = sp.cutoff
        self.maxdim = sp.maxdim

    def _apply_gate(self, gate: np.ndarray, a: np.ndarray, s_ab: np.ndarray, b: np.ndarray, s_ba: np.ndarray):
        return

class TEMPO_SS():

    def __init__(self, bath: Bath, sp: SimulationParams):
        self.N = sp.N
        self.dt = sp.t_max/sp.N
        self.percentage = 10**(-r)

        self.h_0 = sp.H_S
        self.dh = sp.H_S.shape[0]
        self.dl = (self.dh)**2

        self.opA = sp.opA
        self.opB = sp.opB

        self.eta = eta(bath, sp)
        eta_t = np.zeros(self.N+1, dtype=complex)
        eta_t[0] = self.eta.eta_pp_tt_k()
        
        for k in range(1, self.N+1):
            eta_t[k] = self.eta.eta_pp_tt_kk(k)        

    def make_f_inf(self, ):

        # a tensor, b tensor
        
        driver._apply_gate(gate, a, s_ab, b, s_ba)
            
    def get_corrfx(self, h_s: np.ndarray, op_0: np.ndarray, op_1: np.ndarray, n: int) -> np.ndarray:
        """
        Compute steady state correlation function <A_1(t) A_0(0)> for n time steps.

        :param h_s: System Hamiltonian in the eigenbasis of the coupling operator.
        :param rho_0: System initial state in the eigenbasis of the coupling operator.
        :param n: Number of time-steps for the propagation.
        :return: Time evolution of density matrix.
        """
        
        liu_s = np.kron(expm(-1j * h_s * self.delta / 2), expm(1j * h_s * self.delta / 2))
        #liu_s = np.kron(expm(-1j * h_s * self.delta / 2), expm(1j * h_s * self.delta / 2).T)
        u = np.einsum('ab,bc->abc', liu_s.T, liu_s.T)

        A_0 = np.kron(op_0, np.eye(self.s_dim, dtype=op_0.dtype))
        A_1 = np.kron(op_1, np.eye(self.s_dim, dtype=op_1.dtype))

        G_t = np.empty(n+1, dtype=np.complex128)

        evol_tens = ncon([self.f[:, :, :], u], [[-1, 2, -3], [-2, 2, -4]])        
        left_state, right_state, ev = self._full_steadystate(h_s)
        evol_tens = evol_tens / ev
        normalization = ncon([left_state, right_state], [[1, 2], [1, 2]])

        left_vec = np.einsum('bs,sc->bc', left_state, A_0)
        right_vec = right_state
        print(normalization)
        
        print("left vec shape:", left_vec.shape)
        print("A_1 shape:", A_1.shape)

        G_t[0] = ncon([left_vec, A_1, right_vec], [[1, 2], [2, 3], [1, 3]])
        
#        for i in tqdm(range(n), desc='time evolution running'):
        for i in range(n):
            left_vec = ncon([left_vec, evol_tens], [[1, 2], [1, 2, -2, -3]])
            G_t[i + 1] = ncon([left_vec, A_1, right_vec], [[1, 2], [2, 3], [1, 3]])
        return G_t / normalization
    
    def _full_steadystate(self, h_s: np.ndarray) -> np.ndarray:
        """
        Compute the steady state using Lanczos.

        :param h_s: System Hamiltonian in the eigenbasis of the coupling operator.
        :return: Steady state density matrix.
        """

        liu_s = np.kron(expm(-1j * h_s * self.delta / 2), expm(1j * h_s * self.delta / 2).T)
        #liu_s = np.kron(expm(-1j * h_s * self.delta / 2), expm(1j * h_s * self.delta / 2).T)
        u = np.einsum('ab,bc->abc', liu_s.T, liu_s.T)

        evol_tens = ncon([self.f[:, :, :], u], [[-1, 1, -3], [-2, 1, -4]])

        mat = evol_tens.reshape([evol_tens.shape[0] * evol_tens.shape[1], evol_tens.shape[2] * evol_tens.shape[3]]).T
        w_l, v_l = eigs(mat, 50, which='LM')
        l_max = np.argmax(np.abs(w_l))
        w_r, v_r = eigs(mat.T, 50, which='LM')
        r_max = np.argmax(np.abs(w_r))
        print("eigenvalue: ", w_l[l_max], ", ", w_r[r_max])
        
        res_bd = self.f.shape[0]
        rho_ss = v_l[:, l_max].reshape([res_bd, self.s_dim**2])
        tr_ss = v_r[:, r_max].reshape([res_bd, self.s_dim**2])

        return rho_ss, tr_ss, w_l[l_max]
        
    def get_transfer_tensor(self, h_s: np.ndarray) -> np.ndarray:
        """
        Compute the steady state using Lanczos.

        :param h_s: System Hamiltonian in the eigenbasis of the coupling operator.
        :return: Steady state density matrix.
        """

        #liu_s = np.kron(expm(-1j * h_s * self.delta / 2), expm(1j * h_s * self.delta / 2).T)
        liu_s = np.kron(expm(-1j * h_s * self.delta / 2), expm(1j * h_s * self.delta / 2))
        u = np.einsum('ab,bc->abc', liu_s.T, liu_s.T)

        evol_tens = ncon([self.f[:, :, :], u], [[-1, 1, -3], [-2, 1, -4]])

        mat = evol_tens.reshape([evol_tens.shape[0] * evol_tens.shape[1], evol_tens.shape[2] * evol_tens.shape[3]]).T

        return mat
