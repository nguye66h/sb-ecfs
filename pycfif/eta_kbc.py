import numpy as np

from abc import ABC, abstractmethod
from pycfif.bath import Bath
from pycfif.sim_params import SimulationParamsKBC

class Eta_KBC(ABC):
    '''
    Abstract class defining the structure for calculating eta's on the Kadanoff-Baym-like contour

    Attributes
    ----------
    N: int
        Number of timesteps along the real-time part of the contour
    M: int
        Number of timesteps along the half of the imaginary-time part of the contour
    t: float
        Maximum real time of the contour
    dt: float
        Timestep along the real-time part of the contour
    b: float
        Inverse temperature of the equilibrium ensemble
    db: float
        Timestep along the imaginary-time part of the contour
    jw: Callable[[float], float]
        Spectral density of the spin-boson problem
    wc: float
        Frequency cutoff scale of the spectral density
    '''
    def __init__(self, bath: Bath, sp: SimulationParamsKBC):
        self.N = sp.N
        self.M = sp.M

        self.t = sp.t_max
        self.dt = sp.dt
        
        if sp.T == 0.0:
            self.b = np.inf
            self.db = np.inf
        else:
            self.b = 1/sp.T
            self.db = self.b / (2*self.M)
        
        self.jw = bath.jw
        self.wc = bath.wc

    @abstractmethod
    def eta_pp_tt_kk(self, d: int):
        raise NotImplementedError()

    @abstractmethod
    def eta_pm_tt_kk(self, d: int):
        raise NotImplementedError()

    @abstractmethod
    def eta_pm_tt_k(self):
        raise NotImplementedError()

    @abstractmethod
    def eta_pp_tt_k(self):
        raise NotImplementedError()

    ###############

    @abstractmethod
    def eta_pp_bb_kk(self, d: int):
        raise NotImplementedError()

    @abstractmethod
    def eta_pm_bb_kk(self, sum_kkp: int):
        raise NotImplementedError()

    @abstractmethod
    def eta_pp_bb_k(self):
        raise NotImplementedError()

    @abstractmethod
    def eta_pm_bb_k(self, k: int):
        raise NotImplementedError()
    
    ###############
    
    @abstractmethod
    def eta_pp_mix_kk(self, k: int, kp: int):
        raise NotImplementedError()

    @abstractmethod
    def eta_pm_mix_kk(self, k: int, kp: int):
        raise NotImplementedError()
