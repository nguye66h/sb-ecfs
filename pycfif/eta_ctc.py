from abc import ABC, abstractmethod
import numpy as np
from pycfif.bath import Bath
from pycfif.sim_params import SimulationParamsCTC

class Eta_CTC(ABC):
    '''
    Abstract class defining the structure for calculating eta's on the complex-time contour

    Attributes
    ----------
    N: int
        Number of timesteps to take along one leg of the contour
    dt: real
        Real part of the complex timestep along the contour
    db: real
        Imaginary part of the complex timestep along the contour
    t: float
        Maximum real time of the contour
    jw: Callable[[float], float]
        Spectral density of the spin-boson problem
    wc: float
        Frequency cutoff scale of the spectral density
    '''
    def __init__(self, bath: Bath, sp: SimulationParamsCTC):
        self.t = sp.t_list[-1]
        self.b = 1/sp.T
        self.N = int(np.ceil(np.abs(self.t - 1j*self.b/2) / sp.maxdtau))
        
        self.dt = self.t/self.N
        self.db = self.b/(2*self.N)

        self.jw = bath.jw
        self.wc = bath.wc

    def tau_converter(self, k: int) -> complex:
        '''
        Computes the (complex) time at the given number of steps along the complex-time contour
        
        Parameters
        ----------
        k: int
            Number of steps along the contour, ranging from 0 to 2*N

        Returns
        -------
        t_c: complex
            (Complex) time at the given point along the contour
        '''
        
        if k <= self.N:
            return (self.dt + 1j*self.db)*k
        else:
            return self.dt*(-k + 2*self.N) + 1j*(k*self.db)

    @abstractmethod
    def eta_k(self, k: int):
        raise NotImplementedError()

    @abstractmethod
    def eta_kk(self, k: int, kp: int):
        raise NotImplementedError()
