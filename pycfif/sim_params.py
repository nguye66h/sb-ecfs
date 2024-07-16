from abc import ABC
from typing import Optional, Union, List
import numpy as np

class SimulationParams(ABC):

    def __init__(self):

        return

class SimulationParamsKBC(SimulationParams):
    '''
    Class defining the set of parameters required for the calculation of correlation functions on the Kadanoff-Baym-like contour

    Attributes
    ----------
    thermal_corr: bool
        Specifies whether to measure the thermal (True) or symmetrized (False) correlation function on the Kadanoff-Baym-like contour
    N: int
        Number of timesteps along the real-time part of the contour
    M: int
        Number of timesteps along the half of the imaginary-time part of the contour
    T: float
        Temperature of the equilibrium ensemble
    t_max: float
        Maximum (real) time difference in the measurement of the correlation function
    t_list: ndarray
        List of real times at which to measure the correlation function
    dt: float
        Timestep along the real-time part of the contour
    cutoff: float
        Truncation threshold
    maxdim: int
        Maximum bond dimension
    H_S: ndarray
        System Hamiltonian
    opA: ndarray
        System observable for measurement
    opB: ndarray
        System observable for measurement
    '''

    def __init__(self, thermal_corr: bool, N: int, M: int, T: float, 
                       t_max: float,
                       H_S: np.ndarray,
                       opA: np.ndarray, opB: np.ndarray,
                       cutoff: float, maxdim: Optional[int] = 5000
    ):

        # thermal_corr = True if thermal
        #              = False if symmetrized
        self.thermal_corr = thermal_corr

        self.N = N
        self.M = M
        
        self.T = T

        self.t_max = t_max
        self.t_list = np.linspace(0.0, t_max, N+1)
        self.dt = t_max / N
        
        self.cutoff = cutoff
        self.maxdim = maxdim

        self.H_S = H_S
        self.opA = opA
        self.opB = opB

        
class SimulationParamsSS(SimulationParams):
    '''
    Class defining the set of parameters required for the calculation of steady state correlation function

    Attributes
    ----------
    N: int
        Total memory length to include into the calculation of the steady state influence functional, given by the memory time `tmem` divided by the timestep size `dt`. Defaults to an `N` such that the memory time `tmem` is 400 in units of the tunnelling matrix element.
    T: float
        Temperature of the initial bath state, assumed to equal the temperature of the steady state
    dt: float
        Timestep along the real-time part of the contour
    t_max: float
        Maximum (real) time difference in the measurement of the correlation function
    n_sim: int
        Number of times at which to measure the correlation function
    t_list: ndarray
        List of real times, separated by `dt`, at which to measure the correlation function
    cutoff: float
        Truncation threshold
    maxdim: int
        Maximum bond dimension
    H_S: ndarray
        System Hamiltonian
    opA: ndarray
        System observable for measurement
    opB: ndarray
        System observable for measurement
    alg: str
        Algorithm to use for iTEBD. Can be "mbh_tebd", "ov_tebd", or "qr_tebd". Default="mbh_tebd"
    tcut: float
        Timescale over which a smooth cutoff starts to take effect. Default=None
    '''

    def __init__(self, T: float, 
                       dt: float,
                       t_max: float,
                       H_S: np.ndarray,
                       opA: np.ndarray, opB: np.ndarray,
                       cutoff: float, maxdim: Optional[int] = 5000,
                       alg: Optional[str] = "mbh_tebd",
                       tmem: Optional[float] = 400.0,
                       tcut: Optional[float] = None
    ):

        self.T = T

        self.dt = dt
        self.t_max = t_max
        self.n_sim = int(np.ceil(t_max / dt))
        self.t_list = dt * np.linspace(0, self.n_sim, self.n_sim+1)
        
        self.cutoff = cutoff
        self.maxdim = maxdim

        self.H_S = H_S
        self.opA = opA
        self.opB = opB

        self.alg = alg
        self.N = int(np.ceil(tmem / dt))
        self.M = 1
        self.tcut = tcut


class SimulationParamsCTC(SimulationParams):
    '''
    Class defining the set of parameters required for the calculation of symmetrized correlation function on the complex-time contour

    Attributes
    ----------
    maxdtau: float
        Maximum modulus of the complex timestep along the complex-time contour
    T: float
        Temperature of the equilibrium ensemble
    t_list: ndarray
        List of real times at which to measure the correlation function
    cutoff: float
        Truncation threshold
    maxdim: int
        Maximum bond dimension
    H_S: ndarray
        System Hamiltonian
    opA: ndarray
        System observable for measurement
    opB: ndarray
        System observable for measurement
    '''
    def __init__(self, maxdtau: float, T: float, 
                       t_list: Union[np.ndarray, List[float]],
                       H_S: np.ndarray,
                       opA: np.ndarray, opB: np.ndarray,
                       cutoff: float, maxdim: Optional[int] = 5000
    ):

        self.maxdtau = maxdtau
        
        self.T = T

        self.t_list = t_list
        
        self.cutoff = cutoff
        self.maxdim = maxdim

        self.H_S = H_S
        self.opA = opA
        self.opB = opB
