from typing import Any, List, Callable, Optional, Text, Tuple, Union
import numpy as np

class SimulationParams:

    def __init__(self, corr_type: int, contour: int, N: int, M: int, T: float, 
                       t_list: np.ndarray,
                       H_S: np.ndarray,
                       opA: np.ndarray, opB: np.ndarray,
                       cutoff: float, maxdim: Optional[int] = 2000):

        # corr_type = 0 if thermal/steady state
        #           = 1 if symmetrized
        self.corr_type = corr_type
        # contour = 0 if Kadanoff-Baym-like
        #         = 1 if triangular/complex-time
        self.contour = contour
        self.N = N
        self.M = M
        
        # Temperature
        self.T = T

        # List of times at which the correlation function is measured
        self.t_list = t_list
        self.t_spacing = t_list[1] - t_list[0]
        self.t_max = t_list[-1]
        
        self.cutoff = cutoff
        self.maxdim = maxdim

        self.H_S = H_S
        self.opA = opA
        self.opB = opB
