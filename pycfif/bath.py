from abc import ABC
from typing import Callable

class Bath(ABC):
    '''
    Abstract class defining a bath spectral density, used for calculations of eta values defining the influence functional

    Attributes
    ----------
    alpha: float
        System-bath coupling strength
    s: float
        Exponent of the low-frequency part of the spectral density
    wc: float
        Frequency cutoff scale of the spectral density
    jw: Callable[[float], float]
        Spectral density of the spin-boson problem
    '''
    def __init__(self, alpha: float, s: float, wc: float, jw: Callable[[float], float]):

        self.alpha = alpha
        self.s = s
        self.wc = wc

        self.jw = jw

