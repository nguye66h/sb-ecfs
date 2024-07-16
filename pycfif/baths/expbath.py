import numpy as np
from pycfif.bath import Bath

class ExponentialBath(Bath):
    '''
    Abstract class defining baths with exponentially cut-off spectral densities of the form J(ω) = (α/2) * ω_c * (ω/ω_c)^s * exp(-ω/ω_c). Inherits attributes from the abstract Bath class.

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
    def __init__(self, alpha: float, s: float, wc: float):

        # J(w) defined only for w >= 0
        jw = lambda w: (alpha/2) * wc * (np.abs(w/wc)**s) * np.exp(-np.abs(w)/wc)
        
        super().__init__(alpha, s, wc, jw)


