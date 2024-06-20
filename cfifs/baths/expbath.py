from typing import Any
import numpy as np
from cfifs.bath import Bath

class ExponentialBath(Bath):

    def __init__(self, alpha: float, s: float, wc: float):

        # J(w) defined only for w >= 0
        # NOTE TO SELF: THIS DEFINITION IS TWICE THAT OF HAIMI's
        jw = lambda w: (alpha/2) * wc * (np.abs(w/wc)**s) * np.exp(-np.abs(w)/wc)
        
        super().__init__(alpha, s, wc, jw)


