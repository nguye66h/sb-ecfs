from typing import Any
import numpy as np

class ExponentialBath(Bath):

    def __init__(self, T: float,
                       alpha: float, s: float, wc: float):
        super().__init__(T, alpha, s, wc)

        # J(w) defined only for w >= 0
        # NOTE TO SELF: THIS DEFINITION IS TWICE THAT OF HAIMI's
        self.jw = lambda w: (alpha/2) * wc * (np.abs(w/wc)**s) * np.exp(-np.abs(w)/wc)
