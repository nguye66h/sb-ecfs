from abc import ABC, abstractmethod

class Bath(ABC):

    def __init__(self, T: float,
                       alpha: float, s: float, wc: float):

        self.T = T
        self.alpha = alpha
        self.s = s
        self.wc = wc

