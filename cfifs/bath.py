from abc import ABC, abstractmethod
from typing import Callable

class Bath(ABC):

    def __init__(self, alpha: float, s: float, wc: float, jw: Callable[[float], float]):

        self.alpha = alpha
        self.s = s
        self.wc = wc

        self.jw = jw

