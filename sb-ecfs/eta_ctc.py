from abc import ABC, abstractmethod

class Eta_CTC(ABC):

    def __init__(self, bath: Bath, sp: SimulationParams):
        self.t = sp.t_max
        self.b = 1/sp.T
        self.N = sp.N

        self.dt = self.t/self.N
        self.db = self.b/(2*self.N)

        self.jw = bath.jw
        self.wc = bath.wc

    def tau_converter(self,k):

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
