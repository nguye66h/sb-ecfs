from abc import ABC, abstractmethod

class Eta_KBC(ABC):

    def __init__(self, bath: Bath, sp: SimulationParams):
        self.t = sp.t_max
        self.b = 1/sp.T
        self.N = sp.N
        self.M = sp.M

        self.dt = self.t / self.N
        self.db = self.b / (2*self.M)
        self.jw = bath.jw
        self.wc = bath.wc

    @abstractmethod
    def eta_pp_tt_kk(self, d: int):
        raise NotImplementedError()

    @abstractmethod
    def eta_pm_tt_kk(self, d: int):
        raise NotImplementedError()

    @abstractmethod
    def eta_pm_tt_k(self):
        raise NotImplementedError()

    @abstractmethod
    def eta_pp_tt_k(self):
        raise NotImplementedError()

    ###############

    @abstractmethod
    def eta_pp_bb_kk(self, d: int):
        raise NotImplementedError()

    @abstractmethod
    def eta_pm_bb_kk(self, sum_kkp: int):
        raise NotImplementedError()

    @abstractmethod
    def eta_pp_bb_k(self):
        raise NotImplementedError()

    @abstractmethod
    def eta_pm_bb_k(self, k: int):
        raise NotImplementedError()
    
    ###############
    
    @abstractmethod
    def eta_pp_mix_kk(self, k: int, kp: int):
        raise NotImplementedError()

    @abstractmethod
    def eta_pm_mix_kk(self, k: int, kp: int):
        raise NotImplementedError()
