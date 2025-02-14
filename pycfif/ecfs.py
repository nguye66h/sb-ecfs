import numpy as np
from typing import Union, Type
#import multiprocessing as mp

from pycfif.bath import Bath
from pycfif.sim_params import SimulationParams, SimulationParamsKBC, SimulationParamsSS, SimulationParamsCTC
from pycfif.eta_ctc import Eta_CTC
from pycfif.eta_kbc import Eta_KBC
from pycfif.tempo_pt import TEMPO_PT
from pycfif.tempo_ss import TEMPO_SS
from pycfif.tempo_ctc import TEMPO_CTC

class ECFs:
    '''
    Class to define a single simulation for a correlation function

    Attributes
    ----------
    bath: Bath
        Object specifying the type of bath, as defined by the associated spectral density
    sp: SimulationParams
        Object carrying information on the parameters of the calculation for the correlation function
    eta_class: Union[Type[Eta_KBC], Type[Eta_CTC]]
        Class defining objects suitable for calculating the eta values that define the influence functional on a given type of contour
    '''
    def __init__(self, bath: Bath, sp: SimulationParams, eta_class: Union[Type[Eta_KBC], Type[Eta_CTC]]):

        self.bath = bath
        self.sp = sp
        self.eta_class = eta_class

    def run(self) -> np.ndarray:
        '''
        Sets up the relevant calculations for the influence functional and returns the specified correlation function

        Returns
        -------
        res: ndarray
            Correlation function evaluated at specified times
        '''
        sp = self.sp
        t_list = sp.t_list
        
        if isinstance(sp, SimulationParamsKBC):

            assert issubclass(self.eta_class, Eta_KBC), "Simulations on the Kadanoff-Baym contour require an eta object defined on the same contour"
            eta_obj = self.eta_class(self.bath, sp)
            if_kbc = TEMPO_PT(self.bath, sp, eta_obj)
            
            if sp.thermal_corr == True:

                if_kbc.make_thermpt(sp.opA, sp.opB)
                res_corr = if_kbc.get_correlator(sp.opA, sp.opB)
                c0, z = if_kbc.get_therm0(sp.opA, sp.opB)
                res_corr[0] = c0
                
                return np.column_stack((t_list, res_corr / z))

            else:

                if_kbc.make_sympt(sp.opA, sp.opB)
                res_corr = if_kbc.get_correlator(sp.opA, sp.opB)
                _, z = if_kbc.get_therm0(sp.opA, sp.opB)
                c0 = if_kbc.get_sym0(sp.opA, sp.opB)
                res_corr[0] = c0
                
                return np.column_stack((t_list, res_corr / z))
            
        elif isinstance(sp, SimulationParamsCTC):

            assert issubclass(self.eta_class, Eta_CTC), "Simulations on the complex-time contour require an eta object defined on the same contour"
            
            res_corr = np.zeros(len(t_list), dtype=complex)
            z = complex(0)
            iden = np.eye(sp.H_S.shape[0])

            # TODO: parallelize
            for n in range(0, len(t_list)):

                sp_n = SimulationParamsCTC(
                                       sp.maxdtau,
                                       sp.T,
                                       [t_list[n]],
                                       sp.H_S,
                                       sp.opA, sp.opB,
                                       sp.cutoff, sp.maxdim)
                eta_obj = self.eta_class(self.bath, sp_n)
                if_ctc = TEMPO_CTC(self.bath, sp_n, eta_obj)
                res_corr[n] = if_ctc.propagate(sp.opA, sp.opB)# / if_ctc.propagate(iden, iden)

                if n == 0:
                    z = if_ctc.propagate(iden, iden)

            return np.column_stack((t_list, res_corr / z))#np.column_stack((t_list, res_corr))

        elif isinstance(sp, SimulationParamsSS):

            assert issubclass(self.eta_class, Eta_KBC), "Steady state simulations require an eta object defined on the Kadanoff-Baym contour"

            eta_obj = self.eta_class(self.bath, sp)
            if_ss = TEMPO_SS(self.bath, sp, eta_obj)
            if sp.alg == "mbh_tebd":
                print("Using Hastings iTEBD with nc = ", sp.N)
                if_ss.make_finf_mbh()
            elif sp.alg == "ov_tebd":
                print("Using Orus-Vidal iTEBD with nc =", sp.N)
                if_ss.make_finf_ov()

            res_corr = if_ss.get_correlator(sp.opA, sp.opB)
            
            return np.column_stack((t_list, res_corr))

        else:

            raise RuntimeError("Invalid simulation specification")
