import numpy as np
#import multiprocessing as mp

from cfifs.bath import Bath
from cfifs.sim_params import SimulationParams
from cfifs.tempo_pt import TEMPO_PT
from cfifs.tempo_ctc import TEMPO_CTC

class ECFs:

    def __init__(self, bath: Bath, sp: SimulationParams):

        self.bath = bath
        self.sp = sp

    def run(self) -> np.ndarray:

        sp = self.sp
        t_list = sp.t_list

        if sp.corr_type == 0:

            if sp.contour == 0:

                if_kbc = TEMPO_PT(self.bath, sp)
                if_kbc.make_thermpt(sp.opA, sp.opB)
                res_corr = if_kbc.get_correlator(sp.opA, sp.opB)
                c0, z = if_kbc.get_therm0(sp.opA, sp.opB)
                res_corr[0] = c0
                
                return np.column_stack((t_list, res_corr / z))
            
            elif sp.contour == 1:

                raise RuntimeError()# TEMPO_SS

            else:

                raise RuntimeError("Invalid contour specification")
            
        elif sp.corr_type == 1:

            if sp.contour == 0:

                if_kbc = TEMPO_PT(self.bath, sp)
                if_kbc.make_sympt(sp.opA, sp.opB)
                res_corr = if_kbc.get_correlator(sp.opA, sp.opB)
                _, z = if_kbc.get_therm0(sp.opA, sp.opB)
                c0 = if_kbc.get_sym0(sp.opA, sp.opB)
                res_corr[0] = c0
                
                return np.column_stack((t_list, res_corr / z))

            elif sp.contour == 2:

                res_corr = np.zeros(len(t_list), dtype=complex)
                z = complex(0)
                iden = np.eye(sp.H_S.shape[0])
                
                # TODO: parallelize
                for n in range(0, len(t_list)):

                    if_ctc = TEMPO_CTC(self.bath,
                                       SimulationParams(
                                           sp.corr_type, sp.contour, sp.N, 0,
                                           sp.T,
                                           [t_list[n]],
                                           sp.H_S,
                                           sp.opA, sp.opB,
                                           sp.cutoff, sp.maxdim)
                                       )
                    res_corr[n] = if_ctc.propagate(sp.opA, sp.opB)

                    if n == 0:
                        z = if_ctc.propagate(iden, iden)

                return np.column_stack((t_list, res_corr / z))
            
            else:

                raise RuntimeError("Invalid contour specification")

        else:

            raise RuntimeError("Invalid correlation function type")
