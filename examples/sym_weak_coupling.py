import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from pycfif.baths.expbath import ExponentialBath
from pycfif.sim_params import SimulationParamsCTC
from pycfif.ecfs import ECFs
from pycfif.baths.expbath_eta_ctc import ExpBath_Eta_CTC as Eta_gauss
from pycfif.baths.expbath_eta_ctc_quad import ExpBath_Eta_CTC as Eta_quad

if __name__=='__main__':

    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"), exist_ok=True)

    sigmax = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigmaz = np.array([[1.0, 0.0], [0.0, -1.0]])
    sigmay = 1j * sigmax@sigmaz

    expbath = ExponentialBath(alpha = 0.1,
                              s = 1.0,
                              wc = 5.0)
    print(expbath)

    opA = sigmaz
    opB = sigmaz

    sp = SimulationParamsCTC(
                             maxdtau = 0.05,
                             T = 0.2,
                             t_list = [0.2 * n for n in range(0, 51)],
                             H_S = sigmax,
                             opA = opA,
                             opB = opB,
                             cutoff = 10**(-9.0)
                            )

    # Allows for hot-swapping of different implementations for eta calculations
    cf = ECFs(expbath, sp, Eta_gauss)

    res = cf.run()

    np.savetxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "sym_a=0.1_b=5.0_dtau=0.05_1e-9.csv"), res, delimiter=',')

