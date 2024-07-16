import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from pycfif.baths.expbath import ExponentialBath
from pycfif.sim_params import SimulationParamsKBC
from pycfif.ecfs import ECFs
from pycfif.baths.expbath_eta_kbc import ExpBath_Eta_KBC as Eta

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
    
    sp = SimulationParamsKBC(
                             thermal_corr = True,
                             N = 50,
                             M = 20,
                             T = 0.2,
                             t_max = 10.0,
                             H_S = sigmax,
                             opA = opA,
                             opB = opB,
                             cutoff = 10**(-9.0)
                            )

    # Allows for hot-swapping of different implementations for eta calculations
    cf = ECFs(expbath, sp, Eta)

    res = cf.run()

    np.savetxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "pt_therm_a=0.1_b=5.0_N=50_M=20_1e-9.csv"), res, delimiter=',')


    sp = SimulationParamsKBC(
                             thermal_corr = False,
                             N = 50,
                             M = 20,
                             T = 0.2,
                             t_max = 10.0,
                             H_S = sigmax,
                             opA = opA,
                             opB = opB,
                             cutoff = 10**(-9.0)
                            )

    cf = ECFs(expbath, sp, Eta)

    res = cf.run()

    np.savetxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "pt_sym_a=0.1_b=5.0_N=50_M=20_1e-9.csv"), res, delimiter=',')

