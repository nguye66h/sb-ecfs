import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from pycfif.baths.expbath import ExponentialBath
from pycfif.sim_params import SimulationParamsSS
from pycfif.ecfs import ECFs
from pycfif.baths.expbath_eta_kbc import ExpBath_Eta_KBC as Eta

if __name__=='__main__':

    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"), exist_ok=True)

    sigmax = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigmaz = np.array([[1.0, 0.0], [0.0, -1.0]])
    sigmay = 1j * sigmax@sigmaz

    expbath = ExponentialBath(alpha = 0.6,
                              s = 1.0,
                              wc = 5.0)
    print(expbath)

    opA = sigmaz
    opB = sigmaz

    # cutoff = 10**(-7) should give bond dim = 266
    # cutoff = 10**(-7.5) should give bond dim = 383, etc
    sp = SimulationParamsSS(T = 0.0,
                            dt = 0.04,
                            t_max = 10.0,
                            H_S = sigmax,
                            opA = opA,
                            opB = opB,
                            cutoff = 10**(-7.0),
                            alg="ov_tebd", tmem=400.0, tcut=200.0)

    # Allows for hot-swapping of different implementations for eta calculations
    cf = ECFs(expbath, sp, Eta)

    res = cf.run()

    np.savetxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "ss-test_ov.csv"), res, delimiter=',')

