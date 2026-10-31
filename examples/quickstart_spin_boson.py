"""Coarse spin-boson thermal correlator on the Kadanoff-Baym contour.

This is the short run described in the README. It uses the same Ohmic bath
and temperature as the weak-coupling example, on a grid coarse enough to
finish in about a second. The scripts next to this file use the finer grids.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from pycfif import ECFs, ExponentialBath, SimulationParamsKBC
from pycfif.baths.expbath_eta_kbc import ExpBath_Eta_KBC

if __name__ == "__main__":
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])

    # J(w) = (alpha/2) * w * exp(-w/wc), with Delta = 1 and k_B = 1.
    bath = ExponentialBath(alpha=0.1, s=1.0, wc=5.0)
    params = SimulationParamsKBC(
        thermal_corr=True,
        N=8,
        M=4,
        T=0.2,
        t_max=1.0,
        H_S=sigma_x,
        opA=sigma_z,
        opB=sigma_z,
        cutoff=1e-6,
    )

    # Columns are t and G(t) = <sigma_z(t) sigma_z>_beta.
    result = ECFs(bath, params, ExpBath_Eta_KBC).run()
    print(result)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(
        os.path.join(out_dir, "quickstart_kbc.csv"),
        result,
        delimiter=",",
        header="t,G",
    )
