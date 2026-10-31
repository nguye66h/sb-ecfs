"""Small Ohmic spin-boson regression for the two real-time formulations.

The stored Kadanoff-Baym values are the thermal correlator
G(t) = <sigma_z(t) sigma_z> on a coarse grid (alpha=0.1, s=1, wc=5,
T=1, N=6, M=4, t_max=0.6, cutoff=1e-5). They were produced by this
package and pin the contour implementation.

On that same grid the steady-state uMPS, built with either iTEBD
variant and a memory time of 1.5, reproduces the complex conjugate of
the Kadanoff-Baym correlator to about 10^{-3}. Conjugation is the
relation G(t)* = G(-t) for a Hermitian autocorrelation, and it is the
relation these two code paths satisfy. The tolerance below is several
times the discrepancy seen with scipy 1.18, and it is there to absorb
the different discretizations (imaginary-time step versus memory
truncation), not to hide a sign error in the real part.
"""

import numpy as np
import pytest

from pycfif import ECFs, ExponentialBath, SimulationParamsKBC, SimulationParamsSS
from pycfif.baths.expbath_eta_ctc import ExpBath_Eta_CTC
from pycfif.baths.expbath_eta_kbc import ExpBath_Eta_KBC

SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]])
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]])

# Real and imaginary parts of G(t) at t = 0, 0.1, ..., 0.6.
KBC_REFERENCE = np.array(
    [
        1.0000000000000000 + 0.0000000000000000j,
        0.9803314814828390 - 0.1379359258755874j,
        0.9229890788919102 - 0.2675626138551410j,
        0.8321954885501555 - 0.3822249733250007j,
        0.7136744288285538 - 0.4771273613031233j,
        0.5739201844307767 - 0.5490113680761702j,
        0.4198184671733772 - 0.5959655560008333j,
    ]
)


def _bath():
    return ExponentialBath(alpha=0.1, s=1.0, wc=5.0)


def _kadanoff_baym():
    params = SimulationParamsKBC(
        thermal_corr=True,
        N=6,
        M=4,
        T=1.0,
        t_max=0.6,
        H_S=SIGMA_X,
        opA=SIGMA_Z,
        opB=SIGMA_Z,
        cutoff=1e-5,
    )
    return ECFs(_bath(), params, ExpBath_Eta_KBC).run()


def test_kadanoff_baym_matches_stored_reference():
    result = _kadanoff_baym()
    assert result.shape == (7, 2)
    assert np.allclose(result[:, 0], np.arange(7) * 0.1)
    # sigma_z^2 = 1, so the equal-time autocorrelation is identically 1.
    assert result[0, 1] == pytest.approx(1.0 + 0.0j)
    assert np.allclose(result[:, 1], KBC_REFERENCE, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("algorithm", ["mbh_tebd", "ov_tebd"])
def test_steady_state_matches_conjugate_of_kadanoff_baym(algorithm):
    contour = _kadanoff_baym()
    params = SimulationParamsSS(
        T=1.0,
        dt=0.1,
        t_max=0.6,
        H_S=SIGMA_X,
        opA=SIGMA_Z,
        opB=SIGMA_Z,
        cutoff=1e-4,
        alg=algorithm,
        tmem=1.5,
    )
    steady = ECFs(_bath(), params, ExpBath_Eta_KBC).run()
    assert steady.shape == contour.shape
    assert np.allclose(steady[:, 0], contour[:, 0])
    assert steady[0, 1] == pytest.approx(1.0 + 0.0j, abs=1e-8)
    assert np.allclose(steady[:, 1], np.conj(contour[:, 1]), atol=1e-2)


def test_unknown_itebd_algorithm_is_rejected():
    params = SimulationParamsSS(
        T=1.0,
        dt=0.1,
        t_max=0.1,
        H_S=SIGMA_X,
        opA=SIGMA_Z,
        opB=SIGMA_Z,
        cutoff=1e-4,
        alg="qr_tebd",
        tmem=0.2,
    )
    with pytest.raises(RuntimeError, match="qr_tebd"):
        ECFs(ExponentialBath(alpha=0.0, s=1.0, wc=5.0), params, ExpBath_Eta_KBC).run()


def test_contour_and_eta_type_must_agree():
    params = SimulationParamsKBC(
        thermal_corr=True,
        N=4,
        M=4,
        T=1.0,
        t_max=0.4,
        H_S=SIGMA_X,
        opA=SIGMA_Z,
        opB=SIGMA_Z,
        cutoff=1e-6,
    )
    with pytest.raises(AssertionError, match="Kadanoff-Baym"):
        ECFs(_bath(), params, ExpBath_Eta_CTC).run()
