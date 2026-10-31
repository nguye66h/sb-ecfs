"""Decoupled bath: the influence functional is idle and the spin is solvable.

With alpha = 0 the bath spectral density vanishes, so every eta coefficient
is zero and the matrix-product influence functional is a product of
identities. The system Hamiltonian in this package is H = sigma_x
(the paper's Delta = 1). For A = B = sigma_z the equilibrium correlators
are then

    G(t) = Tr(e^{-beta H} e^{i H t} sigma_z e^{-i H t} sigma_z) / Z
         = cos(2t) - i tanh(beta) sin(2t)

    C(t) = Tr(e^{-beta H/2} e^{i H t} sigma_z e^{-i H t} e^{-beta H/2} sigma_z) / Z
         = cos(2t) / cosh(beta)

The symmetric Trotter splitting of a single Hamiltonian is exact, so both
contour formulations reproduce these expressions to roundoff.
"""

import numpy as np
import pytest

from pycfif import ECFs, ExponentialBath, SimulationParamsCTC, SimulationParamsKBC
from pycfif.baths.expbath_eta_ctc import ExpBath_Eta_CTC
from pycfif.baths.expbath_eta_kbc import ExpBath_Eta_KBC

SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]])
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]])
TEMPERATURE = 0.2
BETA = 1.0 / TEMPERATURE


def thermal_autocorrelation(times):
    return np.cos(2.0 * times) - 1j * np.tanh(BETA) * np.sin(2.0 * times)


def symmetrized_autocorrelation(times):
    return np.cos(2.0 * times) / np.cosh(BETA)


def _decoupled_bath():
    return ExponentialBath(alpha=0.0, s=1.0, wc=5.0)


def test_closed_forms_match_exact_diagonalization():
    """Guard the analytic expressions themselves, independent of pycfif."""
    from scipy.linalg import expm

    times = np.linspace(0.0, 0.6, 4)
    hamiltonian = SIGMA_X
    partition = np.trace(expm(-BETA * hamiltonian))
    for t in times:
        evolved = expm(1j * hamiltonian * t) @ SIGMA_Z @ expm(-1j * hamiltonian * t)
        thermal = np.trace(expm(-BETA * hamiltonian) @ evolved @ SIGMA_Z) / partition
        half = expm(-0.5 * BETA * hamiltonian)
        symmetrized = np.trace(half @ evolved @ half @ SIGMA_Z) / partition
        assert thermal == pytest.approx(thermal_autocorrelation(t))
        assert symmetrized == pytest.approx(symmetrized_autocorrelation(t))


def test_kadanoff_baym_thermal_matches_analytic_when_decoupled():
    params = SimulationParamsKBC(
        thermal_corr=True,
        N=6,
        M=4,
        T=TEMPERATURE,
        t_max=0.6,
        H_S=SIGMA_X,
        opA=SIGMA_Z,
        opB=SIGMA_Z,
        cutoff=1e-8,
    )
    result = ECFs(_decoupled_bath(), params, ExpBath_Eta_KBC).run()
    assert result.shape == (7, 2)
    assert np.allclose(result[:, 1], thermal_autocorrelation(result[:, 0]), atol=1e-12)


def test_kadanoff_baym_symmetrized_matches_analytic_when_decoupled():
    params = SimulationParamsKBC(
        thermal_corr=False,
        N=6,
        M=4,
        T=TEMPERATURE,
        t_max=0.6,
        H_S=SIGMA_X,
        opA=SIGMA_Z,
        opB=SIGMA_Z,
        cutoff=1e-8,
    )
    result = ECFs(_decoupled_bath(), params, ExpBath_Eta_KBC).run()
    assert np.allclose(result[:, 1], symmetrized_autocorrelation(result[:, 0]), atol=1e-12)
    assert result[0, 1].real == pytest.approx(1.0 / np.cosh(BETA))


def test_complex_time_contour_matches_analytic_when_decoupled():
    times = [0.0, 0.25, 0.5]
    params = SimulationParamsCTC(
        maxdtau=0.25,
        T=TEMPERATURE,
        t_list=times,
        H_S=SIGMA_X,
        opA=SIGMA_Z,
        opB=SIGMA_Z,
        cutoff=1e-8,
    )
    result = ECFs(_decoupled_bath(), params, ExpBath_Eta_CTC).run()
    assert np.allclose(result[:, 1], symmetrized_autocorrelation(result[:, 0]), atol=1e-12)
