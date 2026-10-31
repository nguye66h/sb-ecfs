"""Fast checks of the spectral density, SVD truncation, and MPS gauges."""

import numpy as np
import pytest

from pycfif import __version__
from pycfif.baths.expbath import ExponentialBath
from pycfif.mps import MPS
from pycfif.utilities import svd_truncate


def _contract(mps):
    """Contract an open-boundary MPS down to its physical tensor."""
    acc = mps.mps[0]
    for tensor in mps.mps[1:-1]:
        acc = np.einsum("...a,bac->...bc", acc, tensor)
    return np.einsum("...a,ba->...b", acc, mps.mps[-1])


def test_version_string():
    assert __version__ == "0.1.0"


def test_exponential_spectral_density_matches_ohmic_formula():
    alpha, s, wc = 0.2, 1.0, 5.0
    bath = ExponentialBath(alpha=alpha, s=s, wc=wc)
    frequencies = np.array([0.0, 0.3, 1.0, 5.0, 20.0])
    # J(w) = (alpha/2) * wc * (|w|/wc)^s * exp(-|w|/wc)
    expected = (alpha / 2) * wc * (frequencies / wc) ** s * np.exp(-frequencies / wc)
    assert np.allclose([bath.jw(w) for w in frequencies], expected)

    bath_s = ExponentialBath(alpha=0.4, s=3.0, wc=2.0)
    w = 1.7
    expected_s = (0.4 / 2) * 2.0 * (w / 2.0) ** 3 * np.exp(-w / 2.0)
    assert bath_s.jw(w) == pytest.approx(expected_s)


def test_svd_truncate_discards_only_negligible_weight():
    singular_values = np.array([1.0, 0.5, 1e-8])
    matrix = np.diag(singular_values)
    u_trunc, s_trunc, vh_trunc = svd_truncate(matrix, cutoff=1e-4, linds=[0], rinds=[1])

    assert s_trunc.shape == (2, 2)
    reconstructed = u_trunc @ s_trunc @ vh_trunc
    assert np.allclose(reconstructed, matrix, atol=1e-7)
    assert np.allclose(np.sort(np.diag(s_trunc))[::-1], singular_values[:2])


def test_svd_truncate_respects_maxdim():
    matrix = np.diag([4.0, 3.0, 2.0, 1.0])
    _, s_trunc, _ = svd_truncate(matrix, cutoff=0.0, linds=[0], rinds=[1], maxdim=2)
    assert s_trunc.shape == (2, 2)


def test_canonicalize_preserves_the_contracted_tensor():
    rng = np.random.default_rng(7)
    physical = 2
    chi = 3
    tensors = [
        rng.normal(size=(physical, chi)),
        rng.normal(size=(physical, chi, chi)),
        rng.normal(size=(physical, chi, chi)),
        rng.normal(size=(physical, chi)),
    ]
    state = MPS([tensor.copy() for tensor in tensors], loc=0)
    reference = _contract(state)

    for center in (0, 2, 3, 1):
        state.canonicalize(center)
        assert state.loc == center
        assert np.allclose(_contract(state), reference)


def test_pop_drops_the_last_site_and_the_orthogonality_center():
    tensors = [
        np.ones((2, 2)),
        np.ones((2, 2, 2)),
        np.ones((2, 2)),
    ]
    state = MPS(tensors, loc=2)
    state.pop()
    assert state.length() == 2
    assert state.loc == 1
    assert state.mps[-1].ndim == 3
