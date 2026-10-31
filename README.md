# pycfif

[![Tests](https://github.com/nguye66h/sb-ecfs/actions/workflows/tests.yml/badge.svg)](https://github.com/nguye66h/sb-ecfs/actions/workflows/tests.yml)

pycfif computes real-time equilibrium correlation functions of a small quantum system coupled to a non-Markovian harmonic bath. The bath is integrated out into a Feynman–Vernon influence functional, that functional (the process tensor) is compressed as a matrix product state, and system observables are inserted into the resulting network. The implementation was written for the unbiased spin-boson model and checked against multilayer multiconfigurational time-dependent Hartree (ML-MCTDH) in Nguyen et al., J. Chem. Phys. **161**, 104111 (2024), an Editors' Choice article in the special collection *Algorithms and Software for Open Quantum System Dynamics*.

## What the method computes

The package evaluates two equilibrium correlators of system operators $A$ and $B$. With $\rho_\beta = e^{-\beta H}/Z$ and $A(t) = e^{iHt} A e^{-iHt}$,

$$
G_{AB}(t) = \mathrm{Tr}\big(\rho_\beta\, A(t)\, B\big),
\qquad
C_{AB}(t) = \mathrm{Tr}\big(\rho_\beta\, A(t - i\beta/2)\, B\big).
$$

$G_{AB}$ is the thermal correlator. $C_{AB}$ is its symmetrized partner, obtained by shifting one operator halfway around the thermal circle. For a bath that is harmonic and coupled linearly to the system, both quantities reduce to a contraction of the bare system propagators with the influence functional. pycfif builds that functional with the time-evolving matrix-product operator (TEMPO) algorithm and compresses it by singular-value truncation.

The model treated by the bundled bath class is the unbiased spin-boson Hamiltonian of the paper, in units where the tunneling amplitude $\Delta$, $\hbar$, and $k_B$ are all 1:

$$
H = \sigma^x + \sigma^z \sum_k g_k (b_k + b_k^\dagger) + \sum_k \omega_k b_k^\dagger b_k,
$$

with an exponentially cut-off spectral density

$$
J(\omega) = \frac{\alpha}{2}\,\omega_c^{1-s}\omega^{s} e^{-\omega/\omega_c}.
$$

Ohmic dissipation is $s = 1$. The normalization of $\alpha$ places the Toulouse point at $\alpha = 0.5$ and the localization transition at $\alpha = 1$ in the scaling limit.

## Two formulations

**Kadanoff–Baym contour.** Imaginary-time propagation prepares the thermal state and real-time propagation measures the correlator. One matrix-product process tensor, built by `TEMPO_PT`, holds every real time on the grid, so a single run returns $G(t)$ or $C(t)$ at all of those times. `SimulationParamsKBC` selects which correlator (`thermal_corr=True` for $G$, `False` for $C$) and sets the number of real-time steps $N$ and imaginary-time steps $M$.

**Steady state from real-time propagation.** If the bath is infinite, a finite system relaxes to the unique thermal state. The influence functional becomes translationally invariant and is stored as a uniform matrix product state, constructed by infinite time-evolving block decimation (iTEBD) over a finite memory window (`TEMPO_SS`). There is no imaginary-time leg. The equilibrium correlator is then read off from the dominant eigenvectors of the one-step transfer tensor. In the paper this route produced the converged correlators at substantially lower cost than the contour calculation; the steady-state runs in that comparison finished within a minute. Two iTEBD updates are implemented: Hastings' variant (`alg="mbh_tebd"`, the default) and the Orús–Vidal update (`alg="ov_tebd"`).

A third driver, `TEMPO_CTC`, propagates on a single complex-time contour and returns the symmetrized correlator $C(t)$ at one time per run. It is the method used by `examples/sym_weak_coupling.py`. The Kadanoff–Baym tensor is usually the better way to obtain $C(t)$ on a whole time grid, because the complex-time contour changes with the measurement time and the number of influence-functional coefficients grows steeply with that time.

## Installation

Python 3.9 or newer is required. The runtime dependencies are NumPy, SciPy, and opt_einsum.

```bash
git clone https://github.com/nguye66h/sb-ecfs.git
cd sb-ecfs
pip install -e ".[test]"
pytest
```

`pip install .` installs the library without pytest. From a source checkout the example scripts also run without installation, because each one inserts the repository root on `sys.path`.

## Quickstart

The following is a weak-coupling Ohmic spin-boson calculation ($\alpha = 0.1$, $\omega_c = 5$, $T = 0.2$, so $\beta = 5$) on a coarse Kadanoff–Baym grid. It returns $G(t) = \langle \sigma^z(t)\sigma^z\rangle$ for $t \in [0, 1]$ and takes about a second.

```python
import numpy as np
from pycfif import ECFs, ExponentialBath, SimulationParamsKBC
from pycfif.baths.expbath_eta_kbc import ExpBath_Eta_KBC

sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])

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
result = ECFs(bath, params, ExpBath_Eta_KBC).run()
# result[:, 0] is t, result[:, 1] is G(t)
```

The same script is `examples/quickstart_spin_boson.py`. It writes `examples/output/quickstart_kbc.csv`.

```bash
python examples/quickstart_spin_boson.py
```

`cutoff` is the relative singular-value threshold used while compressing the process tensor. `N` and `M` are the real- and imaginary-time grids; the timestep is `t_max / N`, and the imaginary-time step is $\beta / (2M)$.

## Examples

| Script | Route | What it runs |
| --- | --- | --- |
| `examples/quickstart_spin_boson.py` | Kadanoff–Baym, thermal | Coarse $\alpha = 0.1$ grid used above. About a second. |
| `examples/pt_weak_coupling.py` | Kadanoff–Baym, thermal and symmetrized | $\alpha = 0.1$, $\omega_c = 5$, $T = 0.2$, $N = 50$, $M = 20$, $t_\max = 10$, cutoff $10^{-9}$. Writes two CSVs. |
| `examples/ss_zeroT.py` | Steady state, Orús–Vidal iTEBD | $\alpha = 0.6$, $T = 0$, $\delta t = 0.04$, memory time 400 with a smooth cutoff starting at 200, cutoff $10^{-7}$. This is a production run: it builds $10^4$ influence-functional coefficients. |
| `examples/sym_weak_coupling.py` | Complex-time contour | Symmetrized correlator at 51 times for the $\alpha = 0.1$ bath, with contour step 0.05 and cutoff $10^{-9}$. Each time is a separate propagation. |

Output files go to `examples/output/`, which is gitignored.

## Layout

```
pycfif/
  bath.py            spectral-density interface
  baths/             exponential cutoff and its eta coefficients
  eta_kbc.py         Kadanoff–Baym and steady-state eta interface
  eta_ctc.py         complex-time eta interface
  inflfn.py          influence-functional tensors
  mps.py             finite MPS and the two-site uniform MPS
  tempo_pt.py        Kadanoff–Baym process tensor
  tempo_ss.py        steady-state uMPS
  tempo_ctc.py       complex-time contour
  ecfs.py            driver used by the examples
  sim_params.py      parameter objects for the three routes
  utilities.py       truncated SVD and QR
```

`ECFs(bath, params, eta_class).run()` is the supported entry point. The `eta_class` argument swaps the quadrature used for the influence-functional coefficients without touching the tensor-network contraction. `ExpBath_Eta_KBC` serves both the Kadanoff–Baym contour and the steady-state route. The complex-time contour has two quadratures, `pycfif.baths.expbath_eta_ctc` and `pycfif.baths.expbath_eta_ctc_quad`. Both expose a class named `ExpBath_Eta_CTC`.

## Tests

```bash
pytest
```

The suite is meant to finish in well under a minute. It checks the Ohmic spectral density, SVD truncation, and MPS canonicalization; the decoupled limit $\alpha = 0$, where $G(t)$ and $C(t)$ are known in closed form, on both the Kadanoff–Baym contour and the complex-time contour; and a small coupled spin-boson problem. That last test compares the Kadanoff–Baym thermal correlator with a stored reference and with the steady-state result. For this Hermitian autocorrelation the steady-state implementation agrees with the complex conjugate of the Kadanoff–Baym correlator (the $t \to -t$ function) to about $10^{-3}$ on the coarse grid used in the test.

## Citation

If you use this code, please cite

H. Nguyen, N. Ng, L. P. Lindoy, G. Park, A. J. Millis, G. Kin-Lic Chan, and D. R. Reichman, "Correlation functions from tensor network influence functionals: The case of the spin-boson model," J. Chem. Phys. **161**, 104111 (2024). https://doi.org/10.1063/5.0224880

The preprint is [arXiv:2406.15737](https://arxiv.org/abs/2406.15737). A machine-readable record is in `CITATION.cff`.

## Authors

The package is by Haimi Nguyen (Department of Chemistry, Columbia University), ORCID [0000-0003-1685-445X](https://orcid.org/0000-0003-1685-445X).

The Journal of Chemical Physics article was written with Nathan Ng, Lachlan P. Lindoy, Gunhee Park, Andrew J. Millis, Garnet Kin-Lic Chan, and David R. Reichman. Correspondence for the paper: David R. Reichman, drr2103@columbia.edu.

## License

[MIT](LICENSE). Copyright 2024 Haimi Nguyen.
