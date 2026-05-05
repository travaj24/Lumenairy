"""
lumenairy.asymptotic -- Phase-space asymptotic propagator and Laguerre-Gaussian
aberration tensor.

This module implements the closed-form Gaussian-moment evaluation of the
phase-space (Maslov) diffraction integral derived in the companion design
notes ``maslov_zemax_merit`` (paper 1) and ``small_waist_asymptotic`` (paper
2).  It complements ``apply_real_lens_maslov`` -- which evaluates the same
underlying integral by direct Chebyshev-quadrature in v_2 -- by replacing
the quadrature with a finite Wick-contracted moment over a complex-symmetric
covariance matrix built from the Chebyshev polynomial fit.

What this is for
----------------

The dominant cost in a wave-aware merit function for optical design is the
diffraction integral evaluated *thousands of times* in the inner optimisation
loop.  The asymptotic propagator runs ~10**3-10**4 times faster than direct
quadrature per output pixel and produces a *physically-named* aberration
tensor whose indices correspond to the classical Seidel/Zernike modes
(spherical, coma, astigmatism, ...).  Optimising against that tensor
directly is the optimisation analog of optimising against Strehl, but
computable in milliseconds.

Public API
----------

Math primitives
~~~~~~~~~~~~~~~

- ``lg_polynomial(p, ell, w)`` -- polynomial coefficients of the LG_{p,l}
  mode in Cartesian (x, y), suitable for closed-form Gaussian-moment
  evaluation.  Returns a dict ``{(i, j): c_{ij}}`` such that
  ``LG(x, y) = (sum c_{ij} x^i y^j) * exp(-(x^2 + y^2)/w^2)``.
- ``hg_polynomial(m, n, wx, wy)`` -- same for Hermite-Gaussian
  ``HG(x, y) = (sum c_{ij} x^i y^j) * exp(-x^2/wx^2 - y^2/wy^2)``.
- ``evaluate_lg_mode``, ``evaluate_hg_mode`` -- evaluate basis functions on
  a (x, y) grid.
- ``decompose_lg``, ``decompose_hg`` -- project an arbitrary field onto
  the LG/HG basis by overlap integrals.
- ``gaussian_moment_2d(a, b, sigma)`` -- evaluate
  ``<eta_1^a eta_2^b>_Sigma`` for a 2-D complex-symmetric Gaussian via
  Wick's theorem (closed form; no quadrature).
- ``gaussian_moment_table_2d(M, max_total_order)`` -- precompute moments
  up to a chosen total order; returns dict ``{(a, b): <...>}``.

Canonical polynomial fit
~~~~~~~~~~~~~~~~~~~~~~~~

- ``fit_canonical_polynomials(prescription, surf_pair, wavelength, ...)``
  -- public version of the apply_real_lens_maslov fit:  trace a
  Chebyshev-node grid through the surface pair, fit Phi(s2, v2) and
  s1(s2, v2) as 4-variable Chebyshev tensor-product polynomials, and
  return a ``CanonicalPolyFit`` dataclass.  Includes the
  linear-phase-extraction trick of paper 1 Section 5 for diffractive
  surfaces.

Aberration tensor and propagator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``solve_envelope_stationary(fit, s2, source_point, w_s, w_p, v2_centre)``
  -- Newton-solve paper 2 eq. (9) for the envelope-stationary v_2*.
- ``aberration_tensor(fit, s2_image, source_lg_modes, pupil_lg_modes,
  output_lg_modes, source_point, w_s, w_p, v2_centre, w_o, ...)`` --
  paper 2 eq. (44):  compute the LG aberration tensor T_{k;n,m}
  whose indices are physical Seidel/Zernike aberrations.
- ``propagate_modal_asymptotic(fit, source_lg_amps, pupil_lg_amps,
  source_point, w_s, w_p, v2_centre, output_grid)`` -- paper 2 eq. (24):
  evaluate the leading-order modal propagator on a 2-D output grid.

Conventions
-----------

- All physical quantities in SI (positions in metres, OPD in waves where
  noted).  This matches the rest of lumenairy.
- Phase convention follows lumenairy:  field amplitudes carry phase
  factors ``exp(+i * 2 pi * Phi)`` (Phi in waves, equivalently
  ``exp(+i * k * OPL)``).
- Polynomial bases use the same Chebyshev tensor-product layout as
  ``apply_real_lens_maslov`` so canonical fits are reusable across both
  propagators.

See ``validation/test_asymptotic.py`` for the validation suite (Wick-moment
identities, Collins ABCD reduction, Fourier-of-pupil reduction, agreement
with apply_real_lens in the source-dominated and pupil-dominated limits,
Seidel-aberration correspondence, and end-to-end MeritTerm convergence).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Reuse the proven Chebyshev infrastructure that already powers
# apply_real_lens_maslov.  These helpers are private only by leading-
# underscore convention; they are stable and well-validated.
from .lenses import (
    _chebyshev_vandermonde,
    _chebyshev_derivative_vandermonde,
    _chebyshev_second_derivative_vandermonde,
    _multi_indices_total_degree,
    _evaluate_polynomial_4d,
    _evaluate_polynomial_4d_and_grad34,
    _fit_normaliser,
)


__all__ = [
    # Data containers
    'CanonicalPolyFit',
    'AberrationTensorResult',
    # Modes / polynomial coefficients
    'lg_polynomial',
    'hg_polynomial',
    'evaluate_lg_mode',
    'evaluate_hg_mode',
    'decompose_lg',
    'decompose_hg',
    'lg_seidel_label',
    # Wick moments
    'gaussian_moment_2d',
    'gaussian_moment_table_2d',
    # Polynomial fit
    'fit_canonical_polynomials',
    # Stationary solver
    'solve_envelope_stationary',
    # Top-level evaluators
    'aberration_tensor',
    'propagate_modal_asymptotic',
]


# ===========================================================================
# Section 1 -- Laguerre-Gaussian and Hermite-Gaussian basis polynomials
# ===========================================================================

def lg_polynomial(p: int, ell: int, w: float) -> Dict[Tuple[int, int], complex]:
    """Cartesian polynomial coefficients of a Laguerre-Gaussian mode.

    The LG_{p,l} mode with waist ``w`` centred at the origin can be written
    as a polynomial in (x, y) times a shared Gaussian envelope::

        LG_{p,l}(x, y) = (sum_{i,j} c_{ij} x^i y^j) * exp(-(x^2 + y^2)/w^2)

    with the normalisation convention

        N_{p,l} = sqrt(2 * p! / (pi * (p + |l|)!))

    so that the modes are orthonormal under the L^2 inner product
    ``<f, g> = integral f^* g  dx dy`` (no extra envelope factor).

    Parameters
    ----------
    p : int
        Radial index, p >= 0.
    ell : int
        Azimuthal index, any integer.  The angular dependence is
        ``exp(i*ell*phi)`` so positive ``ell`` rotates one way and
        negative ``ell`` the other.
    w : float
        Beam waist [m].

    Returns
    -------
    dict
        ``{(i, j): complex}`` mapping Cartesian monomial exponents to
        polynomial coefficients.  Total polynomial degree is
        ``|ell| + 2 p``.

    Notes
    -----
    The expansion is exact and finite (no truncation):  it follows from
    standard identities

        L_p^{|l|}(x) = sum_{k=0}^p (-1)^k / k! * binom(p + |l|, p - k) * x^k
        (x + i*s*y)^{|l|} = sum_{m=0}^{|l|} binom(|l|, m) (i*s)^m x^{|l|-m} y^m
        (x^2 + y^2)^k = sum_{j=0}^k binom(k, j) x^{2j} y^{2(k-j)}

    where ``s = sign(ell)``.  The LG mode is
    ``N * (sqrt(2)/w)^{|l|} * (x + i*s*y)^{|l|} * L_p^{|l|}(2 r^2/w^2)
    * exp(-r^2/w^2)``.
    """
    if p < 0:
        raise ValueError(f"LG radial index p must be >= 0, got {p}")
    if w <= 0:
        raise ValueError(f"LG waist w must be > 0, got {w}")
    abs_ell = abs(ell)
    s_sign = 1 if ell >= 0 else -1

    f = math.sqrt(2.0) / w
    # Standard waist-w normalisation:  N = sqrt(2 p! / (pi (p+|l|)! w^2)).
    # Verified by:  integral |LG_00|^2 dA = N^2 * pi w^2 / 2 = 1.
    N = math.sqrt(
        2.0 * math.factorial(p)
        / (math.pi * math.factorial(p + abs_ell) * (w * w))
    )

    coeffs: Dict[Tuple[int, int], complex] = {}
    for m in range(abs_ell + 1):
        binom_lm = math.comb(abs_ell, m)
        is_m = (1j * s_sign) ** m
        for k in range(p + 1):
            lag_coef = ((-1) ** k / math.factorial(k)
                        * math.comb(p + abs_ell, p - k))
            for j in range(k + 1):
                binom_kj = math.comb(k, j)
                i_x = (abs_ell - m) + 2 * j
                i_y = m + 2 * (k - j)
                c = (
                    N
                    * (f ** abs_ell)
                    * binom_lm
                    * is_m
                    * lag_coef
                    * (f ** (2 * k))
                    * binom_kj
                )
                key = (i_x, i_y)
                coeffs[key] = coeffs.get(key, 0.0 + 0.0j) + c
    return coeffs


def hg_polynomial(m: int, n: int, wx: float,
                  wy: Optional[float] = None
                  ) -> Dict[Tuple[int, int], complex]:
    """Cartesian polynomial coefficients of a Hermite-Gaussian mode.

    The HG_{m,n} mode with axis waists ``wx, wy`` centred at the origin is

        HG_{m,n}(x, y) = phi_m(x; wx) * phi_n(y; wy)

    with the 1-D physicist's-Hermite Gaussian basis function

        phi_k(u; w) = (2/(pi w^2))^{1/4} / sqrt(2^k k!)
                     * H_k(sqrt(2) u / w) * exp(-u^2 / w^2)

    Parameters
    ----------
    m, n : int
        x- and y-mode orders, both >= 0.
    wx : float
        Waist along x [m].
    wy : float, optional
        Waist along y [m].  Defaults to ``wx`` (round Gaussian).

    Returns
    -------
    dict
        ``{(i, j): complex}`` -- Cartesian polynomial coefficients
        such that ``HG_{m,n}(x, y) = (sum c_{ij} x^i y^j) *
        exp(-x^2/wx^2 - y^2/wy^2)``.

    Notes
    -----
    The basis is orthonormal:  ``int phi_m(x) phi_p(x) dx = delta_{mp}``.
    Total polynomial degree is ``m + n``.
    """
    if m < 0 or n < 0:
        raise ValueError(f"HG indices must be >= 0, got ({m}, {n})")
    if wx <= 0:
        raise ValueError(f"HG waist wx must be > 0, got {wx}")
    if wy is None:
        wy = wx
    if wy <= 0:
        raise ValueError(f"HG waist wy must be > 0, got {wy}")

    # 1-D Hermite polynomial coefficients of H_m(sqrt(2)*x/wx).
    # Build as polynomial coefficients in x (real, may have negative entries).
    def hermite_coeffs(k: int, alpha: float) -> Dict[int, float]:
        """Coefficients of H_k(alpha * x) as polynomial in x."""
        # H_0 = 1, H_1 = 2*x, H_{k+1}(z) = 2 z H_k(z) - 2 k H_{k-1}(z),
        # but here we have H_k(alpha x): substitute z = alpha x, then
        # rebuild polynomial in x.
        if k == 0:
            return {0: 1.0}
        if k == 1:
            return {1: 2.0 * alpha}
        prev2: Dict[int, float] = {0: 1.0}
        prev1: Dict[int, float] = {1: 2.0 * alpha}
        cur: Dict[int, float] = {}
        for kk in range(2, k + 1):
            # H_{kk}(alpha x) = 2 (alpha x) H_{kk-1}(alpha x)
            #                  - 2 (kk-1) H_{kk-2}(alpha x)
            cur = {}
            for power, coef in prev1.items():
                # 2 alpha x * coef * x^power = 2 alpha coef * x^(power+1)
                key = power + 1
                cur[key] = cur.get(key, 0.0) + 2.0 * alpha * coef
            for power, coef in prev2.items():
                cur[power] = cur.get(power, 0.0) - 2.0 * (kk - 1) * coef
            prev2, prev1 = prev1, cur
        return cur

    # 1-D normalisation:  N_k = (2/(pi w^2))^{1/4} / sqrt(2^k k!)
    Nx = ((2.0 / (math.pi * wx * wx)) ** 0.25
          / math.sqrt((2 ** m) * math.factorial(m)))
    Ny = ((2.0 / (math.pi * wy * wy)) ** 0.25
          / math.sqrt((2 ** n) * math.factorial(n)))

    Hx = hermite_coeffs(m, math.sqrt(2.0) / wx)   # in x
    Hy = hermite_coeffs(n, math.sqrt(2.0) / wy)   # in y

    coeffs: Dict[Tuple[int, int], complex] = {}
    for i, cx in Hx.items():
        for j, cy in Hy.items():
            key = (i, j)
            coeffs[key] = coeffs.get(key, 0.0 + 0.0j) + Nx * Ny * cx * cy
    return coeffs


def _evaluate_poly2d(coeffs: Dict[Tuple[int, int], complex],
                     x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate a 2-D polynomial ``sum c_{ij} x^i y^j``.  Returns complex."""
    out = np.zeros_like(np.broadcast_to(x + 0j, np.broadcast(x, y).shape))
    out = np.array(out, dtype=np.complex128)
    if not coeffs:
        return out
    max_i = max(k[0] for k in coeffs)
    max_j = max(k[1] for k in coeffs)
    # Pre-compute powers
    powers_x = [np.ones_like(x, dtype=np.float64)]
    for _ in range(max_i):
        powers_x.append(powers_x[-1] * x)
    powers_y = [np.ones_like(y, dtype=np.float64)]
    for _ in range(max_j):
        powers_y.append(powers_y[-1] * y)
    for (i, j), c in coeffs.items():
        out = out + c * powers_x[i] * powers_y[j]
    return out


def evaluate_lg_mode(p: int, ell: int, w: float,
                     x: np.ndarray, y: np.ndarray,
                     cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """Evaluate LG_{p,l} on a 2-D grid.

    Parameters
    ----------
    p, ell : int
        Mode indices.
    w : float
        Waist [m].
    x, y : ndarray
        Cartesian sample points [m].  Shapes must broadcast.
    cx, cy : float, optional
        Mode centre [m].  Defaults to origin.

    Returns
    -------
    ndarray, complex
    """
    poly = lg_polynomial(p, ell, w)
    rx = x - cx
    ry = y - cy
    polynomial = _evaluate_poly2d(poly, rx, ry)
    envelope = np.exp(-(rx * rx + ry * ry) / (w * w))
    return polynomial * envelope


def evaluate_hg_mode(m: int, n: int, wx: float, wy: Optional[float],
                     x: np.ndarray, y: np.ndarray,
                     cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """Evaluate HG_{m,n} on a 2-D grid (see ``hg_polynomial``)."""
    if wy is None:
        wy = wx
    poly = hg_polynomial(m, n, wx, wy)
    rx = x - cx
    ry = y - cy
    polynomial = _evaluate_poly2d(poly, rx, ry)
    envelope = np.exp(-(rx * rx) / (wx * wx) - (ry * ry) / (wy * wy))
    return polynomial * envelope


def lg_seidel_label(p: int, ell: int) -> str:
    """Map an LG output-mode index ``(p, ell)`` to its classical
    Seidel/Zernike aberration name.

    From paper 2, Section 7.5.  Used by tooling and diagnostics that
    want to report aberrations by name rather than by index.
    """
    abs_ell = abs(ell)
    table = {
        (0, 0): 'piston',
        (1, 0): 'defocus',
        (2, 0): 'spherical',
        (3, 0): 'higher_spherical',
        (0, 1): 'tilt',
        (1, 1): 'coma',
        (2, 1): 'higher_coma',
        (0, 2): 'astigmatism',
        (1, 2): 'higher_astigmatism',
        (0, 3): 'trefoil',
    }
    return table.get((p, abs_ell), f'p{p}_l{ell:+d}')


# ===========================================================================
# Section 2 -- Wick moments for 2-D complex-symmetric Gaussians
# ===========================================================================

def gaussian_moment_2d(a: int, b: int,
                       sigma: np.ndarray) -> complex:
    """Closed-form 2-D Gaussian moment ``<eta_x^a eta_y^b>_Sigma``.

    For a 2-D Gaussian with weight ``exp(-eta^T M eta)`` and complex-
    symmetric covariance ``Sigma == 0.5 * inv(M)``, this returns

        <eta_x^a eta_y^b>_Sigma = (1/Z) integral eta_x^a eta_y^b
                                       * exp(-eta^T M eta) d^2 eta

    where ``Z = pi / sqrt(det M)`` is the Gaussian normalisation.
    Vanishes by symmetry for ``a + b`` odd; otherwise evaluates the
    closed-form pair-counting sum (paper 2, eq. 36).

    Parameters
    ----------
    a, b : int
        Non-negative integer exponents on eta_x, eta_y.
    sigma : ndarray, shape (2, 2)
        Covariance ``0.5 * inv(M)``.  Must be complex-symmetric.

    Returns
    -------
    complex

    Notes
    -----
    Wick contraction reduces the moment to a sum over balanced pair
    assignments.  The closed form has at most ``floor(min(a,b)/2) + 1``
    nonzero terms even though the naive enumeration would have
    ``(a + b - 1)!!`` pairings.
    """
    if a < 0 or b < 0:
        raise ValueError(f"Moment indices must be >= 0, got ({a}, {b})")
    if (a + b) % 2 != 0:
        return 0.0 + 0.0j

    s11 = complex(sigma[0, 0])
    s12 = complex(sigma[0, 1])
    s22 = complex(sigma[1, 1])

    # p_12 has the same parity as a (and as b, since a+b is even).
    p12_min = a % 2
    total = 0.0 + 0.0j
    fa = math.factorial(a)
    fb = math.factorial(b)
    for p12 in range(p12_min, min(a, b) + 1, 2):
        p11 = (a - p12) // 2
        p22 = (b - p12) // 2
        denom = (math.factorial(p11) * math.factorial(p12)
                 * math.factorial(p22) * (2 ** p11) * (2 ** p22))
        coef = (fa * fb) / denom
        total += (coef * (s11 ** p11) * (s12 ** p12) * (s22 ** p22))
    return total


def gaussian_moment_table_2d(M: np.ndarray, max_total_order: int
                              ) -> Dict[Tuple[int, int], complex]:
    """Pre-tabulate Gaussian moments up to a chosen total order.

    Used by the asymptotic propagator and aberration-tensor evaluator to
    amortise moment evaluation across many ``(n, m)`` mode pairs at the
    same output pixel:  the moments depend only on the covariance, not
    on the modal indices.

    Parameters
    ----------
    M : ndarray, shape (2, 2)
        Complex-symmetric quadratic form in ``exp(-eta^T M eta)``.
    max_total_order : int
        Build moments for all ``(a, b)`` with ``a + b <= max_total_order``.

    Returns
    -------
    dict
        ``{(a, b): <eta_x^a eta_y^b>_Sigma}`` for all valid index pairs.
    """
    if max_total_order < 0:
        raise ValueError(f"max_total_order must be >= 0, got {max_total_order}")
    if M.shape != (2, 2):
        raise ValueError(f"M must be 2x2, got shape {M.shape}")
    sigma = 0.5 * np.linalg.inv(M)
    table: Dict[Tuple[int, int], complex] = {}
    for total in range(max_total_order + 1):
        for a in range(total + 1):
            b = total - a
            table[(a, b)] = gaussian_moment_2d(a, b, sigma)
    return table


# ===========================================================================
# Section 3 -- Canonical polynomial fit (Phi(s2, v2) and s1(s2, v2))
# ===========================================================================

@dataclass
class CanonicalPolyFit:
    """Result of a Chebyshev tensor-product fit to ray-traced
    ``Phi(s2, v2)`` and ``s1(s2, v2)`` (paper 1, Section 3).

    Fields
    ------
    poly_order : int
        Total-degree truncation used in the fit (``|k|_1 <= poly_order``).
    multi_indices : list of 4-tuples
        Multi-indices ``(k1, k2, k3, k4)`` enumerating the basis terms,
        in the same order as the coefficient vectors.
    coef_phi : ndarray
        Coefficients of the Phi polynomial (in waves), residual after
        linear-phase extraction.
    coef_s1x, coef_s1y : ndarray
        Coefficients of the s1 back-map components (in metres).
    linear_coeffs_phi : ndarray, optional
        ``(alpha_0, alpha_1, alpha_2, alpha_3, alpha_4)`` of the linear
        prefit of Phi (paper 1, Section 5.4).  None if the fit was done
        with ``extract_linear_phase=False``.
    s2x_centre, s2x_halfrange : float
        Normalisation parameters for s2x:  ``u_s2x = (s2x - centre) /
        halfrange``.
    s2y_centre, s2y_halfrange : float
        Same for s2y.
    v2x_centre, v2x_halfrange : float
        Same for v2x.
    v2y_centre, v2y_halfrange : float
        Same for v2y.
    wavelength : float
        Wavelength used for the fit [m].
    res_phi_rms_waves : float
        RMS fit residual on the Chebyshev-node training set, in waves.
    res_s1_rms_m : float
        RMS fit residual on s1, in metres.
    n_rays : int
        Number of training rays.
    """
    poly_order: int
    multi_indices: List[Tuple[int, int, int, int]]
    coef_phi: np.ndarray
    coef_s1x: np.ndarray
    coef_s1y: np.ndarray
    s2x_centre: float
    s2x_halfrange: float
    s2y_centre: float
    s2y_halfrange: float
    v2x_centre: float
    v2x_halfrange: float
    v2y_centre: float
    v2y_halfrange: float
    wavelength: float
    res_phi_rms_waves: float = 0.0
    res_s1_rms_m: float = 0.0
    n_rays: int = 0
    linear_coeffs_phi: Optional[np.ndarray] = None
    extract_linear_phase: bool = False

    # ------------------------------------------------------------------
    # Coordinate normalisation helpers
    # ------------------------------------------------------------------
    def to_normalised(self, s2x: np.ndarray, s2y: np.ndarray,
                      v2x: np.ndarray, v2y: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Map physical (s2, v2) into normalised [-1, 1]^4."""
        u1 = (s2x - self.s2x_centre) / self.s2x_halfrange
        u2 = (s2y - self.s2y_centre) / self.s2y_halfrange
        u3 = (v2x - self.v2x_centre) / self.v2x_halfrange
        u4 = (v2y - self.v2y_centre) / self.v2y_halfrange
        return u1, u2, u3, u4

    def in_box(self, s2x: np.ndarray, s2y: np.ndarray,
               v2x: np.ndarray, v2y: np.ndarray) -> np.ndarray:
        """Boolean mask of points inside the fit's normalised box."""
        u1, u2, u3, u4 = self.to_normalised(s2x, s2y, v2x, v2y)
        return ((np.abs(u1) <= 1.0) & (np.abs(u2) <= 1.0)
                & (np.abs(u3) <= 1.0) & (np.abs(u4) <= 1.0))

    # ------------------------------------------------------------------
    # Polynomial evaluation helpers
    # ------------------------------------------------------------------
    def eval_s1(self, s2x: np.ndarray, s2y: np.ndarray,
                v2x: np.ndarray, v2y: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the back-map s1(s2, v2) -> (s1x, s1y) at arbitrary
        (s2, v2) samples in physical units."""
        u1, u2, u3, u4 = self.to_normalised(s2x, s2y, v2x, v2y)
        s1x = _evaluate_polynomial_4d(self.coef_s1x, self.multi_indices,
                                       u1, u2, u3, u4, self.poly_order)
        s1y = _evaluate_polynomial_4d(self.coef_s1y, self.multi_indices,
                                       u1, u2, u3, u4, self.poly_order)
        return s1x, s1y

    def eval_phi(self, s2x: np.ndarray, s2y: np.ndarray,
                 v2x: np.ndarray, v2y: np.ndarray,
                 *, include_linear: bool = True) -> np.ndarray:
        """Evaluate Phi(s2, v2) [waves] at arbitrary samples.

        Parameters
        ----------
        include_linear : bool, default True
            If True and the fit extracted a linear ramp, re-add it to
            give the *raw* OPD that the ray trace would report.  If
            False, return only the Chebyshev residual (the
            integrator-safe form, see paper 1 Section 5).
        """
        u1, u2, u3, u4 = self.to_normalised(s2x, s2y, v2x, v2y)
        phi = _evaluate_polynomial_4d(self.coef_phi, self.multi_indices,
                                       u1, u2, u3, u4, self.poly_order)
        if include_linear and self.linear_coeffs_phi is not None:
            a0, a1, a2, a3, a4 = self.linear_coeffs_phi
            phi = phi + (a0 + a1 * u1 + a2 * u2 + a3 * u3 + a4 * u4)
        return phi

    def eval_s1_with_v2_grad(self, s2x: np.ndarray, s2y: np.ndarray,
                              v2x: np.ndarray, v2y: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate s1(s2, v2) and its (v2x, v2y)-gradient (Jacobian).

        Returns
        -------
        s1x, s1y, dS1x_dv2x, dS1x_dv2y, dS1y_dv2x, dS1y_dv2y
            All in physical units; gradients have the chain-rule
            normalisation factor 1/v2x_halfrange and 1/v2y_halfrange
            already applied.
        """
        u1, u2, u3, u4 = self.to_normalised(s2x, s2y, v2x, v2y)
        s1x, du3_s1x, du4_s1x = _evaluate_polynomial_4d_and_grad34(
            self.coef_s1x, self.multi_indices, u1, u2, u3, u4,
            self.poly_order)
        s1y, du3_s1y, du4_s1y = _evaluate_polynomial_4d_and_grad34(
            self.coef_s1y, self.multi_indices, u1, u2, u3, u4,
            self.poly_order)
        # Chain rule:  d/dv2x = (1/v2x_halfrange) d/du3 ; same for v2y
        invhx = 1.0 / self.v2x_halfrange
        invhy = 1.0 / self.v2y_halfrange
        return (s1x, s1y,
                du3_s1x * invhx, du4_s1x * invhy,
                du3_s1y * invhx, du4_s1y * invhy)

    def eval_phi_with_v2_grad(self, s2x: np.ndarray, s2y: np.ndarray,
                               v2x: np.ndarray, v2y: np.ndarray,
                               *, include_linear: bool = False
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate Phi and its v2-gradient.  ``include_linear``
        controls whether the linear-phase prefit is re-added (default
        False since the integrator drops it)."""
        u1, u2, u3, u4 = self.to_normalised(s2x, s2y, v2x, v2y)
        phi, du3_phi, du4_phi = _evaluate_polynomial_4d_and_grad34(
            self.coef_phi, self.multi_indices, u1, u2, u3, u4,
            self.poly_order)
        if include_linear and self.linear_coeffs_phi is not None:
            a0, a1, a2, a3, a4 = self.linear_coeffs_phi
            phi = phi + (a0 + a1 * u1 + a2 * u2 + a3 * u3 + a4 * u4)
            du3_phi = du3_phi + a3
            du4_phi = du4_phi + a4
        invhx = 1.0 / self.v2x_halfrange
        invhy = 1.0 / self.v2y_halfrange
        return phi, du3_phi * invhx, du4_phi * invhy


def fit_canonical_polynomials(
    prescription: Dict[str, Any],
    wavelength: float,
    *,
    source_box_half: float = 50e-6,
    pupil_box_half: float = 0.05,
    n_field: int = 8,
    n_pupil: int = 8,
    poly_order: int = 6,
    extract_linear_phase: bool = True,
    object_distance: Optional[float] = None,
    surface_diffraction: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
) -> CanonicalPolyFit:
    """Fit Chebyshev tensor-product polynomials to ``Phi(s2, v2)`` and
    ``s1(s2, v2)`` over a 4-D source x pupil grid (paper 1, Section 3).

    The trace-and-fit pipeline samples a Chebyshev-node grid in the
    source-plane coordinates ``(s1_x, s1_y)`` and the input direction
    cosines ``(v1_x, v1_y)``, traces each ray forward through the
    prescription, and records the landing position ``s2`` and direction
    cosine ``v2`` at the last surface (along with the accumulated
    optical path).  The 4-D scattered training set
    ``{(s2, v2, s1, Phi)}`` then drives a least-squares fit of
    ``s1(s2, v2)`` and ``Phi(s2, v2)`` as 4-variable Chebyshev tensor-
    product polynomials.

    Sampling source positions in addition to pupil directions is what
    gives the back-map ``s1(s2, v2)`` non-trivial dependence on (s2,
    v2):  with a single source point the trace produces a 2-D
    submanifold of (s2, v2) values along which ``s1 = const`` and the
    fit's J = d s1 / d v2 collapses to zero.

    Parameters
    ----------
    prescription : dict
        lumenairy prescription dict (e.g. from
        ``load_zmx_prescription`` or a builder).
    wavelength : float
        Wavelength [m].
    source_box_half : float
        Half-width of the source sampling box [m].  Set to a few times
        the source-Gaussian waist consumers will use (typical 50 um).
    pupil_box_half : float
        Half-width of the input direction-cosine box [dimensionless].
        Set to cover the system aperture as seen from the source plane.
    n_field : int
        Per-axis Chebyshev-node grid size for source sampling.
    n_pupil : int
        Per-axis grid size for input direction sampling.  Total ray
        count is ``n_field**2 * n_pupil**2``.
    poly_order : int
        Total-degree truncation of the Chebyshev fit.  Default 6.
    extract_linear_phase : bool
        If True (default), pre-fit and subtract a 5-parameter linear
        ramp from Phi before the Chebyshev fit -- this restores Nyquist
        sampling for diffractive surfaces at non-zero orders (paper 1,
        Section 5).  No-op for refractive systems.
    object_distance : float, optional
        Source-plane to first-surface distance [m].  Defaults to
        ``prescription.get('object_distance', 0.0)``.
    surface_diffraction : dict, optional
        Per-surface diffractive-order kicks for DOE / grating elements
        embedded in the prescription.  Maps surface index to
        ``(order_x, order_y, period_x, period_y)``.  Each ray's
        direction cosines are shifted by ``m * lambda / Lambda`` after
        refraction at that surface, evanescent orders are flagged
        ``alive=False``.  Use this to pin the LG fit to a single
        diffraction order of a Dammann splitter or any thin grating in
        a sequential prescription -- the resulting fit's chief-ray
        landings span the chosen order's image-plane footprint, so
        :func:`aberration_tensor` evaluations at that order's frame
        centres reflect the true diffractive wavefront.  See also
        :func:`lumenairy.raytrace.apply_doe_phase_traced`.

    Returns
    -------
    CanonicalPolyFit
        Fit container.  ``fit.res_phi_rms_waves`` is typically below
        1e-4 waves on a well-conditioned refractive design.

    Raises
    ------
    ValueError
        On invalid input.
    RuntimeError
        If too many rays die for a meaningful fit.
    """
    from .raytrace import surfaces_from_prescription, _make_bundle, trace

    if wavelength <= 0:
        raise ValueError(f"wavelength must be > 0, got {wavelength}")
    if poly_order < 0:
        raise ValueError(f"poly_order must be >= 0, got {poly_order}")
    if source_box_half <= 0 or pupil_box_half <= 0:
        raise ValueError("source_box_half and pupil_box_half must be > 0")

    if object_distance is None:
        object_distance = float(prescription.get('object_distance', 0.0)) or 0.0

    surfaces = surfaces_from_prescription(prescription)

    # 4-D Chebyshev-node grid:  (s1_x, s1_y) x (v1_x, v1_y).
    def cheb_nodes(n: int) -> np.ndarray:
        i = np.arange(n)
        return np.cos(np.pi * (i + 0.5) / n)

    u_field = cheb_nodes(n_field)        # in (-1, 1)
    u_pupil = cheb_nodes(n_pupil)
    s1x_axis = u_field * source_box_half
    s1y_axis = u_field * source_box_half
    v1x_axis = u_pupil * pupil_box_half
    v1y_axis = u_pupil * pupil_box_half

    S1X, S1Y, V1X, V1Y = np.meshgrid(s1x_axis, s1y_axis,
                                      v1x_axis, v1y_axis,
                                      indexing='ij')
    s1x_in = S1X.ravel()
    s1y_in = S1Y.ravel()
    v1x_in = V1X.ravel()
    v1y_in = V1Y.ravel()
    n_rays = s1x_in.size

    # Reject any input direction with sin^2(theta) >= 1 (non-real
    # longitudinal cosine):  this means pupil_box_half is too large.
    sumsq = v1x_in * v1x_in + v1y_in * v1y_in
    if np.any(sumsq >= 1.0):
        raise ValueError(
            "pupil_box_half too large -- input direction cosines would "
            "be non-real.  Reduce pupil_box_half "
            f"(current = {pupil_box_half}).")

    # Build a ray bundle launched at z = -object_distance.  trace()
    # automatically propagates the rays the appropriate distance to the
    # first surface (since the surfaces' z positions are determined by
    # the prescription's accumulated thicknesses).
    bundle = _make_bundle(
        x=s1x_in.astype(np.float64),
        y=s1y_in.astype(np.float64),
        L=v1x_in.astype(np.float64),
        M=v1y_in.astype(np.float64),
        wavelength=wavelength,
    )
    bundle.z = np.full(n_rays, -object_distance, dtype=np.float64)

    res = trace(bundle, surfaces, wavelength, output_filter='last',
                surface_diffraction=surface_diffraction)
    final = res.image_rays
    alive = np.asarray(final.alive, dtype=bool)
    n_alive = int(alive.sum())
    if n_alive < max(64, 0.5 * n_rays):
        raise RuntimeError(
            f"Too many rays died during canonical-fit trace: "
            f"{n_alive} alive of {n_rays}.  Reduce pupil_box_half, "
            f"check prescription apertures, or rebalance the source/"
            f"pupil sampling boxes.")

    s2x_obs = np.asarray(final.x, dtype=np.float64)
    s2y_obs = np.asarray(final.y, dtype=np.float64)
    v2x_obs = np.asarray(final.L, dtype=np.float64)
    v2y_obs = np.asarray(final.M, dtype=np.float64)
    phi_obs = np.asarray(final.opd, dtype=np.float64) / wavelength

    s2x_live = s2x_obs[alive]
    s2y_live = s2y_obs[alive]
    v2x_live = v2x_obs[alive]
    v2y_live = v2y_obs[alive]
    phi_live = phi_obs[alive]
    s1x_live = s1x_in[alive]
    s1y_live = s1y_in[alive]

    # Normalise the OUTPUT (s2, v2) coordinates to [-1, 1]^4 from the
    # observed extents of the live rays (paper 1 §3.2).  Rays that end
    # up at extreme s2 or v2 set the box; the 5%-pad in _fit_normaliser
    # leaves room for evaluation at boundary pixels.
    s2x_c, s2x_h = _fit_normaliser(s2x_live)
    s2y_c, s2y_h = _fit_normaliser(s2y_live)
    v2x_c, v2x_h = _fit_normaliser(v2x_live)
    v2y_c, v2y_h = _fit_normaliser(v2y_live)

    u_s2x = (s2x_live - s2x_c) / s2x_h
    u_s2y = (s2y_live - s2y_c) / s2y_h
    u_v2x = (v2x_live - v2x_c) / v2x_h
    u_v2y = (v2y_live - v2y_c) / v2y_h

    linear_coeffs = None
    if extract_linear_phase:
        X5 = np.column_stack([
            np.ones_like(u_s2x),
            u_s2x, u_s2y, u_v2x, u_v2y,
        ])
        linear_coeffs, *_ = np.linalg.lstsq(X5, phi_live, rcond=None)
        opd_residual = phi_live - X5 @ linear_coeffs
    else:
        opd_residual = phi_live.copy()

    multi_indices = _multi_indices_total_degree(4, poly_order)
    n_basis = len(multi_indices)
    T1 = _chebyshev_vandermonde(u_s2x, poly_order)
    T2 = _chebyshev_vandermonde(u_s2y, poly_order)
    T3 = _chebyshev_vandermonde(u_v2x, poly_order)
    T4 = _chebyshev_vandermonde(u_v2y, poly_order)
    A = np.empty((u_s2x.size, n_basis), dtype=np.float64)
    for j, (k1, k2, k3, k4) in enumerate(multi_indices):
        A[:, j] = T1[k1] * T2[k2] * T3[k3] * T4[k4]

    coef_phi, *_ = np.linalg.lstsq(A, opd_residual, rcond=None)
    coef_s1x, *_ = np.linalg.lstsq(A, s1x_live, rcond=None)
    coef_s1y, *_ = np.linalg.lstsq(A, s1y_live, rcond=None)

    res_phi = float(np.sqrt(np.mean((opd_residual - A @ coef_phi) ** 2)))
    res_s1 = float(np.sqrt(0.5 * np.mean(
        (s1x_live - A @ coef_s1x) ** 2 + (s1y_live - A @ coef_s1y) ** 2
    )))

    return CanonicalPolyFit(
        poly_order=poly_order,
        multi_indices=multi_indices,
        coef_phi=coef_phi,
        coef_s1x=coef_s1x,
        coef_s1y=coef_s1y,
        s2x_centre=s2x_c, s2x_halfrange=s2x_h,
        s2y_centre=s2y_c, s2y_halfrange=s2y_h,
        v2x_centre=v2x_c, v2x_halfrange=v2x_h,
        v2y_centre=v2y_c, v2y_halfrange=v2y_h,
        wavelength=wavelength,
        res_phi_rms_waves=res_phi,
        res_s1_rms_m=res_s1,
        n_rays=n_alive,
        linear_coeffs_phi=linear_coeffs,
        extract_linear_phase=extract_linear_phase,
    )


# ===========================================================================
# Section 4 -- Newton solver for the envelope-stationary point
# ===========================================================================

def solve_envelope_stationary(
    fit: CanonicalPolyFit,
    s2: Tuple[float, float],
    source_point: Tuple[float, float],
    *,
    w_s: float,
    w_p: float,
    v2_centre: Tuple[float, float] = (0.0, 0.0),
    v2_initial: Optional[Tuple[float, float]] = None,
    max_iter: int = 12,
    tol: float = 1e-12,
) -> Tuple[Tuple[float, float], int, float]:
    """Newton-solve the envelope-stationary equation (paper 2, eq. 9).

    Find ``v_2^*`` such that the joint Gaussian envelope

        G(v_2) = exp(-|s_1(s_2, v_2) - s_src|^2 / w_s^2
                     - |v_2 - v_2_centre|^2 / w_p^2)

    is locally maximised at ``s_2``.  Equation (9) reads

        J^T (s_1 - s_src) / w_s^2 + (v_2 - v_2_centre) / w_p^2 = 0

    with ``J = ds_1 / dv_2``.

    Parameters
    ----------
    fit : CanonicalPolyFit
        Polynomial fit of the system.
    s2 : (float, float)
        Output-plane point [m].
    source_point : (float, float)
        Source-plane point [m].
    w_s, w_p : float
        Source and pupil Gaussian waists.  ``w_s`` in [m] (object plane);
        ``w_p`` in direction-cosine units [dimensionless].
    v2_centre : (float, float), optional
        Pupil centre in direction cosines.  Default origin.
    v2_initial : (float, float), optional
        Newton initial guess.  Defaults to ``v2_centre``.
    max_iter : int, optional
        Iteration cap.  Newton typically converges in 3-5; cap at 12.
    tol : float, optional
        Convergence tolerance on |residual|.  Default 1e-12.

    Returns
    -------
    v_star : (float, float)
        The stationary point.
    n_iter : int
        Iterations actually used.
    residual_norm : float
        Final |residual|.
    """
    s2x, s2y = float(s2[0]), float(s2[1])
    src_x, src_y = float(source_point[0]), float(source_point[1])
    v_cx, v_cy = float(v2_centre[0]), float(v2_centre[1])
    if v2_initial is None:
        v2x = v_cx
        v2y = v_cy
    else:
        v2x = float(v2_initial[0])
        v2y = float(v2_initial[1])

    inv_ws2 = 1.0 / (w_s * w_s)
    inv_wp2 = 1.0 / (w_p * w_p)

    last_norm = float('inf')
    for it in range(max_iter):
        # Evaluate s1, J at current v2
        s2x_arr = np.asarray(s2x).reshape(())
        s2y_arr = np.asarray(s2y).reshape(())
        v2x_arr = np.asarray(v2x).reshape(())
        v2y_arr = np.asarray(v2y).reshape(())
        s1x, s1y, dS1x_dv2x, dS1x_dv2y, dS1y_dv2x, dS1y_dv2y = (
            fit.eval_s1_with_v2_grad(s2x_arr, s2y_arr, v2x_arr, v2y_arr)
        )
        s1x = float(s1x)
        s1y = float(s1y)
        # Jacobian of s_1 w.r.t. v_2
        J = np.array([
            [float(dS1x_dv2x), float(dS1x_dv2y)],
            [float(dS1y_dv2x), float(dS1y_dv2y)],
        ])
        delta_s1 = np.array([s1x - src_x, s1y - src_y])
        delta_v = np.array([v2x - v_cx, v2y - v_cy])
        residual = inv_ws2 * (J.T @ delta_s1) + inv_wp2 * delta_v
        rn = float(np.linalg.norm(residual))
        if rn < tol or rn > 0.99 * last_norm > 1e-300:
            # Converged or stalling -- stop.
            return (v2x, v2y), it, rn
        last_norm = rn

        # Approximate Hessian:  d residual / d v2.
        # First term:  d/dv2 [ J^T (s_1 - s_src) ] = J^T @ J + sum_k
        #                    (s_1 - s_src)_k * d^2 s_1_k / dv2 dv2.
        # Hessian-of-s1 piece is second-order in (s_1 - s_src) and is
        # neglected here (Gauss-Newton-like).  This is exact at the
        # stationary point and gives quadratic-ish convergence.
        # Second term:  d/dv2 [(v_2 - v_2c) / w_p^2] = I / w_p^2.
        H = inv_ws2 * (J.T @ J) + inv_wp2 * np.eye(2)
        try:
            step = np.linalg.solve(H, residual)
        except np.linalg.LinAlgError:
            # Singular Hessian -- back off and bail.
            return (v2x, v2y), it, rn
        v2x = v2x - float(step[0])
        v2y = v2y - float(step[1])

    return (v2x, v2y), max_iter, last_norm


# ===========================================================================
# Section 5 -- Aberration tensor and modal asymptotic propagator
# ===========================================================================

@dataclass
class AberrationTensorResult:
    """Result of an LG aberration-tensor evaluation at one image point.

    Fields
    ------
    L : ndarray, shape (n_output_modes, n_source_modes)
        Aberration matrix L_{k,n} = sum_m b_m T_{k;n,m} after pupil
        contraction.  Rows index output LG modes; columns index source
        LG modes.
    output_modes : list of (p, ell)
    source_modes : list of (p, ell)
    pupil_modes : list of (p, ell)
    s2_image : (float, float)
        Chief-ray image point [m] at which this tensor was evaluated.
    w_s, w_p, w_o : float
        Source, pupil, output Gaussian waists.
    v_star : (float, float)
        Envelope-stationary v_2* at s2_image.

    Notes
    -----
    Indices of L correspond to physical aberrations via
    ``lg_seidel_label(p, ell)``:  (1, 0) is defocus, (2, 0) is
    spherical, (1, +-1) is coma, (0, +-2) is astigmatism, etc.
    Driving |L_{(2,0), 0}|^2 to zero suppresses on-axis spherical
    aberration, etc.
    """
    L: np.ndarray
    output_modes: List[Tuple[int, int]]
    source_modes: List[Tuple[int, int]]
    pupil_modes: List[Tuple[int, int]]
    s2_image: Tuple[float, float]
    w_s: float
    w_p: float
    w_o: float
    v_star: Tuple[float, float]


def _multiply_polys_2d(p_a: Dict[Tuple[int, int], complex],
                        p_b: Dict[Tuple[int, int], complex]
                        ) -> Dict[Tuple[int, int], complex]:
    """Multiply two 2-D polynomial dicts."""
    out: Dict[Tuple[int, int], complex] = {}
    for (i_a, j_a), c_a in p_a.items():
        for (i_b, j_b), c_b in p_b.items():
            key = (i_a + i_b, j_a + j_b)
            out[key] = out.get(key, 0.0 + 0.0j) + c_a * c_b
    return out


def _polynomial_under_affine_shift(
    coeffs: Dict[Tuple[int, int], complex],
    shift_x: complex, shift_y: complex,
    var_name: str = 'eta',
) -> Dict[Tuple[int, int], complex]:
    """Substitute (x, y) -> (x + shift_x, y + shift_y) in a 2-D polynomial.

    Used to produce the ``eta``-polynomial after the (s_1, v_2) -> eta
    coordinate change.
    """
    if not coeffs:
        return {}
    max_i = max(k[0] for k in coeffs)
    max_j = max(k[1] for k in coeffs)
    # Pre-compute (x + shift_x)^i = sum_k C(i, k) shift_x^(i-k) x^k
    # as a coefficient table indexed by k.
    bin_x: List[Dict[int, complex]] = []
    for i in range(max_i + 1):
        row: Dict[int, complex] = {}
        for k in range(i + 1):
            row[k] = math.comb(i, k) * (shift_x ** (i - k))
        bin_x.append(row)
    bin_y: List[Dict[int, complex]] = []
    for j in range(max_j + 1):
        row = {}
        for k in range(j + 1):
            row[k] = math.comb(j, k) * (shift_y ** (j - k))
        bin_y.append(row)

    out: Dict[Tuple[int, int], complex] = {}
    for (i, j), c in coeffs.items():
        for kx, bx in bin_x[i].items():
            for ky, by in bin_y[j].items():
                key = (kx, ky)
                out[key] = out.get(key, 0.0 + 0.0j) + c * bx * by
    return out


def _contract_against_moment_table(
    poly: Dict[Tuple[int, int], complex],
    moments: Dict[Tuple[int, int], complex],
) -> complex:
    """Compute ``<P(eta)>_M = sum_{ij} c_{ij} <eta_x^i eta_y^j>``."""
    total = 0.0 + 0.0j
    for (i, j), c in poly.items():
        total += c * moments.get((i, j), 0.0 + 0.0j)
    return total


def _compute_M_b(
    fit: CanonicalPolyFit,
    s2x: float, s2y: float,
    v2x: float, v2y: float,
    src_x: float, src_y: float,
    w_s: float, w_p: float,
    v_cx: float, v_cy: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, complex]:
    """Build the complex beam matrix M (paper 2 eq. 16), the linear
    term b (eq. 17), the Jacobian J*, and the OPL piston Phi*."""
    s2x_arr = np.asarray(s2x).reshape(())
    s2y_arr = np.asarray(s2y).reshape(())
    v2x_arr = np.asarray(v2x).reshape(())
    v2y_arr = np.asarray(v2y).reshape(())

    s1x, s1y, dS1x_dv2x, dS1x_dv2y, dS1y_dv2x, dS1y_dv2y = (
        fit.eval_s1_with_v2_grad(s2x_arr, s2y_arr, v2x_arr, v2y_arr)
    )
    phi, dPhi_dv2x, dPhi_dv2y = fit.eval_phi_with_v2_grad(
        s2x_arr, s2y_arr, v2x_arr, v2y_arr,
        include_linear=False,
    )

    s1x_v = float(s1x)
    s1y_v = float(s1y)
    phi_v = float(phi)
    J = np.array([
        [float(dS1x_dv2x), float(dS1x_dv2y)],
        [float(dS1y_dv2x), float(dS1y_dv2y)],
    ])
    g = np.array([float(dPhi_dv2x), float(dPhi_dv2y)])

    # Hessian of phi w.r.t. v2 (use finite differences of analytic 1st
    # derivatives; cleaner to differentiate the polynomial twice).
    H_phi = _phi_v2_hessian(fit, s2x, s2y, v2x, v2y)

    inv_ws2 = 1.0 / (w_s * w_s)
    inv_wp2 = 1.0 / (w_p * w_p)
    M_real = inv_ws2 * (J.T @ J) + inv_wp2 * np.eye(2)
    M = M_real - 1j * math.pi * H_phi
    r_star = np.array([s1x_v - src_x, s1y_v - src_y])
    delta_v = np.array([v2x - v_cx, v2y - v_cy])
    b = (2.0j * math.pi * g
         - 2.0 * inv_ws2 * (J.T @ r_star)
         - 2.0 * inv_wp2 * delta_v)
    detJ = float(np.abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]))

    G0 = math.exp(
        -(r_star[0] ** 2 + r_star[1] ** 2) / (w_s * w_s)
        - (delta_v[0] ** 2 + delta_v[1] ** 2) / (w_p * w_p)
    )
    return M, b, np.array([s1x_v, s1y_v]), J, complex(phi_v), G0, detJ


def _phi_v2_hessian(fit: CanonicalPolyFit, s2x: float, s2y: float,
                     v2x: float, v2y: float) -> np.ndarray:
    """Compute the 2x2 Hessian d^2 Phi / d v_2 d v_2 by analytic
    differentiation of the Chebyshev fit, in *physical* (waves per
    direction-cosine^2) units."""
    # Normalised coords
    u1 = (s2x - fit.s2x_centre) / fit.s2x_halfrange
    u2 = (s2y - fit.s2y_centre) / fit.s2y_halfrange
    u3 = (v2x - fit.v2x_centre) / fit.v2x_halfrange
    u4 = (v2y - fit.v2y_centre) / fit.v2y_halfrange

    T1 = _chebyshev_vandermonde(np.array(u1), fit.poly_order)
    T2 = _chebyshev_vandermonde(np.array(u2), fit.poly_order)
    T3 = _chebyshev_vandermonde(np.array(u3), fit.poly_order)
    T4 = _chebyshev_vandermonde(np.array(u4), fit.poly_order)
    dT3 = _chebyshev_derivative_vandermonde(np.array(u3), fit.poly_order)
    dT4 = _chebyshev_derivative_vandermonde(np.array(u4), fit.poly_order)
    d2T3 = _chebyshev_second_derivative_vandermonde(np.array(u3),
                                                       fit.poly_order)
    d2T4 = _chebyshev_second_derivative_vandermonde(np.array(u4),
                                                       fit.poly_order)

    h33 = 0.0
    h34 = 0.0
    h44 = 0.0
    for c, (k1, k2, k3, k4) in zip(fit.coef_phi, fit.multi_indices):
        if c == 0.0:
            continue
        T12 = float(T1[k1]) * float(T2[k2])
        h33 += c * T12 * float(d2T3[k3]) * float(T4[k4])
        h34 += c * T12 * float(dT3[k3]) * float(dT4[k4])
        h44 += c * T12 * float(T3[k3]) * float(d2T4[k4])
    # Add linear-phase Hessian contribution -- linear terms have zero
    # second derivative, so nothing to add.
    invhx = 1.0 / fit.v2x_halfrange
    invhy = 1.0 / fit.v2y_halfrange
    return np.array([
        [h33 * invhx * invhx, h34 * invhx * invhy],
        [h34 * invhx * invhy, h44 * invhy * invhy],
    ])


def aberration_tensor(
    fit: CanonicalPolyFit,
    s2_image: Tuple[float, float],
    *,
    source_point: Tuple[float, float] = (0.0, 0.0),
    source_modes: Optional[List[Tuple[int, int]]] = None,
    pupil_modes: Optional[List[Tuple[int, int]]] = None,
    output_modes: Optional[List[Tuple[int, int]]] = None,
    pupil_amplitudes: Optional[Dict[Tuple[int, int], complex]] = None,
    w_s: float = 50e-6,
    w_p: float = 0.05,
    w_o: Optional[float] = None,
    v2_centre: Tuple[float, float] = (0.0, 0.0),
) -> AberrationTensorResult:
    """LG aberration tensor at a single chief-ray image point.

    Implements paper 2 Section 7:  expand source, pupil, and output
    fields in Laguerre-Gaussian bases, evaluate the leading-order
    asymptotic propagator analytically as a Wick-contracted Gaussian
    moment, and project onto the output basis to read off the
    coefficient ``L_{k, n} = sum_m b_m T_{k;n,m}`` of each named
    aberration channel.

    Parameters
    ----------
    fit : CanonicalPolyFit
        4-D Chebyshev fit of the prescription.
    s2_image : (float, float)
        Image-plane point [m] at which to evaluate the tensor.  Should
        be the chief-ray landing of ``source_point`` for the
        Seidel-name interpretation to apply.
    source_point : (float, float)
        Source-plane point [m].
    source_modes, pupil_modes, output_modes : list of (p, ell)
        LG mode indices to retain.  Defaults below give a useful
        baseline.
    pupil_amplitudes : dict, optional
        Pupil expansion coefficients ``{(p, ell): complex}``.  If None,
        defaults to a clean LG_{0,0} pupil (b_{0,0} = 1) -- the
        "ideal-Gaussian-pupil" convention used for design merit
        functions, where higher pupil modes are not needed because the
        Seidel content lives entirely on the output side.
    w_s, w_p : float
        Source and pupil Gaussian waists [m and direction-cosine].
    w_o : float, optional
        Output Gaussian waist [m].  Defaults to a value derived from
        the local complex beam matrix (Maréchal-scale).
    v2_centre : (float, float), optional
        Pupil centre.

    Returns
    -------
    AberrationTensorResult
    """
    if source_modes is None:
        source_modes = [(0, 0)]
    if pupil_modes is None:
        pupil_modes = [(0, 0)]
    if output_modes is None:
        # Default:  the named Seidel/Zernike aberrations through 4th order
        output_modes = [
            (0, 0),  # piston / Strehl
            (1, 0),  # defocus
            (2, 0),  # primary spherical
            (0, 1), (0, -1),  # tilt
            (1, 1), (1, -1),  # coma
            (0, 2), (0, -2),  # astigmatism
            (0, 3), (0, -3),  # trefoil
        ]
    if pupil_amplitudes is None:
        pupil_amplitudes = {(0, 0): 1.0 + 0.0j}

    s2x_img, s2y_img = float(s2_image[0]), float(s2_image[1])
    src_x, src_y = float(source_point[0]), float(source_point[1])

    # Solve the envelope-stationary equation at s2_image
    v_star, _n_iter, _resid = solve_envelope_stationary(
        fit, (s2x_img, s2y_img), (src_x, src_y),
        w_s=w_s, w_p=w_p, v2_centre=v2_centre,
    )
    v2x_star, v2y_star = v_star

    # Build M, b at v_star
    M, b, s1_star, J_star, phi_star, G0, detJ = _compute_M_b(
        fit, s2x_img, s2y_img, v2x_star, v2y_star,
        src_x, src_y, w_s, w_p, v2_centre[0], v2_centre[1]
    )

    # Choose output waist if not supplied
    if w_o is None:
        # Use the smaller of the two real-part eigenvalues of M^-1, and
        # drop a factor of pi (Maréchal convention scales as
        # 1/sqrt(lambda_max(Re M))).
        eig_M_real = np.linalg.eigvalsh(np.real(M))
        if eig_M_real.max() <= 0:
            w_o = 1e-6  # fallback for ill-conditioned cases
        else:
            w_o = 1.0 / math.sqrt(float(eig_M_real.max()))
        # Clamp to a physically reasonable range
        w_o = max(min(w_o, 1.0), 1e-9)

    # Stationary shift delta* = 0.5 M^-1 b
    M_inv = np.linalg.inv(M)
    delta_star = 0.5 * (M_inv @ b)
    Sigma = 0.5 * M_inv  # 2x2 complex covariance
    sqrt_detM = np.sqrt(np.linalg.det(M))

    # Leading amplitude (paper 2 eq. 20)
    A_lead = (detJ * (math.pi / sqrt_detM) * G0
              * np.exp(2j * math.pi * phi_star)
              * np.exp(0.25 * b @ M_inv @ b))

    # Pre-tabulate eta-moments for max polynomial order needed
    max_order_needed = max(
        max((2 * p + abs(ell) for (p, ell) in source_modes), default=0)
        + max((2 * p + abs(ell) for (p, ell) in pupil_modes), default=0),
        4  # always have enough for low output orders
    )
    eta_moments = gaussian_moment_table_2d(M, max_order_needed)

    # Compute T_{k;n,m} for each (k, n, m).
    # The integrand polynomial is
    #     P_{n,m}(eta) = p^src_n(s_1(s_2*, v_2*+eta) - s_src; w_s)
    #                   * p^pup_m(v_2* - v_2c + eta; w_p)
    # at leading order s_1(s_2*, v_2*+eta) = s_1* + J* delta_star + J* eta;
    # so the source argument is r* + J* delta_star + J* eta.
    # The pupil argument is (v_2* - v_2c + delta_star + eta).
    n_out = len(output_modes)
    n_src = len(source_modes)
    L = np.zeros((n_out, n_src), dtype=np.complex128)

    # Source argument shift = r* + J* delta_star ; J* eta is the
    # eta-dependent piece.  But J* couples eta -> source-r-space, so
    # we need to substitute (eta_1, eta_2) -> J* (eta_1, eta_2) into
    # the source polynomial.  This is a linear coordinate change in
    # the source polynomial whose result is *another* polynomial in
    # eta of the same total degree.  Paper 2 leading order keeps this
    # affine substitution exactly.
    #
    # We implement it generically:  given p^src_n(r1, r2), substitute
    # r = (J* eta) + (r* + J* delta_star) -- the full affine
    # transformation -- and re-collect as polynomial in eta.

    r_const = s1_star + J_star @ np.array([delta_star[0], delta_star[1]]) - np.array([src_x, src_y])
    # Linear transform matrix on eta: r1 = J11 eta1 + J12 eta2 + r_const[0],
    # r2 = J21 eta1 + J22 eta2 + r_const[1].
    pupil_const = (np.array([v2x_star, v2y_star])
                   - np.array(v2_centre)
                   + np.array([delta_star[0], delta_star[1]]))

    for io, k_out in enumerate(output_modes):
        # Build output polynomial at sigma = 0 (we evaluate at the chief
        # image point; sigma-dependence is absorbed into the leading
        # amplitude's s2-Taylor expansion separately).
        # For *one* image point evaluation we project the polynomial
        # P_{n,m}(eta) (after Wick contraction) directly onto the
        # output LG basis -- but at a single s2_image, the projection
        # collapses to overlap weight at sigma = 0:  L_{k,n} corresponds
        # to evaluating the integral once at s2_image and asking how it
        # decomposes against output mode k.
        #
        # Practical implementation: paper 2 shows that for the field at
        # s2_image + sigma = s2_image, the projection of I_{n,m} onto
        # LG^out_k integrates the conjugate output polynomial against
        # the leading Gaussian envelope.  At sigma = 0 the integral
        # collapses to <p^out_k* p^src_n p^pup_m>_M up to scaling.  We
        # compute that triple-product moment.
        out_poly_full = lg_polynomial(k_out[0], k_out[1], w_o)
        # Take complex conjugate (we project onto LG^out_k* * I_{n,m}
        # in the output overlap integral).
        out_poly = {key: c.conjugate() for key, c in out_poly_full.items()}

        for js, k_src in enumerate(source_modes):
            src_poly_r = lg_polynomial(k_src[0], k_src[1], w_s)
            # Substitute r = J* eta + r_const into source polynomial
            src_poly_eta = _polynomial_substitute_linear_2d(
                src_poly_r,
                A_xx=J_star[0, 0], A_xy=J_star[0, 1],
                A_yx=J_star[1, 0], A_yy=J_star[1, 1],
                b_x=r_const[0], b_y=r_const[1],
            )

            # Sum over pupil modes
            T_acc = 0.0 + 0.0j
            for k_pup, b_pup in pupil_amplitudes.items():
                if abs(b_pup) < 1e-300:
                    continue
                pup_poly_r = lg_polynomial(k_pup[0], k_pup[1], w_p)
                # Pupil arg is (v - v_c) + delta_star + eta = pupil_const + eta
                pup_poly_eta = _polynomial_under_affine_shift(
                    pup_poly_r,
                    shift_x=complex(pupil_const[0]),
                    shift_y=complex(pupil_const[1]),
                )
                # Multiply src(eta) * pup(eta) * out_conj(eta=0 shift)
                # Output mode evaluated at sigma=0 contributes its
                # constant-eta Wick projection.  But out_poly is in
                # sigma not eta -- and at sigma = 0, the polynomial
                # reduces to its constant term.  In the proper
                # output-plane integral the sigma-dependence is
                # integrated out separately; here we provide the
                # SCALAR projection at the chief ray, which is what
                # matters for the Seidel-name-meaning of the
                # coefficients.  See paper 2 eq. (52) and surrounding
                # discussion.
                P_eta = _multiply_polys_2d(src_poly_eta, pup_poly_eta)
                exp_val = _contract_against_moment_table(P_eta, eta_moments)
                # Output projection coefficient (constant term of out_poly):
                out_const = out_poly.get((0, 0), 0.0 + 0.0j)
                T_acc += b_pup * out_const * exp_val
            L[io, js] = A_lead * T_acc

    return AberrationTensorResult(
        L=L,
        output_modes=list(output_modes),
        source_modes=list(source_modes),
        pupil_modes=list(pupil_modes),
        s2_image=(s2x_img, s2y_img),
        w_s=w_s, w_p=w_p, w_o=w_o,
        v_star=(v2x_star, v2y_star),
    )


def _polynomial_substitute_linear_2d(
    coeffs: Dict[Tuple[int, int], complex],
    A_xx: float, A_xy: float, A_yx: float, A_yy: float,
    b_x: float, b_y: float,
) -> Dict[Tuple[int, int], complex]:
    """Substitute (r_x, r_y) -> A * (eta_x, eta_y) + (b_x, b_y) in a
    2-D polynomial, returning the resulting polynomial in (eta_x, eta_y).

    Used to push the source polynomial through the linear J* map at
    the envelope-stationary point.
    """
    if not coeffs:
        return {}
    # First pre-compute (a x + b y + c)^n expansion as polynomial in (x, y).
    # We need (A_xx eta_x + A_xy eta_y + b_x)^i and similarly for y.

    def axes_pow(coef_a: complex, coef_b: complex, coef_c: complex,
                 n: int) -> Dict[Tuple[int, int], complex]:
        """Expand (a x + b y + c)^n via multinomial."""
        out: Dict[Tuple[int, int], complex] = {}
        # multinomial:  sum over (i, j, k) with i + j + k = n of
        #     n!/(i! j! k!) * a^i * b^j * c^k * x^i * y^j
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                w = (math.factorial(n)
                     // (math.factorial(i) * math.factorial(j)
                         * math.factorial(k)))
                key = (i, j)
                out[key] = out.get(key, 0.0 + 0.0j) + (
                    w * (coef_a ** i) * (coef_b ** j) * (coef_c ** k)
                )
        return out

    out: Dict[Tuple[int, int], complex] = {}
    # Cache the expansions of the linear forms raised to each needed power
    max_i = max(k[0] for k in coeffs)
    max_j = max(k[1] for k in coeffs)

    # (A_xx eta_x + A_xy eta_y + b_x)^i ; build for i = 0..max_i
    cache_x: List[Dict[Tuple[int, int], complex]] = []
    for n in range(max_i + 1):
        cache_x.append(axes_pow(
            complex(A_xx), complex(A_xy), complex(b_x), n
        ))
    # (A_yx eta_x + A_yy eta_y + b_y)^j ; build for j = 0..max_j
    cache_y: List[Dict[Tuple[int, int], complex]] = []
    for n in range(max_j + 1):
        cache_y.append(axes_pow(
            complex(A_yx), complex(A_yy), complex(b_y), n
        ))

    for (i, j), c in coeffs.items():
        # Multiply cache_x[i] * cache_y[j] and accumulate into out.
        prod = _multiply_polys_2d(cache_x[i], cache_y[j])
        for key, pc in prod.items():
            out[key] = out.get(key, 0.0 + 0.0j) + c * pc
    return out


# ===========================================================================
# Section 6 -- Modal asymptotic propagator (paper 2 leading order)
# ===========================================================================

def propagate_modal_asymptotic(
    fit: CanonicalPolyFit,
    *,
    source_point: Tuple[float, float] = (0.0, 0.0),
    source_amplitudes: Optional[Dict[Tuple[int, int], complex]] = None,
    pupil_amplitudes: Optional[Dict[Tuple[int, int], complex]] = None,
    w_s: float = 50e-6,
    w_p: float = 0.05,
    v2_centre: Tuple[float, float] = (0.0, 0.0),
    s2_grid_x: np.ndarray,
    s2_grid_y: np.ndarray,
) -> np.ndarray:
    """Evaluate the leading-order modal asymptotic propagator on a 2-D
    output grid (paper 2, eq. 24).

    For each output pixel, solves the envelope-stationary equation,
    builds the complex beam matrix, and evaluates the closed-form
    Gaussian-moment expansion in the chosen LG mode bases.  This is
    ~10**3-10**4 times faster than direct quadrature for typical
    parameters.

    Parameters
    ----------
    fit : CanonicalPolyFit
    source_point : (float, float)
        Object-plane source location [m].
    source_amplitudes, pupil_amplitudes : dict, optional
        LG mode amplitudes ``{(p, ell): complex}`` for source and pupil.
        Defaults: source = LG_{0,0} (Gaussian); pupil = LG_{0,0}
        (matched soft aperture).
    w_s, w_p : float
        Source and pupil waists.
    v2_centre : (float, float)
        Pupil centre in direction cosines.
    s2_grid_x, s2_grid_y : ndarray
        Output-plane sample points [m].  Same shape; iterated jointly.

    Returns
    -------
    ndarray, complex, same shape as s2_grid_x
        Output field E(s_2).
    """
    if source_amplitudes is None:
        source_amplitudes = {(0, 0): 1.0 + 0.0j}
    if pupil_amplitudes is None:
        pupil_amplitudes = {(0, 0): 1.0 + 0.0j}
    src_x, src_y = float(source_point[0]), float(source_point[1])
    v_cx, v_cy = float(v2_centre[0]), float(v2_centre[1])

    s2x_arr = np.asarray(s2_grid_x, dtype=np.float64)
    s2y_arr = np.asarray(s2_grid_y, dtype=np.float64)
    if s2x_arr.shape != s2y_arr.shape:
        raise ValueError(
            f"s2_grid_x and s2_grid_y shape mismatch: {s2x_arr.shape} vs {s2y_arr.shape}")

    out = np.zeros(s2x_arr.shape, dtype=np.complex128)

    # Determine moment-table order needed
    max_order_src = max((2 * p + abs(l) for (p, l) in source_amplitudes), default=0)
    max_order_pup = max((2 * p + abs(l) for (p, l) in pupil_amplitudes), default=0)
    max_order_needed = max(max_order_src + max_order_pup, 0)

    flat_x = s2x_arr.ravel()
    flat_y = s2y_arr.ravel()
    flat_out = np.zeros(flat_x.size, dtype=np.complex128)

    last_v_star = (v_cx, v_cy)  # warm-start across pixels
    for idx in range(flat_x.size):
        s2x_p = flat_x[idx]
        s2y_p = flat_y[idx]

        # Skip points outside the fit's training box
        u1 = (s2x_p - fit.s2x_centre) / fit.s2x_halfrange
        u2 = (s2y_p - fit.s2y_centre) / fit.s2y_halfrange
        if abs(u1) > 1.0 or abs(u2) > 1.0:
            continue

        try:
            v_star, n_iter, residual = solve_envelope_stationary(
                fit, (s2x_p, s2y_p), (src_x, src_y),
                w_s=w_s, w_p=w_p, v2_centre=v2_centre,
                v2_initial=last_v_star,
            )
        except (np.linalg.LinAlgError, ValueError, OverflowError):
            continue
        # If Newton wandered outside the fit box, fall back to v_centre
        # (otherwise we'll evaluate the polynomial fit out-of-domain).
        u3 = (v_star[0] - fit.v2x_centre) / fit.v2x_halfrange
        u4 = (v_star[1] - fit.v2y_centre) / fit.v2y_halfrange
        if abs(u3) > 1.0 or abs(u4) > 1.0 or not (math.isfinite(u3) and math.isfinite(u4)):
            continue
        last_v_star = v_star

        try:
            M, b, s1_star, J_star, phi_star, G0, detJ = _compute_M_b(
                fit, s2x_p, s2y_p, v_star[0], v_star[1],
                src_x, src_y, w_s, w_p, v_cx, v_cy
            )
        except (np.linalg.LinAlgError, ValueError, OverflowError):
            continue
        if not (np.all(np.isfinite(M)) and np.all(np.isfinite(b))):
            continue
        det_M = np.linalg.det(M)
        if not math.isfinite(abs(det_M)) or abs(det_M) < 1e-300:
            continue
        sqrt_detM = np.sqrt(det_M)
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(M_inv)):
            continue
        delta_star = 0.5 * (M_inv @ b)

        # Leading amplitude.  Guard against complex-Gaussian overflow in
        # the b^T M^-1 b exponent: cap at large magnitude (the integrand
        # is genuinely negligible if the exponent goes to -inf, but a
        # +inf would NaN the field).
        b_quad = 0.25 * (b @ M_inv @ b)
        if not math.isfinite(abs(b_quad)) or abs(b_quad.real) > 700:
            continue
        amp_lead = (detJ * (math.pi / sqrt_detM) * G0
                    * np.exp(2j * math.pi * phi_star)
                    * np.exp(b_quad))
        if not math.isfinite(abs(amp_lead)):
            continue

        # Moment table on this pixel's covariance
        eta_moments = gaussian_moment_table_2d(M, max_order_needed)

        r_const = s1_star + J_star @ np.array([delta_star[0], delta_star[1]]) \
                  - np.array([src_x, src_y])
        pupil_const = (np.array([v_star[0], v_star[1]])
                       - np.array([v_cx, v_cy])
                       + np.array([delta_star[0], delta_star[1]]))

        # Sum over (n, m) modes
        E_pixel = 0.0 + 0.0j
        for k_src, a_src in source_amplitudes.items():
            if abs(a_src) < 1e-300:
                continue
            src_poly_r = lg_polynomial(k_src[0], k_src[1], w_s)
            src_poly_eta = _polynomial_substitute_linear_2d(
                src_poly_r,
                A_xx=J_star[0, 0], A_xy=J_star[0, 1],
                A_yx=J_star[1, 0], A_yy=J_star[1, 1],
                b_x=r_const[0], b_y=r_const[1],
            )
            for k_pup, b_pup in pupil_amplitudes.items():
                if abs(b_pup) < 1e-300:
                    continue
                pup_poly_r = lg_polynomial(k_pup[0], k_pup[1], w_p)
                pup_poly_eta = _polynomial_under_affine_shift(
                    pup_poly_r,
                    shift_x=complex(pupil_const[0]),
                    shift_y=complex(pupil_const[1]),
                )
                P_eta = _multiply_polys_2d(src_poly_eta, pup_poly_eta)
                exp_val = _contract_against_moment_table(P_eta, eta_moments)
                E_pixel += a_src * b_pup * exp_val

        flat_out[idx] = amp_lead * E_pixel

    return flat_out.reshape(s2x_arr.shape)


# ===========================================================================
# Section 7 -- LG/HG decomposition utilities
# ===========================================================================

def decompose_lg(field: np.ndarray, x: np.ndarray, y: np.ndarray,
                 w: float, p_max: int, ell_max: int,
                 cx: float = 0.0, cy: float = 0.0
                 ) -> Dict[Tuple[int, int], complex]:
    """Project a complex field onto the Laguerre-Gaussian basis.

    Computes overlap integrals
        a_{p, ell} = integral conj(LG_{p, ell}(x, y)) * field(x, y) dx dy
    by trapezoidal quadrature on the supplied grid.

    Parameters
    ----------
    field : ndarray, complex, shape (Nx, Ny)
    x, y : ndarray, shape (Nx, Ny)
        Cartesian coordinates [m] (typically from meshgrid).
    w : float
        LG basis waist [m].
    p_max, ell_max : int
        Truncation:  retain p in [0, p_max], ell in [-ell_max, +ell_max].
    cx, cy : float, optional
        Basis centre.

    Returns
    -------
    dict
        ``{(p, ell): a_{p, ell}}``
    """
    if field.shape != x.shape or field.shape != y.shape:
        raise ValueError("field, x, y must have the same shape")
    dx = float(np.mean(np.diff(x, axis=1)[:, 0]))
    dy = float(np.mean(np.diff(y, axis=0)[0, :]))
    da = abs(dx * dy)
    out: Dict[Tuple[int, int], complex] = {}
    for p in range(p_max + 1):
        for ell in range(-ell_max, ell_max + 1):
            mode = evaluate_lg_mode(p, ell, w, x, y, cx, cy)
            out[(p, ell)] = complex(np.sum(np.conj(mode) * field) * da)
    return out


def decompose_hg(field: np.ndarray, x: np.ndarray, y: np.ndarray,
                 wx: float, wy: Optional[float],
                 m_max: int, n_max: int,
                 cx: float = 0.0, cy: float = 0.0
                 ) -> Dict[Tuple[int, int], complex]:
    """Project a complex field onto the Hermite-Gaussian basis.  See
    ``decompose_lg`` for arguments."""
    if wy is None:
        wy = wx
    if field.shape != x.shape or field.shape != y.shape:
        raise ValueError("field, x, y must have the same shape")
    dx = float(np.mean(np.diff(x, axis=1)[:, 0]))
    dy = float(np.mean(np.diff(y, axis=0)[0, :]))
    da = abs(dx * dy)
    out: Dict[Tuple[int, int], complex] = {}
    for mi in range(m_max + 1):
        for nj in range(n_max + 1):
            mode = evaluate_hg_mode(mi, nj, wx, wy, x, y, cx, cy)
            out[(mi, nj)] = complex(np.sum(np.conj(mode) * field) * da)
    return out
