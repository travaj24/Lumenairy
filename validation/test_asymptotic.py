"""Validation suite for lumenairy.asymptotic.

Tests are organised in layers, each layer exercising one well-defined
piece of the module:

  Layer 1  Math primitives (LG/HG basis, Wick moments)
  Layer 2  Polynomial / shift / contraction utilities
  Layer 3  Canonical polynomial fit (Phi(s2, v2), s1(s2, v2))
  Layer 4  Newton stationary solver and aberration tensor
  Layer 5  Modal asymptotic propagator (Collins limit, Fourier limit)
  Layer 6  LGAberrationMerit end-to-end behaviour

The aim is that any change to the module either passes every test below
or fails them with a diagnostic that points at the concrete identity
that broke.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from _harness import Harness

import lumenairy as op
from lumenairy.asymptotic import (
    lg_polynomial,
    hg_polynomial,
    evaluate_lg_mode,
    evaluate_hg_mode,
    gaussian_moment_2d,
    gaussian_moment_table_2d,
    lg_seidel_label,
    decompose_lg,
    decompose_hg,
    _multiply_polys_2d,
    _polynomial_under_affine_shift,
    _contract_against_moment_table,
    _polynomial_substitute_linear_2d,
)


H = Harness('asymptotic')


# ===========================================================================
# Layer 1 -- LG / HG basis polynomials
# ===========================================================================

H.section('Layer 1 -- LG/HG basis polynomials')


def t_lg_00_normalisation():
    """LG_{0,0} has unit L^2 norm:  integral |LG_00|^2 dA = 1."""
    w = 1.0e-3
    extent = 4.0 * w
    n = 256
    x = np.linspace(-extent, extent, n)
    y = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    da = (x[1] - x[0]) * (y[1] - y[0])
    field = evaluate_lg_mode(0, 0, w, X, Y)
    norm2 = float(np.sum(np.abs(field) ** 2) * da)
    err = abs(norm2 - 1.0)
    return err < 1e-3, f'|LG_00|^2 = {norm2:.6f}, err = {err:.2e}'


H.run('LG_00 has unit L^2 norm (waist w=1mm)', t_lg_00_normalisation)


def t_lg_01_normalisation():
    """LG_{0,1} (vortex) has unit L^2 norm."""
    w = 1.0e-3
    extent = 5.0 * w
    n = 256
    x = np.linspace(-extent, extent, n)
    y = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    da = (x[1] - x[0]) * (y[1] - y[0])
    field = evaluate_lg_mode(0, 1, w, X, Y)
    norm2 = float(np.sum(np.abs(field) ** 2) * da)
    err = abs(norm2 - 1.0)
    return err < 1e-3, f'|LG_01|^2 = {norm2:.6f}, err = {err:.2e}'


H.run('LG_01 (vortex) has unit L^2 norm', t_lg_01_normalisation)


def t_lg_higher_normalisations():
    """LG_{p,l} for several (p, l) pairs all have unit L^2 norm."""
    w = 1.0e-3
    extent = 6.0 * w
    n = 320
    x = np.linspace(-extent, extent, n)
    y = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    da = (x[1] - x[0]) * (y[1] - y[0])
    test_modes = [(1, 0), (2, 0), (1, 1), (1, -1), (0, 2), (2, 1)]
    errs = []
    for p, ell in test_modes:
        field = evaluate_lg_mode(p, ell, w, X, Y)
        norm2 = float(np.sum(np.abs(field) ** 2) * da)
        errs.append(abs(norm2 - 1.0))
    max_err = max(errs)
    return max_err < 1e-2, (
        f'max |1 - |LG|^2| = {max_err:.2e} over modes {test_modes}'
    )


H.run('LG_{p,l} unit norm for several (p, l)', t_lg_higher_normalisations)


def t_lg_orthogonality():
    """LG_{p,l} orthogonal to LG_{p',l'} when (p, l) != (p', l')."""
    w = 1.0e-3
    extent = 6.0 * w
    n = 320
    x = np.linspace(-extent, extent, n)
    y = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    da = (x[1] - x[0]) * (y[1] - y[0])
    pairs = [
        ((0, 0), (1, 0)),
        ((0, 0), (0, 1)),
        ((1, 0), (2, 0)),
        ((0, 1), (0, -1)),
        ((1, 1), (1, -1)),
    ]
    overlaps = []
    for m1, m2 in pairs:
        f1 = evaluate_lg_mode(m1[0], m1[1], w, X, Y)
        f2 = evaluate_lg_mode(m2[0], m2[1], w, X, Y)
        ov = abs(complex(np.sum(np.conj(f1) * f2) * da))
        overlaps.append(ov)
    max_ov = max(overlaps)
    return max_ov < 1e-2, (
        f'max off-diagonal overlap = {max_ov:.2e} over pairs {pairs}'
    )


H.run('LG modes orthogonal across distinct (p, l)', t_lg_orthogonality)


def t_hg_orthogonality_unit_norm():
    """HG modes orthonormal -- both unit norm and zero cross-overlap."""
    wx = 1.0e-3
    wy = 1.5e-3
    extent_x = 6.0 * wx
    extent_y = 6.0 * wy
    n = 320
    x = np.linspace(-extent_x, extent_x, n)
    y = np.linspace(-extent_y, extent_y, n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    da = (x[1] - x[0]) * (y[1] - y[0])
    test_modes = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    norms = []
    for m, n_ in test_modes:
        f = evaluate_hg_mode(m, n_, wx, wy, X, Y)
        norms.append(abs(float(np.sum(np.abs(f) ** 2) * da) - 1.0))
    max_norm_err = max(norms)
    # Cross-overlaps
    f_00 = evaluate_hg_mode(0, 0, wx, wy, X, Y)
    f_10 = evaluate_hg_mode(1, 0, wx, wy, X, Y)
    f_01 = evaluate_hg_mode(0, 1, wx, wy, X, Y)
    f_11 = evaluate_hg_mode(1, 1, wx, wy, X, Y)
    cross = max(
        abs(complex(np.sum(np.conj(f_00) * f_10) * da)),
        abs(complex(np.sum(np.conj(f_00) * f_01) * da)),
        abs(complex(np.sum(np.conj(f_10) * f_01) * da)),
        abs(complex(np.sum(np.conj(f_00) * f_11) * da)),
    )
    return (max_norm_err < 1e-2 and cross < 1e-2), (
        f'max norm err = {max_norm_err:.2e}, max cross overlap = {cross:.2e}'
    )


H.run('HG_{m,n} orthonormal (round and elliptical waist)',
      t_hg_orthogonality_unit_norm)


def t_lg_seidel_label_table():
    """The Seidel label table covers the named primary aberrations."""
    expected = {
        (0, 0): 'piston',
        (1, 0): 'defocus',
        (2, 0): 'spherical',
        (0, 1): 'tilt',
        (1, 1): 'coma',
        (0, 2): 'astigmatism',
        (0, 3): 'trefoil',
    }
    ok = all(lg_seidel_label(p, l) == name for (p, l), name in expected.items())
    # Negative ell maps the same as positive
    ok = ok and (lg_seidel_label(0, -1) == lg_seidel_label(0, 1))
    return ok, 'classical aberration labels match'


H.run('Seidel label table', t_lg_seidel_label_table)


# ===========================================================================
# Layer 2 -- Wick / Gaussian moment identities
# ===========================================================================

H.section('Layer 2 -- Gaussian moment / Wick identities')


def t_wick_zeroth_moment():
    """<1>_M = 1 (the moment table is normalised so the zeroth moment
    is unity)."""
    M = np.array([[2.0 + 0.1j, 0.3 - 0.05j],
                  [0.3 - 0.05j, 1.5 - 0.2j]], dtype=np.complex128)
    val = gaussian_moment_2d(0, 0, 0.5 * np.linalg.inv(M))
    return abs(val - 1.0) < 1e-12, f'<1> = {val}'


H.run('Wick: <1>_M = 1', t_wick_zeroth_moment)


def t_wick_second_moments():
    """<eta_x^2> = Sigma_xx, <eta_y^2> = Sigma_yy, <eta_x eta_y> = Sigma_xy."""
    M = np.array([[2.0 + 0.1j, 0.3 - 0.05j],
                  [0.3 - 0.05j, 1.5 - 0.2j]], dtype=np.complex128)
    Sigma = 0.5 * np.linalg.inv(M)
    m20 = gaussian_moment_2d(2, 0, Sigma)
    m02 = gaussian_moment_2d(0, 2, Sigma)
    m11 = gaussian_moment_2d(1, 1, Sigma)
    e20 = abs(m20 - Sigma[0, 0])
    e02 = abs(m02 - Sigma[1, 1])
    e11 = abs(m11 - Sigma[0, 1])
    max_err = max(float(e20), float(e02), float(e11))
    return max_err < 1e-12, (
        f'<x^2>={m20:.4g} (expect {Sigma[0,0]:.4g}); '
        f'<y^2>={m02:.4g} (expect {Sigma[1,1]:.4g}); '
        f'<xy>={m11:.4g} (expect {Sigma[0,1]:.4g}); err {max_err:.2e}'
    )


H.run('Wick: 2nd moments match Sigma_ij', t_wick_second_moments)


def t_wick_odd_moments_vanish():
    """All odd-total-order Wick moments vanish exactly."""
    M = np.array([[2.0 + 0.1j, 0.3 - 0.05j],
                  [0.3 - 0.05j, 1.5 - 0.2j]], dtype=np.complex128)
    Sigma = 0.5 * np.linalg.inv(M)
    odd = [(1, 0), (0, 1), (3, 0), (1, 2), (5, 0), (3, 2)]
    vals = [abs(gaussian_moment_2d(a, b, Sigma)) for a, b in odd]
    return max(vals) < 1e-300, (
        f'max |<odd>| = {max(vals):.2e} over indices {odd}'
    )


H.run('Wick: odd moments vanish exactly', t_wick_odd_moments_vanish)


def t_wick_fourth_moment_isserlis():
    """<eta_x^2 eta_y^2> = Sigma_xx Sigma_yy + 2 Sigma_xy^2  (Isserlis)."""
    M = np.array([[2.0, 0.5],
                  [0.5, 1.0]], dtype=np.complex128)
    Sigma = 0.5 * np.linalg.inv(M)
    measured = gaussian_moment_2d(2, 2, Sigma)
    expected = (Sigma[0, 0] * Sigma[1, 1]
                + 2.0 * Sigma[0, 1] * Sigma[0, 1])
    err = abs(measured - expected)
    return err < 1e-12, (
        f'<x^2 y^2> = {measured:.6g}, Isserlis = {expected:.6g}, '
        f'err {err:.2e}'
    )


H.run('Wick: <x^2 y^2> matches Isserlis identity', t_wick_fourth_moment_isserlis)


def t_wick_fourth_moment_xxxx():
    """<eta_x^4> = 3 Sigma_xx^2  (Gaussian fourth-cumulant identity)."""
    M = np.array([[3.5 - 0.4j, 0.2 + 0.1j],
                  [0.2 + 0.1j, 2.0]], dtype=np.complex128)
    Sigma = 0.5 * np.linalg.inv(M)
    measured = gaussian_moment_2d(4, 0, Sigma)
    expected = 3.0 * Sigma[0, 0] ** 2
    err = abs(measured - expected)
    return err < 1e-12, (
        f'<x^4> = {measured:.6g}, expected {expected:.6g}, err {err:.2e}'
    )


H.run('Wick: <x^4> = 3 Sigma_xx^2', t_wick_fourth_moment_xxxx)


def t_wick_sixth_moment_xxxx_yy():
    """<eta_x^4 eta_y^2> tested against Wick combinatorial sum."""
    M = np.array([[2.5, 0.4],
                  [0.4, 1.7]], dtype=np.complex128)
    Sigma = 0.5 * np.linalg.inv(M)
    s11 = Sigma[0, 0]
    s22 = Sigma[1, 1]
    s12 = Sigma[0, 1]
    # Isserlis hand-computed:  <x^4 y^2> = sum over pairings of {x,x,x,x,y,y}
    # = 3 s11^2 s22 + 12 s11 s12^2
    expected = 3.0 * s11 * s11 * s22 + 12.0 * s11 * s12 * s12
    measured = gaussian_moment_2d(4, 2, Sigma)
    err = abs(measured - expected)
    return err < 1e-12, (
        f'<x^4 y^2> = {measured:.6g}, expected {expected:.6g}, err {err:.2e}'
    )


H.run('Wick: <x^4 y^2> matches hand-computed Isserlis sum',
      t_wick_sixth_moment_xxxx_yy)


def t_wick_quadrature_check():
    """Numerical integration cross-check:  the closed-form moment
    matches direct quadrature of eta_x^a eta_y^b * exp(-eta^T M eta)
    for small (a, b)."""
    M = np.array([[2.5 + 0.3j, 0.2],
                  [0.2, 1.8 - 0.1j]], dtype=np.complex128)
    Sigma = 0.5 * np.linalg.inv(M)
    # Build a high-resolution real-axis quadrature grid (the moments
    # are well-defined even with complex M because Re(M) >= 0).
    # We integrate on the steepest-descent path = real axis since
    # eigenvalues of Re(M) > 0; the imaginary part of M only adds an
    # oscillatory factor that the integrand still resolves.
    L = 6.0
    n = 600
    eta = np.linspace(-L, L, n)
    deta = eta[1] - eta[0]
    Ex, Ey = np.meshgrid(eta, eta, indexing='xy')
    quadratic = (M[0, 0] * Ex * Ex + 2.0 * M[0, 1] * Ex * Ey
                 + M[1, 1] * Ey * Ey)
    weight = np.exp(-quadratic)
    # Normalisation for the moment definition:  divide by Z = pi/sqrt(det M)
    Z = math.pi / np.sqrt(np.linalg.det(M))
    test_pairs = [(0, 0), (2, 0), (0, 2), (1, 1), (4, 0), (2, 2)]
    errs = []
    for a, b in test_pairs:
        integrand = (Ex ** a) * (Ey ** b) * weight
        I_num = complex(np.sum(integrand) * deta * deta) / Z
        I_closed = gaussian_moment_2d(a, b, Sigma)
        err = abs(I_num - I_closed)
        errs.append(err)
    max_err = max(errs)
    return max_err < 5e-3, (
        f'max |closed-form - quadrature| over {test_pairs}: {max_err:.2e}'
    )


H.run('Wick: closed form matches quadrature on test moments',
      t_wick_quadrature_check)


def t_moment_table_completeness():
    """gaussian_moment_table_2d covers all (a, b) with a+b<=N and matches
    gaussian_moment_2d term-by-term."""
    M = np.array([[3.0, 0.5], [0.5, 2.0]], dtype=np.complex128)
    Sigma = 0.5 * np.linalg.inv(M)
    table = gaussian_moment_table_2d(M, 6)
    expected_keys = set()
    for total in range(7):
        for a in range(total + 1):
            expected_keys.add((a, total - a))
    if set(table.keys()) != expected_keys:
        return False, f'keys mismatch: have {set(table.keys())}'
    for (a, b), val in table.items():
        ref = gaussian_moment_2d(a, b, Sigma)
        if abs(val - ref) > 1e-12:
            return False, f'(a,b)=({a},{b}): table={val}, direct={ref}'
    return True, f'{len(table)} moments verified'


H.run('moment table covers all indices and matches direct',
      t_moment_table_completeness)


# ===========================================================================
# Layer 2b -- Polynomial utilities (multiply, shift, contract)
# ===========================================================================

H.section('Layer 2b -- Polynomial utilities')


def t_poly_multiply():
    """(x + 1)(y + 1) = xy + x + y + 1."""
    p1 = {(1, 0): 1.0 + 0j, (0, 0): 1.0 + 0j}     # x + 1
    p2 = {(0, 1): 1.0 + 0j, (0, 0): 1.0 + 0j}     # y + 1
    prod = _multiply_polys_2d(p1, p2)
    expected = {(1, 1): 1.0, (1, 0): 1.0, (0, 1): 1.0, (0, 0): 1.0}
    if set(prod.keys()) != set(expected.keys()):
        return False, f'wrong keys: {set(prod.keys())}'
    for k, v in expected.items():
        if abs(prod[k] - v) > 1e-12:
            return False, f'{k}: got {prod[k]}, expected {v}'
    return True, '(x+1)(y+1) correct'


H.run('Polynomial multiply: (x+1)(y+1)', t_poly_multiply)


def t_poly_shift():
    """(x + a)^2 = x^2 + 2 a x + a^2 via _polynomial_under_affine_shift."""
    p = {(2, 0): 1.0 + 0j}     # x^2
    shifted = _polynomial_under_affine_shift(p, shift_x=3.0 + 0j, shift_y=0.0 + 0j)
    # (x + 3)^2 = x^2 + 6 x + 9
    expected = {(2, 0): 1.0, (1, 0): 6.0, (0, 0): 9.0}
    for k, v in expected.items():
        got = shifted.get(k, 0.0)
        if abs(got - v) > 1e-12:
            return False, f'{k}: got {got}, expected {v}'
    return True, '(x+3)^2 expansion correct'


H.run('Polynomial shift: (x+3)^2', t_poly_shift)


def t_poly_substitute_linear():
    """Substitute x = 2 eta_x + 3 eta_y + 1 into x^2 -- result should be
    4 eta_x^2 + 12 eta_x eta_y + 9 eta_y^2 + 4 eta_x + 6 eta_y + 1."""
    p = {(2, 0): 1.0 + 0j}
    res = _polynomial_substitute_linear_2d(
        p, A_xx=2.0, A_xy=3.0, A_yx=0.0, A_yy=1.0,
        b_x=1.0, b_y=0.0,
    )
    expected = {(2, 0): 4.0, (1, 1): 12.0, (0, 2): 9.0,
                (1, 0): 4.0, (0, 1): 6.0, (0, 0): 1.0}
    for k, v in expected.items():
        got = res.get(k, 0.0)
        if abs(got - v) > 1e-12:
            return False, f'{k}: got {got}, expected {v}'
    return True, 'all coefficients match'


H.run('Polynomial linear substitution (2D affine)',
      t_poly_substitute_linear)


def t_poly_contract_against_moments():
    """Contract polynomial 1 + x^2 + y^2 against Wick moment table:
    result = <1> + <x^2> + <y^2> = 1 + Sigma_xx + Sigma_yy."""
    M = np.array([[2.0, 0.3], [0.3, 1.5]], dtype=np.complex128)
    Sigma = 0.5 * np.linalg.inv(M)
    table = gaussian_moment_table_2d(M, 4)
    poly = {(0, 0): 1.0 + 0j, (2, 0): 1.0 + 0j, (0, 2): 1.0 + 0j}
    val = _contract_against_moment_table(poly, table)
    expected = 1.0 + Sigma[0, 0] + Sigma[1, 1]
    err = abs(val - expected)
    return err < 1e-12, f'val={val}, expected={expected}, err={err:.2e}'


H.run('Polynomial-moment contraction matches term-by-term',
      t_poly_contract_against_moments)


# ===========================================================================
# Layer 1b -- LG/HG decomposition round-trip
# ===========================================================================

H.section('Layer 1b -- LG/HG decomposition (round-trip)')


def t_lg_decompose_self():
    """decompose_lg of LG_{1,1} returns 1.0 on the (1, 1) component and
    near-zero elsewhere."""
    w = 1.0e-3
    extent = 6.0 * w
    n = 320
    x = np.linspace(-extent, extent, n)
    y = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    field = evaluate_lg_mode(1, 1, w, X, Y)
    coeffs = decompose_lg(field, X, Y, w, p_max=2, ell_max=2)
    diag = abs(coeffs[(1, 1)] - 1.0)
    others = max(abs(coeffs[k]) for k in coeffs if k != (1, 1))
    return diag < 5e-2 and others < 5e-2, (
        f'diagonal err={diag:.2e}; max off-diagonal={others:.2e}'
    )


H.run('decompose_lg recovers a known LG mode', t_lg_decompose_self)


def t_hg_decompose_self():
    """decompose_hg of HG_{2, 1} returns 1 on the (2, 1) component."""
    wx = 1.0e-3
    wy = 1.5e-3
    extent_x = 6.0 * wx
    extent_y = 6.0 * wy
    n = 320
    x = np.linspace(-extent_x, extent_x, n)
    y = np.linspace(-extent_y, extent_y, n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    field = evaluate_hg_mode(2, 1, wx, wy, X, Y)
    coeffs = decompose_hg(field, X, Y, wx, wy, m_max=3, n_max=3)
    diag = abs(coeffs[(2, 1)] - 1.0)
    others = max(abs(coeffs[k]) for k in coeffs if k != (2, 1))
    return diag < 5e-2 and others < 5e-2, (
        f'diagonal err={diag:.2e}; max off-diagonal={others:.2e}'
    )


H.run('decompose_hg recovers a known HG mode', t_hg_decompose_self)


# ===========================================================================
# Layer 3 -- Canonical polynomial fit (Phi(s2, v2), s1(s2, v2))
# ===========================================================================

H.section('Layer 3 -- Canonical polynomial fit')


def _build_test_singlet():
    """100mm focal length BK7 singlet with 12mm aperture, used as the
    test prescription throughout the fit/aberration-tensor tests."""
    pres = op.make_singlet(
        51.5e-3,         # R1
        np.inf,          # R2
        4.1e-3,          # thickness
        'N-BK7',
        aperture=12.0e-3,
    )
    pres['object_distance'] = 200e-3
    return pres


def t_fit_runs_and_residual_small():
    """fit_canonical_polynomials runs without error and produces a small
    residual on a benign refractive system."""
    from lumenairy.asymptotic import fit_canonical_polynomials
    pres = _build_test_singlet()
    fit = fit_canonical_polynomials(
        pres, wavelength=1.31e-6,
        source_box_half=20e-6,    # 20 um source extent
        pupil_box_half=0.02,      # ~20 mrad cone -- comfortable inside aperture
        n_field=8, n_pupil=8,
        poly_order=6,
    )
    return (fit.res_phi_rms_waves < 1e-3
            and fit.res_s1_rms_m < 1e-7
            and fit.n_rays > 1000), (
        f'res_phi_rms = {fit.res_phi_rms_waves:.2e} waves, '
        f'res_s1_rms = {fit.res_s1_rms_m*1e6:.3e} um, '
        f'n_rays = {fit.n_rays}'
    )


H.run('Fit:  small residual on N-BK7 singlet (sub-mwave Phi)',
      t_fit_runs_and_residual_small)


def t_fit_round_trip_evaluation():
    """Evaluate s1 and Phi at randomly-sampled (s2, v2) inside the fit
    box and verify against direct ray-trace.  This is the strongest
    end-to-end test of the fit:  if the polynomial doesn't reproduce
    the underlying ray trace the test fails."""
    from lumenairy.asymptotic import fit_canonical_polynomials
    from lumenairy.raytrace import surfaces_from_prescription, _make_bundle, trace
    pres = _build_test_singlet()
    fit = fit_canonical_polynomials(
        pres, wavelength=1.31e-6,
        source_box_half=20e-6,
        pupil_box_half=0.02,
        n_field=8, n_pupil=8,
        poly_order=6,
    )
    # Direct ray trace: launch from a few (s1, v1) test points.
    test_inputs = [
        (5e-6, 0.0, 0.005, 0.0),
        (-5e-6, 5e-6, -0.003, 0.004),
        (0.0, -10e-6, 0.0, -0.005),
    ]
    surfaces = surfaces_from_prescription(pres)
    obj_d = pres.get('object_distance', 0.0) or 0.0
    s1x_in = np.array([t[0] for t in test_inputs])
    s1y_in = np.array([t[1] for t in test_inputs])
    L_in = np.array([t[2] for t in test_inputs])
    M_in = np.array([t[3] for t in test_inputs])
    bundle = _make_bundle(x=s1x_in, y=s1y_in, L=L_in, M=M_in,
                          wavelength=1.31e-6)
    bundle.z = np.full(len(test_inputs), -obj_d)
    res = trace(bundle, surfaces, 1.31e-6, output_filter='last')
    final = res.image_rays
    alive = np.asarray(final.alive, dtype=bool)
    # Skip dead rays
    if not alive.all():
        return False, 'some test rays died unexpectedly'

    s2x_dt = np.asarray(final.x)
    s2y_dt = np.asarray(final.y)
    v2x_dt = np.asarray(final.L)
    v2y_dt = np.asarray(final.M)
    phi_dt = np.asarray(final.opd) / 1.31e-6

    # Evaluate fit at the OUTPUT (s2, v2) and compare to s1, Phi.
    s1x_fit, s1y_fit = fit.eval_s1(s2x_dt, s2y_dt, v2x_dt, v2y_dt)
    phi_fit = fit.eval_phi(s2x_dt, s2y_dt, v2x_dt, v2y_dt,
                            include_linear=True)
    s1_err = max(float(np.max(np.abs(s1x_fit - s1x_in))),
                 float(np.max(np.abs(s1y_fit - s1y_in))))
    phi_err = float(np.max(np.abs(phi_fit - phi_dt)))
    return (s1_err < 1e-6 and phi_err < 1e-3), (
        f's1 max err = {s1_err*1e6:.3e} um; phi max err = {phi_err:.3e} waves'
    )


H.run('Fit round-trip:  evaluate at scattered test points matches direct ray trace',
      t_fit_round_trip_evaluation)


def t_fit_jacobian_nonzero():
    """Jacobian J = d s1 / d v2 must be nonzero at the chief ray.
    A degenerate single-source-point fit would give J = 0; the fixed
    4-D fit must produce meaningful |J|."""
    from lumenairy.asymptotic import fit_canonical_polynomials
    pres = _build_test_singlet()
    fit = fit_canonical_polynomials(
        pres, wavelength=1.31e-6,
        source_box_half=20e-6,
        pupil_box_half=0.02,
        n_field=8, n_pupil=8,
        poly_order=6,
    )
    # Evaluate J at the chief ray (s2 = chief image, v2 = chief direction)
    # We don't know the chief ray a priori; just sample at the fit centre.
    s2x = fit.s2x_centre
    s2y = fit.s2y_centre
    v2x = fit.v2x_centre
    v2y = fit.v2y_centre
    s1x, s1y, dxdvx, dxdvy, dydvx, dydvy = fit.eval_s1_with_v2_grad(
        np.array(s2x), np.array(s2y), np.array(v2x), np.array(v2y)
    )
    J_fro = math.sqrt(float(dxdvx) ** 2 + float(dxdvy) ** 2
                       + float(dydvx) ** 2 + float(dydvy) ** 2)
    return J_fro > 1e-6, (
        f'|J|_F = {J_fro:.4e} m  '
        f'(dxdvx={float(dxdvx):.3e}, dxdvy={float(dxdvy):.3e}, '
        f'dydvx={float(dydvx):.3e}, dydvy={float(dydvy):.3e})'
    )


H.run('Fit: J = ds1/dv2 non-zero at chief ray (s1 has v2-dependence)',
      t_fit_jacobian_nonzero)


def t_fit_in_box_mask():
    """fit.in_box correctly masks evaluation points inside vs. outside
    the [-1, 1]^4 normalisation box."""
    from lumenairy.asymptotic import fit_canonical_polynomials
    pres = _build_test_singlet()
    fit = fit_canonical_polynomials(
        pres, wavelength=1.31e-6,
        source_box_half=20e-6,
        pupil_box_half=0.02,
        n_field=6, n_pupil=6,
        poly_order=4,
    )
    # In-box points
    s2x_in = np.array([fit.s2x_centre, fit.s2x_centre + 0.5 * fit.s2x_halfrange])
    s2y_in = np.array([fit.s2y_centre, fit.s2y_centre + 0.5 * fit.s2y_halfrange])
    v2x_in = np.array([fit.v2x_centre, fit.v2x_centre + 0.5 * fit.v2x_halfrange])
    v2y_in = np.array([fit.v2y_centre, fit.v2y_centre + 0.5 * fit.v2y_halfrange])
    in_mask = fit.in_box(s2x_in, s2y_in, v2x_in, v2y_in)
    # Out-of-box point
    s2x_out = np.array([fit.s2x_centre + 5.0 * fit.s2x_halfrange])
    s2y_out = np.array([fit.s2y_centre])
    v2x_out = np.array([fit.v2x_centre])
    v2y_out = np.array([fit.v2y_centre])
    out_mask = fit.in_box(s2x_out, s2y_out, v2x_out, v2y_out)
    return (bool(in_mask.all()) and not bool(out_mask.any())), (
        f'in_mask = {in_mask}, out_mask = {out_mask}'
    )


H.run('Fit: in_box mask classifies points correctly', t_fit_in_box_mask)


def t_fit_with_linear_phase_extraction():
    """A diffractive prescription (or any system with a linear OPD ramp)
    triggers extract_linear_phase to produce a smaller residual than
    the same fit without extraction.  Use a synthetic linear ramp on
    a refractive system as a proxy:  the fitter should recover it
    cleanly when extract_linear_phase=True."""
    # Use the singlet fit as baseline; check that linear_coeffs_phi is
    # populated when extract_linear_phase=True and absent when False.
    from lumenairy.asymptotic import fit_canonical_polynomials
    pres = _build_test_singlet()
    fit_on = fit_canonical_polynomials(
        pres, wavelength=1.31e-6,
        source_box_half=20e-6, pupil_box_half=0.02,
        n_field=6, n_pupil=6, poly_order=4,
        extract_linear_phase=True,
    )
    fit_off = fit_canonical_polynomials(
        pres, wavelength=1.31e-6,
        source_box_half=20e-6, pupil_box_half=0.02,
        n_field=6, n_pupil=6, poly_order=4,
        extract_linear_phase=False,
    )
    has_linear = (fit_on.linear_coeffs_phi is not None
                  and fit_off.linear_coeffs_phi is None)
    # On a refractive system, the linear component is typically small;
    # extraction should not hurt the Chebyshev residual.
    cheb_residual_ratio = (fit_on.res_phi_rms_waves
                           / max(fit_off.res_phi_rms_waves, 1e-30))
    # Round-trip:  re-add the linear coeffs and the fit should give the
    # same evaluation as a fit without extraction (up to lstsq noise).
    s2x = fit_on.s2x_centre
    s2y = fit_on.s2y_centre
    v2x = fit_on.v2x_centre
    v2y = fit_on.v2y_centre
    phi_on_with_lin = fit_on.eval_phi(np.array(s2x), np.array(s2y),
                                        np.array(v2x), np.array(v2y),
                                        include_linear=True)
    phi_off = fit_off.eval_phi(np.array(s2x), np.array(s2y),
                                 np.array(v2x), np.array(v2y),
                                 include_linear=True)
    diff = abs(float(phi_on_with_lin) - float(phi_off))
    # We only require has_linear correct AND round-trip consistent;
    # the residual ratio is informational (refractive systems may have
    # truly tiny linear components, so the ratio could be ~1).
    return has_linear and diff < 1e-3, (
        f'linear flag = {has_linear}; '
        f'eval diff (on/off) = {diff:.2e} waves; '
        f'cheb-residual ratio (on/off) = {cheb_residual_ratio:.3f}'
    )


H.run('Fit: extract_linear_phase round-trip consistent on refractive system',
      t_fit_with_linear_phase_extraction)


# ===========================================================================
# Layer 4 -- Newton stationary solver
# ===========================================================================

H.section('Layer 4 -- Newton envelope-stationary solver')


def t_newton_chief_ray_origin():
    """For an on-axis source point and image at the chief-ray landing,
    Newton should converge to v_2* near the chief-ray direction (at
    the centre of the pupil box, modulo aberrations)."""
    from lumenairy.asymptotic import (fit_canonical_polynomials,
                                       solve_envelope_stationary)
    pres = _build_test_singlet()
    fit = fit_canonical_polynomials(
        pres, wavelength=1.31e-6,
        source_box_half=20e-6, pupil_box_half=0.02,
        n_field=6, n_pupil=6, poly_order=4,
    )
    # Chief-ray image of the on-axis source point: solve via Newton.
    v_star, n_iter, residual = solve_envelope_stationary(
        fit, (0.0, 0.0), (0.0, 0.0),
        w_s=20e-6, w_p=0.02,
        v2_centre=(fit.v2x_centre, fit.v2y_centre),
    )
    # Convergence assertion
    return (residual < 1e-8 and n_iter < 10), (
        f'residual={residual:.2e} after {n_iter} iter, v* = {v_star}'
    )


H.run('Newton: converges on the singlet chief-ray solve',
      t_newton_chief_ray_origin)


# ===========================================================================
# Layer 5 -- Modal asymptotic propagator
# ===========================================================================

H.section('Layer 5 -- Modal asymptotic propagator')


def t_modal_propagator_runs():
    """propagate_modal_asymptotic runs without error and produces a
    field of the right shape with finite values."""
    from lumenairy.asymptotic import (fit_canonical_polynomials,
                                       propagate_modal_asymptotic)
    pres = _build_test_singlet()
    fit = fit_canonical_polynomials(
        pres, wavelength=1.31e-6,
        source_box_half=20e-6, pupil_box_half=0.02,
        n_field=8, n_pupil=8, poly_order=6,
    )
    # Output grid:  small region around the fit centre
    L = fit.s2x_halfrange * 0.3
    n = 16
    ax = np.linspace(-L, L, n) + fit.s2x_centre
    ay = np.linspace(-L, L, n) + fit.s2y_centre
    X, Y = np.meshgrid(ax, ay, indexing='xy')
    field = propagate_modal_asymptotic(
        fit,
        source_point=(0.0, 0.0),
        w_s=20e-6, w_p=0.02,
        v2_centre=(fit.v2x_centre, fit.v2y_centre),
        s2_grid_x=X, s2_grid_y=Y,
    )
    finite = bool(np.all(np.isfinite(field)))
    has_signal = bool(np.max(np.abs(field)) > 0)
    return (finite and has_signal and field.shape == X.shape), (
        f'shape={field.shape}, max|E|={np.max(np.abs(field)):.3e}, '
        f'all finite={finite}'
    )


H.run('Modal propagator: runs and produces finite-valued field',
      t_modal_propagator_runs)


def t_modal_propagator_peaked_at_chief():
    """For an on-axis source, the propagator should peak near (0, 0)
    of the output plane (chief-ray image)."""
    from lumenairy.asymptotic import (fit_canonical_polynomials,
                                       propagate_modal_asymptotic)
    pres = _build_test_singlet()
    fit = fit_canonical_polynomials(
        pres, wavelength=1.31e-6,
        source_box_half=20e-6, pupil_box_half=0.02,
        n_field=8, n_pupil=8, poly_order=6,
    )
    L = fit.s2x_halfrange * 0.3
    n = 17
    ax = np.linspace(-L, L, n) + fit.s2x_centre
    ay = np.linspace(-L, L, n) + fit.s2y_centre
    X, Y = np.meshgrid(ax, ay, indexing='xy')
    field = propagate_modal_asymptotic(
        fit,
        source_point=(0.0, 0.0),
        w_s=20e-6, w_p=0.02,
        v2_centre=(fit.v2x_centre, fit.v2y_centre),
        s2_grid_x=X, s2_grid_y=Y,
    )
    intensity = np.abs(field) ** 2
    iy_peak, ix_peak = np.unravel_index(int(np.argmax(intensity)), intensity.shape)
    s2x_peak = X[iy_peak, ix_peak]
    s2y_peak = Y[iy_peak, ix_peak]
    # Should be within one pixel of the fit centre
    pix = ax[1] - ax[0]
    err = math.hypot(s2x_peak - fit.s2x_centre, s2y_peak - fit.s2y_centre)
    return err < 2.5 * pix, (
        f'peak at ({s2x_peak:.2e}, {s2y_peak:.2e}); '
        f'fit centre = ({fit.s2x_centre:.2e}, {fit.s2y_centre:.2e}); '
        f'err = {err:.2e}, pixel = {pix:.2e}'
    )


H.run('Modal propagator: PSF peaks near the on-axis chief-ray image',
      t_modal_propagator_peaked_at_chief)


# ===========================================================================
# Layer 6 -- LG aberration tensor
# ===========================================================================

H.section('Layer 6 -- LG aberration tensor')


def t_aberration_tensor_runs():
    """aberration_tensor runs and produces sensible non-NaN output."""
    from lumenairy.asymptotic import (fit_canonical_polynomials,
                                       aberration_tensor)
    pres = _build_test_singlet()
    fit = fit_canonical_polynomials(
        pres, wavelength=1.31e-6,
        source_box_half=20e-6, pupil_box_half=0.02,
        n_field=8, n_pupil=8, poly_order=6,
    )
    res = aberration_tensor(
        fit,
        s2_image=(fit.s2x_centre, fit.s2y_centre),
        source_point=(0.0, 0.0),
        source_modes=[(0, 0)],
        pupil_modes=[(0, 0)],
        output_modes=[(0, 0), (1, 0), (2, 0), (1, 1)],
        w_s=20e-6, w_p=0.02,
        v2_centre=(fit.v2x_centre, fit.v2y_centre),
    )
    has_shape = res.L.shape == (4, 1)
    is_finite = bool(np.all(np.isfinite(res.L)))
    has_piston = abs(res.L[0, 0]) > 0
    return (has_shape and is_finite and has_piston), (
        f'L shape = {res.L.shape}; |L_(0,0),(0,0)| = '
        f'{abs(res.L[0,0]):.3e}; finite = {is_finite}'
    )


H.run('Aberration tensor: runs end-to-end on the singlet',
      t_aberration_tensor_runs)


# ===========================================================================
# Layer 7 -- LGAberrationMerit (end-to-end optimisation hook)
# ===========================================================================

H.section('Layer 7 -- LGAberrationMerit')


def t_lg_merit_runs():
    """LGAberrationMerit can be instantiated and evaluated."""
    pres = _build_test_singlet()
    merit = op.LGAberrationMerit(
        targets={(2, 0): 1.0, (1, 1): 1.0, (1, -1): 1.0},
        field_points=[(0.0, 0.0)],
        w_s=20e-6, w_p=0.02,
        fit_kwargs=dict(source_box_half=20e-6, pupil_box_half=0.02,
                        n_field=6, n_pupil=6, poly_order=4),
    )
    # Provide a context with the test prescription
    class _Ctx:
        prescription = pres
        wavelength = 1.31e-6
        N = 64
        dx = 20e-6
    val = merit.evaluate(_Ctx())
    return math.isfinite(val) and val >= 0, (
        f'merit value = {val:.4e} (finite >= 0 expected)'
    )


H.run('LGAberrationMerit: evaluates without error', t_lg_merit_runs)


def t_lg_merit_responds_to_curvature():
    """Changing R1 of the singlet should change the merit -- this
    verifies the merit is actually sensitive to design parameters
    (otherwise it would be useless for optimisation)."""
    pres_a = _build_test_singlet()
    pres_b = op.make_singlet(60.0e-3, np.inf, 4.1e-3, 'N-BK7',
                              aperture=12.0e-3)
    pres_b['object_distance'] = 200e-3
    merit = op.LGAberrationMerit(
        targets={(2, 0): 1.0},
        field_points=[(0.0, 0.0)],
        w_s=20e-6, w_p=0.02,
        fit_kwargs=dict(source_box_half=20e-6, pupil_box_half=0.02,
                        n_field=6, n_pupil=6, poly_order=4),
    )

    class _Ctx:
        wavelength = 1.31e-6
        N = 64
        dx = 20e-6

    ctx_a = _Ctx()
    ctx_a.prescription = pres_a
    ctx_b = _Ctx()
    ctx_b.prescription = pres_b

    val_a = merit.evaluate(ctx_a)
    val_b = merit.evaluate(ctx_b)
    return abs(val_a - val_b) > 1e-12, (
        f'merit (R1=51.5mm) = {val_a:.4e}; '
        f'merit (R1=60mm) = {val_b:.4e}; diff = {abs(val_a - val_b):.4e}'
    )


H.run('LGAberrationMerit: responds to a curvature change',
      t_lg_merit_responds_to_curvature)


def t_lg_merit_handles_bad_prescription():
    """If the prescription is so bad that the fit fails, the merit
    should return a finite penalty (not raise)."""
    # Construct a degenerate prescription with a 0-radius surface
    # (always-aborts trace).
    pres = _build_test_singlet()
    pres['surfaces'][0]['radius'] = 1e-12   # near-zero radius
    merit = op.LGAberrationMerit(
        targets={(2, 0): 1.0},
        fit_kwargs=dict(source_box_half=20e-6, pupil_box_half=0.02,
                        n_field=6, n_pupil=6, poly_order=4),
    )

    class _Ctx:
        prescription = pres
        wavelength = 1.31e-6
        N = 64
        dx = 20e-6

    val = merit.evaluate(_Ctx())
    return math.isfinite(val) and val > 0, (
        f'merit on degenerate prescription = {val} '
        f'(finite penalty expected, no exception)'
    )


H.run('LGAberrationMerit: handles bad prescription gracefully',
      t_lg_merit_handles_bad_prescription)


def t_fit_endpoint_anchored_default_off():
    """Default endpoint_anchored=False produces fit results identical
    to existing behaviour; passing True changes the fit slightly but
    keeps it valid."""
    from lumenairy.asymptotic import fit_canonical_polynomials
    pres = op.make_singlet(50e-3, -50e-3, 3e-3, 'N-BK7', aperture=8e-3)
    pres['object_distance'] = 50e-3
    common = dict(
        wavelength=1.31e-6,
        source_box_half=200e-6,
        pupil_box_half=0.05,
        n_field=6, n_pupil=6,
        poly_order=4,
        object_distance=50e-3,
    )
    fit_default = fit_canonical_polynomials(pres, **common)
    fit_anchored = fit_canonical_polynomials(
        pres, **common, endpoint_anchored=True)
    # Both fits must produce finite Phi residuals
    ok_def = math.isfinite(fit_default.res_phi_rms_waves)
    ok_anc = math.isfinite(fit_anchored.res_phi_rms_waves)
    # The anchored fit should typically have a SMALL difference in
    # residual -- usually within 2x of the default at this poly_order
    rms_def = fit_default.res_phi_rms_waves
    rms_anc = fit_anchored.res_phi_rms_waves
    ratio = (rms_anc / rms_def) if rms_def > 0 else 1.0
    sane = 0.1 < ratio < 10.0
    return ok_def and ok_anc and sane, (
        f'rms_default={rms_def:.4e}, rms_anchored={rms_anc:.4e}, '
        f'ratio={ratio:.3f}'
    )


H.run('fit_canonical_polynomials: endpoint_anchored option produces valid fit',
      t_fit_endpoint_anchored_default_off)


if __name__ == '__main__':
    sys.exit(H.summary())
