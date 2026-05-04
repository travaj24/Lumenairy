"""
Lens and focusing element phase screens.

This module provides phase-screen models for refractive focusing elements:
thin lenses (paraxial through aplanatic), thick singlets (spherical and
aspheric), multi-surface real lenses with split-step propagation through
glass, cylindrical lenses, axicons, and GRIN rod lenses.

All functions follow the exp(-i*omega*t) time convention and use SI meters
for spatial quantities.

Backends
--------
Most functions use NumPy by default.  ``apply_thin_lens``,
``apply_spherical_lens``, and ``apply_aspheric_lens`` accept a *use_gpu*
flag and will dispatch to CuPy when available.  ``apply_real_lens``
currently runs on the CPU only.

Author: Andrew Traverso
"""

import warnings

import numpy as np

# GPU backend ----------------------------------------------------------------
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    # Sentinel so ``xp is cp`` checks below don't NameError when cupy
    # isn't installed (CI / pure-CPU users).
    cp = None
    CUPY_AVAILABLE = False

# Optional fused-expression backend ------------------------------------------
# numexpr evaluates array expressions in chunked, multi-threaded passes
# without materialising full N x N intermediates.  Used by apply_real_lens
# to fuse the ``E * exp(-1j*k0*opd)`` phase-screen multiply, which at
# N=32768 otherwise allocates three 17 GB complex128 temporaries.
try:
    import numexpr as _ne
    NUMEXPR_AVAILABLE = True
except ImportError:
    NUMEXPR_AVAILABLE = False

# Optional Numba JIT.  Used by ``_aspheric_sag_accum_numba`` (fused
# polynomial-aspheric loop, 3.2.14) and the Maslov Chebyshev evaluator
# (``_cheb2d_val_grad_numba``).  Both have pure-NumPy fallbacks below.
try:
    import numba as _numba
    from numba import njit as _njit, prange as _prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _numba = None
    _NUMBA_AVAILABLE = False


_NEWTON_MAX_ITERS = 12

# Minimum field size at which the numexpr phase-screen path beats the
# straight numpy multiply (overhead of expression compile + thread
# dispatch is fixed; benefit scales with array size).
_NUMEXPR_MIN_SIZE = 1 << 20  # 1 Mi elements (~1024 x 1024)


def _newton_invert_chunk(args):
    """Module-level worker for ``apply_real_lens_traced`` Newton inversion.

    Rebuilds the three ``RectBivariateSpline`` objects from their knot
    data in-process (so we avoid pickling the SciPy spline objects,
    which is expensive) and runs the Newton loop on ``(x_chunk,
    y_chunk)`` for up to ``_NEWTON_MAX_ITERS`` iterations.  Returns
    the OPL at the converged entrance positions with NaN for any
    points that landed outside the fit domain.

    Lives at module scope so ``ProcessPoolExecutor`` can pickle it on
    Windows (spawn) workers.  The caller is ``_invert_newton`` inside
    :func:`apply_real_lens_traced`.
    """
    (knot_data, x_chunk, y_chunk) = args
    from scipy.interpolate import RectBivariateSpline
    xs_in = knot_data['xs_in']
    x_out_grid = knot_data['x_out_grid']
    y_out_grid = knot_data['y_out_grid']
    opl_grid = knot_data['opl_grid']
    launch_radius = knot_data['launch_radius']
    dx = knot_data['dx']
    bound = knot_data['bound']
    # Paraxial-magnification initial-guess factors.  See the docstring
    # in ``apply_real_lens_traced`` where these are computed from the
    # central finite-difference slope of the forward map.  Older knot
    # data written by pre-3.1.3 callers won't have these keys -- fall
    # back to the historical 1.10 multiplier so the worker stays
    # backwards compatible.
    inv_M_x = float(knot_data.get('inv_M_x', 1.10))
    inv_M_y = float(knot_data.get('inv_M_y', 1.10))

    Sx = RectBivariateSpline(xs_in, xs_in, x_out_grid, kx=3, ky=3)
    Sy = RectBivariateSpline(xs_in, xs_in, y_out_grid, kx=3, ky=3)
    So = RectBivariateSpline(xs_in, xs_in, opl_grid, kx=3, ky=3)

    xe = x_chunk.copy() * inv_M_x
    ye = y_chunk.copy() * inv_M_y
    tol = 0.01 * dx
    active = np.ones(xe.size, dtype=bool)
    for _it in range(_NEWTON_MAX_ITERS):
        if not active.any():
            break
        xa = xe[active]; ya = ye[active]
        xw = x_chunk[active]; yw = y_chunk[active]
        rx = Sx.ev(xa, ya) - xw
        ry = Sy.ev(xa, ya) - yw
        jxx = Sx.ev(xa, ya, dx=1)
        jxy = Sx.ev(xa, ya, dy=1)
        jyx = Sy.ev(xa, ya, dx=1)
        jyy = Sy.ev(xa, ya, dy=1)
        det = jxx * jyy - jxy * jyx
        safe = np.abs(det) > 1e-12
        inv_det = np.where(safe, 1.0 / det, 0.0)
        dxe = (jyy * rx - jxy * ry) * inv_det
        dye = (-jyx * rx + jxx * ry) * inv_det
        xa_new = np.clip(xa - dxe, -bound, bound)
        ya_new = np.clip(ya - dye, -bound, bound)
        xe[active] = xa_new
        ye[active] = ya_new
        res = np.sqrt(rx * rx + ry * ry)
        converged = res < tol
        idx_active = np.where(active)[0]
        active[idx_active[converged]] = False

    opl_flat = So.ev(xe, ye)
    out_of_domain = (xe * xe + ye * ye > (launch_radius * 0.99) ** 2)
    return np.where(out_of_domain, np.nan, opl_flat)


def _is_cupy_array(x):
    """
    Reliable CuPy array check.  ``hasattr(x, 'device')`` used to be a
    duck-type test for a CuPy device array but broke in NumPy 2.x
    (``ndarray`` now exposes ``.device`` via the Array API standard),
    causing every NumPy array to get routed into the CuPy branch.
    """
    if not CUPY_AVAILABLE:
        return False
    return isinstance(x, cp.ndarray)

# Sibling-module imports (created separately in this package) ----------------
from .propagation import angular_spectrum_propagate
from .glass import get_glass_index, get_glass_index_complex

# Typing: the Maslov section (merged in 3.2.2 from the former
# lens_maslov.py) uses Any / Dict / Optional / Tuple in function
# annotations.
from typing import Any, Dict, Optional, Tuple
# The Maslov section uses ``time`` for internal progress timing.
import time  # noqa: F401


# ---------------------------------------------------------------------------
# Helper: general conic + aspheric surface sag
# ---------------------------------------------------------------------------

if _NUMBA_AVAILABLE:
    @_njit(cache=True, parallel=True, fastmath=True)
    def _aspheric_sag_accum_numba(h_sq, sag, powers, coeffs):
        """In-place accumulate sum_i coeff_i * h_sq**(power_i // 2) onto
        ``sag``.  Single fused pass over h_sq, no temporary arrays.

        ``h_sq`` and ``sag`` must be contiguous float64 arrays of the
        same shape.  ``powers`` is int32, ``coeffs`` is float64; both
        1-D and same length.
        """
        flat_h = h_sq.ravel()
        flat_s = sag.ravel()
        n = flat_h.size
        n_terms = powers.size
        for i in _prange(n):
            v = flat_h[i]
            acc = 0.0
            for j in range(n_terms):
                p = powers[j] // 2
                # h_sq^p via repeated squaring keeps the inner loop
                # branch-free (Numba unrolls small fixed-power loops).
                hp = 1.0
                for _ in range(p):
                    hp *= v
                acc += coeffs[j] * hp
            flat_s[i] += acc
else:
    _aspheric_sag_accum_numba = None  # noqa


def surface_sag_general(h_sq, R, conic=0.0, aspheric_coeffs=None):
    """
    Compute surface sag for a general conic + even-aspheric surface.

    This function is used by both the lens and mirror modules, so it is
    exported at module level (no leading underscore).

    Parameters
    ----------
    h_sq : ndarray
        Squared radial distance from the optical axis, x**2 + y**2  [m**2].
    R : float
        Radius of curvature [m].  Use ``float('inf')`` or ``np.inf`` for a
        flat surface.
    conic : float, optional
        Conic constant (default 0 = sphere, -1 = paraboloid, < -1 =
        hyperboloid, -1 < k < 0 = prolate ellipsoid, > 0 = oblate ellipsoid).
    aspheric_coeffs : dict or None, optional
        Even polynomial aspheric coefficients ``{power: coeff}``, e.g.
        ``{4: A4, 6: A6, 8: A8, 10: A10}``.  Each term contributes
        ``coeff * h_sq**(power // 2)`` to the sag.

    Returns
    -------
    sag : ndarray
        Signed surface sag (positive when R > 0).
    """
    # Array-API polymorphic: detect cupy vs numpy from the input.
    # This keeps the helper usable from both the CPU and GPU paths of
    # apply_real_lens without duplicating code.  Arithmetic
    # broadcasting (np.where, np.sqrt on cupy arrays) silently
    # converts to host, so we dispatch explicitly.
    xp = cp if _is_cupy_array(h_sq) else np
    sag = xp.zeros_like(h_sq)

    if R is not None and not np.isinf(R):
        # Conic sag: h^2 / (R * (1 + sqrt(1 - (1+k)*h^2/R^2)))
        norm = (1 + conic) * h_sq / R**2
        valid = norm < 0.9999
        denom_arg = xp.where(valid, 1 - norm, 0.01)
        conic_sag = xp.where(
            valid,
            h_sq / (R * (1 + xp.sqrt(denom_arg))),
            0.0,
        )
        sag = conic_sag

    if aspheric_coeffs:
        # 3.2.14: fused single-pass numba kernel when available.
        # Skips the per-term temporary array allocation that the
        # legacy NumPy fallback required (5 aspheric coeffs at N=4096
        # is ~640 MB of transient memory in that path).  CuPy stays
        # on the legacy path because numba targets host arrays.
        if (xp is np and _NUMBA_AVAILABLE
                and _aspheric_sag_accum_numba is not None
                and h_sq.dtype == np.float64
                and sag.dtype == np.float64):
            powers_arr = np.fromiter(
                (int(p) for p in aspheric_coeffs.keys()), dtype=np.int32)
            coeffs_arr = np.fromiter(
                (float(c) for c in aspheric_coeffs.values()),
                dtype=np.float64)
            # In-place accumulate; sag is contiguous from xp.zeros_like
            # above so .ravel() inside the kernel is a view.
            _aspheric_sag_accum_numba(
                np.ascontiguousarray(h_sq), sag, powers_arr, coeffs_arr)
        else:
            for power, coeff in aspheric_coeffs.items():
                sag = sag + coeff * h_sq ** (power // 2)

    return sag


# Keep the private alias so internal callers (apply_real_lens) can use either
# name without changing semantics.
_surface_sag_general = surface_sag_general


def surface_sag_biconic(X, Y, R_x, R_y=None, conic_x=0.0, conic_y=None,
                        aspheric_coeffs=None, aspheric_coeffs_y=None):
    """Biconic / cylindrical / toroidal surface sag.

    Generalises :func:`surface_sag_general` to surfaces that have
    different curvatures and conics along the x and y axes.  Covers:

    * **Biconic** (Zemax "Biconic"): independent R_x, R_y, K_x, K_y.
      Each axis contributes its own conic sag:

          z(x,y) = C_x*x² / (1 + sqrt(1 - (1+K_x)*C_x²*x²))
                 + C_y*y² / (1 + sqrt(1 - (1+K_y)*C_y²*y²))

      where C_x = 1/R_x, C_y = 1/R_y.
    * **Cylindrical**: pass ``R_y = inf`` (focusing in x only) or
      ``R_x = inf`` (focusing in y only).
    * **Toroidal** (approx., Zemax "Toroidal"): pass R_x (rotation-axis
      radius) and R_y (cross-section radius).
    * **Rotationally symmetric**: if ``R_y is None`` the function
      reduces to :func:`surface_sag_general` via h² = x² + y².

    Aspheric coefficients may be given per-axis for fully general
    anamorphic surfaces; ``aspheric_coeffs`` (x-axis) and
    ``aspheric_coeffs_y`` (y-axis) are separate dicts of
    ``{power: coeff}`` contributing ``coeff * h² ** (power // 2)``
    along each axis.

    Parameters
    ----------
    X, Y : ndarray
        Surface-local coordinates [m] (after any decenter/tilt).
        ``X`` and ``Y`` must have the same shape; meshgrid indexing is
        up to the caller.
    R_x : float
        Radius of curvature along x-axis [m] (``inf`` = flat in x).
    R_y : float, optional
        Radius of curvature along y-axis [m].  If ``None``, the surface
        is treated as rotationally symmetric with R = R_x (legacy).
    conic_x : float, default 0
        Conic constant along x.
    conic_y : float, optional
        Conic constant along y.  Defaults to ``conic_x`` if not given.
    aspheric_coeffs : dict or None
        Even-aspheric coefficients along x, ``{power: coeff}``.
    aspheric_coeffs_y : dict or None
        Even-aspheric coefficients along y.  If ``None`` and
        ``aspheric_coeffs`` is given, the x coefficients are reused for
        y (isotropic asphere).

    Returns
    -------
    sag : ndarray
        Signed surface sag, same shape as ``X`` / ``Y``.
    """
    # Detect array backend and use cupy ops when X/Y are device arrays.
    xp = cp if _is_cupy_array(X) else np
    X = xp.asarray(X)
    Y = xp.asarray(Y)

    if R_y is None:
        # Reduce to the rotationally-symmetric formula for backward
        # compatibility.
        h_sq = X ** 2 + Y ** 2
        return surface_sag_general(h_sq, R_x, conic_x, aspheric_coeffs)

    if conic_y is None:
        conic_y = conic_x

    def _axis_sag(h_sq, R, K, asph):
        s = xp.zeros_like(h_sq)
        if R is not None and not np.isinf(R):
            norm = (1 + K) * h_sq / R ** 2
            valid = norm < 0.9999
            denom_arg = xp.where(valid, 1 - norm, 0.01)
            s = xp.where(
                valid,
                h_sq / (R * (1 + xp.sqrt(denom_arg))),
                0.0,
            )
        if asph:
            for power, coeff in asph.items():
                s = s + coeff * h_sq ** (power // 2)
        return s

    sag_x = _axis_sag(X ** 2, R_x, conic_x, aspheric_coeffs)
    sag_y = _axis_sag(Y ** 2, R_y, conic_y,
                      aspheric_coeffs_y if aspheric_coeffs_y is not None
                      else aspheric_coeffs)
    return sag_x + sag_y


# ---------------------------------------------------------------------------
# Thin lens models
# ---------------------------------------------------------------------------

def apply_thin_lens(E_in, f, wavelength, dx, dy=None, xc=0, yc=0,
                    use_gpu=False, lens_model='paraxial'):
    """
    Apply a thin-lens phase to an optical field.

    Parameters
    ----------
    E_in : ndarray (complex), shape (Ny, Nx)
        Input electric field.
    f : float
        Focal length [m].  Positive = converging, negative = diverging.
    wavelength : float
        Optical wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float or None
        Grid spacing in y [m].  Defaults to *dx*.
    xc, yc : float
        Center of the lens [m] (for decentered lenses).
    use_gpu : bool
        If True and CuPy is available, run on the GPU.
    lens_model : str
        Phase model.  One of:

        ``'paraxial'``
            Quadratic approximation: phi = -k/(2f) * r**2.
            Valid for r/f < ~0.1 (half-angle < ~6 deg).
        ``'nonparaxial'``
            Exact spherical wavefront: phi = k * (f - sqrt(f**2 + r**2)).
            Accurate up to r/f ~ 0.5 (half-angle ~30 deg).
        ``'aplanatic'``
            Satisfies the Abbe sine condition (sin(theta) = r/f).
            phi = -k * f * (1 - sqrt(1 - r**2/f**2)) for r < |f|.
            Ideal for imaging systems; eliminates coma.
        ``'local_only'``
            Quadratic focusing about the decentered point *without* the
            linear tilt that a decentered paraxial lens would produce.
            Useful for micro-lens arrays where each lenslet should focus
            locally without steering the beam.

    Returns
    -------
    E_out : ndarray (complex), same shape as *E_in*
    """
    # Determine array library
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        xp = cp
        if not _is_cupy_array(E_in):
            E_in = cp.asarray(E_in)
    else:
        xp = np

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx

    k = 2 * np.pi / wavelength

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)
    r_sq = (X - xc) ** 2 + (Y - yc) ** 2

    if lens_model == 'paraxial':
        lens_phase = xp.exp(-1j * k / (2 * f) * r_sq)

    elif lens_model == 'nonparaxial':
        lens_phase = xp.exp(1j * k * (f - xp.sqrt(f ** 2 + r_sq)))

    elif lens_model == 'aplanatic':
        r_over_f_sq = r_sq / f ** 2
        valid = r_over_f_sq < 1.0
        sqrt_term = xp.sqrt(xp.maximum(1.0 - r_over_f_sq, 0.0))
        phase = k * f * (1.0 - sqrt_term)
        lens_phase = xp.where(valid, xp.exp(-1j * phase), 0.0 + 0.0j)

    elif lens_model == 'local_only':
        # Pure local focusing: the standard decentered quadratic minus the
        # linear tilt k/f * (xc*x + yc*y) that would otherwise steer the beam.
        decentered_phase = -k / (2 * f) * r_sq
        tilt_cancel = -k / f * (xc * X + yc * Y)
        lens_phase = xp.exp(1j * (decentered_phase + tilt_cancel))

    else:
        raise ValueError(
            f"Unknown lens_model: {lens_model!r}. "
            f"Choose from 'paraxial', 'nonparaxial', 'aplanatic', 'local_only'."
        )

    return E_in * lens_phase


# ---------------------------------------------------------------------------
# Thick spherical singlet
# ---------------------------------------------------------------------------

def apply_spherical_lens(E_in, R1, R2, d, n_lens, wavelength, dx, dy=None,
                         aperture_diameter=None, xc=0, yc=0, use_gpu=False):
    """
    Apply the phase of a thick singlet with spherical surfaces.

    Computes the exact optical-path difference through a glass element with
    two spherical surfaces, naturally including spherical aberration and all
    higher-order monochromatic aberrations.

    Parameters
    ----------
    E_in : ndarray (complex), shape (Ny, Nx)
        Input electric field.
    R1 : float
        Radius of curvature of the front surface [m].
        Positive = center of curvature on the transmission side (convex
        toward input).  ``np.inf`` for a flat surface.
    R2 : float
        Radius of curvature of the back surface [m].
        Negative = center of curvature on the input side (convex toward
        output).  Example: biconvex lens has R1 > 0, R2 < 0.
    d : float
        Center thickness [m].
    n_lens : float
        Refractive index of the lens material.
    wavelength : float
        Optical wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float or None
        Grid spacing in y [m].  Defaults to *dx*.
    aperture_diameter : float or None
        Clear aperture diameter [m].  If None the aperture is set by the
        surface radii of curvature.
    xc, yc : float
        Lens center [m].
    use_gpu : bool
        Use GPU if available.

    Returns
    -------
    E_out : ndarray (complex), same shape as *E_in*

    Notes
    -----
    The thickness profile is ``t(h) = d - sag1(h) - sag2(h)`` where each
    signed sag is ``sag(h) = R - sign(R) * sqrt(R**2 - h**2)``.

    The OPD relative to the center is:

        delta_phi(h) = -k * (n - 1) * (sag1(h) - sag2(h))

    which reduces to ``-k/(2f) * h**2`` in the paraxial limit with
    ``1/f = (n-1) * (1/R1 - 1/R2)`` (lensmaker's equation).
    """
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        xp = cp
        if not _is_cupy_array(E_in):
            E_in = cp.asarray(E_in)
    else:
        xp = np

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx

    k = 2 * np.pi / wavelength

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)
    h_sq = (X - xc) ** 2 + (Y - yc) ** 2

    def _surface_sag(h_sq, R):
        """Signed spherical sag: positive for convex (R > 0)."""
        if R is None or np.isinf(R):
            return xp.zeros_like(h_sq)
        h_sq_safe = xp.minimum(h_sq, R ** 2 * 0.9999)
        return R - np.sign(R) * xp.sqrt(R ** 2 - h_sq_safe)

    sag1 = _surface_sag(h_sq, R1)
    sag2 = _surface_sag(h_sq, R2)

    phase = -k * (n_lens - 1) * (sag1 - sag2)
    lens_field = xp.exp(1j * phase)

    # Clear aperture
    if aperture_diameter is not None:
        lens_field = xp.where(
            h_sq <= (aperture_diameter / 2) ** 2, lens_field, 0.0 + 0.0j
        )
    else:
        max_h_sq = np.inf
        if not np.isinf(R1):
            max_h_sq = min(max_h_sq, R1 ** 2)
        if not np.isinf(R2):
            max_h_sq = min(max_h_sq, R2 ** 2)
        if max_h_sq < np.inf:
            lens_field = xp.where(
                h_sq < max_h_sq * 0.9999, lens_field, 0.0 + 0.0j
            )

    return E_in * lens_field


# ---------------------------------------------------------------------------
# Thick aspheric singlet (conic + even polynomial)
# ---------------------------------------------------------------------------

def apply_aspheric_lens(E_in, R1, R2, d, n_lens, wavelength, dx, dy=None,
                        k1=0, k2=0, A1=None, A2=None,
                        aperture_diameter=None, xc=0, yc=0, use_gpu=False):
    """
    Apply an aspheric singlet lens phase based on exact OPD through thick glass.

    Each surface follows the standard aspheric sag equation:

        sag(h) = h**2 / (R * (1 + sqrt(1 - (1+k)*h**2/R**2)))
                 + A4*h**4 + A6*h**6 + A8*h**8 + A10*h**10

    Parameters
    ----------
    E_in : ndarray (complex), shape (Ny, Nx)
        Input electric field.
    R1, R2 : float
        Radii of curvature [m] (same sign convention as
        :func:`apply_spherical_lens`).
    d : float
        Center thickness [m].
    n_lens : float
        Refractive index at the operating wavelength.
    wavelength : float
        Optical wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float or None
        Grid spacing in y [m].  Defaults to *dx*.
    k1, k2 : float
        Conic constants for surfaces 1 and 2 (0 = sphere, -1 = paraboloid).
    A1, A2 : dict or None
        Even aspheric polynomial coefficients for each surface.
        Keys are the powers of h: ``{4: A4, 6: A6, 8: A8, 10: A10}``.
    aperture_diameter : float or None
        Clear aperture diameter [m].
    xc, yc : float
        Lens center [m].
    use_gpu : bool
        Use GPU if available.

    Returns
    -------
    E_out : ndarray (complex), same shape as *E_in*

    Notes
    -----
    With ``k1=k2=0`` and ``A1=A2=None`` this reduces to
    :func:`apply_spherical_lens`.

    A plano-convex lens with ``k1 = -n_lens**2`` on the curved surface
    eliminates third-order spherical aberration for collimated input.
    """
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        xp = cp
        if not _is_cupy_array(E_in):
            E_in = cp.asarray(E_in)
    else:
        xp = np

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx

    kw = 2 * np.pi / wavelength  # wavenumber (avoid shadowing conic k)

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)
    h_sq = (X - xc) ** 2 + (Y - yc) ** 2

    def _aspheric_sag(h_sq, R, k_conic, A_coeffs):
        """Signed aspheric sag for one surface."""
        if R is None or np.isinf(R):
            sag = xp.zeros_like(h_sq)
            if A_coeffs:
                for power, coeff in A_coeffs.items():
                    sag = sag + coeff * h_sq ** (power // 2)
            return sag

        R_abs = abs(R)
        norm_h_sq = h_sq / R_abs ** 2
        denom_arg = 1 - (1 + k_conic) * norm_h_sq
        denom_arg_safe = xp.maximum(denom_arg, 1e-12)
        sag_unsigned = h_sq / (R_abs * (1 + xp.sqrt(denom_arg_safe)))
        sag = np.sign(R) * sag_unsigned

        if A_coeffs:
            for power, coeff in A_coeffs.items():
                sag = sag + coeff * h_sq ** (power // 2)

        return sag

    sag1 = _aspheric_sag(h_sq, R1, k1, A1)
    sag2 = _aspheric_sag(h_sq, R2, k2, A2)

    phase = -kw * (n_lens - 1) * (sag1 - sag2)
    lens_field = xp.exp(1j * phase)

    # Apply aperture
    if aperture_diameter is not None:
        lens_field = xp.where(
            h_sq <= (aperture_diameter / 2) ** 2, lens_field, 0.0 + 0.0j
        )
    else:
        max_h_sq = np.inf
        if R1 is not None and not np.isinf(R1):
            if (1 + k1) > 0:
                max_h_sq = min(max_h_sq, R1 ** 2 / (1 + k1))
        if R2 is not None and not np.isinf(R2):
            if (1 + k2) > 0:
                max_h_sq = min(max_h_sq, R2 ** 2 / (1 + k2))
        if max_h_sq < np.inf:
            lens_field = xp.where(
                h_sq < max_h_sq * 0.9999, lens_field, 0.0 + 0.0j
            )

    return E_in * lens_field


# ---------------------------------------------------------------------------
# Grid-vs-aperture safety check
# ---------------------------------------------------------------------------

def _collect_semi_diameters(prescription):
    """Return [(label, semi_diameter_m)] for every surface in the
    prescription that has a ``semi_diameter`` set.

    Looks at both ``prescription['elements']`` (Zemax-loaded path,
    where ``semi_diameter`` is the per-surface CLAP) and
    ``prescription['surfaces']`` (builder-style).  The top-level
    ``aperture_diameter`` (system-wide clear aperture) is also
    included if present.  Surfaces without a finite ``semi_diameter``
    are skipped.
    """
    out = []
    seen_labels = set()
    elements = prescription.get('elements')
    if isinstance(elements, list):
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            sd = elem.get('semi_diameter')
            if sd is None or not np.isfinite(float(sd)):
                continue
            surf_num = elem.get('surf_num', '?')
            comment = (elem.get('comment') or elem.get('name') or '').strip()
            label = f"surf {surf_num}"
            if comment:
                label = f"{label} '{comment}'"
            if label not in seen_labels:
                out.append((label, float(sd)))
                seen_labels.add(label)
    surfaces = prescription.get('surfaces')
    if isinstance(surfaces, list):
        for i, surf in enumerate(surfaces):
            if not isinstance(surf, dict):
                continue
            sd = surf.get('semi_diameter')
            if sd is None or not np.isfinite(float(sd)):
                continue
            label = f"surfaces[{i}]"
            if label not in seen_labels:
                out.append((label, float(sd)))
                seen_labels.add(label)
    ap = prescription.get('aperture_diameter')
    if ap is not None and np.isfinite(float(ap)):
        out.append(('system aperture_diameter', 0.5 * float(ap)))
    return out


def check_grid_vs_apertures(prescription, N, dx, *, safety_factor=1.0):
    """Identify every prescription surface whose semi-aperture exceeds
    the simulation grid's half-extent.

    Use this as a pre-flight check before a full ASM-through-lens run.
    A surface whose ``semi_diameter`` is larger than ``safety_factor *
    N * dx / 2`` will silently truncate the field's outer rim during
    propagation, dropping any energy the real hardware would have
    transmitted past the grid edge.  This is a sim-infrastructure
    artifact, not a physical clipping by the lens itself, and will
    show up in centroid measurements as a uniform inward bias.

    Parameters
    ----------
    prescription : dict
        lumenairy prescription dict (loaded or builder-built).
    N : int
        Simulation grid size (assumed square).
    dx : float
        Grid pitch [m].
    safety_factor : float, optional
        Margin between the grid semi and the largest surface
        semi-aperture.  Pass 0.95 to flag surfaces whose semi exceeds
        95% of the grid half-extent (recommended for clean Gaussian
        wing containment); pass 1.0 (default) to flag only surfaces
        that exceed the grid outright.

    Returns
    -------
    issues : list of (label, semi_aperture_m, grid_semi_m, gap_m)
        One tuple per offending surface.  Empty if every aperture fits.

    See Also
    --------
    apply_real_lens, apply_real_lens_traced, apply_real_lens_maslov
        These functions automatically run this check on entry and emit
        a UserWarning if any aperture exceeds the grid.
    """
    if N <= 0 or dx <= 0:
        raise ValueError(f"N and dx must be positive, got N={N}, dx={dx}")
    grid_semi = 0.5 * float(N) * float(dx)
    threshold = float(safety_factor) * grid_semi
    issues = []
    for label, sd in _collect_semi_diameters(prescription):
        if sd > threshold:
            issues.append((label, sd, grid_semi, sd - grid_semi))
    return issues


def _warn_if_aperture_exceeds_grid(prescription, N, dx, *,
                                    source='apply_real_lens',
                                    safety_factor=1.0,
                                    stacklevel=3):
    """Emit a UserWarning if any prescription aperture exceeds the
    simulation grid.  Called at the top of ``apply_real_lens``,
    ``apply_real_lens_traced``, and ``apply_real_lens_maslov``.

    Python's default warning filter dedups by ``(module, lineno)`` so
    repeated calls from the same site only warn once.
    """
    try:
        issues = check_grid_vs_apertures(
            prescription, N, dx, safety_factor=safety_factor)
    except Exception:
        return
    if not issues:
        return
    grid_semi = 0.5 * N * dx
    issues_sorted = sorted(issues, key=lambda r: -r[1])
    biggest_label, biggest_sd, _, biggest_gap = issues_sorted[0]
    body = ", ".join(
        f"{lab}={sd*1e3:.2f}mm"
        for lab, sd, _, _ in issues_sorted[:5]
    )
    if len(issues_sorted) > 5:
        body = body + f", ... (+{len(issues_sorted)-5} more)"
    msg = (
        f"{source}: {len(issues)} prescription aperture(s) exceed the "
        f"simulation grid (N={N}, dx={dx*1e6:.3f} um, "
        f"semi={grid_semi*1e3:.3f} mm). "
        f"Largest is {biggest_label} with semi_diameter="
        f"{biggest_sd*1e3:.3f} mm "
        f"({biggest_gap*1e3:+.3f} mm beyond the grid); the field will "
        f"be truncated at the grid edge during propagation, "
        f"silently dropping energy the real lens would have "
        f"transmitted. Consider increasing N or dx so "
        f"N*dx/2 >= max(semi_diameter). "
        f"Affected surfaces: {body}."
    )
    warnings.warn(msg, UserWarning, stacklevel=stacklevel)


# ---------------------------------------------------------------------------
# Multi-surface real lens (split-step refraction + ASM in glass)
# ---------------------------------------------------------------------------

def apply_real_lens(E_in, lens_prescription, wavelength, dx,
                    bandlimit=True, fresnel=False, slant_correction=False,
                    absorption=False, seidel_correction=False,
                    seidel_poly_order=6, progress=None,
                    use_gpu=False, wave_propagator='asm'):
    """
    Propagate a field through a real lens defined by a surface prescription.

    Models the lens as a sequence of refracting phase screens (one per
    surface) with angular-spectrum propagation through the glass between
    them.  Captures exact surface sag (spherical aberration and higher
    orders), diffraction during in-glass propagation, thickness effects, and
    compound lenses (doublets, triplets, etc.).

    The default behaviour uses the **paraxial** thin-element OPD
    ``(n2-n1)*sag`` for the per-surface phase screen.  Empirically this
    gives equally good or better OPD agreement with a geometric ray
    trace as the slant-corrected formula, because the angular-spectrum
    propagation between surfaces already encodes most of the obliquity
    physics.  Pass ``slant_correction=True`` to use the generalised
    ``n2*sag/cos(theta_t) - n1*sag/cos(theta_i)`` formula -- helpful in
    a few specific geometries (asymmetric meniscus, very steep
    asphere) but not a universal improvement.

    Optional opt-in features add further physical realism:

    * ``fresnel=True`` -- multiply by s/p-averaged Fresnel amplitude
      transmission at each surface using local angle of incidence derived
      from the surface normal.  Captures wavelength/index-dependent
      throughput (~4% loss per uncoated air-glass interface) and works
      naturally with complex refractive indices.
    * ``slant_correction=True`` -- replace the paraxial OPD
      ``(n2-n1)*sag`` with the generalized thin-element OPD
      ``n2*sag/cos(theta_t) - n1*sag/cos(theta_i)``, which is accurate at
      larger angles of incidence (faster lenses, off-axis input).
    * ``absorption=True`` -- apply bulk attenuation
      ``exp(-2*pi*kappa*thickness/wavelength)`` between surfaces using the
      imaginary part of the in-medium index from
      :func:`get_glass_index_complex`.

    Per-surface realism additions (set in the prescription dict, all
    optional and backward-compatible):

    * ``"clear_aperture"`` -- float, mechanical clear aperture diameter at
      this surface [m].  Field outside is zeroed (vignetting).
    * ``"decenter"`` -- ``(dx, dy)`` lateral offset of this surface [m].
    * ``"tilt"`` -- ``(tx, ty)`` small-angle surface tilt [rad].  Adds a
      linear sag ramp ``tx*x + ty*y`` to the surface.
    * ``"form_error"`` -- 2D ndarray (same shape as the field) of additive
      sag perturbation [m].  Use to inject measured figure error or
      synthetic Zernike form error.

    The prescription dict may also specify ``"stop_index"`` (int) to apply
    the global ``"aperture_diameter"`` at a specific surface (the aperture
    stop) rather than at the entrance.

    Parameters
    ----------
    E_in : ndarray (complex, N x N)
        Input electric field.
    lens_prescription : dict
        Required keys:

        ``"surfaces"`` : list of dict
            Each surface dict contains:

            - ``"radius"`` : float -- radius of curvature [m] (inf = flat)
            - ``"conic"`` : float -- conic constant (0 = sphere)
            - ``"aspheric_coeffs"`` : dict or None -- {4: A4, 6: A6, ...}
            - ``"glass_before"`` : str -- glass name before this surface
            - ``"glass_after"``  : str -- glass name after this surface
            - ``"clear_aperture"`` : float, optional -- per-surface aperture [m]
            - ``"decenter"`` : (dx, dy), optional -- lateral offset [m]
            - ``"tilt"`` : (tx, ty), optional -- small-angle tilt [rad]
            - ``"form_error"`` : ndarray, optional -- additive sag map [m]

        ``"thicknesses"`` : list of float
            Center spacing [m] between consecutive surfaces.

        Optional keys:

        ``"aperture_diameter"`` : float -- clear aperture [m] (entrance, or
            applied at ``stop_index`` if provided).
        ``"stop_index"`` : int -- index of the surface that holds the
            aperture stop.
        ``"name"`` : str -- human-readable label.

    wavelength : float
        Free-space wavelength [m].
    dx : float
        Grid spacing [m] (square grid assumed).
    bandlimit : bool
        Apply band-limiting in ASM propagation steps (default True).
    fresnel : bool
        Apply Fresnel amplitude transmission at each surface.
    slant_correction : bool, default False
        Use the generalised thin-element OPD with local angle of
        incidence: ``n2*sag/cos(theta_t) - n1*sag/cos(theta_i)``.  Off
        by default because the simple paraxial formula
        ``(n2-n1)*sag`` typically gives equal or better agreement
        with geometric ray-traced OPD (see
        ``validation/real_lens_opd``).
    absorption : bool
        Apply bulk attenuation through each glass region using the
        extinction coefficient from :func:`get_glass_index_complex`.
    seidel_correction : bool, default False
        Add a "Seidel-style" radially-symmetric OPD correction at the
        exit pupil derived from a 1-D geometric ray-trace fan.  A
        polynomial is fit to the difference between the geometric
        ray OPL and the analytic thin-element OPL (``(n2-n1)*sag``),
        then applied as a radial phase screen on the way out.
        Captures ~3-5x improvement on cemented doublets at essentially
        no extra cost (~41 rays traced, one polynomial fit, one 2-D
        phase multiplication).  **Off by default** because: (a) well-
        corrected singlets already achieve sub-30 nm residual against
        the geometric ray trace via the thin-element model alone, and
        (b) the Seidel correction can inject polynomial-fit artefacts
        of order 100 nm on such systems (the analytic formula doesn't
        model the Fresnel ASM contribution exactly).  A 50 nm RMS
        correction-amplitude threshold is applied internally to skip
        the correction when the thin-element model is already good
        enough.  Recommended: turn on for ``AC254*``-class cemented
        doublets and similar multi-surface curved-interface systems;
        leave off for plano-convex singlets and similar well-behaved
        cases, or use :func:`apply_real_lens_traced` for uniformly
        high accuracy.
    seidel_poly_order : int, default 8
        Highest even power of the radial polynomial fit used for the
        Seidel correction.  Order 4 is classical spherical-aberration
        (``a*r^4``); order 8 includes 6th and 8th-order spherical
        terms; higher is rarely beneficial because the fit is limited
        by the 1-D sampling rather than by the polynomial basis.

    Returns
    -------
    E_out : ndarray (complex, N x N)

    Notes
    -----
    With ``slant_correction=False`` and all other optional features off,
    the function reduces to the original paraxial-OPD, lossless,
    perfectly-aligned, single-aperture model and is bit-for-bit backward
    compatible with prescriptions that omit the new keys.

    GPU usage (3.1.10+)
    -------------------
    Pass ``use_gpu=True`` or a CuPy array as ``E_in`` to run the whole
    phase-screen + in-glass ASM pipeline on GPU.  Default is ``False``
    to preserve the existing CPU path bit-for-bit.  When enabled:

    * ``E_in`` is promoted to the device via ``cp.asarray`` (or kept
      as-is if already a CuPy array).
    * All meshgrids, sag arrays, and per-surface phase screens are
      built natively on the device using the CuPy namespace.
    * Internal ``angular_spectrum_propagate`` calls auto-detect the
      CuPy input and use the library's existing cuFFT-backed ASM.
    * The numexpr fused-phase-screen path is skipped on GPU (numexpr
      is CPU-only); CuPy's native elementwise kernels are used
      instead.
    * The return value is a CuPy array when ``use_gpu=True``.  Use
      ``cp.asnumpy(E_out)`` to pull it back to the host when needed.

    The returned array type follows ``use_gpu``: host -> host, device
    -> device.  Mixed-dtype callers (e.g. a complex64 host array
    promoted to the device) remain in their starting precision.
    """
    # Pre-flight grid vs prescription-aperture check.  If any surface's
    # semi-aperture exceeds the simulation grid, ASM will silently
    # truncate the field at the grid edge and lose energy that the real
    # hardware would have transmitted.  Issue a UserWarning once per
    # call site (Python's default warning filter dedups by source line).
    try:
        N_grid = int(np.shape(E_in)[0])
        _warn_if_aperture_exceeds_grid(
            lens_prescription, N_grid, dx, source='apply_real_lens')
    except Exception:
        pass

    # Select the array namespace: numpy by default; cupy if the caller
    # opted in via ``use_gpu=True`` OR passed in a cupy array.
    if use_gpu or _is_cupy_array(E_in):
        if not CUPY_AVAILABLE:
            raise ImportError(
                "use_gpu=True (or CuPy input) requires the 'cupy' package.  "
                "Install cupy-cuda12x (or matching CUDA version) or call "
                "with use_gpu=False to stay on the CPU path.")
        xp = cp
    else:
        xp = np

    surfaces = lens_prescription['surfaces']
    thicknesses = lens_prescription['thicknesses']
    aperture = lens_prescription.get('aperture_diameter')
    stop_index = lens_prescription.get('stop_index')

    assert len(thicknesses) == len(surfaces) - 1, (
        f"Need {len(surfaces) - 1} thicknesses for {len(surfaces)} surfaces, "
        f"got {len(thicknesses)}"
    )

    Ny, Nx = E_in.shape
    k0 = 2 * np.pi / wavelength

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dx
    X, Y = xp.meshgrid(x, y)
    h_sq_axis = X ** 2 + Y ** 2  # axis-centered distance, used for stop aperture

    # Preserve the caller's complex dtype (complex128 or complex64).
    # The numexpr ``out=E`` path below evaluates the phase screen
    # expression in complex128 internally and casts to E.dtype at
    # the final store, which is the documented mitigation that keeps
    # the per-surface OPD accurate even for large ``k0 * opd``
    # arguments regardless of storage precision.  The numpy fallback
    # restores the original dtype explicitly at the end of each
    # surface.  On the GPU path, numexpr isn't available so we use
    # the plain multiplication fallback throughout.
    if xp is cp:
        # Ensure E is a device array of appropriate complex dtype
        if not _is_cupy_array(E_in):
            E = cp.asarray(E_in)
        else:
            E = E_in.copy()
        if not cp.iscomplexobj(E):
            from .propagation import DEFAULT_COMPLEX_DTYPE
            E = E.astype(DEFAULT_COMPLEX_DTYPE)
    else:
        if np.iscomplexobj(E_in):
            E = E_in.copy()
        else:
            from .propagation import DEFAULT_COMPLEX_DTYPE
            E = E_in.astype(DEFAULT_COMPLEX_DTYPE)

    # Entrance aperture (only if no explicit stop surface specified)
    if aperture is not None and stop_index is None:
        E = xp.where(h_sq_axis <= (aperture / 2) ** 2, E, 0.0 + 0.0j)

    # Resolve glass names once.  Use complex form so we can recover kappa for
    # absorption while still having the real part for geometry/Snell.
    resolved = []
    for surf in surfaces:
        if absorption or fresnel:
            n1c = get_glass_index_complex(surf['glass_before'], wavelength)
            n2c = get_glass_index_complex(surf['glass_after'], wavelength)
        else:
            n1c = complex(get_glass_index(surf['glass_before'], wavelength), 0.0)
            n2c = complex(get_glass_index(surf['glass_after'], wavelength), 0.0)
        resolved.append((n1c, n2c))


    from .progress import call_progress
    n_surf = len(surfaces)
    for i, surf in enumerate(surfaces):
        call_progress(progress, 'apply_real_lens',
                      i / max(n_surf, 1),
                      f'surface {i + 1}/{n_surf}')
        R = surf['radius']
        kc = surf.get('conic', 0.0)
        asph = surf.get('aspheric_coeffs')
        # Optional anamorphic fields (backward-compatible -- present
        # only on biconic / cylindrical / toroidal surfaces).
        R_y = surf.get('radius_y')
        kc_y = surf.get('conic_y')
        asph_y = surf.get('aspheric_coeffs_y')
        n1c, n2c = resolved[i]
        n1r, n2r = n1c.real, n2c.real

        # ---- Decenter --------------------------------------------------
        decenter = surf.get('decenter') or (0.0, 0.0)
        if decenter[0] == 0.0 and decenter[1] == 0.0:
            # Alias the axis-centered grids.  Downstream code only reads
            # Xs/Ys/h_sq and creates new arrays when combining them
            # (e.g. ``sag + tilt[0]*Xs``), so aliasing is safe.  Saves
            # three float64 N x N allocations per surface (~24 GB at
            # N=32768).
            Xs = X
            Ys = Y
            h_sq = h_sq_axis
        else:
            Xs = X - decenter[0]
            Ys = Y - decenter[1]
            h_sq = Xs ** 2 + Ys ** 2

        # ---- Base sag (conic + asphere; biconic if radius_y given) ----
        if R_y is not None:
            sag = surface_sag_biconic(
                Xs, Ys, R_x=R, R_y=R_y,
                conic_x=kc, conic_y=kc_y,
                aspheric_coeffs=asph,
                aspheric_coeffs_y=asph_y)
        else:
            sag = _surface_sag_general(h_sq, R, kc, asph)

        # ---- Tilt (small-angle linear ramp added to sag) --------------
        tilt = surf.get('tilt') or (0.0, 0.0)
        if tilt[0] != 0.0 or tilt[1] != 0.0:
            sag = sag + tilt[0] * Xs + tilt[1] * Ys

        # ---- Form error map -------------------------------------------
        form_err = surf.get('form_error')
        if form_err is not None:
            sag = sag + form_err

        # ---- Local surface normal -> angles of incidence/refraction ---
        # Needed only for fresnel and/or slant_correction.  When on,
        # the legacy code keeps cos_ti / cos_tt / sin2_tt / etc. all
        # alive simultaneously (~6 N x N float64 arrays), which dwarfs
        # the field memory at N >= 4096.  3.2.14: drop intermediates
        # immediately and free the grad components after grad_sq is
        # built.  At N=8192 this cuts peak refraction-step memory
        # from ~5 GB to ~1.5 GB without changing the math.
        if fresnel or slant_correction:
            dsag_dy, dsag_dx = xp.gradient(sag, dx, dx)
            grad_sq = dsag_dx ** 2 + dsag_dy ** 2
            # Free the gradient components -- only grad_sq is needed
            # for the rest of the refraction pipeline.
            del dsag_dx, dsag_dy
            # cos_ti / sin2_ti share `grad_sq + 1.0`; build the safe
            # versions directly to avoid two extra full-grid arrays.
            one_plus_g = 1.0 + grad_sq
            cos_ti = 1.0 / xp.sqrt(one_plus_g)
            sin2_ti = grad_sq / one_plus_g
            del one_plus_g
            del grad_sq
            sin2_tt = (n1r / n2r) ** 2 * sin2_ti
            cos_tt = xp.sqrt(xp.maximum(1.0 - sin2_tt, 0.0))
            cos_ti_safe = xp.maximum(cos_ti, 1e-3)
            cos_tt_safe = xp.maximum(cos_tt, 1e-3)
            # cos_ti / cos_tt are no longer needed -- only the _safe
            # versions and sin2_tt (for TIR mask) survive.
            del cos_ti, cos_tt

        # ---- Refraction OPD (thin-element phase screen) ---------------
        # Note: a BPM-style "interface sub-slicing" mode was
        # prototyped (see git history) but does not deliver the
        # accuracy improvement it promises on sharp air-glass
        # interfaces: simple single-reference-medium BPM requires
        # sub-wavelength axial slabs, which for realistic
        # interface thicknesses (~100 um) means 1000s of slabs --
        # too slow.  Sub-wavelength slabs are needed because the
        # BPM approximation (reference-medium Fresnel kernel + local
        # phase correction) breaks down for step-discontinuous
        # media.  Users needing better than thin-element accuracy
        # should use ``apply_real_lens_traced`` which bypasses this
        # limitation entirely by ray-tracing each pixel.
        if slant_correction:
            opd = n2r * sag / cos_tt_safe - n1r * sag / cos_ti_safe
        else:
            opd = (n2r - n1r) * sag
        if xp is np and NUMEXPR_AVAILABLE and E.size >= _NUMEXPR_MIN_SIZE:
            # Fused multiply + complex exp in one threaded, chunked pass
            # -- avoids the three complex128 N x N temporaries that
            # ``E * np.exp(-1j * k0 * opd)`` otherwise materialises.
            # With ``out=E``, numexpr evaluates the expression at
            # complex128 internal precision and casts only at the
            # final store, so complex64 E gets a double-precision
            # phase accumulation + single-precision storage.
            # CPU only -- numexpr has no GPU backend.
            _ne.evaluate(
                'E * exp(-1j * k0 * opd)',
                local_dict={'E': E, 'k0': k0, 'opd': opd},
                out=E,
            )
        else:
            # Fallback for GPU (xp is cp) or small CPU arrays: compute
            # exp() in the array backend's precision, then cast back
            # to E's dtype so we don't silently upcast a complex64
            # field to complex128.  CuPy's exp is fused at kernel
            # level so the "three temporaries" concern doesn't apply
            # the same way on device.
            phase_exp = xp.exp(-1j * k0 * opd)
            if phase_exp.dtype != E.dtype:
                phase_exp = phase_exp.astype(E.dtype)
            E = E * phase_exp

        # ---- Fresnel amplitude transmission ---------------------------
        if fresnel:
            # s and p amplitude transmission, complex n supported
            denom_s = n1c * cos_ti_safe + n2c * cos_tt_safe
            denom_p = n2c * cos_ti_safe + n1c * cos_tt_safe
            t_s = 2.0 * n1c * cos_ti_safe / denom_s
            t_p = 2.0 * n1c * cos_ti_safe / denom_p
            # Scalar approximation: average the two amplitude coefficients
            E = E * 0.5 * (t_s + t_p)
            # Suppress regions that went into TIR
            E = xp.where(sin2_tt < 1.0, E, 0.0 + 0.0j)

        # ---- Per-surface clear aperture (vignetting) ------------------
        clear_ap = surf.get('clear_aperture')
        if clear_ap is not None:
            E = xp.where(h_sq <= (clear_ap / 2) ** 2, E, 0.0 + 0.0j)

        # ---- Aperture stop applied at this surface --------------------
        if stop_index is not None and i == stop_index and aperture is not None:
            E = xp.where(h_sq_axis <= (aperture / 2) ** 2, E, 0.0 + 0.0j)

        # ---- Propagate through glass to the next surface --------------
        if i < len(surfaces) - 1:
            n_medium_r = n2r
            n_medium_kappa = n2c.imag
            thickness = thicknesses[i]
            lam_medium = wavelength / n_medium_r
            # Default path: ASM (auto-detects cupy backend from E dtype).
            # Expert override via ``wave_propagator`` -- supports four
            # values:
            #
            #   'asm'                (default) angular spectrum method
            #   'sas'                scalable angular spectrum + resample
            #   'fresnel'            single-FFT Fresnel + resample
            #   'rayleigh_sommerfeld' Rayleigh-Sommerfeld convolution
            #
            # Physically ASM is the right choice for the short (mm) glass
            # thicknesses typical of lenses; the other three are exposed
            # for research, cross-validation, and pipelines that want a
            # single propagator used consistently throughout.  Both
            # Fresnel and SAS produce an output grid with a much
            # smaller pitch than the input when z is small (mm-scale),
            # so the back-resample to ``dx`` loses most of the
            # high-spatial-frequency content the chirp produced; that
            # loss is a feature of the physical regime, not a bug in
            # the dispatcher.
            if wave_propagator == 'sas':
                from .propagation import (
                    scalable_angular_spectrum_propagate, resample_field)
                E, dx_new, _ = scalable_angular_spectrum_propagate(
                    E, thickness, lam_medium, dx)
                if abs(dx_new - dx) > dx * 1e-6:
                    E, _ = resample_field(
                        E, dx_new, dx, N_out=E.shape[-1])
            elif wave_propagator == 'fresnel':
                from .propagation import (
                    fresnel_propagate, resample_field)
                E, dx_new, _ = fresnel_propagate(
                    E, thickness, lam_medium, dx)
                if abs(dx_new - dx) > dx * 1e-6:
                    E, _ = resample_field(
                        E, dx_new, dx, N_out=E.shape[-1])
            elif wave_propagator == 'rayleigh_sommerfeld':
                from .propagation import rayleigh_sommerfeld_propagate
                E = rayleigh_sommerfeld_propagate(
                    E, thickness, lam_medium, dx)
            elif wave_propagator == 'asm':
                E = angular_spectrum_propagate(
                    E, thickness, lam_medium, dx, bandlimit=bandlimit)
            else:
                raise ValueError(
                    f"apply_real_lens: unknown wave_propagator "
                    f"{wave_propagator!r}.  Supported: 'asm', 'sas', "
                    f"'fresnel', 'rayleigh_sommerfeld'.")
            # Bulk absorption: exp(-2*pi * kappa * t / lambda0)
            if absorption and n_medium_kappa != 0.0:
                E = E * xp.exp(-k0 * n_medium_kappa * thickness)

    # ----- Seidel correction ------------------------------------------
    # Apply a ray-trace-derived radial phase correction that captures
    # the residual OPD the thin-element model misses.  This is a
    # generalised "Seidel"-style correction: we ray-trace a 1-D fan,
    # take the difference between the geometric OPL and the analytic
    # thin-element OPL at each height, fit a radial even polynomial,
    # and apply that as an additional phase screen at the exit pupil.
    #
    # Captures all orders of spherical aberration up to
    # ``seidel_poly_order``, plus any residual caused by the uniform-
    # slab approximation at each interface.  For rotationally
    # symmetric on-axis collimated input the correction is radially
    # symmetric and applied per pixel via r = sqrt(x^2 + y^2).
    if seidel_correction and aperture is not None:
        # Local imports to avoid circular dep at module load
        from .raytrace import (
            trace as _rt_trace, _make_bundle as _rt_make_bundle,
            surfaces_from_prescription as _rt_surfaces_from_prescription,
        )
        r_pupil = 0.5 * aperture
        n_fan = 41
        h_fan = np.linspace(-0.9 * r_pupil, 0.9 * r_pupil, n_fan)
        z_arr = np.zeros_like(h_fan)
        fan = _rt_make_bundle(
            x=h_fan, y=z_arr, L=z_arr, M=z_arr,
            wavelength=wavelength)
        surfs_fan = _rt_surfaces_from_prescription(lens_prescription)
        res_fan = _rt_trace(fan, surfs_fan, wavelength)
        final_fan = res_fan.image_rays
        alive_fan = final_fan.alive
        if alive_fan.sum() >= 5:
            opl_ray = final_fan.opd[alive_fan]
            h_alive = h_fan[alive_fan]
            # Analytic height-dependent OPD deposited by the phase-
            # screens above: sum over surfaces of (n2-n1)*sag_i(h).
            # With the sign convention ``phase_screen = exp(-i*k*opd)``
            # this DECREASES the wave's OPL at edges for a positive
            # lens -- the wave's height-dependent OPL is therefore
            # ``-sum (n2-n1)*sag``.  The full wave output includes
            # further Fresnel ASM contributions we don't try to model
            # analytically.
            opl_analytic = np.zeros_like(h_alive)
            for surf_i in surfaces:
                R_i = surf_i['radius']
                kc_i = surf_i.get('conic', 0.0)
                asph_i = surf_i.get('aspheric_coeffs')
                R_y_i = surf_i.get('radius_y')
                n1r_i = get_glass_index(surf_i['glass_before'], wavelength)
                n2r_i = get_glass_index(surf_i['glass_after'], wavelength)
                if R_y_i is not None:
                    sag_fan_i = surface_sag_biconic(
                        h_alive, np.zeros_like(h_alive),
                        R_x=R_i, R_y=R_y_i,
                        conic_x=kc_i,
                        conic_y=surf_i.get('conic_y'),
                        aspheric_coeffs=asph_i,
                        aspheric_coeffs_y=surf_i.get('aspheric_coeffs_y'))
                else:
                    sag_fan_i = _surface_sag_general(
                        h_alive * h_alive, R_i, kc_i, asph_i)
                opl_analytic = opl_analytic + (
                    (n2r_i - n1r_i) * sag_fan_i)
            i_ax = int(np.argmin(np.abs(h_alive)))
            delta_ray = opl_ray - opl_ray[i_ax]
            opl_wave_rel = -(opl_analytic - opl_analytic[i_ax])
            correction = delta_ray - opl_wave_rel
            # Fit even-power polynomial in normalised pupil coord.
            rho = h_alive / r_pupil
            max_order = max(2, int(seidel_poly_order))
            even_powers = np.arange(2, max_order + 2, 2)
            A = np.column_stack([rho ** p for p in even_powers])
            coeffs, *_ = np.linalg.lstsq(A, correction, rcond=None)
            # Suppress fitting noise: if the RMS correction across the
            # fan is already well below typical simulation residual
            # (~ a few nm), skip application to avoid injecting
            # polynomial-fit artefacts into otherwise-clean fields.
            corr_rms = float(np.sqrt(np.mean(correction ** 2)))
            if corr_rms > 50e-9:  # > 50 nm RMS to be worth applying
                # h_sq_axis is in the array backend (xp) already;
                # coeffs came from a CPU lstsq so scalar-broadcast
                # them into xp.  Final phase screen multiplies E on
                # the target device.
                rho_map_sq = h_sq_axis / (r_pupil ** 2)
                corr_map = xp.zeros_like(rho_map_sq)
                for p, c in zip(even_powers, coeffs):
                    corr_map = corr_map + float(c) * rho_map_sq ** (p // 2)
                corr_map = xp.where(rho_map_sq <= 1.0, corr_map, 0.0)
                E = E * xp.exp(+1j * k0 * corr_map)

    call_progress(progress, 'apply_real_lens', 1.0, 'done')
    return E


# ---------------------------------------------------------------------------
# Cylindrical lens
# ---------------------------------------------------------------------------
# Per-pixel ray-traced phase-screen lens model
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Optional Numba JIT for the polynomial-evaluator inner loop.
#
# The hot path of _Cheb2DEvaluator.ev_value_and_grad is a doubly-nested
# reduction over (basis_term, output_sample).  Plain NumPy executes it
# as a chain of broadcast multiplies and sum-reductions with a handful
# of allocated temporaries; a @njit kernel collapses that to a single
# tight loop with zero temporaries and thread-parallel output rows.
#
# Guarded import -- fallback to pure-xp path (which is fine on NumPy and
# REQUIRED on CuPy) when numba isn't installed.
# Note: ``_NUMBA_AVAILABLE`` / ``_njit`` / ``_prange`` are imported once
# at module top (see top of file).  This block historically had its own
# import guard; the early import pulled the names earlier so the
# ``_aspheric_sag_accum_numba`` helper at line ~150 can use them too.
# ---------------------------------------------------------------------------


if _NUMBA_AVAILABLE:
    @_njit(cache=True, parallel=True, fastmath=True)
    def _cheb2d_val_grad_numba(coeffs, K1, K2, u_flat, v_flat, max_order):
        """Combined Chebyshev value + gradient via in-place recurrence.

        Computes f(u, v), df/du, df/dv at every (u_flat[i], v_flat[i])
        sample in parallel.  Chebyshev T and U (second kind) values are
        generated by 3-term recurrence on the stack per sample -- no
        Vandermonde matrices are materialised.  This implements the
        Clenshaw-style "#3" optimisation: O(order) stack work per
        sample instead of an O(N x order) materialised Vandermonde.

        Parameters
        ----------
        coeffs : (M,) float64   -- polynomial coefficients in total-degree order
        K1, K2 : (M,) int64     -- multi-indices (kx, ky) for each term
        u_flat, v_flat : (N,) float64 -- normalised sample coords in [-1, 1]
        max_order : int         -- maximum individual Chebyshev order

        Returns
        -------
        f, fx_u, fy_v : three (N,) float64 arrays: value and du/dv-partials
        in normalised coordinates.  Caller applies chain rule for
        physical derivatives.
        """
        N = u_flat.shape[0]
        M = coeffs.shape[0]
        f = np.zeros(N)
        fx = np.zeros(N)
        fy = np.zeros(N)

        for i in _prange(N):
            u = u_flat[i]
            v = v_flat[i]

            # T_n(u), T_n(v): first kind, by 3-term recurrence
            # T_0 = 1, T_1 = u, T_{n+1} = 2u T_n - T_{n-1}
            Tu = np.empty(max_order + 1)
            Tv = np.empty(max_order + 1)
            Tu[0] = 1.0; Tv[0] = 1.0
            if max_order >= 1:
                Tu[1] = u; Tv[1] = v
            for n in range(2, max_order + 1):
                Tu[n] = 2.0 * u * Tu[n - 1] - Tu[n - 2]
                Tv[n] = 2.0 * v * Tv[n - 1] - Tv[n - 2]

            # T'_n(u) = n * U_{n-1}(u); U_0 = 1, U_1 = 2u, U_{n+1} = 2u U_n - U_{n-1}
            # We store dTu[n] = T'_n(u) directly for n = 0..max_order
            dTu = np.zeros(max_order + 1)
            dTv = np.zeros(max_order + 1)
            if max_order >= 1:
                dTu[1] = 1.0          # T'_1 = 1 * U_0 = 1
                dTv[1] = 1.0
                if max_order >= 2:
                    U_prev_u = 1.0    # U_0
                    U_u = 2.0 * u     # U_1
                    U_prev_v = 1.0
                    U_v = 2.0 * v
                    dTu[2] = 2.0 * U_u    # T'_2 = 2 * U_1
                    dTv[2] = 2.0 * U_v
                    for n in range(3, max_order + 1):
                        U_next_u = 2.0 * u * U_u - U_prev_u
                        U_next_v = 2.0 * v * U_v - U_prev_v
                        U_prev_u = U_u; U_u = U_next_u
                        U_prev_v = U_v; U_v = U_next_v
                        dTu[n] = n * U_u
                        dTv[n] = n * U_v

            # Accumulate coefficient-weighted sum over multi-indices
            acc_f = 0.0
            acc_fx = 0.0
            acc_fy = 0.0
            for m in range(M):
                kx = K1[m]
                ky = K2[m]
                c = coeffs[m]
                tu = Tu[kx]
                tv = Tv[ky]
                acc_f  += c * tu * tv
                acc_fx += c * dTu[kx] * tv
                acc_fy += c * tu * dTv[ky]
            f[i] = acc_f
            fx[i] = acc_fx
            fy[i] = acc_fy
        return f, fx, fy


def _get_array_module(arr):
    """Return the array namespace (numpy or cupy) for ``arr``.

    Enables array-API polymorphism: code that uses only namespace-
    agnostic operations (xp.asarray, xp.sum, xp.meshgrid, ...) runs
    unchanged on NumPy or CuPy arrays.  Gracefully degrades to NumPy
    when CuPy isn't installed.
    """
    try:
        import cupy as _cp
        if isinstance(arr, _cp.ndarray):
            return _cp
    except ImportError:
        pass
    return np


class _Cheb2DEvaluator:
    """2-D Chebyshev tensor-product polynomial fit with an API compatible
    with a SciPy ``RectBivariateSpline`` for the subset used by
    :func:`apply_real_lens_traced` -- specifically the ``ev(x, y)``,
    ``ev(x, y, dx=1)``, and ``ev(x, y, dy=1)`` methods.

    This is the polynomial equivalent of the default spline interpolation
    used by ``apply_real_lens_traced``'s Newton inversion, enabled when
    ``newton_fit='polynomial'``.  For smooth refractive lenses where the
    entrance->exit coordinate map and the OPL are essentially polynomials
    of total degree up to 6 (all Seidel + higher-order aberrations of
    reasonable orders), this is both faster (closed-form analytic
    derivatives, no Fortran spline calls) and more accurate (no cubic
    truncation error on the coarse grid).

    Architecture
    ------------
    * **Array-API polymorphic**: the ``xp`` constructor kwarg selects
      the array backend (default :mod:`numpy`).  Pass ``xp=cupy`` to
      run the fit and evaluation on the GPU with zero code changes
      here -- every internal operation uses ``self.xp``'s namespace.
    * **Combined value + gradient** (``ev_value_and_grad``): returns
      ``(f, df/dx, df/dy)`` in one shared-basis pass, avoiding the 3x
      redundant Vandermonde builds that the separate ``.ev(dx=1)`` and
      ``.ev(dy=1)`` calls would do.
    * **Optional Numba JIT fastpath**: on the NumPy backend, if
      :mod:`numba` is importable, the combined evaluation drops into a
      ``@njit(parallel=True, fastmath=True)`` kernel that runs the
      Chebyshev recurrence inline per sample (no Vandermonde
      materialised).  Typical 3-10x speedup over the pure-NumPy path.
      Silently skipped on the CuPy backend -- the pure-xp fallback
      runs on GPU instead.

    GPU note
    --------
    To use this class on GPU with CuPy::

        import cupy as cp
        ev = _Cheb2DEvaluator(xs_in_cp, ys_in_cp, values_cp,
                              order=6, xp=cp)
        # Later:
        f, fx, fy = ev.ev_value_and_grad(xa_cp, ya_cp)

    All arrays (inputs, outputs, internal state) stay on the GPU;
    there is no implicit host-device copy.  The Newton loop in
    :func:`apply_real_lens_traced` is unchanged as long as ``xa, ya``
    are CuPy arrays.  A future ``use_gpu=True`` kwarg could dispatch
    this automatically.
    """

    __slots__ = ('order', 'coeffs', 'xmin', 'xmax', 'ymin', 'ymax',
                 '_mi', '_K1', '_K2', 'xp')

    def __init__(self, xs_in, ys_in, values, order=6, xp=None):
        if xp is None:
            xp = _get_array_module(values)
        self.xp = xp
        # The fit itself (a tiny lstsq -- typically a few hundred rows
        # by 28-70 terms) is always performed on CPU via NumPy, even
        # when xp=cupy.  Three reasons:
        #   1. NumPy lstsq is reliable and dependency-free; cupy.linalg.
        #      lstsq needs cuSOLVER which isn't guaranteed to be
        #      present on every cupy install.
        #   2. The fit is O(1) per apply_real_lens_traced call (one-
        #      time cost) and is negligible vs per-pixel Newton work.
        #      Routing it via the CPU has no measurable impact.
        #   3. The payoff from xp=cupy is in the Newton hot loop (N^2
        #      evaluator calls), where it does matter -- only the
        #      fitted coefficients need to live on the device.
        xs_np = np.asarray(xp.asnumpy(xs_in) if xp is not np else xs_in,
                            dtype=np.float64)
        ys_np = np.asarray(xp.asnumpy(ys_in) if xp is not np else ys_in,
                            dtype=np.float64)
        vals_np = np.asarray(xp.asnumpy(values) if xp is not np else values,
                              dtype=np.float64)
        self.order = int(order)
        # Scalars extracted as Python floats so chain-rule multiplies
        # stay backend-agnostic and don't pull host-device copies later.
        self.xmin = float(xs_np.min()); self.xmax = float(xs_np.max())
        self.ymin = float(ys_np.min()); self.ymax = float(ys_np.max())
        # Build total-degree multi-indices (kx, ky) with kx + ky <= order
        self._mi = [(kx, ky)
                     for kx in range(order + 1)
                     for ky in range(order + 1 - kx)]
        n_terms = len(self._mi)
        # Fit on CPU using NumPy
        X_np, Y_np = np.meshgrid(xs_np, ys_np, indexing='ij')
        u_np = (2.0 * X_np - (self.xmin + self.xmax)) / (self.xmax - self.xmin)
        v_np = (2.0 * Y_np - (self.ymin + self.ymax)) / (self.ymax - self.ymin)
        K1_np = np.asarray([m[0] for m in self._mi], dtype=np.int64)
        K2_np = np.asarray([m[1] for m in self._mi], dtype=np.int64)
        Tu_np = _cheb_vand_2d(u_np, order, np)
        Tv_np = _cheb_vand_2d(v_np, order, np)
        A_full = (Tu_np[K1_np] * Tv_np[K2_np]).reshape(n_terms, -1).T
        vals_flat = vals_np.ravel()
        finite = np.isfinite(vals_flat)
        if finite.all():
            A = A_full
            rhs = vals_flat
        else:
            A = A_full[finite, :]
            rhs = vals_flat[finite]
        c_np, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        # Push coefficients + index arrays onto the target backend
        self.coeffs = xp.asarray(c_np, dtype=xp.float64)
        self._K1 = xp.asarray(K1_np, dtype=xp.int64)
        self._K2 = xp.asarray(K2_np, dtype=xp.int64)

    def _to_u(self, x):
        return (2.0 * x - (self.xmin + self.xmax)) / \
                 (self.xmax - self.xmin)

    def _to_v(self, y):
        return (2.0 * y - (self.ymin + self.ymax)) / \
                 (self.ymax - self.ymin)

    # ----------------------------------------------------------------
    # Backward-compat single-quantity API (RectBivariateSpline.ev()).
    # Supports dx=0/1 and dy=0/1 (up to first derivatives).
    # ----------------------------------------------------------------
    def ev(self, x, y, dx=0, dy=0):
        """Evaluate polynomial (or partial derivative) at (x, y).

        Compatible subset of SciPy RectBivariateSpline.ev: supports
        dx=0/1 and dy=0/1 (up to first derivatives).  When multiple
        derivatives are needed at the same (x, y), prefer
        :meth:`ev_value_and_grad` -- one call returns all three.
        """
        if (dx, dy) in ((0, 0), (1, 0), (0, 1)):
            f, fx, fy = self.ev_value_and_grad(x, y)
            if dx == 0 and dy == 0:
                return f
            if dx == 1 and dy == 0:
                return fx
            return fy
        raise NotImplementedError(
            f"_Cheb2DEvaluator.ev with dx={dx}, dy={dy} not supported; "
            f"only 0th and 1st derivatives in a single axis.")

    # ----------------------------------------------------------------
    # Combined value + gradient (#6) -- primary entry point for the
    # Newton loop in apply_real_lens_traced.  Shares Chebyshev basis
    # work across all three quantities.  Uses the Numba fastpath (#1)
    # when available on the NumPy backend; otherwise a pure-xp
    # implementation that runs on NumPy or CuPy alike.
    # ----------------------------------------------------------------
    def ev_value_and_grad(self, x, y):
        """Evaluate the polynomial and both partial derivatives in one
        pass.

        Returns
        -------
        f, df/dx, df/dy : arrays with the broadcast shape of (x, y)
            Value and physical-space partial derivatives (chain rule
            applied to undo the ``[-1, 1]`` normalisation).
        """
        xp = self.xp
        x = xp.asarray(x, dtype=xp.float64)
        y = xp.asarray(y, dtype=xp.float64)
        u = self._to_u(x)
        v = self._to_v(y)
        sx = 2.0 / (self.xmax - self.xmin)
        sy = 2.0 / (self.ymax - self.ymin)

        # Numba fastpath on the NumPy backend
        if _NUMBA_AVAILABLE and xp is np:
            u_flat = np.ascontiguousarray(u.ravel(), dtype=np.float64)
            v_flat = np.ascontiguousarray(v.ravel(), dtype=np.float64)
            coeffs = np.ascontiguousarray(self.coeffs, dtype=np.float64)
            K1 = np.ascontiguousarray(self._K1, dtype=np.int64)
            K2 = np.ascontiguousarray(self._K2, dtype=np.int64)
            f_flat, fx_u_flat, fy_v_flat = _cheb2d_val_grad_numba(
                coeffs, K1, K2, u_flat, v_flat, self.order)
            shape = u.shape
            return (f_flat.reshape(shape),
                    fx_u_flat.reshape(shape) * sx,
                    fy_v_flat.reshape(shape) * sy)

        # Pure-xp fallback (always-on; REQUIRED for CuPy backend).
        # Build T and T' Vandermondes once, gather by multi-index, and
        # contract against the coefficient vector with one sum each.
        Tu = _cheb_vand_2d(u, self.order, xp)
        Tv = _cheb_vand_2d(v, self.order, xp)
        dTu = _cheb_deriv_vand_2d(u, self.order, xp)
        dTv = _cheb_deriv_vand_2d(v, self.order, xp)
        # Gather per-basis-term arrays: shape (M, ...u.shape)
        Tu_K = Tu[self._K1]
        Tv_K = Tv[self._K2]
        dTu_K = dTu[self._K1]
        dTv_K = dTv[self._K2]
        # Broadcast coefficients and sum over the basis-term axis.
        c_shape = (len(self._mi),) + (1,) * u.ndim
        c_b = self.coeffs.reshape(c_shape)
        f    = xp.sum(c_b * Tu_K  * Tv_K , axis=0)
        fx_u = xp.sum(c_b * dTu_K * Tv_K , axis=0)
        fy_v = xp.sum(c_b * Tu_K  * dTv_K, axis=0)
        return f, fx_u * sx, fy_v * sy


def _cheb_vand_2d(u, max_k, xp=None):
    """Chebyshev T_k(u) for k=0..max_k as (max_k+1,) + u.shape array.

    Backend-agnostic: pass ``xp=numpy`` (default) or ``xp=cupy`` to run
    on host or device respectively.
    """
    if xp is None:
        xp = _get_array_module(u)
    T = xp.empty((max_k + 1,) + u.shape, dtype=xp.float64)
    T[0] = 1.0
    if max_k >= 1:
        T[1] = u
    for n in range(2, max_k + 1):
        T[n] = 2.0 * u * T[n - 1] - T[n - 2]
    return T


def _cheb_deriv_vand_2d(u, max_k, xp=None):
    """T'_k(u) via T'_n = n U_{n-1}; shape (max_k+1,) + u.shape.

    Backend-agnostic: pass ``xp=numpy`` (default) or ``xp=cupy``.
    """
    if xp is None:
        xp = _get_array_module(u)
    Tp = xp.zeros((max_k + 1,) + u.shape, dtype=xp.float64)
    if max_k < 1:
        return Tp
    U = xp.empty((max_k + 1,) + u.shape, dtype=xp.float64)
    U[0] = 1.0
    if max_k >= 1:
        U[1] = 2.0 * u
    for n in range(2, max_k + 1):
        U[n] = 2.0 * u * U[n - 1] - U[n - 2]
    for n in range(1, max_k + 1):
        Tp[n] = float(n) * U[n - 1]
    return Tp


def _geometric_lens_phase(lens_prescription, wavelength, dx, N):
    """Compute the analytic per-surface sag-phase-screen sum for a lens.

    Returns the *geometric* component of the phase a plane wave would
    acquire after passing through the lens -- equivalent to
    ``np.angle(apply_real_lens(ones, ...))`` except that the ASM
    diffractive correction between surfaces is omitted.

    For smooth refractive lens prescriptions the omitted correction
    scales as ``t * k_perp^2 / (2k)`` where t is glass thickness and
    k_perp is the characteristic spatial-frequency of the sag.  On
    typical F/10+ refractive lenses this is under 10 nm OPL; for
    faster lenses (F/3 or below) validate before trusting.

    Parameters
    ----------
    lens_prescription : dict
        Same format as :func:`apply_real_lens`.
    wavelength : float
        Free-space wavelength [m].
    dx : float
        Grid spacing [m].
    N : int
        Grid size (N x N square).

    Returns
    -------
    phase : ndarray (N, N) float64
        Analytic geometric phase in radians, wrapped to the [-pi, pi]
        range so it can be used interchangeably with
        ``np.angle(E_analytic_pw)``.
    """
    from . import raytrace as _rt
    surfaces = _rt.surfaces_from_prescription(lens_prescription)
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    k0 = 2.0 * np.pi / wavelength

    phase = np.zeros((N, N), dtype=np.float64)

    # Accumulate per-surface sag phase: phi += -k0 * (n_after - n_before) * sag(x, y)
    # This matches the thin-element OPD used inside apply_real_lens's
    # phase-screen model (the default paraxial formula) -- so dropping
    # the ASM step is the only physics difference.
    for surf in surfaces:
        n1 = get_glass_index(surf.glass_before, wavelength)
        n2 = get_glass_index(surf.glass_after, wavelength)
        if abs(n2 - n1) < 1e-15:
            continue   # no refraction
        sag = _rt._surface_sag_xy(X, Y, surf)
        phase = phase + (-k0 * (n2 - n1) * sag)

    # Also add the bulk glass piston (constant k*n*t_i in each glass)
    # since the full apply_real_lens includes this via the ASM in-glass
    # propagation.  The piston is a rigid offset but keeping it
    # preserves absolute-phase consistency when this function is used
    # for the phase_analytic_lens reference.
    for surf in surfaces[:-1]:
        n_mid = get_glass_index(surf.glass_after, wavelength)
        phase = phase + k0 * n_mid * float(surf.thickness)

    # Wrap to match np.angle convention
    return np.angle(np.exp(1j * phase))


def _sample_local_tilts(E_in, wavelength, dx, entrance_x, entrance_y,
                         max_sin=0.5, smooth_sigma_px=4.0,
                         multimode_diagnostic=None):
    """Extract ``(L, M)`` direction cosines for each entrance ray from
    the local phase gradient of ``E_in``.

    For a field ``E_in = A(x,y) * exp(i*phi(x,y))``, the local wavevector
    at each pixel is ``k_local = grad(phi)``.  A ray launched from
    that pixel should carry direction cosines ``L = k_x / k0``,
    ``M = k_y / k0``, where ``k0 = 2*pi/wavelength``.

    We compute ``grad(phi)`` via the conjugate-product trick
    ``angle(E_shifted * conj(E))`` so the wrap-to-(-pi, pi] happens
    once per pair without a separate unwrap pass.  Low-amplitude
    pixels (below 0.1 % of peak) and NaN/inf phase are returned as
    zero tilt.  The final cosines are clipped to ``|L|, |M| <=
    max_sin`` for numerical safety.

    Why this function has to be careful
    -----------------------------------
    A single-mode field has ONE well-defined phase gradient at every
    pixel (plane wave, smooth Gaussian, MLA-tilted beamlet).  A
    multi-mode field -- a superposition of several plane-wave
    components like a post-DOE diffraction pattern -- has NO
    well-defined local direction: neighbouring pixels can report wildly
    different ``np.angle(E_shift * conj(E))`` values because the sum of
    components interferes coherently.  Feeding those aliased per-pixel
    directions straight into the entrance->exit spline in
    :func:`apply_real_lens_traced` produces a chaotic map that Newton
    cannot invert, resulting in an all-NaN OPL and a zero output field.

    Fix: **amplitude-weighted Gaussian smoothing of the tilt field**
    before it's returned.  The smoothing is a low-pass on the local
    wavevector, with the physical interpretation that a ray launched
    from an entrance pixel carries the *mean* direction of the wave
    components within a few-wavelength neighbourhood, rather than the
    single-pixel aliased fringe phase.

    *   Single-mode fields: the true tilt is a slowly-varying function
        of position, so a Gaussian of sigma a few pixels leaves it
        essentially unchanged.  MLA-modulated fields keep their
        per-beamlet tilts.
    *   Multi-mode fields: the tilt oscillates pixel-to-pixel with
        mean zero (for a balanced set of orders).  Gaussian smoothing
        pulls the tilt toward that zero mean, naturally degenerating
        to a classical collimated launch for post-DOE inputs.
    *   Amplitude weighting ensures low-amplitude pixels (between
        DOE orders, outside MLA beamlets, etc.) don't drag the
        smoothed tilt toward the noisy phase readings those pixels
        contribute.

    No threshold to tune, no global "reject" decision -- the smoothing
    is the universal fix.

    Parameters
    ----------
    smooth_sigma_px : float, default 4.0
        Gaussian smoothing radius (pixels) applied to the tilt field.
        Set to 0 to disable smoothing (pre-smoothing behaviour, NOT
        recommended for multi-mode inputs).  A few pixels is enough
        to suppress single-pixel aliasing while preserving tilts that
        vary on the scale of typical beam features (MLA beamlet
        diameters, Gaussian waists, etc.).
    multimode_diagnostic : dict, optional
        If provided, gets populated with tilt-field statistics before
        and after smoothing (``raw_rms_L``, ``smoothed_rms_L``,
        ``raw_rms_M``, ``smoothed_rms_M``, ``smoothing_ratio``).
        Useful for callers that want to log or verify the smoothing
        is doing what's expected.
    """
    k0 = 2.0 * np.pi / wavelength
    N_y, N_x = E_in.shape

    # Phase gradient: d(phi)/dx ~ angle(E[:, 1:] * conj(E[:, :-1])) / dx
    # Use np.roll so shapes match; the rolled-into-the-boundary pixels
    # get low weights after the amplitude mask.
    E_shift_x = np.roll(E_in, -1, axis=1)
    E_shift_y = np.roll(E_in, -1, axis=0)
    grad_phi_x = np.angle(E_shift_x * np.conj(E_in)) / dx
    grad_phi_y = np.angle(E_shift_y * np.conj(E_in)) / dx

    L_grid = grad_phi_x / k0
    M_grid = grad_phi_y / k0

    # Zero-out noise-floor pixels and boundary wrap
    amp = np.abs(E_in)
    amp_thresh = 1e-3 * float(amp.max()) if amp.size else 0.0
    mask = (amp > amp_thresh) & np.isfinite(L_grid) & np.isfinite(M_grid)
    L_grid = np.where(mask, L_grid, 0.0)
    M_grid = np.where(mask, M_grid, 0.0)

    # Statistics before smoothing -- for diagnostics and as the "raw"
    # baseline the smoothing is operating on.
    raw_rms_L = float(np.sqrt(np.mean(L_grid[mask] ** 2))) if mask.any() else 0.0
    raw_rms_M = float(np.sqrt(np.mean(M_grid[mask] ** 2))) if mask.any() else 0.0

    # ---- Amplitude-weighted Gaussian smoothing ---------------------
    # Low-pass the tilt field with an intensity-weighted kernel:
    #
    #     L_smooth = blur(|E|^2 * L) / blur(|E|^2)
    #     M_smooth = blur(|E|^2 * M) / blur(|E|^2)
    #
    # This averages neighbouring pixels' tilts using their amplitude
    # squared (intensity) as weights.  On a smooth single-mode field
    # this leaves L and M essentially unchanged (neighbours already
    # agree).  On a multi-mode superposition with pixel-to-pixel
    # aliased phase gradients, the oscillations average out and
    # amplitude-weighting discounts the low-amplitude interference
    # nulls where the phase is noisiest.  Low-amplitude regions
    # (between beamlets, outside the main field) where the raw
    # gradient is unreliable naturally decay to zero because both
    # numerator and denominator weight them out.
    if smooth_sigma_px > 0:
        from scipy.ndimage import gaussian_filter
        I = (amp * amp).astype(np.float64)
        sigma = float(smooth_sigma_px)
        num_L = gaussian_filter(I * L_grid, sigma=sigma, mode='nearest')
        num_M = gaussian_filter(I * M_grid, sigma=sigma, mode='nearest')
        den = gaussian_filter(I, sigma=sigma, mode='nearest')
        # Guard against division by zero far from the field support
        safe = den > (den.max() * 1e-6)
        L_grid = np.where(safe, num_L / np.where(safe, den, 1.0), 0.0)
        M_grid = np.where(safe, num_M / np.where(safe, den, 1.0), 0.0)

    smoothed_rms_L = float(np.sqrt(np.mean(L_grid[mask] ** 2))) if mask.any() else 0.0
    smoothed_rms_M = float(np.sqrt(np.mean(M_grid[mask] ** 2))) if mask.any() else 0.0
    if multimode_diagnostic is not None:
        multimode_diagnostic['raw_rms_L'] = raw_rms_L
        multimode_diagnostic['raw_rms_M'] = raw_rms_M
        multimode_diagnostic['smoothed_rms_L'] = smoothed_rms_L
        multimode_diagnostic['smoothed_rms_M'] = smoothed_rms_M
        # Ratio < 1 means smoothing reduced the tilt magnitude (i.e.
        # noise was averaged out); ratio ~= 1 means smoothing was a
        # no-op (field was already smooth).
        raw_mag = np.hypot(raw_rms_L, raw_rms_M)
        smoothed_mag = np.hypot(smoothed_rms_L, smoothed_rms_M)
        multimode_diagnostic['smoothing_ratio'] = (
            smoothed_mag / raw_mag if raw_mag > 0 else 1.0)

    # Clip to physical range -- rays with |sin(theta)| > max_sin are
    # unphysical for most lens designs and will overwhelm the Newton
    # fit domain.  After smoothing this clip typically never triggers,
    # but we keep it as a defence against pathological inputs.
    np.clip(L_grid, -max_sin, max_sin, out=L_grid)
    np.clip(M_grid, -max_sin, max_sin, out=M_grid)

    # Interpolate to launch positions (physical -> pixel index,
    # bilinear sample).  Launch positions outside the E_in grid
    # (|x| > N*dx/2) fall back to zero tilt (edge -- no information).
    from scipy.ndimage import map_coordinates
    pix_x = entrance_x / dx + N_x / 2.0
    pix_y = entrance_y / dx + N_y / 2.0
    coords = np.vstack([pix_y.ravel(), pix_x.ravel()])
    L = map_coordinates(L_grid, coords, order=1,
                        mode='constant', cval=0.0).reshape(entrance_x.shape)
    M = map_coordinates(M_grid, coords, order=1,
                        mode='constant', cval=0.0).reshape(entrance_x.shape)
    return L, M


def _reverse_prescription(prescription):
    """Build a prescription describing the same lens traversed in the
    backward direction.

    Used by the experimental backward-trace OPL inversion in
    :func:`apply_real_lens_traced`.  Reversing amounts to:

    *   Swap surface order.
    *   Negate every radius of curvature (curvature direction flips
        when viewed from the opposite side).  Conic constants and
        even-power aspheric coefficients are invariant under this
        reflection.
    *   Swap ``glass_before`` and ``glass_after`` on each surface.
    *   Reverse the thickness list (the gap AFTER surface i in the
        forward prescription is the gap BEFORE surface (N-1-i) in
        the reversed one, which is the same list read right-to-left).
    """
    surfaces = prescription['surfaces']
    thicknesses = prescription.get('thicknesses', [])
    rev_surfaces = []
    for s in reversed(surfaces):
        rs = dict(s)
        rs['radius'] = -rs['radius']
        if rs.get('radius_y') is not None:
            rs['radius_y'] = -rs['radius_y']
        rs['glass_before'], rs['glass_after'] = (
            rs['glass_after'], rs['glass_before'])
        rev_surfaces.append(rs)
    rev = {
        'surfaces': rev_surfaces,
        'thicknesses': list(reversed(thicknesses)),
    }
    if 'aperture_diameter' in prescription:
        rev['aperture_diameter'] = prescription['aperture_diameter']
    return rev


def _opl_by_backward_trace(E_analytic, lens_prescription, wavelength, dx,
                           N_grid, ray_subsample,
                           tilt_smooth_sigma_px=4.0):
    """Alternative to the Newton-based forward-map inversion in
    :func:`apply_real_lens_traced`.

    **Validation** (2026-04-18):

    *   Single-ray forward-vs-backward OPL on a plano-convex singlet:
        **< 1 pm** (machine-precision agreement) when the exit-vertex
        correction is applied to both ends.
    *   End-to-end ``apply_real_lens_traced`` OPD RMS vs the Newton
        path: **~35-40 nm** on singlets at N=512.  The residual is
        not a bug in the reversal; it comes from using the
        finite-difference phase gradient of ``E_analytic`` as the
        backward-launch direction estimate (Newton uses the
        forward-trace's exact entrance-plane direction).  For
        design-verification work at lambda/10 tolerance this is deep
        in the margin; for sub-nm precision use Newton.

    Measured speed at N=512: ~1.7x faster than Newton on a singlet.
    Scales better to large N because the work is ``O(N^2)`` rather
    than ``O(N^2 * newton_iters)``.

    Algorithm in brief:

    Instead of ray-tracing the entrance grid forward and then
    Newton-inverting the spline of that map to find each exit pixel's
    entrance ray, we trace rays BACKWARD from a coarse subsample of
    the exit grid through the lens to the entrance, accumulating
    OPL along the way.  Fermat's principle makes OPL path-reversible,
    so the backward-trace OPL is numerically the same as the
    forward-trace OPL up to a sign convention.

    The exit-plane ray directions are derived from the local phase
    gradient of ``E_analytic`` (same mechanism as the input-aware
    forward launch, just applied at the exit).  The
    amplitude-weighted Gaussian smoothing keeps this robust on
    multi-mode inputs.

    Advantages over the Newton path (when it works):
        *   No spline fit, no Newton iteration.  The entire
            computation is a single forward pass of ``trace()``
            through a reversed prescription plus interpolation
            of the OPL map to the wave grid.
        *   Embarrassingly parallel in the trace itself (no
            dependencies between rays).

    Disadvantages / caveats:
        *   Accuracy depends on how well the exit-plane direction
            is extracted from ``E_analytic``.  Near a focus the
            true direction varies rapidly and the smoothed
            gradient is less representative; Newton handles this
            via the spline without needing a direction estimate.
        *   Only tested on singlet and doublet geometries so far;
            compound systems with intermediate foci may behave
            unexpectedly.  **Labelled experimental.**
    """
    from .raytrace import surfaces_from_prescription, trace, _make_bundle

    N = int(N_grid)
    sub = max(1, int(ray_subsample))
    # Coarse exit-plane sampling (same stride pattern as the Newton
    # path's ``X[::sub, ::sub]`` slice so the final interpolation
    # grids line up identically).
    idx_c = np.arange(0, N, sub)
    N_c = idx_c.size
    x_c = (idx_c - N / 2.0) * dx
    Xc, Yc = np.meshgrid(x_c, x_c)

    # Extract exit-plane direction cosines from the phase gradient
    # of E_analytic, smoothed per the 3.1.3 multi-mode fix.
    L_out, M_out = _sample_local_tilts(
        E_analytic, wavelength, dx, Xc, Yc,
        smooth_sigma_px=tilt_smooth_sigma_px)

    # Build the reversed prescription + its surface list.  Note:
    # surfaces_from_prescription uses the per-element semi-diameter
    # plus the prescription-level aperture_diameter for vignetting;
    # both carry through to the reverse automatically.
    rev_rx = _reverse_prescription(lens_prescription)
    rev_surfaces = surfaces_from_prescription(rev_rx)

    # Rays start at the exit vertex plane (z=0) with direction
    # cosines (-L_out, -M_out, +sqrt(1-L^2-M^2)).  The sign flip on
    # (L, M) accounts for tracing in the reversed-axis frame:
    # "forward" here == backward in the original frame.  _make_bundle
    # computes N = +sqrt(1-L^2-M^2) which is the correct "forward"
    # direction in the reversed frame.
    rays = _make_bundle(
        x=Xc.ravel(), y=Yc.ravel(),
        L=-L_out.ravel(), M=-M_out.ravel(),
        wavelength=wavelength,
    )
    result = trace(rays, rev_surfaces, wavelength)
    final = result.image_rays

    # ---- Exit-vertex correction on the backward trace ----
    # trace() leaves rays at z = sag(last_surface_in_reversed_frame)
    # = sag of original S1 (the original entrance-side vertex) in
    # the reversed frame.  Without propagating each ray to z=0 of
    # this reversed-frame last surface (the original entrance
    # vertex plane), we under-count the OPL by the
    # vertex-to-sag leg in the final medium -- exactly the same
    # correction the forward path applies in apply_real_lens_traced
    # at lenses.py:1548-1556.  For on-axis rays this is zero; for
    # marginal rays on a strong-curvature lens it's tens of nm to
    # hundreds of nm.  Missing this is what made the first draft
    # of this function disagree with Newton by ~343 nm RMS.
    rev_surfaces_list = rev_surfaces
    n_exit_backward = get_glass_index(
        rev_surfaces_list[-1].glass_after, wavelength)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_to_vertex = np.where(
            final.alive & (np.abs(final.N) > 1e-30),
            -final.z / final.N, 0.0)
    final.opd = final.opd + n_exit_backward * t_to_vertex
    # (We don't actually need to update x/y/z since we only
    # consume final.opd downstream, but keep it consistent.)
    final.x = final.x + final.L * t_to_vertex
    final.y = final.y + final.M * t_to_vertex
    final.z = np.zeros_like(final.z)

    # OPL: set NaN for dead rays (TIR / vignetted during the
    # reverse trace) so downstream NaN-propagation matches the
    # Newton path's treatment of out-of-domain points.
    opl_flat = np.where(final.alive, final.opd, np.nan)
    opl_coarse = opl_flat.reshape(Xc.shape)

    # Reference to on-axis so the returned OPL has the same origin
    # as the Newton path.  (Forward Newton does this at the spline
    # fit step via ``opl_grid = opl_grid - opl_grid[i_axis, i_axis]``.)
    i_c = N_c // 2
    ref = opl_coarse[i_c, i_c]
    if np.isfinite(ref):
        opl_coarse = opl_coarse - ref

    # Interpolate coarse OPL to the full wave grid, with the same
    # mode='nearest' + NaN-majority masking the Newton path uses.
    from scipy.ndimage import map_coordinates
    ii, jj = np.indices((N, N), dtype=np.float64)
    coords = np.array([ii * N_c / N, jj * N_c / N])
    opl_map = map_coordinates(
        np.where(np.isnan(opl_coarse), 0.0, opl_coarse),
        coords, order=1, mode='nearest')
    nan_coarse = np.isnan(opl_coarse).astype(np.float64)
    nan_full = map_coordinates(
        nan_coarse, coords, order=1, mode='nearest')
    opl_map = np.where(nan_full > 0.5, np.nan, opl_map)
    return opl_map


def apply_real_lens_traced(E_in, lens_prescription, wavelength, dx,
                           bandlimit=True, ray_subsample=8,
                           n_workers=None, progress=None,
                           min_coarse_samples_per_aperture=32,
                           on_undersample='error',
                           preserve_input_phase=True,
                           tilt_aware_rays=False,
                           parallel_amp=True,
                           parallel_amp_min_free_gb=48.0,
                           newton_amp_mask_rel=1e-4,
                           newton_mask_dilate_coarse_px=2,
                           inversion_method='newton',
                           fast_analytic_phase=False,
                           newton_fit='polynomial',
                           newton_poly_order=6,
                           use_gpu=False,
                           amp_use_gpu=False,
                           wave_propagator='asm'):
    """Wave + per-pixel ray-traced phase variant of :func:`apply_real_lens`.

    For each pixel of the simulation grid, a geometric ray is launched
    from the entrance plane straight through the prescription using the
    sequential ray tracer in :mod:`lumenairy.raytrace`.  The
    accumulated optical path length (OPL) per pixel is used as the
    exit-plane phase, while the wave's *amplitude* envelope (vignetting,
    diffraction, edge effects) comes from a single ASM propagation of
    the entrance aperture to the exit-vertex plane.

    This eliminates the uniform-glass-slab approximation that limits
    the closed-form thin-element model on cemented doublets and other
    multi-surface curved-interface systems: each pixel sees the
    geometrically-correct glass path for its (x,y) position.  In
    practice the OPD agrees with the geometric ray trace to the
    sampling limit of the grid, at the cost of one ray trace per
    pixel (~3-10x slowdown relative to the analytic phase-screen
    model).

    Critical sampling rule
    ----------------------
    Extracting OPD from a converging wavefront requires

        dx <= lambda * f / aperture

    where ``f`` is the back focal length and ``aperture`` is the pupil
    diameter.  Coarser sampling makes ``np.unwrap`` lose cycles at the
    pupil edge, giving catastrophically wrong OPD values there.  Run
    :func:`lumenairy.analysis.check_opd_sampling` before a
    large ``apply_real_lens_traced`` call to verify.  If a coarser
    grid is required, use :func:`apply_real_lens` (with
    ``seidel_correction=True`` for doublets) instead.

    Limitations
    -----------
    * Assumes the input field is approximately a collimated plane wave
      (each pixel ray launched parallel to z).  For converging or
      tilted input, fall back to :func:`apply_real_lens`.
    * Replaces the wave's exit phase with the geometric OPL; this
      gives correct OPD by construction but bypasses any wave-physics
      phase content that the ASM would have introduced (negligible for
      typical lens systems but worth noting).
    * Fresnel transmission and absorption are NOT applied here -- if
      you need them, run both this function and
      :func:`apply_real_lens` and combine.

    Parameters
    ----------
    E_in : ndarray, complex, shape (N, N)
    lens_prescription : dict
        Same format as :func:`apply_real_lens`.
    wavelength : float
    dx : float
    bandlimit : bool, default True
        Passed to the (single) ASM propagation used for amplitude
        evolution.
    ray_subsample : int, default 1
        Compute the ray-trace OPL on every ``ray_subsample``-th pixel
        and bilinearly interpolate to the full grid.  OPL is a very
        smooth function of pupil position, so ``ray_subsample=4``
        typically loses < 1 nm of fidelity while cutting cost ~16x.
        Recommended for production use on large grids.
    min_coarse_samples_per_aperture : int, default 32
        Guardrail against undersampled Newton inversion.  After
        ``ray_subsample`` is applied, the coarse output grid must have
        at least this many samples spanning the lens aperture (or
        ``launch_radius`` if no explicit aperture is set), otherwise
        the cubic-spline interpolation of the wavefront will alias and
        the result will be wrong.

        Empirical scaling on a singlet at lambda = 1.31 um:

        ====================  ==================
        coarse-samples / ap   typical RMS phase
        ====================  ==================
        64                    ~20 nm
        32 (default safe)     ~85 nm
        16                    ~350 nm  (unusable)
        ====================  ==================

        Pass ``0`` to disable the check entirely.
    on_undersample : ``'error'`` (default) / ``'warn'`` / ``'silent'``
        What to do when the coarse-sample count falls below
        ``min_coarse_samples_per_aperture``.  ``'error'`` raises
        ``ValueError`` with the safe ``ray_subsample`` value computed
        for the current grid; ``'warn'`` logs via the ``warnings``
        module and continues; ``'silent'`` is the explicit "I know
        what I'm doing" escape hatch.
    n_workers : int, optional
        Number of worker *processes* for the Newton-inversion step.
        Defaults to
        :func:`lumenairy._backends.available_cpus` -- the
        affinity-aware count of CPUs this process can actually use
        (respects cgroup limits, ``taskset`` masks, Python 3.13+
        ``process_cpu_count``, Windows process affinity).  Pass 1 to
        force the in-process serial path (useful for reproducible
        timings or when called from a parent pool that already
        saturates the machine).
    tilt_aware_rays : bool, default False
        If True, each ray's initial direction ``(L, M)`` is derived
        from the local phase gradient of ``E_in`` at the entrance
        position (the "Tier 1 input-aware ray launch" added in 3.1.2).
        If False (the default), collimated rays are launched
        (L = M = 0 everywhere) and the plane-wave lens-OPL reference
        is used.

        **Why the default flipped from True to False in 3.1.3:**  When
        ``preserve_input_phase=True`` (also the default), the exit
        field is assembled as

            E_out = E_analytic * exp(i * delta_phase)
            delta_phase = k0 * opl_traced - phase_analytic_lens

        where ``phase_analytic_lens`` is the phase produced by running
        :func:`apply_real_lens` on a unit PLANE WAVE -- i.e. a
        plane-wave reference.  For ``delta_phase`` to be a
        mathematically clean "ray-traced minus analytic" correction,
        ``opl_traced`` must use the same reference: a plane-wave
        entrance launch.  With ``tilt_aware_rays=True``, ``opl_traced``
        instead mixes the lens-model correction with per-pixel
        tilt-induced phase shifts that the plane-wave ``phase_analytic_lens``
        does not contain.  The resulting ``delta_phase`` is only
        approximately right for small/uniform input tilts, and breaks
        materially on multi-mode inputs (post-DOE fields, strongly
        off-axis compound beams) where the per-pixel tilts vary
        significantly across the pupil.

        The 3.1.4 default ``tilt_aware_rays=False`` restores the
        reference-consistent plane-wave launch that pre-3.1.2 releases
        used, so ``delta_phase`` remains well-defined for any input the
        wave model can represent.  If you have a specifically small,
        uniform input tilt and want the per-ray OPL variation (e.g.
        rigorous off-axis lens characterisation with a single tilted
        input), pass ``tilt_aware_rays=True`` explicitly and validate
        against the default on your specific case.

        When this flag is True, tilts are clipped to
        ``|sin(theta)| <= 0.5`` (~30 deg) for numerical safety and
        amplitude-weighted-Gaussian-smoothed (``smooth_sigma_px=4``
        by default inside :func:`_sample_local_tilts`) to tame
        multi-mode aliasing; neither applies when the flag is False
        (the default).

    preserve_input_phase : bool, default True
        If True, the input field's phase structure (source tilts,
        MLA / DOE phase modulation, off-axis wavefronts, etc.) is
        preserved through the lens and combined with the ray-traced
        OPL correction.  This is the physically-correct behaviour
        and matches what :func:`apply_real_lens` does (with the added
        benefit of corrected geometric OPL).

        If False (legacy behaviour prior to v3.1.2), the output is
        ``|E_analytic| * exp(i*k0*OPL_traced)`` -- the input-field
        phase is discarded entirely and only the lens's ray-traced
        OPL is retained.  Use this mode when you specifically want
        the lens-only OPD response on a synthetic plane wave;
        otherwise keep the default.

        Cost: ``preserve_input_phase=True`` runs the analytic
        apply_real_lens *twice* (once for the input field, once for
        a unit plane-wave reference so we can subtract the analytic
        lens phase before adding the traced one).  This roughly
        doubles the ~40 % amplitude-leg budget.  At large N the
        total overhead is ~20 %.

        Implementation note: the work is dispatched via
        ``concurrent.futures.ProcessPoolExecutor`` rather than threads
        because SciPy's ``RectBivariateSpline.ev`` does not release
        the GIL in current versions, so threading delivers no
        speedup.  Each worker rebuilds the splines locally from
        their knot data (cheap), avoiding the pickle cost of the
        spline objects themselves.  Sequential fallback is used when
        the coarse grid is below ~200 k pixels (pool startup cost
        dominates) or when pool spawn fails.  Measured speedup on
        large grids: ~8x on 16 cores.

    Returns
    -------
    E_out : ndarray, complex, shape (N, N)
        Field at the exit-vertex plane of the last surface.
    """
    # Local import to avoid a circular dep at module load time
    from .raytrace import (
        surfaces_from_prescription, trace, _make_bundle,
    )
    from .propagation import angular_spectrum_propagate
    from .progress import call_progress, ProgressScaler

    call_progress(progress, 'real_lens_traced', 0.0, 'initialising')

    # Pre-flight grid vs prescription-aperture check.
    try:
        _warn_if_aperture_exceeds_grid(
            lens_prescription, int(np.shape(E_in)[0]), dx,
            source='apply_real_lens_traced')
    except Exception:
        pass

    Ny, Nx = E_in.shape
    if Ny != Nx:
        raise ValueError("apply_real_lens_traced requires a square grid")
    N = Nx

    aperture = lens_prescription.get('aperture_diameter')
    thicknesses = lens_prescription['thicknesses']
    total_thickness = float(sum(thicknesses))

    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)

    # ----- Step 1: amplitude envelope from the ANALYTIC lens model -----
    #
    # WHY WE CALL apply_real_lens HERE (the "double call"):
    #
    # apply_real_lens_traced is a HYBRID method.  It combines:
    #   (a) AMPLITUDE from wave optics — diffraction, vignetting, and
    #       the physically correct in-glass beam evolution (Fresnel
    #       effects at curved surfaces, edge ripples, aperture clipping)
    #   (b) PHASE from geometric ray tracing — the exact OPL through
    #       every curved glass/air interface, per pixel, via vector
    #       Snell's law at each surface
    #
    # The thin-element model's accuracy limitation is in its PHASE
    # (it approximates curved surfaces as phase screens at a single
    # z-plane), NOT in its amplitude (ASM through a uniform glass slab
    # handles diffraction correctly).  So we:
    #   1. Run apply_real_lens to get the full exit-plane field
    #   2. Keep only |E| (amplitude) — the wave-optics part
    #   3. Replace the phase with the geometrically exact ray-traced
    #      OPL map computed in Step 2 below
    #
    # This gives sub-nanometre OPD agreement with the geometric ray
    # trace (the "truth") while retaining physically correct
    # diffraction effects that pure ray tracing cannot capture.
    #
    # An earlier version used a simple air-ASM for the amplitude,
    # which produced a ~3.5 mm focus offset because air propagation
    # ≠ glass propagation (different wavenumber k = n·k0).  Using
    # apply_real_lens for the amplitude solves this because it
    # propagates through the correct glass/air refractive index
    # sequence.
    # Allocate 40% of the budget to the amplitude (which runs a full
    # apply_real_lens with its own per-surface cost), 50% to the ray
    # trace + Newton inversion, and 10% to the final field assembly.
    # ---------- Parallelism decision for amp and amp(pw) --------------
    # The two apply_real_lens calls (``amp`` on the real input, and
    # ``amp(pw)`` on a unit plane wave to recover the analytic lens
    # OPD) are data-independent and can run concurrently.  We dispatch
    # them on a ThreadPoolExecutor so the non-FFT work (sag, phase
    # screens, numexpr-fused multiplies, glass-interval setup)
    # overlaps.  The pyFFTW plan cache in ``propagation._fft2`` /
    # ``_ifft2`` holds a per-plan lock so the actual FFT execution
    # serialises safely on the shared aligned buffer; overlap is
    # therefore bounded by the FFT share of each call (~45-50 %) but
    # still gives ~40 % wall-time savings on the combined amp step.
    #
    # Memory cost of parallelism: two E fields and two sets of lens
    # intermediates alive simultaneously (~2x the peak of a single
    # call).  The ``parallel_amp_min_free_gb`` guard drops back to
    # sequential execution when available RAM is too tight for this
    # doubled working set -- tuned for the N=32768 complex128 case,
    # where the single-call transient peak is ~25 GB and doubling
    # brings it to ~50 GB.
    _use_parallel_amp = (preserve_input_phase and parallel_amp)
    if _use_parallel_amp:
        try:
            import psutil as _psutil
            _free_gb = _psutil.virtual_memory().available / 1e9
            if _free_gb < parallel_amp_min_free_gb:
                _use_parallel_amp = False
        except Exception:
            # psutil missing -- leave parallel_amp enabled but the
            # user can still force off via the kwarg.
            pass

    amp_cb = ProgressScaler(progress, 'real_lens_traced',
                            lo=0.0, hi=0.50 if _use_parallel_amp else 0.40)

    if _use_parallel_amp:
        # Parallel path: run amp and amp(pw) concurrently.  Only the
        # amp call reports progress (0-50%); amp(pw) runs silently to
        # avoid interleaved status lines from two threads.  The ones-
        # like plane wave is materialised outside the thread so the
        # 17 GB allocation happens once, synchronously, with clear
        # OOM semantics.
        from concurrent.futures import ThreadPoolExecutor

        def _amp_call():
            return apply_real_lens(
                E_in, lens_prescription, wavelength, dx,
                bandlimit=bandlimit, use_gpu=amp_use_gpu,
                wave_propagator=wave_propagator,
                progress=lambda stage, frac, msg='':
                    amp_cb(frac, f'amp: {msg}'))

        if fast_analytic_phase and preserve_input_phase:
            # Skip the full amp(pw) ASM pass; compute the geometric
            # lens phase analytically from per-surface sag.
            E_analytic = _amp_call()
            # np.abs and np.angle work on cupy arrays via __array_function__
            # in recent numpy; but to be explicit, use xp.abs/xp.angle via
            # the module selector below.
            _xp = cp if _is_cupy_array(E_analytic) else np
            amp = _xp.abs(E_analytic)
            phase_analytic_lens = _geometric_lens_phase(
                lens_prescription, wavelength, dx, E_in.shape[0])
            if _xp is cp:
                phase_analytic_lens = cp.asarray(phase_analytic_lens)
        else:
            ones_input = np.ones_like(E_in)

            def _amp_pw_call():
                return apply_real_lens(
                    ones_input, lens_prescription, wavelength, dx,
                    bandlimit=bandlimit, use_gpu=amp_use_gpu,
                    wave_propagator=wave_propagator, progress=None)

            with ThreadPoolExecutor(max_workers=2,
                                    thread_name_prefix='rlt_amp') as _tp:
                fut_amp = _tp.submit(_amp_call)
                fut_pw = _tp.submit(_amp_pw_call)
                E_analytic = fut_amp.result()
                E_analytic_pw = fut_pw.result()
            del ones_input
            _xp = cp if _is_cupy_array(E_analytic) else np
            amp = _xp.abs(E_analytic)
            phase_analytic_lens = _xp.angle(E_analytic_pw)
            del E_analytic_pw  # free ~17 GB at N=32768 before Newton starts
    else:
        # Sequential fallback (preserve_input_phase=False or RAM tight).
        E_analytic = apply_real_lens(
            E_in, lens_prescription, wavelength, dx, bandlimit=bandlimit,
            use_gpu=amp_use_gpu, wave_propagator=wave_propagator,
            progress=lambda stage, frac, msg='': amp_cb(frac, f'amp: {msg}'))
        _xp = cp if _is_cupy_array(E_analytic) else np
        amp = _xp.abs(E_analytic)
        # When preserving input phase (the physically-correct default),
        # we also need to know the *analytic model's lens-only phase* so
        # we can subtract it out before adding the ray-traced OPL back in.
        # We extract it by running apply_real_lens on a unit plane wave --
        # the result's phase is exactly the analytic lens's OPL
        # (plus small wave-propagation-through-glass effects) applied to
        # a flat input.
        if preserve_input_phase:
            if fast_analytic_phase:
                # Analytic geometric phase: per-surface sag phase
                # screens summed locally, no ASM through glass.  On
                # Design 51 lenses this introduces at most ~10 nm OPL
                # error (L4, F/6.8 doublet) and essentially none on
                # slower singlets -- below the numerical noise floor
                # of the rest of the pipeline.
                phase_analytic_lens = _geometric_lens_phase(
                    lens_prescription, wavelength, dx, E_in.shape[0])
                if _xp is cp:
                    phase_analytic_lens = cp.asarray(phase_analytic_lens)
            else:
                analytic_pw_cb = ProgressScaler(progress, 'real_lens_traced',
                                                 lo=0.40, hi=0.50)
                E_analytic_pw = apply_real_lens(
                    np.ones_like(E_in), lens_prescription, wavelength, dx,
                    bandlimit=bandlimit, use_gpu=amp_use_gpu,
                    wave_propagator=wave_propagator,
                    progress=lambda stage, frac, msg='':
                        analytic_pw_cb(frac, f'amp(pw): {msg}'))
                phase_analytic_lens = _xp.angle(E_analytic_pw)
                del E_analytic_pw
        else:
            phase_analytic_lens = None
    # When amp_use_gpu=True the amp pipeline returns CuPy arrays.  The
    # rest of apply_real_lens_traced (ray trace, Newton, final E_out
    # assembly) is CPU-only, so pull the amp outputs back to the host
    # here rather than xp-ifying the final-assembly section.
    if _is_cupy_array(E_analytic):
        E_analytic = cp.asnumpy(E_analytic)
    if _is_cupy_array(amp):
        amp = cp.asnumpy(amp)
    if phase_analytic_lens is not None and _is_cupy_array(phase_analytic_lens):
        phase_analytic_lens = cp.asnumpy(phase_analytic_lens)
    call_progress(progress, 'real_lens_traced', 0.40,
                  'ray-tracing exit pupil')

    # ----- Step 2: ray-traced OPL per (subsampled) pixel ---------------
    # Launch a dense grid of rays from the entrance pupil; each ray
    # bends through the lens and lands at a *different* (x_out, y_out)
    # at the exit plane.  We need OPL associated with the exit
    # position (matching the wave's exit-plane grid), not the entrance
    # position, so we scatter-interpolate ``opl(x_out, y_out)`` onto
    # the wave grid.
    #
    # IMPORTANT: build surfaces WITHOUT the prescription aperture so
    # rays launched slightly beyond the entrance pupil are not
    # vignetted -- they may end up landing *inside* the wave grid
    # after refraction-induced inward shift.  But we DO restrict the
    # entrance launch positions to a modest over-margin around the
    # actual aperture so ultra-marginal rays (at huge angles of
    # incidence on the first surface) don't contaminate the OPL
    # function with non-paraxial branches.  The wave amplitude mask is
    # applied separately and zeros any spurious phase outside the
    # physical aperture anyway.
    pres_no_ap = dict(lens_prescription)
    pres_no_ap.pop('aperture_diameter', None)
    surfaces = surfaces_from_prescription(pres_no_ap)

    sub = max(1, int(ray_subsample))
    # Pick the launch radius: aperture (if specified) plus a 50 %
    # over-margin so that the entrance-grid sampling covers all wave-
    # grid exit positions even for fast lenses (rays bend inward so
    # exit positions are closer to axis than entrance).
    if aperture is not None:
        launch_radius = 0.5 * aperture * 1.50
    else:
        launch_radius = 0.5 * N * dx

    # ----- Subsampling guardrail --------------------------------------
    # The Newton-inversion step builds a cubic-spline interpolant of the
    # entrance->exit map on a coarse grid and uses bilinear interp to
    # back-fill the full grid.  If the coarse grid is too sparse
    # relative to the lens aperture the interpolant aliases and the
    # whole exit-pupil OPD is wrong (RMS phase err blows up roughly
    # as (samples_per_aperture)^-2 from the benchmark sweep -- 32
    # samples gives ~85 nm at lambda = 1.31 um, 16 samples gives ~350
    # nm and is unusable).
    if min_coarse_samples_per_aperture and aperture is not None:
        ap_diameter = float(aperture)
        coarse_dx = dx * sub
        n_coarse_across = ap_diameter / coarse_dx if coarse_dx > 0 else 0
        if n_coarse_across < min_coarse_samples_per_aperture:
            # Compute the largest sub that *would* be safe so the
            # error message gives the user an actionable number.
            safe_sub = max(1, int(np.floor(
                ap_diameter / (dx * min_coarse_samples_per_aperture))))
            msg = (
                f'apply_real_lens_traced: ray_subsample={ray_subsample} '
                f'gives only {n_coarse_across:.1f} coarse samples across '
                f'the {ap_diameter*1e3:.2f}-mm aperture (threshold '
                f'{min_coarse_samples_per_aperture}).  At this density '
                f'the spline interpolation of the wavefront will alias '
                f'and the OPD will be wrong by ~lambda/4 or more.  '
                f'Drop to ray_subsample <= {safe_sub} (or pass '
                f'min_coarse_samples_per_aperture=0 to override).'
            )
            if on_undersample == 'error':
                raise ValueError(msg)
            elif on_undersample == 'warn':
                import warnings
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            elif on_undersample != 'silent':
                raise ValueError(
                    f"on_undersample must be 'error', 'warn', or "
                    f"'silent' (got {on_undersample!r})")

    # Number of samples across the launch grid (subsampled).  Keep it
    # at least proportional to the grid resolution so the OPL
    # function is well sampled.
    n_launch = max(8, int(2 * launch_radius / (dx * sub)))
    # Ensure odd so there's a sample on the optical axis (entrance
    # centre) -- makes on-axis piston subtraction exact.
    if n_launch % 2 == 0:
        n_launch += 1
    xs_in = np.linspace(-launch_radius, launch_radius, n_launch)
    # Use indexing='ij' so that after reshaping trace results to
    # (n_launch, n_launch), array[i, j] corresponds to entrance
    # (X = xs_in[i], Y = xs_in[j]) -- matching scipy's
    # RectBivariateSpline(x_knots, y_knots, values) convention where
    # values[i, j] is the value at (x_knots[i], y_knots[j]).  With the
    # default 'xy' indexing the reshape transposes x/y, which makes
    # the spline's Jacobian wrong and Newton converges to bogus
    # points for 2D wave pixels off the symmetry axes.
    Xs_in, Ys_in = np.meshgrid(xs_in, xs_in, indexing='ij')
    h_x = Xs_in.ravel()
    h_y = Ys_in.ravel()
    # Tier 1 input-aware ray launch: derive each ray's direction from
    # the local phase gradient of E_in at its entrance position.  For
    # plane-wave inputs this reduces to L = M = 0 (identical to the
    # classical collimated launch); for structured inputs (MLA
    # modulation, off-axis sources, pre-aberrated wavefronts) the
    # rays correctly start at the angle implied by E_in, giving the
    # lens its actual per-ray OPL instead of a plane-wave-reference
    # OPL map.  See :func:`_sample_local_tilts` for the extraction.
    if tilt_aware_rays:
        L_in, M_in = _sample_local_tilts(E_in, wavelength, dx, Xs_in, Ys_in)
        L_in = L_in.ravel()
        M_in = M_in.ravel()
    else:
        L_in = np.zeros_like(h_x)
        M_in = np.zeros_like(h_x)
    rays = _make_bundle(x=h_x, y=h_y, L=L_in, M=M_in,
                        wavelength=wavelength)
    # output_filter='last': only keep the image-plane bundle.  We do
    # not consume any intermediate per-surface state here, so saving
    # ray_history for all surfaces would allocate ~1 GB per surface
    # at N=32768 and ~250 MB per surface at N=4096 (for an
    # apply_real_lens_traced call at ray_subsample=8) for no benefit.
    result = trace(rays, surfaces, wavelength, output_filter='last')
    final = result.image_rays
    if not final.alive.any():
        raise RuntimeError(
            'apply_real_lens_traced: no rays survived the prescription; '
            'check aperture and clear-aperture settings.')

    # ---- EXIT-VERTEX CORRECTION ----------------------------------------
    # trace() leaves rays at the SAG of the last surface, i.e. at
    # z = sag(h) ≠ 0 for curved exit surfaces.  But the wave model's
    # exit field is defined at the flat exit VERTEX plane (z = 0).
    # Without this correction, the OPL comparison between on-axis
    # (z = 0) and off-axis (z = sag < 0 for concave) rays is made
    # at DIFFERENT z-planes, which introduces a systematic defocus
    # error equal to n_exit * sag(h) — enough to shift the implied
    # focal length by tens of percent for cemented doublets with
    # curved rear surfaces.
    #
    # Fix: propagate each ray from its current sag position to z = 0
    # in the exit medium, accumulating the remaining OPL and updating
    # the exit position to the vertex plane.
    #
    # IMPORTANT: use SIGNED t, not abs(t).  For concave rear surfaces
    # (sag < 0, z < 0) the ray must go forward (t > 0) → add OPL.
    # For convex rear surfaces (sag > 0, z > 0) the ray is AHEAD of
    # the vertex and must go backward (t < 0) → subtract OPL.
    # Using abs() forces the wrong sign for convex exits (e.g.
    # negative meniscus lenses), producing ~45x worse OPD.
    n_exit = get_glass_index(surfaces[-1].glass_after, wavelength)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_to_vertex = np.where(
            final.alive & (np.abs(final.N) > 1e-30),
            -final.z / final.N, 0.0)
    final.opd = final.opd + n_exit * t_to_vertex
    final.x = final.x + final.L * t_to_vertex
    final.y = final.y + final.M * t_to_vertex
    final.z = np.zeros_like(final.z)

    # Reshape final.x, final.y, final.opd onto the regular ENTRANCE
    # grid.  Dead rays would break RectBivariateSpline (which requires
    # strictly regular data); vignetting is rare for normal lenses but
    # we guard against it by filling dead entries with NaN and
    # extrapolating with the spline's natural extrapolation (OK inside
    # the entrance disc of interest).
    x_out_grid = final.x.reshape(n_launch, n_launch)
    y_out_grid = final.y.reshape(n_launch, n_launch)
    opl_grid = final.opd.reshape(n_launch, n_launch)
    if not final.alive.all():
        alive_grid = final.alive.reshape(n_launch, n_launch)
        # Fill NaN into dead entries to make spline fitting fail
        # cleanly (rare path -- vignetted prescriptions)
        x_out_grid = np.where(alive_grid, x_out_grid, np.nan)
        y_out_grid = np.where(alive_grid, y_out_grid, np.nan)
        opl_grid = np.where(alive_grid, opl_grid, np.nan)

    # Reference OPL to on-axis (center of the entrance grid is an
    # exact sample because n_launch is odd)
    i_axis = n_launch // 2
    opl_grid = opl_grid - opl_grid[i_axis, i_axis]

    # ----- OPTION B: RectBivariateSpline + Newton-inversion of the
    # entrance->exit mapping ------------------------------------------
    #
    # Because the rays were launched on a regular (xs_in, xs_in) grid,
    # final.x, final.y, final.opl are regular-grid functions of the
    # entrance position.  We build three 2-D splines:
    #
    #     Sx(xe, ye) = x_out at entrance (xe, ye)
    #     Sy(xe, ye) = y_out at entrance (xe, ye)
    #     So(xe, ye) = OPL   at entrance (xe, ye)
    #
    # For each wave-grid exit pixel (Xw, Yw) we find the entrance
    # (xe, ye) that lands there via Newton iteration on the residual
    # r = (Sx(xe,ye) - Xw, Sy(xe,ye) - Yw) = 0.  Then OPL at that
    # wave pixel = So(xe, ye).
    #
    # Advantages over the previous scatter-to-grid (griddata) path:
    #   * C^2 smooth interpolation (no Delaunay-edge spikes).
    #   * RectBivariateSpline.ev() is implemented in Fortran and DOES
    #     release the GIL, so we CAN multi-thread the Newton loop.
    #   * Works correctly even for fast lenses with caustic-like
    #     behaviour near the exit-pupil edge (the mapping is still
    #     single-valued on the entrance grid; inversion is stable).
    # ---- Validate use_gpu combination ---------------------------------
    _newton_xp = np  # default Newton array backend
    if use_gpu:
        if newton_fit != 'polynomial':
            raise ValueError(
                f"use_gpu=True requires newton_fit='polynomial'; "
                f"got newton_fit={newton_fit!r}.  The spline path uses "
                f"SciPy RectBivariateSpline which has no GPU backend.")
        if not CUPY_AVAILABLE:
            raise ImportError(
                "use_gpu=True requires the 'cupy' package.  Install with "
                "'pip install cupy-cuda12x' (matching your CUDA version) "
                "or set use_gpu=False to stay on the CPU path.")
        _newton_xp = cp

    if newton_fit == 'polynomial':
        # 2-D Chebyshev tensor-product fit -- closed-form evaluation and
        # analytic derivatives, better accuracy than bicubic spline on
        # smooth refractive-lens data.  Same .ev(...) API so the
        # Newton loop below is untouched.
        #
        # When use_gpu=True, build the evaluator on GPU (all arrays
        # pushed to device via cp.asarray).  The Newton loop below
        # auto-detects the evaluator backend and runs on the matching
        # device.
        _xp = _newton_xp
        _xs_xp = _xp.asarray(xs_in)
        _xout_xp = _xp.asarray(x_out_grid)
        _yout_xp = _xp.asarray(y_out_grid)
        _opl_xp = _xp.asarray(opl_grid)
        Sx = _Cheb2DEvaluator(_xs_xp, _xs_xp, _xout_xp,
                               order=newton_poly_order, xp=_xp)
        Sy = _Cheb2DEvaluator(_xs_xp, _xs_xp, _yout_xp,
                               order=newton_poly_order, xp=_xp)
        So = _Cheb2DEvaluator(_xs_xp, _xs_xp, _opl_xp,
                               order=newton_poly_order, xp=_xp)
    elif newton_fit == 'spline':
        try:
            from scipy.interpolate import RectBivariateSpline
        except ImportError:
            raise ImportError(
                'apply_real_lens_traced requires SciPy for spline '
                'interpolation.')
        Sx = RectBivariateSpline(xs_in, xs_in, x_out_grid, kx=3, ky=3)
        Sy = RectBivariateSpline(xs_in, xs_in, y_out_grid, kx=3, ky=3)
        So = RectBivariateSpline(xs_in, xs_in, opl_grid, kx=3, ky=3)
    else:
        raise ValueError(
            f"newton_fit must be 'spline' or 'polynomial', "
            f"got {newton_fit!r}")

    # ---- Paraxial magnification from the already-computed forward
    # trace.  Used as the Newton initial guess: (xe, ye) ~ (Xw, Yw) / M.
    #
    # We read the central finite-difference slope of the forward map:
    #     M_x = [x_out(i_c, i_c+1) - x_out(i_c, i_c-1)] / (2 * d_xs_in)
    #     M_y = [y_out(i_c+1, i_c) - y_out(i_c-1, i_c)] / (2 * d_xs_in)
    # where (i_c, i_c) is the on-axis entrance grid point (exact sample
    # because n_launch is odd).  The central-difference stencil uses a
    # few-micron neighborhood so it captures the paraxial slope without
    # aberration contamination from the pupil edge.
    #
    # This is strictly better than the previous hard-coded 1.10 multiplier:
    # the old heuristic assumed M ~ 0.91 (converging system "shrinks 10%")
    # which is approximately right for singlets at their exit vertex (M ~ 1)
    # but wildly off for compound systems with real imaging magnification
    # (TX Design 36 full-system inversion would have M = 0.25; using 1.10
    # as the guess puts Newton 4x from the answer and costs several extra
    # iterations per pixel).  Zero additional compute -- the grid values
    # are already in memory from the forward trace above.
    i_c = n_launch // 2
    d_xs = float(xs_in[1] - xs_in[0])
    try:
        M_x = (float(x_out_grid[i_c, i_c + 1])
               - float(x_out_grid[i_c, i_c - 1])) / (2.0 * d_xs)
        M_y = (float(y_out_grid[i_c + 1, i_c])
               - float(y_out_grid[i_c - 1, i_c])) / (2.0 * d_xs)
    except (IndexError, ValueError):
        M_x = M_y = 0.91  # fallback to pre-3.1.3 heuristic (1/1.10)
    # Guard against NaNs from dead rays at the center (unlikely -- the
    # axial ray always survives in a well-posed prescription) and
    # against extreme values that would blow up the initial guess.
    if not (np.isfinite(M_x) and np.isfinite(M_y)):
        M_x = M_y = 0.91
    M_x = float(np.clip(abs(M_x), 1e-3, 1e3))
    M_y = float(np.clip(abs(M_y), 1e-3, 1e3))

    # Store spline knot data for potential process-pool pickling.
    # Include the inverse magnification so the process-pool path (which
    # rebuilds splines inside each worker) can seed Newton identically.
    _spline_data = {
        'xs_in': xs_in,
        'x_out_grid': x_out_grid,
        'y_out_grid': y_out_grid,
        'opl_grid': opl_grid,
        'launch_radius': launch_radius,
        'dx': dx,
        'bound': launch_radius * 0.999,
        'inv_M_x': 1.0 / M_x,
        'inv_M_y': 1.0 / M_y,
    }

    # Bound for the clipped Newton update (stay inside fitted domain)
    bound = launch_radius * 0.999

    MAX_NEWTON_ITERS = 12

    def _invert_newton(Xw, Yw, sub_progress=None):
        """Run Newton iteration to find (xe, ye) such that (Sx, Sy)
        evaluated at (xe, ye) equals (Xw, Yw).  Returns OPL at the
        converged entrance positions plus a validity mask.

        Fully vectorised over the input arrays -- ``Xw`` and ``Yw``
        may be any shape; result has the same shape.

        ``sub_progress`` is an optional ``ProgressScaler`` (or any
        callable ``f(frac, msg)``) driven once per Newton iteration.
        """
        # Detect Newton-loop array backend from the evaluator.  The
        # evaluator's xp is either numpy (CPU) or cupy (GPU when
        # use_gpu=True was set earlier).  Using xp uniformly inside
        # the Newton loop keeps this code device-agnostic -- the only
        # other GPU plumbing needed is pushing xe/ye/active/idx_active
        # to xp and pulling opl_flat back to numpy at the end.
        xp = getattr(Sx, 'xp', np)
        # Push wave-grid coordinates to the Newton backend.  On the
        # CPU path this is a zero-cost view; on GPU it's a H->D copy
        # of order (N_wave^2) floats, incurred once per Newton call.
        x_w_flat = xp.asarray(Xw.ravel())
        y_w_flat = xp.asarray(Yw.ravel())
        n_total = int(x_w_flat.size)
        # Initial guess: entrance ~ exit / M, where M is the paraxial
        # magnification measured from the central finite-difference slope
        # of the forward map (see `inv_M_x` / `inv_M_y` computed above from
        # the already-traced ray grid -- no extra compute).  This is a
        # strictly better guess than the pre-3.1.3 hard-coded 1.10
        # multiplier: for singlets with M ~ 1 the two are nearly identical,
        # but for compound systems or unusual magnifications the measured
        # value avoids putting Newton several iterations away from
        # convergence.
        xe = x_w_flat * _spline_data['inv_M_x']
        ye = y_w_flat * _spline_data['inv_M_y']
        tol = 0.01 * dx
        active = xp.ones(xe.size, dtype=bool)  # pixels still iterating
        if sub_progress is not None:
            sub_progress(0.0, f'newton 0/{MAX_NEWTON_ITERS}: {n_total} pixels')
        # When the fit objects support combined value+gradient
        # (polynomial path via _Cheb2DEvaluator), use it to halve the
        # number of Newton-hot-path evaluator calls per iteration from
        # 6 down to 2, and share Chebyshev basis work across f/fx/fy.
        _has_combined = (hasattr(Sx, 'ev_value_and_grad')
                          and hasattr(Sy, 'ev_value_and_grad'))
        for _it in range(MAX_NEWTON_ITERS):
            if not bool(active.any()):
                if sub_progress is not None:
                    sub_progress(1.0,
                                 f'newton converged after {_it} iters')
                break
            # Only evaluate splines at active (unconverged) pixels
            xa = xe[active]; ya = ye[active]
            xw = x_w_flat[active]; yw = y_w_flat[active]
            if _has_combined:
                fx_val, jxx, jxy = Sx.ev_value_and_grad(xa, ya)
                fy_val, jyx, jyy = Sy.ev_value_and_grad(xa, ya)
                rx = fx_val - xw
                ry = fy_val - yw
            else:
                rx = Sx.ev(xa, ya) - xw
                ry = Sy.ev(xa, ya) - yw
                jxx = Sx.ev(xa, ya, dx=1)
                jxy = Sx.ev(xa, ya, dy=1)
                jyx = Sy.ev(xa, ya, dx=1)
                jyy = Sy.ev(xa, ya, dy=1)
            det = jxx * jyy - jxy * jyx
            safe = xp.abs(det) > 1e-12
            inv_det = xp.where(safe, 1.0 / det, 0.0)
            dxe = (jyy * rx - jxy * ry) * inv_det
            dye = (-jyx * rx + jxx * ry) * inv_det
            xa_new = xp.clip(xa - dxe, -bound, bound)
            ya_new = xp.clip(ya - dye, -bound, bound)
            xe[active] = xa_new
            ye[active] = ya_new
            # Mark converged pixels as inactive
            res = xp.sqrt(rx * rx + ry * ry)
            converged = res < tol
            idx_active = xp.where(active)[0]
            active[idx_active[converged]] = False
            if sub_progress is not None:
                remaining = int(active.sum())
                pct_done = 1.0 - remaining / max(n_total, 1)
                # Emit max(iteration-based, convergence-based) fraction,
                # bounded to <1 so the final "assembling" tick owns 1.0.
                frac = min(max((_it + 1) / MAX_NEWTON_ITERS, pct_done),
                           0.99)
                sub_progress(
                    frac,
                    f'newton {_it + 1}/{MAX_NEWTON_ITERS}: '
                    f'{remaining}/{n_total} pixels unconverged')
        opl_flat = So.ev(xe, ye)
        out_of_domain = (xe * xe + ye * ye > (launch_radius * 0.99) ** 2)
        opl_flat = xp.where(out_of_domain, xp.nan, opl_flat)
        # If we ran on GPU, pull the result back to the host so the
        # rest of apply_real_lens_traced -- which is CPU-only
        # (amplitude from apply_real_lens, final field assembly) --
        # sees a NumPy array.
        if xp is not np:
            opl_flat = cp.asnumpy(opl_flat)
        return opl_flat.reshape(Xw.shape)

    # ----- Coarse-grid Newton + interpolation --------------------------
    # The OPL map is extremely smooth (well-approximated by a
    # low-order polynomial), so evaluating the expensive Newton
    # inversion at every wave-grid pixel is wasteful.  Instead we
    # evaluate on a COARSER output grid and bilinearly interpolate to
    # the full wave grid.  ``ray_subsample`` controls the output
    # sub-sampling factor:
    #
    #   ray_subsample=1  -> Newton at every pixel (exact, slow)
    #   ray_subsample=4  -> Newton at every 4th pixel, interp rest
    #   ray_subsample=8  -> Newton at every 8th pixel (fastest)
    #
    # Parallelism: Newton is embarrassingly parallel (per-pixel
    # independent, immutable splines).  We dispatch to a process pool
    # when the grid is large enough that pool startup + knot-pickling
    # is worth it.  Threads don't help here: scipy's
    # ``RectBivariateSpline.ev`` does not release the GIL in current
    # versions, so threading delivers no speedup.

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from ._backends import available_cpus

    # Affinity-aware: respect cgroup limits, taskset masks, Python 3.13+
    # process_cpu_count so we don't oversubscribe a restricted machine.
    # If the user pinned half the cores via taskset (or the container
    # has a CPU quota) we'll see the restricted count here, whereas
    # os.cpu_count() would still return the raw logical total.
    n_cpu = n_workers if n_workers is not None else available_cpus()
    n_cpu = max(1, int(n_cpu))

    # Heuristic: only spin up the pool when the chunk count can actually
    # fill it AND the work per chunk amortises the startup cost.  On
    # Windows spawn mode, pool startup is ~200-400 ms per worker.
    _POOL_MIN_PIXELS = 200_000

    def _invert_newton_parallel(Xw, Yw, sub_progress=None):
        """Dispatch ``_invert_newton`` work across a process pool when
        useful; fall back to the in-process serial path otherwise.

        Preserves the serial path's numerical behaviour exactly (same
        Newton iteration count, same convergence tolerance, same
        out-of-domain NaN policy -- see :func:`_newton_invert_chunk`).
        """
        # GPU path must stay in-process: the worker function
        # ``_newton_invert_chunk`` rebuilds SciPy splines per worker
        # (CPU-only), and shipping CuPy device arrays through a
        # ProcessPoolExecutor would host-copy them anyway.  Go direct.
        if use_gpu:
            return _invert_newton(Xw, Yw, sub_progress=sub_progress)
        # Spline-path worker pool is also incompatible with
        # newton_fit='polynomial' because the worker builds
        # RectBivariateSpline rather than _Cheb2DEvaluator.  Force
        # serial for polynomial until a worker-side polynomial path
        # is added (cheap on Newton-time at subsample=8 anyway).
        if newton_fit == 'polynomial':
            return _invert_newton(Xw, Yw, sub_progress=sub_progress)
        x_w_flat = Xw.ravel()
        y_w_flat = Yw.ravel()
        n_total = x_w_flat.size
        if n_cpu <= 1 or n_total < _POOL_MIN_PIXELS:
            return _invert_newton(Xw, Yw, sub_progress=sub_progress)

        if sub_progress is not None:
            sub_progress(0.0,
                         f'newton pool: {n_total} pts across '
                         f'{n_cpu} workers')
        # Split indices into roughly-equal chunks.  ``np.array_split``
        # handles the n_total % n_cpu != 0 case cleanly.
        chunk_idx = np.array_split(np.arange(n_total), n_cpu)
        args_list = [
            (_spline_data, x_w_flat[c].copy(), y_w_flat[c].copy())
            for c in chunk_idx]
        results = [None] * len(args_list)

        try:
            with ProcessPoolExecutor(max_workers=n_cpu) as ex:
                future_to_idx = {
                    ex.submit(_newton_invert_chunk, a): i
                    for i, a in enumerate(args_list)}
                done = 0
                for fut in as_completed(future_to_idx):
                    i = future_to_idx[fut]
                    results[i] = fut.result()
                    done += 1
                    if sub_progress is not None:
                        frac = min(done / max(len(args_list), 1), 0.99)
                        sub_progress(
                            frac,
                            f'newton chunk {done}/{len(args_list)} done')
        except Exception:
            # Any pool failure (Windows antivirus, spawn error, ...)
            # falls through to the serial path so the caller isn't
            # left without a result.
            return _invert_newton(Xw, Yw, sub_progress=sub_progress)

        opl_flat = np.concatenate(results)
        return opl_flat.reshape(Xw.shape)

    call_progress(progress, 'real_lens_traced', 0.55,
                  'inverting entrance->exit map')
    # Give the Newton loop its own slice of the parent budget
    # (0.55 -> 0.88) so the bar advances through the iterations
    # instead of sitting still between the 0.55 and 0.90 ticks.
    newton_cb = ProgressScaler(progress, 'real_lens_traced',
                               lo=0.55, hi=0.88)

    # ---------- Amplitude-mask the Newton work --------------------
    # Pixels where ``amp`` is well below peak produce a final field
    # of ``|E_analytic| * exp(...)`` that is already ~zero no matter
    # what OPL we compute for them, so running Newton there is
    # wasted effort.  We build a boolean mask on the coarse output
    # grid, dilate by ``newton_mask_dilate_coarse_px`` so bilinear
    # interpolation at the full grid always has real data in its
    # support near mask boundaries, and run Newton only on the
    # masked coarse pixels.  Skipped pixels get ``NaN`` which the
    # existing NaN-propagation logic below treats exactly like the
    # ray-domain-failure NaNs from Newton itself.
    #
    # Controls:
    #   newton_amp_mask_rel=0  disables masking (runs Newton on the
    #                          entire coarse grid, bit-identical to
    #                          pre-mask behaviour).
    #   newton_amp_mask_rel>0  threshold = that fraction of amp.max().
    #   newton_mask_dilate_coarse_px  0 for no dilation, else that
    #                          many iterations of binary_dilation.
    #
    # The mask is SKIPPED if it would capture essentially everything
    # (>95 %) -- in that case the filter overhead isn't worth it --
    # or essentially nothing (<1 %) -- which signals a pathological
    # amp field and we fall back to full-grid Newton rather than
    # returning garbage.
    def _build_newton_mask(amp_grid):
        if newton_amp_mask_rel <= 0.0:
            return None
        amp_max = float(amp_grid.max())
        if amp_max <= 0.0:
            return None
        thresh = amp_max * float(newton_amp_mask_rel)
        m = amp_grid > thresh
        if newton_mask_dilate_coarse_px > 0:
            from scipy.ndimage import binary_dilation
            m = binary_dilation(
                m, iterations=int(newton_mask_dilate_coarse_px))
        frac = float(m.mean())
        if frac > 0.95 or frac < 0.01:
            return None
        return m

    # Dispatch the OPL inversion to Newton (default) or the experimental
    # backward-trace alternative.  Both produce a wave-grid OPL map
    # with the same axis convention (on-axis referenced to zero, NaN
    # for out-of-domain / dead-ray pixels).
    if inversion_method == 'backward_trace':
        # Experimental path.  Bypasses the forward ray trace + Newton
        # spline inversion entirely; see _opl_by_backward_trace for
        # the algorithm and caveats.  Kept as an opt-in because the
        # accuracy on focused-beam exit planes has not been as
        # thoroughly validated as the Newton path.
        opl_map = _opl_by_backward_trace(
            E_analytic, lens_prescription, wavelength, dx,
            N_grid=N, ray_subsample=sub)
    elif sub > 1:
        # Evaluate Newton on sub-sampled output grid
        Xs = X[::sub, ::sub]
        Ys = Y[::sub, ::sub]
        amp_coarse = amp[::sub, ::sub]
        mask_coarse = _build_newton_mask(amp_coarse)
        if mask_coarse is None:
            opl_coarse = _invert_newton_parallel(
                Xs, Ys, sub_progress=newton_cb)
        else:
            Xs_masked = Xs[mask_coarse]
            Ys_masked = Ys[mask_coarse]
            opl_1d = _invert_newton_parallel(
                Xs_masked, Ys_masked, sub_progress=newton_cb)
            opl_coarse = np.full(Xs.shape, np.nan, dtype=opl_1d.dtype)
            opl_coarse[mask_coarse] = opl_1d
        # Bilinearly interpolate to full grid
        from scipy.ndimage import map_coordinates
        Ns = opl_coarse.shape[0]
        ii, jj = np.indices((N, N), dtype=np.float64)
        opl_map = map_coordinates(
            np.where(np.isnan(opl_coarse), 0.0, opl_coarse),
            np.array([ii * Ns / N, jj * Ns / N]),
            order=1, mode='nearest')
        # Propagate NaN mask
        nan_coarse = np.isnan(opl_coarse).astype(np.float64)
        nan_full = map_coordinates(
            nan_coarse,
            np.array([ii * Ns / N, jj * Ns / N]),
            order=1, mode='nearest')
        opl_map = np.where(nan_full > 0.5, np.nan, opl_map)
    else:
        mask_full = _build_newton_mask(amp)
        if mask_full is None:
            opl_map = _invert_newton_parallel(
                X, Y, sub_progress=newton_cb)
        else:
            X_masked = X[mask_full]
            Y_masked = Y[mask_full]
            opl_1d = _invert_newton_parallel(
                X_masked, Y_masked, sub_progress=newton_cb)
            opl_map = np.full(X.shape, np.nan, dtype=opl_1d.dtype)
            opl_map[mask_full] = opl_1d
    call_progress(progress, 'real_lens_traced', 0.90,
                  'assembling exit field')

    # ----- Step 3: combine amplitude with geom phase -------------------
    # When preserve_input_phase=True (default, physically correct):
    #   We KEEP the full complex E_analytic (which already contains the
    #   input field's phase correctly propagated through the glass
    #   split-step) and APPLY A CORRECTION that replaces the analytic
    #   model's lens-only phase with the ray-traced OPL.
    #
    #   delta_phase = k0 * opl_traced - phase_analytic_lens
    #   E_out = E_analytic * exp(i * delta_phase)
    #
    # This preserves any input-field phase structure (source tilts, MLA
    # patterns, off-axis aberrations) that apply_real_lens correctly
    # carried through.  Before this fix, the input phase was silently
    # discarded -- tilted inputs focused on-axis, MLA-modulated inputs
    # came out as a featureless envelope, etc.
    #
    # When preserve_input_phase=False (legacy behaviour):
    #   E_out = |E_analytic| * exp(i * k0 * opl_traced).  Useful for
    #   measuring the lens-only OPD on a plane-wave input, where the
    #   input-phase question is moot.
    k0 = 2.0 * np.pi / wavelength
    valid = np.isfinite(opl_map)
    # Preserve the caller's complex dtype: apply_real_lens (called
    # above to build E_analytic / amp) already returns a field in
    # E_in.dtype, but the ``* np.exp(1j * ...)`` multiply here would
    # silently upcast to complex128 unless we cast the exp() result.
    target_cdtype = E_in.dtype if np.iscomplexobj(E_in) else np.complex128
    if preserve_input_phase:
        delta_phase = np.where(valid, k0 * opl_map - phase_analytic_lens, 0.0)
        phase_exp = np.exp(1j * delta_phase)
        if phase_exp.dtype != target_cdtype:
            phase_exp = phase_exp.astype(target_cdtype)
        E_out = E_analytic * phase_exp
    else:
        phase = np.where(valid, k0 * opl_map, 0.0)
        phase_exp = np.exp(1j * phase)
        if phase_exp.dtype != target_cdtype:
            phase_exp = phase_exp.astype(target_cdtype)
        E_out = amp * phase_exp
    # Zero outside the exit-pupil (ray-coverage) region
    E_out = np.where(valid, E_out, target_cdtype.type(0))
    # And outside the entrance aperture (defensive: in practice the
    # ray-coverage region is a subset of the entrance aperture, so
    # this is a no-op except in pathological configurations)
    if aperture is not None:
        E_out = np.where(X ** 2 + Y ** 2 <= (aperture / 2) ** 2,
                         E_out, target_cdtype.type(0))
    if E_out.dtype != target_cdtype:
        E_out = E_out.astype(target_cdtype)
    call_progress(progress, 'real_lens_traced', 1.0, 'done')
    return E_out


# ---------------------------------------------------------------------------

def apply_cylindrical_lens(E_in, f, wavelength, dx, dy=None, axis='x',
                           xc=0, yc=0):
    """
    Apply a cylindrical thin-lens phase (focusing in one axis only).

    Parameters
    ----------
    E_in : ndarray (complex, N x N)
        Input electric field.
    f : float
        Focal length [m].  Positive = converging.
    wavelength : float
        Optical wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float or None
        Grid spacing in y [m].  Defaults to *dx*.
    axis : ``'x'`` or ``'y'``
        Focusing axis.  ``'x'`` applies phi = -k/(2f) * (x - xc)**2;
        ``'y'`` applies phi = -k/(2f) * (y - yc)**2.
    xc, yc : float
        Lens center [m].

    Returns
    -------
    E_out : ndarray (complex, N x N)

    Notes
    -----
    Produces a line focus (orthogonal to the focusing axis) instead of a
    point focus.
    """
    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx
    k = 2 * np.pi / wavelength

    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy

    if axis == 'x':
        phase_1d = -k / (2 * f) * (x - xc) ** 2
        phase = phase_1d[np.newaxis, :]
    elif axis == 'y':
        phase_1d = -k / (2 * f) * (y - yc) ** 2
        phase = phase_1d[:, np.newaxis]
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    return E_in * np.exp(1j * phase)


# ---------------------------------------------------------------------------
# GRIN lens
# ---------------------------------------------------------------------------

def apply_grin_lens(E_in, n0, g, d, wavelength, dx, dy=None, xc=0, yc=0):
    """
    Apply a gradient-index (GRIN) rod lens phase (thin approximation).

    Models a GRIN rod with parabolic index profile:

        n(r) = n0 * (1 - g**2 / 2 * r**2)

    Parameters
    ----------
    E_in : ndarray (complex, N x N)
        Input electric field.
    n0 : float
        On-axis refractive index.
    g : float
        Gradient constant [1/m] (also called sqrt(A)).
        Pitch P = 2 pi / g.
    d : float
        Rod length (thickness) [m].
    wavelength : float
        Optical wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float or None
        Grid spacing in y [m].  Defaults to *dx*.
    xc, yc : float
        GRIN lens center [m].

    Returns
    -------
    E_out : ndarray (complex, N x N)

    Notes
    -----
    The quadratic OPD through the rod gives an effective focal length

        f = 1 / (n0 * g**2 * d)      (thin approximation, g*d << 1)

    For longer rods the exact result is ``f = 1 / (n0 * g * sin(g*d))``.
    Quarter-pitch (g*d = pi/2) collimates a point source at the front face;
    half-pitch (g*d = pi) reimages 1:1 inverted.
    """
    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx
    k = 2 * np.pi / wavelength

    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)
    r_sq = (X - xc) ** 2 + (Y - yc) ** 2

    phase = -k * n0 * (g ** 2 / 2) * d * r_sq
    return E_in * np.exp(1j * phase)


# ---------------------------------------------------------------------------
# Axicon
# ---------------------------------------------------------------------------

def apply_axicon(E_in, alpha, n_axicon, wavelength, dx, dy=None, xc=0, yc=0):
    """
    Apply an axicon (conical lens) phase to generate a Bessel-like beam.

    Parameters
    ----------
    E_in : ndarray (complex, N x N)
        Input electric field.
    alpha : float
        Physical half-angle of the cone [radians].
        Typical range: 0.5--5 degrees (0.009--0.087 rad).
    n_axicon : float or str
        Refractive index of the axicon material.  If a string is passed it
        is resolved via :func:`get_glass_index`.
    wavelength : float
        Optical wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float or None
        Grid spacing in y [m].  Defaults to *dx*.
    xc, yc : float
        Axicon center [m].

    Returns
    -------
    E_out : ndarray (complex, N x N)

    Notes
    -----
    The axicon imparts a phase linear in radial distance:

        phi(r) = -k * (n - 1) * alpha * r

    A collimated input beam produces a non-diffracting Bessel-beam region
    extending over ``z_max ~ w0 / ((n - 1) * alpha)`` where *w0* is the
    input beam radius.
    """
    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx
    k = 2 * np.pi / wavelength

    if isinstance(n_axicon, str):
        n = get_glass_index(n_axicon, wavelength)
    else:
        n = float(n_axicon)

    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)

    phase = -k * (n - 1) * alpha * r
    return E_in * np.exp(1j * phase)


# ============================================================
# MASLOV PHASE-SPACE PROPAGATOR
# ============================================================
# Moved inline from the former top-level lens_maslov.py
# module in v3.2.2; kept in the same file as the rest of the
# lens pipeline so the Maslov variant sits alongside its
# siblings (apply_real_lens, apply_real_lens_traced).

from . import raytrace as rt


# ---------------------------------------------------------------------------
# Chebyshev polynomial helpers (numpy-native, vectorised)
# ---------------------------------------------------------------------------

def _chebyshev_vandermonde(u: np.ndarray, max_k: int) -> np.ndarray:
    """
    Build the Chebyshev Vandermonde-like array T[n](u) for n = 0..max_k.

    Parameters
    ----------
    u : ndarray, any shape, values in [-1, 1]
    max_k : int

    Returns
    -------
    T : ndarray of shape (max_k+1,) + u.shape
        T[n] is T_n(u), computed by the standard 3-term recurrence.
    """
    u = np.asarray(u)
    T = np.empty((max_k + 1,) + u.shape, dtype=np.float64)
    T[0] = 1.0
    if max_k >= 1:
        T[1] = u
    for n in range(2, max_k + 1):
        T[n] = 2.0 * u * T[n - 1] - T[n - 2]
    return T


def _chebyshev_derivative_vandermonde(u: np.ndarray, max_k: int
                                       ) -> np.ndarray:
    """
    Build T_n'(u) for n = 0..max_k.  Uses T_n'(x) = n * U_{n-1}(x),
    where U is the Chebyshev polynomial of the second kind.

    Returns
    -------
    Tp : ndarray of shape (max_k+1,) + u.shape
    """
    u = np.asarray(u)
    Tp = np.zeros((max_k + 1,) + u.shape, dtype=np.float64)
    if max_k < 1:
        return Tp
    # U_0(x) = 1, U_1(x) = 2x, U_{n+1} = 2x U_n - U_{n-1}
    U = np.empty((max_k + 1,) + u.shape, dtype=np.float64)
    U[0] = 1.0
    if max_k >= 1:
        U[1] = 2.0 * u
    for n in range(2, max_k + 1):
        U[n] = 2.0 * u * U[n - 1] - U[n - 2]
    # T_n'(x) = n * U_{n-1}(x)  for n >= 1
    for n in range(1, max_k + 1):
        Tp[n] = float(n) * U[n - 1]
    return Tp


def _chebyshev_second_derivative_vandermonde(u: np.ndarray, max_k: int
                                              ) -> np.ndarray:
    """
    Build T_n''(u) for n = 0..max_k.

    T''_n(x) can be derived from T_n and U_n via the identity
        T''_n(x) = n * ((n+1) T_n(x) - U_n(x)) / (x^2 - 1)   (x != +/- 1)
    but this has singular denominators at the endpoints.  A more stable
    recurrence is obtained by differentiating the standard T recurrence
    once more:
        T''_0 = 0,  T''_1 = 0,  T''_2 = 4,
        T''_{n+1} = 2 x T''_n + 4 T'_n - T''_{n-1}

    Uses the same 3-term recurrence style as the first-derivative
    helper, so the cost is O(max_k) per evaluation point.

    Returns
    -------
    Tpp : ndarray of shape (max_k+1,) + u.shape
    """
    u = np.asarray(u)
    shape = u.shape
    Tpp = np.zeros((max_k + 1,) + shape, dtype=np.float64)
    if max_k < 2:
        return Tpp
    # We'll need T'_n to drive the recurrence
    Tp = _chebyshev_derivative_vandermonde(u, max_k)
    # T''_0 = 0, T''_1 = 0, T''_2 = 4 (constant)
    Tpp[2] = 4.0 * np.ones(shape, dtype=np.float64)
    for n in range(2, max_k):
        Tpp[n + 1] = 2.0 * u * Tpp[n] + 4.0 * Tp[n] - Tpp[n - 1]
    return Tpp


def _multi_indices_total_degree(n_vars: int, max_order: int):
    """Enumerate multi-indices k with sum(k) <= max_order, as list of tuples."""
    out = []
    def recurse(prefix, remaining, depth):
        if depth == n_vars:
            out.append(tuple(prefix))
            return
        for k in range(remaining + 1):
            recurse(prefix + [k], remaining - k, depth + 1)
    recurse([], max_order, 0)
    return out


def _evaluate_polynomial_4d(coeffs: np.ndarray,
                              multi_indices,
                              u1: np.ndarray, u2: np.ndarray,
                              u3: np.ndarray, u4: np.ndarray,
                              max_order: int) -> np.ndarray:
    """
    Evaluate a 4-variable Chebyshev tensor-product polynomial in
    total-degree subspace at arbitrary (u1, u2, u3, u4) samples.

    Parameters
    ----------
    coeffs : ndarray, shape (M,)
        Polynomial coefficients in the same order as ``multi_indices``.
    multi_indices : list of 4-tuples
        Multi-indices (k1, k2, k3, k4) enumerating the basis.
    u1, u2, u3, u4 : ndarrays with identical shape
        Evaluation points in [-1, 1]^4.
    max_order : int
        Maximum individual index (== total-degree cap in this call).

    Returns
    -------
    value : ndarray with the broadcast shape of (u1, ..., u4)
    """
    # Broadcast shapes
    shape = np.broadcast(u1, u2, u3, u4).shape
    T1 = _chebyshev_vandermonde(u1, max_order)   # (max_k+1,) + shape
    T2 = _chebyshev_vandermonde(u2, max_order)
    T3 = _chebyshev_vandermonde(u3, max_order)
    T4 = _chebyshev_vandermonde(u4, max_order)
    out = np.zeros(shape, dtype=np.float64)
    for c, (k1, k2, k3, k4) in zip(coeffs, multi_indices):
        if c == 0.0:
            continue
        out = out + c * T1[k1] * T2[k2] * T3[k3] * T4[k4]
    return out


def _evaluate_polynomial_4d_and_grad34(coeffs: np.ndarray,
                                         multi_indices,
                                         u1, u2, u3, u4,
                                         max_order: int
                                         ) -> Tuple[np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray]:
    """
    Evaluate the 4-variable polynomial at (u1, u2, u3, u4) and also its
    partial derivatives d/du3 and d/du4 (used for the Jacobian w.r.t.
    the v2 coordinates).

    Returns
    -------
    f, df_du3, df_du4
    """
    shape = np.broadcast(u1, u2, u3, u4).shape
    T1 = _chebyshev_vandermonde(u1, max_order)
    T2 = _chebyshev_vandermonde(u2, max_order)
    T3 = _chebyshev_vandermonde(u3, max_order)
    T4 = _chebyshev_vandermonde(u4, max_order)
    dT3 = _chebyshev_derivative_vandermonde(u3, max_order)
    dT4 = _chebyshev_derivative_vandermonde(u4, max_order)
    f = np.zeros(shape, dtype=np.float64)
    df3 = np.zeros(shape, dtype=np.float64)
    df4 = np.zeros(shape, dtype=np.float64)
    for c, (k1, k2, k3, k4) in zip(coeffs, multi_indices):
        if c == 0.0:
            continue
        T12 = T1[k1] * T2[k2]
        f = f + c * T12 * T3[k3] * T4[k4]
        df3 = df3 + c * T12 * dT3[k3] * T4[k4]
        df4 = df4 + c * T12 * T3[k3] * dT4[k4]
    return f, df3, df4


# ---------------------------------------------------------------------------
# Data normalisation helpers
# ---------------------------------------------------------------------------

def _fit_normaliser(v: np.ndarray, pad: float = 0.05):
    """Return (center, half_range) such that (v - center)/half_range sits
    in [-(1-pad), (1-pad)].

    pad leaves a narrow margin so that mild extrapolation by the
    propagator is still bounded.
    """
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    center = 0.5 * (vmin + vmax)
    half = 0.5 * (vmax - vmin) * (1.0 + pad)
    if half == 0.0:
        half = 1.0
    return center, half


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def apply_real_lens_maslov(
    E_in: np.ndarray,
    lens_prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    *,
    ray_field_samples: int = 16,
    ray_pupil_samples: int = 16,
    poly_order: int = 4,
    n_v2: int = 32,
    output_subsample: int = 1,
    extract_linear_phase: bool = True,
    chunk_v2: int = 64,
    use_numexpr: Optional[bool] = None,
    integration_method: str = 'quadrature',
    stationary_newton_iter: int = 12,
    stationary_newton_tol: float = 1e-10,
    local_n_samples: int = 8,
    local_window_sigma: float = 3.0,
    collimated_input: bool = False,
    normalize_output: str = 'power',
    verbose: bool = False,
    progress: Optional[Any] = None,
) -> np.ndarray:
    """
    Phase-space / Maslov propagator through a thick-lens prescription.

    Traces a Chebyshev-node grid of rays from the entrance plane of
    ``lens_prescription`` to the exit plane, fits a 4-variable
    Chebyshev tensor-product polynomial to

        s1(s2, v2),  OPD(s2, v2)

    (back-map and wavefront), and evaluates the Maslov integral

        E(s2) = integral  E_in(s1(s2, v2)) * exp(2 pi i OPD(s2, v2))
                          * |det(ds1/dv2)|  d^2 v2

    at each output pixel.

    Parameters
    ----------
    E_in : ndarray, complex, shape (N, N)
        Input field at the entrance plane of the lens.
    lens_prescription : dict
        Same format used by :func:`apply_real_lens` /
        :func:`apply_real_lens_traced`.
    wavelength : float
        Vacuum wavelength [m].
    dx : float
        Grid spacing [m] (square grid assumed).
    ray_field_samples, ray_pupil_samples : int
        Number of Chebyshev-node samples along each of the two "field"
        (entrance-position) axes and two "pupil" (entrance-direction)
        axes respectively.  Total rays = ray_field_samples^2 *
        ray_pupil_samples^2.  Defaults (16, 16) give 65 536 rays, which
        in practice saturates a 4th-order polynomial fit without
        runaway cost.
    poly_order : int, default 4
        Max total degree of the Chebyshev tensor-product fit.  4 -> 70
        coefficients per polynomial (fast, accurate for most refractive
        systems).  6 -> 210 (captures strongly-aberrated and
        diffractive systems per OPDGPU maslov_zemax_merit.tex).
    n_v2 : int, default 32
        Number of quadrature samples per axis in the v2 direction
        cosine integral.  Maslov quadrature-validity bound:
        ``source waist >= entrance pupil diameter / n_v2``.
    output_subsample : int, default 1
        Evaluate the Maslov integral only on every ``output_subsample``-
        th pixel; bilinearly interpolate to the full output grid.
        Analogous to apply_real_lens_traced's ``ray_subsample``.  4 is a
        good production value.
    extract_linear_phase : bool, default True
        Pre-fit and drop the linear component of OPD(s2, v2) before the
        Chebyshev fit.  Repairs the missing grating kick for diffractive
        surfaces at nonzero orders (see OPDGPU Section 4); no-op for
        zeroth-order / refractive-only systems.
    verbose : bool, default False
        If True, print timing and diagnostic info.
    progress : callable or None
        Optional callback ``progress(phase, fraction, elapsed_s, note)``
        for UI integration.

    Returns
    -------
    E_out : ndarray, complex, shape (N, N)
        Exit-plane field.

    Notes
    -----
    Prototype implementation, pure numpy on CPU.  For large grids (N >
    ~4096) set ``output_subsample`` to 4 or 8; for GPU acceleration a
    CuPy drop-in can be added to :func:`_integrate_maslov_vectorised`.

    References
    ----------
    Kravtsov & Orlov, *Caustics, Catastrophes and Wave Fields* (1999).
    Forbes & Alonso, JOSA A 15, 1341 (1998).
    """
    t0 = time.perf_counter()
    E_in = np.asarray(E_in)
    if E_in.ndim != 2 or E_in.shape[0] != E_in.shape[1]:
        raise ValueError(
            f"E_in must be square 2D, got shape {E_in.shape}")
    N = E_in.shape[0]

    # Pre-flight grid vs prescription-aperture check.
    try:
        _warn_if_aperture_exceeds_grid(
            lens_prescription, N, dx, source='apply_real_lens_maslov')
    except Exception:
        pass

    def _progress(phase, frac, note=''):
        if progress is not None:
            try:
                progress(phase=phase, fraction=float(frac),
                         elapsed=time.perf_counter() - t0, note=note)
            except TypeError:
                progress(phase, float(frac), time.perf_counter() - t0,
                         note)
        if verbose:
            dt = time.perf_counter() - t0
            print(f"  maslov {phase:>10s}  {frac*100:5.1f}%  "
                  f"({dt:6.1f}s) {note}", flush=True)

    # -----------------------------------------------------------------
    # Step 1: Trace rays on a Chebyshev-node (h, p) grid
    # -----------------------------------------------------------------
    _progress('trace', 0.0, 'building ray bundle')

    surfaces = rt.surfaces_from_prescription(lens_prescription)
    if not surfaces:
        raise ValueError("Lens prescription has no surfaces.")

    # Entrance aperture for ray sampling
    aperture_m = lens_prescription.get('aperture_diameter', None)
    if aperture_m is None:
        # Fall back to smallest surface semi-diameter
        sds = [s.semi_diameter for s in surfaces if np.isfinite(s.semi_diameter)]
        if sds:
            aperture_m = 2.0 * min(sds)
        else:
            aperture_m = N * dx * 0.5  # last-resort: half the grid
    r_aperture = 0.5 * aperture_m

    # Cosine-spaced (Chebyshev) nodes h_i = cos((i - 1/2) pi / N) on [-1, 1]
    def cheb_nodes(n):
        i = np.arange(n)
        return np.cos((i + 0.5) * np.pi / n)

    hx = cheb_nodes(ray_field_samples)   # entrance-position x normalised
    hy = cheb_nodes(ray_field_samples)   # entrance-position y normalised
    px = cheb_nodes(ray_pupil_samples)   # entrance-direction x normalised
    py = cheb_nodes(ray_pupil_samples)

    HX, HY, PX, PY = np.meshgrid(hx, hy, px, py, indexing='ij')
    HX = HX.ravel(); HY = HY.ravel()
    PX = PX.ravel(); PY = PY.ravel()

    # Discard pupil samples outside the unit disc (p_x^2 + p_y^2 > 1)
    keep = (PX**2 + PY**2) <= 1.0
    HX, HY, PX, PY = HX[keep], HY[keep], PX[keep], PY[keep]
    n_rays = len(HX)
    if n_rays < 1.5 * _count_multi_indices_4d(poly_order):
        raise ValueError(
            f"Only {n_rays} rays survived pupil masking; need at least "
            f"~{int(1.5 * _count_multi_indices_4d(poly_order))} "
            f"for a well-conditioned order-{poly_order} fit.")

    # Map (hx, hy, px, py) -> physical entrance coordinates & directions
    s1x = HX * r_aperture
    s1y = HY * r_aperture

    # Max pupil half-angle: use the lens's paraxial acceptance NA derived
    # from its effective focal length via system_abcd_prescription.  This
    # is much tighter (and physically correct) than the old
    # aperture/lens_total_thickness heuristic, which over-filled the
    # 4-D phase space with vignettable / TIR-bound rays.
    if collimated_input:
        # Collimated input: v1 = 0 across the aperture.  No "pupil" angle
        # to sweep; the pupil dimensions (px, py) are redundant but kept
        # for code-path symmetry and polynomial-fit conditioning.
        # Populate v1 with a very small angular spread so the polynomial
        # fit has enough numerical conditioning.
        na_proxy = 1e-5   # essentially 0
    else:
        try:
            _M, _efl, _bfl, _ffl = rt.system_abcd_prescription(
                lens_prescription, wavelength)
            efl_abs = float(abs(_efl))
            if np.isfinite(efl_abs) and efl_abs > 0:
                na_proxy = r_aperture / max(efl_abs, r_aperture * 10)
            else:
                lens_total_thickness = sum(s.thickness for s in surfaces)
                na_proxy = r_aperture / max(lens_total_thickness,
                                             r_aperture * 10)
        except Exception:
            lens_total_thickness = sum(s.thickness for s in surfaces)
            na_proxy = r_aperture / max(lens_total_thickness,
                                         r_aperture * 10)

    if verbose:
        print(f"  NA_proxy = {na_proxy:.5f}  "
              f"(collimated_input={collimated_input})")

    v1x = PX * na_proxy
    v1y = PY * na_proxy
    # Build direction cosines (L, M, N) with L^2+M^2+N^2=1, N>0
    N_dir = np.sqrt(np.maximum(1.0 - v1x**2 - v1y**2, 0.0))
    _progress('trace', 0.05, f'{n_rays} rays prepared')

    rays = rt.RayBundle(
        x=s1x.copy(), y=s1y.copy(), z=np.zeros_like(s1x),
        L=v1x.copy(), M=v1y.copy(), N=N_dir,
        wavelength=wavelength,
        alive=np.ones(n_rays, dtype=bool),
        opd=np.zeros(n_rays),
    )

    tr = rt.trace(rays, surfaces, wavelength)
    exit_rays = tr.image_rays
    alive = exit_rays.alive
    if alive.sum() < 1.5 * _count_multi_indices_4d(poly_order):
        raise ValueError(
            f"Only {alive.sum()}/{n_rays} rays survived the trace; "
            f"likely aperture / TIR issue.  Check prescription.")

    s2x = exit_rays.x[alive]
    s2y = exit_rays.y[alive]
    v2x = exit_rays.L[alive]
    v2y = exit_rays.M[alive]
    opd_m = exit_rays.opd[alive] - rays.opd[alive]   # only the lens OPL
    opd_w = opd_m / wavelength
    s1x_live = s1x[alive]
    s1y_live = s1y[alive]
    _progress('trace', 0.15, f'{alive.sum()} alive rays; '
              f'OPD p-v = {opd_w.max()-opd_w.min():.3f} waves')

    # -----------------------------------------------------------------
    # Step 2: Normalise (s2, v2) to [-1, 1]^4 and fit Chebyshev polys
    # -----------------------------------------------------------------
    _progress('fit', 0.15, 'normalising inputs')
    s2x_c, s2x_h = _fit_normaliser(s2x)
    s2y_c, s2y_h = _fit_normaliser(s2y)
    v2x_c, v2x_h = _fit_normaliser(v2x)
    v2y_c, v2y_h = _fit_normaliser(v2y)

    u_s2x = (s2x - s2x_c) / s2x_h
    u_s2y = (s2y - s2y_c) / s2y_h
    u_v2x = (v2x - v2x_c) / v2x_h
    u_v2y = (v2y - v2y_c) / v2y_h

    # Optional linear-phase extraction (OPDGPU Section 4)
    linear_coeffs = None
    if extract_linear_phase:
        X5 = np.column_stack([
            np.ones_like(u_s2x),
            u_s2x, u_s2y, u_v2x, u_v2y,
        ])
        linear_coeffs, *_ = np.linalg.lstsq(X5, opd_w, rcond=None)
        opd_linear = X5 @ linear_coeffs
        opd_residual = opd_w - opd_linear
    else:
        opd_residual = opd_w.copy()

    # Chebyshev tensor-product fit, total-degree <= poly_order
    mi = _multi_indices_total_degree(4, poly_order)
    M = len(mi)
    # Build design matrix by stacking per-ray basis evaluations
    _progress('fit', 0.25, f'building design matrix ({n_rays} x {M})')
    # (max_k+1,) + shape
    T1 = _chebyshev_vandermonde(u_s2x, poly_order)
    T2 = _chebyshev_vandermonde(u_s2y, poly_order)
    T3 = _chebyshev_vandermonde(u_v2x, poly_order)
    T4 = _chebyshev_vandermonde(u_v2y, poly_order)
    A = np.empty((len(u_s2x), M), dtype=np.float64)
    for j, (k1, k2, k3, k4) in enumerate(mi):
        A[:, j] = T1[k1] * T2[k2] * T3[k3] * T4[k4]

    _progress('fit', 0.35, 'solving lstsq for OPD')
    coef_opd, *_ = np.linalg.lstsq(A, opd_residual, rcond=None)
    _progress('fit', 0.45, 'solving lstsq for s1x')
    coef_s1x, *_ = np.linalg.lstsq(A, s1x_live, rcond=None)
    _progress('fit', 0.55, 'solving lstsq for s1y')
    coef_s1y, *_ = np.linalg.lstsq(A, s1y_live, rcond=None)

    # Fit quality diagnostics
    opd_pred = A @ coef_opd
    s1x_pred = A @ coef_s1x
    s1y_pred = A @ coef_s1y
    res_opd = np.sqrt(np.mean((opd_residual - opd_pred)**2))
    res_s1x = np.sqrt(np.mean((s1x_live - s1x_pred)**2)) * 1e6  # um
    res_s1y = np.sqrt(np.mean((s1y_live - s1y_pred)**2)) * 1e6

    _progress('fit', 0.60,
              f'RMS OPD residual = {res_opd:.2e} waves; '
              f's1x RMS = {res_s1x:.2e} um, s1y RMS = {res_s1y:.2e} um')

    # -----------------------------------------------------------------
    # Step 3: Build output grids (possibly subsampled)
    # -----------------------------------------------------------------
    _progress('grid', 0.60, 'setting up output and v2 grids')
    if output_subsample < 1:
        output_subsample = 1
    N_out_coarse = N // output_subsample

    # Output pixel positions, centred on the grid
    out_axis = (np.arange(N_out_coarse) - N_out_coarse / 2) * \
               (dx * output_subsample)
    s2x_grid, s2y_grid = np.meshgrid(out_axis, out_axis, indexing='xy')

    # Normalise output positions to the fit box
    u_s2x_out = (s2x_grid - s2x_c) / s2x_h
    u_s2y_out = (s2y_grid - s2y_c) / s2y_h
    # Mask: evaluate only where the fit is valid
    inbox = (np.abs(u_s2x_out) <= 1.0) & (np.abs(u_s2y_out) <= 1.0)

    # v2 quadrature grid -- uniform on normalised [-1, 1] with Tukey
    # apodisation to avoid ringing from hard truncation.
    u_v2x_samples = np.linspace(-1.0, 1.0, n_v2)
    u_v2y_samples = np.linspace(-1.0, 1.0, n_v2)
    du = u_v2x_samples[1] - u_v2x_samples[0]

    # Tukey window (cosine taper on 20% of range) per axis
    def tukey(n, alpha=0.2):
        u = np.linspace(-1, 1, n)
        abs_u = np.abs(u)
        w = np.ones_like(u)
        taper_start = 1.0 - alpha
        tmask = abs_u > taper_start
        w[tmask] = 0.5 * (1 + np.cos(np.pi * (abs_u[tmask] - taper_start) / alpha))
        return w
    tuk_x = tukey(n_v2)
    tuk_y = tukey(n_v2)
    tuk_2d = tuk_x[None, :] * tuk_y[:, None]

    # Physical v2 samples (for diagnostics / Jacobian scaling)
    v2x_samples = v2x_c + u_v2x_samples * v2x_h
    v2y_samples = v2y_c + u_v2y_samples * v2y_h

    # Input-field bilinear sampling helper
    def sample_E_bilinear(s1x_q: np.ndarray, s1y_q: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of E_in at arbitrary (s1x, s1y)."""
        in_axis = (np.arange(N) - N / 2) * dx
        # Convert to fractional pixel index
        fx = (s1x_q - in_axis[0]) / dx
        fy = (s1y_q - in_axis[0]) / dx
        ix = np.floor(fx).astype(np.int64)
        iy = np.floor(fy).astype(np.int64)
        wx = fx - ix
        wy = fy - iy
        # Mask valid indices
        ok = (ix >= 0) & (ix < N - 1) & (iy >= 0) & (iy < N - 1)
        ix_c = np.clip(ix, 0, N - 2)
        iy_c = np.clip(iy, 0, N - 2)
        e00 = E_in[iy_c, ix_c]
        e10 = E_in[iy_c, ix_c + 1]
        e01 = E_in[iy_c + 1, ix_c]
        e11 = E_in[iy_c + 1, ix_c + 1]
        val = ((1 - wx) * (1 - wy) * e00
               + wx * (1 - wy) * e10
               + (1 - wx) * wy * e01
               + wx * wy * e11)
        val = np.where(ok, val, 0.0 + 0.0j)
        return val

    # -----------------------------------------------------------------
    # Step 4: Integrate
    #
    # Two methods available via the ``integration_method`` kwarg:
    #
    #   'quadrature'        (default) -- Riemann sum on a Tukey-windowed
    #                       uniform v2 grid.  Correct for extended
    #                       sources well inside the quadrature validity
    #                       bound w_s >= D_s1 / n_v2.
    #
    #   'stationary_phase'  Leading-order asymptotic.  For each output
    #                       pixel, Newton-iterate on the fitted OPD
    #                       polynomial to locate the stationary point
    #                       v2* where grad_v2 OPD = 0, then evaluate
    #
    #                           E(s2) = |det J_{s1v2}|
    #                                 * E_obj(s1(s2, v2*))
    #                                 * exp(2 pi i * OPD(s2, v2*))
    #                                 * exp(i pi sigma/4)
    #                                 / sqrt(|det H_{v2v2} OPD|)
    #
    #                       with Hessian H and signature sigma.  This
    #                       is the analytic Gaussian-moment limit of
    #                       Forbes-Alonso / small_waist_asymptotic.tex;
    #                       it gives the correct answer when the
    #                       integrand is delta-like in v2 (collimated
    #                       input, a geometric image) -- the case where
    #                       uniform quadrature silently undercounts.
    # -----------------------------------------------------------------
    if integration_method not in ('quadrature', 'stationary_phase',
                                    'local_quadrature'):
        raise ValueError(
            f"integration_method must be one of 'quadrature', "
            f"'stationary_phase', 'local_quadrature', "
            f"got {integration_method!r}")

    # -------- shared precompute used by both paths --------
    _progress('integrate', 0.60,
              f'method={integration_method}; '
              f'precomputing (s2)-basis on {N_out_coarse}^2 output grid')

    Tx_1d = _chebyshev_vandermonde(
        (out_axis - s2x_c) / s2x_h, poly_order)
    Ty_1d = _chebyshev_vandermonde(
        (out_axis - s2y_c) / s2y_h, poly_order)

    M = len(mi)
    G = np.empty((N_out_coarse * N_out_coarse, M), dtype=np.float64)
    for m, (k1, k2, _, _) in enumerate(mi):
        G[:, m] = np.outer(Ty_1d[k2], Tx_1d[k1]).ravel()
    _progress('integrate', 0.63,
              f'G matrix {G.shape} = {G.nbytes/1e6:.1f} MB')

    K1_arr = np.array([k[0] for k in mi], dtype=np.int64)
    K2_arr = np.array([k[1] for k in mi], dtype=np.int64)
    K3_arr = np.array([k[2] for k in mi], dtype=np.int64)
    K4_arr = np.array([k[3] for k in mi], dtype=np.int64)

    inbox_flat = inbox.ravel()

    if integration_method == 'stationary_phase':
        E_out_coarse = _integrate_stationary_phase(
            coef_opd, coef_s1x, coef_s1y, mi,
            K1_arr, K2_arr, K3_arr, K4_arr,
            poly_order, G, N_out_coarse,
            u_s2x_out, u_s2y_out, inbox_flat,
            v2x_c, v2y_c, v2x_h, v2y_h,
            sample_E_bilinear,
            stationary_newton_iter, stationary_newton_tol,
            _progress, verbose,
        )
    elif integration_method == 'local_quadrature':
        E_out_coarse = _integrate_local_quadrature(
            coef_opd, coef_s1x, coef_s1y, mi,
            K1_arr, K2_arr, K3_arr, K4_arr,
            poly_order, G, N_out_coarse,
            u_s2x_out, u_s2y_out, inbox_flat,
            v2x_c, v2y_c, v2x_h, v2y_h,
            sample_E_bilinear,
            stationary_newton_iter, stationary_newton_tol,
            local_n_samples, local_window_sigma,
            _progress, verbose,
        )
    else:
        E_out_coarse = _integrate_quadrature(
            coef_opd, coef_s1x, coef_s1y, mi,
            K1_arr, K2_arr, K3_arr, K4_arr,
            poly_order, G, N_out_coarse,
            u_v2x_samples, u_v2y_samples, tuk_2d, du,
            v2x_h, v2y_h, chunk_v2, inbox_flat,
            sample_E_bilinear,
            use_numexpr, _progress,
        )

    # -----------------------------------------------------------------
    # Step 5: Upsample to the full grid if output_subsample > 1
    # -----------------------------------------------------------------
    if output_subsample > 1:
        _progress('upsample', 0.95,
                  f'interpolating {N_out_coarse}^2 -> {N}^2 (cubic)')
        from scipy.ndimage import zoom
        # Cubic (order=3) on the AMPLITUDE and PHASE separately to avoid
        # the bilinear interpolation of a rapidly-varying complex field
        # that produced ~4% RMS errors in the first implementation.
        #
        # Interpolating |E| and unwrapped phase independently is more
        # faithful to the Maslov output than interpolating re/im parts
        # of the complex field directly, because both |E| and the
        # stationary-phase OPD are smooth functions of s2 while the
        # complex field oscillates rapidly.
        zoom_factor = float(N) / float(N_out_coarse)
        amp = np.abs(E_out_coarse)
        # Use a "reference" phase tracked across the coarse grid -- we
        # don't have the OPD polynomial handy here, so fall back to
        # unwrapping the coarse-grid phase before zoom.  For large
        # grids, prefer output_subsample=1 instead.
        phase_c = np.angle(E_out_coarse)
        amp_z   = zoom(amp,     zoom_factor, order=3, mode='nearest')
        # Unwrap only along rows then columns -- cheap and good enough
        # for a smooth OPD.
        phase_unw = np.unwrap(np.unwrap(phase_c, axis=1), axis=0)
        phase_z = zoom(phase_unw, zoom_factor, order=3, mode='nearest')
        E_out_re = amp_z * np.cos(phase_z)
        E_out_im = amp_z * np.sin(phase_z)
        # Crop or pad to ensure exactly (N, N)
        def _fit(a):
            if a.shape == (N, N):
                return a
            out = np.zeros((N, N), dtype=a.dtype)
            rows = min(a.shape[0], N)
            cols = min(a.shape[1], N)
            out[:rows, :cols] = a[:rows, :cols]
            return out
        E_out = _fit(E_out_re) + 1j * _fit(E_out_im)
    else:
        E_out = E_out_coarse

    # -----------------------------------------------------------------
    # Step 6: Absolute-amplitude normalization.
    # The stationary-phase formula as written omits the Huygens-Fresnel
    # global prefactor (~-i*k/(2*pi*z) for planar propagation) so the
    # raw output amplitude is miscalibrated by a constant.  We scale
    # here to recover a physically-meaningful amplitude; default is
    # 'power' which preserves total |E|^2 (lossless lens limit).
    # Other options: 'peak' matches max|E|, 'none' leaves the raw
    # formula output, or pass a scalar to multiply directly.
    # -----------------------------------------------------------------
    if normalize_output == 'power':
        p_in = float((np.abs(E_in)**2).sum())
        p_out = float((np.abs(E_out)**2).sum())
        if p_out > 0 and p_in > 0:
            scale = np.sqrt(p_in / p_out)
            E_out = E_out * scale
    elif normalize_output == 'peak':
        a_in = float(np.abs(E_in).max())
        a_out = float(np.abs(E_out).max())
        if a_out > 0 and a_in > 0:
            E_out = E_out * (a_in / a_out)
    elif normalize_output == 'none':
        pass
    elif isinstance(normalize_output, (int, float, complex)):
        E_out = E_out * normalize_output
    else:
        raise ValueError(f"normalize_output={normalize_output!r}; "
                          f"expected 'power', 'peak', 'none', or scalar")

    _progress('done', 1.0,
              f'total {time.perf_counter()-t0:.1f}s')
    return E_out


def _count_multi_indices_4d(max_order: int) -> int:
    """Number of 4-variable multi-indices with total degree <= max_order
    (== C(n+4, 4) for n = max_order).
    """
    from math import comb
    return comb(max_order + 4, 4)


# ---------------------------------------------------------------------------
# Integration method helpers
# ---------------------------------------------------------------------------

def _integrate_quadrature(
    coef_opd, coef_s1x, coef_s1y, mi,
    K1_arr, K2_arr, K3_arr, K4_arr,
    poly_order, G, N_out_coarse,
    u_v2x_samples, u_v2y_samples, tuk_2d, du,
    v2x_h, v2y_h, chunk_v2, inbox_flat,
    sample_E_bilinear,
    use_numexpr, _progress,
):
    """Uniform Tukey-windowed quadrature on the (v2x, v2y) grid."""
    n_v2 = len(u_v2x_samples)
    n_v2_total = n_v2 * n_v2

    Tu3_all  = _chebyshev_vandermonde(u_v2x_samples, poly_order)
    Tu4_all  = _chebyshev_vandermonde(u_v2y_samples, poly_order)
    dTu3_all = _chebyshev_derivative_vandermonde(u_v2x_samples, poly_order)
    dTu4_all = _chebyshev_derivative_vandermonde(u_v2y_samples, poly_order)

    iy_grid, ix_grid = np.meshgrid(np.arange(n_v2), np.arange(n_v2),
                                     indexing='ij')
    v2x_idx = ix_grid.ravel()
    v2y_idx = iy_grid.ravel()

    T3bj  = Tu3_all [K3_arr[:, None], v2x_idx[None, :]]
    T4bj  = Tu4_all [K4_arr[:, None], v2y_idx[None, :]]
    dT3bj = dTu3_all[K3_arr[:, None], v2x_idx[None, :]]
    dT4bj = dTu4_all[K4_arr[:, None], v2y_idx[None, :]]
    T3_T4  = T3bj * T4bj
    dT3_T4 = dT3bj * T4bj
    T3_dT4 = T3bj * dT4bj

    H_opd      = coef_opd[:, None] * T3_T4
    H_s1x      = coef_s1x[:, None] * T3_T4
    H_s1y      = coef_s1y[:, None] * T3_T4
    H_ds1x_du3 = coef_s1x[:, None] * dT3_T4
    H_ds1x_du4 = coef_s1x[:, None] * T3_dT4
    H_ds1y_du3 = coef_s1y[:, None] * dT3_T4
    H_ds1y_du4 = coef_s1y[:, None] * T3_dT4

    weight_per_sample = tuk_2d.ravel() * du * du * (v2x_h * v2y_h)

    if use_numexpr is None:
        use_numexpr = NUMEXPR_AVAILABLE
    use_numexpr = bool(use_numexpr) and NUMEXPR_AVAILABLE
    _progress('integrate', 0.65,
              f'quadrature: {n_v2_total} v2 samples, chunk={chunk_v2}, '
              f'numexpr={use_numexpr}')

    if chunk_v2 <= 0:
        chunk_v2 = n_v2_total
    chunk_v2 = min(chunk_v2, n_v2_total)

    E_out_flat = np.zeros(N_out_coarse * N_out_coarse, dtype=np.complex128)
    t_int_start = time.perf_counter()

    for c_start in range(0, n_v2_total, chunk_v2):
        c_end = min(c_start + chunk_v2, n_v2_total)

        opd_c      = G @ H_opd     [:, c_start:c_end]
        s1x_c      = G @ H_s1x     [:, c_start:c_end]
        s1y_c      = G @ H_s1y     [:, c_start:c_end]
        ds1x_du3_c = G @ H_ds1x_du3[:, c_start:c_end]
        ds1x_du4_c = G @ H_ds1x_du4[:, c_start:c_end]
        ds1y_du3_c = G @ H_ds1y_du3[:, c_start:c_end]
        ds1y_du4_c = G @ H_ds1y_du4[:, c_start:c_end]

        det_J_c = (ds1x_du3_c * ds1y_du4_c
                   - ds1x_du4_c * ds1y_du3_c)
        abs_J_c = np.abs(det_J_c) / (v2x_h * v2y_h)

        Eobj_c = sample_E_bilinear(s1x_c, s1y_c)
        weights_c = weight_per_sample[c_start:c_end]

        if use_numexpr:
            twopi = 2.0 * np.pi
            cos_term = _ne.evaluate("cos(twopi * opd_c)")
            sin_term = _ne.evaluate("sin(twopi * opd_c)")
            Er = Eobj_c.real; Ei = Eobj_c.imag
            contrib_r = _ne.evaluate(
                "(Er*cos_term - Ei*sin_term) * abs_J_c * weights_c")
            contrib_i = _ne.evaluate(
                "(Ei*cos_term + Er*sin_term) * abs_J_c * weights_c")
            contrib_sum = contrib_r.sum(axis=1) + 1j * contrib_i.sum(axis=1)
        else:
            contrib_c = (Eobj_c
                          * np.exp(2j * np.pi * opd_c)
                          * abs_J_c
                          * weights_c)
            contrib_sum = contrib_c.sum(axis=1)

        E_out_flat[inbox_flat] += contrib_sum[inbox_flat]

    t_int = time.perf_counter() - t_int_start
    _progress('integrate', 0.95,
              f'quadrature: {n_v2_total} v2 samples in {t_int:.1f}s '
              f'({"numexpr" if use_numexpr else "numpy"}, '
              f'chunk={chunk_v2})')

    return E_out_flat.reshape(N_out_coarse, N_out_coarse)


def _integrate_stationary_phase(
    coef_opd, coef_s1x, coef_s1y, mi,
    K1_arr, K2_arr, K3_arr, K4_arr,
    poly_order, G, N_out_coarse,
    u_s2x_out, u_s2y_out, inbox_flat,
    v2x_c, v2y_c, v2x_h, v2y_h,
    sample_E_bilinear,
    newton_iter, newton_tol,
    _progress, verbose,
):
    """Leading-order stationary-phase (Gaussian-moment) evaluation of
    the Maslov integral.

    For each output pixel s2, Newton-iterate on the fitted OPD
    polynomial to locate the stationary point v2* where
    grad_{v2} OPD(s2, v2*) = 0.  Evaluate the saddle-point formula

        E(s2) ~ |det J_{s1,v2}| * E_obj(s1(s2, v2*))
              * exp(2 pi i * OPD*)
              * exp(i pi sigma/4) / sqrt(|det H_{v2,v2} OPD*|)

    where H is the 2x2 Hessian of OPD w.r.t. v2 and sigma is its
    signature (number of positive eigenvalues minus number negative).

    The Newton iteration runs fully vectorised over all output pixels
    in normalised (u_v2x, u_v2y) coordinates, so convergence is as
    fast as the slowest pixel needs.
    """
    M = len(mi)
    t_int_start = time.perf_counter()
    _progress('integrate', 0.65,
              f'stationary-phase Newton ({newton_iter} max iters)')

    N_px = N_out_coarse * N_out_coarse

    # --- Build per-pixel (u_s2x, u_s2y) arrays ---------------------
    u_s2x_flat = u_s2x_out.ravel()   # (N_px,)
    u_s2y_flat = u_s2y_out.ravel()

    # Initial guess: v2 = 0 (chief ray direction) in normalised coords
    u_v2x = np.zeros(N_px, dtype=np.float64)
    u_v2y = np.zeros(N_px, dtype=np.float64)

    # ---------------------------------------------------------------
    # Helper: evaluate OPD, its gradient, and its Hessian at
    # (u_s2x_flat, u_s2y_flat, u_v2x, u_v2y) using the polynomial
    # coefficients.  All arrays have length N_px.
    # ---------------------------------------------------------------
    def _opd_and_derivs(coef, u1, u2, u3, u4):
        """Return (f, df/du3, df/du4, d2f/du3^2, d2f/du3du4, d2f/du4^2)
        each of shape (N_px,)."""
        # Per-pixel Vandermondes
        T1 = _chebyshev_vandermonde(u1, poly_order)    # (K+1, N_px)
        T2 = _chebyshev_vandermonde(u2, poly_order)
        T3 = _chebyshev_vandermonde(u3, poly_order)
        T4 = _chebyshev_vandermonde(u4, poly_order)
        dT3 = _chebyshev_derivative_vandermonde(u3, poly_order)
        dT4 = _chebyshev_derivative_vandermonde(u4, poly_order)
        d2T3 = _chebyshev_second_derivative_vandermonde(u3, poly_order)
        d2T4 = _chebyshev_second_derivative_vandermonde(u4, poly_order)
        # Gather basis values per multi-index
        T1b = T1[K1_arr]    # (M, N_px)
        T2b = T2[K2_arr]
        T3b = T3[K3_arr]
        T4b = T4[K4_arr]
        dT3b = dT3[K3_arr]
        dT4b = dT4[K4_arr]
        d2T3b = d2T3[K3_arr]
        d2T4b = d2T4[K4_arr]

        T12 = T1b * T2b
        # Sum over basis axis (M) with coefficient weights
        c = coef[:, None]    # (M, 1)
        f        = np.sum(c * T12 * T3b  * T4b , axis=0)
        df_du3   = np.sum(c * T12 * dT3b * T4b , axis=0)
        df_du4   = np.sum(c * T12 * T3b  * dT4b, axis=0)
        d2f_33   = np.sum(c * T12 * d2T3b* T4b , axis=0)
        d2f_44   = np.sum(c * T12 * T3b  * d2T4b, axis=0)
        d2f_34   = np.sum(c * T12 * dT3b * dT4b, axis=0)
        return f, df_du3, df_du4, d2f_33, d2f_34, d2f_44

    # ---------------------------------------------------------------
    # Newton iterate on grad_{u_v2} OPD = 0
    # ---------------------------------------------------------------
    converged_mask = np.zeros(N_px, dtype=bool)
    converged_mask[~inbox_flat] = True  # skip out-of-box pixels

    for it in range(newton_iter):
        if converged_mask.all():
            break
        active = ~converged_mask
        u1 = u_s2x_flat[active]
        u2 = u_s2y_flat[active]
        u3 = u_v2x[active]
        u4 = u_v2y[active]
        _, g3, g4, H33, H34, H44 = _opd_and_derivs(
            coef_opd, u1, u2, u3, u4)
        # 2x2 Newton step: H * dv = -g
        det_H = H33 * H44 - H34 * H34
        det_safe = np.where(np.abs(det_H) < 1e-30,
                             np.sign(det_H) * 1e-30 + 1e-30, det_H)
        dv3 = -(H44 * g3 - H34 * g4) / det_safe
        dv4 = -(-H34 * g3 + H33 * g4) / det_safe
        # Damp the step if it would leave [-1, 1]
        step_limit = 0.5
        step_size = np.sqrt(dv3**2 + dv4**2)
        damp = np.where(step_size > step_limit,
                         step_limit / np.maximum(step_size, 1e-30),
                         1.0)
        dv3 *= damp
        dv4 *= damp
        # Update
        u_v2x_new = u_v2x[active] + dv3
        u_v2y_new = u_v2y[active] + dv4
        # Clamp to [-1, 1]
        u_v2x_new = np.clip(u_v2x_new, -1.0, 1.0)
        u_v2y_new = np.clip(u_v2y_new, -1.0, 1.0)
        u_v2x[active] = u_v2x_new
        u_v2y[active] = u_v2y_new
        # Check convergence (gradient magnitude)
        grad_mag = np.sqrt(g3**2 + g4**2)
        newly = np.zeros(N_px, dtype=bool)
        newly[active] = grad_mag < newton_tol
        converged_mask |= newly
        if verbose and (it == 0 or it == newton_iter - 1 or
                         it % max(1, newton_iter // 4) == 0):
            n_conv = converged_mask.sum()
            _progress('integrate', 0.65 + 0.15 * it / newton_iter,
                      f'Newton iter {it+1}/{newton_iter}, '
                      f'{n_conv}/{N_px} pixels converged '
                      f'(max grad {grad_mag.max():.2e})')

    # ---------------------------------------------------------------
    # Evaluate the stationary-phase formula at (u_v2x, u_v2y)
    # ---------------------------------------------------------------
    _progress('integrate', 0.85, 'evaluating saddle-point formula')

    opd_star, g3, g4, H33, H34, H44 = _opd_and_derivs(
        coef_opd, u_s2x_flat, u_s2y_flat, u_v2x, u_v2y)
    s1x_star, ds1x_du3, ds1x_du4, _, _, _ = _opd_and_derivs(
        coef_s1x, u_s2x_flat, u_s2y_flat, u_v2x, u_v2y)
    s1y_star, ds1y_du3, ds1y_du4, _, _, _ = _opd_and_derivs(
        coef_s1y, u_s2x_flat, u_s2y_flat, u_v2x, u_v2y)

    # Jacobian of the s1 <-> v2 map at v2*
    det_J_norm = ds1x_du3 * ds1y_du4 - ds1x_du4 * ds1y_du3
    abs_J = np.abs(det_J_norm) / (v2x_h * v2y_h)

    # Hessian of OPD w.r.t. physical v2 = (v2x_c + v2x_h * u_v2x, ...)
    # Chain rule: d/dv2x = (1/v2x_h) d/du_v2x.  So the Hessian in
    # physical v2 is H_phys = diag(1/v2x_h, 1/v2y_h) H_norm diag(1/v2x_h, 1/v2y_h).
    H33_phys = H33 / (v2x_h * v2x_h)
    H34_phys = H34 / (v2x_h * v2y_h)
    H44_phys = H44 / (v2y_h * v2y_h)
    det_H_phys = H33_phys * H44_phys - H34_phys * H34_phys
    # Signature: +2 if both eigenvalues positive, -2 if both negative,
    # 0 if saddle (eigenvalues of opposite sign).
    trace_H = H33_phys + H44_phys
    # For 2x2 sym: eigenvalues = 0.5*(trace +/- sqrt(trace^2 - 4 det))
    # If det > 0 and trace > 0 -> both positive (sig=+2)
    # If det > 0 and trace < 0 -> both negative (sig=-2)
    # If det < 0 -> opposite signs (sig=0)
    sig = np.where(det_H_phys > 0,
                    np.where(trace_H > 0, 2, -2),
                    0)
    # Stationary-phase prefactor: includes the 2pi scaling of the
    # quadratic form.  For phi(v2) = 2pi * OPD(v2), the second
    # derivative of phi is 2pi * H_phys, so det(phi'') = (2pi)^2 *
    # det(H_phys).  The 2pi Gaussian integral of
    # exp(i * (1/2) x^T A x) d^2x equals 2pi / sqrt(|det A|) *
    # exp(i pi sig/4).  Therefore the amplitude factor is
    #     2pi / (2pi * sqrt(|det H_phys|)) = 1 / sqrt(|det H_phys|)
    amp_sp = 1.0 / np.sqrt(np.maximum(np.abs(det_H_phys), 1e-300))

    # Phase from Maslov signature
    phase_sp = np.exp(1j * (np.pi / 4.0) * sig)

    # Sample input field at the stationary s1
    Eobj_star = sample_E_bilinear(s1x_star, s1y_star)

    E_flat = (Eobj_star
              * np.exp(2j * np.pi * opd_star)
              * abs_J
              * amp_sp
              * phase_sp)

    # Zero the pixels that never converged (rough safety)
    not_conv = ~converged_mask
    if not_conv.any():
        E_flat[not_conv] = 0.0
        if verbose:
            _progress('integrate', 0.92,
                      f'{not_conv.sum()}/{N_px} pixels did not converge, '
                      f'zeroed')

    # Zero out-of-box pixels
    E_flat[~inbox_flat] = 0.0

    t_int = time.perf_counter() - t_int_start
    _progress('integrate', 0.95,
              f'stationary_phase: {N_px} pixels in {t_int:.1f}s')

    return E_flat.reshape(N_out_coarse, N_out_coarse)


def _integrate_local_quadrature(
    coef_opd, coef_s1x, coef_s1y, mi,
    K1_arr, K2_arr, K3_arr, K4_arr,
    poly_order, G, N_out_coarse,
    u_s2x_out, u_s2y_out, inbox_flat,
    v2x_c, v2y_c, v2x_h, v2y_h,
    sample_E_bilinear,
    newton_iter, newton_tol,
    n_samples, window_sigma,
    _progress, verbose,
):
    """Hybrid stationary-phase + local quadrature.

    Per output pixel:
      1. Newton to locate v2* where grad_v2 OPD = 0     (cost ~ stationary phase).
      2. Compute Hessian H_phys at v2*; extract eigenvalues lambda_1, lambda_2.
      3. Define local sampling scales sigma_k = 1 / sqrt(2 pi |lambda_k|)
         -- the natural Gaussian waist of the leading-order stationary-
         phase integrand.
      4. Sample a (n_samples x n_samples) uniform grid on
            v2 in [v2* - window_sigma*sigma_k, v2* + window_sigma*sigma_k]
         in each Hessian eigendirection.
      5. Evaluate the full polynomial integrand at every sample and
         accumulate with uniform-quadrature weights (times the sample
         pitch).  No stationary-phase prefactor is applied -- the
         Gaussian factor is sampled directly, so leading-order SP is
         an automatic limit when n_samples is large enough.

    This is the "beyond leading-order" extension of stationary phase.
    It captures asymptotic corrections (cubic+quartic OPD, first-order
    Eobj gradient) that leading-order stationary phase truncates.

    NUFFT connection:  the local-quadrature step is a direct numerical
    evaluation of what OPDGPU's small_waist_asymptotic paper writes as
    a closed-form Wick-contracted Gaussian moment sum.  A genuine
    NUFFT-accelerated path would replace the uniform quadrature with
    analytic moment evaluation against Laguerre-Gaussian basis
    functions; this is a stricter-accuracy, more-complex alternative.
    """
    M = len(mi)
    t_int_start = time.perf_counter()
    _progress('integrate', 0.60,
              f'local_quadrature: Newton phase ({newton_iter} max iters)')

    N_px = N_out_coarse * N_out_coarse
    u_s2x_flat = u_s2x_out.ravel()
    u_s2y_flat = u_s2y_out.ravel()

    u_v2x = np.zeros(N_px, dtype=np.float64)
    u_v2y = np.zeros(N_px, dtype=np.float64)

    def _opd_and_derivs(coef, u1, u2, u3, u4):
        T1 = _chebyshev_vandermonde(u1, poly_order)
        T2 = _chebyshev_vandermonde(u2, poly_order)
        T3 = _chebyshev_vandermonde(u3, poly_order)
        T4 = _chebyshev_vandermonde(u4, poly_order)
        dT3 = _chebyshev_derivative_vandermonde(u3, poly_order)
        dT4 = _chebyshev_derivative_vandermonde(u4, poly_order)
        d2T3 = _chebyshev_second_derivative_vandermonde(u3, poly_order)
        d2T4 = _chebyshev_second_derivative_vandermonde(u4, poly_order)
        T1b = T1[K1_arr]; T2b = T2[K2_arr]
        T3b = T3[K3_arr]; T4b = T4[K4_arr]
        dT3b = dT3[K3_arr]; dT4b = dT4[K4_arr]
        d2T3b = d2T3[K3_arr]; d2T4b = d2T4[K4_arr]
        T12 = T1b * T2b
        c = coef[:, None]
        f        = np.sum(c * T12 * T3b  * T4b , axis=0)
        df_du3   = np.sum(c * T12 * dT3b * T4b , axis=0)
        df_du4   = np.sum(c * T12 * T3b  * dT4b, axis=0)
        d2f_33   = np.sum(c * T12 * d2T3b* T4b , axis=0)
        d2f_44   = np.sum(c * T12 * T3b  * d2T4b, axis=0)
        d2f_34   = np.sum(c * T12 * dT3b * dT4b, axis=0)
        return f, df_du3, df_du4, d2f_33, d2f_34, d2f_44

    # Step 1-2: Newton to locate v2* (reuse logic from stationary_phase)
    converged = np.zeros(N_px, dtype=bool)
    converged[~inbox_flat] = True
    for it in range(newton_iter):
        if converged.all():
            break
        active = ~converged
        u1 = u_s2x_flat[active]
        u2 = u_s2y_flat[active]
        u3 = u_v2x[active]
        u4 = u_v2y[active]
        _, g3, g4, H33, H34, H44 = _opd_and_derivs(coef_opd, u1, u2, u3, u4)
        det_H = H33 * H44 - H34 * H34
        det_safe = np.where(np.abs(det_H) < 1e-30,
                             np.sign(det_H) * 1e-30 + 1e-30, det_H)
        dv3 = -(H44 * g3 - H34 * g4) / det_safe
        dv4 = -(-H34 * g3 + H33 * g4) / det_safe
        step_size = np.sqrt(dv3 ** 2 + dv4 ** 2)
        damp = np.where(step_size > 0.5,
                         0.5 / np.maximum(step_size, 1e-30), 1.0)
        dv3 *= damp; dv4 *= damp
        u_v2x[active] = np.clip(u_v2x[active] + dv3, -1.0, 1.0)
        u_v2y[active] = np.clip(u_v2y[active] + dv4, -1.0, 1.0)
        grad_mag = np.sqrt(g3 ** 2 + g4 ** 2)
        newly = np.zeros(N_px, dtype=bool)
        newly[active] = grad_mag < newton_tol
        converged |= newly

    # Hessian at v2* for local sampling scale
    _progress('integrate', 0.72, 'computing Hessian eigen-scales')
    _, _, _, H33, H34, H44 = _opd_and_derivs(
        coef_opd, u_s2x_flat, u_s2y_flat, u_v2x, u_v2y)
    # Physical-v2 Hessian (chain rule from normalized)
    H33_phys = H33 / (v2x_h ** 2)
    H34_phys = H34 / (v2x_h * v2y_h)
    H44_phys = H44 / (v2y_h ** 2)
    # 2x2 eigendecomposition: tau = trace, det = determinant
    tau = H33_phys + H44_phys
    detH = H33_phys * H44_phys - H34_phys ** 2
    disc = np.maximum(tau ** 2 / 4.0 - detH, 0.0)
    sqrt_disc = np.sqrt(disc)
    lam1 = tau / 2.0 + sqrt_disc
    lam2 = tau / 2.0 - sqrt_disc
    # Natural Gaussian waists in physical v2 units
    sigma1_phys = 1.0 / np.sqrt(np.maximum(np.abs(lam1), 1e-30) * np.pi)
    sigma2_phys = 1.0 / np.sqrt(np.maximum(np.abs(lam2), 1e-30) * np.pi)
    # In normalized coords:
    sigma1_norm = sigma1_phys / v2x_h   # rough -- uses axis-aligned scaling
    sigma2_norm = sigma2_phys / v2y_h
    # Keep eigenvector orientation simple (axis-aligned) for the local
    # sampling.  Treating H as diagonal is a leading-order simplification
    # that's exact when H34 = 0; the orthogonal-sampling error is
    # O(H34 / sqrt(H33 * H44)) which is small for well-conditioned
    # Hessians of near-circular lens apertures.  Full eigen-rotation
    # could be added later if needed.

    # Step 3-4: Local uniform sampling grid around v2*
    _progress('integrate', 0.75,
              f'local uniform sampling: {n_samples}x{n_samples} pts, '
              f'window={window_sigma}sigma')
    lin = np.linspace(-window_sigma, window_sigma, n_samples)
    dxi = lin[1] - lin[0]
    Xlin, Ylin = np.meshgrid(lin, lin, indexing='xy')   # (n_s, n_s)
    Xlin_flat = Xlin.ravel()
    Ylin_flat = Ylin.ravel()   # (n_s^2,)

    # Broadcasted sample positions in normalized coords
    # u_v2x_samples[pixel, node] = u_v2x[pixel] + sigma1_norm[pixel] * Xlin[node]
    # Shapes: (N_px, 1) + (N_px, 1) * (1, n_s^2) -> (N_px, n_s^2)
    u_v2x_samp = (u_v2x[:, None]
                   + (sigma1_norm[:, None]) * Xlin_flat[None, :])
    u_v2y_samp = (u_v2y[:, None]
                   + (sigma2_norm[:, None]) * Ylin_flat[None, :])
    # Clamp to fit box
    u_v2x_samp = np.clip(u_v2x_samp, -1.0, 1.0)
    u_v2y_samp = np.clip(u_v2y_samp, -1.0, 1.0)

    # For polynomial evaluations we tile s2 across samples:
    # u_s2x_pix[pixel, node] = u_s2x_flat[pixel]  (broadcast)
    n_s2 = n_samples * n_samples
    u_s2x_tile = np.broadcast_to(u_s2x_flat[:, None], (N_px, n_s2))
    u_s2y_tile = np.broadcast_to(u_s2y_flat[:, None], (N_px, n_s2))

    # Integrand evaluation: do it chunked over pixel rows to manage memory
    _progress('integrate', 0.78,
              f'evaluating integrand on {N_px*n_s2:,} (pixel,sample) pairs')

    E_flat = np.zeros(N_px, dtype=np.complex128)
    # Uniform-quadrature weight:
    # physical d^2v2 = (sigma1_phys * dxi) * (sigma2_phys * dxi)
    w2d_phys = (sigma1_phys * sigma2_phys) * (dxi ** 2)
    # w2d_phys has shape (N_px,)

    # Chunk over pixels to cap memory
    PX_CHUNK = max(1, min(N_px, 1024 * 64 // max(1, n_s2 // 16)))
    # target ~1 GB per chunk at n_s^2 * PX_CHUNK complex
    for p_start in range(0, N_px, PX_CHUNK):
        p_end = min(p_start + PX_CHUNK, N_px)
        u3 = u_v2x_samp[p_start:p_end].ravel()        # (chunk * n_s^2,)
        u4 = u_v2y_samp[p_start:p_end].ravel()
        u1 = u_s2x_tile[p_start:p_end].ravel()
        u2 = u_s2y_tile[p_start:p_end].ravel()
        opd_v, _, _, _, _, _        = _opd_and_derivs(coef_opd, u1, u2, u3, u4)
        s1x_v, ds1x_du3, ds1x_du4, *_ = _opd_and_derivs(coef_s1x, u1, u2, u3, u4)
        s1y_v, ds1y_du3, ds1y_du4, *_ = _opd_and_derivs(coef_s1y, u1, u2, u3, u4)
        det_J = ds1x_du3 * ds1y_du4 - ds1x_du4 * ds1y_du3
        abs_J = np.abs(det_J) / (v2x_h * v2y_h)

        Eobj_v = sample_E_bilinear(s1x_v, s1y_v)

        contrib = (Eobj_v
                    * np.exp(2j * np.pi * opd_v)
                    * abs_J)
        # Reshape back to (chunk, n_s^2) and sum over samples axis,
        # applying the per-pixel physical quadrature weight.
        contrib_r = contrib.reshape(p_end - p_start, n_s2)
        E_flat[p_start:p_end] = contrib_r.sum(axis=1) * \
                                  w2d_phys[p_start:p_end]
        if verbose and (p_start % (PX_CHUNK * 8) == 0):
            _progress('integrate',
                      0.78 + 0.15 * (p_end / N_px),
                      f'pixel chunk {p_end}/{N_px}')

    # Zero non-converged / out-of-box pixels
    E_flat[~converged] = 0.0
    E_flat[~inbox_flat] = 0.0

    t_int = time.perf_counter() - t_int_start
    _progress('integrate', 0.95,
              f'local_quadrature: {N_px} pixels, '
              f'{n_s2} samples/pixel, {t_int:.1f}s')

    return E_flat.reshape(N_out_coarse, N_out_coarse)
