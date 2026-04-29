"""
Beam source and mode generators for optical propagation simulations.

This module provides functions to create common laser beam profiles on a
discrete 2-D grid:

- Fundamental Gaussian beams (with optional GPU acceleration via CuPy)
- Hermite-Gaussian (HG_mn) modes
- Laguerre-Gaussian (LG_pl) modes

All fields are returned at the beam waist (flat phase) and are suitable for
use as input to angular-spectrum or other propagation routines.

Author: Andrew Traverso
"""

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    # Sentinel so ``xp is cp`` checks below don't NameError when cupy
    # isn't installed.
    cp = None
    CUPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fundamental Gaussian beam
# ---------------------------------------------------------------------------

def create_gaussian_beam(N, dx, sigma, wavelength=None, x0=0, y0=0, use_gpu=False):
    """
    Create a Gaussian beam field.

    Parameters
    ----------
    N : int or tuple
        Grid size. If int, creates an N x N grid. If tuple, interpreted as
        (Ny, Nx).
    dx : float
        Grid spacing [m].
    sigma : float
        Gaussian width parameter (field standard deviation) [m].
        The 1/e field amplitude radius is sigma * sqrt(2).
        The 1/e^2 intensity radius (beam waist w0) is also sigma * sqrt(2).
    wavelength : float, optional
        Reserved for future use (e.g. adding a spherical phase for a
        focused beam). Currently unused -- the returned field has flat phase.
    x0, y0 : float, default 0
        Center position of the beam [m].
    use_gpu : bool, default False
        If True and CuPy is available, create the arrays on the GPU.

    Returns
    -------
    E : ndarray, complex
        Gaussian beam field (Ny x Nx).
    x : ndarray
        1-D x-coordinate array [m].
    y : ndarray
        1-D y-coordinate array [m].
    """
    if CUPY_AVAILABLE and use_gpu:
        xp = cp
    else:
        xp = np

    if isinstance(N, int):
        Ny, Nx = N, N
    else:
        Ny, Nx = N

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dx
    X, Y = xp.meshgrid(x, y)

    # Gaussian amplitude: exp(-r^2 / (2 sigma^2))
    E = xp.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    E = E.astype(complex)

    return E, x, y


# ---------------------------------------------------------------------------
# Hermite-Gaussian modes
# ---------------------------------------------------------------------------

def hermite_physicist(n, x):
    """
    Evaluate the physicist's Hermite polynomial H_n(x) via recurrence.

    Uses the three-term recurrence relation:

        H_0(x) = 1
        H_1(x) = 2x
        H_k(x) = 2x H_{k-1}(x) - 2(k-1) H_{k-2}(x)

    Parameters
    ----------
    n : int
        Polynomial order (>= 0).
    x : ndarray
        Points at which to evaluate H_n.

    Returns
    -------
    H_n : ndarray
        Values of the physicist's Hermite polynomial of order *n*.
    """
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        H_prev2 = np.ones_like(x)
        H_prev1 = 2 * x
        for k in range(2, n + 1):
            H_curr = 2 * x * H_prev1 - 2 * (k - 1) * H_prev2
            H_prev2 = H_prev1
            H_prev1 = H_curr
        return H_curr


def create_hermite_gauss(N, dx, w0, wavelength, m=0, n=0, x0=0, y0=0):
    """
    Create a Hermite-Gaussian (HG_mn) beam mode at the waist.

    Parameters
    ----------
    N : int
        Grid size (N x N).
    dx : float
        Grid spacing [m].
    w0 : float
        Beam waist (1/e^2 intensity radius) [m].
    wavelength : float
        Wavelength [m]. Currently unused -- the field is returned at the
        waist with flat phase.
    m, n : int, default 0
        Transverse mode indices. HG_00 is the fundamental Gaussian.
    x0, y0 : float, default 0
        Beam center [m].

    Returns
    -------
    E : ndarray, complex (N x N)
        Hermite-Gaussian mode field, power-normalised.
    x : ndarray
        1-D x-coordinate array [m].
    y : ndarray
        1-D y-coordinate array [m].

    Notes
    -----
    The (un-normalised) field is

        E_mn(x, y) = H_m(sqrt(2) x / w0) * H_n(sqrt(2) y / w0)
                      * exp(-(x^2 + y^2) / w0^2)

    where H_m is the physicist's Hermite polynomial of order m.
    """
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, y)

    u = np.sqrt(2) * (X - x0) / w0
    v = np.sqrt(2) * (Y - y0) / w0

    Hm = hermite_physicist(m, u)
    Hn = hermite_physicist(n, v)

    gaussian = np.exp(-((X - x0)**2 + (Y - y0)**2) / w0**2)
    E = (Hm * Hn * gaussian).astype(complex)

    # Power-normalise
    norm = np.sqrt(np.sum(np.abs(E)**2) * dx**2)
    if norm > 0:
        E /= norm

    return E, x, y


# ---------------------------------------------------------------------------
# Laguerre-Gaussian modes
# ---------------------------------------------------------------------------

def laguerre_generalized(p, l_abs, x):
    """
    Evaluate the generalized Laguerre polynomial L_p^l(x) via recurrence.

    Uses the three-term recurrence relation:

        L_0^l(x) = 1
        L_1^l(x) = 1 + l - x
        L_k^l(x) = ((2k - 1 + l - x) L_{k-1}^l(x)
                     - (k - 1 + l) L_{k-2}^l(x)) / k

    Parameters
    ----------
    p : int
        Polynomial order (radial index, >= 0).
    l_abs : int
        Associated (generalized) index (|l|, >= 0).
    x : ndarray
        Points at which to evaluate L_p^{l_abs}.

    Returns
    -------
    L_p : ndarray
        Values of the generalized Laguerre polynomial.
    """
    if p == 0:
        return np.ones_like(x)
    elif p == 1:
        return 1 + l_abs - x
    else:
        L_prev2 = np.ones_like(x)
        L_prev1 = 1 + l_abs - x
        for k in range(2, p + 1):
            L_curr = ((2 * k - 1 + l_abs - x) * L_prev1
                      - (k - 1 + l_abs) * L_prev2) / k
            L_prev2 = L_prev1
            L_prev1 = L_curr
        return L_curr


def create_laguerre_gauss(N, dx, w0, wavelength, p=0, l=0, x0=0, y0=0):
    """
    Create a Laguerre-Gaussian (LG_pl) beam mode at the waist.

    Parameters
    ----------
    N : int
        Grid size (N x N).
    dx : float
        Grid spacing [m].
    w0 : float
        Beam waist (1/e^2 intensity radius) [m].
    wavelength : float
        Wavelength [m]. Currently unused -- the field is returned at the
        waist with flat phase.
    p : int, default 0
        Radial index (number of radial nodes).
    l : int, default 0
        Azimuthal index (topological charge / orbital angular momentum).
        LG_00 is the fundamental Gaussian.
    x0, y0 : float, default 0
        Beam center [m].

    Returns
    -------
    E : ndarray, complex (N x N)
        Laguerre-Gaussian mode field, power-normalised.
    x : ndarray
        1-D x-coordinate array [m].
    y : ndarray
        1-D y-coordinate array [m].

    Notes
    -----
    The (un-normalised) field is

        E_pl(r, theta) = (r sqrt(2) / w0)^|l| * L_p^|l|(2 r^2 / w0^2)
                          * exp(-r^2 / w0^2) * exp(i l theta)

    where L_p^|l| is the generalized Laguerre polynomial.
    """
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, y)

    r = np.sqrt((X - x0)**2 + (Y - y0)**2)
    theta = np.arctan2(Y - y0, X - x0)

    rho = np.sqrt(2) * r / w0
    rho_sq = rho**2

    # Generalized Laguerre polynomial L_p^|l|
    L = laguerre_generalized(p, abs(l), rho_sq)

    gaussian = np.exp(-r**2 / w0**2)
    E = (rho**abs(l) * L * gaussian * np.exp(1j * l * theta))

    # Power-normalise
    norm = np.sqrt(np.sum(np.abs(E)**2) * dx**2)
    if norm > 0:
        E /= norm

    return E, x, y


# ---------------------------------------------------------------------------
# Off-axis / tilted plane-wave sources
# ---------------------------------------------------------------------------

def create_tilted_plane_wave(N, dx, wavelength, angle_x=0.0, angle_y=0.0,
                             amplitude=1.0, dy=None):
    """Create a tilted (off-axis) plane wave on an N x N grid.

    A tilted plane wave has a linear phase ramp across the pupil,
    representing a collimated beam arriving from a direction offset
    from the optical axis by ``angle_x`` (horizontal) and ``angle_y``
    (vertical).  This is the standard source for evaluating off-axis
    imaging performance -- pass it through ``apply_real_lens`` and
    compare the exit-pupil OPD or PSF to the on-axis case.

    Parameters
    ----------
    N : int
        Grid dimension (square N x N).
    dx : float
        Grid spacing in x [m].
    wavelength : float
        Vacuum wavelength [m].
    angle_x : float, default 0
        Field angle in the x-z plane [rad].  Positive = source
        tilted toward +x.
    angle_y : float, default 0
        Field angle in the y-z plane [rad].  Positive = source
        tilted toward +y.
    amplitude : float, default 1
        Uniform amplitude.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.

    Returns
    -------
    E : ndarray, complex, shape (N, N)
        Complex field on the grid.
    x, y : ndarray
        1-D coordinate arrays [m].
    """
    if dy is None:
        dy = dx
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = np.meshgrid(x, y)
    k0 = 2 * np.pi / wavelength
    phase = k0 * (np.sin(angle_x) * X + np.sin(angle_y) * Y)
    E = amplitude * np.exp(1j * phase)
    return E, x, y


def create_point_source(N, dx, wavelength, x0=0.0, y0=0.0, z0=0.0,
                        amplitude=1.0, dy=None):
    """Create a diverging spherical wave from a point source at
    ``(x0, y0, z0)`` evaluated at z=0.

    For ``z0 < 0`` the source is *before* the grid (diverging);
    for ``z0 > 0`` it is *after* (converging).  ``z0 = 0`` gives
    a delta at (x0, y0).

    Parameters
    ----------
    N, dx, wavelength : usual
    x0, y0 : float
        Transverse position of the point source [m].
    z0 : float
        Axial position of the point source [m] relative to the
        grid plane at z=0.  Negative = source before grid (diverging).
    amplitude : float
    dy : float, optional

    Returns
    -------
    E : ndarray, complex, shape (N, N)
    x, y : ndarray
    """
    if dy is None:
        dy = dx
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = np.meshgrid(x, y)
    k0 = 2 * np.pi / wavelength
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2 + z0 ** 2)
    r = np.maximum(r, 1e-30)
    E = amplitude * np.exp(1j * k0 * r) / r
    return E, x, y


def create_multi_field_sources(N, dx, wavelength, field_angles,
                               amplitude=1.0, dy=None):
    """Generate a list of tilted plane waves at the given field angles.

    Convenience wrapper around :func:`create_tilted_plane_wave` for
    setting up multi-field analyses.

    Parameters
    ----------
    N, dx, wavelength : usual
    field_angles : sequence of float or tuple
        Each element is either a scalar (y-tilt only) or a
        ``(angle_x, angle_y)`` tuple.
    amplitude : float
    dy : float, optional

    Returns
    -------
    sources : list of (E, angle_x, angle_y)
        One per field angle.  **Note the return shape differs from
        the scalar ``create_*`` helpers** (which return ``(E, x, y)``
        directly): this is a *list of tilted-plane-wave sources*,
        not a single field.
    x, y : ndarray
        Shared 1-D coordinate arrays.
    """
    sources = []
    x = y = None
    for a in field_angles:
        if isinstance(a, (list, tuple)):
            ax, ay = float(a[0]), float(a[1])
        else:
            ax, ay = 0.0, float(a)
        E, x, y = create_tilted_plane_wave(
            N, dx, wavelength, angle_x=ax, angle_y=ay,
            amplitude=amplitude, dy=dy)
        sources.append((E, ax, ay))
    return sources, x, y


# ---------------------------------------------------------------------------
# Extended source models (LED, fiber, top-hat, annular, Bessel)
# ---------------------------------------------------------------------------

def create_top_hat_beam(N, dx, diameter, wavelength=None, x0=0, y0=0):
    """Uniform-intensity circular beam (top-hat / flat-top).

    Parameters
    ----------
    N, dx : int, float
    diameter : float
        Beam diameter [m].
    wavelength : float, optional (reserved)
    x0, y0 : float
        Center [m].

    Returns
    -------
    E, x, y : ndarray
    """
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    E = np.where(r <= diameter / 2, 1.0, 0.0).astype(np.complex128)
    norm = np.sqrt(np.sum(np.abs(E) ** 2) * dx ** 2)
    if norm > 0:
        E /= norm
    return E, x, y


def create_annular_beam(N, dx, outer_diameter, inner_diameter,
                        wavelength=None, x0=0, y0=0):
    """Annular (donut) beam.

    Parameters
    ----------
    N, dx : int, float
    outer_diameter, inner_diameter : float [m]

    Returns
    -------
    E, x, y : ndarray
    """
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    E = np.where((r <= outer_diameter / 2) & (r >= inner_diameter / 2),
                  1.0, 0.0).astype(np.complex128)
    norm = np.sqrt(np.sum(np.abs(E) ** 2) * dx ** 2)
    if norm > 0:
        E /= norm
    return E, x, y


def create_fiber_mode(N, dx, mode_field_diameter, wavelength,
                      x0=0, y0=0, na=0.12):
    """Single-mode fiber output (Gaussian with NA-defined divergence).

    The mode-field diameter (MFD) is the 1/e^2 intensity diameter.
    The field is a Gaussian with w0 = MFD/2, and the NA is encoded
    in the far-field divergence angle.

    Parameters
    ----------
    N, dx : int, float
    mode_field_diameter : float [m]
    wavelength : float [m]
    x0, y0 : float
    na : float
        Fiber numerical aperture (informational; the near-field
        profile is MFD-determined).

    Returns
    -------
    E, x, y : ndarray
    """
    w0 = mode_field_diameter / 2.0
    sigma = w0 / np.sqrt(2)
    return create_gaussian_beam(N, dx, sigma, wavelength=wavelength,
                                 x0=x0, y0=y0)


def create_led_source(N, dx, diameter, divergence_angle,
                      wavelength, x0=0, y0=0):
    """Lambertian LED source (incoherent; returns the intensity
    envelope as a complex field for use with partial-coherence
    imaging).

    The spatial extent is a uniform disk of given diameter; the
    angular extent (divergence) determines how many source angles
    to sample when using ``koehler_image`` or
    ``extended_source_image``.

    Parameters
    ----------
    N, dx : int, float
    diameter : float [m]
        Emitting area diameter.
    divergence_angle : float [rad]
        Half-angle of the emission cone.
    wavelength : float [m]

    Returns
    -------
    E : ndarray (complex)
        Amplitude envelope (uniform inside disk, zero outside).
    source_angles : list of (float, float)
        Suggested source angles for partial-coherence integration,
        covering the divergence cone with ~21 samples.
    x, y : ndarray
    """
    E, x, y = create_top_hat_beam(N, dx, diameter, wavelength, x0, y0)
    # Generate suggested source angles
    n_ring = 3
    angles = [(0.0, 0.0)]
    for ring in range(1, n_ring + 1):
        r = divergence_angle * ring / n_ring
        for k in range(6 * ring):
            theta = 2 * np.pi * k / (6 * ring)
            angles.append((r * np.cos(theta), r * np.sin(theta)))
    return E, angles, x, y


def create_bessel_beam(N, dx, wavelength, cone_angle, x0=0, y0=0):
    """Ideal Bessel beam (J_0 profile).

    Creates the field proportional to J_0(k_r * r) where
    k_r = k * sin(cone_angle).  This is an idealized non-diffracting
    beam; in practice it's produced by an axicon or annular aperture.

    Parameters
    ----------
    N, dx : int, float
    wavelength : float [m]
    cone_angle : float [rad]
        Half-angle of the Bessel cone.

    Returns
    -------
    E, x, y : ndarray
    """
    from scipy.special import j0

    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    k_r = 2 * np.pi / wavelength * np.sin(cone_angle)
    E = j0(k_r * r).astype(np.complex128)
    return E, x, y
