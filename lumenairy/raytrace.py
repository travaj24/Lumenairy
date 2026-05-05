"""
Sequential geometric ray tracer.

A vectorised 3-D sequential ray tracer that works with the same lens
prescription format used by :func:`lenses.apply_real_lens`.  It provides:

* Exact Snell's-law refraction and reflection at conic/aspheric surfaces
  (matching the Zemax standard sag equation in :func:`lenses.surface_sag_general`).
* ABCD (paraxial ray) matrix extraction from traced marginal and chief rays.
* Third-order (Seidel) aberration coefficient computation.
* Spot diagram, ray fan, and wavefront analysis utilities.
* Ray generation helpers: fans, grids, rings, and single rays.

All spatial quantities are in SI metres.  Direction cosines (L, M, N) are
used for the ray direction so that ``L**2 + M**2 + N**2 == 1``.

The module is designed for cross-validation against the ASM wave-optics
results from the rest of the library.

Usage
-----
::

    from lumenairy.raytrace import (
        trace_prescription, spot_diagram, ray_fan_plot,
        system_abcd, seidel_coefficients,
    )
    from lumenairy.prescriptions import load_zemax_prescription_txt

    rx = load_zemax_prescription_txt('design.txt', surface_range=(1, 13))
    result = trace_prescription(rx, wavelength=1.31e-6, num_rings=10)
    spot_diagram(result)

Author: Andrew Traverso
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from .glass import get_glass_index
from .lenses import surface_sag_general, surface_sag_biconic


# ============================================================================
# Data structures
# ============================================================================

# Error codes for RayBundle.error_code.  Codes are cumulative "first
# failure wins": once set non-zero, a ray's error_code sticks for the
# rest of the trace so downstream surfaces can't overwrite the root
# cause.  alive = (error_code == 0) by invariant -- helpers that
# vignette rays MUST set error_code at the same time.
RAY_OK              = 0    # ray is alive
RAY_TIR             = 1    # total internal reflection at refract
RAY_APERTURE        = 2    # clipped by a surface semi_diameter
RAY_MISSED_SURFACE  = 3    # intersection Newton failed / no real root
RAY_NAN             = 4    # arithmetic produced NaN/Inf (numerical fault)
RAY_EVANESCENT      = 5    # diffraction order does not propagate
                           # (L'^2 + M'^2 > 1 after a grating k-shift)


@dataclass
class RayBundle:
    """A bundle of rays represented as parallel numpy arrays.

    Each array has shape ``(N_rays,)``.  All spatial coordinates are in
    metres; direction cosines satisfy ``L**2 + M**2 + N**2 == 1``.

    Attributes
    ----------
    x, y, z : ndarray
        Ray positions [m].
    L, M, N : ndarray
        Direction cosines (x, y, z components).
    wavelength : float
        Vacuum wavelength [m].
    alive : ndarray of bool
        ``False`` for rays that have been vignetted or suffered TIR.
        Derived quantity: ``alive = (error_code == RAY_OK)``.
    opd : ndarray
        Accumulated optical path length [m] along each ray.
    error_code : ndarray of uint8
        Per-ray diagnostic code (``RAY_OK`` / ``RAY_TIR`` /
        ``RAY_APERTURE`` / ``RAY_MISSED_SURFACE`` / ``RAY_NAN``).
        First-failure-wins: once a ray is killed with a non-zero
        code, subsequent surfaces do NOT overwrite the root cause.
        Useful for post-trace diagnostics -- see
        :func:`trace_summary` for the breakdown.
    """
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    L: np.ndarray
    M: np.ndarray
    N: np.ndarray
    wavelength: float
    alive: np.ndarray
    opd: np.ndarray
    # 3.1.9: per-ray diagnostic code.  Default-factory keeps older
    # constructors (pickled bundles, user code that creates bundles
    # directly without this field) working -- a missing error_code is
    # synthesised from alive as "alive -> OK, dead -> TIR (unknown)."
    error_code: Optional[np.ndarray] = None

    def __post_init__(self):
        # Synthesise error_code if the caller didn't supply one.  This
        # keeps any pre-3.1.9 code paths (user-constructed bundles,
        # pickled objects from older versions) working transparently.
        if self.error_code is None:
            ec = np.zeros(len(self.x), dtype=np.uint8)
            ec[~np.asarray(self.alive, dtype=bool)] = RAY_TIR
            # NB: "unknown dead" defaults to RAY_TIR as a placeholder;
            # actual downstream codes are set at kill time by
            # _intersect_surface / _refract / _reflect.
            self.error_code = ec

    @property
    def n_rays(self):
        return len(self.x)

    def copy(self):
        return RayBundle(
            x=self.x.copy(), y=self.y.copy(), z=self.z.copy(),
            L=self.L.copy(), M=self.M.copy(), N=self.N.copy(),
            wavelength=self.wavelength,
            alive=self.alive.copy(), opd=self.opd.copy(),
            error_code=(self.error_code.copy()
                         if self.error_code is not None else None),
        )


@dataclass
class Surface:
    """A single optical surface in the sequential model.

    Attributes
    ----------
    radius : float
        Radius of curvature [m].  ``inf`` for flat.
    conic : float
        Conic constant (0 = sphere, -1 = paraboloid).
    aspheric_coeffs : dict or None
        Even polynomial coefficients ``{power: coeff}``
        (e.g. ``{4: A4, 6: A6}``).
    semi_diameter : float
        Clear semi-aperture [m].  Rays outside are vignetted.
    glass_before : str
        Glass name on the input side (e.g. ``'air'``).
    glass_after : str
        Glass name on the output side (e.g. ``'N-BK7'``).
    is_mirror : bool
        If True, the surface reflects rather than refracts.
    is_stop : bool
        If True, marks this surface as the aperture stop of the
        system.  Used by stop-aware helpers (``find_stop``,
        ``compute_pupils``, ``seidel_coefficients``, etc.) to anchor
        the chief ray.  Zemax ``.zmx`` / ``.txt`` loaders set this
        from the STOP keyword; the legacy fallback behaviour (first
        surface with a finite semi-diameter, else surface 0) is
        preserved by ``find_stop`` when no surface is flagged.
    thickness : float
        Axial distance to the *next* surface [m].
    label : str
        Human-readable label for the surface.
    surf_num : int
        Zemax surface number (for reference).
    """
    radius: float = np.inf
    conic: float = 0.0
    aspheric_coeffs: Optional[Dict] = None
    semi_diameter: float = np.inf
    glass_before: str = 'air'
    glass_after: str = 'air'
    is_mirror: bool = False
    is_stop: bool = False
    thickness: float = 0.0
    label: str = ''
    surf_num: int = -1
    # Biconic / anamorphic extensions (all optional; None => rotationally
    # symmetric surface using radius / conic / aspheric_coeffs above).
    radius_y: Optional[float] = None
    conic_y: Optional[float] = None
    aspheric_coeffs_y: Optional[Dict] = None
    # Optional freeform departure layered on top of the (biconic) base
    # sag.  Dict keys depend on ``kind``; see freeform.surface_sag_freeform.
    # Recognised forms:
    #   {'kind': 'xy_polynomial', 'coefficients': {(i,j): a_ij, ...}}
    #   {'kind': 'zernike',      'coefficients': {(n,m): c_nm, ...},
    #    'aperture_radius': r}
    #   {'kind': 'chebyshev',    'coefficients': {(i,j): c_ij, ...},
    #    'normalization_radius': r}
    # Surface normals through freeform departures use a finite-difference
    # gradient at the ray's intersection point.
    freeform: Optional[Dict] = None
    # Optional BSDF (bidirectional scattering distribution function) for
    # stray-light analysis.  Either a BSDFModel instance or a dict spec
    # consumed by :func:`bsdf.make_bsdf`.  Does not affect the specular
    # trace; invoke :func:`bsdf.sample_scatter_rays` to spawn scatter
    # rays from a surface that carries this field.
    bsdf: Optional[object] = None


@dataclass
class TraceResult:
    """Result of tracing a ray bundle through a sequential system.

    Attributes
    ----------
    surfaces : list of Surface
        The surface list that was traced.
    ray_history : list of RayBundle
        Ray state *after* each surface (index 0 = after surface 0).
        ``ray_history[-1]`` is the final image-plane intercept.
    input_rays : RayBundle
        The original input rays (before any surface).
    wavelength : float
        Vacuum wavelength [m].
    """
    surfaces: List[Surface]
    ray_history: List[RayBundle]
    input_rays: RayBundle
    wavelength: float

    @property
    def image_rays(self) -> RayBundle:
        """Rays at the final (image) surface."""
        return self.ray_history[-1]

    def rays_at(self, surface_index: int) -> RayBundle:
        """Return the ray bundle after the given surface."""
        return self.ray_history[surface_index]


# ============================================================================
# Surface sag and normal computation
# ============================================================================

def _surface_sag_scalar(h_sq, R, conic=0.0, aspheric_coeffs=None):
    """Sag for scalar or array h_sq, thin wrapper around surface_sag_general.

    Note: this function assumes rotational symmetry (sag is a function
    of h² only).  For biconic surfaces use :func:`_surface_sag_xy`.
    """
    h_sq = np.asarray(h_sq, dtype=np.float64)
    return surface_sag_general(h_sq, R, conic, aspheric_coeffs)


def _surface_sag_xy(x, y, surface):
    """Sag z = f(x, y) for an arbitrary (possibly biconic / freeform)
    surface.

    Dispatch order:

    1. ``surface.freeform`` set       -> :func:`freeform.surface_sag_freeform`
       (uses the surface's own ``radius`` + ``conic`` as the base
       conic; departure is XY-poly / Zernike / Chebyshev per the spec).
    2. ``surface.radius_y`` set       -> :func:`surface_sag_biconic`.
    3. otherwise                      -> :func:`surface_sag_general`.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    spec = getattr(surface, 'freeform', None)
    if spec:
        # Freeform surfaces include their own base sphere/conic, so
        # they're an "instead of" rather than "on top of" the biconic
        # base.  Rotationally-symmetric base only -- biconic-base
        # freeforms are not yet supported by freeform.surface_sag_freeform
        # and would silently drop the radius_y term, which is worse
        # than warning the user up front.
        if getattr(surface, 'radius_y', None) is not None:
            raise NotImplementedError(
                "Freeform departure on a biconic base is not supported "
                "yet: leave radius_y unset, or extend "
                "freeform.surface_sag_freeform with biconic support.")
        from .freeform import surface_sag_freeform
        sd = dict(spec)
        # surface_sag_freeform reads 'freeform_type', 'radius', 'conic',
        # plus per-kind keys.  The Surface dataclass already supplies
        # radius/conic, so merge them in if not overridden.
        sd.setdefault('radius', surface.radius)
        sd.setdefault('conic', surface.conic)
        return surface_sag_freeform(x, y, sd)

    if getattr(surface, 'radius_y', None) is None:
        return surface_sag_general(
            x * x + y * y, surface.radius, surface.conic,
            surface.aspheric_coeffs)
    # Biconic path
    return surface_sag_biconic(
        x, y, R_x=surface.radius, R_y=surface.radius_y,
        conic_x=surface.conic,
        conic_y=surface.conic_y,
        aspheric_coeffs=surface.aspheric_coeffs,
        aspheric_coeffs_y=surface.aspheric_coeffs_y)


def _surface_sag_derivatives_xy(x, y, surface):
    """Partial derivatives (dz/dx, dz/dy) at (x, y) on the given surface.

    Used for the surface normal in refraction / reflection and for the
    slant-correction formula.  Handles biconic surfaces by summing the
    per-axis derivatives.  Freeform surfaces use a centred finite
    difference because the freeform sag basis functions don't expose
    analytic gradients.
    """
    # Freeform path: centred FD with h scaled to local feature size.
    if getattr(surface, 'freeform', None):
        # Step size: small fraction of typical aperture (use R or 1 mm).
        R = surface.radius if (surface.radius is not None
                               and np.isfinite(surface.radius)) else 1e-3
        h_step = max(abs(R) * 1e-6, 1e-9)
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        z_xp = _surface_sag_xy(x_arr + h_step, y_arr, surface)
        z_xm = _surface_sag_xy(x_arr - h_step, y_arr, surface)
        z_yp = _surface_sag_xy(x_arr, y_arr + h_step, surface)
        z_ym = _surface_sag_xy(x_arr, y_arr - h_step, surface)
        return ((z_xp - z_xm) / (2 * h_step),
                (z_yp - z_ym) / (2 * h_step))
    if getattr(surface, 'radius_y', None) is None:
        # Rotationally symmetric -- reuse the scalar helpers.
        h = np.sqrt(x * x + y * y)
        h_safe = np.maximum(h, 1e-30)
        dz_dh = _surface_sag_derivative(
            h, surface.radius, surface.conic, surface.aspheric_coeffs)
        dz_dx = np.where(h > 0, dz_dh * x / h_safe, 0.0)
        dz_dy = np.where(h > 0, dz_dh * y / h_safe, 0.0)
        return dz_dx, dz_dy

    # Biconic -- derivative of each axis independently.
    def _axis_deriv(u, R, K, asph):
        if R is None or np.isinf(R):
            d = np.zeros_like(u)
        else:
            h_sq = u * u
            norm = (1 + K) * h_sq / R ** 2
            valid = norm < 0.9999
            denom = np.where(valid, np.sqrt(np.maximum(1 - norm, 1e-30)), 1.0)
            d = np.where(valid, u / (R * denom), 0.0)
        if asph:
            for power, coeff in asph.items():
                # d/du of coeff * u^power = power * coeff * u^(power-1)
                d = d + power * coeff * u ** (power - 1)
        return d

    asph_y = (surface.aspheric_coeffs_y
              if surface.aspheric_coeffs_y is not None
              else surface.aspheric_coeffs)
    dz_dx = _axis_deriv(x, surface.radius, surface.conic,
                        surface.aspheric_coeffs)
    dz_dy = _axis_deriv(y, surface.radius_y, surface.conic_y,
                        asph_y)
    return dz_dx, dz_dy


def _surface_sag_derivative(h, R, conic=0.0, aspheric_coeffs=None):
    """Derivative of sag with respect to radial distance h = sqrt(h_sq).

    Returns dz/dh for computing surface normals.
    """
    h = np.asarray(h, dtype=np.float64)

    dz_dh = np.zeros_like(h)

    if R is not None and not np.isinf(R):
        h_sq = h ** 2
        norm = (1 + conic) * h_sq / R ** 2
        valid = norm < 0.9999
        denom = np.where(valid, np.sqrt(np.maximum(1 - norm, 1e-30)), 1.0)
        # d(sag)/dh for conic: h / (R * sqrt(1 - (1+k)*h^2/R^2))
        dz_dh = np.where(valid, h / (R * denom), 0.0)

    if aspheric_coeffs:
        for power, coeff in aspheric_coeffs.items():
            # d/dh of coeff * h^power = power * coeff * h^(power-1)
            dz_dh = dz_dh + power * coeff * h ** (power - 1)

    return dz_dh


def _surface_normal(x, y, surface):
    """Outward unit normal at point (x, y) on the given surface.

    Returns (nx, ny, nz) arrays.  The normal points from glass_before
    toward glass_after (i.e. in the +z direction for a flat surface).
    Handles biconic / anamorphic surfaces via ``_surface_sag_derivatives_xy``.
    """
    dz_dx, dz_dy = _surface_sag_derivatives_xy(x, y, surface)
    # Normal = (-dz/dx, -dz/dy, 1), normalised
    mag = np.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1.0)
    return -dz_dx / mag, -dz_dy / mag, 1.0 / mag


# ============================================================================
# Ray-surface intersection (Newton iteration)
# ============================================================================

def _intersect_surface(rays, surface, n_medium=1.0):
    """Find intersection of rays with a surface centred at z=0.

    Modifies ``rays`` in place: updates (x, y, z) to the intersection
    point and **accumulates OPD for the path travelled during the
    intersection** (in the medium the rays are currently in, which is
    the medium *before* this surface).  Rays that miss the clear
    aperture are marked dead.

    Parameters
    ----------
    rays : RayBundle
        Rays approaching the surface.  Their current z is the plane
        the previous transfer left them on (typically the vertex plane
        of this surface, z = 0 in this surface's frame).
    surface : Surface
        The surface to intersect.
    n_medium : float, default 1.0
        Refractive index of the medium the rays are travelling through
        on the way to this surface (i.e. the *glass_before* of this
        surface).  Used to accumulate the OPL of the small "vertex
        plane to actual sag" leg, which is critical for thick lenses
        with curved surfaces -- without it the trace under-counts (or
        over-counts, for concave) the true ray path between
        intersections.
    """
    R = surface.radius
    kc = surface.conic
    asph = surface.aspheric_coeffs

    if np.isinf(R) and not asph:
        # Flat surface: intersect at z = 0
        # t such that z + N*t = 0  =>  t = -z / N
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(rays.alive & (np.abs(rays.N) > 1e-30),
                         -rays.z / rays.N, 0.0)
    else:
        # Newton's method to find ray-surface intersection.
        # The surface is z = sag(x, y).  We need to find t such that
        # z + N*t = sag(x + L*t, y + M*t).
        t = np.zeros(rays.n_rays)

        # Initial guess: paraxial approximation for a sphere
        if not np.isinf(R):
            # For a sphere: x^2 + y^2 + (z-R)^2 = R^2
            # Approximate t from the ray-sphere intersection
            x0, y0, z0 = rays.x, rays.y, rays.z
            Ld, Md, Nd = rays.L, rays.M, rays.N

            # Centre of curvature at (0, 0, R)
            dx, dy, dz = x0, y0, z0 - R
            a = 1.0  # L^2 + M^2 + N^2
            b = 2.0 * (Ld * dx + Md * dy + Nd * dz)
            c = dx ** 2 + dy ** 2 + dz ** 2 - R ** 2
            disc = b ** 2 - 4 * a * c
            disc_safe = np.maximum(disc, 0.0)
            sqrt_disc = np.sqrt(disc_safe)

            # Pick the smaller positive root (closer intersection)
            t1 = (-b - sqrt_disc) / (2 * a)
            t2 = (-b + sqrt_disc) / (2 * a)
            # For rays travelling forward (N > 0), we want the intersection
            # closest to z=0, which is typically t1 for R > 0, t2 for R < 0
            if not np.isscalar(R):
                t = np.where(R > 0, t1, t2)
            else:
                t = t1 if R > 0 else t2

            t = np.where(disc > 0, t, 0.0)
        else:
            # Flat surface with aspheric terms only: start at z=0
            with np.errstate(divide='ignore', invalid='ignore'):
                t = np.where(np.abs(rays.N) > 1e-30, -rays.z / rays.N, 0.0)

        # Newton iterations
        for _ in range(10):
            xi = rays.x + rays.L * t
            yi = rays.y + rays.M * t
            zi = rays.z + rays.N * t
            sag_i = _surface_sag_xy(xi, yi, surface)

            # Residual: F(t) = zi - sag(xi, yi) = 0
            F = zi - sag_i

            # Derivative of F with respect to t:
            # dF/dt = N - dz/dx * L - dz/dy * M
            dz_dx, dz_dy = _surface_sag_derivatives_xy(xi, yi, surface)
            dF_dt = rays.N - dz_dx * rays.L - dz_dy * rays.M

            # Newton step
            dt = np.where(np.abs(dF_dt) > 1e-30, F / dF_dt, 0.0)
            t = t - dt

            if np.all(np.abs(dt) < 1e-15):
                break

    # Update ray positions
    t = np.where(rays.alive, t, 0.0)
    rays.x = rays.x + rays.L * t
    rays.y = rays.y + rays.M * t
    rays.z = rays.z + rays.N * t

    # Accumulate OPL for the vertex-plane -> actual-sag-intersection
    # leg.  ``t`` is the parametric distance along the ray (with
    # |(L,M,N)| = 1 by construction), so |t| is the geometric path
    # length.  Use the SIGNED contribution: a negative t (which
    # happens when the surface is concave and the ray has already
    # passed it after the previous transfer) corresponds to back-
    # tracking, and we should subtract the over-counted OPL.
    rays.opd = rays.opd + n_medium * t

    # Vignette rays outside the clear aperture
    if np.isfinite(surface.semi_diameter):
        h_sq = rays.x ** 2 + rays.y ** 2
        clipped = (h_sq > surface.semi_diameter ** 2) & rays.alive
        if clipped.any():
            rays.alive = rays.alive & ~clipped
            if rays.error_code is not None:
                # First-failure-wins: only set RAY_APERTURE on rays
                # that were alive up to this surface.
                rays.error_code = np.where(clipped, RAY_APERTURE,
                                             rays.error_code)


# ============================================================================
# Vector Snell's law (refraction and reflection)
# ============================================================================

def _refract(rays, surface, n1, n2):
    """Apply vector Snell's law at the surface.

    Updates direction cosines (L, M, N) in place.  Rays that undergo
    total internal reflection are marked dead.

    Convention: n̂ points into the incident medium (against the
    incoming ray).  cos_i = -(d̂ · n̂) > 0.

    Refracted direction:
        d̂_t = mu * d̂_i + (mu * cos_i - cos_t) * n̂
    where mu = n1 / n2.
    """
    nx, ny, nz = _surface_normal(rays.x, rays.y, surface)

    # Ensure normal points into the incident medium (against the ray)
    cos_i = rays.L * nx + rays.M * ny + rays.N * nz
    flip = cos_i > 0
    nx = np.where(flip, -nx, nx)
    ny = np.where(flip, -ny, ny)
    nz = np.where(flip, -nz, nz)
    # cos_i = -(d · n̂) = |d · n_original|  (always positive)
    cos_i = np.abs(cos_i)

    # Snell's law: n1 * sin(theta_i) = n2 * sin(theta_t)
    mu = n1 / n2
    sin2_t = mu ** 2 * (1.0 - cos_i ** 2)

    # Total internal reflection check
    tir = sin2_t > 1.0
    newly_tir = tir & rays.alive
    rays.alive = rays.alive & ~tir
    if newly_tir.any() and rays.error_code is not None:
        # First-failure-wins: RAY_TIR overwrites only RAY_OK entries.
        rays.error_code = np.where(newly_tir, RAY_TIR, rays.error_code)

    cos_t = np.sqrt(np.maximum(1.0 - sin2_t, 0.0))

    # Refracted direction: d_t = mu * d_i + (mu * cos_i - cos_t) * n̂
    factor = mu * cos_i - cos_t
    rays.L = np.where(rays.alive, mu * rays.L + factor * nx, rays.L)
    rays.M = np.where(rays.alive, mu * rays.M + factor * ny, rays.M)
    rays.N = np.where(rays.alive, mu * rays.N + factor * nz, rays.N)

    # Renormalise (numerical safety)
    mag = np.sqrt(rays.L ** 2 + rays.M ** 2 + rays.N ** 2)
    mag = np.maximum(mag, 1e-30)
    rays.L /= mag
    rays.M /= mag
    rays.N /= mag

    # Accumulate OPD at this surface
    # OPD contribution from the refraction surface itself is zero
    # (OPD is accumulated during transfer between surfaces)


def _reflect(rays, surface):
    """Reflect rays at a mirror surface.

    Updates direction cosines in place.

    Convention: n̂ points into the incident medium (against the
    incoming ray).  cos_i = -(d̂ · n̂) > 0.

    Reflected direction:
        d̂_r = d̂_i + 2 * cos_i * n̂
    """
    nx, ny, nz = _surface_normal(rays.x, rays.y, surface)

    # Ensure normal points into the incident medium (against the ray)
    cos_i = rays.L * nx + rays.M * ny + rays.N * nz
    flip = cos_i > 0
    nx = np.where(flip, -nx, nx)
    ny = np.where(flip, -ny, ny)
    nz = np.where(flip, -nz, nz)
    cos_i = np.abs(cos_i)

    # Reflected direction: d_r = d_i + 2 * cos_i * n̂
    rays.L = rays.L + 2.0 * cos_i * nx
    rays.M = rays.M + 2.0 * cos_i * ny
    rays.N = rays.N + 2.0 * cos_i * nz

    # Renormalise
    mag = np.sqrt(rays.L ** 2 + rays.M ** 2 + rays.N ** 2)
    mag = np.maximum(mag, 1e-30)
    rays.L /= mag
    rays.M /= mag
    rays.N /= mag


def _transfer(rays, thickness, n_medium):
    """Transfer rays by the given axial thickness in a medium of index n.

    Translates ray positions so they arrive at the next surface vertex
    plane (z = 0) and accumulates OPD.
    """
    if thickness == 0:
        return

    # Transfer: advance each ray along its direction until it reaches
    # z = thickness (the next surface vertex plane).
    # t = (thickness - z) / N
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(rays.alive & (np.abs(rays.N) > 1e-30),
                     (thickness - rays.z) / rays.N, 0.0)

    # Accumulate OPD: geometric path * refractive index
    path = np.abs(t)
    rays.opd = rays.opd + n_medium * path

    rays.x = rays.x + rays.L * t
    rays.y = rays.y + rays.M * t
    rays.z = np.zeros_like(rays.z)  # reset to vertex of next surface


# ============================================================================
# Sequential trace engine
# ============================================================================

def trace(rays, surfaces, wavelength, output_filter='all',
          surface_diffraction=None):
    """Trace a ray bundle through a sequential list of surfaces.

    Parameters
    ----------
    rays : RayBundle
        Input rays.  The bundle is *not* modified; a copy is traced.
    surfaces : list of Surface
        Ordered surface list.  ``surface.thickness`` gives the axial
        distance from this surface to the next.
    wavelength : float
        Vacuum wavelength [m] (used to resolve glass indices).
    output_filter : ``'all'`` (default) | ``'last'`` | callable
        Controls what per-surface state is retained in
        ``result.ray_history``.

        * ``'all'``  -- save a ``RayBundle.copy()`` after every
          surface (legacy behaviour).
        * ``'last'`` -- save only the final post-last-surface bundle
          in a one-element ``ray_history``.  ``result.image_rays``
          is still the expected object; every ``rays_at(i)`` for
          ``i < len(surfaces)-1`` raises ``IndexError``.  Use this
          for memory-constrained workloads where only the image-
          plane bundle is consumed -- most notably
          :func:`apply_real_lens_traced`, which at N=32768 avoids
          ~1-5 GB of transient ``RayBundle.copy()`` allocations per
          call.
        * ``callable`` -- any ``fn(rays, surf, index) -> Any``.  The
          return value is appended to ``ray_history``; return
          ``None`` to skip.  Enables user-defined per-surface
          recording (e.g. store only (x, y, opd) as a
          ``NamedTuple``, or accumulate running spot centroids).
    surface_diffraction : dict or None, optional
        Per-surface diffractive-order kicks.  Maps surface index
        ``i`` (zero-based) to a tuple ``(order_x, order_y, period_x,
        period_y)`` interpreted as the grating equation::

            L_new = L + order_x * wavelength / period_x
            M_new = M + order_y * wavelength / period_y

        applied AFTER refraction at surface ``i`` (so the rays
        continue propagation through the post-surface medium with the
        diffractive kick applied).  ``period_y`` may be ``np.inf`` for
        a 1-D grating; ``order_x`` / ``order_y`` may be half-integer
        (Dammann-style even-N splitters).  Orders that turn evanescent
        (``L_new**2 + M_new**2 > 1``) are flagged
        ``alive=False`` with ``error_code=RAY_EVANESCENT``.  See also
        :func:`apply_doe_phase_traced`.

    Returns
    -------
    result : TraceResult
    """
    r = rays.copy()
    history = [] if output_filter != 'last' else None
    final = None
    _diff = dict(surface_diffraction) if surface_diffraction else {}

    # Pre-resolve all glass indices once per wavelength.  Each
    # get_glass_index call has module-level LRU caching, so repeated
    # Python dispatch overhead is the only cost saved -- tiny per
    # surface, noticeable at high repeated-trace counts (focus
    # sweeps pre-3.1.8 retraced from scratch; the pattern survives
    # in user code that does its own iteration).  Underscore-
    # prefixed names to avoid colliding with the transfer-step
    # ``n_after = n2`` rebind later in this loop.
    _n_pre  = [get_glass_index(s.glass_before, wavelength) for s in surfaces]
    _n_post = [get_glass_index(s.glass_after,  wavelength) for s in surfaces]

    for i, surf in enumerate(surfaces):
        n1 = _n_pre[i]
        n2 = _n_post[i]

        # 1. Intersect with surface (accumulates OPL in glass_before)
        _intersect_surface(r, surf, n_medium=n1)

        # 2. Refract or reflect
        if surf.is_mirror:
            _reflect(r, surf)
        else:
            _refract(r, surf, n1, n2)

        # 2.5. Diffractive-order kick (if this surface is registered as
        # a grating in surface_diffraction).  Modifies (L, M, N) in
        # place AND adds the DOE's linear OPL contribution
        # ``m * lambda * (x, y) / Lambda`` -- which apply_doe_phase_traced
        # explicitly excludes but the LG aberration fit needs to see in
        # order to give correct (0, 0) piston phases per emitter.  The
        # linear part of this OPL is geometric and gets absorbed by the
        # piston-coherence merit's linear fit; any non-linear part comes
        # from the per-emitter chief rays hitting the DOE at non-paraxial
        # positions, and is precisely the corner-frame coherence content
        # we want the optimizer to see.
        _diff_spec = _diff.get(i)
        if _diff_spec is not None:
            _mx, _my, _px, _py = _diff_spec
            _dL = float(_mx) * wavelength / float(_px)
            _dM = float(_my) * wavelength / float(_py)
            r.L = r.L + _dL
            r.M = r.M + _dM
            _sumsq = r.L * r.L + r.M * r.M
            _evan = _sumsq > 1.0
            _propagating = ~_evan
            _N_new = np.zeros_like(r.N)
            np.sqrt(np.maximum(1.0 - _sumsq, 0.0),
                    out=_N_new, where=_propagating)
            # Preserve the sign of the longitudinal cosine (forward
            # vs. backward propagation).  The original N's sign was
            # set by the propagation direction; the diffraction kick
            # only shifts (L, M) so the new N has the same sign.
            r.N = np.where(r.N < 0, -_N_new, _N_new)
            # Add the constant grating-order OPL contribution evaluated
            # at the ray's DOE-plane intersection (x, y).  The factor
            # ``m * lambda / period`` is the same gradient applied to
            # (L, M) above, so this is the integral of that phase
            # gradient evaluated at the surface.
            r.opd = r.opd + _dL * r.x + _dM * r.y
            if np.any(_evan) and r.alive is not None:
                r.alive = r.alive & _propagating
                if r.error_code is not None:
                    r.error_code = np.where(
                        _evan & (r.error_code == RAY_OK),
                        np.uint8(RAY_EVANESCENT),
                        r.error_code,
                    )

        # Save state after this surface, per output_filter
        if output_filter == 'all':
            history.append(r.copy())
        elif callable(output_filter):
            item = output_filter(r, surf, i)
            if item is not None:
                history.append(item)
        # 'last' branch: retain only the final bundle, copied below

        # Remember the final bundle so 'last' mode can cheaply snapshot
        # after the loop without an extra walk.
        if i == len(surfaces) - 1:
            final = r.copy() if output_filter == 'last' else None

        # 3. Transfer to next surface (accumulates the bulk
        # vertex-to-vertex axial leg in glass_after; the small
        # sag-correction at the next surface is added by the next
        # _intersect_surface call).
        if i < len(surfaces) - 1:
            n_after = n2  # medium after this surface
            _transfer(r, surf.thickness, n_after)

    if output_filter == 'last':
        history = [final] if final is not None else []

    return TraceResult(
        surfaces=surfaces,
        ray_history=history,
        input_rays=rays,
        wavelength=wavelength,
    )


# ============================================================================
# Prescription → Surface list conversion
# ============================================================================

def surfaces_from_prescription(prescription):
    """Convert a lens prescription dict to a list of Surface objects.

    Accepts the same prescription format returned by
    :func:`prescriptions.load_zemax_prescription_txt`,
    :func:`prescriptions.load_zmx_prescription`,
    :func:`prescriptions.make_singlet`, etc.

    Parameters
    ----------
    prescription : dict
        Must contain ``'surfaces'`` and ``'thicknesses'`` keys.
        Optionally ``'aperture_diameter'``.

    Returns
    -------
    surfaces : list of Surface
    """
    p_surfs = prescription['surfaces']
    p_thick = prescription['thicknesses']
    aperture = prescription.get('aperture_diameter')

    # If the prescription has 'elements' with semi_diameter, use those
    elements = prescription.get('elements', None)

    surface_list = []
    for i, ps in enumerate(p_surfs):
        # Determine semi-diameter
        sd = np.inf
        if aperture is not None:
            sd = aperture / 2.0
        # If elements list has per-surface semi-diameters, use the tighter one
        if elements is not None:
            # Match by index within refracting surfaces
            refr_elems = [e for e in elements if e.get('element_type') == 'surface']
            if i < len(refr_elems):
                elem_sd = refr_elems[i].get('semi_diameter', np.inf)
                if elem_sd > 0 and np.isfinite(elem_sd):
                    sd = min(sd, elem_sd)

        thickness = p_thick[i] if i < len(p_thick) else 0.0

        # Freeform departure (optional).  Accept either a unified
        # 'freeform' dict or the legacy flat keys used by the
        # prescription-level freeform helpers.
        ff = ps.get('freeform')
        if ff is None and ps.get('freeform_type') is not None:
            ff = {k: v for k, v in ps.items()
                  if k in ('freeform_type', 'xy_coeffs',
                           'zernike_coeffs', 'cheb_coeffs',
                           'norm_radius', 'norm_x', 'norm_y')}

        # Aperture-stop flag.  Zemax parsers store the STOP keyword
        # on the per-surface dict ('is_stop': True); the prescription
        # dict may also carry a 'stop_index' (the index of the stop
        # in the surface list), which the wave-optics side already
        # honours.  Prefer per-surface flag if both are present.
        is_stop_flag = bool(ps.get('is_stop', False))
        if not is_stop_flag:
            stop_idx = prescription.get('stop_index')
            if stop_idx is not None and int(stop_idx) == i:
                is_stop_flag = True

        surface_list.append(Surface(
            radius=ps['radius'],
            conic=ps.get('conic', 0.0),
            aspheric_coeffs=ps.get('aspheric_coeffs'),
            semi_diameter=sd,
            glass_before=ps['glass_before'],
            glass_after=ps['glass_after'],
            is_mirror=False,
            is_stop=is_stop_flag,
            thickness=thickness,
            label=ps.get('comment', f'S{i+1}'),
            surf_num=ps.get('surf_num', i + 1),
            # Biconic / anamorphic (optional, default None = rotationally
            # symmetric)
            radius_y=ps.get('radius_y'),
            conic_y=ps.get('conic_y'),
            aspheric_coeffs_y=ps.get('aspheric_coeffs_y'),
            freeform=ff,
        ))

    return surface_list


def find_stop(surfaces):
    """Return the index of the aperture stop in ``surfaces``.

    Dispatch order:

    1. First surface with ``is_stop=True``.  If multiple surfaces
       are flagged, a ``RuntimeWarning`` is emitted and the earliest
       match is returned -- callers should explicitly set one stop
       per system to avoid ambiguity.
    2. First surface with a finite, user-declared ``semi_diameter``
       (legacy fallback, matches pre-3.1.8 implicit behaviour).
    3. Surface 0, with a ``UserWarning`` when the system has more
       than one surface (the stop guess is almost certainly wrong
       in that case, but we preserve behaviour rather than raise).

    Parameters
    ----------
    surfaces : list of Surface

    Returns
    -------
    stop_index : int

    Notes
    -----
    No trace work is performed here.  The function is O(N_surfaces)
    and safe to call inside hot loops, though ``compute_pupils`` and
    ``seidel_coefficients`` only need it once per system.
    """
    if not surfaces:
        raise ValueError("find_stop: empty surface list")
    flagged = [i for i, s in enumerate(surfaces) if s.is_stop]
    if len(flagged) > 1:
        import warnings
        warnings.warn(
            f"find_stop: multiple surfaces marked is_stop=True "
            f"(indices {flagged}); returning the first match "
            f"{flagged[0]}. Clear the extras to disambiguate.",
            RuntimeWarning, stacklevel=2)
    if flagged:
        return flagged[0]
    # Legacy fallback: first finite semi-diameter
    for i, s in enumerate(surfaces):
        if np.isfinite(s.semi_diameter):
            return i
    if len(surfaces) > 1:
        import warnings
        warnings.warn(
            "find_stop: no surface flagged is_stop=True and none have "
            "a finite semi_diameter; defaulting to surface 0.  "
            "Set is_stop=True on the intended aperture-stop surface "
            "for correct chief-ray behaviour.",
            UserWarning, stacklevel=2)
    return 0


# ============================================================================
# Ray generation helpers
# ============================================================================

def _make_bundle(x, y, L, M, wavelength):
    """Create a RayBundle from position and direction arrays."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    L = np.atleast_1d(np.asarray(L, dtype=np.float64))
    M = np.atleast_1d(np.asarray(M, dtype=np.float64))

    n = max(len(x), len(y), len(L), len(M))
    x = np.broadcast_to(x, n).copy()
    y = np.broadcast_to(y, n).copy()
    L = np.broadcast_to(L, n).copy()
    M = np.broadcast_to(M, n).copy()
    N = np.sqrt(np.maximum(1.0 - L ** 2 - M ** 2, 0.0))

    return RayBundle(
        x=x, y=y, z=np.zeros(n),
        L=L, M=M, N=N,
        wavelength=wavelength,
        alive=np.ones(n, dtype=bool),
        opd=np.zeros(n),
    )


def make_ray(x=0.0, y=0.0, L=0.0, M=0.0, wavelength=550e-9):
    """Create a single ray.

    Parameters
    ----------
    x, y : float
        Ray position at z = 0 [m].
    L, M : float
        Direction cosines in x and y.
    wavelength : float
        Vacuum wavelength [m].

    Returns
    -------
    RayBundle with one ray.
    """
    return _make_bundle([x], [y], [L], [M], wavelength)


def make_fan(axis='y', semi_aperture=12.7e-3, n_rays=21,
             field_angle=0.0, wavelength=550e-9):
    """Create a 1-D fan of rays across the pupil.

    Parameters
    ----------
    axis : str
        ``'x'`` or ``'y'`` — fan direction.
    semi_aperture : float
        Pupil semi-diameter [m].
    n_rays : int
        Number of rays (odd recommended to include the chief ray).
    field_angle : float
        Off-axis field angle [radians].  Applied as a direction cosine
        tilt in the fan axis.
    wavelength : float
        Vacuum wavelength [m].

    Returns
    -------
    RayBundle
    """
    t = np.linspace(-1, 1, n_rays)
    if axis == 'y':
        x = np.zeros(n_rays)
        y = t * semi_aperture
        L = np.zeros(n_rays)
        M = np.full(n_rays, np.sin(field_angle))
    else:
        x = t * semi_aperture
        y = np.zeros(n_rays)
        L = np.full(n_rays, np.sin(field_angle))
        M = np.zeros(n_rays)

    return _make_bundle(x, y, L, M, wavelength)


def make_ring(semi_aperture=12.7e-3, n_rays=36, field_angle=0.0,
              wavelength=550e-9, fraction=1.0):
    """Create a ring of rays at a given fractional pupil radius.

    Parameters
    ----------
    semi_aperture : float
        Pupil semi-diameter [m].
    n_rays : int
        Number of rays around the ring.
    field_angle : float
        Off-axis angle [radians] applied as M direction cosine.
    wavelength : float
        Vacuum wavelength [m].
    fraction : float
        Fractional pupil radius (0 to 1).

    Returns
    -------
    RayBundle
    """
    theta = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    r = semi_aperture * fraction
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    L = np.zeros(n_rays)
    M = np.full(n_rays, np.sin(field_angle))
    return _make_bundle(x, y, L, M, wavelength)


def make_grid(semi_aperture=12.7e-3, n_across=11, field_angle=0.0,
              wavelength=550e-9, pattern='square'):
    """Create a 2-D grid of rays across the pupil.

    Parameters
    ----------
    semi_aperture : float
        Pupil semi-diameter [m].
    n_across : int
        Number of rays along each axis.
    field_angle : float
        Off-axis angle [radians] applied as M direction cosine.
    wavelength : float
        Vacuum wavelength [m].
    pattern : str
        ``'square'`` — full rectangular grid.
        ``'circular'`` — only rays inside the pupil circle.

    Returns
    -------
    RayBundle
    """
    t = np.linspace(-1, 1, n_across)
    tx, ty = np.meshgrid(t, t)
    tx = tx.ravel()
    ty = ty.ravel()

    if pattern == 'circular':
        r_sq = tx ** 2 + ty ** 2
        mask = r_sq <= 1.0
        tx = tx[mask]
        ty = ty[mask]

    x = tx * semi_aperture
    y = ty * semi_aperture
    L = np.zeros_like(x)
    M = np.full_like(y, np.sin(field_angle))
    return _make_bundle(x, y, L, M, wavelength)


def make_rings(semi_aperture=12.7e-3, num_rings=6, rays_per_ring=36,
               field_angle=0.0, wavelength=550e-9, include_chief=True):
    """Create concentric rings of rays (good for spot diagrams).

    Parameters
    ----------
    semi_aperture : float
        Pupil semi-diameter [m].
    num_rings : int
        Number of concentric rings.
    rays_per_ring : int
        Rays per ring (each ring has this many).
    field_angle : float
        Off-axis angle [radians].
    wavelength : float
        Vacuum wavelength [m].
    include_chief : bool
        If True, add the on-axis chief ray at the centre.

    Returns
    -------
    RayBundle
    """
    all_x = []
    all_y = []

    if include_chief:
        all_x.append(0.0)
        all_y.append(0.0)

    for ring in range(1, num_rings + 1):
        frac = ring / num_rings
        theta = np.linspace(0, 2 * np.pi, rays_per_ring, endpoint=False)
        r = semi_aperture * frac
        all_x.append(r * np.cos(theta))
        all_y.append(r * np.sin(theta))

    x = np.concatenate([np.atleast_1d(xi) for xi in all_x])
    y = np.concatenate([np.atleast_1d(yi) for yi in all_y])
    L = np.zeros_like(x)
    M = np.full_like(y, np.sin(field_angle))
    return _make_bundle(x, y, L, M, wavelength)


# ============================================================================
# Diffraction-order direction shift (gratings / DOEs in the traced path)
# ============================================================================

def apply_doe_phase_traced(rays, order_x, order_y=0, *,
                           period_x, period_y=None, wavelength=None):
    """Apply a grating diffraction-order direction shift to a ray bundle.

    Each ray's transverse direction cosines are shifted by the grating
    equation::

        L_new = L + order_x * lambda / period_x
        M_new = M + order_y * lambda / period_y

    The longitudinal cosine is recomputed from
    ``L_new**2 + M_new**2 + N_new**2 == 1``.  Orders for which
    ``L_new**2 + M_new**2 > 1`` are evanescent (do not propagate); those
    rays are flagged ``alive=False`` with ``error_code = RAY_EVANESCENT``.

    Ray positions ``(x, y, z)`` and the OPL accumulator are *not*
    modified -- the grating is treated as a thin diffractive surface
    that only redirects each ray.  If you need to add the constant
    grating-order phase shift to ``opd``, do so manually after the call.

    The function supports two calling conventions:

    1. **Single order** -- pass scalar ``order_x`` and ``order_y``.
       The returned bundle has the same length as ``rays``.

    2. **Order array** -- pass 1-D arrays of equal length for
       ``order_x`` and ``order_y``.  The returned bundle is replicated
       ``len(order_x)`` times in *order-major* layout::

           out[order=k, ray=i] = out[k * n_rays + i]

       i.e. all rays for order 0, then all rays for order 1, ...

    Typical use: split a ray bundle at a Dammann-grating plane into a
    set of diffraction orders, then continue tracing each order through
    the post-grating optics with a single :func:`trace` call on the
    flattened bundle.

    Parameters
    ----------
    rays : RayBundle
        Input bundle.  Not modified in place.
    order_x : float, int, or 1-D array-like
        Diffraction order along the grating's x-axis.  Half-integer
        orders are allowed (e.g. for even-N Dammann splitters).
    order_y : float, int, or 1-D array-like, default 0
        Diffraction order along the grating's y-axis.  When passing
        arrays, ``order_x`` and ``order_y`` must broadcast to the same
        1-D length.
    period_x : float
        Grating period along x [m].  Required keyword.
    period_y : float, optional
        Grating period along y [m].  Defaults to ``period_x`` (square
        crossed grating).  Use ``np.inf`` to disable diffraction along
        one axis (1-D grating).
    wavelength : float, optional
        Vacuum wavelength [m].  Defaults to ``rays.wavelength``.

    Returns
    -------
    RayBundle
        New bundle (positions copied, directions shifted).  Length equals
        ``len(rays)`` for scalar orders or ``n_orders * len(rays)`` for
        order arrays.

    Notes
    -----
    The grating equation here is the small-angle / paraxial direction-
    cosine form: ``sin(theta_diff) - sin(theta_in) = m * lambda / Lambda``
    expressed as ``L_new = L_in + m * lambda / Lambda``.  This is the
    standard 1-st-order DOE / Dammann ray-tracing convention; it neglects
    the cosine factor that distinguishes ``sin`` from the direction
    cosine for very large grating angles.  For modest deflections
    (sub-100 mrad) the two are interchangeable to <1% even at the
    pupil edge.

    See Also
    --------
    trace : Geometric ray tracer (call after this function with the
        post-grating surfaces).
    lumenairy.doe.makedammann2d : 2-D Dammann period derivation.
    """
    if wavelength is None:
        wavelength = rays.wavelength
    if period_y is None:
        period_y = period_x

    # Normalize order args; track whether the caller passed scalars
    # (single-order convention) or arrays (multi-order replication).
    mx = np.asarray(order_x, dtype=np.float64)
    my = np.asarray(order_y, dtype=np.float64)
    scalar_input = (mx.ndim == 0 and my.ndim == 0)
    mx = np.atleast_1d(mx)
    my = np.atleast_1d(my)
    if mx.ndim != 1 or my.ndim != 1:
        raise ValueError(
            f"order_x and order_y must be scalar or 1-D, got shapes "
            f"{mx.shape} and {my.shape}")
    try:
        mx, my = np.broadcast_arrays(mx, my)
    except ValueError as e:
        raise ValueError(
            f"order_x (length {len(mx)}) and order_y (length {len(my)}) "
            f"must broadcast to the same 1-D length") from e
    n_orders = mx.size
    n_rays = len(rays.x)

    # Per-order direction increments.
    dL = (mx * wavelength / period_x).reshape(n_orders, 1)
    dM = (my * wavelength / period_y).reshape(n_orders, 1)

    # Broadcast to (n_orders, n_rays); reshape input direction cosines.
    L_new = rays.L.reshape(1, n_rays) + dL
    M_new = rays.M.reshape(1, n_rays) + dM

    sum_sq = L_new ** 2 + M_new ** 2
    propagating = sum_sq <= 1.0
    N_new = np.zeros_like(L_new)
    np.sqrt(np.maximum(1.0 - sum_sq, 0.0), out=N_new, where=propagating)

    # Per-order alive / error_code grids.
    alive_in = np.asarray(rays.alive, dtype=bool).reshape(1, n_rays)
    alive_new = alive_in & propagating

    if rays.error_code is not None:
        ec_in = np.asarray(rays.error_code).reshape(1, n_rays)
        ec_new = np.broadcast_to(ec_in, (n_orders, n_rays)).copy()
    else:
        ec_new = np.zeros((n_orders, n_rays), dtype=np.uint8)
        ec_new[~alive_in.repeat(n_orders, axis=0)] = RAY_TIR
    # First-failure-wins: only stamp RAY_EVANESCENT on rays that were
    # alive coming in but became non-propagating from the order shift.
    newly_dead = (~propagating) & alive_in
    ec_new[newly_dead] = RAY_EVANESCENT

    if scalar_input:
        # Single-order convention: same shape as input.
        return RayBundle(
            x=rays.x.copy(), y=rays.y.copy(), z=rays.z.copy(),
            L=L_new[0], M=M_new[0], N=N_new[0],
            wavelength=wavelength,
            alive=alive_new[0],
            opd=rays.opd.copy(),
            error_code=ec_new[0],
        )

    # Order-major flatten: all rays for order 0, then order 1, ...
    return RayBundle(
        x=np.tile(rays.x, n_orders),
        y=np.tile(rays.y, n_orders),
        z=np.tile(rays.z, n_orders),
        L=L_new.reshape(-1),
        M=M_new.reshape(-1),
        N=N_new.reshape(-1),
        wavelength=wavelength,
        alive=alive_new.reshape(-1),
        opd=np.tile(rays.opd, n_orders),
        error_code=ec_new.reshape(-1),
    )


# ============================================================================
# High-level trace functions
# ============================================================================

def trace_prescription(prescription, wavelength, semi_aperture=None,
                       field_angle=0.0, num_rings=6, rays_per_ring=36,
                       ray_pattern='rings', n_across=11,
                       image_distance=None):
    """Trace rays through a lens prescription.

    Convenience wrapper that converts a prescription dict to surfaces,
    generates rays, traces, and optionally propagates to a custom image
    distance.

    Parameters
    ----------
    prescription : dict
        Lens prescription (from :func:`prescriptions.make_singlet` etc.).
    wavelength : float
        Vacuum wavelength [m].
    semi_aperture : float or None
        Pupil semi-diameter [m].  If None, uses
        ``prescription['aperture_diameter'] / 2``.
    field_angle : float
        Off-axis field angle [radians].
    num_rings, rays_per_ring : int
        Parameters for the ``'rings'`` pattern.
    ray_pattern : str
        ``'rings'``, ``'grid'``, or ``'fan_xy'``.
    n_across : int
        Grid size for the ``'grid'`` pattern.
    image_distance : float or None
        If given, add a final flat surface at this distance after the
        last prescription surface.  Useful for evaluating the spot at a
        specific image plane.

    Returns
    -------
    TraceResult
    """
    surfaces = surfaces_from_prescription(prescription)

    if semi_aperture is None:
        ap = prescription.get('aperture_diameter')
        semi_aperture = ap / 2.0 if ap else 12.7e-3

    # Generate rays
    if ray_pattern == 'rings':
        rays = make_rings(semi_aperture, num_rings, rays_per_ring,
                          field_angle, wavelength)
    elif ray_pattern == 'grid':
        rays = make_grid(semi_aperture, n_across, field_angle,
                         wavelength, pattern='circular')
    elif ray_pattern == 'fan_xy':
        fan_y = make_fan('y', semi_aperture, 2 * rays_per_ring + 1,
                         field_angle, wavelength)
        fan_x = make_fan('x', semi_aperture, 2 * rays_per_ring + 1,
                         field_angle, wavelength)
        # Merge (skip duplicate chief ray)
        rays = _make_bundle(
            np.concatenate([fan_y.x, fan_x.x[fan_x.x != 0]]),
            np.concatenate([fan_y.y, fan_x.y[fan_x.x != 0]]),
            np.concatenate([fan_y.L, fan_x.L[fan_x.x != 0]]),
            np.concatenate([fan_y.M, fan_x.M[fan_x.x != 0]]),
            wavelength,
        )
    else:
        raise ValueError(f"Unknown ray_pattern: {ray_pattern!r}")

    # If image_distance is specified, set the last surface thickness and
    # append a flat image-plane surface so the trace engine transfers
    # the rays to the image plane before the final intersection.
    if image_distance is not None and surfaces:
        # Determine the medium after the last optical surface
        last_glass = surfaces[-1].glass_after
        surfaces[-1].thickness = image_distance
        surfaces.append(Surface(
            radius=np.inf, conic=0.0,
            semi_diameter=np.inf,
            glass_before=last_glass, glass_after=last_glass,
            is_mirror=False, thickness=0.0,
            label='Image',
        ))

    return trace(rays, surfaces, wavelength)


# ============================================================================
# Paraxial ray trace and ABCD matrix
# ============================================================================

def _paraxial_trace(surfaces, wavelength, y_in=0.0, u_in=0.0):
    """Trace a single paraxial ray (y, u) through the system.

    Uses the exact paraxial recursion:
        y' = y + u * t          (transfer)
        u' = u - y * phi / n'   (refraction, phi = (n'-n)/R)

    Returns lists of (y, u) at each surface (after refraction).
    """
    y_hist = []
    u_hist = []

    y = float(y_in)
    u = float(u_in)

    for i, surf in enumerate(surfaces):
        n1 = get_glass_index(surf.glass_before, wavelength)
        n2 = get_glass_index(surf.glass_after, wavelength)
        R = surf.radius

        # Refraction (power).  Paraxial refraction equation:
        #    n2 * u2  =  n1 * u1  -  y * (n2 - n1) / R
        # So the correct update for u ( = u' in the new medium) is:
        #    u  <-  (n1 * u_prev - y * phi) / n2
        # Historical equivalent form below (u prior to update is still
        # n1-normalised):  u <- u - y * phi / n2 .  Both agree because
        # u_prev on the right-hand side already satisfies u_prev = u
        # at this point in the loop (no intervening rewrite).
        if surf.is_mirror:
            phi = 2.0 * n1 / R if np.isfinite(R) else 0.0
            u = u - y * phi / n1
            n2 = n1  # medium doesn't change for mirrors
        else:
            phi = (n2 - n1) / R if np.isfinite(R) else 0.0
            u = u - y * phi / n2

        y_hist.append(y)
        u_hist.append(u)

        # Transfer to next surface
        if i < len(surfaces) - 1:
            t = surf.thickness
            n_medium = n2
            y = y + u * t

    return y_hist, u_hist


def _paraxial_refract(y, n1_u, R, n1, n2):
    """Paraxial refraction: returns n2*u2 given n1*u1."""
    if np.isinf(R):
        return n1_u  # flat surface, no power
    return n1_u - y * (n2 - n1) / R


def _paraxial_transfer(y, n_u, t, n):
    """Paraxial transfer: y2 = y1 + u1 * t, n*u unchanged."""
    u = n_u / n
    return y + u * t, n_u


def system_abcd(surfaces, wavelength):
    """Compute the system ABCD matrix using paraxial ray tracing.

    Traces a marginal ray (y=1, u=0) and an axial ray (y=0, u=1)
    through the surface list and constructs the 2x2 system matrix.

    Parameters
    ----------
    surfaces : list of Surface
        Surface list (from :func:`surfaces_from_prescription`).
    wavelength : float
        Vacuum wavelength [m].

    Returns
    -------
    abcd : ndarray, shape (2, 2)
        System matrix ``[[A, B], [C, D]]``.
    efl : float
        Effective focal length ``-1/C`` [m].
    bfl : float
        Back focal length (distance from last surface to rear focus) [m].
    ffl : float
        Front focal length (distance from first surface to front focus) [m].
    """
    # Build system matrix by multiplying surface-by-surface
    M = np.eye(2)

    for i, surf in enumerate(surfaces):
        n1 = get_glass_index(surf.glass_before, wavelength)
        n2 = get_glass_index(surf.glass_after, wavelength)
        R = surf.radius

        # Refraction matrix
        if np.isfinite(R) and not surf.is_mirror:
            phi = (n2 - n1) / R
            R_mat = np.array([[1.0, 0.0],
                              [-phi, 1.0]])
        elif surf.is_mirror and np.isfinite(R):
            phi = 2.0 * n1 / R
            R_mat = np.array([[1.0, 0.0],
                              [-phi, 1.0]])
        else:
            R_mat = np.eye(2)

        M = R_mat @ M

        # Transfer matrix to next surface
        if i < len(surfaces) - 1:
            t = surf.thickness
            n_after = n2
            T_mat = np.array([[1.0, t / n_after],
                              [0.0, 1.0]])
            M = T_mat @ M

    A, B = M[0, 0], M[0, 1]
    C, D = M[1, 0], M[1, 1]

    efl = -1.0 / C if abs(C) > 1e-30 else np.inf
    bfl = -A / C if abs(C) > 1e-30 else np.inf
    ffl = -D / C if abs(C) > 1e-30 else np.inf

    return M, efl, bfl, ffl


def system_abcd_prescription(prescription, wavelength):
    """Compute the ABCD matrix from a lens prescription dict.

    Convenience wrapper around :func:`system_abcd`.
    """
    surfaces = surfaces_from_prescription(prescription)
    return system_abcd(surfaces, wavelength)


# ============================================================================
# Per-lens ABCD helpers (3.1.9)
# ============================================================================

@dataclass
class LensInfo:
    """Paraxial characterisation of a single lens element.

    Returned by :func:`lens_abcd` and :func:`find_lenses`.  Captures
    everything you'd typically quote from a datasheet: effective focal
    length, back/front focal lengths, principal-plane positions, and
    the underlying ABCD matrix so callers can compose it with other
    transfer matrices.

    All lengths in metres.  ``principal_planes`` returns ``(H, H')``
    where H is measured from the first surface (positive = rearward)
    and H' is measured from the last surface (positive = forward).
    This follows the usual Welford / Hecht convention; Zemax
    reports H and H' with the opposite sign convention on
    ``radius < 0`` systems -- cross-check signs before comparing.
    """
    abcd: np.ndarray               # (2, 2) air-to-air
    efl: float
    bfl: float
    ffl: float
    principal_planes: tuple        # (H, H')
    thickness: float               # total center thickness
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    label: str = ''


def _surface_copy_with(surf, **overrides):
    """Return a new Surface with the given fields overridden.

    Propagates all optional fields (``radius_y``, ``conic_y``,
    ``aspheric_coeffs_y``, ``freeform``, ``is_stop``) so anamorphic,
    freeform, and stop-flagged surfaces survive the clone.  This is a
    lightweight drop-in for ``dataclasses.replace`` that keeps the
    fallback ``getattr(..., None)`` for bundles unpickled from older
    library versions.
    """
    return Surface(
        radius=overrides.get('radius', surf.radius),
        conic=overrides.get('conic', surf.conic),
        aspheric_coeffs=overrides.get('aspheric_coeffs', surf.aspheric_coeffs),
        semi_diameter=overrides.get('semi_diameter', surf.semi_diameter),
        glass_before=overrides.get('glass_before', surf.glass_before),
        glass_after=overrides.get('glass_after', surf.glass_after),
        is_mirror=overrides.get('is_mirror', surf.is_mirror),
        is_stop=overrides.get('is_stop', getattr(surf, 'is_stop', False)),
        thickness=overrides.get('thickness', surf.thickness),
        label=overrides.get('label', surf.label),
        surf_num=overrides.get('surf_num', surf.surf_num),
        radius_y=overrides.get('radius_y',
                                 getattr(surf, 'radius_y', None)),
        conic_y=overrides.get('conic_y',
                                getattr(surf, 'conic_y', None)),
        aspheric_coeffs_y=overrides.get(
            'aspheric_coeffs_y',
            getattr(surf, 'aspheric_coeffs_y', None)),
        freeform=overrides.get('freeform',
                                 getattr(surf, 'freeform', None)),
        bsdf=overrides.get('bsdf', getattr(surf, 'bsdf', None)),
    )


def lens_abcd(lens, wavelength, *, start=None, end=None, label=None,
              surfaces=None):
    """Compute paraxial ABCD + EFL/BFL/FFL for a single lens element.

    Accepts any of the following forms of ``lens``:

    - **Prescription dict** (has ``'surfaces'`` + ``'thicknesses'``):
      treats the whole prescription as one lens (entire air-to-air
      section).  Same format consumed by
      :func:`apply_real_lens` / :func:`apply_real_lens_traced`.
    - **List of `Surface`**: treated as the lens in its entirety.
      Pass ``start`` / ``end`` to select a contiguous slice
      (inclusive both ends) -- useful when you have a full system's
      surface list and know surfaces ``[s:e+1]`` constitute one
      physical element (cemented doublet, etc.).
    - **Single `Surface`** (e.g. a mirror treated as a "lens"):
      characterised as a one-surface optic.
    - **`LensInfo`** (as returned by :func:`find_lenses`): requires the
      ``surfaces`` kwarg to be the original surface list the LensInfo
      was derived from.  Useful for re-analyzing a detected lens at a
      different ``wavelength``.  The returned LensInfo preserves the
      original ``start_index``, ``end_index``, and ``label``.

    The trailing-thickness gap on the last surface is stripped before
    the ABCD computation so the returned ABCD is the lens alone
    (air-to-air at the last vertex), not "lens plus whatever
    propagation came after."  If you need lens + downstream gap,
    build a ``T(d)`` transfer matrix separately and compose.

    Note on the polymorphic dispatch
    --------------------------------
    The 2026-04 roadmap proposed a larger API accepting
    :mod:`system.py`-style element dicts (``{'type': 'lens', ...}``)
    as well.  That dispatch is intentionally omitted: callers with a
    system-element dict have a one-line unwrap path
    (``surfaces_from_elements([elem], wavelength)``), and the
    narrower API here is easier to reason about and type-check.

    Parameters
    ----------
    lens : dict | list[Surface] | Surface | LensInfo
    wavelength : float
        Vacuum wavelength [m].  ABCD is wavelength-dependent through
        glass dispersion, so no silent default.
    start, end : int, optional
        Inclusive surface-index range when ``lens`` is a list of
        surfaces; ignored otherwise.
    label : str, optional
        Override the auto-generated label on the returned LensInfo.
    surfaces : list[Surface], optional
        Required when ``lens`` is a ``LensInfo``: the original surface
        list the LensInfo was derived from.  Ignored otherwise.

    Returns
    -------
    LensInfo

    Examples
    --------
    >>> rx = make_doublet(...)
    >>> info = lens_abcd(rx, 1.31e-6)
    >>> print(f"EFL = {info.efl*1e3:.2f} mm, H' = {info.principal_planes[1]*1e3:.2f} mm")

    >>> # Auto-detect all lenses in a compound system
    >>> surfaces = surfaces_from_prescription(full_rx)
    >>> for L in find_lenses(surfaces, 1.31e-6):
    ...     print(f"{L.label}: EFL = {L.efl*1e3:.2f} mm")

    >>> # Re-analyze a detected lens at a different wavelength
    >>> lenses = find_lenses(surfaces, 1.31e-6)
    >>> L_vis = lens_abcd(lenses[0], 0.55e-6, surfaces=surfaces)
    """
    # ---- Dispatch input to a surface slice -----------------------
    if isinstance(lens, LensInfo):
        if surfaces is None:
            raise ValueError(
                "lens_abcd: LensInfo input requires the 'surfaces' kwarg "
                "(the original surface list the LensInfo was derived from).")
        if not isinstance(surfaces, list) or \
                not all(isinstance(s, Surface) for s in surfaces):
            raise TypeError(
                "lens_abcd: 'surfaces' must be a list of Surface objects.")
        if lens.start_index is None or lens.end_index is None:
            raise ValueError(
                "lens_abcd: LensInfo has no start_index/end_index -- it "
                "was not produced by find_lenses and cannot be sliced.")
        s_idx = int(lens.start_index)
        e_idx = int(lens.end_index)
        if not (0 <= s_idx <= e_idx < len(surfaces)):
            raise ValueError(
                f"lens_abcd: LensInfo indices [{s_idx}..{e_idx}] out of "
                f"range for surfaces list of length {len(surfaces)}.")
        sub = list(surfaces[s_idx:e_idx + 1])
        auto_label = lens.label or f'Lens@surfaces[{s_idx}..{e_idx}]'
    elif isinstance(lens, Surface):
        sub = [lens]
        auto_label = lens.label or 'Lens'
        s_idx, e_idx = None, None
    elif isinstance(lens, dict):
        if 'surfaces' not in lens:
            raise ValueError(
                "lens_abcd: dict input must be a prescription with "
                "a 'surfaces' key.  Got keys: "
                f"{sorted(lens.keys())!r}.")
        sub = surfaces_from_prescription(lens)
        auto_label = lens.get('name', 'Lens')
        s_idx, e_idx = None, None
    elif isinstance(lens, list):
        if not lens:
            raise ValueError("lens_abcd: empty surface list.")
        if not all(isinstance(s, Surface) for s in lens):
            raise TypeError(
                "lens_abcd: list input must contain Surface objects.")
        s_idx = 0 if start is None else int(start)
        e_idx = (len(lens) - 1) if end is None else int(end)
        if not (0 <= s_idx <= e_idx < len(lens)):
            raise ValueError(
                f"lens_abcd: start={start} / end={end} out of range "
                f"for surface list of length {len(lens)}.")
        sub = list(lens[s_idx:e_idx + 1])
        auto_label = (sub[0].label
                       if (sub[0].label and len(sub) == 1)
                       else f'Lens@surfaces[{s_idx}..{e_idx}]')
    else:
        raise TypeError(
            f"lens_abcd: unsupported lens type {type(lens).__name__}.  "
            f"Pass a prescription dict, list of Surface, a single "
            f"Surface, or a LensInfo (with the 'surfaces' kwarg).")

    # ---- Strip trailing thickness so ABCD is air-to-air ----------
    last = len(sub) - 1
    sub = [
        _surface_copy_with(s, thickness=(0.0 if i == last else s.thickness))
        for i, s in enumerate(sub)
    ]

    # ---- ABCD + paraxial focal quantities ------------------------
    M, efl, bfl, ffl = system_abcd(sub, wavelength)
    A, B, C, D = float(M[0, 0]), float(M[0, 1]), \
                  float(M[1, 0]), float(M[1, 1])
    if abs(C) > 1e-30:
        # Welford convention: H = (D-1)/C (from front vertex),
        # H' = (1-A)/C (from rear vertex).
        H  = (D - 1.0) / C
        Hp = (1.0 - A) / C
    else:
        H, Hp = float('inf'), float('inf')

    thickness = sum(sub[k].thickness for k in range(len(sub) - 1))

    return LensInfo(
        abcd=np.asarray(M, dtype=np.float64),
        efl=float(efl),
        bfl=float(bfl),
        ffl=float(ffl),
        principal_planes=(H, Hp),
        thickness=float(thickness),
        start_index=s_idx,
        end_index=e_idx,
        label=(label if label is not None else auto_label),
    )


@dataclass
class PupilInfo:
    """Paraxial entrance and exit pupil characterisation.

    Both EP and XP positions are given relative to the lens's own
    reference surfaces:
    * ``ep_z``: axial distance from ``surfaces[0]`` to EP.
      Negative = EP is to the left (object side) of the first surface.
    * ``xp_z``: axial distance from ``surfaces[-1]`` to XP.
      Positive = XP is to the right (image side) of the last surface.
    """
    ep_z: float
    ep_radius: float
    xp_z: float
    xp_radius: float
    stop_index: int


def compute_pupils(surfaces, wavelength, stop_index=None):
    """Paraxial entrance and exit pupil positions and radii.

    Images the aperture stop backward through the pre-stop optics to
    find the entrance pupil, and forward through the post-stop optics
    to find the exit pupil.  Both are computed from the sub-system
    ABCD matrices; no ray tracing needed.

    Parameters
    ----------
    surfaces : list of Surface
    wavelength : float
        Vacuum wavelength [m].
    stop_index : int, optional
        Explicit stop surface index.  Defaults to the result of
        :func:`find_stop` (i.e. the surface flagged ``is_stop=True``,
        or the first finite-semi-diameter surface, or 0).

    Returns
    -------
    PupilInfo

    Notes
    -----
    For the EP, we seek the object-space conjugate of the stop:
    image distance from surface 0 at which an object placed there
    would image onto the stop plane.  Equivalently, treat the stop
    as the "source" and propagate in reverse through the pre-stop
    sub-system.  For a reversed sub-system M_rev = T(-t1)
    L1^{-1} T(-t2) L2^{-1} ..., but the cleanest implementation is
    the imaging condition on the forward sub-system's ABCD:
    if M_pre = [[A, B], [C, D]] maps (y_obj, u_obj) at surface 0
    to (y_stop, u_stop) at the stop, then the object-space position
    z_ep (measured from surface 0, negative = to the left) that
    images to the stop satisfies A + B / (z_ep * ... ) = 0 after
    prepending T(|z_ep|).  Equivalently, solve B_new = 0 for the
    prepended distance:  B + z_ep * A = 0  =>  z_ep = -B / A.

    For the XP: same logic on the post-stop sub-system in the
    forward direction, with the stop as the object.
    """
    if not surfaces:
        raise ValueError("compute_pupils: empty surface list.")
    if stop_index is None:
        stop_index = find_stop(surfaces)
    if not (0 <= stop_index < len(surfaces)):
        raise ValueError(
            f"compute_pupils: stop_index={stop_index} out of range "
            f"[0, {len(surfaces)})")

    stop_surf = surfaces[stop_index]
    # Stop radius from the surface's semi_diameter (fall back to a
    # reasonable default when infinite, with a warning -- infinite
    # semi-diameter means "no stop was really declared here").
    stop_radius = stop_surf.semi_diameter
    if not np.isfinite(stop_radius):
        import warnings
        warnings.warn(
            f"compute_pupils: stop surface at index {stop_index} has "
            f"infinite semi_diameter; pupil radii will be reported "
            f"as NaN.  Declare a finite semi_diameter to get "
            f"meaningful pupil sizes.",
            UserWarning, stacklevel=2)
        stop_radius = float('nan')

    # ---- Entrance pupil -------------------------------------------
    # Pre-stop sub-system: surfaces[0 .. stop_index-1], ending with
    # the thickness from the last pre-stop surface to the stop's
    # vertex (i.e. include the propagation gap up to the stop).
    if stop_index == 0:
        # Stop is at the first surface; EP coincides with it.
        ep_z = 0.0
        ep_radius = stop_radius
    else:
        pre = [_surface_copy_with(s) for s in surfaces[:stop_index]]
        # Append the propagation leg to the stop as a trailing
        # thickness on the last pre-surface.  system_abcd walks
        # thicknesses between surfaces, so we need to insert an extra
        # "transfer only" leg.  Easiest: append a dummy flat air
        # surface at the stop vertex (zero power, just for the
        # transfer) -- its ABCD contribution is identity; the
        # thickness accumulates from pre[-1].thickness which is the
        # pre->stop gap.
        # Actually the pre-stop sub-system already walks
        # thicknesses[0..stop_index-1] which includes the gap from
        # surface stop_index-1 to stop (since s.thickness = distance
        # to NEXT surface).  So no dummy needed; system_abcd(pre)
        # already lands the ray at the stop vertex.
        M_pre, _, _, _ = system_abcd(pre, wavelength)
        A_pre, B_pre = float(M_pre[0, 0]), float(M_pre[0, 1])
        C_pre, D_pre = float(M_pre[1, 0]), float(M_pre[1, 1])
        # Imaging condition: prepend T(z_obj) so B_total = 0.
        # B + z * A = 0 ?  No: T(z) applied on the RIGHT gives
        # M . T(z) = [[A, A*z+B], [C, C*z+D]].  So B_total = A*z + B.
        # z_ep = -B / A (distance from surface 0 back to EP).
        # Magnification through pre: m_pre = A_pre when B=0.
        if abs(A_pre) > 1e-30:
            z_ep = -B_pre / A_pre
            # Radius: EP is the reverse image of the stop with
            # magnification 1/A_pre (because the forward sub-system
            # maps object height to stop height with factor A when
            # B=0 -> object height = stop / A).
            ep_radius = abs(stop_radius / A_pre) if np.isfinite(stop_radius) else float('nan')
        else:
            z_ep = float('inf')
            ep_radius = float('inf')

    # ---- Exit pupil -----------------------------------------------
    if stop_index == len(surfaces) - 1:
        xp_z = 0.0
        xp_radius = stop_radius
    else:
        post = [_surface_copy_with(s) for s in surfaces[stop_index + 1:]]
        # Propagation from stop to first post-surface is the
        # thickness attribute on surfaces[stop_index], which lives on
        # the stop surface itself.  To include it in the post
        # sub-system we prepend a dummy air surface with that
        # thickness.  Cleanest: pass a fake Surface at stop_index with
        # zero power and the correct thickness.
        stop_to_first_post = surfaces[stop_index].thickness
        dummy = Surface(
            radius=np.inf, conic=0.0,
            semi_diameter=np.inf,
            glass_before='air', glass_after='air',
            is_mirror=False, is_stop=False,
            thickness=stop_to_first_post,
            label='(stop->XP transfer)')
        post_full = [dummy] + post
        M_post, _, _, _ = system_abcd(post_full, wavelength)
        A_post, B_post = float(M_post[0, 0]), float(M_post[0, 1])
        C_post, D_post = float(M_post[1, 0]), float(M_post[1, 1])
        # XP is the image-space conjugate of the stop.  Append
        # T(z_img) on the LEFT (image side) so B_total = 0:
        # T(z) . M = [[A + z*C, B + z*D], [C, D]].  B_new = B + z*D = 0
        # => z_xp = -B_post / D_post.
        if abs(D_post) > 1e-30:
            xp_z = -B_post / D_post
            # Magnification for the image of the stop through post:
            # m_post = D_post when B=0.  XP radius = stop_radius * m_post.
            xp_radius = abs(stop_radius * D_post) if np.isfinite(stop_radius) else float('nan')
        else:
            xp_z = float('inf')
            xp_radius = float('inf')

    return PupilInfo(
        ep_z=float(ep_z), ep_radius=float(ep_radius),
        xp_z=float(xp_z), xp_radius=float(xp_radius),
        stop_index=int(stop_index),
    )


def find_lenses(surfaces, wavelength):
    """Auto-detect individual lens elements in a surface list.

    Scans for air -> glass -> air blocks; each block becomes one
    ``LensInfo``.  Cemented multi-element lenses (glass -> glass
    interfaces in the middle) stay grouped.  Mirrors are treated
    as their own single-surface elements.  Air-only runs (gaps
    between lenses, dummy COORDBRK surfaces) are skipped.

    Limitations
    -----------
    * Pure air -> air surfaces (DOE phase masks represented as
      air-to-air elements, COORDBRK carriers, dummy reference
      planes) are not detected as "lenses" and are silently
      skipped.  That's usually the right thing -- they contribute
      no power -- but a phase-grating element imparts real optical
      power at non-zero diffraction orders that ``find_lenses``
      won't see.
    * A system that ends inside glass (last surface's
      ``glass_after != 'air'``) is malformed for this purpose.  The
      straggling partial block is omitted from the result.
    """
    lenses = []
    n_surf = len(surfaces)
    i = 0
    while i < n_surf:
        s = surfaces[i]
        n_b = get_glass_index(s.glass_before, wavelength)
        n_a = get_glass_index(s.glass_after, wavelength)
        # Detect air -> glass transition (entry of a lens element)
        if abs(n_b - 1.0) < 1e-6 and abs(n_a - 1.0) > 1e-6:
            start = i
            j = i
            end = None
            while j < n_surf:
                sj = surfaces[j]
                nb_j = get_glass_index(sj.glass_before, wavelength)
                na_j = get_glass_index(sj.glass_after, wavelength)
                if abs(nb_j - 1.0) > 1e-6 and abs(na_j - 1.0) < 1e-6:
                    # glass -> air (exit of lens block)
                    end = j
                    break
                j += 1
            if end is None:
                # Malformed: entered glass but never exited.  Skip.
                break
            lenses.append(lens_abcd(surfaces, wavelength,
                                     start=start, end=end))
            i = end + 1
        # Handle a free-standing mirror as its own element
        elif s.is_mirror:
            lenses.append(lens_abcd(surfaces, wavelength, start=i, end=i))
            i += 1
        else:
            i += 1
    return lenses


# ============================================================================
# Seidel aberration coefficients
# ============================================================================

def seidel_coefficients(surfaces, wavelength, object_distance=np.inf,
                        stop_index=None, field_angle=0.01):
    """Compute the five Seidel (third-order) aberration coefficients.

    Uses the Buchdahl-Hopkins formulation based on paraxial marginal
    and chief ray data at each surface.  **Stop-aware** (3.1.11): the
    chief ray is constrained to pass through the centre of the
    declared aperture stop (``y_c = 0`` at the stop surface), and the
    marginal ray fills the stop (``y_m = r_stop`` at the stop).  The
    initial conditions at surface 0 are derived from the pre-stop
    ABCD so both ray constraints are satisfied automatically.

    When the stop is at surface 0 (the legacy assumption), behaviour
    is bit-for-bit backward compatible with 3.1.10.

    Parameters
    ----------
    surfaces : list of Surface
    wavelength : float
        Vacuum wavelength [m].
    object_distance : float
        Object distance from the first surface [m].  ``np.inf`` for
        an object at infinity (collimated input).
    stop_index : int, optional
        Explicit stop surface index.  Defaults to :func:`find_stop`
        -- i.e. the surface flagged ``is_stop=True``, or the first
        surface with a finite ``semi_diameter``, or 0.  When the
        resolved stop is somewhere other than surface 0, the chief
        ray initial conditions are back-propagated through the
        pre-stop ABCD so that ``y_c = 0`` at the stop.
    field_angle : float, default 0.01
        Unreduced field half-angle [rad] for the chief ray.  Only
        the shape of the Seidel sums is reported; absolute
        magnitudes scale linearly with this value (and quadratically
        for astigmatism/Petzval).  0.01 rad (~0.57 deg) is the
        conventional small-angle normalisation.

    Returns
    -------
    seidel : dict
        Keys: ``'S1'`` (spherical), ``'S2'`` (coma), ``'S3'``
        (astigmatism), ``'S4'`` (Petzval), ``'S5'`` (distortion).
        Each value is a 1-D per-surface array.  Also contains:

        * ``'total'`` : dict with the sums.
        * ``'labels'`` : dict with human-readable names.
        * ``'y_marginal'`` / ``'y_chief'`` : per-surface ray heights.
        * ``'stop_index'`` : the stop index used (for diagnostics).
    abcd : ndarray
        System ABCD matrix.
    """
    n_surf = len(surfaces)
    n_first = get_glass_index(surfaces[0].glass_before, wavelength)

    # ---- Resolve the stop surface and its radius ------------------
    if stop_index is None:
        stop_index = find_stop(surfaces)
    if not (0 <= stop_index < n_surf):
        raise ValueError(
            f"seidel_coefficients: stop_index={stop_index} out of "
            f"range [0, {n_surf})")
    r_stop = surfaces[stop_index].semi_diameter
    if not np.isfinite(r_stop):
        # No explicit stop radius declared; fall back to surface 0's
        # semi-diameter for normalisation (legacy behaviour).  The
        # absolute magnitude of the Seidel sums then depends on this
        # default; callers who care should declare a proper stop.
        r_stop = surfaces[0].semi_diameter
        if not np.isfinite(r_stop):
            r_stop = 12.7e-3  # last-resort fallback

    # ---- Pre-stop ABCD (surface 0 -> stop vertex) -----------------
    # system_abcd walks surfaces but only applies the transfer matrix
    # between SURFACES (``if i < len(surfaces) - 1``).  For the
    # pre-stop sub-system we need one additional transfer: from the
    # last pre-stop surface's vertex to the stop's vertex, using the
    # stop-ward glass index.  Build it explicitly.
    if stop_index == 0:
        A_pre, B_pre = 1.0, 0.0
    else:
        M_pre, _, _, _ = system_abcd(surfaces[:stop_index], wavelength)
        # Transfer from surface (stop_index-1) to the stop vertex in
        # the medium on its image side.
        t_last = float(surfaces[stop_index - 1].thickness)
        n_last = get_glass_index(
            surfaces[stop_index - 1].glass_after, wavelength)
        T_last = np.array([[1.0, t_last / n_last],
                            [0.0, 1.0]])
        M_pre = T_last @ M_pre
        A_pre = float(M_pre[0, 0])
        B_pre = float(M_pre[0, 1])

    # ---- Initial conditions at surface 0 --------------------------
    # Marginal ray: on-axis object (u_0 = 0, nu_0 = 0), filling the
    # stop (y_stop = r_stop).  In reduced-coord matrix form:
    #    y_stop = A_pre * y_0 + B_pre * 0  =>  y_0 = r_stop / A_pre.
    # For a finite object the marginal ray is launched from the
    # axial object point; we keep the legacy u-driven form in that
    # case (stop-awareness is a collimated-input concept primarily).
    if np.isinf(object_distance):
        y_m_init = (r_stop / A_pre) if abs(A_pre) > 1e-30 else r_stop
        nu_m_init = 0.0
    else:
        y_m_init = 0.0
        nu_m_init = n_first * r_stop / object_distance

    # Chief ray: from edge of field (angle = field_angle), through
    # centre of stop (y_stop = 0).  system_abcd works in reduced
    # coordinates (y, nu) where nu = n*u, so the transfer is
    #    y_stop = A_pre * y_0 + B_pre * nu_0
    # Setting y_stop = 0 with nu_0 = n_first * field_angle gives
    #    y_0 = -B_pre * nu_0 / A_pre = -B_pre * n_first * field_angle / A_pre.
    u_0_c = float(field_angle)
    nu_c_init = n_first * u_0_c
    if abs(A_pre) > 1e-30:
        y_c_init = -B_pre * nu_c_init / A_pre
    else:
        y_c_init = 0.0

    y_m = np.zeros(n_surf)
    nu_m = np.zeros(n_surf)  # n*u product

    y_c = np.zeros(n_surf)
    nu_c = np.zeros(n_surf)  # chief ray

    # Trace marginal and chief rays
    y_val_m = y_m_init
    nu_val_m = nu_m_init
    y_val_c = y_c_init
    nu_val_c = nu_c_init

    # Per-surface Seidel contributions
    S1 = np.zeros(n_surf)
    S2 = np.zeros(n_surf)
    S3 = np.zeros(n_surf)
    S4 = np.zeros(n_surf)
    S5 = np.zeros(n_surf)

    for i, surf in enumerate(surfaces):
        n1 = get_glass_index(surf.glass_before, wavelength)
        n2 = get_glass_index(surf.glass_after, wavelength)
        R = surf.radius

        # Store ray heights at this surface
        y_m[i] = y_val_m
        nu_m[i] = nu_val_m
        y_c[i] = y_val_c
        nu_c[i] = nu_val_c

        if np.isfinite(R) and not surf.is_mirror:
            c = 1.0 / R  # curvature

            # Incidence angle (paraxial): i = c*y + u = c*y + nu/n
            u_m = nu_val_m / n1
            u_c = nu_val_c / n1

            i_m = c * y_val_m + u_m
            i_c = c * y_val_c + u_c

            # Refract
            nu_m_after = nu_val_m - y_val_m * (n2 - n1) * c
            nu_c_after = nu_val_c - y_val_c * (n2 - n1) * c

            u_m_after = nu_m_after / n2
            u_c_after = nu_c_after / n2

            i_m_after = c * y_val_m + u_m_after
            i_c_after = c * y_val_c + u_c_after

            # Abbe invariant
            A_m = n1 * i_m  # = n2 * i_m_after (Snell)
            A_c = n1 * i_c

            # Delta(u/n): 1/n2 - 1/n1
            delta_un = 1.0 / n2 - 1.0 / n1

            # Seidel sums (Hopkins notation)
            # S_I   = -A_m^2 * y_m * (u_m_after/n2 - u_m/n1)
            h = y_val_m
            hbar = y_val_c

            delta_un_m = u_m_after - u_m  # this is (nu_after - nu_before) * ...

            # Standard Hopkins/Welford per-surface Seidel sums (the
            # Lagrange-invariant H^2 factor is pulled out of the sum
            # by convention, so each term is H^2-less):
            #    S_I   = -A_m^2 * h * delta(u/n)           spherical
            #    S_II  = -A_m A_c h * delta(u/n)           coma
            #    S_III = -A_c^2 * h * delta(u/n)           astigmatism
            #    S_IV  = -c * (n2 - n1) / (n1 * n2)        Petzval
            #    S_V   = -(A_c / A_m) * (S_III + S_IV)     distortion
            # The pre-3.1.8 code assigned S4[i] twice; the first line
            # had (n2-n1) * (1/n2 - 1/n1) which squares the index
            # difference and is wrong -- the second line, which wins,
            # uses the correct -(n2-n1)/(n1 n2) * c form above.  The
            # errant first line is removed to make the intent
            # unambiguous.
            S1[i] = -(A_m ** 2) * h * delta_un
            S2[i] = -(A_m * A_c) * h * delta_un
            S3[i] = -(A_c ** 2) * h * delta_un
            S4[i] = -(1.0 / (n2 * n1)) * c * (n2 - n1)
            S5[i] = -(A_c / A_m) * (S3[i] + S4[i]) if abs(A_m) > 1e-30 else 0.0

            nu_val_m = nu_m_after
            nu_val_c = nu_c_after

        elif surf.is_mirror and np.isfinite(R):
            c = 1.0 / R
            u_m = nu_val_m / n1
            u_c = nu_val_c / n1
            i_m = c * y_val_m + u_m
            i_c = c * y_val_c + u_c

            # Mirror refraction: n2 = -n1 (sign reversal)
            phi = 2.0 * n1 * c
            nu_m_after = nu_val_m - y_val_m * phi
            nu_c_after = nu_val_c - y_val_c * phi

            nu_val_m = nu_m_after
            nu_val_c = nu_c_after
        else:
            # Flat surface: no power, no aberration contribution
            nu_m_after = nu_val_m - y_val_m * (n2 - n1) * 0  # zero power
            nu_c_after = nu_val_c - y_val_c * (n2 - n1) * 0
            nu_val_m = nu_m_after
            nu_val_c = nu_c_after

        # Transfer to next surface
        if i < len(surfaces) - 1:
            t = surf.thickness
            n_after = n2
            u_m_t = nu_val_m / n_after
            u_c_t = nu_val_c / n_after
            y_val_m = y_val_m + u_m_t * t
            y_val_c = y_val_c + u_c_t * t

    abcd, _, _, _ = system_abcd(surfaces, wavelength)

    return {
        'S1': S1, 'S2': S2, 'S3': S3, 'S4': S4, 'S5': S5,
        'total': {
            'S1': np.sum(S1), 'S2': np.sum(S2), 'S3': np.sum(S3),
            'S4': np.sum(S4), 'S5': np.sum(S5),
        },
        'labels': {
            'S1': 'Spherical', 'S2': 'Coma', 'S3': 'Astigmatism',
            'S4': 'Petzval', 'S5': 'Distortion',
        },
        'y_marginal': y_m,
        'y_chief': y_c,
        'stop_index': stop_index,
    }, abcd


def seidel_prescription(prescription, wavelength, object_distance=np.inf,
                        stop_index=None, field_angle=0.01):
    """Compute Seidel coefficients from a lens prescription dict.

    Passes ``stop_index`` and ``field_angle`` through to
    :func:`seidel_coefficients`; the prescription dict's own
    ``stop_index`` key (used by the wave-optics pipeline) is
    propagated onto the surface list automatically via
    :func:`surfaces_from_prescription` and picked up by
    :func:`find_stop` when ``stop_index`` is left ``None`` here.
    """
    surfaces = surfaces_from_prescription(prescription)
    return seidel_coefficients(
        surfaces, wavelength, object_distance=object_distance,
        stop_index=stop_index, field_angle=field_angle)


# ============================================================================
# Analysis: spot diagram
# ============================================================================

def spot_rms(result):
    """Compute RMS spot radius from a trace result.

    Parameters
    ----------
    result : TraceResult

    Returns
    -------
    rms : float
        RMS spot radius [m] at the final surface.
    centroid : tuple (cx, cy)
        Spot centroid [m].
    """
    r = result.image_rays
    alive = r.alive
    if not np.any(alive):
        return np.inf, (0.0, 0.0)

    cx = np.mean(r.x[alive])
    cy = np.mean(r.y[alive])

    dx = r.x[alive] - cx
    dy = r.y[alive] - cy
    rms = np.sqrt(np.mean(dx ** 2 + dy ** 2))

    return rms, (cx, cy)


def spot_geo_radius(result):
    """Compute the geometric (maximum) spot radius.

    Parameters
    ----------
    result : TraceResult

    Returns
    -------
    geo_radius : float
        Maximum distance from centroid [m].
    """
    r = result.image_rays
    alive = r.alive
    if not np.any(alive):
        return np.inf

    cx = np.mean(r.x[alive])
    cy = np.mean(r.y[alive])
    dist = np.sqrt((r.x[alive] - cx) ** 2 + (r.y[alive] - cy) ** 2)
    return np.max(dist)


def spot_diagram(result, ax=None, title=None, units='um', **kwargs):
    """Plot a spot diagram from a trace result.

    Parameters
    ----------
    result : TraceResult
    ax : matplotlib Axes or None
        If None, creates a new figure.
    title : str or None
    units : str
        ``'um'`` (micrometres) or ``'mm'`` (millimetres).
    **kwargs
        Passed to ``ax.scatter()``.

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt

    scale = {'um': 1e6, 'mm': 1e3, 'm': 1.0}[units]
    label = {'um': '\u00b5m', 'mm': 'mm', 'm': 'm'}[units]

    r = result.image_rays
    alive = r.alive

    rms, (cx, cy) = spot_rms(result)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.figure

    scatter_kw = dict(s=4, alpha=0.6, edgecolors='none')
    scatter_kw.update(kwargs)

    ax.scatter((r.x[alive] - cx) * scale,
               (r.y[alive] - cy) * scale,
               **scatter_kw)

    ax.set_xlabel(f'x [{label}]')
    ax.set_ylabel(f'y [{label}]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if title is None:
        n_alive = np.sum(alive)
        title = (f'Spot Diagram  ({n_alive}/{r.n_rays} rays)\n'
                 f'RMS = {rms * scale:.3f} {label},  '
                 f'GEO = {spot_geo_radius(result) * scale:.3f} {label}')
    ax.set_title(title)

    # Draw Airy disc for reference
    airy_r = 1.22 * result.wavelength / (
        2 * result.surfaces[0].semi_diameter) if np.isfinite(
        result.surfaces[0].semi_diameter) else None
    if airy_r is not None and airy_r * scale < ax.get_xlim()[1] * 5:
        circle = plt.Circle((0, 0), airy_r * scale,
                             fill=False, color='red', linestyle='--',
                             linewidth=0.8, label=f'Airy ({airy_r*scale:.3f} {label})')
        ax.add_patch(circle)
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig, ax


# ============================================================================
# Analysis: ray fan (transverse aberration) plots
# ============================================================================

def ray_fan_data(surfaces, wavelength, semi_aperture, field_angle=0.0,
                 n_rays=101):
    """Compute transverse ray aberration vs normalised pupil coordinate.

    Parameters
    ----------
    surfaces : list of Surface
    wavelength : float
    semi_aperture : float
    field_angle : float
    n_rays : int

    Returns
    -------
    py : ndarray
        Normalised pupil coordinate in Y (tangential fan).
    ey : ndarray
        Transverse ray error in Y [m] (tangential).
    px : ndarray
        Normalised pupil coordinate in X (sagittal fan).
    ex : ndarray
        Transverse ray error in X [m] (sagittal).
    """
    # Tangential fan (Y)
    fan_y = make_fan('y', semi_aperture, n_rays, field_angle, wavelength)
    res_y = trace(fan_y, surfaces, wavelength)
    img_y = res_y.image_rays

    # Reference: chief ray position
    chief = make_ray(0, 0, 0, np.sin(field_angle), wavelength)
    res_chief = trace(chief, surfaces, wavelength)
    y_ref = res_chief.image_rays.y[0]
    x_ref = res_chief.image_rays.x[0]

    py = np.linspace(-1, 1, n_rays)
    ey = np.where(img_y.alive, img_y.y - y_ref, np.nan)

    # Sagittal fan (X)
    fan_x = make_fan('x', semi_aperture, n_rays, field_angle, wavelength)
    res_x = trace(fan_x, surfaces, wavelength)
    img_x = res_x.image_rays

    px = np.linspace(-1, 1, n_rays)
    ex = np.where(img_x.alive, img_x.x - x_ref, np.nan)

    return py, ey, px, ex


def ray_fan_plot(surfaces, wavelength, semi_aperture, field_angles=None,
                 n_rays=101, ax=None, units='um'):
    """Plot transverse ray aberration fans.

    Parameters
    ----------
    surfaces : list of Surface
    wavelength : float
    semi_aperture : float
    field_angles : list of float or None
        Field angles [rad] to plot.  Default: [0].
    n_rays : int
    ax : pair of Axes or None
        ``(ax_tangential, ax_sagittal)``.
    units : str

    Returns
    -------
    fig : Figure
    axes : pair of Axes
    """
    import matplotlib.pyplot as plt

    if field_angles is None:
        field_angles = [0.0]

    scale = {'um': 1e6, 'mm': 1e3, 'm': 1.0}[units]
    label = {'um': '\u00b5m', 'mm': 'mm', 'm': 'm'}[units]

    if ax is None:
        fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        ax_t, ax_s = ax
        fig = ax_t.figure

    for fa in field_angles:
        py, ey, px, ex = ray_fan_data(surfaces, wavelength, semi_aperture,
                                      fa, n_rays)
        fa_deg = np.degrees(fa)
        ax_t.plot(py, ey * scale, label=f'{fa_deg:.1f}\u00b0')
        ax_s.plot(px, ex * scale, label=f'{fa_deg:.1f}\u00b0')

    ax_t.set_xlabel('Normalised pupil (PY)')
    ax_t.set_ylabel(f'EY [{label}]')
    ax_t.set_title('Tangential ray fan')
    ax_t.axhline(0, color='k', linewidth=0.5)
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(fontsize=8)

    ax_s.set_xlabel('Normalised pupil (PX)')
    ax_s.set_ylabel(f'EX [{label}]')
    ax_s.set_title('Sagittal ray fan')
    ax_s.axhline(0, color='k', linewidth=0.5)
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(fontsize=8)

    fig.tight_layout()
    return fig, (ax_t, ax_s)


def ray_fan_plot_prescription(prescription, wavelength, field_angles=None,
                              n_rays=101, units='um'):
    """Ray fan plot from a lens prescription dict."""
    surfaces = surfaces_from_prescription(prescription)
    ap = prescription.get('aperture_diameter')
    sa = ap / 2.0 if ap else 12.7e-3
    return ray_fan_plot(surfaces, wavelength, sa, field_angles, n_rays,
                        units=units)


# ============================================================================
# Analysis: OPD (wavefront error)
# ============================================================================

def opd_fan_data(surfaces, wavelength, semi_aperture, field_angle=0.0,
                 n_rays=101):
    """Compute OPD vs pupil coordinate for tangential and sagittal fans.

    Parameters
    ----------
    surfaces : list of Surface
    wavelength : float
    semi_aperture : float
    field_angle : float
    n_rays : int

    Returns
    -------
    py, opd_y, px, opd_x : ndarray
        Normalised pupil and OPD [waves] for each fan.
    """
    # Tangential fan
    fan_y = make_fan('y', semi_aperture, n_rays, field_angle, wavelength)
    res_y = trace(fan_y, surfaces, wavelength)
    img_y = res_y.image_rays

    # Chief ray reference OPD
    chief = make_ray(0, 0, 0, np.sin(field_angle), wavelength)
    res_chief = trace(chief, surfaces, wavelength)
    opd_ref = res_chief.image_rays.opd[0]

    py = np.linspace(-1, 1, n_rays)
    opd_y = np.where(img_y.alive, (img_y.opd - opd_ref) / wavelength, np.nan)

    # Sagittal fan
    fan_x = make_fan('x', semi_aperture, n_rays, field_angle, wavelength)
    res_x = trace(fan_x, surfaces, wavelength)
    img_x = res_x.image_rays

    px = np.linspace(-1, 1, n_rays)
    opd_x = np.where(img_x.alive, (img_x.opd - opd_ref) / wavelength, np.nan)

    return py, opd_y, px, opd_x


# ============================================================================
# Analysis: through-focus spot
# ============================================================================

def refocus(result, delta_z, wavelength=None):
    """Project the final bundle of a traced result to an image plane
    at ``delta_z`` downstream of the last surface's vertex, returning
    a new ``TraceResult``.

    Equivalent to appending a flat image-plane surface at
    ``thickness=delta_z`` after the last refracting surface and
    re-tracing, but closed-form -- each ray is advanced in a straight
    line from its current (post-refraction) position to the target
    z-plane, using its (L, M, N) direction cosines.  Orders of
    magnitude cheaper than the retrace when used in a focus sweep.

    Parameters
    ----------
    result : TraceResult
        Output of a previous :func:`trace` call.  Must have the
        final ray bundle available (``result.image_rays``); works
        with both ``output_filter='all'`` and ``output_filter='last'``.
    delta_z : float
        Axial distance from the last surface's VERTEX to the image
        plane [m].  Signed -- pass a negative value to move *toward*
        the lens (pre-focus).  Note: rays after :func:`trace` end at
        ``z = sag(h)`` of the last surface (not at the vertex plane
        z=0), so the effective arc length each ray travels is
        ``(delta_z - sag(h)) / N``, not ``delta_z / N``.
        ``refocus`` handles the sag-to-vertex correction internally,
        so the caller just specifies the target image distance and
        the math Just Works on curved exit surfaces.
    wavelength : float, optional
        Wavelength for resolving the image-space refractive index.
        Defaults to ``result.wavelength`` if unset.  If the
        image-space medium is glass rather than air (rare -- only
        relevant for tests that place the "image" inside a refractive
        element), the OPL update uses the correct n.

    Returns
    -------
    new_result : TraceResult
        Same surface list, same input rays, same wavelength.
        ``new_result.image_rays`` is the refocused bundle at
        z = delta_z (in the last surface's frame).

    Notes
    -----
    * Rays that were at z = sag (off-axis on a curved exit surface)
      travel a slightly longer path than rays at z = 0 (on-axis).
      The correction ``(delta_z - z_start)`` in the transfer keeps
      both paths geometrically consistent -- this is what matches
      the full-retrace behaviour that appends a flat image plane
      at ``thickness=delta_z`` after the last surface.
    * The OPL update is ``n * arc_length`` where arc length is the
      ray-path distance from (x, y, z_start) to the image plane.
      Signed, so negative ``delta_z`` subtracts OPL as expected.
    * For GRIN or highly aberrated image spaces where the
      "last-medium-is-uniform" assumption fails, use a full
      :func:`trace` with the image plane inserted rather than
      ``refocus``.
    """
    if wavelength is None:
        wavelength = result.wavelength
    n_image = get_glass_index(result.surfaces[-1].glass_after, wavelength)

    last = result.image_rays.copy()
    # Advance each ray to the image plane at z = delta_z (measured
    # from the last surface's vertex).  Rays currently sit at
    # z = sag(h), so the axial distance to travel is
    # (delta_z - z_current), and the arc length along each ray is
    # (delta_z - z_current) / N.
    with np.errstate(divide='ignore', invalid='ignore'):
        dz_remaining = delta_z - last.z
        t = np.where(last.alive & (np.abs(last.N) > 1e-30),
                     dz_remaining / last.N, 0.0)
    last.x = last.x + last.L * t
    last.y = last.y + last.M * t
    last.z = last.z + last.N * t
    # Signed OPL update: +n*t moves forward; for t < 0 this is a
    # physical "undo" of part of the previous propagation leg.
    last.opd = last.opd + n_image * t

    # Splice the refocused bundle into ray_history.  When the source
    # result used output_filter='last', ray_history has a single
    # entry; we overwrite it.  Otherwise we replace the last entry
    # (image_rays) only, leaving upstream surface-by-surface state
    # intact for callers that want it.
    if len(result.ray_history) <= 1:
        new_history = [last]
    else:
        new_history = list(result.ray_history[:-1]) + [last]

    return TraceResult(
        surfaces=result.surfaces,
        ray_history=new_history,
        input_rays=result.input_rays,
        wavelength=result.wavelength,
    )


def through_focus_rms(surfaces, wavelength, semi_aperture,
                      focus_shifts, field_angle=0.0,
                      num_rings=6, rays_per_ring=36):
    """Compute RMS spot size at a series of focus positions.

    Useful for finding best focus.

    Performance note (3.1.8)
    ------------------------
    Earlier versions rebuilt the entire surface list on every focus
    shift and retraced from surface 0.  This version traces once
    (through the real surfaces only) and uses :func:`refocus` for
    each shift -- effectively a closed-form straight-line transfer
    in the image-space medium.  Speedup is roughly proportional to
    the number of surfaces (typically 5-20x).  Numerical output is
    identical to the pre-3.1.8 path on well-behaved systems, since
    ``refocus`` is the exact operator that ``trace`` would apply for
    the extra image-plane transfer surface.

    Parameters
    ----------
    surfaces : list of Surface
    wavelength : float
    semi_aperture : float
    focus_shifts : array-like
        Image distances [m] from the last surface vertex.  Pass
        e.g. ``bfl + np.linspace(-1e-3, 1e-3, 51)`` to scan around
        the paraxial focus.
    field_angle : float
    num_rings, rays_per_ring : int

    Returns
    -------
    shifts : ndarray
        Image distances [m].
    rms_values : ndarray
        RMS spot radius [m] at each position.
    best_shift : float
        Image distance giving minimum RMS.
    """
    focus_shifts = np.asarray(focus_shifts, dtype=np.float64)
    rms_values = np.zeros_like(focus_shifts)

    rays = make_rings(semi_aperture, num_rings, rays_per_ring,
                      field_angle, wavelength)

    # Single base trace through the surfaces as specified.  Use
    # output_filter='last' because we only need the final bundle
    # for refocus + spot_rms.  Saves ~N_surfaces memory copies on
    # large ring counts.
    base = trace(rays, surfaces, wavelength, output_filter='last')

    for j, img_dist in enumerate(focus_shifts):
        shifted = refocus(base, float(img_dist), wavelength=wavelength)
        rms_values[j], _ = spot_rms(shifted)

    best_idx = np.argmin(rms_values)
    return focus_shifts, rms_values, focus_shifts[best_idx]


# ============================================================================
# Analysis: find paraxial focus
# ============================================================================

def find_paraxial_focus(surfaces, wavelength):
    """Find the paraxial image distance from the last surface.

    Parameters
    ----------
    surfaces : list of Surface
    wavelength : float

    Returns
    -------
    image_distance : float
        Axial distance from the last surface vertex to paraxial focus [m].
    """
    _, _, bfl, _ = system_abcd(surfaces, wavelength)
    return bfl


# ============================================================================
# Utility: trace summary
# ============================================================================

def trace_summary(result, units='mm'):
    """Print a summary of the trace result.

    Parameters
    ----------
    result : TraceResult
    units : str
        ``'mm'`` or ``'um'``.
    """
    scale = {'um': 1e6, 'mm': 1e3, 'm': 1.0}[units]
    label = {'um': '\u00b5m', 'mm': 'mm', 'm': 'm'}[units]

    rms, (cx, cy) = spot_rms(result)
    geo = spot_geo_radius(result)

    final = result.image_rays
    n_alive = int(np.sum(final.alive))
    n_total = final.n_rays
    vignetting = 100 * (1 - n_alive / n_total)

    # Break down the loss by cause if error_code is available
    # (added 3.1.9).  Pre-3.1.9 bundles may lack the field; fall
    # back silently to an aggregate vignetting number.
    ec = getattr(final, 'error_code', None)
    if ec is not None and n_alive < n_total:
        n_tir   = int(np.sum(ec == RAY_TIR))
        n_ap    = int(np.sum(ec == RAY_APERTURE))
        n_miss  = int(np.sum(ec == RAY_MISSED_SURFACE))
        n_nan   = int(np.sum(ec == RAY_NAN))
        loss_detail = (f" [TIR={n_tir}, aperture={n_ap}, "
                        f"miss={n_miss}, nan={n_nan}]")
    else:
        loss_detail = ''

    print(f"Ray trace summary")
    print(f"  Wavelength:   {result.wavelength * 1e9:.2f} nm")
    print(f"  Surfaces:     {len(result.surfaces)}")
    print(f"  Rays:         {n_alive}/{n_total} alive "
          f"({vignetting:.1f}% lost{loss_detail})")
    print(f"  Centroid:     ({cx * scale:.4f}, {cy * scale:.4f}) {label}")
    print(f"  RMS spot:     {rms * scale:.4f} {label}")
    print(f"  GEO radius:   {geo * scale:.4f} {label}")

    # Airy disc
    sd = result.surfaces[0].semi_diameter
    if np.isfinite(sd):
        na = sd  # approximate entrance pupil radius
        airy = 1.22 * result.wavelength / (2 * na)
        print(f"  Airy radius:  {airy * scale:.4f} {label}")
        print(f"  Spot/Airy:    {rms / airy:.2f}")


def prescription_summary(prescription, wavelength, units='mm'):
    """Print a system summary from a prescription dict.

    Parameters
    ----------
    prescription : dict
    wavelength : float
    units : str
    """
    scale = {'um': 1e6, 'mm': 1e3, 'm': 1.0}[units]
    label = {'um': '\u00b5m', 'mm': 'mm', 'm': 'm'}[units]

    surfaces = surfaces_from_prescription(prescription)
    abcd, efl, bfl, ffl = system_abcd(surfaces, wavelength)

    name = prescription.get('name', 'Unnamed')
    print(f"System: {name}")
    print(f"  Wavelength:   {wavelength * 1e9:.2f} nm")
    print(f"  Surfaces:     {len(surfaces)}")
    print(f"  EFL:          {efl * scale:.4f} {label}")
    print(f"  BFL:          {bfl * scale:.4f} {label}")
    print(f"  FFL:          {ffl * scale:.4f} {label}")
    print(f"  ABCD matrix:")
    print(f"    A = {abcd[0,0]:.6f}   B = {abcd[0,1] * scale:.6f} {label}")
    print(f"    C = {abcd[1,0] / scale:.6f} 1/{label}   D = {abcd[1,1]:.6f}")

    ap = prescription.get('aperture_diameter')
    if ap:
        f_number = abs(efl) / ap if np.isfinite(efl) else np.inf
        print(f"  Aperture:     {ap * scale:.4f} {label}")
        print(f"  f/#:          {f_number:.2f}")


# ============================================================================
# Compatibility bridge: system.py element-list format → Surface list
# ============================================================================

def surfaces_from_elements(elements, wavelength):
    """Convert a ``propagate_through_system`` element list to Surfaces.

    This allows the same element-list used for wave-optics simulation
    to be ray-traced geometrically, enabling quick cross-validation::

        # Wave-optics
        E_out, _ = propagate_through_system(E_in, elements, wv, dx)

        # Geometric ray trace — same element list
        result = raytrace_system(elements, wv, semi_aperture=5e-3)
        spot_diagram(result)

    Supported element types:

    - ``'propagate'`` — free-space gap (converted to thickness on the
      preceding surface).
    - ``'lens'`` — thin lens (one surface with power = 1/f).
    - ``'real_lens'`` — multi-surface prescription (expanded in-line).
    - ``'mirror'`` — flat or curved reflector.
    - ``'aperture'`` — sets the semi-diameter of the preceding surface.

    Parameters
    ----------
    elements : list of dict
        Element list in the same format as :func:`system.propagate_through_system`.
    wavelength : float
        Vacuum wavelength [m] (needed to resolve glass indices for
        real-lens prescriptions).

    Returns
    -------
    surfaces : list of Surface
        Sequential surface list for :func:`trace`.
    """
    surfaces = []
    pending_thickness = 0.0  # accumulated free-space before next surface

    for elem in elements:
        etype = elem['type']

        if etype in ('propagate', 'propagate_tilted'):
            pending_thickness += elem['z']

        elif etype == 'lens':
            f = elem['f']
            # A thin lens is a flat surface with power phi = 1/f.
            # We model it as two flat air→air surfaces separated by zero
            # thickness, with the refraction equivalent encoded as a
            # curved surface with R = 2*f (mirror equivalent of a thin lens).
            # Simpler: single surface with radius = -f (for convergent).
            # Actually the cleanest approach: use the ABCD-equivalent
            # pair: a flat surface that applies the thin-lens deflection.
            # For ray tracing, a thin lens with focal length f is equivalent
            # to a curved mirror surface with R = 2f, but since we want
            # refraction not reflection, we use a single surface with
            # R such that phi = (n2-n1)/R = 1/f.  With n1=n2=1 (air),
            # this doesn't work.  Instead, we encode thin lenses as two
            # surfaces of a fictitious glass element:
            # Surface 1: R = f*(n-1)/1 = 2*f, glass air→glass (n=2)
            # This is too hacky.  Better: just store the focal length
            # and handle thin lenses specially in the trace engine.
            #
            # Pragmatic solution: approximate a thin lens as a very thin
            # high-index singlet.  With d≈0 and n_lens chosen so
            # 1/f = (n-1)*(1/R1 - 1/R2), a symmetric biconvex with
            # R1 = -R2 = R gives 1/f = (n-1)*2/R → R = 2*f*(n-1).
            # With n=1.5, R = f.  This is exact in the paraxial limit.
            n_thin = 1.5
            R_val = f  # R = 2*f*(n-1) = 2*f*0.5 = f for n=1.5
            sd = np.inf
            if 'aperture_diameter' in elem:
                sd = elem['aperture_diameter'] / 2.0

            # Flush any pending thickness
            if surfaces:
                surfaces[-1].thickness += pending_thickness
            pending_thickness = 0.0

            surfaces.append(Surface(
                radius=R_val, conic=0.0, semi_diameter=sd,
                glass_before='air', glass_after='__thin_lens__',
                thickness=0.0, label=f'Lens f={f*1e3:.1f}mm (front)',
            ))
            surfaces.append(Surface(
                radius=-R_val, conic=0.0, semi_diameter=sd,
                glass_before='__thin_lens__', glass_after='air',
                thickness=0.0, label=f'Lens f={f*1e3:.1f}mm (back)',
            ))

        elif etype == 'real_lens':
            rx = elem['prescription']
            rx_surfaces = surfaces_from_prescription(rx)

            # Flush pending thickness
            if surfaces:
                surfaces[-1].thickness += pending_thickness
            pending_thickness = 0.0

            surfaces.extend(rx_surfaces)

        elif etype == 'mirror':
            R = elem.get('radius', np.inf)
            sd = np.inf
            if 'aperture_diameter' in elem:
                sd = elem['aperture_diameter'] / 2.0

            if surfaces:
                surfaces[-1].thickness += pending_thickness
            pending_thickness = 0.0

            surfaces.append(Surface(
                radius=R if R is not None else np.inf,
                conic=elem.get('conic', 0.0),
                semi_diameter=sd,
                glass_before='air', glass_after='air',
                is_mirror=True, thickness=0.0,
                label='Mirror',
            ))

        elif etype == 'aperture':
            # Apply aperture as a semi-diameter constraint on the
            # most recent surface, or add a dummy flat surface.
            params = elem.get('params', {})
            diameter = params.get('diameter', np.inf)
            sd = diameter / 2.0 if np.isfinite(diameter) else np.inf

            if surfaces:
                surfaces[-1].thickness += pending_thickness
            pending_thickness = 0.0

            surfaces.append(Surface(
                radius=np.inf, semi_diameter=sd,
                glass_before='air', glass_after='air',
                thickness=0.0, label='Aperture',
            ))

        elif etype == 'spherical_lens':
            if surfaces:
                surfaces[-1].thickness += pending_thickness
            pending_thickness = 0.0

            n_lens = elem['n_lens']
            sd = np.inf
            if 'aperture_diameter' in elem:
                sd = elem['aperture_diameter'] / 2.0

            # Register the glass temporarily
            _glass_name = f'__spherical_{id(elem)}'
            from .glass import GLASS_REGISTRY, _glass_cache
            # Store a fixed-index material
            _register_fixed_index(_glass_name, n_lens, wavelength)

            surfaces.append(Surface(
                radius=elem['R1'], conic=0.0, semi_diameter=sd,
                glass_before='air', glass_after=_glass_name,
                thickness=elem['d'],
                label='Spherical lens (front)',
            ))
            surfaces.append(Surface(
                radius=elem['R2'], conic=0.0, semi_diameter=sd,
                glass_before=_glass_name, glass_after='air',
                thickness=0.0,
                label='Spherical lens (back)',
            ))

        elif etype == 'aspheric_lens':
            if surfaces:
                surfaces[-1].thickness += pending_thickness
            pending_thickness = 0.0

            n_lens = elem['n_lens']
            sd = np.inf
            if 'aperture_diameter' in elem:
                sd = elem['aperture_diameter'] / 2.0

            _glass_name = f'__aspheric_{id(elem)}'
            _register_fixed_index(_glass_name, n_lens, wavelength)

            surfaces.append(Surface(
                radius=elem['R1'],
                conic=elem.get('k1', 0.0),
                aspheric_coeffs=elem.get('A1'),
                semi_diameter=sd,
                glass_before='air', glass_after=_glass_name,
                thickness=elem['d'],
                label='Aspheric lens (front)',
            ))
            surfaces.append(Surface(
                radius=elem['R2'],
                conic=elem.get('k2', 0.0),
                aspheric_coeffs=elem.get('A2'),
                semi_diameter=sd,
                glass_before=_glass_name, glass_after='air',
                thickness=0.0,
                label='Aspheric lens (back)',
            ))

        # Silently skip unsupported element types (mask, zernike, etc.)
        # — these have no geometric-optics equivalent.

    # Flush any trailing thickness
    if surfaces and pending_thickness > 0:
        surfaces[-1].thickness += pending_thickness

    return surfaces


# Thin-lens helper: register a fixed-index "glass" for spherical/aspheric lenses
def _register_fixed_index(name, n, wavelength):
    """Register a fixed refractive index as a temporary glass entry."""
    from .glass import GLASS_REGISTRY, _glass_cache

    class _FixedIndex:
        def __init__(self, n_val):
            self._n = n_val
        def get_refractive_index(self, wv_nm, unit='nm'):
            return self._n

    GLASS_REGISTRY[name] = ('__fixed__', '__fixed__', '__fixed__')
    _glass_cache[name] = _FixedIndex(n)

# Also register the thin-lens pseudo-glass
_register_fixed_index('__thin_lens__', 1.5, 550e-9)


def raytrace_system(elements, wavelength, semi_aperture=None,
                    field_angle=0.0, num_rings=6, rays_per_ring=36,
                    ray_pattern='rings', n_across=11,
                    image_distance=None):
    """Ray-trace the same element list used by propagate_through_system.

    This is the geometric-optics counterpart to
    :func:`system.propagate_through_system`.  It accepts the same
    element-list format, converts it to a sequential surface list,
    generates rays, and traces them.

    Parameters
    ----------
    elements : list of dict
        Element list (same format as ``propagate_through_system``).
    wavelength : float
        Vacuum wavelength [m].
    semi_aperture : float or None
        Entrance pupil semi-diameter [m].  If None, inferred from the
        first aperture or lens element.
    field_angle : float
        Off-axis field angle [radians].
    num_rings, rays_per_ring, ray_pattern, n_across : int/str
        Ray generation parameters (see :func:`trace_prescription`).
    image_distance : float or None
        Distance from last surface to image plane [m].  If None, uses
        the paraxial back focal length.

    Returns
    -------
    result : TraceResult
    surfaces : list of Surface
        The converted surface list (useful for further analysis).
    """
    surfaces = surfaces_from_elements(elements, wavelength)

    if not surfaces:
        raise ValueError("No traceable surfaces found in the element list.")

    # Infer semi-aperture if not given
    if semi_aperture is None:
        for s in surfaces:
            if np.isfinite(s.semi_diameter):
                semi_aperture = s.semi_diameter
                break
        if semi_aperture is None:
            semi_aperture = 12.7e-3

    # Find image distance
    if image_distance is None:
        try:
            _, _, bfl, _ = system_abcd(surfaces, wavelength)
            if np.isfinite(bfl) and bfl > 0:
                image_distance = bfl
        except Exception:
            pass

    # Generate rays
    if ray_pattern == 'rings':
        rays = make_rings(semi_aperture, num_rings, rays_per_ring,
                          field_angle, wavelength)
    elif ray_pattern == 'grid':
        rays = make_grid(semi_aperture, n_across, field_angle,
                         wavelength, pattern='circular')
    else:
        rays = make_rings(semi_aperture, num_rings, rays_per_ring,
                          field_angle, wavelength)

    # Add image plane if we have a distance
    if image_distance is not None and surfaces:
        last_glass = surfaces[-1].glass_after
        surfaces[-1].thickness = image_distance
        surfaces.append(Surface(
            radius=np.inf, semi_diameter=np.inf,
            glass_before=last_glass, glass_after=last_glass,
            label='Image',
        ))

    result = trace(rays, surfaces, wavelength)
    return result, surfaces
