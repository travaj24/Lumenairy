"""
Hybrid wave/ray optical-design optimization.

Combines the fast, differentiable-in-parameters paraxial ray trace
(``raytrace`` module) with the full wave-optics propagation
(``apply_real_lens`` / ``apply_real_lens_traced``) to optimize lens
prescriptions against a user-specified merit function.

Architecture
------------
A lens design is specified by a *parameter vector* mapped onto a
*prescription template*.  :class:`DesignParameterization` handles the
mapping:

    free params     ->   prescription dict (for apply_real_lens etc.)

Each iteration the optimizer:

    1. Builds the current prescription from the parameter vector.
    2. Evaluates fast geometric figures (focal length, Seidel
       coefficients, ray fans) via the ray tracer.
    3. Optionally evaluates wave figures (Strehl ratio at best
       focus, RMS wavefront error via Zernike decomposition, spot
       size in a through-focus scan) via the wave-optics path.
    4. Combines these into a scalar merit via a sum of
       :class:`MeritTerm` objects, each weighted.

``scipy.optimize.minimize`` (or ``scipy.optimize.least_squares`` for
Gauss-Newton / Levenberg-Marquardt) drives the parameter updates.
Finite-difference gradients are used by default; users can supply
an analytic Jacobian where available.

Typical usage
-------------

.. code-block:: python

    import lumenairy as op
    from lumenairy.optimize import (
        DesignParameterization, design_optimize,
        FocalLengthMerit, StrehlMerit, RMSWavefrontMerit,
    )

    # Start from a Thorlabs AC254-100-C achromat, free up R1/R2/R3/d1.
    template = op.thorlabs_lens('AC254-100-C')
    template['aperture_diameter'] = 10e-3

    param = DesignParameterization(template,
        free_vars=[
            ('surfaces', 0, 'radius'),
            ('surfaces', 1, 'radius'),
            ('surfaces', 2, 'radius'),
            ('thicknesses', 0),
        ],
        bounds=[
            (50e-3, 80e-3),
            (-60e-3, -30e-3),
            (-250e-3, -150e-3),
            (4e-3, 8e-3),
        ])

    merit = [
        FocalLengthMerit(target=100e-3, weight=1.0),
        StrehlMerit(min_strehl=0.95, weight=10.0),
        RMSWavefrontMerit(max_rms_waves=0.05, weight=50.0),
    ]

    result = design_optimize(param, merit,
                             wavelength=1.31e-6,
                             N=512, dx=20e-6,
                             method='L-BFGS-B', verbose=True)

    print('Optimized prescription:', result.prescription)
    print('Merit:', result.merit, '  Strehl:', result.strehl_best)
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .lenses import apply_real_lens, apply_real_lens_traced
from .raytrace import (
    surfaces_from_prescription, system_abcd, trace,
    seidel_coefficients,
)
from .analysis import wave_opd_2d, zernike_decompose
from .through_focus import (
    through_focus_scan, find_best_focus, diffraction_limited_peak,
)


# =========================================================================
# Parameterization
# =========================================================================

@dataclass
class DesignParameterization:
    """Map a flat parameter vector to a lens prescription dict.

    Attributes
    ----------
    template : dict
        Base prescription.  Deep-copied on each ``build()`` call;
        the copy has the free variables replaced by the current
        parameter values.
    free_vars : list of tuple
        Each entry is a "path" into the prescription dict.  Supported
        forms:
          - ``('surfaces', i, key)`` -- surface-dict field like
            ``radius``, ``conic``, ``radius_y`` etc.
          - ``('thicknesses', i)`` -- ``thicknesses[i]``
          - ``('aperture_diameter',)`` -- top-level field
    bounds : list of tuple or None
        (lower, upper) for each parameter.  Used by bounded
        scipy solvers.  Pass ``None`` to disable bounds for a given
        parameter.
    """

    template: Dict[str, Any]
    free_vars: List[Tuple[Any, ...]]
    bounds: Optional[List[Optional[Tuple[float, float]]]] = None

    def __post_init__(self):
        if self.bounds is not None:
            if len(self.bounds) != len(self.free_vars):
                raise ValueError(
                    f"bounds length {len(self.bounds)} != free_vars "
                    f"length {len(self.free_vars)}")

    @property
    def n_params(self) -> int:
        return len(self.free_vars)

    def initial_values(self) -> np.ndarray:
        """Read the free-var values from the template as the starting x0."""
        x0 = np.empty(self.n_params, dtype=np.float64)
        for i, path in enumerate(self.free_vars):
            x0[i] = _read_path(self.template, path)
        return x0

    def build(self, x: np.ndarray) -> Dict[str, Any]:
        """Return a deep copy of the template with free vars set to x."""
        pres = copy.deepcopy(self.template)
        for i, path in enumerate(self.free_vars):
            _write_path(pres, path, float(x[i]))
        return pres


def _read_path(pres, path):
    """Read a value from a prescription dict along a tuple path."""
    cur = pres
    for p in path:
        if isinstance(cur, dict):
            cur = cur[p]
        else:  # list or tuple
            cur = cur[p]
    return float(cur)


def _write_path(pres, path, value):
    """Write a value into a prescription dict along a tuple path."""
    cur = pres
    for p in path[:-1]:
        if isinstance(cur, dict):
            cur = cur[p]
        else:
            cur = cur[p]
    last = path[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        cur[last] = value


@dataclass
class MultiPrescriptionParameterization:
    """Optimize multiple lens prescriptions jointly.

    Like :class:`DesignParameterization` but holds a LIST of template
    prescriptions and a free-var list whose entries are
    ``(prescription_index, *inner_path)`` tuples pointing into a
    specific template.  ``build()`` returns a list of concrete
    prescriptions of the same length; the context stores them as
    ``ctx.prescriptions``.

    Use this when an architecture has more than one lens whose
    parameters vary independently -- e.g. a 4f imaging system's two
    achromats, a Keplerian telescope's objective + eyepiece, or a
    zoom stage.

    Example
    -------
    >>> import lumenairy as op
    >>> obj = op.thorlabs_lens('AC254-200-C')
    >>> eye = op.thorlabs_lens('AC254-050-C')
    >>> param = op.MultiPrescriptionParameterization(
    ...     templates=[obj, eye],
    ...     free_vars=[
    ...         (0, 'surfaces', 0, 'radius'),  # obj R1
    ...         (0, 'surfaces', 2, 'radius'),  # obj R3
    ...         (1, 'surfaces', 0, 'radius'),  # eye R1
    ...         (1, 'thicknesses', 1),          # eye inner thickness
    ...     ],
    ...     bounds=[(100e-3, 300e-3), (-300e-3, -50e-3),
    ...             (20e-3, 100e-3), (1e-3, 5e-3)])
    >>> # The merit must know which prescription slot maps to which lens:
    >>> merit = op.MatchIdealSystemMerit(
    ...     ideal_elements=[
    ...         {'type': 'lens', 'f': 200e-3},
    ...         {'type': 'propagate', 'z': 250e-3},
    ...         {'type': 'lens', 'f': 50e-3},
    ...         {'type': 'propagate', 'z': 50e-3},
    ...     ],
    ...     real_elements=[
    ...         {'type': '_prescription_', 'index': 0},
    ...         {'type': 'propagate', 'z': 250e-3},
    ...         {'type': '_prescription_', 'index': 1},
    ...         {'type': 'propagate', 'z': 50e-3},
    ...     ],
    ...     match='field_overlap')

    Attributes
    ----------
    templates : list of dict
        Base prescriptions, one per lens.
    free_vars : list of tuple
        Each tuple starts with an int (the prescription index in
        ``templates``) followed by the same path format accepted by
        :class:`DesignParameterization`.
    bounds : list of (lo, hi) or None
    """

    templates: List[Dict[str, Any]]
    free_vars: List[Tuple[Any, ...]]
    bounds: Optional[List[Optional[Tuple[float, float]]]] = None

    def __post_init__(self):
        for fv in self.free_vars:
            if not fv or not isinstance(fv[0], (int, np.integer)):
                raise ValueError(
                    f"MultiPrescriptionParameterization free_vars entries "
                    f"must start with an int prescription index; got {fv!r}")
            if not (0 <= int(fv[0]) < len(self.templates)):
                raise ValueError(
                    f"free_var {fv!r} refers to template index "
                    f"{fv[0]}, but only {len(self.templates)} templates "
                    f"were provided")
        if self.bounds is not None:
            if len(self.bounds) != len(self.free_vars):
                raise ValueError(
                    f"bounds length {len(self.bounds)} != free_vars "
                    f"length {len(self.free_vars)}")

    @property
    def n_params(self) -> int:
        return len(self.free_vars)

    @property
    def n_prescriptions(self) -> int:
        return len(self.templates)

    def initial_values(self) -> np.ndarray:
        x0 = np.empty(self.n_params, dtype=np.float64)
        for i, fv in enumerate(self.free_vars):
            pres_idx = int(fv[0])
            inner_path = fv[1:]
            x0[i] = _read_path(self.templates[pres_idx], inner_path)
        return x0

    def build(self, x: np.ndarray) -> List[Dict[str, Any]]:
        """Return a list of deep-copied prescriptions with free vars
        set to ``x``."""
        prescriptions = [copy.deepcopy(t) for t in self.templates]
        for i, fv in enumerate(self.free_vars):
            pres_idx = int(fv[0])
            inner_path = fv[1:]
            _write_path(prescriptions[pres_idx], inner_path, float(x[i]))
        return prescriptions


# =========================================================================
# Merit terms
# =========================================================================

# Sentinel used by EvaluationContext when ABCD extraction failed.  Merit
# terms that consume ``ctx.efl`` / ``ctx.bfl`` should route through
# :func:`ctx_is_valid` rather than blindly plugging the sentinel into
# their formulas (a naive ``(1e9 - target)^2`` is astronomical and drags
# the optimizer away from good regions).
_INVALID_FL_SENTINEL = 1e9


def ctx_is_valid(ctx, field) -> bool:
    """Return True if ``ctx.<field>`` holds a usable physical value.

    Guards against the sentinels set when the ray-leg failed (``1e9``
    for focal lengths) and against NaN/Inf from downstream computations.
    """
    try:
        v = getattr(ctx, field)
    except AttributeError:
        return False
    import numpy as _np
    if v is None:
        return False
    if not _np.isfinite(v):
        return False
    if abs(v) >= _INVALID_FL_SENTINEL * 0.5:
        return False
    return True


class MeritTerm:
    """Base class for a single term in the merit function.

    Each merit term takes the full ``EvaluationContext`` (ray-trace
    results, wave field, etc.) and returns a scalar contribution
    (already weighted).  Concrete subclasses override
    :meth:`evaluate`.

    Attributes
    ----------
    weight : float
        Multiplier applied to the raw term value.  Squared residuals
        in a least-squares sense, or additive penalty in a general
        minimize sense.
    needs_wave : bool, default False
        If True, the optimizer will run the wave-optics pipeline
        (``apply_real_lens_traced`` + through-focus) for each
        evaluation.  Set False for pure-geometric terms -- the
        optimizer will skip the expensive wave leg if NO merit
        terms need it.
    """

    weight: float = 1.0
    needs_wave: bool = False
    name: str = 'MeritTerm'

    def evaluate(self, ctx) -> float:
        raise NotImplementedError


class FocalLengthMerit(MeritTerm):
    """Penalise deviation from target focal length.

    ``contribution = weight * (efl - target)^2 / target^2``

    When ``target == 0`` (afocal / collimator), the normalised-error
    formula is ill-defined; this class falls back to a penalty on
    ``efl^2`` scaled to unit metres so that pushing EFL toward
    infinity (i.e. toward truly afocal) drives the merit to zero.
    """

    needs_wave = False
    name = 'FocalLength'

    def __init__(self, target: float, weight: float = 1.0):
        self.target = float(target)
        self.weight = float(weight)

    def evaluate(self, ctx) -> float:
        efl = getattr(ctx, 'efl', float('nan'))
        if not ctx_is_valid(ctx, 'efl'):
            return self.weight  # graceful large-but-finite penalty
        if self.target == 0.0:
            # Collimator / afocal target: minimise |EFL| directly.  A
            # metre-scaled unit keeps the contribution comparable to
            # other unit-scaled merits.
            return self.weight * efl * efl
        err = (efl - self.target) / self.target
        return self.weight * err * err


class BackFocalLengthMerit(MeritTerm):
    """Penalise deviation from target back focal length.

    Same zero-target behaviour as :class:`FocalLengthMerit`.
    """

    needs_wave = False
    name = 'BackFocalLength'

    def __init__(self, target: float, weight: float = 1.0):
        self.target = float(target)
        self.weight = float(weight)

    def evaluate(self, ctx) -> float:
        bfl = getattr(ctx, 'bfl', float('nan'))
        if not ctx_is_valid(ctx, 'bfl'):
            return self.weight
        if self.target == 0.0:
            return self.weight * bfl * bfl
        err = (bfl - self.target) / self.target
        return self.weight * err * err


class SphericalSeidelMerit(MeritTerm):
    """Minimise Seidel spherical aberration coefficient S_I.

    Fast, geometric-only term.
    """

    needs_wave = False
    name = 'SphericalSeidel'

    def __init__(self, weight: float = 1.0):
        self.weight = float(weight)

    def evaluate(self, ctx) -> float:
        return self.weight * ctx.seidel[0] ** 2


class StrehlMerit(MeritTerm):
    """Penalise Strehl ratio below ``min_strehl``.

    ``contribution = weight * max(0, min_strehl - best_strehl)^2``
    """

    needs_wave = True
    name = 'Strehl'

    def __init__(self, min_strehl: float = 0.8, weight: float = 1.0):
        self.min_strehl = float(min_strehl)
        self.weight = float(weight)

    def evaluate(self, ctx) -> float:
        deficit = max(0.0, self.min_strehl - ctx.strehl_best)
        return self.weight * deficit * deficit


class RMSWavefrontMerit(MeritTerm):
    """Penalise RMS wavefront error above a target (waves).

    Uses Zernike decomposition to exclude the first
    ``exclude_low_order`` modes (default 4: piston + 2 tilts +
    defocus), matching the optics-design convention of reporting
    'image-quality' RMS after best-focus.  Set ``exclude_low_order=3``
    to keep defocus in the RMS (penalises focus shift as well as
    high-order aberrations).
    """

    needs_wave = True
    name = 'RMSWavefront'

    def __init__(self, max_rms_waves: float = 0.07,
                 n_modes: int = 21,
                 exclude_low_order: int = 4,
                 weight: float = 1.0):
        self.max_rms_waves = float(max_rms_waves)
        self.n_modes = int(n_modes)
        self.exclude_low_order = int(exclude_low_order)
        self.weight = float(weight)

    def evaluate(self, ctx) -> float:
        rms_waves = ctx.rms_wavefront_waves(
            n_modes=self.n_modes,
            exclude_low_order=self.exclude_low_order)
        excess = max(0.0, rms_waves - self.max_rms_waves)
        return self.weight * excess * excess


class SpotSizeMerit(MeritTerm):
    """Penalise RMS spot radius at best focus above a target."""

    needs_wave = True
    name = 'SpotSize'

    def __init__(self, max_rms_radius: float, weight: float = 1.0):
        self.max_rms_radius = float(max_rms_radius)
        self.weight = float(weight)

    def evaluate(self, ctx) -> float:
        r = ctx.rms_radius_best
        excess = max(0.0, r - self.max_rms_radius)
        return self.weight * excess * excess


class MatchIdealThinLensMerit(MeritTerm):
    """Penalise deviation of the actual exit-pupil OPD from an
    idealised thin-lens wavefront with the same target focal length.

    This is the merit term you want when you're asking "make this real
    lens behave as much like an ideal thin lens of focal length f as
    possible".  At each evaluation:

    1. The actual exit-pupil OPD is extracted from the wave-optics
       output (using `wave_opd_2d`, with reference-sphere subtraction
       for numerical stability).
    2. An ideal thin-lens OPD ``OPD_ideal(r) = -r^2 / (2*f_target)`` is
       computed on the same grid.
    3. Their difference -- the *aberration wavefront* -- is masked to
       the pupil and its RMS computed.
    4. ``contribution = weight * RMS_diff^2 / wavelength^2`` (in
       wavelength-squared units, so weights of 1.0 produce
       ``waves^2`` of penalty).

    The result of optimizing against this merit is a wavefront whose
    departure from a perfect spherical converging wave is minimised
    -- which is the formal definition of "diffraction-limited" up to
    a tolerable RMS wavefront error.

    Parameters
    ----------
    target_focal_length : float
        Focal length [m] of the ideal thin lens to match.
    weight : float, default 1.0
    exclude_low_order : int, default 1
        Number of Zernike modes to exclude from the RMS (default 1 =
        piston only).  Set to 3 to also exclude tilts (so a lateral
        decenter doesn't dominate the merit), or 4 to also exclude
        defocus (so the merit becomes "match the ideal except for an
        arbitrary focus shift").
    n_modes : int, default 21
        How many Zernike modes the OPD is decomposed into to compute
        the high-order RMS.
    """

    needs_wave = True
    name = 'MatchIdealThinLens'

    def __init__(self, target_focal_length, weight=1.0,
                 exclude_low_order=1, n_modes=21):
        self.target_focal_length = float(target_focal_length)
        self.weight = float(weight)
        self.exclude_low_order = int(exclude_low_order)
        self.n_modes = int(n_modes)

    def evaluate(self, ctx) -> float:
        if ctx.opd_map is None:
            return 0.0
        ap = ctx.prescription.get('aperture_diameter')
        if ap is None:
            return 0.0
        # Compute the ideal thin-lens OPD on the pupil grid
        Ny, Nx = ctx.opd_map.shape
        x = (np.arange(Nx) - Nx / 2) * ctx.dx
        y = (np.arange(Ny) - Ny / 2) * ctx.dx
        X, Y = np.meshgrid(x, y)
        opd_ideal = -(X ** 2 + Y ** 2) / (2.0 * self.target_focal_length)
        diff = ctx.opd_map - opd_ideal
        # Decompose into Zernikes; high-order RMS = aberration RMS
        # (excluding the requested low-order modes which represent
        # piston / tilt / defocus -- usually NOT what you want to
        # penalise unless you've fixed alignment).
        from .analysis import zernike_decompose
        finite = np.isfinite(diff)
        # Replace NaN with 0 in the masked region; decompose handles it
        diff_clean = np.where(finite, diff, 0.0)
        try:
            coeffs, _ = zernike_decompose(
                diff_clean, ctx.dx, ap, n_modes=self.n_modes)
        except Exception:
            return 0.0
        higher = coeffs[self.exclude_low_order:]
        rms_m = float(np.sqrt(np.sum(higher ** 2)))
        rms_waves = rms_m / ctx.wavelength
        return self.weight * rms_waves * rms_waves


# =========================================================================
# Full-system "match this ideal thin-lens architecture" merit
# =========================================================================

class MatchIdealSystemMerit(MeritTerm):
    """Match the real system's output field to that of an idealised
    thin-lens reference system.

    Unlike :class:`MatchIdealThinLensMerit` -- which operates on the
    exit-pupil OPD of a single lens and compares it to a bare
    converging sphere -- this merit propagates a reference source
    through BOTH an ideal thin-lens element list AND the real
    prescription (wrapped in an optional pre/post-propagation envelope)
    and then compares the resulting **complex output fields**.

    Use cases:

    * Replace a paraxial thin lens with a singlet / doublet / aspheric
      while preserving the output radiation pattern + relative phase.
    * Replace a Keplerian / Galilean telescope's thin-lens pair with
      real achromats.
    * Replace a 4f imaging system's two thin lenses with two real
      lenses, jointly optimised.
    * Any architecture expressible as
      ``propagate_through_system(source, ideal_elements)``.

    The merit is designed for the common situation where the real
    system is **slightly longer** than the ideal (because real lenses
    have nonzero thickness) but otherwise functionally equivalent.
    Approximately preserving aperture and inter-element distances is
    the user's responsibility; the merit will drive whatever free
    variables it can to make the output fields match.

    Parameters
    ----------
    ideal_elements : list of dict
        Element list for the ideal reference system, in the same
        format as :func:`propagate_through_system`.  Typically a mix
        of ``{'type': 'lens', 'f': ...}`` and
        ``{'type': 'propagate', 'z': ...}`` elements, optionally with
        apertures / mirrors / masks.
    real_elements : list of dict, optional
        Element list for the real system.  Dicts with
        ``type='_prescription_'`` are replaced at evaluation time
        with the current ``ctx.prescription`` wrapped as a
        ``'real_lens'`` element (or ``'real_lens_traced'`` if
        ``use_traced_lens=True``).  Default: a single-lens drop-in,
        ``[{'type': '_prescription_'}]``, which is correct when the
        ideal is a single thin lens + propagate pair and the real
        prescription replaces that thin lens.
    source_fn : callable or None
        Factory returning the complex input field at the first plane.
        Signature ``source_fn(N, dx, wavelength) -> ndarray``.  The
        merit uses ``ctx.N``, ``ctx.dx``, ``ctx.wavelength``.  Default
        (``None``) is a uniform plane wave of unit amplitude inside
        the prescription's aperture (if specified) or the full grid
        (otherwise).
    match : ``'field_overlap'`` | ``'field_mse'`` | ``'intensity_mse'`` | ``'intensity_overlap'``
        Similarity metric:

        * ``'field_overlap'`` (**default**, recommended): coupling
          efficiency
          ``|<E_ideal | E_real>|^2 / (||E_ideal||^2 ||E_real||^2)``.
          Bounded in [0, 1], invariant to a global phase and to an
          overall amplitude scaling.  Merit = ``weight * (1 - overlap)``;
          drops to zero when the real field differs from the ideal only
          by a global phase factor and overall amplitude.  **This is
          the right choice for "match the radiation pattern + relative
          phase"** as requested -- global phase is not a physical
          observable.
        * ``'field_mse'``: power-normalised + phase-aligned MSE of the
          field difference.  Merit is roughly the squared "fraction"
          of energy in the difference.  More sensitive to absolute
          phase shape than ``field_overlap``.
        * ``'intensity_mse'``: MSE of ``|E|^2``, phase-blind.  Use
          this when only the radiation pattern (not phase) matters
          (e.g. matching a target irradiance profile).
        * ``'intensity_overlap'``: correlation of ``|E|^2`` patterns,
          phase-blind.
    aperture_mask : ndarray or None
        Optional boolean / real mask applied to BOTH output fields
        before the comparison.  Use it to restrict the match to a
        region of interest (e.g. the intended image area) and avoid
        letting low-intensity grid edges dominate.
    use_traced_lens : bool, default False
        If True, propagate the real prescription via
        ``apply_real_lens_traced`` (sub-nm OPD agreement with the
        ray trace, 10-30x slower) rather than ``apply_real_lens``.
    ray_subsample : int, default 4
        Passed to ``apply_real_lens_traced`` when used.
    focus_search : bool, default False
        If True, scan a small range of axial offsets on the real
        system's output plane and report the BEST (lowest-penalty)
        match.  Decouples "correct focal plane" from "aberration
        quality" so a small BFL shift caused by real-lens thickness
        doesn't dominate the penalty.  Not valid for
        ``match='intensity_mse'`` (no unique optimum under
        translation); enable it with any of the other three metrics.
    focus_search_range : tuple (z_lo, z_hi) or None
        Axial-offset bracket for the focus search, relative to the
        nominal output plane [m].  Default (None): +/- f/20 computed
        from ``ctx.bfl`` or ``ctx.efl``, falling back to +/- 5 mm.
    focus_search_n : int, default 9
        Number of samples in the z-offset scan.
    wavelengths : list of float, optional
        If given, evaluate the merit at each wavelength and average
        the results.  Drives the glass-index dispersion through
        ``apply_real_lens`` + ``propagate_through_system``.  Useful
        for broadband / chromatic-matching optimisation without
        needing a separate ``MultiWavelengthMerit`` wrapper.
    field_angles : list of (theta_x, theta_y) tuples, optional
        Off-axis tilts (radians) to evaluate.  Each field angle adds
        a carrier phase to the source so the merit penalises the
        real lens's output for multiple input beam directions
        simultaneously.  Combines Cartesian-product-wise with
        ``wavelengths``.
    weight : float

    Notes
    -----
    * The ideal system is propagated with ``propagate_through_system``;
      each thin lens applies the paraxial phase screen
      ``exp(-i k r^2 / 2f)``.  This is exact in the small-angle limit
      and close to correct for f/2 or slower systems.  For systems
      with non-paraxial focusing, consider using
      ``{'type': 'lens', 'f': ..., 'lens_model': 'nonparaxial'}`` in
      ``ideal_elements``.
    * ``field_overlap`` is the physical "coupling efficiency" metric
      used in fiber / optical-mode matching.  It's bounded and
      dimensionless, which makes it numerically well-behaved for the
      optimizer and meaningful as an absolute number (0.99 = nearly
      perfect; 0.50 = significant mismatch).
    * Multi-lens architectures where two or more prescriptions are
      independently varied require multiple
      :class:`DesignParameterization` -- not yet supported by the
      single-template design.  Open a PR if you hit this.
    """

    needs_wave = True
    name = 'MatchIdealSystem'

    def __init__(self, ideal_elements,
                 real_elements=None,
                 source_fn=None,
                 match='field_overlap',
                 aperture_mask=None,
                 use_traced_lens=False,
                 ray_subsample=4,
                 focus_search=False,
                 focus_search_range=None,
                 focus_search_n=9,
                 wavelengths=None,
                 field_angles=None,
                 weight=1.0):
        self.ideal_elements = list(ideal_elements)
        self.real_elements = (list(real_elements)
                              if real_elements is not None
                              else [{'type': '_prescription_'}])
        self.source_fn = source_fn
        self.match = str(match)
        self.aperture_mask = aperture_mask
        self.use_traced_lens = bool(use_traced_lens)
        self.ray_subsample = int(ray_subsample)
        self.weight = float(weight)
        self.focus_search = bool(focus_search)
        self.focus_search_range = focus_search_range
        self.focus_search_n = int(focus_search_n)
        # ``wavelengths`` and ``field_angles`` drive built-in sweeps
        # (averaged penalty).  Both default to None = single
        # wavelength / on-axis.
        self.wavelengths = (list(wavelengths)
                            if wavelengths is not None else None)
        self.field_angles = (list(field_angles)
                             if field_angles is not None else None)
        valid = ('field_overlap', 'field_mse',
                 'intensity_mse', 'intensity_overlap')
        if self.match not in valid:
            raise ValueError(
                f"match must be one of {valid}; got {self.match!r}")
        if self.focus_search and self.match not in (
                'field_overlap', 'field_mse', 'intensity_overlap'):
            raise ValueError(
                f"focus_search requires match in "
                f"('field_overlap', 'field_mse', 'intensity_overlap'); "
                f"got {self.match!r}.  intensity_mse doesn't have a "
                f"unique optimum under axial translation.")

    # -- Helpers -----------------------------------------------------

    def _make_source(self, ctx, wavelength, field_angle=(0.0, 0.0)):
        """Build the reference input field at the first plane.

        Parameters
        ----------
        ctx : EvaluationContext
            Provides ``N``, ``dx``, ``prescription`` (for default
            aperture clipping).
        wavelength : float
            Wavelength [m] used both by ``source_fn`` (if supplied)
            and for the field-angle carrier phase.
        field_angle : (float, float)
            Off-axis tilt ``(theta_x, theta_y)`` in radians.  A linear
            phase ``exp(i * k_x X + i * k_y Y)`` is applied on top of
            whatever source the factory produced, with
            ``k_x = 2 pi sin(theta_x) / wavelength``.
        """
        if self.source_fn is not None:
            E = self.source_fn(ctx.N, ctx.dx, wavelength)
            E = np.asarray(E, dtype=np.complex128)
        else:
            E = np.ones((ctx.N, ctx.N), dtype=np.complex128)
            ap = (ctx.prescription.get('aperture_diameter')
                  if ctx.prescription else None)
            if ap is not None and np.isfinite(ap) and ap > 0:
                x = (np.arange(ctx.N) - ctx.N / 2) * ctx.dx
                X, Y = np.meshgrid(x, x)
                mask = (X * X + Y * Y) <= (ap / 2.0) ** 2
                E = np.where(mask, E, 0.0 + 0.0j)

        # Field-angle tilt: apply a linear phase ramp for off-axis
        # illumination.  Identity for on-axis (0, 0).
        tx, ty = field_angle
        if tx or ty:
            x = (np.arange(ctx.N) - ctx.N / 2) * ctx.dx
            X, Y = np.meshgrid(x, x)
            k0 = 2.0 * np.pi / wavelength
            E = E * np.exp(1j * (k0 * np.sin(tx) * X
                                   + k0 * np.sin(ty) * Y))
        return E

    def _build_real_elements(self, ctx):
        """Expand ``_prescription_`` sentinels into real-lens elements.

        The sentinel supports an optional ``'index'`` key that selects
        from ``ctx.prescriptions`` (populated when using
        :class:`MultiPrescriptionParameterization`).  With no index
        supplied, falls back to ``ctx.prescription`` -- the
        single-prescription (backward-compatible) case.
        """
        lens_type = ('real_lens_traced' if self.use_traced_lens
                     else 'real_lens')
        extras = {}
        if self.use_traced_lens:
            extras['ray_subsample'] = self.ray_subsample

        prescriptions = ctx.prescriptions
        if prescriptions is None:
            prescriptions = [ctx.prescription]

        expanded = []
        for elem in self.real_elements:
            if elem.get('type') == '_prescription_':
                idx = int(elem.get('index', 0))
                if not (0 <= idx < len(prescriptions)):
                    raise IndexError(
                        f"'_prescription_' placeholder index {idx} out "
                        f"of range (ctx has {len(prescriptions)} "
                        f"prescriptions)")
                expanded.append({
                    'type': lens_type,
                    'prescription': prescriptions[idx],
                    **extras,
                    # Preserve any user-specified bandlimit / per-
                    # element overrides passed through the sentinel,
                    # except for the meta keys we've already consumed.
                    **{k: v for k, v in elem.items()
                       if k not in ('type', 'index')},
                })
            else:
                expanded.append(dict(elem))
        return expanded

    def _propagate(self, elements, E_in, ctx, wavelength):
        from .system import propagate_through_system
        E_out, _ = propagate_through_system(
            E_in, elements, wavelength, ctx.dx)
        return E_out

    # -- Main evaluate ----------------------------------------------

    def evaluate(self, ctx) -> float:
        if ctx.prescription is None:
            return self.weight

        wavelengths = self.wavelengths or [ctx.wavelength]
        field_angles = self.field_angles or [(0.0, 0.0)]

        penalties = []
        for wl in wavelengths:
            for fa in field_angles:
                try:
                    p = self._evaluate_one(ctx, wavelength=float(wl),
                                            field_angle=tuple(fa))
                except Exception:
                    p = self.weight
                penalties.append(p)
        # Arithmetic mean across all (wavelength, field) combinations.
        return float(np.mean(penalties))

    def _evaluate_one(self, ctx, wavelength, field_angle):
        """Compute the merit for a single wavelength + field-angle pair."""
        E_in = self._make_source(ctx, wavelength, field_angle)
        E_ideal = self._propagate(self.ideal_elements, E_in, ctx, wavelength)
        real_elems = self._build_real_elements(ctx)
        E_real = self._propagate(real_elems, E_in, ctx, wavelength)

        if E_ideal.shape != E_real.shape:
            return self.weight

        mask = self.aperture_mask
        if mask is not None:
            E_ideal = E_ideal * mask
            E_real = E_real * mask

        # Optional axial focus search: find the z-offset where the
        # real field best matches the ideal's radiation pattern.  This
        # decouples "correct focal plane" from "aberration quality" so
        # a small BFL shift introduced by lens thickness doesn't
        # dominate the penalty.
        if self.focus_search:
            return self._focus_search_penalty(
                E_ideal, E_real, ctx, wavelength)

        return self._compute_penalty(E_ideal, E_real)

    def _focus_search_penalty(self, E_ideal, E_real, ctx, wavelength):
        """Propagate E_real through a small range of z offsets, pick
        the one that minimises the penalty (i.e. maximises overlap),
        and return that value.  Uses ASM (fast, exact, preserves dx).
        """
        from .propagation import angular_spectrum_propagate
        # Default range: +-f/20 where f ~= ctx.efl or ctx.bfl; fall
        # back to +-5 mm if neither is available.
        if self.focus_search_range is not None:
            z_lo, z_hi = self.focus_search_range
        else:
            ref = ctx.bfl if (ctx.bfl and np.isfinite(ctx.bfl)
                                and abs(ctx.bfl) < 10) else ctx.efl
            if ref and np.isfinite(ref) and abs(ref) < 10:
                half = max(abs(ref) / 20.0, 1e-4)
            else:
                half = 5e-3
            z_lo, z_hi = -half, +half
        zs = np.linspace(z_lo, z_hi, max(3, self.focus_search_n))
        best = self.weight  # worst-case sentinel
        for dz in zs:
            E_shifted = (E_real if dz == 0.0
                         else angular_spectrum_propagate(
                             E_real, float(dz), wavelength, ctx.dx,
                             bandlimit=True))
            p = self._compute_penalty(E_ideal, E_shifted)
            if p < best:
                best = p
        return best

    def _compute_penalty(self, E_ideal, E_real):
        if self.match == 'field_overlap':
            return self._field_overlap_penalty(E_ideal, E_real)
        if self.match == 'field_mse':
            return self._field_mse_penalty(E_ideal, E_real)
        if self.match == 'intensity_mse':
            return self._intensity_mse_penalty(E_ideal, E_real)
        # intensity_overlap
        return self._intensity_overlap_penalty(E_ideal, E_real)

    # -- Metric kernels ---------------------------------------------

    @staticmethod
    def _field_overlap_penalty_raw(E_ideal, E_real):
        """Return (1 - coupling_efficiency).  Returns 1.0 if either
        field is zero (worst case)."""
        num = abs(np.vdot(E_ideal.ravel(), E_real.ravel())) ** 2
        p_i = float(np.sum(np.abs(E_ideal) ** 2))
        p_r = float(np.sum(np.abs(E_real) ** 2))
        den = p_i * p_r
        if den < 1e-60:
            return 1.0
        overlap = float(num / den)
        return 1.0 - overlap

    def _field_overlap_penalty(self, E_ideal, E_real):
        return self.weight * self._field_overlap_penalty_raw(E_ideal, E_real)

    def _field_mse_penalty(self, E_ideal, E_real):
        """Power-normalised, global-phase-aligned, squared L2 of the
        field residual.  Roughly the "fraction of energy in the
        difference" when amplitude-normalised."""
        p_i = float(np.sum(np.abs(E_ideal) ** 2))
        p_r = float(np.sum(np.abs(E_real) ** 2))
        if p_i < 1e-30 or p_r < 1e-30:
            return self.weight
        scale = np.sqrt(p_i / p_r)
        inner = np.vdot(E_ideal.ravel(), E_real.ravel())
        phase_align = (np.conj(inner) / abs(inner)) if abs(inner) > 1e-30 else 1.0 + 0.0j
        E_real_aligned = E_real * scale * phase_align
        mse = float(np.sum(np.abs(E_ideal - E_real_aligned) ** 2) / p_i)
        return self.weight * mse

    def _intensity_mse_penalty(self, E_ideal, E_real):
        """Phase-blind: compares |E|^2 patterns, normalised to equal
        total power."""
        I_i = np.abs(E_ideal) ** 2
        I_r = np.abs(E_real) ** 2
        p_i = float(np.sum(I_i))
        p_r = float(np.sum(I_r))
        if p_i < 1e-30 or p_r < 1e-30:
            return self.weight
        I_r_norm = I_r * (p_i / p_r)
        return self.weight * float(np.sum((I_i - I_r_norm) ** 2) / (p_i ** 2))

    def _intensity_overlap_penalty(self, E_ideal, E_real):
        I_i = np.abs(E_ideal) ** 2
        I_r = np.abs(E_real) ** 2
        num = float(np.sum(I_i * I_r))
        den = np.sqrt(float(np.sum(I_i ** 2)) * float(np.sum(I_r ** 2)))
        if den < 1e-30:
            return self.weight
        return self.weight * (1.0 - num / den)

    # -- Convenience -------------------------------------------------

    @classmethod
    def single_lens(cls, focal_length, post_distance=None, **kwargs):
        """Shortcut for the single-lens drop-in replacement case.

        Generates ``ideal_elements`` = [thin_lens(f), propagate(z=f
        or post_distance)].  Equivalent real_elements is the default
        single-``_prescription_`` drop-in.

        Parameters
        ----------
        focal_length : float
            Ideal thin-lens focal length [m].
        post_distance : float, optional
            Propagation distance from the lens to the output plane
            [m].  Default: ``focal_length`` (i.e. evaluate at the
            paraxial focus).
        kwargs : forwarded to :meth:`__init__`.
        """
        post = float(focal_length) if post_distance is None else float(post_distance)
        ideal = [
            {'type': 'lens', 'f': float(focal_length)},
            {'type': 'propagate', 'z': post},
        ]
        real_elems = kwargs.pop('real_elements', None)
        if real_elems is None:
            real_elems = [
                {'type': '_prescription_'},
                {'type': 'propagate', 'z': post},
            ]
        return cls(ideal_elements=ideal,
                   real_elements=real_elems,
                   **kwargs)


class MatchTargetOPDMerit(MeritTerm):
    """Penalise deviation of the actual exit-pupil OPD from a
    user-supplied target OPD map (or callable returning one).

    Use this when you have a desired wavefront -- not necessarily a
    perfect sphere -- that the lens should produce at its exit pupil.
    Examples: matching a measured wavefront, copying an existing
    well-corrected design, or shaping a beam with a target phase
    profile.

    Parameters
    ----------
    target_opd : ndarray or callable
        - If ndarray (shape (Ny, Nx) matching the simulation grid):
          used directly.
        - If callable: called as ``target_opd(X, Y, prescription)``
          and expected to return an ndarray of OPD [m] over the
          pupil grid.  Useful when the target depends on the current
          prescription (e.g. "match a target with the same EFL").
    weight : float, default 1.0
    exclude_low_order : int, default 1
        Number of Zernike modes to remove from the residual before
        computing RMS.
    n_modes : int, default 21
    """

    needs_wave = True
    name = 'MatchTargetOPD'

    def __init__(self, target_opd, weight=1.0,
                 exclude_low_order=1, n_modes=21):
        self.target_opd = target_opd
        self.weight = float(weight)
        self.exclude_low_order = int(exclude_low_order)
        self.n_modes = int(n_modes)

    def evaluate(self, ctx) -> float:
        if ctx.opd_map is None:
            return 0.0
        ap = ctx.prescription.get('aperture_diameter')
        if ap is None:
            return 0.0
        Ny, Nx = ctx.opd_map.shape
        x = (np.arange(Nx) - Nx / 2) * ctx.dx
        y = (np.arange(Ny) - Ny / 2) * ctx.dx
        X, Y = np.meshgrid(x, y)
        if callable(self.target_opd):
            target = np.asarray(self.target_opd(X, Y, ctx.prescription))
        else:
            target = np.asarray(self.target_opd)
        if target.shape != ctx.opd_map.shape:
            raise ValueError(
                f'target_opd shape {target.shape} does not match '
                f'opd_map shape {ctx.opd_map.shape}')
        diff = ctx.opd_map - target
        from .analysis import zernike_decompose
        finite = np.isfinite(diff)
        diff_clean = np.where(finite, diff, 0.0)
        try:
            coeffs, _ = zernike_decompose(
                diff_clean, ctx.dx, ap, n_modes=self.n_modes)
        except Exception:
            return 0.0
        higher = coeffs[self.exclude_low_order:]
        rms_m = float(np.sqrt(np.sum(higher ** 2)))
        rms_waves = rms_m / ctx.wavelength
        return self.weight * rms_waves * rms_waves


class ZernikeCoefficientMerit(MeritTerm):
    """Penalise (or target) specific Zernike-mode coefficients of the
    actual exit-pupil OPD.

    Lets you express design intents like:
    - "Eliminate spherical aberration" -- target mode 12 (Z_4^0) = 0
    - "Allow some defocus but no tilt or coma" -- target tilts (1,2),
      vertical/horizontal coma (7,8), and trefoils (6,9) all = 0
    - "Match a measured aberration profile mode-by-mode"

    Parameters
    ----------
    targets : dict of {int: float}
        Map from OSA Zernike index to target coefficient [m].  Modes
        not in this dict are unconstrained.
    weight : float, default 1.0
    n_modes : int, default 21
        Number of modes to fit (must exceed max key in ``targets``).
    """

    needs_wave = True
    name = 'ZernikeCoefficient'

    def __init__(self, targets, weight=1.0, n_modes=21):
        self.targets = {int(j): float(v) for j, v in targets.items()}
        self.weight = float(weight)
        self.n_modes = max(int(n_modes), max(self.targets) + 1
                           if self.targets else int(n_modes))

    def evaluate(self, ctx) -> float:
        if ctx.opd_map is None:
            return 0.0
        ap = ctx.prescription.get('aperture_diameter')
        if ap is None:
            return 0.0
        from .analysis import zernike_decompose
        finite = np.isfinite(ctx.opd_map)
        opd_clean = np.where(finite, ctx.opd_map, 0.0)
        try:
            coeffs, _ = zernike_decompose(
                opd_clean, ctx.dx, ap, n_modes=self.n_modes)
        except Exception:
            return 0.0
        total = 0.0
        for j, target in self.targets.items():
            err_waves = (coeffs[j] - target) / ctx.wavelength
            total = total + err_waves * err_waves
        return self.weight * total


class LGAberrationMerit(MeritTerm):
    """Penalise specified Laguerre-Gaussian aberration-tensor channels
    via the closed-form modal asymptotic propagator (paper 2,
    Section 7).

    Each entry ``L_{(p, ell), n}(s_2^img)`` of the LG aberration tensor
    is the projection of the system's leading-order asymptotic
    image-plane field onto a named classical aberration channel:
    ``(0, 0)`` is piston/Strehl, ``(1, 0)`` is defocus, ``(2, 0)`` is
    primary spherical, ``(0, +-1)`` is tilt, ``(1, +-1)`` is coma,
    ``(0, +-2)`` is astigmatism, ``(0, +-3)`` is trefoil.  Driving a
    given ``|L_{(p, ell), 0}|^2`` to zero suppresses that aberration
    in the merit-function loop without invoking the wave leg.

    The merit is computed from a Chebyshev tensor-product fit of the
    prescription's canonical map ``Phi(s2, v2), s1(s2, v2)``
    (paper 1, Section 3) -- a single fit drives all targeted
    aberration channels at all chosen field points.

    Parameters
    ----------
    targets : dict
        Map from output LG index ``(p, ell)`` to a float weight.  Each
        listed channel contributes ``|L_{(p, ell), 0}(s_2^img)|^2``
        times the entry weight.  Channels not listed are unconstrained.
        Common targets:

            {(2, 0): 1.0, (1, 1): 1.0, (1, -1): 1.0, (0, 2): 1.0, (0, -2): 1.0}

        suppresses primary spherical, both coma orientations, and
        both astigmatism orientations.
    field_points : list of (float, float), optional
        Source-plane points [m] at which to evaluate the tensor.
        Default: a single on-axis point ``[(0.0, 0.0)]``.  Each field
        point's contribution is summed.
    image_points : list of (float, float), optional
        Image-plane evaluation points (one per field point).  If None,
        defaults to the chief-ray landing of each source point (which
        the merit evaluator finds via Newton).
    w_s, w_p : float
        Source-plane and pupil-plane Gaussian waists [m and direction
        cosine].  Defaults: ``w_s = 50e-6`` (50 um), ``w_p = 0.05``
        (50 mrad).
    w_o : float, optional
        Output Gaussian waist [m].  Default: derived per-pixel from
        the local complex beam matrix.
    fit_kwargs : dict, optional
        Additional keyword arguments passed to
        ``fit_canonical_polynomials``:  ``poly_order``,
        ``source_box_half``, ``pupil_box_half``, ``n_field``,
        ``n_pupil``, ``extract_linear_phase``, ``object_distance``.
    weight : float, default 1.0
    name : str, optional

    See Also
    --------
    lumenairy.asymptotic.aberration_tensor :  raw tensor evaluator.
    SphericalSeidelMerit :  Seidel-coefficient-based primary-spherical
        merit (uses paraxial coefficients; LGAberrationMerit's full
        non-paraxial generalisation is preferred for high-NA work).
    """

    needs_wave = False
    name = 'LGAberration'

    def __init__(self, targets,
                 field_points=None,
                 image_points=None,
                 w_s=50e-6, w_p=0.05, w_o=None,
                 fit_kwargs=None,
                 weight=1.0,
                 name=None):
        if not targets:
            raise ValueError("LGAberrationMerit: targets dict is empty")
        self.targets = {tuple(k): float(v) for k, v in targets.items()}
        if field_points is None:
            field_points = [(0.0, 0.0)]
        self.field_points = [tuple(p) for p in field_points]
        if image_points is None:
            self.image_points = None
        else:
            ips = list(image_points)
            if len(ips) != len(self.field_points):
                raise ValueError(
                    f"LGAberrationMerit: image_points length "
                    f"{len(ips)} must match field_points length "
                    f"{len(self.field_points)}")
            self.image_points = [tuple(p) for p in ips]
        self.w_s = float(w_s)
        self.w_p = float(w_p)
        self.w_o = None if w_o is None else float(w_o)
        self.fit_kwargs = dict(fit_kwargs) if fit_kwargs else {}
        self.weight = float(weight)
        if name is not None:
            self.name = str(name)

    def evaluate(self, ctx) -> float:
        # Lazy import to avoid bootstrap cycles.
        from .asymptotic import (fit_canonical_polynomials,
                                  aberration_tensor)
        try:
            fit = fit_canonical_polynomials(
                ctx.prescription,
                wavelength=ctx.wavelength,
                **self.fit_kwargs,
            )
        except Exception:
            # If the fit can't be built (e.g., aperture clipping kills
            # too many rays for the current prescription), assign a
            # large penalty so the optimiser steers away.
            return 1e20

        # The output modes are exactly the target keys, plus (0, 0)
        # for piston (always useful diagnostically).
        target_keys = list(self.targets.keys())
        output_modes = list(set([(0, 0)] + target_keys))

        total = 0.0
        for ifp, src in enumerate(self.field_points):
            if self.image_points is None:
                # Use the nominal chief-ray landing computed from the
                # paraxial back-map at v2_centre = 0.  fit's s2_centre/
                # halfrange box is centered on the actual landing
                # distribution, so s2_centre is a good chief estimate.
                s2_img = (fit.s2x_centre, fit.s2y_centre)
            else:
                s2_img = self.image_points[ifp]
            try:
                tensor = aberration_tensor(
                    fit,
                    s2_image=s2_img,
                    source_point=src,
                    source_modes=[(0, 0)],
                    pupil_modes=[(0, 0)],
                    output_modes=output_modes,
                    w_s=self.w_s, w_p=self.w_p, w_o=self.w_o,
                )
            except Exception:
                return 1e20
            # Index of each target in output_modes
            idx_map = {m: i for i, m in enumerate(output_modes)}
            for (p, ell), wgt in self.targets.items():
                try:
                    i = idx_map[(p, ell)]
                except KeyError:
                    continue
                val = complex(tensor.L[i, 0])
                total = total + wgt * (val.real * val.real
                                       + val.imag * val.imag)

        return self.weight * total


class CompositeMerit(MeritTerm):
    """Combine multiple sub-merits into one weighted sum.

    Useful for composing a complex objective from simpler pieces,
    or for grouping merits that share an expensive intermediate
    (e.g., the Zernike decomposition of the exit-pupil OPD).
    """

    name = 'Composite'

    def __init__(self, sub_merits, weight=1.0):
        self.sub_merits = list(sub_merits)
        self.weight = float(weight)
        self.needs_wave = any(m.needs_wave for m in self.sub_merits)

    def evaluate(self, ctx) -> float:
        s = 0.0
        for m in self.sub_merits:
            s = s + m.evaluate(ctx)
        return self.weight * s


class CallableMerit(MeritTerm):
    """Generic merit term that delegates to a user-supplied callable.

    Use for one-off custom objectives that don't fit the prebuilt
    classes:

        def my_merit(ctx):
            # ctx.efl, ctx.bfl, ctx.seidel, ctx.E_exit, ctx.opd_map,
            # ctx.strehl_best, ctx.rms_radius_best, ctx.prescription, ...
            return some_scalar

        merit = CallableMerit(my_merit, weight=1.0, needs_wave=True)
    """

    name = 'Callable'

    def __init__(self, fn, weight=1.0, needs_wave=False, name=None):
        self.fn = fn
        self.weight = float(weight)
        self.needs_wave = bool(needs_wave)
        if name is not None:
            self.name = name

    def evaluate(self, ctx) -> float:
        return self.weight * float(self.fn(ctx))


class ChromaticFocalShiftMerit(MeritTerm):
    """Penalise focal-length variation across wavelengths.

    Evaluates the EFL at each wavelength from the stored
    ``efls_per_wavelength`` on the context and penalises the PV
    (max - min).  For full use, call ``design_optimize`` with a
    ``MultiWavelengthMerit`` wrapper that populates this field.
    """

    needs_wave = False
    name = 'ChromaticFocalShift'

    def __init__(self, weight: float = 1.0):
        self.weight = float(weight)

    def evaluate(self, ctx) -> float:
        if ctx.efls_per_wavelength is None:
            return 0.0
        pv = (np.max(ctx.efls_per_wavelength)
              - np.min(ctx.efls_per_wavelength))
        return self.weight * pv * pv


# =========================================================================
# Multi-wavelength support
# =========================================================================

class MultiWavelengthMerit(MeritTerm):
    """Evaluate a sub-merit at multiple wavelengths and average.

    Populates ``ctx.efls_per_wavelength`` with per-wavelength EFLs
    (computed geometrically, cheap).  The sub-merit is evaluated at
    each wavelength and the results are summed.

    Parameters
    ----------
    wavelengths : sequence of float
        Wavelengths [m] to evaluate at.
    sub_merit : MeritTerm
        Merit term to evaluate at each wavelength.  Its ``evaluate``
        receives a modified ``ctx`` with the corresponding wavelength.
    weight : float
    """

    name = 'MultiWavelength'

    def __init__(self, wavelengths, sub_merit, weight=1.0):
        self.wavelengths = [float(w) for w in wavelengths]
        self.sub_merit = sub_merit
        self.weight = float(weight)
        self.needs_wave = sub_merit.needs_wave

    def evaluate(self, ctx) -> float:
        efls = []
        total = 0.0
        for wl in self.wavelengths:
            # Geometric EFL at this wavelength
            surfs = surfaces_from_prescription(ctx.prescription)
            try:
                _, efl, bfl, _ = system_abcd(surfs, wl)
            except Exception:
                efl = bfl = 1e9
            efls.append(float(efl))
            # Create a sub-context at this wavelength
            sub_ctx = EvaluationContext(
                prescription=ctx.prescription, wavelength=wl,
                N=ctx.N, dx=ctx.dx, efl=float(efl), bfl=float(bfl),
                seidel=ctx.seidel, E_exit=ctx.E_exit,
                opd_map=ctx.opd_map, strehl_best=ctx.strehl_best,
                rms_radius_best=ctx.rms_radius_best, z_best=ctx.z_best)
            total = total + self.sub_merit.evaluate(sub_ctx)
        ctx.efls_per_wavelength = np.array(efls)
        return self.weight * total


# =========================================================================
# Multi-field support (off-axis)
# =========================================================================

class MultiFieldMerit(MeritTerm):
    """Evaluate a sub-merit at multiple field angles.

    At each field angle a tilted plane wave is built, propagated
    through the lens, and the sub-merit is evaluated on the
    resulting wave field.

    Parameters
    ----------
    field_angles : sequence of float
        Field angles in radians (half-angle from optical axis).
        ``0`` = on-axis.
    sub_merit : MeritTerm
        Wave-based merit term to evaluate at each field.
    weight : float
    """

    name = 'MultiField'

    def __init__(self, field_angles, sub_merit, weight=1.0):
        self.field_angles = [float(a) for a in field_angles]
        self.sub_merit = sub_merit
        self.weight = float(weight)
        self.needs_wave = True

    def evaluate(self, ctx) -> float:
        total = 0.0
        for angle in self.field_angles:
            # Build tilted plane wave
            Ny, Nx = ctx.N, ctx.N
            x = (np.arange(Nx) - Nx / 2) * ctx.dx
            y = (np.arange(Ny) - Ny / 2) * ctx.dx
            X, Y = np.meshgrid(x, y)
            k0 = 2 * np.pi / ctx.wavelength
            # Tilt in y direction (tangential)
            tilt_phase = k0 * np.sin(angle) * Y
            E_tilted = np.exp(1j * tilt_phase)
            # Propagate through lens
            E_exit = apply_real_lens(
                E_tilted, ctx.prescription, ctx.wavelength, ctx.dx)
            # Build sub-context
            sub_ctx = EvaluationContext(
                prescription=ctx.prescription,
                wavelength=ctx.wavelength, N=ctx.N, dx=ctx.dx,
                efl=ctx.efl, bfl=ctx.bfl, seidel=ctx.seidel,
                E_exit=E_exit)
            # Through-focus for this field
            if np.isfinite(ctx.bfl) and abs(ctx.bfl) < 10:
                half = max(abs(ctx.bfl) / 20.0, 1e-3)
                z_values = np.linspace(ctx.bfl - half, ctx.bfl + half, 21)
                try:
                    ideal = diffraction_limited_peak(
                        E_exit, ctx.wavelength, ctx.bfl, ctx.dx)
                    scan = through_focus_scan(
                        E_exit, ctx.dx, ctx.wavelength, z_values,
                        ideal_peak=ideal, verbose=False)
                    z_best, strehl_best = find_best_focus(scan, 'strehl')
                    sub_ctx.strehl_best = float(strehl_best)
                    i_best = int(np.argmax(scan.strehl))
                    sub_ctx.rms_radius_best = float(scan.rms_radius[i_best])
                except Exception:
                    sub_ctx.strehl_best = 0.0
            # OPD map if needed
            ap = ctx.prescription.get('aperture_diameter')
            if ap and hasattr(self.sub_merit, 'needs_wave') and self.sub_merit.needs_wave:
                try:
                    from .analysis import wave_opd_2d
                    _, _, opd = wave_opd_2d(
                        E_exit, ctx.dx, ctx.wavelength,
                        aperture=ap, focal_length=ctx.bfl, f_ref=ctx.bfl)
                    sub_ctx.opd_map = opd
                except Exception:
                    sub_ctx.opd_map = None
            total = total + self.sub_merit.evaluate(sub_ctx)
        return self.weight * total / max(len(self.field_angles), 1)


# =========================================================================
# Constraint-style merits
# =========================================================================

class MinThicknessMerit(MeritTerm):
    """Penalise any glass thickness below a minimum.

    ``contribution = weight * sum_surfaces max(0, min_t - t_i)^2``

    Parameters
    ----------
    min_thickness : float
        Minimum acceptable glass/air-gap thickness [m].
    weight : float
    """

    needs_wave = False
    name = 'MinThickness'

    def __init__(self, min_thickness=1e-3, weight=1.0):
        self.min_thickness = float(min_thickness)
        self.weight = float(weight)

    def evaluate(self, ctx) -> float:
        thicknesses = ctx.prescription.get('thicknesses', [])
        total = 0.0
        for t in thicknesses:
            deficit = max(0.0, self.min_thickness - float(t))
            total = total + deficit * deficit
        return self.weight * total


class MaxThicknessMerit(MeritTerm):
    """Penalise any glass thickness above a maximum."""

    needs_wave = False
    name = 'MaxThickness'

    def __init__(self, max_thickness=20e-3, weight=1.0):
        self.max_thickness = float(max_thickness)
        self.weight = float(weight)

    def evaluate(self, ctx) -> float:
        thicknesses = ctx.prescription.get('thicknesses', [])
        total = 0.0
        for t in thicknesses:
            excess = max(0.0, float(t) - self.max_thickness)
            total = total + excess * excess
        return self.weight * total


class MinBackFocalLengthMerit(MeritTerm):
    """Penalise BFL below a minimum (e.g. to keep clearance for
    a sensor package)."""

    needs_wave = False
    name = 'MinBFL'

    def __init__(self, min_bfl=5e-3, weight=1.0):
        self.min_bfl = float(min_bfl)
        self.weight = float(weight)

    def evaluate(self, ctx) -> float:
        deficit = max(0.0, self.min_bfl - ctx.bfl)
        return self.weight * deficit * deficit


class MaxFNumberMerit(MeritTerm):
    """Penalise an f/# above a maximum (force faster lens)."""

    needs_wave = False
    name = 'MaxFNumber'

    def __init__(self, max_f_number=8.0, weight=1.0):
        self.max_f_number = float(max_f_number)
        self.weight = float(weight)

    def evaluate(self, ctx) -> float:
        ap = ctx.prescription.get('aperture_diameter', 1e-3)
        fnum = abs(ctx.efl) / ap if ap > 0 else 1e9
        excess = max(0.0, fnum - self.max_f_number)
        return self.weight * excess * excess


# =========================================================================
# Tolerance-aware merit
# =========================================================================

class ToleranceAwareMerit(MeritTerm):
    """Optimise the MEAN of a sub-merit across a set of random
    perturbations.

    Instead of optimising the *nominal* Strehl / wavefront, this
    optimises the *average* over a Monte-Carlo perturbation set.
    Produces designs that are robust to manufacturing tolerances
    rather than fragile at the nominal but excellent on paper.

    Parameters
    ----------
    sub_merit : MeritTerm
        The merit evaluated at each perturbation (typically
        ``StrehlMerit`` or ``RMSWavefrontMerit``).
    perturbation_spec : list of dict
        Same format as for :func:`monte_carlo_tolerancing`:
        ``[{'surface_index': i, 'decenter_std': ..., 'tilt_std': ...,
            'form_error_rms': ...}]``
    n_trials : int
        Number of random perturbation draws per evaluation.
    seed : int
        Base seed for reproducibility.
    weight : float
    """

    name = 'ToleranceAware'

    def __init__(self, sub_merit, perturbation_spec,
                 n_trials=5, seed=42, weight=1.0):
        self.sub_merit = sub_merit
        self.perturbation_spec = list(perturbation_spec)
        self.n_trials = int(n_trials)
        self.seed = int(seed)
        self.weight = float(weight)
        self.needs_wave = sub_merit.needs_wave

    def evaluate(self, ctx) -> float:
        from .through_focus import apply_perturbations, Perturbation

        total = 0.0
        for t in range(self.n_trials):
            rng = np.random.default_rng(self.seed + t)
            perts = []
            for spec_idx, spec in enumerate(self.perturbation_spec):
                d_std = spec.get('decenter_std', 0.0)
                t_std = spec.get('tilt_std', 0.0)
                f_rms = spec.get('form_error_rms', 0.0)
                # Deterministic form-error seed: tying it directly to
                # the trial index + surface index means two runs with
                # the same ``self.seed`` produce identical form-error
                # realisations regardless of the global RNG state.
                # Mask to 31 bits to match the Perturbation API.
                fe_seed = ((self.seed + t) * 1_000_003
                           + spec['surface_index']
                           + spec_idx * 17) & 0x7FFFFFFF
                perts.append(Perturbation(
                    surface_index=spec['surface_index'],
                    decenter=(rng.normal(0, d_std) if d_std > 0 else 0.0,
                              rng.normal(0, d_std) if d_std > 0 else 0.0),
                    tilt=(rng.normal(0, t_std) if t_std > 0 else 0.0,
                          rng.normal(0, t_std) if t_std > 0 else 0.0),
                    form_error_rms=f_rms,
                    random_seed=fe_seed,
                    name=f'tol_trial_{t}_s{spec["surface_index"]}'))
            pres_pert = apply_perturbations(
                ctx.prescription, perts, N=ctx.N, dx=ctx.dx)

            # Per-trial ABCD: the perturbed prescription generally has a
            # different EFL/BFL from the nominal, and scanning around
            # the nominal BFL misses the actual best focus (giving an
            # artificially low Strehl that drags the optimizer away).
            try:
                surfs_p = surfaces_from_prescription(pres_pert)
                _, efl_p, bfl_p, _ = system_abcd(surfs_p, ctx.wavelength)
                efl_p = float(efl_p) if np.isfinite(efl_p) else ctx.efl
                bfl_p = float(bfl_p) if np.isfinite(bfl_p) else ctx.bfl
            except Exception:
                efl_p, bfl_p = ctx.efl, ctx.bfl

            # Re-run wave propagation for this perturbation
            E_in = np.ones((ctx.N, ctx.N), dtype=np.complex128)
            E_exit = apply_real_lens(
                E_in, pres_pert, ctx.wavelength, ctx.dx)
            sub_ctx = EvaluationContext(
                prescription=pres_pert, wavelength=ctx.wavelength,
                N=ctx.N, dx=ctx.dx, efl=efl_p, bfl=bfl_p)
            # Through-focus scan around the PERTURBED BFL, not the
            # nominal BFL.
            if np.isfinite(bfl_p) and abs(bfl_p) < 10:
                half = max(abs(bfl_p) / 20.0, 1e-3)
                z_values = np.linspace(bfl_p - half, bfl_p + half, 11)
                try:
                    ideal = diffraction_limited_peak(
                        E_exit, ctx.wavelength, bfl_p, ctx.dx)
                    scan = through_focus_scan(
                        E_exit, ctx.dx, ctx.wavelength, z_values,
                        ideal_peak=ideal, verbose=False)
                    z_best, strehl_best = find_best_focus(scan, 'strehl')
                    sub_ctx.strehl_best = float(strehl_best)
                except Exception:
                    sub_ctx.strehl_best = 0.0
            total = total + self.sub_merit.evaluate(sub_ctx)
        return self.weight * total / max(self.n_trials, 1)


# =========================================================================
# Evaluation context
# =========================================================================

@dataclass
class EvaluationContext:
    prescription: Dict[str, Any]
    wavelength: float
    N: int
    dx: float
    efl: float = 0.0
    bfl: float = 0.0
    seidel: np.ndarray = field(default_factory=lambda: np.zeros(5))
    E_exit: Optional[np.ndarray] = None  # wave leg output
    strehl_best: float = 0.0
    rms_radius_best: float = np.inf
    z_best: float = 0.0
    opd_map: Optional[np.ndarray] = None
    efls_per_wavelength: Optional[np.ndarray] = None
    # Populated when a MultiPrescriptionParameterization is used.
    # ``prescription`` stays == ``prescriptions[0]`` for backward
    # compatibility so single-prescription merit terms keep working.
    prescriptions: Optional[List[Dict[str, Any]]] = None

    def rms_wavefront_waves(self, n_modes: int = 21,
                             exclude_low_order: int = 3) -> float:
        """RMS wavefront error in waves, excluding the first
        ``exclude_low_order`` Zernike modes (default: piston, tilt X,
        tilt Y).  Computed from the current OPD map.
        """
        if self.opd_map is None:
            return np.inf
        ap = self.prescription.get('aperture_diameter')
        if ap is None:
            return np.inf
        coeffs, _ = zernike_decompose(
            self.opd_map, self.dx, ap, n_modes=n_modes)
        # rms of higher-order modes, in meters
        higher = coeffs[exclude_low_order:]
        rms_m = float(np.sqrt(np.sum(higher ** 2)))
        return rms_m / self.wavelength  # waves


# =========================================================================
# Main entry point
# =========================================================================

@dataclass
class DesignResult:
    x: np.ndarray
    prescription: Dict[str, Any]
    merit: float
    converged: bool
    iterations: int
    time_sec: float
    context_final: EvaluationContext
    scipy_result: Any = None
    # Populated when a MultiPrescriptionParameterization was used.
    # Otherwise None (use ``prescription`` for the single-lens case).
    prescriptions: Optional[List[Dict[str, Any]]] = None


def design_optimize(parameterization,
                    merit_terms: Sequence[MeritTerm],
                    wavelength: float,
                    N: int = 512,
                    dx: float = 20e-6,
                    E_in: Optional[np.ndarray] = None,
                    method: str = 'L-BFGS-B',
                    max_iter: int = 100,
                    wave_traced: bool = False,
                    ray_subsample: int = 4,
                    z_scan_range: Optional[Tuple[float, float]] = None,
                    z_scan_n: int = 31,
                    verbose: bool = True,
                    progress=None) -> DesignResult:
    """Optimize a lens prescription against a set of merit terms.

    Parameters
    ----------
    parameterization : DesignParameterization
        Template + free variables + bounds.
    merit_terms : sequence of MeritTerm
        Each contributes an (already-weighted) scalar term that is
        summed into the total merit.  ``SphericalSeidelMerit``,
        ``FocalLengthMerit`` etc. are pure-geometric and fast;
        ``StrehlMerit`` / ``RMSWavefrontMerit`` / ``SpotSizeMerit``
        require the wave leg (slower).
    wavelength : float
        Optimization wavelength [m].  For chromatic merits pass a
        list of wavelengths as ``ChromaticFocalShiftMerit``
        dependency (not yet wired; geometric-only chromatic shift).
    N, dx : int, float
        Wave-grid size and spacing.  Only used when any merit term
        has ``needs_wave = True``.
    E_in : ndarray, optional
        Input field for the wave leg.  Defaults to a unit plane wave.
    method : str
        scipy.optimize method.  ``'L-BFGS-B'`` (bounded quasi-Newton,
        default), ``'trust-constr'``, ``'SLSQP'``, or ``'Powell'``.
        For Gauss-Newton / LM treatment, pass ``'lm'`` and the
        optimizer will switch to ``least_squares``.
    max_iter : int
        Maximum outer iterations.
    wave_traced : bool, default False
        If True, use :func:`apply_real_lens_traced` for the wave
        leg (sub-nm OPD accuracy but slower).  Otherwise use
        :func:`apply_real_lens` (fast analytic model).
    ray_subsample : int, default 4
        Passed to ``apply_real_lens_traced`` when used.
    z_scan_range : tuple, optional
        (``z_min``, ``z_max``) relative to the nominal back focal
        length, for the through-focus scan.  Default: ±f/20.
    z_scan_n : int, default 31
        Points in the through-focus scan.
    verbose : bool
    progress : callable, optional
        ``ProgressCallback`` (see :mod:`lumenairy.progress`).
        Fired with ``stage='design_optimize'`` at start (frac=0.0),
        on every merit-function evaluation (``'eval N: ...'``), on
        every scipy iteration where available (``'iter N: ...'``),
        and at completion (frac=1.0).  Monotonic: the bar never
        moves backwards even when eval and iter series leapfrog.
        ``'lm'`` / ``least_squares`` emits eval-only (no scipy iter
        callback exists for it).

    Returns
    -------
    DesignResult
    """
    import scipy.optimize as so
    from .progress import call_progress

    need_wave = any(m.needs_wave for m in merit_terms)
    n_params = parameterization.n_params
    x0 = parameterization.initial_values()
    bounds = parameterization.bounds

    call_count = [0]
    iter_count = [0]
    last_value = [float('inf')]
    last_efl = [0.0]
    last_frac = [0.0]  # monotonic guard: progress bar never moves backwards
    call_progress(progress, 'design_optimize', 0.0,
                  f'method={method}, {len(merit_terms)} merit term(s)')

    multi_mode = isinstance(parameterization, MultiPrescriptionParameterization)

    def _emit_progress(frac: float, msg: str) -> None:
        # Clamp to [last_frac, 0.99] so the bar is monotonic.  The
        # eval-based and iter-based progress series can leapfrog each
        # other; we always take the larger value.
        frac = max(0.0, min(float(frac), 0.99))
        if frac < last_frac[0]:
            frac = last_frac[0]
        last_frac[0] = frac
        call_progress(progress, 'design_optimize', frac, msg)

    def _emit_iter_progress():
        # Fired from scipy's per-iteration callback (accurate iteration
        # counter, unlike merit_fn which fires on every FD gradient eval).
        iter_count[0] += 1
        frac = iter_count[0] / max(max_iter, 1)
        _emit_progress(
            frac,
            f'iter {iter_count[0]}: merit={last_value[0]:.4g}  '
            f'efl={last_efl[0]*1e3:.3f}mm')

    def evaluate(x):
        built = parameterization.build(x)
        if multi_mode:
            prescriptions = list(built)
            # Use the first prescription as the "primary" for backward-
            # compatible single-prescription merits (ABCD, Seidel, etc.).
            pres = prescriptions[0]
        else:
            pres = built
            prescriptions = [pres]
        ctx = EvaluationContext(
            prescription=pres, wavelength=wavelength, N=N, dx=dx,
            prescriptions=prescriptions)
        # Ray-leg (always)
        surfs = surfaces_from_prescription(pres)
        try:
            _, efl, bfl, _ = system_abcd(surfs, wavelength)
        except Exception:
            efl = bfl = float('inf')
        try:
            seidel_raw = seidel_coefficients(surfs, wavelength)
            # seidel_coefficients returns (per-surface-dict, totals-dict)
            if (isinstance(seidel_raw, tuple) and len(seidel_raw) == 2
                    and isinstance(seidel_raw[0], dict)):
                per_surf = seidel_raw[0]
                # Sum each aberration coefficient over surfaces
                seidel = np.array([
                    np.sum(per_surf.get(f'S{k}', np.zeros(1)))
                    for k in range(1, 6)], dtype=np.float64)
            else:
                seidel = np.asarray(seidel_raw, dtype=np.float64)
        except Exception:
            seidel = np.zeros(5)
        ctx.efl = float(efl) if np.isfinite(efl) else 1e9
        ctx.bfl = float(bfl) if np.isfinite(bfl) else 1e9
        ctx.seidel = np.asarray(seidel, dtype=np.float64).ravel()
        # Wave leg (only if any merit term needs it)
        if need_wave:
            if E_in is None:
                E0 = np.ones((N, N), dtype=np.complex128)
            else:
                E0 = E_in
            if wave_traced:
                E_exit = apply_real_lens_traced(
                    E0, pres, wavelength, dx,
                    ray_subsample=ray_subsample, n_workers=1)
            else:
                E_exit = apply_real_lens(E0, pres, wavelength, dx)
            ctx.E_exit = E_exit
            # Through-focus scan
            if not np.isfinite(ctx.bfl) or abs(ctx.bfl) > 10:
                # Probably a bad prescription; skip wave metrics
                return _sum_merits(ctx, merit_terms)
            if z_scan_range is None:
                half = max(abs(ctx.bfl) / 20.0, 1e-3)
                z0, z1 = -half, +half
            else:
                z0, z1 = z_scan_range
            z_values = np.linspace(ctx.bfl + z0, ctx.bfl + z1, z_scan_n)
            ideal = diffraction_limited_peak(
                E_exit, wavelength, ctx.bfl, dx)
            scan = through_focus_scan(
                E_exit, dx, wavelength, z_values,
                ideal_peak=ideal, verbose=False)
            z_best, strehl_best = find_best_focus(scan, 'strehl')
            ctx.z_best = float(z_best)
            ctx.strehl_best = float(strehl_best)
            i_best = int(np.argmax(scan.strehl))
            ctx.rms_radius_best = float(scan.rms_radius[i_best])
            # Build OPD map for Zernike fit
            ap = pres.get('aperture_diameter') or (0.4 * N * dx)
            try:
                _, _, opd_map = wave_opd_2d(
                    E_exit, dx, wavelength, aperture=ap,
                    focal_length=ctx.bfl, f_ref=ctx.bfl)
                ctx.opd_map = opd_map
            except Exception:
                ctx.opd_map = None

        return _sum_merits(ctx, merit_terms), ctx

    def merit_fn(x):
        call_count[0] += 1
        value, ctx = evaluate(x)
        last_value[0] = float(value)
        last_efl[0] = float(ctx.efl) if np.isfinite(ctx.efl) else 0.0
        # Fallback eval-counter progress for methods without a per-
        # iteration callback hook (Powell, DE, dual_annealing, basin-
        # hopping).  For methods with a scipy callback we also emit
        # from there, which is more accurate iteration-wise; the
        # monotonic guard in _emit_progress prevents the bar from
        # going backwards when the two series leapfrog.
        frac = call_count[0] / max(max_iter * 5, 1)
        _emit_progress(
            frac,
            f'eval {call_count[0]}: merit={value:.4g}  '
            f'efl={ctx.efl*1e3:.3f}mm')
        if verbose and call_count[0] % 5 == 1:
            print(f'  iter {call_count[0]}: merit = {value:.6g}  '
                  f'efl = {ctx.efl*1e3:.3f} mm  '
                  f'strehl = {ctx.strehl_best:.4f}')
        return value

    def _scipy_cb_minimize(xk, *args, **kwargs):
        # Callback signature varies by method (some pass xk only,
        # trust-constr passes (xk, state), SLSQP passes xk).  Accept
        # anything.
        _emit_iter_progress()

    def _scipy_cb_de(xk, convergence):
        _emit_iter_progress()

    def _scipy_cb_basin(xk, f, accept):
        last_value[0] = float(f)
        _emit_iter_progress()

    t0 = time.time()
    if method == 'lm':
        # Gauss-Newton / Levenberg-Marquardt via least_squares.  No
        # per-iteration callback is available, so emit progress from
        # inside residuals() using the eval counter.
        def residuals(x):
            call_count[0] += 1
            value, ctx = evaluate(x)
            last_value[0] = float(value)
            last_efl[0] = float(ctx.efl) if np.isfinite(ctx.efl) else 0.0
            frac = call_count[0] / max(max_iter * 5, 1)
            _emit_progress(
                frac,
                f'eval {call_count[0]}: merit={value:.4g}  '
                f'efl={ctx.efl*1e3:.3f}mm')
            return np.array(
                [np.sqrt(max(m.evaluate(ctx), 0.0)) for m in merit_terms],
                dtype=np.float64)
        lb = np.array([b[0] if b else -np.inf for b in (bounds or [None] * n_params)])
        ub = np.array([b[1] if b else +np.inf for b in (bounds or [None] * n_params)])
        res = so.least_squares(
            residuals, x0, method='lm' if not (bounds is not None) else 'trf',
            bounds=(lb, ub) if bounds is not None else (-np.inf, np.inf),
            max_nfev=max_iter, verbose=1 if verbose else 0)
        x_opt = res.x
    elif method in ('differential_evolution', 'de', 'global'):
        # Differential evolution: stochastic global optimizer.
        # Requires bounds for all variables.
        if bounds is None:
            raise ValueError(
                'differential_evolution requires bounds for all variables.')
        res = so.differential_evolution(
            merit_fn, bounds, maxiter=max_iter, seed=42,
            tol=1e-8, disp=verbose, polish=True,
            callback=_scipy_cb_de)
        x_opt = res.x
    elif method == 'basin_hopping':
        # Basin-hopping: global optimizer with local minimisation steps.
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': bounds,
            'options': {'maxiter': 50},
        }
        res = so.basinhopping(
            merit_fn, x0, niter=max_iter,
            minimizer_kwargs=minimizer_kwargs, seed=42,
            disp=verbose, callback=_scipy_cb_basin)
        x_opt = res.x
    elif method == 'dual_annealing':
        if bounds is None:
            raise ValueError(
                'dual_annealing requires bounds for all variables.')
        res = so.dual_annealing(
            merit_fn, bounds, maxiter=max_iter, seed=42,
            callback=lambda x, f, ctx: (
                last_value.__setitem__(0, float(f)),
                _emit_iter_progress()))
        x_opt = res.x
    else:
        res = so.minimize(
            merit_fn, x0, method=method,
            bounds=bounds if method in ('L-BFGS-B', 'SLSQP', 'trust-constr') else None,
            options={'maxiter': max_iter, 'disp': verbose},
            callback=_scipy_cb_minimize)
        x_opt = res.x

    # Final evaluation for the returned context
    final_value, final_ctx = evaluate(x_opt)
    dt = time.time() - t0
    iter_tag = (f'{iter_count[0]} iters, '
                if iter_count[0] > 0 else '')
    call_progress(progress, 'design_optimize', 1.0,
                  f'converged: merit={final_value:.4g} '
                  f'({iter_tag}{call_count[0]} evals, {dt:.1f}s)')

    return DesignResult(
        x=x_opt,
        prescription=final_ctx.prescription,
        merit=float(final_value),
        converged=getattr(res, 'success', True),
        iterations=call_count[0],
        time_sec=dt,
        context_final=final_ctx,
        scipy_result=res,
        prescriptions=(final_ctx.prescriptions if multi_mode else None))


def _sum_merits(ctx, merit_terms):
    total = 0.0
    for m in merit_terms:
        total = total + m.evaluate(ctx)
    return total
