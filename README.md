# lumenairy

[![PyPI](https://img.shields.io/pypi/v/lumenairy.svg)](https://pypi.org/project/lumenairy/)
[![Validate](https://github.com/travaj24/LumenAiry/actions/workflows/validate.yml/badge.svg)](https://github.com/travaj24/LumenAiry/actions/workflows/validate.yml)
[![Python](https://img.shields.io/pypi/pyversions/lumenairy.svg)](https://pypi.org/project/lumenairy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive Python library for coherent optical beam propagation and
manipulation using the Angular Spectrum Method (ASM) and related techniques.

**Author:** Andrew Traverso

## What's new in 3.3.2

- **Embedded grating diffraction in `trace()` and
  `fit_canonical_polynomials`** — new `surface_diffraction` kwarg
  pins a DOE / grating order at a specific surface inside a
  sequential prescription.  Required to do LG-aberration-tensor or
  asymptotic-propagator analysis at non-zero diffraction orders
  (Dammann splitter corner orders, etc.).  Applies the angular kick
  AND adds the DOE's linear-phase OPL contribution `m * lambda *
  (x, y) / period` at the surface, so per-emitter (0, 0) pistons are
  correct even at corner orders.

## What's new in 3.3.1

- **Pre-flight grid-vs-aperture check** — `apply_real_lens`,
  `apply_real_lens_traced`, and `apply_real_lens_maslov` now
  inspect every surface's `semi_diameter` against the simulation
  grid's half-extent (`N*dx/2`) and emit a `UserWarning` if any
  surface exceeds the grid.  This catches the silent-energy-loss
  case where the lens itself would have transmitted past the
  grid edge but the simulation truncates at `N*dx/2`, which
  otherwise manifests downstream as a uniform inward centroid
  bias and missing power.  A new public helper
  `check_grid_vs_apertures(prescription, N, dx,
  safety_factor=1.0)` returns the offending surfaces explicitly
  for use in pre-flight scripts.  Warning fires once per call
  site (Python's default warning filter dedups by source line).

- **Quadoa Optikos `.qos` import/export (best-effort)** —
  `export_quadoa_qos` / `load_quadoa_qos` add round-trip support
  for a Quadoa-Optikos-style JSON system file.  The schema
  (version `QUADOA_SCHEMA_VERSION = '1.0'`) captures every field
  a lumenairy prescription holds — radii (incl. biconic Y),
  conics, asphere coefficients (per axis), glasses, thicknesses,
  semi-diameters, aperture, stop index, wavelength, units.
  Round-trips lossless inside lumenairy; external Quadoa
  readability is unverified pending a reference `.qos`.  The
  library now has full prescription I/O for Zemax (`.zmx`,
  `.txt`), Code V (`.seq`), and Quadoa Optikos (`.qos`).

## What's new in 3.3.0

A new module **`lumenairy.asymptotic`** implementing the closed-form
phase-space (Maslov) diffraction propagator and Laguerre-Gaussian
aberration tensor.  This is the
"missing middle tier" between expensive wave-leg merits
(`StrehlMerit`, `RMSWavefrontMerit`) and cheap ray-leg-only merits
(`SphericalSeidelMerit`, `FocalLengthMerit`):  wave-leg-faithful
quantities (the named aberrations the diffraction integral sees) at
ray-leg-only evaluation cost.

- **`fit_canonical_polynomials`** — Trace a 4-D Chebyshev-node grid
  through any prescription, fit Phi(s2, v2) and s1(s2, v2) as
  4-variable Chebyshev tensor-product polynomials, expose analytic
  gradient evaluation.  Sub-microwave residual on refractive
  systems.

- **`aberration_tensor`** — Evaluate the LG aberration tensor
  T_{k;n,m} at a chief-ray image point.  Indices (p, ell)
  correspond directly to classical Seidel/Zernike aberrations:
  (1, 0) is defocus, (2, 0) is primary spherical, (1, +-1) is coma,
  (0, +-2) is astigmatism, etc.  Closed-form Wick-contracted
  Gaussian moment, no quadrature.

- **`propagate_modal_asymptotic`** — Closed-form leading-order
  asymptotic propagator on a 2-D output grid.  Reduces to Collins'
  ABCD in the source-dominated limit and Fourier-of-pupil in the
  pupil-dominated limit; interpolates smoothly with no caustic
  pathology.  ~10**3-10**4 times faster per pixel than direct
  quadrature.

- **`LGAberrationMerit`** — New `MeritTerm` subclass that drops
  into `design_optimize`.  Specifies named aberration channels to
  suppress (defocus, spherical, coma, ...); evaluates the closed-
  form tensor in milliseconds per merit call.  No wave leg required.

  ```python
  merit = op.LGAberrationMerit(
      targets={(2, 0): 1.0,    # primary spherical
               (1, 1): 1.0,    # coma (sin)
               (1, -1): 1.0,   # coma (cos)
               (0, 2): 1.0, (0, -2): 1.0},   # astigmatism
      field_points=[(0.0, 0.0), (5e-3, 0.0), (0.0, 5e-3)],
  )
  ```

- **LG / HG basis utilities** — `lg_polynomial`, `hg_polynomial`,
  `evaluate_lg_mode`, `evaluate_hg_mode`, `decompose_lg`,
  `decompose_hg`, `lg_seidel_label`.

- **Wick moment utilities** — `gaussian_moment_2d`,
  `gaussian_moment_table_2d`.  Closed-form 2-D Gaussian moments for
  complex-symmetric covariances.

> Validation:  32 new physics-faithful tests in
> `validation/test_asymptotic.py` (LG orthonormality to 1e-14,
> Wick / Isserlis identities, fit round-trip, modal propagator PSF
> peak location, end-to-end LGAberrationMerit).  Full existing test
> suite re-runs green:  no regressions.  No breaking API changes.

## What's new in 3.2.15

- **`apply_doe_phase_traced`** — new ray-trace primitive for splitting
  a `RayBundle` at a thin grating / DOE plane into one or more
  diffraction orders.  Applies the grating-equation direction-cosine
  shift `L_new = L + m_x * lambda / period_x` (and same on y) per
  ray, recomputes `N` from the unit-norm constraint, and flags
  evanescent orders (`L'^2 + M'^2 > 1`) `alive=False` with a new
  `RAY_EVANESCENT = 5` error code.  Two calling conventions: scalar
  orders return a same-shape bundle; 1-D order arrays return a
  bundle replicated `n_orders * n_rays` in order-major layout, ready
  for a single `trace()` call through the post-grating optics.  Use
  case: ray-trace through a Dammann splitter or any thin grating
  in a sequential prescription.

  > Validation: 6 new tests in `test_raytrace.py` cover zero-order
  > no-op, the grating equation, unit-norm preservation, evanescent
  > flagging, order-major layout, and free-space round-trip;
  > 32/32 raytrace + 17/17 optimize tests pass.

## What's new in 3.2.14

A focused performance pass on the highest-traffic paths.  All
behaviour preserved (full validation suite still 16/16 PASS,
298 assertions); changes are transparent caches or opt-in toggles.

- **ASM transfer-function `H` cache** keyed by
  `(N_y, N_x, dy, dx, λ, z, bandlimit, dtype)`.  Repeat propagations
  at the same geometry skip the chunked kernel build entirely.
  Measured **1.5× speedup at N=2048** on cache-hit (the H build is
  ~30-50% of total ASM call time on 2k+ grids).  Tunables:
  `set_asm_cache_size(...)`, `clear_asm_caches()`.

- **Frequency-grid + band-limit caches** complement the H cache so
  even on H-miss the kernel reconstruction skips spatial-frequency
  vector recomputation.

- **Multi-slot pyFFTW plan cache** (was single-slot per direction
  → 8 entries by default).  No more thrashing when calls oscillate
  between two shapes (JonesField stacking, Maslov inner work, batched
  vs scalar grids).  Tunable: `set_fft_plan_cache_size(n)`.

- **Single-precision (complex64) toggles**: `set_default_complex_dtype(np.complex64)`
  flips the library default for fields allocated when callers pass
  real-valued inputs.  All propagators preserve the caller's complex
  dtype; the kernel-phase mod-2π folding keeps single-precision ASM
  accurate at the float32 noise floor.  **2.18× speedup** on N=2048
  ASM, ~2× memory headroom.

- **`angular_spectrum_propagate_batch(E_3d, ...)`** runs a stack of
  fields `(B, Ny, Nx)` through one fused FFT pair across the
  trailing two axes.  `JonesField.propagate` stacks `[Ex, Ey]` and
  routes through it for grids ≥ 512 (below that, dispatch overhead
  exceeds the benefit and the H cache already serves the second
  component for free).

- **Numba-fused aspheric-sum kernel** in `surface_sag_general`.  The
  legacy aspheric loop allocated one fresh N×N array per coefficient
  term; the new path walks `h_sq` once and accumulates all terms in
  a single threaded pass.  **4.75× speedup** at N=2048 with 5
  aspheric coefficients.  Pure spheres unaffected; CuPy stays on
  the legacy path.

- **Memory-leaner refraction step** in `apply_real_lens` (Fresnel /
  slant_correction branches): gradient and intermediate arrays
  freed eagerly.  Drops peak transient memory at N=8192 from ~5 GB
  to ~1.5 GB.  Math unchanged.

## What's new in 3.2.13

- **Validation suite expanded** by ~70 new physics + interop test
  cases.  16 files, 298 Harness assertions across topic suites,
  all PASS.  Highlights: ASM linearity / reciprocity / D4σ growth;
  thin-vs-real lens interop; doublet ABCD-EFL ↔ paraxial focus
  agreement; Strehl-Maréchal; Parseval; Zernike linearity; Malus
  + crossed polarizers + circular S₃; Köhler imaging smoke;
  Cauchy-Schwarz on mutual coherence; Keplerian |M|=f₁/f₂;
  end-to-end singlet wave-vs-trace consistency;
  `propagate_through_system` matches manual `apply_real_lens`+ASM.

## What's new in 3.2.10 - 3.2.12

These were **UI-focused releases** that did not change core
library behaviour.  See `Optical_Propagation_Library_UI/CHANGELOG.md`
for the GUI-specific feature additions (workspace tabs, Welcome
dock, embedded Python REPL, persistent status-bar metrics,
drag-and-drop file open, keyboard shortcuts for workspaces, compact
mode, etc.).  Core API unchanged.

## What's new in 3.2.3

- **`wave_propagator='fresnel'` and `wave_propagator='rayleigh_sommerfeld'`**
  added to `apply_real_lens` (and threaded through `apply_real_lens_traced`).
  Together with the pre-existing `'asm'` (default) and `'sas'`, the
  through-glass propagator can now be switched to any of the four
  physically-sensible choices.  ASM remains the right tool for the mm-scale
  glass distances typical of lenses; the others are exposed for research
  and pipelines that want one propagator used consistently throughout.
  Fresnel and SAS resample back to the input `dx` automatically.  RS
  preserves pitch and agrees with ASM to ~1e-13 at typical through-glass
  distances.  Unknown `wave_propagator` values now raise `ValueError`.

## What's new in 3.2.2

- **`lens_maslov.py` retired; `apply_real_lens_maslov` now lives inside
  `lenses.py`** alongside `apply_real_lens` and `apply_real_lens_traced`.
  Was conceptually a third real-lens wave-optics pipeline all along,
  not a separate subsystem.  Public API unchanged:
  `op.apply_real_lens_maslov` and
  `from lumenairy.lenses import apply_real_lens_maslov`
  both still work.  Only the legacy
  `from lumenairy.lens_maslov import ...` path broke; nothing
  in the library or validation suite used it.

## What's new in 3.2.1

- **SAS integration hooks** — the Scalable Angular Spectrum propagator
  added in 3.2.0 is now a first-class peer of ASM/Fresnel in three
  more places:

    * `propagate_through_system({'type': 'propagate', ..., 'method': 'sas'})`
      — SAS alongside `'asm'` and `'fresnel'` as a per-element method.
      Pipeline auto-resamples back to `dx` so downstream elements keep
      their coordinates.
    * `apply_real_lens(..., wave_propagator='sas')` (+ forwarded through
      `apply_real_lens_traced`) — swap the through-glass propagator.
    * `JonesField.sas_propagate(z, wavelength, pad=2,
      skip_final_phase=False)` — polarization-aware SAS wrapper that
      applies the scalar SAS kernel to `Ex` and `Ey` and updates
      `self.dx` / `self.dy` to the new output pitch.

## What's new in 3.2.0

- **Scalable Angular Spectrum (SAS) propagator**
  (`scalable_angular_spectrum_propagate`).  Three-FFT kernel from
  Heintzmann-Loetgering-Wechsler 2023: ASM-minus-Fresnel precompensation
  phase + band-limit filter + Fresnel chirp + FFT + optional final
  quadratic phase.  Output pitch is `lambda*z/(pad*N*dx)` — a zoom-out
  that avoids the impractical-N trap of plain ASM at long z.  The
  paper's closed-form `z_limit` check warns when exceeded.  Includes a
  Fresnel-style `1/(i·λ·z)·dx²` amplitude prefactor for power
  conservation (the reference PyTorch notebook is amplitude-agnostic).
  Validated against `fresnel_propagate` / `fraunhofer_propagate` at
  moderate / far-field z in their respective limits.

- **CODE V `.seq` import/export**
  (`export_codev_seq` + `load_codev_seq`).  Canonical CODE V sequence
  syntax (`LEN NEW` / `DIM M|MM|IN` / `WL` / `S<i>` / `RDY` / `THI` /
  `GLA` / `CON` / `STO` / `APE`).  Bit-exact round-trip for radii,
  thicknesses, conic, glass, stop index, and aperture.

- **BSDF surface scatter model** (new module `bsdf.py`) with
  `LambertianBSDF`, `GaussianBSDF`, `HarveyShackBSDF`.  Common
  interface: `evaluate(inc, scat)`, `sample(inc, n, rng)`,
  `total_integrated_scatter()`.  Attached to `Surface` via a new
  `bsdf` field.  Helper `sample_scatter_rays(surface, incident,
  n_per_ray)` spawns a `RayBundle` of scattered rays for Monte Carlo
  stray-light propagation through the rest of a system.

- **Jones pupil spatial-map visualization**
  (`plot_jones_pupil` + `compute_jones_pupil`).
  `compute_jones_pupil(apply_fn, N, dx, wavelength)` probes a
  polarization-capable system with orthogonal x/y plane-wave inputs
  and returns the full `(Ny, Nx, 2, 2)` Jones matrix.
  `plot_jones_pupil(J, ...)` produces the canonical 2x4 grid
  (amplitude + phase for each of Jxx/Jxy/Jyx/Jyy) with phase masked
  below an amplitude threshold.

## What's new in 3.1.11

- **Stop-aware `seidel_coefficients`** — new `stop_index` and
  `field_angle` kwargs.  When the declared stop is not at surface 0,
  the chief ray's initial conditions are now derived from the pre-stop
  ABCD so that `y_chief = 0` at the stop by construction.  Default
  uses `find_stop` (which falls back to surface 0 when no surface is
  flagged `is_stop=True`).
- **`refocus(result, delta_z, wavelength=None)`** — closed-form
  image-space transfer of a traced bundle.  `through_focus_rms`
  uses it internally for a 5-20x speedup on focus sweeps.
- **`find_stop`, `compute_pupils`, `find_lenses`, `lens_abcd`,
  `LensInfo`, `PupilInfo`** — new stop / pupil / per-lens paraxial
  helpers.  `lens_abcd` accepts a prescription dict, surface-list
  slice, single `Surface`, or a `LensInfo` (with the original
  `surfaces` passed as a kwarg, for re-analysis at a different
  wavelength).
- **GPU path for `apply_real_lens`** (opt-in `use_gpu=True`).
  `apply_real_lens_traced` forwards `amp_use_gpu` to its two internal
  `apply_real_lens` calls.

## What's new in 3.1.10

- **`apply_real_lens(use_gpu=False)`** — opt-in CuPy backend for
  the full phase-screen + ASM-through-glass pipeline.  Default is
  ``False`` (unchanged CPU behaviour).  When enabled, per-surface
  sag arrays, phase screens, and ASM propagation all run on GPU.
- **`apply_real_lens_traced(amp_use_gpu=False)`** — new kwarg
  pipes ``use_gpu`` through to the internal ``apply_real_lens``
  calls that build the amp + amp(pw) arrays.  The rest of the
  traced pipeline (ray trace, Newton, assembly) stays CPU;
  results are pulled back automatically at the amp-block exit.
  Combines cleanly with the existing ``use_gpu=True`` kwarg for
  the Newton inversion.
- **`surface_sag_general` / `surface_sag_biconic`** now array-API
  polymorphic (accept NumPy or CuPy inputs transparently).

## What's new in 3.1.9

- **`lens_abcd(lens, wavelength)`** + **`find_lenses(surfaces,
  wavelength)`** — paraxial characterisation of individual lens
  elements.  Accepts prescription dicts, surface-list slices, or
  single surfaces.  Auto-detects cemented-doublet grouping.
  Returns EFL, BFL, FFL, principal planes, and the underlying
  ABCD for composition.
- **`compute_pupils(surfaces, wavelength)`** — paraxial entrance
  / exit pupil positions and radii, from the stop surface's ABCD
  sub-system images.  Foundation for future chief-ray aiming
  and rigorous reference-sphere OPD.
- **`RayBundle.error_code`** per-ray diagnostic field recording
  why each dead ray was killed (`RAY_TIR`, `RAY_APERTURE`, etc.).
  `trace_summary` now prints the breakdown.
- **Glass indices pre-resolved** once per `trace()` call; tiny
  per-call speedup, meaningful at high repeated-trace counts
  (aim iteration, focus sweeps, optimisation loops).

## What's new in 3.1.8

- **`trace(output_filter='last')`** eliminates per-surface
  ``RayBundle.copy()`` allocations when only the final bundle
  is consumed.  Saves ~1.4 GB per `apply_real_lens_traced`
  call on a 6-surface doublet at N=32768.  Wired into
  `apply_real_lens_traced` automatically.
- **`refocus(result, delta_z)`** — closed-form image-space
  transfer of a traced bundle.  `through_focus_rms` rewritten
  to use it, giving ~5-20x speedup on focus sweeps.
- **`is_stop` field on `Surface`** + **`find_stop(surfaces)`**
  helper.  Aperture-stop surface can be explicitly flagged
  (Zemax loaders populate it from the STOP keyword); helpers
  for pupil calculation and chief-ray aiming are the natural
  next steps.
- Seidel `S4` (Petzval) double-assignment fixed; dead-code in
  `_paraxial_trace` removed.  Numerical output unchanged.

## What's new in 3.1.7

- **`apply_real_lens_maslov`** — a third thick-lens propagator
  complementing `apply_real_lens` and `apply_real_lens_traced`.  Fits
  a 4-variable Chebyshev tensor-product polynomial to the ray-traced
  canonical map (`s1(s2, v2)`, `OPD(s2, v2)`) and evaluates the
  Maslov phase-space diffraction integral by stationary-phase
  (recommended, closed-form per-pixel), uniform Tukey quadrature
  (extended-source regime), or Hessian-oriented local quadrature
  (asymptotic corrections beyond leading stationary phase).
  Caustic-safe by construction, no critical-sampling constraint,
  analytically differentiable w.r.t. the polynomial coefficients.
  See `CHANGELOG.md` for validation numbers and regime guidance.

- **`apply_real_lens_traced` speedup kwargs** (all opt-in, default
  physics unchanged):

    * `fast_analytic_phase=True` — skip the full ASM-through-glass
      reference-phase pass in favour of a per-pixel sum of sag phase
      screens.  ~25 % wall-time savings when `parallel_amp=False`.
      Introduces <10 nm OPL phase error on typical refractive
      prescriptions.

    * `newton_fit='polynomial'` (new default) or `'spline'` — the
      polynomial path uses a 2-D Chebyshev tensor-product fit
      instead of `scipy.interpolate.RectBivariateSpline` for the
      entrance->exit map.  Implements combined value+gradient
      evaluation and an optional Numba `@njit(parallel=True)`
      fastpath: **~12x faster than the spline path on the Newton
      hot loop** (4M-sample isolated benchmark).  For smooth
      refractive systems (all Seidel and higher-order aberrations
      are polynomials), same or better accuracy with closed-form
      analytic derivatives; flip back to `'spline'` for high-order
      freeforms or sharp non-polynomial surface features.

    * `use_gpu=True` — dispatches the polynomial evaluator and
      Newton inversion to GPU via CuPy (amp/amp(pw)/ray-trace/final
      assembly stay CPU-only).  Requires
      `newton_fit='polynomial'` and cupy installed.  Output is
      bit-equivalent to CPU (0 % RMS error).  Modest absolute
      speedup at typical workloads (~1.4x vs numba CPU on 1-4M
      Newton samples); best for iterated design-optimisation
      workflows or very large grids.

- **Optional `numba` dependency** for the polynomial-Newton CPU
  fastpath.  `requirements.txt` now lists it under "optional"; the
  library falls back to a pure-NumPy path when numba is missing.

- **`apply_real_lens_traced` default `ray_subsample` bumped 1 → 8.**
  At typical production grid sizes (N=2048 and above) this gives
  hundreds to thousands of spline / polynomial samples across every
  lens aperture — far above the internal safety floor.  Small-grid
  users who would drop below 32 samples across aperture get a clear
  error message from the existing `on_undersample='error'` guardrail.

## What's new in 3.1.6

- **Zarr storage reliability fix for Windows + Python 3.14.**
  ``append_plane`` and ``write_sim_metadata`` no longer raise
  ``FileExistsError`` when reopening an existing zarr store.  See
  ``CHANGELOG.md`` for the root-cause breakdown; no API change for
  callers.

## What's new in 3.1.5

- **Zemax loaders preserve object-space distance.**
  `load_zmx_prescription` and `load_zemax_prescription_txt` now
  return a new `object_distance` key on the prescription dict,
  computed as the sum of `DISZ` values from the STOP (or SURF 0 if
  no STOP) up to the first active refractive surface.  This
  recovers design-intended obj-space geometry (coordinate breaks,
  field-reference planes, MLA mount surfaces, etc.) that earlier
  loader versions dropped.  Wave-optics driver scripts should
  propagate their source field by this distance before invoking the
  first lens operator; failing to do so collapses the obj-space
  geometry and produces a defocus-like blur at the image plane
  proportional to the dropped distance (observed on the Design 51
  .zmx: 96.67 mm of dropped air gap → ~235 µm defocus blur at the
  metasurface plane when the distance was not re-injected).

## What's new in 3.1.4

- **`apply_real_lens_traced(..., tilt_aware_rays=...)` default changed
  from `True` to `False`.**  The "Tier 1 input-aware ray launch"
  added in 3.1.2 mixed reference frames between the traced OPL and
  the plane-wave-reference analytic lens phase used in the
  `preserve_input_phase=True` subtraction.  On single-mode
  plane-wave-like inputs the mismatch was small; on multi-mode
  inputs (post-DOE fields, compound superpositions) it produced
  materially wrong output fields.  The plane-wave launch
  (`tilt_aware_rays=False`) is reference-consistent and correct for
  any input the wave model can represent.  See `CHANGELOG.md` for
  the full reasoning.
- **Paraxial-magnification Newton initial guess**: measured from the
  central finite-difference slope of the already-computed forward
  map (zero extra compute) instead of the hard-coded 1.10 multiplier.
  For compound-system callers this saves several Newton iterations
  per pixel; for singlets the improvement is marginal but the guess
  is always at least as good as before.
- **`inversion_method='backward_trace'` opt-in** on
  `apply_real_lens_traced` (experimental).  Replaces the
  forward-trace + Newton-spline-inversion with a direct backward
  ray trace from a coarse subsample of the exit grid through a
  reversed prescription.  Validated to agree with the Newton path
  to sub-pm on single-ray tests and ~30 nm OPD RMS end-to-end at
  N=1024 with a ~3x speedup.  Default stays `'newton'` while the
  backward path is validated on a wider set of prescriptions.

## What's new in 3.1.3

- **Multi-mode-safe `_sample_local_tilts`**: amplitude-weighted
  Gaussian smoothing of the tilt field (new `smooth_sigma_px` kwarg,
  default 4) before clipping, so post-DOE / interferometric inputs
  to `apply_real_lens_traced` degenerate gracefully to a collimated
  launch instead of injecting aliased per-pixel tilts that collapse
  the output field at large N.
- **Complex-dtype flexibility**: `apply_real_lens`, `apply_real_lens_traced`,
  `apply_mirror`, and `angular_spectrum_propagate` preserve the
  caller's complex dtype (complex128 or complex64) end-to-end.
  Kernel-phase and per-surface phase screens always compute in
  float64 + modulo-2-pi reduction before casting, so complex64 mode
  avoids the ~0.02-rad-per-Fourier-pixel precision floor that would
  otherwise swamp large-z ASM propagations.  New top-level export
  `DEFAULT_COMPLEX_DTYPE`.
- **FFT backend refresh**: pyFFTW now uses a single-slot per-direction
  plan cache with in-place aligned buffers (no more 30 s TTL alloc
  churn) and a per-plan `threading.Lock` so parallel callers share
  the cache safely.  Clean `reset_fft_backend()` support.
- **Parallel amp + amp(pw)** (new `parallel_amp=True` default) runs
  the two internal `apply_real_lens` calls inside `apply_real_lens_traced`
  on a `ThreadPoolExecutor`.  FFT-serialised, non-FFT work
  overlapped.  Auto-disables when available RAM is tight
  (`parallel_amp_min_free_gb`).
- **Amplitude-masked Newton** (`newton_amp_mask_rel=1e-4`): skip
  coarse-grid Newton pixels where the analytic amplitude is below
  threshold, since they'd be multiplied by ~zero in the final
  assembly anyway.  Big speedup on post-DOE / sparse fields; mask
  self-disables on dense fields.
- **Numexpr-fused phase screen in `apply_real_lens`** (optional
  dependency `[perf]`): `ne.evaluate('E * exp(-1j*k0*opd)', out=E)`
  eliminates ~50 GB of complex128 intermediates per surface at
  N=32768 and threads the operation.  Numpy fallback preserved.
- **Decenter-aliased entrance grids** in `apply_real_lens`: `Xs, Ys,
  h_sq` alias the axis-centred grids when decenter is zero (the
  common case), saving three float64 NxN allocations per surface.
- New top-level export `NUMEXPR_AVAILABLE` for runners that want to
  gate tunables on the fast path being importable.

## What's new in 3.1

- **Progress hooks** (`progress.py`) — optional `progress=callback` kwarg
  on `apply_real_lens`, `apply_real_lens_traced`, and
  `propagate_through_system`.  Drive a progress bar from any script or
  GUI; callback signature is `(stage, fraction, message)`.
  `ProgressScaler` lets long pipelines nest sub-tasks inside a parent
  budget.
- **`'real_lens_traced'` element type** for `propagate_through_system`
  so the hybrid wave/ray lens model is reachable through the unified
  element-list API.
- **Codegen promoted to public API** — `generate_simulation_script`,
  `generate_script_from_zmx`, and `generate_script_from_txt` are now
  exported from the top-level `lumenairy` namespace.
- **`remove_wavefront_modes` accepts `weights=`** for intensity-weighted
  piston / tilt / defocus fits; crucial on vignetted or annular pupils.
- **Freeform sags ray-traceable** — `Surface.freeform` plumbed through
  `_surface_sag_xy`, `_surface_sag_derivatives_xy`, and
  `surfaces_from_prescription` so XY-polynomial / Zernike / Chebyshev
  freeforms work in both wave and ray paths.
- **`make_singlet` / `make_doublet`** always emit biconic keys (set to
  `None`) for diff-friendly, round-trippable prescriptions.
- **`optical_table.py` + `.html` removed** — the bundled HTML simulator
  was unreferenced and its launcher functions were never exported.

## What's new in 3.0

- **`apply_real_lens_traced`** — high-accuracy hybrid wave/ray lens model.
  Sub-nanometer OPD agreement with the geometric ray trace on cemented
  doublets and other multi-surface curved-interface systems.  Uses
  `RectBivariateSpline` over the entrance grid + vectorised Newton
  inversion of the entrance→exit mapping; orders of magnitude better than
  the analytic thin-element model where it matters.  Amplitude from
  `apply_real_lens` (full ASM-through-glass), phase from ray-traced OPL.
- **Exit-vertex OPL correction** — `apply_real_lens_traced` now transfers
  rays from the last surface's sag to the flat exit vertex plane before
  computing OPD, using the **signed** parametric distance (not absolute
  value) so both concave and convex rear surfaces are handled correctly.
  Previously, off-axis rays ended at `z = sag(h) ≠ 0` while on-axis rays
  ended at `z = 0`, injecting systematic defocus (43 % on doublets) or
  catastrophic sign errors (200,000× on negative meniscus lenses).
  Doublet focus error: 10 mm → **0.000 mm**.  Negative meniscus residual:
  33,742 nm → **0.17 nm**.
- **Critical raytrace OPL bookkeeping fix** — `_intersect_surface` now
  accounts for the small "vertex-plane → actual-sag-intersection" leg in
  the right medium.  Singlet wave-vs-geom residuals dropped 17×–130×;
  every consumer of `raytrace.trace` benefits automatically.
- **Biconic / cylindrical / toroidal surfaces** — `surface_sag_biconic`,
  `make_cylindrical`, `make_biconic`, plus `radius_y` / `conic_y` /
  `aspheric_coeffs_y` keys on any prescription surface.  All downstream
  consumers (ray tracer, OPD analysis, both `apply_real_lens` variants,
  Seidel, ABCD) handle anamorphic surfaces transparently.
- **Zernike decomposition** — `zernike_decompose(opd_map, dx, aperture,
  n_modes)` fits OSA-normalised Zernikes via Householder QR with column
  pivoting (numerically stable for high-order, partial-pupil cases).
  Round-trips to ~1e-12 precision.  Plus `zernike_reconstruct`,
  `zernike_polynomial`, OSA index helpers.
- **Hybrid wave/ray design optimizer** (`lumenairy.optimize`)
  — refine a lens prescription against geometric and/or wave-based
  merit terms (focal length, Seidel S₁, Strehl ratio, RMS wavefront,
  spot size, chromatic focal shift).  Wraps `scipy.optimize` (L-BFGS-B
  by default; `lm` routes through Householder-QR-based Levenberg-
  Marquardt).  Pure-geometric optimization runs sub-second; wave-based
  metrics scale with grid size.
- **OPD-extraction Nyquist tooling** — `check_opd_sampling()` helper +
  built-in `RuntimeWarning` in `wave_opd_1d` / `wave_opd_2d` when
  sampling near or below the Nyquist edge for the lens's converging
  wavefront.  Optional `f_ref` parameter divides out a reference sphere
  before unwrap for users who want coarser grids.
- **SciPy FFT default** — `USE_SCIPY_FFT = True`, `SCIPY_FFT_WORKERS = -1`
  by default.  All wave-propagation calls now multithreaded; 2-4× speedup
  with zero memory overhead.  pyFFTW is opt-in.
- **`slant_correction` default reverted to `False`** — empirical
  validation showed paraxial `(n2−n1)·sag` is equal-or-better for almost
  every test case because the angular-spectrum propagation between
  surfaces already encodes the obliquity.  Slant correction remains
  available as opt-in.
- **Validation suite** — `validation/real_lens_opd/` now compares three
  methods (paraxial / slant / ray-traced) on 21 reference lenses with
  matching Zemax LDE + `.zmx` exports for cross-verification.

## Overview

`lumenairy` provides a physically accurate, modular toolkit for
simulating free-space coherent optics. It handles everything from basic
Gaussian beam propagation to multi-surface real lens modeling with glass
dispersion, metasurface design, and full Jones-vector polarization.

Features are implemented with well-tested physics (verified against textbook
formulas and audited for sign conventions), SI units throughout, and optional
GPU / multi-threaded FFT acceleration.

## Key Features

### Propagation
- **Angular Spectrum Method (ASM)** — exact, band-limited, with rectangular
  anti-aliasing filter
- **Tilted / off-axis ASM** — for beams with a non-zero carrier angle
- **Single-FFT Fresnel** — paraxial, changes output grid spacing
- **Fraunhofer (far-field)** — simplest far-field computation
- **Rayleigh-Sommerfeld** — convolution with the free-space Green's function,
  exact near-field diffraction without band-limiting approximation
- **Scalable Angular Spectrum (SAS)** — Heintzmann-Loetgering-Wechsler 2023
  three-FFT kernel with variable output pitch (`λz/(pad·N·dx)`); the right
  tool when z is long enough that plain ASM needs an impractically large
  grid

### Lenses
- **Thin lens** (paraxial, non-paraxial, aplanatic, local-only)
- **Spherical singlet** — exact OPD through thick glass
- **Aspheric singlet** — conic + even polynomial coefficients
- **Multi-surface real lens** (`apply_real_lens`) — split-step refraction
  with ASM between surfaces.  Optional Fresnel transmission, bulk
  absorption, slant correction, Seidel correction, biconic surfaces.
- **Hybrid wave/ray real lens** (`apply_real_lens_traced`) — high-accuracy
  variant: per-pixel ray-traced OPL combined with wave-amplitude
  envelope.  Sub-nm OPD agreement with the geometric ray trace on
  multi-surface curved-interface systems.  3.1.7: optional
  `fast_analytic_phase`, `newton_fit='polynomial'`, and
  default `ray_subsample=8` for large-grid speedups.
- **Phase-space / Maslov real lens** (`apply_real_lens_maslov`) — third
  real-lens pipeline (added 3.1.7; merged into `lenses.py` in 3.2.2).
  Chebyshev polynomial fit of the ray-traced canonical map +
  closed-form stationary-phase or phase-space quadrature evaluation.
  Caustic-safe; no critical-sampling constraint; analytically
  differentiable.  Best for caustic-near output planes, very coarse
  grids, or design-optimization loops that need gradient information.
- **Pluggable through-glass propagator** on `apply_real_lens` (and
  forwarded through `apply_real_lens_traced`) via `wave_propagator`:
  `'asm'` (default), `'sas'`, `'fresnel'`, `'rayleigh_sommerfeld'`.
  ASM is the right physics for mm-scale glass gaps; the others are
  exposed for cross-validation and pipelines that want a single
  propagator used consistently throughout.
- **Biconic / cylindrical / toroidal** elements via `make_biconic` and
  `make_cylindrical` plus optional `radius_y`/`conic_y` keys on any
  prescription surface
- **GRIN rod lens** — gradient-index parabolic profile
- **Axicon** — conical lens (Bessel-beam generator)

### Geometric Ray Tracing
- **Sequential 3-D ray tracer** — vectorised Snell's law with exact
  conic/aspheric surface intersection (Newton iteration)
- **Surface types** — sphere, conic, aspheric (Zemax standard sag), flat, mirror
- **ABCD matrix** extraction — EFL, BFL, FFL from paraxial marginal ray
- **Seidel aberrations** — per-surface third-order coefficients (S1–S5)
- **Spot diagrams** — with RMS/GEO radius, Airy disc overlay
- **Ray fan plots** — transverse ray aberration vs normalised pupil
- **OPD analysis** — wavefront error fans
- **Through-focus** — RMS spot vs defocus with best-focus finder
- **Ray generators** — fans, grids, concentric rings, single rays
- **Diffraction-order shift** — `apply_doe_phase_traced` splits a ray
  bundle at a thin grating / DOE into one or more orders (single or
  array), with evanescent flagging via `RAY_EVANESCENT`
- **Prescription compatible** — same prescription dicts as `apply_real_lens`
- **System compatible** — `raytrace_system()` accepts the same element list
  as `propagate_through_system()` for instant wave-optics ↔ ray-optics switching

### Mirrors
- Flat or curved (spherical / conic) with optional aperture

### Apertures and Masks
- **Hard apertures** — circular, rectangular, annular
- **Soft apertures** — Gaussian
- **Arbitrary complex masks** — for SLMs, metasurfaces, custom DOEs

### Wavefront
- **Zernike aberrations on a pupil** — `apply_zernike_aberration` for
  generating wavefronts from Zernike coefficients
- **Zernike decomposition of OPD maps** — `zernike_decompose` /
  `zernike_reconstruct` using Householder QR with column pivoting
  (numerically stable, OSA-normalised, RMS coefficients in meters)
- **Zernike basis primitives** — `zernike_polynomial(n, m, rho, theta)`,
  `zernike_basis_matrix`, `zernike_index_to_nm`, `zernike_nm_to_index`
- **OPD extraction from wave fields** — `wave_opd_1d`, `wave_opd_2d`
  with Nyquist sampling warnings, optional reference-sphere subtraction
- **Sampling-rule helper** — `check_opd_sampling` reports the Nyquist
  margin and recommends grid sizing for clean OPD extraction
- **Low-order mode removal** — `remove_wavefront_modes` (piston / tilt /
  defocus least-squares fit and subtract)
- **Turbulence phase screens** — Kolmogorov and von Karman statistics

### Polarization (Jones calculus)
- **`JonesField` class** — wraps (Ex, Ey) with all standard propagators
  (ASM, Fresnel, Fraunhofer, tilted ASM, SAS as of 3.2.1) and all
  non-polarizing element methods (thin/spherical/real lens, aperture,
  mirror, mask)
- **Polarization elements** — polarizers, waveplates, rotators, arbitrary Jones matrices
- **Polarized sources** — linear, circular, elliptical
- **Analysis** — Stokes parameters, degree of polarization, polarization ellipse
- **Jones-pupil spatial map** (3.2.0) — `compute_jones_pupil(apply_fn, ...)`
  probes a system with orthogonal x/y inputs to extract the full 2×2
  exit-pupil Jones matrix, and `plot_jones_pupil` renders it as the
  canonical 2×4 amplitude + phase grid

### Beam Sources
- Fundamental Gaussian (TEM00)
- Hermite-Gauss modes (HG_{mn})
- Laguerre-Gauss modes (LG_{pl}) with OAM
- **Tilted plane wave** — off-axis collimated source at arbitrary field angles
- **Point source** — diverging spherical wave from (x0, y0, z0)
- **Multi-field source generator** — batch-produce tilted plane waves for field analysis

### Beam Analysis
- Centroid (center of mass)
- D4sigma (ISO 11146) beam diameter
- Power-in-bucket (circular or rectangular)
- Strehl ratio
- PSF, OTF, MTF (including radial MTF profiles)
- Sampling-condition diagnostics
- **Chromatic focal shift** — per-wavelength EFL/BFL and axial colour PV
- **Polychromatic Strehl** — weighted Strehl average across wavelengths

### High-NA Vector Diffraction (Richards-Wolf)
- **`richards_wolf_focus`** — compute (Ex, Ey, Ez) vectorial focal field
  from a scalar or Jones-vector pupil, handling NA > 0.5 where scalar
  diffraction breaks down (longitudinal Ez component)
- **`debye_wolf_psf`** — intensity PSF |E|^2 including all polarisation
  components
- Supports arbitrary input polarisation (x, y, circular, or custom Jones)
- Multi-z-plane evaluation for 3-D focal-volume tomography

### Partial Coherence / Extended-Source Imaging
- **`koehler_image`** — Koehler condenser illumination model: integrates
  coherent sub-images over the condenser NA to produce a partially-
  coherent image
- **`extended_source_image`** — arbitrary source angle distribution with
  per-direction weights
- **`mutual_coherence`** — compute Gamma(r1, r2) from a field ensemble

### Detector / Wavefront-Sensor Simulation
- **`apply_detector`** — pixel-integrate a field onto a detector grid
  with Poisson shot noise, Gaussian read noise, dark current, full-well
  saturation, and quantum efficiency
- **`shack_hartmann`** — simulate a Shack-Hartmann wavefront sensor:
  sub-aperture extraction, lenslet focusing, centroid detection, and
  wavefront reconstruction via slope integration

### Diffractive Optical Elements
- Periodic phase mask tiling (for DOEs, gratings)
- Microlens array
- **Dammann grating IFTA design** — generates uniform spot arrays

### Glass Catalog
- Refractive index lookup via the [refractiveindex.info](https://refractiveindex.info) database
- Includes Schott, Ohara, fused silica, silicon, CaF₂, etc.
- Cached material objects for fast repeated lookups

### Lens Prescriptions
- Build singlets and cemented doublets by glass name + geometry
- **Thorlabs catalog presets** — LA1050-C, LA1509-C, AC254-050-C, AC254-100-C, etc.
- **Zemax `.zmx` parser** — import real lens prescriptions from Zemax files
- **Zemax `.txt` parser** — import Zemax prescription-report text files
- **Zemax `.zmx` / `.txt` exporters** — round-trip designs back to Zemax
- **CODE V `.seq` import / export** — round-trips prescriptions through the
  canonical CODE V sequence syntax (units M/MM/IN, spherical + conic
  surfaces, stop flag, aperture; unknown directives skipped on import)

### Stray-Light / BSDF (3.2.0)
- **Three BSDF models** with a common `evaluate` / `sample` /
  `total_integrated_scatter` interface: `LambertianBSDF` (uniform
  diffuse), `GaussianBSDF` (small-angle lobe around specular),
  `HarveyShackBSDF` (three-parameter ABC model for polished
  microroughness with optional wavelength scaling)
- **`Surface.bsdf` field** — attach a BSDF to any sequential-system
  surface
- **`sample_scatter_rays(surface, incident, n_per_ray)`** — spawn a
  `RayBundle` of scattered rays sampled from the surface's BSDF for
  Monte Carlo stray-light propagation through the remainder of a
  system

### User Library
- **Persistent material catalog** — save custom glasses (fixed index or from
  refractiveindex.info) for reuse across sessions and scripts
- **Lens library** — save and load prescription dicts (singlets, doublets,
  custom designs, Thorlabs catalog lenses)
- **Phase mask library** — save mathematical expressions (e.g. spiral phase
  plates), pre-computed arrays, or glass-block definitions
- All stored as JSON in ``~/.lumenairy/library/``
- Saved materials auto-register in ``GLASS_REGISTRY`` on import

### Phase Retrieval
- **Gerchberg-Saxton** — phase-only CGH design between source and target
- **Error Reduction** — coherent diffractive imaging
- **Hybrid Input-Output (HIO)** — Fienup's feedback algorithm

### Prescription → Simulation Script (codegen)
- **`generate_simulation_script`** — turn a prescription dict into a
  standalone, runnable Python script that imports `lumenairy`,
  defines the prescription inline, builds an element list, and propagates
  a source through it. Useful for archiving a simulation alongside a
  design, sending a reproducible reference to a collaborator, or dropping
  the generated code into a Jupyter notebook as a starting point.
- **`generate_script_from_zmx`** / **`generate_script_from_txt`** —
  one-call path from a `.zmx` file or Zemax prescription text export
  straight to a simulation script.
- Styles: `'unrolled'` (one call per element, easy to edit) or
  `'system'` (single `propagate_through_system` call, compact).
- Toggle `include_analysis` and `include_plotting` to control how much
  post-propagation code is emitted.

### I/O
- CSV phase file read/write (with metadata header)
- FITS file read/write (optional, requires `astropy`)
- **HDF5 file read/write** (optional, requires `h5py`) — single fields,
  multi-plane propagation datasets, and polarized JonesFields with
  hierarchical groups, compression, and rich metadata attributes

### Plotting (optional, requires `matplotlib`)
- **Field visualizations** — intensity (log/linear), phase (with
  low-intensity masking), combined intensity + phase panels
- **Cross-sections** — 1D cuts through any axis with optional phase overlay
- **Multi-plane grids** — automatic layout for propagation simulation
  results with per-plane labels and z-positions
- **PSF / MTF** — 2D PSF plots and radial MTF profiles (with optional
  diffraction-limit overlay)
- **Polarization** — 4-panel Stokes parameter maps and polarization
  ellipse overlays on intensity images
- **Beam profile** — 1D intensity cross-section with D4σ markers and
  optional Gaussian fit

### System Propagation
- `propagate_through_system()` — pass a field through an ordered list of
  elements with one function call
- `raytrace_system()` — geometric ray-trace the **same** element list for
  quick cross-validation against wave-optics

### Through-focus / Tolerancing
- **`through_focus_scan`** — propagate the exit field across a range of
  axial planes and tabulate Strehl, peak intensity, D4σ spot, RMS
  radius, encircled energy at each plane
- **`find_best_focus`** — optimise the metric of choice across the scan
- **`single_plane_metrics`** — full set of beam metrics at a single z
- **`diffraction_limited_peak`** — reference for Strehl computations
- **`tolerancing_sweep`** — apply a list of `Perturbation` (decenter,
  tilt, form-error) and compare best-focus Strehl/spot for each
- **`monte_carlo_tolerancing`** — random perturbation draws from
  user-specified distributions, aggregate Strehl statistics

### Hybrid Wave/Ray Design Optimization (`lumenairy.optimize`)
- **`DesignParameterization`** — flat-vector ↔ prescription dict mapping
  with arbitrary path-based free variables and bounds
- **`MeritTerm`** building blocks: `FocalLengthMerit`,
  `BackFocalLengthMerit`, `SphericalSeidelMerit`, `StrehlMerit`,
  `RMSWavefrontMerit`, `SpotSizeMerit`, `ChromaticFocalShiftMerit`,
  **`LGAberrationMerit`** (closed-form named aberration suppression
  via the LG aberration tensor — see Phase-space asymptotic propagator)
- **`design_optimize`** — main driver wrapping `scipy.optimize`
  (L-BFGS-B / SLSQP / trust-constr / `lm`).  Wave leg only runs when a
  wave-based merit term needs it; pure-geometric optimization is
  sub-second for typical lenses
- `lm` method routes through `scipy.optimize.least_squares`, which uses
  Householder QR with column pivoting under the hood

### Phase-space asymptotic propagator (`lumenairy.asymptotic`)
- **`fit_canonical_polynomials`** — 4-variable Chebyshev tensor-product
  fit of `Phi(s2, v2)` and `s1(s2, v2)` from a ray-traced grid; sub-
  microwave residuals on refractive systems
- **`aberration_tensor`** — closed-form Laguerre-Gaussian aberration
  tensor whose indices correspond directly to classical Seidel/Zernike
  aberrations (defocus, spherical, coma, astigmatism, trefoil, ...)
- **`propagate_modal_asymptotic`** — closed-form leading-order Maslov
  propagator; Collins-ABCD in source-dominated limit, Fourier-of-pupil
  in pupil-dominated limit; ~10³-10⁴× faster per pixel than direct
  Maslov quadrature
- **`solve_envelope_stationary`** — Newton-solve the chief-ray
  envelope-stationary equation directly on the Chebyshev fit
- **LG / HG basis utilities** — `lg_polynomial`, `hg_polynomial`,
  `evaluate_lg_mode`, `evaluate_hg_mode`, `decompose_lg`, `decompose_hg`,
  `lg_seidel_label`
- **Wick moment utilities** — `gaussian_moment_2d`,
  `gaussian_moment_table_2d` for 2-D Gaussian moments under complex-
  symmetric covariance

## Installation

### Development install (editable)

From the project root (the `Optical_Propagation_Library/` directory),
install the package in editable mode:

```bash
cd Optical_Propagation_Library
pip install -e .
```

This makes `lumenairy` importable from anywhere and any edits
to the source files take effect immediately without reinstalling.

### Standard install

```bash
cd Optical_Propagation_Library
pip install .
```

### Manual (no install)

If you prefer not to install the package, you can place your scripts
next to the `Optical_Propagation_Library` directory and add it to
`sys.path`:

```python
import sys
sys.path.insert(0, 'Optical_Propagation_Library')
import lumenairy as op
```

### Usage

Once installed (or on `sys.path`):

```python
import lumenairy as op
# or
from lumenairy import angular_spectrum_propagate, JonesField
```

## Dependencies

### Required
- `numpy` — core numerics
- `refractiveindex` — glass catalog lookups (only needed if using
  `get_glass_index`, `apply_real_lens`, or the Thorlabs/Zemax helpers)

### Optional
- `scipy` — used by default for multi-threaded FFTs
  (`USE_SCIPY_FFT = True`, `SCIPY_FFT_WORKERS = -1`), Zernike
  decomposition (Householder QR), and `design_optimize`
- `pyfftw` — extra ~10-20% on top of SciPy FFT (opt-in via
  `op.propagation.USE_PYFFTW = True`); ~2× memory per FFT plan
- `cupy` — GPU acceleration (auto-detected; pass `use_gpu=True` to supported
  functions)
- `astropy` — FITS file I/O for `load_fits_field` / `save_fits_field`
- `h5py` — HDF5 field storage (`save_field_h5`, `save_planes_h5`, etc.)
- `matplotlib` — all plotting utilities (`plot_intensity`, `plot_stokes`,
  etc.) and the `makedammann2d` progress display

Install the required dependencies:

```bash
pip install numpy refractiveindex
```

Install optional dependencies as needed:

```bash
pip install pyfftw astropy h5py matplotlib
```

## Quick Start

### Basic propagation

```python
import numpy as np
import lumenairy as op

# Create a Gaussian beam
E, x, y = op.create_gaussian_beam(N=512, dx=2e-6, sigma=50e-6)

# Propagate 10 cm through free space
E_prop = op.angular_spectrum_propagate(E, z=0.1, wavelength=1.3e-6, dx=2e-6)

# Analyze
cx, cy = op.beam_centroid(E_prop, 2e-6)
dx_b, dy_b = op.beam_d4sigma(E_prop, 2e-6)
print(f"Centroid: ({cx*1e6:.1f}, {cy*1e6:.1f}) um")
print(f"D4sigma:  {dx_b*1e6:.0f} x {dy_b*1e6:.0f} um")
```

### Geometric ray tracing

```python
import lumenairy as op

# Load a prescription and ray-trace it
rx = op.thorlabs_lens('AC254-100-C')
surfaces = op.surfaces_from_prescription(rx)

# ABCD matrix and focal lengths
abcd, efl, bfl, ffl = op.system_abcd(surfaces, wavelength=1.31e-6)
print(f"EFL = {efl*1e3:.1f} mm, BFL = {bfl*1e3:.1f} mm")

# Trace rays and generate a spot diagram
result = op.trace_prescription(rx, wavelength=1.31e-6, num_rings=8,
                               image_distance=bfl)
op.spot_diagram(result, units='um')
op.trace_summary(result)

# Same element list for wave-optics AND ray-optics
elements = [
    {'type': 'propagate', 'z': 50e-3},
    {'type': 'lens', 'f': 100e-3},
    {'type': 'propagate', 'z': 100e-3},
]
# Wave-optics
E_out, _ = op.propagate_through_system(E_in, elements, 1.31e-6, dx)
# Geometric ray trace — same element list
result, surfs = op.raytrace_system(elements, 1.31e-6, semi_aperture=5e-3)
```

### Real lens from Zemax file

```python
# Load a lens prescription from a Zemax .zmx file
rx = op.load_zmx_prescription('path/to/lens.zmx')

# Or use a Thorlabs catalog lens
rx = op.thorlabs_lens('AC254-200-C')

# Fast analytic thin-element model (default)
E_out = op.apply_real_lens(E_in, rx, wavelength=1.3e-6, dx=2e-6)

# Higher-accuracy hybrid wave/ray model -- sub-nm OPD on doublets
E_out = op.apply_real_lens_traced(E_in, rx, wavelength=1.3e-6, dx=2e-6,
                                   ray_subsample=4)
```

### Generate a simulation script from a prescription

```python
# Turn a Zemax .zmx into a self-contained Python sim script
import lumenairy as op

rx = op.load_zmx_prescription('AC254-100-C.zmx')
code = op.generate_simulation_script(
    rx,
    wavelength=1.31e-6,
    N=2048,
    style='unrolled',          # or 'system' for a single propagate_through_system call
    include_analysis=True,
    include_plotting=True,
)
with open('sim_AC254_100C.py', 'w') as f:
    f.write(code)

# Or one-shot from a file path
code = op.generate_script_from_zmx('AC254-100-C.zmx', wavelength=1.31e-6)
```

The output is a runnable script with the prescription data inline, ready
to drop into version control alongside the design or hand to a
collaborator.

### Anamorphic / cylindrical / biconic elements

```python
# Cylindrical lens (focuses in x only)
pres = op.make_cylindrical(R_focus=50e-3, d=3e-3, glass='N-BK7', axis='x')
E_line_focus = op.apply_real_lens(E_in, pres, wavelength=1.3e-6, dx=2e-6)

# Biconic singlet (independent x and y curvatures)
pres = op.make_biconic(R1_x=50e-3, R1_y=70e-3,
                        R2_x=-30e-3, R2_y=-40e-3,
                        d=4e-3, glass='N-BK7')
E_anam = op.apply_real_lens(E_in, pres, wavelength=1.3e-6, dx=2e-6)
```

### Zernike decomposition of an OPD map

```python
# Extract the OPD map from a wave field
E_exit = op.apply_real_lens(E_in, prescription, wavelength, dx)
X, Y, opd = op.wave_opd_2d(E_exit, dx, wavelength,
                            aperture=10e-3, focal_length=100e-3,
                            f_ref=100e-3)

# Decompose into 21 Zernike modes (covers up through 5th-order spherical)
coeffs, names = op.zernike_decompose(opd, dx, aperture=10e-3, n_modes=21)
for j, (c, n) in enumerate(zip(coeffs, names)):
    print(f'  Z{j:2d} {n:30s}: {c*1e9:+8.2f} nm RMS')

# Reconstruct from a coefficient set
opd_recon = op.zernike_reconstruct(coeffs, dx, opd.shape, aperture=10e-3)
```

### Sampling check for OPD extraction

```python
# Before committing to a long simulation, verify the grid is fine
# enough for clean OPD unwrap at the pupil edge
samp = op.check_opd_sampling(dx=4e-6, wavelength=1.31e-6,
                              aperture=12e-3, focal_length=45e-3)
print(f'  Nyquist margin: {samp["margin"]:.2f}  (>= 2 = safe)')
if not samp['ok']:
    for rec in samp['recommendations']:
        print('  Suggestion:', rec)
```

### Hybrid wave/ray lens-design optimization

```python
# Refine a Thorlabs achromat to hit a custom focal-length target
template = op.thorlabs_lens('AC254-100-C')
template['aperture_diameter'] = 10e-3

param = op.DesignParameterization(
    template=template,
    free_vars=[
        ('surfaces', 0, 'radius'),
        ('surfaces', 1, 'radius'),
        ('surfaces', 2, 'radius'),
        ('thicknesses', 0),
    ],
    bounds=[(50e-3, 80e-3),
            (-60e-3, -30e-3),
            (-250e-3, -150e-3),
            (4e-3, 8e-3)])

merit = [
    op.FocalLengthMerit(target=110e-3, weight=1.0),
    op.SphericalSeidelMerit(weight=1e-10),
    op.StrehlMerit(min_strehl=0.95, weight=10.0),
]

result = op.design_optimize(parameterization=param,
                             merit_terms=merit,
                             wavelength=1.31e-6,
                             N=256, dx=20e-6,
                             method='L-BFGS-B',
                             max_iter=50)
print(f'Final EFL: {result.context_final.efl*1e3:.3f} mm')
print(f'Best Strehl: {result.context_final.strehl_best:.4f}')
print('Optimised prescription:', result.prescription)
```

### Progress reporting from long-running operations

Any of the core library's slow entry points accept an optional
`progress` callback so scripts and GUIs can drive a progress bar
from the same hook:

```python
import lumenairy as op

def cb(stage, fraction, message=''):
    print(f'{stage}: {fraction*100:5.1f}%  {message}')

# Wave-optics pipeline
E_out = op.apply_real_lens_traced(
    E_in, prescription, wavelength=1.31e-6, dx=2e-6,
    ray_subsample=4, progress=cb)
E_out, _ = op.propagate_through_system(
    E_in, elements, wavelength=1.31e-6, dx=2e-6, progress=cb)

# Through-focus and tolerancing
scan = op.through_focus_scan(E_exit, dx, wavelength, z_values, progress=cb)
results = op.tolerancing_sweep(prescription, wavelength, N, dx, E_source,
                                perturbations, focal_length=bfl,
                                aperture=ap, progress=cb)
stats = op.monte_carlo_tolerancing(prescription, wavelength, N, dx, E_source,
                                    spec, focal_length=bfl, aperture=ap,
                                    n_trials=100, progress=cb)

# Design optimization (progress is per merit-function evaluation)
result = op.design_optimize(parameterization=param, merit_terms=merits,
                             wavelength=1.31e-6, max_iter=200, progress=cb)
```

The callback signature is `(stage: str, fraction: float, message: str)`
where `fraction` is in `[0, 1]`.  Implementations should be cheap and
thread-safe; exceptions raised inside the callback are swallowed so a
broken progress UI cannot crash a simulation.

`ProgressScaler` lets a parent caller nest sub-tasks within a budget
so long pipelines (`apply_real_lens` inside `apply_real_lens_traced`,
which itself is one of many surfaces inside `propagate_through_system`,
which might be inside `tolerancing_sweep`) report a single monotonic
0\u20131 timeline.  See `lumenairy/progress.py` for the full
protocol.

### Through-focus and tolerancing

```python
# Run a 21-plane through-focus scan
E_exit = op.apply_real_lens(E_in, prescription, wavelength, dx)
ideal_peak = op.diffraction_limited_peak(E_exit, wavelength, bfl, dx)
z_values = bfl + np.linspace(-1e-3, +1e-3, 21)
scan = op.through_focus_scan(E_exit, dx, wavelength, z_values,
                              ideal_peak=ideal_peak,
                              bucket_radius=20e-6)
z_best, strehl_best = op.find_best_focus(scan, 'strehl')
op.plot_through_focus(scan, best_z=z_best, path='through_focus.png')

# Tolerancing: how does Strehl change with surface tilt / decenter?
perts = [
    op.Perturbation(surface_index=0, tilt=(1e-3, 0),       name='S0 tilt 1 mrad'),
    op.Perturbation(surface_index=1, decenter=(50e-6, 0),  name='S1 decenter 50 um'),
    op.Perturbation(surface_index=2, form_error_rms=100e-9,
                    random_seed=42, name='S2 form error 100 nm RMS'),
]
results = op.tolerancing_sweep(prescription, wavelength, N, dx,
                                E_in, perts,
                                focal_length=bfl, aperture=10e-3,
                                bucket_radius=20e-6)
```

### Polarization

```python
# Create a right-hand circularly polarized Gaussian beam
scalar, _, _ = op.create_gaussian_beam(256, 2e-6, 30e-6)
field = op.create_circular_polarized(scalar, dx=2e-6, handedness='right')

# Propagate through a half-wave plate at 22.5°
op.apply_half_wave_plate(field, angle=np.pi/8)

# Apply a lens (polarization-preserving)
field.apply_thin_lens(f=100e-3, wavelength=1.3e-6)

# Propagate
field.propagate(z=100e-3, wavelength=1.3e-6)

# Measure Stokes parameters
S = op.stokes_parameters(field)
print(f"S3/S0 = {S['S3'].mean() / S['S0'].mean():+.3f}")
```

### Phase retrieval (Gerchberg-Saxton CGH design)

```python
# Design a phase-only DOE to turn a Gaussian into a flat-top
x = np.linspace(-1, 1, 256)
X, Y = np.meshgrid(x, x)
source = np.exp(-(X**2 + Y**2) / 0.3**2)
target = (np.sqrt(X**2 + Y**2) < 0.4).astype(float)

phase, err = op.gerchberg_saxton(source, target, n_iter=500)
# 'phase' is the design phase-only DOE
```

### Save and load multi-plane simulations (HDF5)

```python
# Save a propagation simulation with multiple planes
planes = [
    {'field': E0, 'dx': 2e-6, 'z': 0.0,    'label': 'source'},
    {'field': E1, 'dx': 2e-6, 'z': 10e-3,  'label': 'after lens'},
    {'field': E2, 'dx': 2e-6, 'z': 100e-3, 'label': 'focal plane'},
]
op.save_planes_h5('simulation.h5', planes, wavelength=1.3e-6)

# Load back later
planes, meta = op.load_planes_h5('simulation.h5')
print(f"Wavelength: {meta['wavelength']*1e9:.0f} nm")
for p in planes:
    print(f"  {p['label']}: z={p['z']*1e3:.1f} mm, shape={p['field'].shape}")

# Append planes incrementally during a long simulation
op.append_plane_h5('simulation.h5', E_new, dx=2e-6, z=200e-3,
                   label='detector plane')

# Save a polarized Jones field
op.save_jones_field_h5('polarized.h5', jones_field, wavelength=1.3e-6)
```

### Plotting

```python
import matplotlib.pyplot as plt

# Single field intensity and phase
fig, axes = op.plot_field(E, dx=2e-6, title='Focal plane')

# Log-scale intensity
fig, ax = op.plot_intensity(E, dx=2e-6, log=True)

# Cross-section with phase overlay
fig, ax = op.plot_cross_section(E, dx=2e-6, axis='x', show_phase=True)

# Multi-plane grid from a loaded HDF5 file
planes, _ = op.load_planes_h5('simulation.h5')
fig, axes = op.plot_planes_grid(planes, n_cols=3, suptitle='Propagation')

# PSF and MTF
fig1, ax1 = op.plot_psf(psf, dx_psf=dx_psf, log=True)
fig2, ax2 = op.plot_mtf(freq, mtf_profile, diffraction_limit=100)

# Stokes parameters for a polarized field
fig, axes = op.plot_stokes(jones_field)

# Polarization ellipses overlaid on intensity
fig, ax = op.plot_polarization_ellipses(jones_field, n_ellipses=12)

plt.show()
```

### PSF / MTF analysis

```python
# Build a circular pupil with some spherical aberration
pupil = op.apply_aperture(
    np.ones((256, 256), dtype=complex),
    dx=25e-6, shape='circular',
    params={'diameter': 5e-3}
)
pupil = op.apply_zernike_aberration(
    pupil, dx=25e-6,
    coefficients={(4, 0): 0.25},       # 1/4 wave spherical
    aperture_radius=2.5e-3
)

# Compute PSF and MTF
psf, dx_psf = op.compute_psf(pupil, wavelength=1.3e-6, f=50e-3, dx_pupil=25e-6)
mtf = op.compute_mtf(psf)
freq, mtf_profile = op.mtf_radial(mtf, dx_psf, 1.3e-6, 50e-3)
```

## Project Layout

```
Optical_Propagation_Library/            # project root
    README.md                            # this file
    LICENSE                              # MIT license
    CHANGELOG.md                         # version-by-version release notes
    pyproject.toml                       # build / install configuration
    requirements.txt                     # runtime dependencies
    lumenairy/                 # the importable package
        __init__.py                      # public API re-exports (272 symbols)
        propagation.py                   # ASM, tilted ASM, Fresnel, Fraunhofer,
                                         # Rayleigh-Sommerfeld, Scalable ASM
                                         # (SciPy-FFT default, multithreaded)
        lenses.py                        # Thin/thick/aspheric/real lenses +
                                         # apply_real_lens_traced (hybrid) +
                                         # apply_real_lens_maslov (phase-space),
                                         # surface_sag_general / _biconic,
                                         # cylindrical / GRIN / axicon
        glass.py                         # Glass catalog + refractiveindex.info
        coatings.py                      # Thin-film stack (TMM) + QW / broadband
                                         # AR coating designs
        bsdf.py                          # Surface-scatter BSDF models
                                         # (Lambertian, Gaussian, Harvey-Shack)
        elements.py                      # Mirrors, apertures, masks, Zernike
                                         # aberrations, turbulence phase screens
        sources.py                       # Gaussian, HG, LG, top-hat, annular,
                                         # Bessel, LED, fiber, tilted plane,
                                         # point source, multi-field generator
        freeform.py                      # XY-poly / Zernike / Chebyshev
                                         # freeform surface sags
        analysis.py                      # Centroid, D4σ, Strehl, PSF/OTF/MTF,
                                         # OPD extraction (1D/2D),
                                         # Zernike decomposition (QR-based),
                                         # check_opd_sampling, chromatic
        doe.py                           # Gratings, MLA, Dammann, FITS / phase
                                         # file I/O
        interferometry.py                # Interferogram synthesis + PSI
        phase_retrieval.py               # Gerchberg-Saxton, HIO, ER
        prescriptions.py                 # Singlet, doublet, biconic, cylindrical,
                                         # Thorlabs catalog, Zemax I/O, CODE V I/O
        system.py                        # Sequential element-list propagator
                                         # (accepts method='asm'|'fresnel'|'sas')
        raytrace.py                      # Geometric ray tracer (Snell, ABCD,
                                         # Seidel, pupils, find_lenses,
                                         # refocus, through_focus_rms, spot,
                                         # ray fan, OPD fan, error codes;
                                         # biconic-aware; OPL fix)
        through_focus.py                 # Strehl, best-focus, single-plane
                                         # metrics, tolerancing, MC analysis
        optimize.py                      # Hybrid wave/ray design optimizer
                                         # (DesignParameterization, 12+ MeritTerms,
                                         #  scipy.optimize wrapper; DE, basin-hop)
        vector_diffraction.py            # Richards-Wolf high-NA vectorial focus
        coherence.py                     # Koehler/extended-source partially-
                                         # coherent imaging, mutual coherence
        detector.py                      # Detector pixel model (shot/read noise,
                                         # dark current, full-well saturation),
                                         # Shack-Hartmann WFS simulator
        polarization.py                  # Jones vector / JonesField (with SAS
                                         # propagate), waveplates, Stokes
        ghost.py                         # Ghost-path analysis for a multi-
                                         # surface lens
        multiconfig.py                   # Multi-configuration / afocal
                                         # (Keplerian, beam expander)
        rcwa.py                          # 1-D rigorous coupled-wave analysis
        storage.py                       # Unified HDF5/Zarr auto-dispatch backend
        hdf5_io.py                       # Back-compat re-export shim for storage
        memory.py                        # RAM budget and batch-size helpers
        user_library.py                  # Persistent user material/lens/mask library
        codegen.py                       # Auto-generate sim scripts from Zemax
        plotting.py                      # Matplotlib plots (field, PSF, MTF,
                                         # Stokes, polarization ellipses,
                                         # Jones pupil 2×4 grid)
        progress.py                      # ProgressCallback + ProgressScaler
        _backends.py                     # CPU / thread helpers
    validation/                          # topic-based validation suite
        _harness.py                      # Shared Harness class
        run_all.py                       # discovers test_*.py + runs in
                                         # fresh subprocesses (per-file PASS/FAIL
                                         # + aggregate summary)
        test_propagation.py              # ASM, Fresnel, Fraunhofer, R-S, SAS,
                                         # tilted, sampling
        test_lenses.py                   # thin/real/biconic/cyl/asph/GRIN/axicon
                                         # + Maslov/traced, wave_propagator switch
        test_raytrace.py                 # Snell, ABCD, Seidel, pupils, refocus,
                                         # through_focus, spot/fans/off-axis
        test_analysis.py                 # Zernike, Strehl, MTF, Airy, Gaussian-
                                         # ABCD, OPD metrics, overlap invariance
        test_sources.py                  # 10 beam generators
        test_elements.py                 # apertures, mirror, turbulence, Zernike
        test_polarization.py             # HWP, QWP, Stokes, Jones, Jones pupil
        test_advanced_diffraction.py     # Richards-Wolf + Koehler + mutual
        test_optimize.py                 # all merits + L-BFGS / DE / basin-hop
        test_io.py                       # HDF5 / Zemax / CODE V / user_library
        test_features.py                 # coatings / DOE / RCWA / freeform /
                                         # ghost / interferometry / multi-config
                                         # + BSDF
        test_detector.py                 # detector / Shack-Hartmann / GS
        test_glass_tolerancing.py        # dispersion / achromat / tolerances
        test_integration.py              # API exports / compositions / memory
                                         # / plotting smoke
        test_subsample.py                # apply_real_lens_traced subsample
                                         # guardrail + ProcessPool + scaling
        test_validation_lens.py          # known-answer lens harness
        real_lens_opd/                   # 3-method OPD comparison sweep
            run_validation.py             # paraxial / slant / ray-traced
            results/                      # per-case PNGs + report.md/csv
            zemax_prescriptions/          # matching .txt/.zmx for cross-check
        through_focus_smoke/             # quick Strehl + tolerancing demo
```

## Conventions

- **Time dependence:** `exp(-i*omega*t)` throughout
- **Units:** SI meters for all spatial quantities
- **Sign convention for radii:** positive = center of curvature to the right
  of the surface (standard optics / Zemax convention)
- **Grid:** always square (N × N), centered at the origin

## Physics Notes

The library is designed around the Angular Spectrum Method, which is exact
(within sampling limits) for free-space propagation. All band-limiting uses
the Matsushima-Shimobaba (2009) rectangular criterion for anti-aliasing.

### Real-lens accuracy strategy

Two complementary models are provided:

- **`apply_real_lens` (analytic thin-element)** — split-step refraction
  with ASM between surfaces.  Each refracting surface is treated as a
  phase screen computed from the exact surface sag (conic + polynomial
  aspheric + biconic if specified), and the wave propagates through
  the glass between surfaces in the in-medium wavelength `lambda/n`.
  Captures diffraction during in-glass propagation; sub-100-nm RMS
  agreement with geometric ray trace on most singlets.  Has a hard
  accuracy ceiling on cemented doublets and other multi-surface
  curved-interface systems because the wave model treats each glass
  region as a vertex-to-vertex uniform slab while real rays cross
  interior interfaces at z = sag(h).
- **`apply_real_lens_traced` (hybrid wave/ray)** — bypasses that ceiling
  by computing the exit-pupil OPD from a geometric ray trace (per-pixel
  Newton inversion of the entrance→exit map via cubic splines) and
  combining with a wave-optics amplitude envelope.  Sub-nanometer OPD
  agreement with the geometric ray trace when properly sampled, at the
  cost of ~10–30 s on N=4096 grids.

A critical OPL-bookkeeping fix in `raytrace._intersect_surface` (the
small "vertex-plane → actual-sag-intersection" leg now correctly
accumulates `n·path` in the right medium) cut singlet wave-vs-geom
residuals by 17×–130× and brought the geometric reference itself into
agreement with sequential ray tracers used elsewhere in the optics
community.

### Sampling rule for OPD extraction

Extracting OPD from a converging-wavefront simulation requires
`dx ≤ λ·f / aperture` so that `np.unwrap` doesn't lose cycles at the
pupil edge.  `check_opd_sampling()` reports the Nyquist margin and
flags marginal sampling.  `wave_opd_1d` / `wave_opd_2d` accept a
`focal_length` argument to emit a `RuntimeWarning` when the sampling
is risky, and an `f_ref` argument to subtract a reference sphere
before unwrap (allows coarser grids at the cost of needing the
focal length up front).

## License

MIT License — see `LICENSE` file.

## Acknowledgments

The Dammann grating design function (`makedammann2d`) is a Python port by
Andrew Traverso of the original Octave/MATLAB implementation by Daniel Marks.
