# Changelog — lumenairy

All notable changes to the core library are documented here.

## [3.3.2] — 2026-05-04

### Feature — Embedded grating diffraction in `trace()` and `fit_canonical_polynomials`

`trace()` and `fit_canonical_polynomials` gain a new `surface_diffraction`
keyword argument that pins a chosen DOE / grating order at a specific
surface inside the prescription.  This unblocks LG-aberration-tensor
analysis (and the asymptotic propagator) at non-zero diffraction
orders -- previously, geometric tracing only saw the (0, 0) order
because `apply_doe_phase_traced` operates on standalone `RayBundle`
objects, not surfaces in a sequential prescription.

```python
fit = op.fit_canonical_polynomials(
    prescription, wavelength,
    source_box_half=...,
    surface_diffraction={
        doe_surf_idx: (m_x, m_y, period_x, period_y),
    },
)
```

The kick obeys the standard grating equation
`L_new = L + m_x * lambda / period_x` (and same on y) at the
specified surface, applied AFTER refraction.  Evanescent orders
(`L_new**2 + M_new**2 > 1`) flag rays `alive=False` with
`error_code=RAY_EVANESCENT`.

**Importantly, the OPL accumulator IS updated** with the grating's
linear phase contribution `m * lambda * (x, y) / period` evaluated at
the ray's DOE-plane intersection -- the "constant phase shift"
`apply_doe_phase_traced` explicitly does NOT add but the LG
aberration fit needs to see in order to give correct (0, 0)-piston
phases per emitter.  Without this, the per-emitter pistons at a
non-zero order are inconsistent with the fit's chief-ray landing,
and the LG aberration tensor's piston channel reports nonsensical
inter-emitter phase relationships.

3 new tests in `validation/test_raytrace.py` cover the angular kick,
the OPL contribution, and evanescent-order flagging.

## [3.3.1] — 2026-05-02

### Feature — Pre-flight grid vs prescription-aperture check

`apply_real_lens`, `apply_real_lens_traced`, and `apply_real_lens_maslov`
now run a one-shot check at entry that compares each surface's
`semi_diameter` against the simulation grid's half-extent (`N*dx/2`)
and emits a `UserWarning` if any surface exceeds the grid.

This is the silent-energy-loss case where the lens itself would have
transmitted energy past `N*dx/2` but the simulation grid's hard
boundary clips it.  It manifests downstream as a uniform inward
centroid bias and missing power, and is otherwise difficult to
distinguish from real aberration.  The warning lists the offending
surfaces with their semi-diameters and the largest gap, and points
the user to either grow `N` or coarsen `dx`.

**New public API:**

- **`check_grid_vs_apertures(prescription, N, dx, *, safety_factor=1.0)`**.
  Returns a list of `(label, semi_aperture_m, grid_semi_m, gap_m)`
  for every prescription surface whose `semi_diameter` exceeds
  `safety_factor * N * dx / 2`.  Empty list means the grid is wide
  enough.  Pass `safety_factor=0.95` to flag surfaces that come
  within 5% of the grid edge (recommended for clean Gaussian-wing
  containment).

The warning fires once per call site (Python's default warning
filter dedups by source line), so heavy multi-element systems do
not get spammed.

### Feature — Quadoa Optikos `.qos` import/export (best-effort)

`export_quadoa_qos` / `load_quadoa_qos` add round-trip support for a
Quadoa-Optikos-style JSON system file.  Quadoa's official schema is
not fully publicly documented, so this writer emits a self-defined
JSON layout (schema version `QUADOA_SCHEMA_VERSION = '1.0'`) that
captures every field a lumenairy prescription holds:

- per-surface radii (incl. biconic `radius_y`),
- conics (incl. `conic_y`),
- aspheric coefficients (incl. per-Y axis),
- glasses on both sides of the surface,
- thicknesses, semi-diameters, comments,
- aperture diameter, stop index, wavelength, and units.

Round-trips losslessly inside lumenairy.  External Quadoa
readability is **not yet verified** — for verified interchange,
validate against a known-good reference `.qos`; the docstring
calls this out explicitly.

The library now has full I/O support for Zemax (`.zmx`, `.txt`),
Code V (`.seq`), and Quadoa Optikos (`.qos`).

Validation: 4 new tests in `validation/test_io.py` covering doublet
round-trip, `units='MM'` round-trip, asphere coefficients +
semi_diameter + biconic Y round-trip, and a sanity check that a
round-tripped prescription drives `apply_real_lens` without error.

## [3.3.0] — 2026-05-03

### Feature — Phase-space asymptotic propagator and Laguerre-Gaussian aberration tensor

A new module `lumenairy.asymptotic` implementing the closed-form
Gaussian-moment evaluation of the phase-space (Maslov) diffraction
integral.  This complements
the existing `apply_real_lens_maslov` -- which evaluates the same
underlying integral by direct Chebyshev-quadrature in v_2 -- by
replacing the quadrature with a finite Wick-contracted moment over a
complex-symmetric covariance matrix built from the Chebyshev
polynomial fit.

**What's new:**

- **`fit_canonical_polynomials(prescription, wavelength, ...)` ->
  `CanonicalPolyFit`**.  Trace a 4-D Chebyshev-node grid through any
  prescription, fit Phi(s2, v2) and s1(s2, v2) as 4-variable
  Chebyshev tensor-product polynomials, and return a fit container
  with analytic gradient evaluation.  Sub-microwave residual on
  refractive systems; includes the linear-phase-extraction trick of
  paper 1 Section 5 for diffractive surfaces at non-zero orders.

- **`aberration_tensor(fit, s2_image, ...)` -> `AberrationTensorResult`**.
  Compute the Laguerre-Gaussian aberration tensor T_{k;n,m} at a
  chief-ray image point.  Indices (p, ell) of the output basis
  correspond directly to classical Seidel/Zernike aberrations:
  (0, 0) is piston/Strehl, (1, 0) is defocus, (2, 0) is primary
  spherical, (1, +-1) is coma, (0, +-2) is astigmatism, (0, +-3)
  is trefoil, etc.  Closed-form Wick-contracted Gaussian moment;
  no quadrature.

- **`propagate_modal_asymptotic(fit, source_amplitudes,
  pupil_amplitudes, ...)` -> ndarray**.  Closed-form leading-order
  asymptotic propagator on a 2-D output grid.  Reduces to Collins'
  ABCD law in the source-dominated limit (large source waist) and
  to the Fourier-of-pupil diffraction-limited spot in the
  pupil-dominated limit; interpolates smoothly between with no
  special handling of caustics.  ~10**3 to 10**4 times faster per
  pixel than direct quadrature; with NaN guards on Newton
  divergence near caustics or out-of-box pixels.

- **`solve_envelope_stationary(fit, s2, source_point, w_s, w_p, ...)`**.
  Newton-solve the envelope-stationary equation (paper 2 eq. 9) for
  the v_2* that maximises the joint Gaussian envelope.  Used inside
  the propagator and the aberration tensor; exposed for users who
  want to inspect the chief-ray geometry directly.

- **`LGAberrationMerit(targets={(p, ell): weight, ...},
  field_points=[...], ...)`**.  A new `MeritTerm` subclass that
  drops directly into `design_optimize`.  Targets named aberration
  channels (defocus, spherical, coma, ...) by output LG index;
  single-call evaluation via `aberration_tensor`.  No wave leg
  required (`needs_wave = False`), so the merit runs at
  millisecond-per-evaluation cost while measuring the same
  physically-named aberrations the wave leg cares about.

- **LG / HG basis utilities** (`lg_polynomial`, `hg_polynomial`,
  `evaluate_lg_mode`, `evaluate_hg_mode`, `decompose_lg`,
  `decompose_hg`, `lg_seidel_label`).  Polynomial-coefficient
  representation of the Laguerre-Gaussian and Hermite-Gaussian
  bases as Cartesian polynomial * shared Gaussian envelope -- the
  form needed by the closed-form Gaussian-moment integrators.
  Verified orthonormal to machine precision on circular
  (LG, w=1mm) and elliptical (HG, wx=1mm wy=1.5mm) cases.

- **Wick moment utilities** (`gaussian_moment_2d`,
  `gaussian_moment_table_2d`).  Closed-form 2-D Gaussian moment
  evaluator for complex-symmetric covariances, with a moment-table
  builder for amortising across many mode-pair contractions.
  Verified against Isserlis identities and direct numerical
  quadrature.

**Why this matters for design optimisation:**

The wave-leg-aware merits (`StrehlMerit`, `RMSWavefrontMerit`, etc.)
are physically faithful but expensive (full ASM propagation per
evaluation).  The ray-leg-only merits (`SphericalSeidelMerit`,
`FocalLengthMerit`) are cheap but only see paraxial geometry; on
high-NA / strongly-aberrated systems they can drive an optimisation
in directions the wave leg disagrees with.

`LGAberrationMerit` is the missing middle tier:  wave-leg-faithful
quantities (the named aberrations the diffraction integral sees) at
ray-leg-only cost.  It is the recommended primary merit for
diffraction-limited design optimisation that needs to converge
quickly across many parameter sweeps (e.g. radii + thicknesses +
conics + aspherics simultaneously).

**Validation:**

A new test file `validation/test_asymptotic.py` covers all 32
identities and end-to-end paths:

- LG / HG basis orthonormality (round / elliptical waist) to 1e-14.
- Wick moment identities:  unit zeroth moment, second moments
  match Sigma_ij to 1e-12, fourth-moment Isserlis identities,
  hand-computed sixth-moment correctness, closed-form vs.
  numerical quadrature agreement to 1e-15.
- Polynomial multiply, shift, and linear substitution unit tests.
- LG / HG decomposition round-trip recovers a known mode.
- Canonical fit:  sub-microwave Phi residual on N-BK7 singlet,
  round-trip evaluation matches direct ray trace, J = ds1/dv2
  has non-trivial magnitude (catches single-source-point
  degeneracy), in_box mask correctness, linear-phase round-trip.
- Newton stationary solver converges in 1 iteration on a clean
  on-axis singlet test.
- Modal propagator:  finite-valued field, PSF peaks at the
  on-axis chief-ray image point.
- Aberration tensor:  evaluates end-to-end with the right shape
  and finite content.
- LGAberrationMerit:  evaluates without error, responds to
  curvature changes, returns a finite penalty when the prescription
  is degenerate (no exceptions propagated to the optimiser).

> Validation: 32/32 new tests pass.  Full library suite of 17
> existing files re-runs green:  no regressions introduced.

**Compatibility:**

No breaking changes.  All existing APIs unchanged.  New module is
purely additive; new merit term subclasses `MeritTerm` and uses the
same `EvaluationContext` as every other merit.

## [3.2.15] — 2026-05-03

### Feature — `apply_doe_phase_traced`: grating diffraction-order shift for ray bundles

New public function in `lumenairy.raytrace` for splitting a
`RayBundle` into one or more diffraction orders at a thin grating /
DOE plane.  Applies the grating-equation direction-cosine shift
`L_new = L + m_x * lambda / period_x` (and the same on the y-axis)
to every ray, recomputes `N` from the unit-norm constraint, and
flags evanescent orders (`L'^2 + M'^2 > 1`) as `alive=False` with a
new error code `RAY_EVANESCENT = 5`.

Two calling conventions:

- **Scalar orders** -- pass `order_x`, `order_y` as scalars; returns a
  bundle the same length as the input.
- **Order arrays** -- pass 1-D arrays of equal length; returns a
  replicated bundle in *order-major* layout (all rays for order 0,
  then order 1, ...).  This is the form used to split a single
  pre-DOE bundle into N orders for one downstream `trace()` call.

Use case: ray-trace through a Dammann splitter or any thin grating
in a sequential prescription.  Before this, callers had to construct
`RayBundle` instances directly and apply the k-shift inline; this
function packages the bookkeeping (broadcast, evanescent flagging,
`error_code` propagation under the first-failure-wins invariant)
and matches the public `trace` / `make_*` API conventions.

Exports: `apply_doe_phase_traced`, `RAY_EVANESCENT`.

> Validation: all 32 raytrace tests pass (6 new), 17 optimize tests
> pass.

## [3.2.14.1] — 2026-04-25

### Bugfix — H-cache OOM at very large N (regression introduced in 3.2.14)

Mirrors the core 3.2.14.1 fix.  The 3.2.14 ASM transfer-function
`H` cache was bounded by entry count (default 8) but not by
bytes; at N=32768 each H is 16 GB so the cache could hold up to
~128 GB of transfer functions, starving `apply_real_lens` of the
RAM it needs for its own sag intermediates.  Caught running
Design 51 traced simulations at N=32768 -- the run failed with
`numpy._core._exceptions._ArrayMemoryError: Unable to allocate
8.00 GiB ...` deep inside `surface_sag_general` partway through
the second lens group.

The H cache now enforces a **per-entry size cap** (default 2 GB,
silently rejects entries above) and a **total bytes budget**
(default 8 GB, LRU-evicts to fit).  At N=32768 the cache
transparently disables itself; lookups miss, H is rebuilt per
call, the result is still correct.  Tunable via
`set_asm_cache_size(h_max_bytes_per_entry=, h_max_total_bytes=)`.

No GUI-side changes; the GUI inherits the safer cache policy
automatically.

> Validation: all 16 files / 298 assertions pass on both
> libraries.

## [3.2.14] — 2026-04-24

### Performance — ASM caches + multi-slot FFTW + batched JonesField + numba aspherics (mirrors core)

Mirrors the core library's 3.2.14 perf pass.  No GUI changes; UI
library version bumped in lock-step.  See the core CHANGELOG for
the per-feature breakdown.

Highlights for UI users:
- Wave Optics simulations that propagate at the same z multiple
  times (multi-config sweeps, per-wavelength loops, optimization
  iterations) now cache the ASM transfer function H — repeat
  propagations are ~1.55× faster at N=2048.
- `JonesField.propagate` runs Ex/Ey through a single batched FFT
  pair on grids ≥ 512.
- `set_default_complex_dtype(np.complex64)` is now exposed at the
  package top level — flip it once in your wave-optics preset for
  ~1.6× FFT throughput and ~2× memory headroom (all propagators
  preserve the caller's dtype, and the existing kernel-phase
  mod-2π folding keeps accuracy at the float32 noise floor).
- pyFFTW now keeps up to 8 plans resident (multi-slot LRU) so
  switching between (Ex, Ey) shape and a 3-D batch shape no longer
  thrashes the plan cache.
- `apply_real_lens` aspheric loop is JIT-fused via numba — pure
  spheres unaffected, aspherics get a single threaded pass.

> **Total**: 16 files, 298 Harness assertions, all PASS.

## [3.2.13] — 2026-04-24

### Validation — physics & interop hammer expansion (mirrors core)

Roughly +70 new test cases added to the core library's validation
suite covering cross-pipeline interop and physics invariants.  No
GUI changes in this release; UI library version bumped to track the
core in lock-step.  See the core CHANGELOG for the per-file
breakdown.

> **Total**: 16 files, 298 Harness assertions across topic suites
> (74 net new vs. 3.2.9 baseline).  All pass.

## [3.2.12] — 2026-04-24

### UI — full polish pass: keyboard, drag-drop, persistent metrics, REPL, compact mode

A round of quality-of-life enhancements covering navigation,
visibility, customization, and ad-hoc analysis.  Core library
unchanged.

**Quick wins**

- **`Ctrl+1` … `Ctrl+9` jump between workspace tabs.** Match the
  Zemax/optiland muscle memory; no more mouse trips to the tab bar.

- **Window title reflects the loaded file + dirty state.** Format is
  `Optical Designer — file.zmx*`, with `*` appended when the design
  has unsaved changes.  Cleared by Save / Open / New.

- **Drag-and-drop `.zmx`, `.txt`, `.seq`, `.json` onto the window**
  to load.  Uses the same paths as File > Open.

- **Permanent right-aligned status-bar metrics**: EFL, BFL, f/#, EPD,
  wavelength.  Visible on every workspace, no need to keep System
  Data open just to glance at headline numbers.

**Workspace upgrades**

- **Pinned docks across all workspaces.**  New
  `View > Workspace > Pin Docks Across All Workspaces…` dialog lets
  you mark docks that should be visible on every tab — handy for the
  Element Table or System Data dock you always want at hand.
  Pinning state persists in `QSettings`.

- **Workspace export/import.**  `View > Workspace > Export
  Workspaces to File…` writes the full workspace set (titles, dock
  membership, saved geometry, pinned set) as a JSON `.workspace`
  file.  Import restores everything in one go — share custom layouts
  with collaborators.

- **`defaults_revision` migration.**  Saved-blob migration appends
  any new default workspace to existing user setups, so previously-
  shipped users automatically get the new `Optimize` tab from 3.2.11
  the next time they launch.  Customizations are preserved.

- **Optimizer progress badge.**  While the optimizer runs, the
  *Optimize* tab title becomes `Optimize • iter N` and the status
  bar reports merit; both clear when finished.  Lets you stay on
  Analysis or Wave Optics while the optimizer runs.

**New docks**

- **`Welcome` (`ui/welcome_dock.py`).** Empty-state landing panel
  with quick-start buttons (Open Demo, Insert Singlet, Insert
  Achromat, Browse Library, Keyboard Shortcuts) and a recent-files
  list backed by `QSettings`.  Default in the Design workspace; auto-
  populates from your last 10 opens / saves.

- **`Python` (`ui/repl_dock.py`).** Embedded Python REPL with
  `model`, `np`, `plt`, `result`, `wave` pre-bound to the live
  system, latest geometric trace, and latest wave-optics result.
  Up/Down arrow history, expression-vs-statement detection (echoes
  values like the standard REPL), captured stdout/stderr.

**Element-table polish (`ui/element_table.py`)**

- **Right-click context menu** on element rows: Duplicate, Delete,
  Move Up/Down, Toggle Distance Variable.  Endpoint rows (Source /
  Detector) and Source-distance variable are correctly disabled.

- **Variable highlighting.** The `Elem#` cell turns amber when the
  element has any optimization variable on it; the `Distance` cell
  turns amber when *distance* itself is a variable.  Quick visual
  for which surfaces the optimizer is allowed to touch.

- **Search box** in the toolbar (`Search elements…`).  Hides rows
  whose Name doesn't contain the substring (case-insensitive).

**Other**

- **F11 / View > Compact Mode.**  Hides the menu bar and replaces
  every dock's title bar with an empty widget — maximises working
  area for laptop screens.  F11 toggles back; the workspace tab bar
  stays visible the whole time.

- **`closeEvent` saves recent files** alongside workspace state.
  `File > New` now resets path + dirty so the title goes back to
  plain "Optical Designer".

> Saved layouts from 3.2.10 / 3.2.11 are auto-migrated: missing
> default workspaces (Welcome, Optimize, etc.) are appended without
> overwriting your customizations.  If you want a clean slate use
> `View > Workspace > Reset Workspaces to Defaults`.

## [3.2.11] — 2026-04-24

### UI — default workspace tweaks: Optimize tab + leaner Design / Wave Optics

Refined the default tab membership in response to a still-too-crowded
Design tab.  Core library and workspace machinery unchanged.

- **New `Optimize` tab** between Design and Analysis.  Holds Optimizer,
  Sliders, Multi-Config, Snapshots, and 2D Layout + System Data for
  context.  These docks were pulled out of the Design tab so the
  optimization workflow lives in its own focused space.

- **Leaner `Design` tab.** Now just 2D Layout, 3D Layout, System Data,
  and Library — the four docks you actually look at while *building*
  the optical layout.  Optimizer/Sliders/Multi-Config/Snapshots moved
  to Optimize.

- **Jones Pupil dropped from `Wave Optics` defaults.**  Still
  available via View > Jones Pupil or by adding it through Manage
  Docks; just no longer shown by default since it is a specialized
  polarization tool not needed for most wave-optics work.

> Existing users with a saved layout from 3.2.10 will continue to
> see their old defaults until they pick *View > Workspace > Reset
> Workspaces to Defaults*.

## [3.2.10] — 2026-04-24

### UI — top-of-window workspace tabs grouping docks by topic

Reduced GUI clutter by introducing a tabbed-workspace system at the top
of the main window.  Each tab shows only the docks relevant to that
phase of design work; the user can create and customize their own.
Core library unchanged.

- **`ui/workspace.py`** — new module with:
  - `Workspace` — named layout (title, dock_names list, saved
    `QMainWindow.saveState()` blob).
  - `WorkspaceBar` — top-of-window QToolBar containing a `QTabBar`
    plus a `＋` button.  Right-click any tab for Manage Docks /
    Rename / Duplicate / Delete; double-click to rename.
  - `ManageWorkspaceDialog` — checkbox list of every dock, with All /
    None bulk toggles, for picking which docks belong to a workspace.
  - `WorkspaceManager` — owns the workspace list and the "current"
    index; `apply_index(i)` switches tabs by hiding non-member docks
    and restoring the per-tab `restoreState()` blob; tracks user-
    initiated dock visibility changes (close button + View menu) and
    updates the active workspace's dock_names so the membership
    sticks; serializes to JSON for `QSettings` persistence.

- **Default workspaces** (loaded on first run, restored thereafter):
  - **Design** — 2D/3D Layout, System Data, Library, Multi-Config,
    Optimizer, Sliders, Snapshots.
  - **Analysis** — 2D Layout, Spot, Ray Fan / OPD, Footprint,
    Distortion, Spot vs Field, Through-focus, PSF/MTF, Field
    Browser, System Data.
  - **Wave Optics** — 2D Layout, Wave Optics, Zernike,
    Interferometry, Phase Retrieval, Jones Pupil, Ghost.
  - **Tolerancing** — 2D Layout, Tolerance, Sensitivity, System Data.
  - **Materials** — Materials, Glass Map, Library, System Data.

- **`ui/main_window.py`** — wired the workspace system into the shell:
  - Added `_build_workspace_bar()` that places the tab strip in the
    top toolbar area with `addToolBarBreak` underneath, so the main
    toolbar lands on the row below the tabs.
  - Added `_init_workspaces()` that builds the dock registry from
    `findChildren(QDockWidget)`, restores from `QSettings` if
    available (else loads defaults), wires every dock's
    `toggleViewAction().toggled` so user toggles update the current
    workspace's dock list, and applies the active workspace.
  - Added handlers for add / rename / duplicate / delete / manage,
    plus a `View > Workspace` submenu (with Reset to Defaults).
  - Added `closeEvent` to flush the current layout and persist all
    workspaces to `QSettings('lumenairy', 'OpticalDesigner')` so
    custom workspaces survive restart.
  - Per-tab dock geometry preserved: `save_current_layout()` snapshots
    `saveState()` into the outgoing workspace before each switch, so
    drags / resizes within a workspace are not clobbered.

### Why

The main window was getting too crowded — at 3.2.9 we had 27 dock
widgets stacked into 3 dock-area tab groups.  Tabbed workspaces let
the user focus on one phase at a time (designing the layout, then
analyzing it, then doing wave optics) without losing access to any
dock — and they can build their own analysis tabs ("MTF only",
"Distortion only", etc.) for whatever they want to plot.

## [3.2.9] — 2026-04-24

### UI — three new analysis docks + command palette + system-data additions

Filled out the comparison vs optiland's GUI feature set.  Core
library unchanged.

- **`ui/footprint_dock.py`** — per-surface ray-bundle outline.  For
  every surface in the system, plots `(x, y)` of the alive rays from
  a `make_rings(rings, per_ring)` launch with the clear-aperture
  circle drawn as a reference.  Multi-field overlay (configurable).
  Standard tool for verifying surface diameters, stop placement, and
  vignetting at every interface, not just the image plane.

- **`ui/distortion_dock.py`** — chief-ray distortion vs field +
  distortion grid.  Sweeps field angles 0..max, traces the chief
  ray, plots `100·(h_chief - f·tan θ) / (f·tan θ)` vs field, and
  also draws a reference paraxial grid (red) overlaid on the actual
  image-plane chief-ray positions (blue).  Status line reports max
  distortion + Pincushion/Barrel tag.

- **`ui/spot_field_dock.py`** — N×M array of spot diagrams across
  the configured `model.field_angles_deg`, on a shared scale so
  cross-field aberration growth is visible at a glance.  Optional
  Airy-disc overlay; per-panel RMS in titles; configurable
  rings / per-ring.

- **`ui/command_palette.py` (Ctrl+K / Ctrl+Shift+P)** — VS-Code-style
  fuzzy-search dialog over every menu action.  Indexes the live
  `QMenuBar` at popup time so any menu action labelled "Foo > Bar
  > Baz" is reachable by typing "fbb" (or "baz", or "f bar").
  Character-subsequence fuzzy match with word-boundary boost and
  prefix-match priority.  Up/Down navigation, Enter to fire, Esc
  to dismiss.  Hooked into MainWindow via `install_command_palette`
  in `__init__`; also added under `Help > Command Palette...` for
  discoverability.

### UI — extended SystemSummary

- `SystemSummaryWidget` now also reports:
  - Multiple wavelengths (when more than one is configured)
  - Configured field angles + total FOV
  - Working f/# (= image-space f/# for object at infinity)
  - Image-space NA
  - Airy disc radius
  - Front and rear principal planes (Welford convention)
  - Stop surface index (via `find_stop`)
  - Paraxial entrance/exit pupil positions and radii (via
    `compute_pupils`, when defined for the system)

### Notes

- 33 UI modules import cleanly (was 29 in 3.2.8; +3 new docks +
  command_palette).
- Footprint, Distortion, and Spot-vs-Field docks all run end-to-end
  against the AC254-100-C demo doublet:
  - Footprint draws S1 / S2 / S3 / Image with vignetting visible.
  - Distortion reports 0.0277 % Barrel at ±5° on AC254-100-C.
  - Spot-vs-Field shows 0°/1°/2° panels with RMS 41.5 / 39.0 /
    49.0 µm and the 5.29 µm Airy disc overlay.
- Command palette fuzzy match verified for "psf", "ghost",
  "arcoat", "field" against the indexed menu actions.
- Tier 3.1 (3D rays in Layout3DView) was already implemented by the
  existing `_draw_rays` method (line 235 of `layout_3d.py`); no
  duplication added.
- Core regression 16/16 (251 assertions) still passes on both
  libraries.

## [3.2.8] — 2026-04-24

### UI — Lens-function options dialog

- **New top-level `&Options` menu** (between Preferences and Help)
  with a `Lens function options...` entry that opens a tabbed
  dialog for configuring kwargs of the three real-lens pipelines.

- **New `ui/lens_options_dialog.py`** (`LensOptionsDialog`).
  QTabWidget with one tab per function — `apply_real_lens` (7
  kwargs), `apply_real_lens_traced` (11 kwargs),
  `apply_real_lens_maslov` (9 kwargs).  Widgets are built
  procedurally from a single `LENS_KWARG_REGISTRY` mapping; adding
  a new kwarg to a function is a one-line registry entry.  Each
  field has a tooltip explaining what it does.

- **Widget kinds**: bool → QCheckBox, int → QSpinBox (with min/
  max/step), float → QDoubleSpinBox, enum → QComboBox.  "Reset
  this tab" and "Reset all to defaults" buttons clear overrides.

- **Default filtering**: only kwargs whose value differs from the
  library default are persisted on `model.lens_options`.  Keeps
  the stored state minimal and means an omitted kwarg gets the
  current library default automatically — useful when the library
  changes a default and the user doesn't want to track it.

- **`SystemModel.lens_options`** — new dict-of-dicts attribute
  (`{function_name: {kwarg: value}}`) holding the user's overrides.
  Initialised empty in `__init__`.

- **WaveOpticsDock integration** — when delegating to a real-lens
  function, the worker reads `model.lens_options[func_name]` and
  splats it onto the call.  Dock-level controls (`ray_subsample`,
  `tilt_aware_rays`) remain authoritative; the dialog only
  contributes a value when the dock didn't already.

### Notable kwargs newly exposed

  apply_real_lens:
    bandlimit, fresnel, absorption, slant_correction,
    seidel_correction, seidel_poly_order, wave_propagator

  apply_real_lens_traced:
    bandlimit, ray_subsample, preserve_input_phase,
    tilt_aware_rays, fast_analytic_phase, parallel_amp,
    inversion_method, newton_fit, newton_poly_order,
    on_undersample, wave_propagator

  apply_real_lens_maslov:
    integration_method, poly_order, n_v2, ray_field_samples,
    ray_pupil_samples, extract_linear_phase, collimated_input,
    output_subsample, normalize_output

### Verification

- 29 UI modules import cleanly (was 28; +1 new dialog file).
- Dialog instantiates with 3 tabs, 27 kwargs total.
- Default-filtering verified: only changed values persist.
- Reset-this-tab and Reset-all-to-defaults rebuild widgets in place.
- End-to-end: enabling `fresnel=True` via the dialog produces a
  measured ~8% power loss through a BK7 singlet at 1.31 µm —
  confirming the kwarg flows correctly from dialog → model →
  WaveOpticsDock → `apply_real_lens`.
- Core regression 16/16 (251 assertions) on both libraries.

## [3.2.7] — 2026-04-24

### UI — hardware-self-calibrating forecast model

- The Wave-Optics dock's `forecast_resources` previously hardcoded
  a single "12 ms ASM at N=1024" reference, which over-predicted on
  fast workstations and dramatically under-predicted on laptops or
  slow sandboxes (16× under on a 192-ms-ASM box).  It now
  **self-calibrates against the local CPU** on first use:

  * **`_local_asm_baseline_ms()`** — runs one warmup + two timed
    `angular_spectrum_propagate` calls at N=512 (~50-300 ms total),
    extrapolates to the N=1024 reference via the standard
    ``N² · log N`` cost model, and caches the result for the rest
    of the session.

  * Every CPU-bound coefficient in the time model (Newton per-pixel,
    setup costs) is now multiplied by a `hw_scale = local / 12`
    factor, so the entire forecast scales **linearly** with the
    measured baseline.  A 4-ms-ASM workstation gets ~3× shorter
    forecasts than the 12-ms reference; a 100-ms laptop gets ~8×
    longer.  Validated: forecast for a 3-surface doublet sub=8 at
    N=1024 reads 78 ms / 234 ms / 1.9 s for ASM-1024 baselines of
    4 / 12 / 100 ms respectively — perfectly proportional.

  * **Recalibrate button** added to the Wave-Optics dock just above
    the forecast strip.  Shows the current baseline ("ASM-1024 =
    14.2 ms (self-measured)") and force-re-measures on click.
    Useful after switching FFT backend (NumPy → SciPy → pyFFTW →
    CuPy) or after moving the process to a different machine via
    hibernate / VM migration.  Disables the button while the (sub-
    300 ms) measurement runs so a double-click can't kick off two
    timed propagations at once.

  * **Fallback**: if `lumenairy.propagation` can't be
    imported (broken CuPy install, etc.), the calibration falls
    back silently to the historical 12 ms reference rather than
    crash the dock.

### Notes

- Pure additive change to the UI subpackage.  Core library is
  unchanged from 3.2.6 (Maslov `_ne` fix + recalibrated multipliers).
- Auto-calibration cost is one-time per process: ~50-300 ms on the
  first `_update_forecast` call.  Subsequent forecasts hit the
  cache in microseconds.
- All 28 UI modules still import cleanly; core regression
  16/16 (251 assertions) passes on both libraries.

## [3.2.6] — 2026-04-24

### Fixed — core library

- **`apply_real_lens_maslov` — `NameError: name 'ne' is not defined`**
  in the 3.2.2 Maslov→lenses merge.  The Maslov section uses
  ``numexpr`` which the rest of ``lenses.py`` imports as ``_ne``
  (with an underscore); the merged code's bare ``ne.evaluate(...)``
  references were never rewritten.  Blocked every Maslov call from
  running at all.  Renamed the four ``ne.evaluate`` sites inside
  ``_integrate_quadrature`` to ``_ne.evaluate``.  Regression caught
  by the benchmark sweep (below).

### UI

- **``forecast_resources`` recalibrated** for the perf work that has
  landed across 3.1.3–3.2.2: numexpr-fused phase screens,
  pre-resolved glass indices, polynomial-Newton default, parallel
  amp+amp(pw) pass, amplitude-masked Newton.  New ratios to ASM
  (benchmarked against an AC254-100-C doublet at N=1024, April 2026):

      ASM                  1.0  (reference)
      Fresnel              0.8  (was 1.3 — Fresnel is actually faster
                                 than ASM, no bandlimit kernel)
      Fraunhofer           0.6
      Rayleigh-Sommerfeld  3.3  (was 2.8)
      SAS                  5.0  (new; 3 FFTs at 2N-padded grid)

  Added `real_lens_maslov` branch (~600× ASM on defaults; dominated
  by 2-D quadrature integration).  Replaced the stale "6 FFTs per
  surface" model for `apply_real_lens` with the physically-correct
  ``(n_surfaces - 1)`` ASM-through-glass calls plus a small
  phase-screen overhead.  Replaced the pre-3.1.7 spline-Newton
  constant (``0.8e-6`` s/px) with a polynomial-Newton constant
  (``6e-6`` s/px); net effect is that traced-sub=8 (the new default)
  forecasts at ~230 ms vs ~980 ms previously, matching the ~261 ms
  actual.  Forecasts now within ~2× of measured wall-clock time
  across all code paths (was 10-20× overestimate on some paths).

- **Fixed I/O forecast bug**: a ``max(n_save_planes, 1)`` clamp was
  adding ~800 ms of phantom disk-save time even when the caller
  specified zero planes.  Changed to gated ``if n_save_planes > 0``
  so forecasts for no-save runs are now correct.

### Notes

- Core regression (16 test files, 251 assertions) still passes on
  both libraries after the Maslov fix.
- UI smoke test: 28 UI modules import cleanly; `WaveOpticsDock`
  instantiates with 4 lens-model options (including Maslov) and 5
  propagator options (including SAS).

## [3.2.5] — 2026-04-24

### UI — Tier 1 feature additions

Core library unchanged.  UI-subpackage changes only (UI-variant library).

- **Ghost-analysis dock** (new ``ui/ghost_dock.py``) — one-click
  enumeration of all ordered surface pairs with bare-Fresnel ghost
  intensities ``R_i * R_j``, rendered as a sortable table.  Registered
  in the Bottom dock area and wired to the View menu and
  Analysis > "Ghost analysis".
- **Jones-pupil visualization dock** (new ``ui/jones_pupil_dock.py``)
  — probes the current lens at configurable N/dx with pure-x and
  pure-y plane-wave inputs, renders the canonical 2x4 Jones pupil
  (amplitude + phase for Jxx/Jxy/Jyx/Jyy).  Registered in the Right
  dock area with a View-menu toggle and Analysis > "Jones pupil"
  shortcut.  Verified: scalar demo doublet gives an exactly diagonal
  pupil (``|Jxy| = |Jyx| = 0``).
- **Codegen "Export Python Sim Script..."** menu item (File menu,
  already wired in the codebase) confirmed functional end-to-end
  after fixing the ``SystemModel.to_prescription`` bug described
  below — one-click reproducibility for the current system.

### Fixed

- **``SystemModel.to_prescription`` indentation bug** in
  ``ui/model.py``.  A pair of module-level helper functions
  (``_nice_dx``, ``_next_nice_N``) had been dedented out of the
  ``SystemModel`` class body, which silently pulled
  ``to_prescription`` out of the class as well (Python parsed it
  as dead code inside ``_next_nice_N``).  ``MainWindow``'s File >
  Export menu items (Zemax ZMX / CODE V SEQ / Python Sim Script,
  all of which call ``self.model.to_prescription()``) would have
  failed at runtime.  Moved ``to_prescription`` above the
  module-level helpers so it's a genuine method again.

### Verification

- 28 UI modules still import cleanly (was 26 pre-3.2.5; up 2 from
  the new ghost + Jones-pupil docks).
- 223 imported symbols resolve (was 216).
- ``MainWindow`` builds all docks without segfault in the offscreen
  headless harness (except the VTK 3D renderer path, which is an
  OpenGL/platform issue unrelated to this release).
- Both new docks run end-to-end against the ``AC254-100-C`` demo:
  Ghost dock produces 3 rows; Jones pupil dock returns the
  expected diagonal ``|Jxx| = |Jyy| = 1.32``, ``|Jxy| = 0``.
- Full core validation suite still 16/16 (251 assertions) on both
  libraries.

## [3.2.4] — 2026-04-24

### UI — compatibility audit + new-feature exposure

Core library unchanged in this release; all changes are in the
`lumenairy.ui` subpackage of the UI-variant library.

- **Fixed broken import** in `ui/materials_dock.py`: was importing
  `lumenairy.ui.glassmap_dock` (no underscore) but the
  actual module is `glass_map_dock`.  The tab loaded silently
  via the try/except fallback; glass-map tab now appears again.

- **SAS propagator exposed in `waveoptics_dock.py`** — added `'SAS'`
  to the "free-space propagator between elements" dropdown with an
  auto-resample-back-to-dx dispatch mirroring the existing Fresnel
  handler.  Covered in both the between-surfaces loop and the
  propagate-to-focus block.  Forecast-resources time model updated
  with a conservative 2× multiplier (SAS = 3 FFTs + resample).

- **Maslov lens model exposed in `waveoptics_dock.py`** — added
  `'apply_real_lens_maslov (phase-space, caustic-safe)'` to the
  lens-model dropdown alongside the existing ASM / apply_real_lens
  / apply_real_lens_traced options.  Dispatch routes through the
  lens-router branch; tooltip explains the phase-space /
  stationary-phase rationale.

- **CODE V `.seq` file I/O exposed in `main_window.py`** —
  added `.seq` to the File > Open dialog's filter (sibling of
  `.zmx` / `.txt`), the File > Export Prescription dialog
  (sibling of `.json` / `.zmx`), and the CLI file-load path.
  Dispatches to :func:`export_codev_seq` / :func:`load_codev_seq`.

### Verification

- All 26 UI modules import cleanly.
- 216 imported symbols resolve (was 208 before the UI update
  added 8 new imports for the three features above).
- `waveoptics_dock.py`, `main_window.py`, `materials_dock.py`
  parse cleanly with AST.
- Full core validation suite (16 files, 251 assertions) still
  passes on both libraries.

## [3.2.3] — 2026-04-24

### Added

- **``wave_propagator='fresnel'`` and ``wave_propagator='rayleigh_sommerfeld'``**
  options for :func:`apply_real_lens` (and threaded through
  :func:`apply_real_lens_traced`).  Rounds out the through-glass
  propagator switch to all four physically-sensible choices:
  ``'asm'`` (default), ``'sas'``, ``'fresnel'``,
  ``'rayleigh_sommerfeld'``.  Each follows the same resample-back-to-dx
  pattern (Fresnel, SAS) or preserves the input pitch natively (ASM,
  R-S).  RS was verified to match ASM to ~1e-13 at mm-scale
  through-glass distances as expected.  Unknown values now raise
  ``ValueError`` with a list of supported options.
  4 new assertions in ``test_lenses.py``.

## [3.2.2] — 2026-04-24

### Changed — Maslov propagator merged into lenses module

- The former top-level ``lens_maslov.py`` has been deleted.  Its sole
  public function ``apply_real_lens_maslov`` now lives in
  :mod:`lumenairy.lenses` alongside ``apply_real_lens`` and
  ``apply_real_lens_traced``.  This matches the fact that it is a
  third real-lens wave-optics pipeline (phase-space / Maslov), not a
  separate subsystem.
- Public API unchanged: ``lumenairy.apply_real_lens_maslov``
  and ``from lumenairy.lenses import apply_real_lens_maslov``
  both work.  The legacy path
  ``from lumenairy.lens_maslov import apply_real_lens_maslov``
  is the only thing that breaks; nothing in the library or its
  validation suite was using it.
- All 251 validation assertions pass; 272 public symbols still
  resolve through the connectivity audit.

## [3.2.1] — 2026-04-24

### Added — SAS integration hooks

Three places where the library's built-in propagation path was
hard-wired to ASM now accept SAS as a first-class alternative.

- **`propagate_through_system(method='sas')`**.  On `'propagate'`
  elements, setting `method='sas'` (globally or per-element) routes
  through :func:`scalable_angular_spectrum_propagate` instead of ASM.
  The pipeline auto-resamples the SAS output back to the original
  `dx` between elements so downstream lenses / apertures keep their
  physical coordinates.  Extra per-element keys: ``pad`` (default 2),
  ``skip_final_phase`` (default False).

- **`apply_real_lens(wave_propagator='sas')`** (and forwarded through
  ``apply_real_lens_traced``).  Swaps the through-glass
  ``angular_spectrum_propagate`` call for
  ``scalable_angular_spectrum_propagate`` + resample-back-to-grid.
  Physically ASM remains the appropriate choice inside a lens (glass
  thicknesses are mm-scale, high-Fresnel-number); this switch is
  exposed for research and for pipelines that want a single
  propagator used consistently throughout.

- **`JonesField.sas_propagate(z, wavelength, pad=2, skip_final_phase=False)`**.
  Polarization-aware SAS wrapper that applies the scalar
  :func:`scalable_angular_spectrum_propagate` to ``Ex`` and ``Ey``
  independently (both on the same grid and with the same kernel),
  then updates ``self.dx`` / ``self.dy`` to the new output pitch.
  Requires ``dx == dy`` (square grid); raises ``ValueError`` otherwise.

### Notes

- Additive change; no existing behaviour modified.  All 246
  previously-passing assertions still pass.  New total: 251
  assertions across 16 test files.

## [3.2.0] — 2026-04-24

### Added

- **Scalable Angular Spectrum propagator**
  (``scalable_angular_spectrum_propagate`` in ``propagation.py``).
  Implements the Heintzmann-Loetgering-Wechsler 2023 three-FFT
  kernel: ASM-minus-Fresnel precompensation phase + chirp + FFT.
  Output pitch is ``lambda*z/(pad*N*dx)`` — larger than input at
  long ``z``, avoiding the impractical-N problem of plain ASM.
  Includes paper's closed-form ``z_limit`` check, Fresnel-style
  physical-amplitude prefactor (so power is conserved; the
  reference notebook is amplitude-agnostic), ``skip_final_phase``
  toggle, ``pad`` factor, ``use_gpu`` path, ``verbose`` diagnostics.
  Validated against ``fresnel_propagate`` / ``fraunhofer_propagate``
  in the respective limits.  5 new assertions in
  ``test_propagation.py``.

- **CODE V ``.seq`` import/export**
  (``export_codev_seq`` + ``load_codev_seq`` in
  ``prescriptions.py``).  Round-trips the library prescription dict
  through the canonical CODE V sequence syntax.  Units M/MM/IN.
  4 new assertions in ``test_io.py``.

- **BSDF surface scatter model** (new module ``bsdf.py``) with
  ``LambertianBSDF``, ``GaussianBSDF``, ``HarveyShackBSDF``.
  Common interface: ``evaluate``, ``sample``,
  ``total_integrated_scatter``.  Attached to ``Surface`` via new
  ``bsdf`` field.  Helper ``sample_scatter_rays`` spawns a
  ``RayBundle`` of scattered rays for Monte Carlo stray-light
  propagation.  8 new assertions in ``test_features.py``.

- **Jones pupil spatial-map visualization**
  (``plot_jones_pupil`` + ``compute_jones_pupil`` in
  ``plotting.py``).  ``compute_jones_pupil`` probes a system with
  orthogonal x/y plane-wave inputs and returns the full
  ``(Ny, Nx, 2, 2)`` Jones matrix.  ``plot_jones_pupil`` produces
  the canonical 2x4 grid (amplitude + phase for each matrix
  element) with phase masked below an amplitude threshold.
  4 new assertions in ``test_polarization.py``.

### Notes

- No breaking changes.  All 225 previously-passing assertions
  still pass.  New total: 246 assertions across 16 test files.
- ``Surface`` dataclass gains a new optional ``bsdf`` field.
  ``_surface_copy_with`` propagates it through edits.  Older
  pickled bundles/prescriptions without this field are handled
  transparently via ``getattr(..., None)``.

## [3.1.11] — 2026-04-24

### Added

- **Stop-aware `seidel_coefficients`**.  Added ``stop_index`` and
  ``field_angle`` kwargs.  When the declared stop is not at surface
  0, the chief ray's initial conditions at surface 0 are now derived
  from the pre-stop ABCD so that ``y_chief = 0`` at the stop by
  construction.  Backward-compat: the default behaviour uses
  :func:`find_stop` (which falls back to surface 0 when no surface
  is flagged ``is_stop=True``), matching the pre-3.1.11 assumption
  bit-for-bit.  Output dict now also contains ``'stop_index'`` for
  diagnostics.  ``seidel_prescription`` passes the new kwargs
  through.

  Hopkins/Welford convention + ``H^2``-factored-out sums are
  preserved; chief-ray-dependent coefficients (S2 coma, S3
  astigmatism, S5 distortion) reflect the new stop position
  correctly, while S4 (Petzval, curvature-only) remains invariant
  as expected.

- **`validation/test_validation_lens.py`** — known-answer
  regression test harness covering the major library APIs:
  lensmaker's formula, manual ABCD composition, ``find_lenses``
  auto-detection, stop-aware Seidel invariants, ``compute_pupils``,
  ``refocus`` vs full retrace, ``through_focus_rms``,
  ``apply_real_lens`` vs ``apply_real_lens_traced`` vs
  ``apply_real_lens_maslov``, and per-ray error codes.  Runs
  standalone; exits 0 on all-pass.

### Fixed

- **`refocus` now projects to the requested image plane** instead
  of advancing rays by ``delta_z`` from their current (post-
  refraction) position, which was at ``z = sag`` of the last
  surface rather than at the vertex plane.  The old semantics
  caused off-axis rays to land short of the intended image plane
  by up to ``sag * (1 - 1/cos_angle)`` (~100 um at h=8 mm on an
  F/5 singlet).  New behaviour uses ``(delta_z - z_current)`` in
  the arc-length computation so rays land exactly on ``z =
  delta_z`` in the last surface's frame -- bit-identical to what
  ``trace`` would produce with a flat image plane appended at
  ``thickness=delta_z``.  ``through_focus_rms`` inherits the fix
  automatically.

## [3.1.10] — 2026-04-24

### Added

- **`apply_real_lens(..., use_gpu=False)`** — new kwarg making the
  whole phase-screen + in-glass ASM pipeline array-API polymorphic.
  Default remains ``False`` so CPU output is bit-for-bit backward
  compatible with 3.1.9 (verified: 0.0e+00 max difference on all
  tested inputs).  When ``use_gpu=True`` or ``E_in`` is already a
  CuPy array:

    * Every meshgrid, sag array, phase screen, and per-surface
      multiplication runs natively on device via ``cp.*`` operations.
    * Internal ``angular_spectrum_propagate`` auto-detects the CuPy
      backend (it already had a GPU path) and uses cuFFT for the
      in-glass ASM propagation between surfaces.
    * The numexpr-fused phase-screen path is automatically skipped
      on GPU (numexpr is CPU-only); CuPy's fused elementwise kernels
      handle the ``E * exp(-i k OPD)`` update.
    * The returned array is a CuPy array (not automatically pulled
      back to host) so downstream callers can keep the field on GPU
      for further propagation or masking.  Use ``cp.asnumpy()`` to
      pull back when needed.

- **`apply_real_lens_traced(..., amp_use_gpu=False)`** — new kwarg
  passing ``use_gpu`` through to the internal ``apply_real_lens``
  calls that build the amplitude envelope and the reference phase.
  Default ``False``: no behaviour change unless explicitly enabled.
  When ``amp_use_gpu=True``:

    * The ``amp`` and ``amp(pw)`` passes run on GPU (or just the
      ``amp`` pass when ``fast_analytic_phase=True``).
    * GPU results are pulled back to the host via ``cp.asnumpy()``
      at the end of the amp block, so the ray trace, Newton
      inversion, and final field assembly run unchanged on CPU.
      This lets the existing stable CPU pipeline drive the
      ray-trace side while the FFT-bound amp side offloads.
    * Independent of the Newton-inversion ``use_gpu`` kwarg added
      in 3.1.7: the two GPU flags can be enabled independently.

### Changed

- **`surface_sag_general` / `surface_sag_biconic`** made array-API
  polymorphic.  Detect CuPy vs NumPy from the input array and
  dispatch all internal ops (``zeros_like``, ``where``, ``sqrt``)
  accordingly.  Needed by the GPU ``apply_real_lens`` path; CPU
  callers see no change.

### Performance

- Not measured at production scale on this host (the local CuPy
  install is missing cuSOLVER and cuFFT DLLs, so the GPU path was
  validated for correctness via code-path inspection but couldn't
  be benchmarked end-to-end).  Expected speedup when a complete
  CUDA stack is available: ~5-10x on the amp + amp(pw) passes
  (they're ASM-FFT-bound and cuFFT is substantially faster than
  pyFFTW for large grids), dropping the wall-time contribution of
  those passes from ~50% of ``apply_real_lens_traced`` to ~10-20%.

### Known limitations

- The GPU path requires a complete CuPy install (cuFFT, cuBLAS;
  cuSOLVER only if you separately enable
  ``newton_fit='polynomial'`` with GPU).  Pass ``use_gpu=True``
  explicitly opts in; missing components raise ``ImportError`` at
  the first GPU call.

## [3.1.9] — 2026-04-24

### Added

- **`lens_abcd(lens, wavelength, start=None, end=None, label=None)`**
  — paraxial characterisation of a single lens element.  Accepts
  a prescription dict, a list of ``Surface`` (with optional
  ``start`` / ``end`` slice indices), or a single ``Surface``.
  Strips the trailing-thickness air gap so the returned ABCD is
  air-to-air at the element's own vertex, not "lens plus
  downstream propagation."  Returns a ``LensInfo`` dataclass with
  ``abcd``, ``efl``, ``bfl``, ``ffl``, ``principal_planes``
  (Welford convention), ``thickness``, and surface-index range.

- **`find_lenses(surfaces, wavelength)`** — auto-detect individual
  lens elements in a surface list by scanning air -> glass ->
  air transitions.  Cemented multi-element groups (glass -> glass
  interfaces in the middle) stay grouped.  Mirrors are reported
  as their own one-surface elements.  Returns
  ``List[LensInfo]``.

- **`compute_pupils(surfaces, wavelength, stop_index=None)`** —
  paraxial entrance / exit pupil positions and radii.  Images the
  aperture stop backward (for EP) and forward (for XP) through
  the pre- and post-stop sub-systems using their ABCD matrices;
  no ray tracing.  Returns a ``PupilInfo`` dataclass with
  ``ep_z``, ``ep_radius``, ``xp_z``, ``xp_radius``, and the
  resolved ``stop_index``.

- **Per-ray diagnostic `error_code` on `RayBundle`** — a ``uint8``
  array (1 byte / ray) recording the reason each dead ray was
  killed:

    * ``RAY_OK = 0``             — alive
    * ``RAY_TIR = 1``            — total internal reflection
    * ``RAY_APERTURE = 2``       — clipped by a surface semi-diameter
    * ``RAY_MISSED_SURFACE = 3`` — intersection Newton failed
    * ``RAY_NAN = 4``            — arithmetic produced NaN/Inf

  First-failure-wins: once a ray is killed with a non-zero code,
  subsequent surfaces do not overwrite the root cause.  The
  ``_refract`` and ``_intersect_surface`` helpers now set the
  appropriate code at kill time.  ``trace_summary`` prints the
  breakdown (``[TIR=N, aperture=M, miss=K, nan=L]``).  Bundles
  constructed before 3.1.9 (or without an explicit ``error_code``)
  are handled transparently: ``__post_init__`` synthesises the
  field from ``alive`` as a best-effort placeholder.

### Performance

- **Glass indices pre-resolved once per `trace()`** call via an
  up-front list comprehension, instead of two `get_glass_index`
  lookups per surface inside the hot loop.  Underlying
  ``refractiveindex.info`` cache was already avoiding the
  dispersion calculation; this removes the Python dispatch
  overhead too.  Small per-call win, more useful at high
  repeated-trace counts.

## [3.1.8] — 2026-04-24

### Added

- **`trace(..., output_filter='all' | 'last' | callable)`** — new kwarg
  controlling what per-surface state is retained in
  ``result.ray_history``.  Default ``'all'`` preserves existing
  behaviour.  ``'last'`` keeps only the final image bundle, eliding
  every intermediate ``RayBundle.copy()``.  On large ray counts
  this is a significant memory win:

  - N=32768 `apply_real_lens_traced` call at `ray_subsample=8`:
    ~4M coarse rays × 7 float64 arrays + 1 bool = ~228 MB per
    surface copy.  A 6-surface doublet therefore saves ~1.4 GB
    per call.
  - At larger grids / finer ray sub-sampling the savings scale
    linearly with ray count.

  `apply_real_lens_traced` now calls `trace(..., output_filter='last')`
  internally since it only consumes the image bundle; existing
  callers of `trace` see no change unless they opt in.

- **`refocus(result, delta_z, wavelength=None)`** — closed-form
  image-space transfer of a traced ray bundle.  Advances every
  ray by ``delta_z`` along its direction cosines; updates
  positions and OPL with the correct image-space refractive
  index (reads ``surfaces[-1].glass_after``, not assumed
  ``n=1``).  Signed ``delta_z`` (negative = move toward the
  lens, pre-focus) — the OPL update is ``n * delta_z / N``
  without an absolute-value, so defocus in both directions is
  handled consistently.

- **`through_focus_rms` rewritten on top of `refocus`**: single
  base trace through the real surfaces followed by a
  closed-form transfer at each focus shift, instead of
  rebuilding the surface list and re-tracing from surface 0.
  Expected speedup scales with the number of surfaces (~5-20x
  typical); numerical output matches the pre-3.1.8 full-retrace
  path to RMS spot-size precision (verified on a doublet:
  identical best-focus location, ~1e-4 max RMS difference which
  is finite-ring-sampling noise).

- **`is_stop: bool = False`** field on `Surface`.  Explicitly
  marks the aperture stop when set by a loader or caller.
  Zemax `.zmx` / `.txt` loaders propagate the STOP keyword onto
  the surface via the per-surface ``'is_stop'`` key.

- **`find_stop(surfaces)`** — locate the aperture-stop surface
  index.  Dispatch: first surface with ``is_stop=True`` (warns
  on multiple flags); else first surface with a finite
  ``semi_diameter``; else surface 0 (with a ``UserWarning`` if
  multi-surface).  Foundation for future stop-aware Seidel,
  pupil, and chief-ray aim work.

### Fixed

- **Dead-code line in `_paraxial_trace`** — removed the
  ``if False else u`` clause on the refraction update that was a
  commented-out alternative form of the paraxial refraction
  equation; the active preceding line already implements the
  correct ``u <- u - y * phi / n2`` update.  No observable
  behaviour change (the function was mathematically correct), but
  the expression now states its intent.

- **Seidel ``S4`` (Petzval) double-assignment** in
  ``seidel_coefficients`` — the first of two adjacent
  assignments used ``-c * (n2 - n1) * (1/n2 - 1/n1)``, which
  squares the index difference and is dimensionally wrong; the
  second, which won, used the correct ``-(n2-n1)/(n1*n2) * c``
  form (standard Hopkins/Welford convention with ``H^2``
  factored out).  The errant line is removed.  Output
  ``total['S4']`` is unchanged (the correct line was already
  winning); the cleanup clarifies intent.

### Performance

- `apply_real_lens_traced` at N=32768: ~1.4 GB of transient
  ``RayBundle.copy()`` allocations eliminated per call via
  `output_filter='last'`.  No measurable wall-time regression
  at small N; larger-grid runs should see moderate gains from
  reduced memory pressure and allocator churn.

## [3.1.7] — 2026-04-23

### Added

- **`apply_real_lens_maslov`** (new public function in `lens_maslov.py`).
  Phase-space / Maslov propagator complementing `apply_real_lens` and
  `apply_real_lens_traced`.  Fits a 4-variable Chebyshev tensor-product
  polynomial to the ray-traced back-map `s1(s2, v2)` and `OPD(s2, v2)`,
  then evaluates the Maslov integral by one of three methods selected
  by `integration_method`:

    * `'stationary_phase'` (recommended) — closed-form saddle-point
      evaluation per output pixel.  Caustic-safe by construction, no
      critical-sampling constraint, analytically differentiable w.r.t.
      the polynomial coefficients.  On Design 51 L1 at N=1024
      (collimated Gaussian input, `collimated_input=True`,
      `output_subsample=4`) matches `apply_real_lens_traced` to
      ~1.2 % RMS intensity in 2.2 s — faster than traced.

    * `'quadrature'` — uniform Tukey-windowed Riemann sum on the v2
      grid.  Correct for extended multi-source inputs inside the
      quadrature-validity bound `w_s >= D_s1 / n_v2`; not suitable for
      single-source collimated inputs where the integrand is
      delta-like in v2.

    * `'local_quadrature'` — Hessian-oriented uniform quadrature in a
      small window around the stationary point.  Captures asymptotic
      corrections beyond leading stationary phase.  Extended-source
      regime only.

  Four fast-path speedups are applied throughout:

    1. Precompute the `s2`-basis Chebyshev Vandermonde over the output
       grid once; per-`v2`-sample evaluations reduce to `G @ h`.
    2. Batched BLAS GEMM across chunks of `v2` samples (tunable
       `chunk_v2=64` default) replaces 7 × N_v2² matvec calls.
    3. Vectorised weight-vector assembly via fancy-indexing of the
       multi-index arrays, eliminating the Python loop over basis
       terms.
    4. `numexpr` fused integrand + reduction on the hot path, falling
       back to plain NumPy when `numexpr` isn't importable.

  Combined ~30× speedup over the naive per-sample Python loop.

  Supports explicit `collimated_input=True`, `poly_order`, `n_v2`,
  `ray_field_samples` / `ray_pupil_samples`, `output_subsample`,
  `normalize_output` (`'power'` default, preserving total |E|²),
  `extract_linear_phase` (for diffractive grating surfaces at nonzero
  orders), and per-method Newton-iteration controls.

- **`apply_real_lens_traced` speedup kwargs** (opt-in, default behaviour
  unchanged):

    * `fast_analytic_phase=False` — when `True`, skips the full
      `apply_real_lens(ones, ...)` ASM-through-glass pass used to
      extract the reference phase and computes it analytically from
      per-surface sag instead (`_geometric_lens_phase` helper).
      Preserves intensity to 0.000 % and introduces <10 nm OPL
      phase error on refractive systems up to ~F/7 (Design 51 L3/L4
      scale); below the numerical noise floor for most coherent-imaging
      workflows.  Saves ~25 % wall time when `parallel_amp=False`.

    * `_Cheb2DEvaluator` refactored as array-API polymorphic with
      a combined value+gradient evaluator.  The Newton loop in
      `apply_real_lens_traced` now detects the combined-evaluation
      API (`ev_value_and_grad`) and uses it when the polynomial
      path is active, dropping from 6 evaluator calls per iteration
      down to 2 (one per coordinate) with shared Chebyshev basis
      work.  In isolated benchmarks on a 4M-sample Newton-style
      workload the refactored polynomial path runs **~12.6x faster
      than the 6-call spline baseline** (189 ms vs 2376 ms) when
      numba is installed; roughly ~4x without numba.  Combined with
      `#3` (Clenshaw-style inline Chebyshev recurrence, no
      Vandermonde materialised), `#6` (one-pass value + both
      partial derivatives), and `#1` (optional `@njit(parallel,
      fastmath)` kernel).

    * **GPU support via ``use_gpu=True``** (requires
      ``newton_fit='polynomial'`` and a working CuPy install).
      Dispatches the ``_Cheb2DEvaluator`` construction and the Newton
      inversion loop to GPU while keeping amp, amp(pw), ray-trace,
      and final field assembly on CPU (those remain CPU-only for
      now).  The polynomial fit (tiny lstsq) is always done on CPU
      to avoid a cuSOLVER dependency; only fitted coefficients +
      index arrays are pushed to the device.  The Newton loop uses
      an ``xp``-namespace throughout so the same code runs on NumPy
      or CuPy; the numba fastpath is skipped when the backend is
      CuPy and the pure-xp path (which CuPy dispatches to cuBLAS
      and elementwise kernels) runs instead.

      Validated on a 33-knot singlet fit:

          N_samples    CPU (numba)    GPU (cupy)    GPU speedup
          100,000        1.4 ms         4.8 ms       0.3x
          1,000,000     22.1 ms        15.4 ms       1.4x
          4,000,000    105.8 ms        74.1 ms       1.4x

      Modest absolute speedup at typical workloads because numba
      already parallelises the CPU path aggressively; the bigger
      payoff is for iterated workflows (many Newton calls per
      optimisation step) or very large grids (N>>4k) where CPU
      memory bandwidth starts to saturate.  Output is bit-equivalent
      to the CPU path (0.0 % RMS intensity error on Design 51 L1).

      Process-pool Newton is auto-disabled when ``use_gpu=True``
      (device arrays don't cross subprocess boundaries cheaply) and
      when ``newton_fit='polynomial'`` (the worker function currently
      only supports splines).  Both fall back to the in-process
      Newton path, which at ``ray_subsample=8`` is already fast
      enough that the pool's spawn overhead often dominates anyway.

    * `newton_fit='polynomial'` (default in 3.1.7; was `'spline'` in
      earlier versions) or `'spline'` — polynomial path replaces
      `scipy.interpolate.RectBivariateSpline` with a 2-D Chebyshev
      tensor-product fit (`_Cheb2DEvaluator`) providing the same
      `.ev(x, y, dx=0/1, dy=0/1)` API used by the Newton loop.  For
      smooth refractive lens prescriptions (all Seidel and
      higher-order aberrations are polynomials by definition),
      polynomial matches or exceeds spline accuracy on the fit and
      provides closed-form analytic derivatives.  Tunable
      `newton_poly_order=6` default (order-6 total-degree captures
      higher-order aberrations out to the 8th Seidel).  Flip back to
      `'spline'` for high-order freeforms / metasurfaces / kinoforms
      with sharp non-polynomial surface features.

    * Default `ray_subsample` bumped from `1` to `8`.  At N=32768 on
      Design 51 lenses this gives 2000–3800 samples across the
      aperture — far above the `min_coarse_samples_per_aperture=32`
      safety floor (which gave ~85 nm RMS OPD error per the library's
      internal benchmark).  At 2000 samples the projected RMS phase
      error is ~0.02 nm (λ/60 000 at 1.31 µm).  Small-grid users who
      would drop below 32 samples across the aperture are protected
      by the existing `on_undersample='error'` guardrail, which now
      raises with a message telling them to reduce to a safe value
      (typically `ray_subsample=4` or lower).

### Performance

On the Design 51 L1 benchmark (N=32768, `parallel_amp=False`, projected):

| Config | L1 time | % of baseline |
|---|---|---|
| 3.1.6 default (`ray_subsample=4`) | ~1285 s | 100 % |
| 3.1.7 default (`ray_subsample=8`) | ~1015 s | 79 % |
| + `fast_analytic_phase=True` | ~685 s | 53 % |
| + `parallel_amp=True` | ~360 s | 28 % |

Accuracy unchanged vs 3.1.6 default, within the stated tolerances.

## [3.1.6] — 2026-04-21

### Fixed

- **Zarr storage on Windows + Python 3.14 + zarr v3.**  Writing to an
  existing zarr store failed with
  ``FileExistsError: [WinError 183] Cannot create a file when that
  file already exists`` whenever ``append_plane`` or
  ``write_sim_metadata`` reopened an already-created zarr directory.
  Root cause: zarr v3's ``LocalStore._open`` unconditionally calls
  ``Path.mkdir(parents=True, exist_ok=True)`` on the store root, and
  on this specific platform combination that call raises
  ``FileExistsError`` even with the ``exist_ok`` flag -- a
  regression relative to standard Python semantics on other OSes and
  earlier Python versions.

  Fixed by adding a ``_open_zarr_group_safe`` helper that picks the
  right open-mode for the situation:

  * Store directory already exists on disk -> open with ``mode='r+'``
    (read/write, must exist), which skips the internal mkdir call
    entirely and therefore doesn't trigger the regression.
  * Store directory doesn't exist yet -> open with ``mode='a'`` to
    create it, with a ``FileExistsError`` fallback to ``'r+'`` for
    concurrent-creation races.

  The helper is used internally by ``_zarr_append_plane`` and
  ``_zarr_write_sim_metadata``; no API change for callers.  All
  existing callers (``append_plane``, ``save_planes``,
  ``write_sim_metadata``, the unified dispatch shims) benefit
  automatically.

### Compatibility

- Read-path APIs (``load_planes``, ``list_planes``,
  ``load_plane_slice``, etc.) open with ``mode='r'``, which doesn't
  mkdir and was never affected by the bug.  No changes to those code
  paths.

## [3.1.5] — 2026-04-20

### Added

- **`load_zmx_prescription` and `load_zemax_prescription_txt` now
  return `prescription['object_distance']`.**  Zemax sequential files
  typically carry non-refractive "dummy" surfaces between the object
  (or STOP / source plane) and the first active lens surface —
  coordinate breaks, field-reference planes, MLA mounting planes,
  etc.  Previous loader versions filtered these out and discarded
  their `DISZ` (z-thickness) values, which meant any wave-optics
  simulation driven by the returned prescription implicitly placed
  its source field AT the first refractive surface, collapsing that
  design-intended obj-space geometry.  The symptom was a
  defocus-like blur at the downstream image plane proportional to
  the dropped distance.

  The new key `object_distance` (float, meters) is the sum of `DISZ`
  values from the STOP surface up to but not including the first
  active surface.  If the file has no STOP, the sum runs from SURF 0
  onward.  Non-finite `DISZ` (`INFINITY`) contributes 0 so
  object-at-infinity configurations behave the same as before.

  Downstream callers driving a simulation from the loaded
  prescription should now propagate their source field by
  `prescription['object_distance']` before invoking the first lens
  operator to recover the .zmx's original paraxial geometry.

  **Detection example (Design 51 tx4designstudy51.zmx):**

      rx = op.load_zmx_prescription('tx4designstudy51.zmx')
      print(rx['object_distance'])   # -> 0.096669 (m), = 96.67 mm

  of previously-dropped dummy-surface thickness, which (without
  this change) caused ~235 µm of defocus blur on each MLA-
  collimated beam at the metasurface plane.

### Compatibility

- The new key is additive; all existing callers continue to work
  unchanged.  Prescriptions built manually (`make_singlet`,
  `make_doublet`, `make_cylindrical`, etc.) don't set
  `object_distance` at all — callers reading this key should use
  `rx.get('object_distance', 0.0)` for safety.

## [3.1.4] — 2026-04-18

### Changed (default flip)

- **`apply_real_lens_traced(..., tilt_aware_rays=...)` default changed
  from ``True`` to ``False``.**  The Tier 1 input-aware ray launch
  added in 3.1.2 was meant to extract per-pixel ray directions from
  the input field's local phase gradient so the lens OPL would reflect
  actual angles of incidence.  In combination with the (also default)
  ``preserve_input_phase=True`` path, however, it creates a
  reference-frame inconsistency that does not affect single-mode
  plane-wave-like inputs but produces materially wrong output on
  multi-mode inputs (post-DOE diffraction patterns, compound
  superpositions).

  Specifically, the ``preserve_input_phase`` output is assembled as

        E_out = E_analytic * exp(i * delta_phase)
        delta_phase = k0 * opl_traced - phase_analytic_lens

  where ``phase_analytic_lens`` is extracted by running
  ``apply_real_lens`` on a unit plane wave -- a plane-wave reference.
  For ``delta_phase`` to be a clean "ray-traced minus analytic"
  correction, ``opl_traced`` must share that reference (rays launched
  collimated at the entrance).  With ``tilt_aware_rays=True``,
  ``opl_traced`` is instead evaluated at per-pixel launch angles; for
  multi-mode inputs those angles vary wildly across the pupil,
  ``delta_phase`` mixes lens-model correction with tilt-induced phase
  shifts the plane-wave reference does not contain, and the output
  field collapses to a "power-lost-to-bandlimit" state (TX Design 36
  rerun on 2026-04-17 showed 0.55 % power conservation on a 4-lens
  post-DOE system, vs 92.5 % with the plane-wave default).

  The 3.1.3 multi-mode Gaussian-smoothing of the extracted tilts was a
  mitigation for the pathological gradient aliasing but could not
  address the underlying reference-frame inconsistency -- with enough
  smoothing the tilts collapse toward zero anyway, and the legitimate
  per-order tilt structure of the post-DOE field is destroyed in the
  process.  Flipping the default to ``False`` side-steps the whole
  issue by using the reference-consistent plane-wave launch that
  pre-3.1.2 releases used.  Users with specifically small, uniform
  input tilts who want the per-ray OPL variation can still pass
  ``tilt_aware_rays=True`` explicitly and validate on their case.

  The 3.1.3 ``_sample_local_tilts`` Gaussian smoothing + ``max_sin``
  clip stay in the library -- they are still consulted when
  ``tilt_aware_rays=True`` is explicit or when the experimental
  ``inversion_method='backward_trace'`` needs an exit-direction
  estimate.  They just don't run on the default path anymore.

### Performance

- **Paraxial-magnification Newton initial guess in
  `apply_real_lens_traced`.**  The pre-3.1.4 initial guess was a
  hard-coded ``(xe, ye) = 1.10 * (Xw, Yw)`` (implicitly assuming
  every lens has a paraxial magnification of ~0.91 at its exit
  vertex).  For singlets this was approximately right; for compound
  systems with real imaging magnification (TX Design 36 full-system
  inversion has M = 0.25) it puts Newton 4x from the answer and costs
  several extra iterations per pixel.

  The new path measures the per-lens paraxial magnification directly
  from the already-computed forward-map slope at the central launch
  point (the central-finite-difference ``dx_out/dx_in`` and
  ``dy_out/dy_in``) and uses ``(Xw/M_x, Yw/M_y)`` as the initial
  guess.  Zero additional compute (the values are already in the
  forward-trace output array), and for singlets the result is
  essentially the same as before; for compound-system callers the
  speedup is several iterations.  Parallel-pool workers get the
  inverse-magnification factors through ``_spline_data`` so the
  in-process serial path and the out-of-process chunked path seed
  Newton identically.

- **`inversion_method='backward_trace'` opt-in on
  `apply_real_lens_traced`** (experimental).  Replaces the forward
  ray trace + Newton spline inversion with a single backward pass
  from a coarse subsample of the exit-plane wave grid, driven by
  the phase-gradient-extracted exit direction.  Validated to
  reproduce the Newton path's OPL to sub-picometre on a plano-convex
  singlet single-ray test; end-to-end ``apply_real_lens_traced`` at
  N=1024 shows ~30 nm OPD RMS agreement and ~3.2x speedup vs the
  Newton default.  Accuracy is bounded by the finite-difference +
  smoothed phase-gradient direction estimate, not by the reversal
  itself.  Default stays ``'newton'`` while the backward path is
  validated on a wider set of prescriptions; opt in by passing
  ``inversion_method='backward_trace'`` to trade some accuracy
  (~30 nm OPD per lens) for a substantial speedup on large grids.

## [3.1.3] — 2026-04-17

This release is a set of targeted performance, robustness, and
precision-flexibility improvements to the hot path for N=32768
coherent propagation runs.  All changes are backwards-compatible
defaults (bit-identical at complex128 with no new kwargs passed)
except where explicitly noted.

### Performance

- **Numexpr-fused phase-screen multiply in `apply_real_lens`.**  The
  per-surface `E * np.exp(-1j * k0 * opd)` step used to materialise
  three complex128 NxN intermediates (the broadcast `-1j*k0*opd`,
  the `exp()` output, and the multiply result -- ~50 GB of churn
  per surface at N=32768).  When `numexpr` is available (optional
  dependency `[perf]`) and the field has at least 2^20 elements, the
  library routes the expression through `ne.evaluate('E * exp(-1j*k0*opd)', out=E)`
  instead.  Threaded, chunked, fully in-place, and numerically
  identical to the numpy path at double precision (`max |diff| = 0`
  on a singlet test).  Measured 1.66x on `apply_real_lens` at N=4096.
  Automatic fallback to the numpy path when numexpr is not installed
  or when the field is too small for the overhead to pay off.
  New top-level export: `NUMEXPR_AVAILABLE`.

- **Decenter-aliased entrance grids in `apply_real_lens`.**  When a
  surface has `decenter == (0, 0)` (the common case), the library
  now aliases `Xs = X`, `Ys = Y`, `h_sq = h_sq_axis` instead of
  allocating three new float64 NxN arrays per surface (~24 GB at
  N=32768).  Safe because downstream code only reads these arrays
  and creates fresh arrays for `sag + tilt[i]*Xs` / `sag + form_err`.

- **Single-slot pyFFTW plan cache with per-plan threading.Lock in
  `_fft2` / `_ifft2`.**  Replaces the previous
  `pyfftw.interfaces.numpy_fft` shim, which allocated fresh 16 GB
  aligned buffers on every call and held them for 30 s -- the root
  cause of the Windows contiguous-address-space fragmentation that
  forced `USE_PYFFTW = False` as a workaround on N=32768 runs.  The
  new path holds exactly one `pyfftw.FFTW` plan per direction (forward,
  inverse), each backed by an in-place 16 GB aligned buffer that is
  allocated once at first use and reused for the lifetime of the
  process.  Shape/dtype/threads are keyed on the cache; any mismatch
  drops the old plan + buffer (GC'd) and reallocates.  A per-plan
  `threading.Lock` serialises concurrent execution from parallel
  callers (e.g. the `parallel_amp` path in `apply_real_lens_traced`).
  Measured 1.50x on an N=4096 ASM call.  `reset_fft_backend()` now
  clears the new plan slots in addition to the bad-shape blacklist.

- **Parallelised `amp` and `amp(pw)` passes in `apply_real_lens_traced`**
  via a 2-worker `ThreadPoolExecutor`.  The two internal
  `apply_real_lens` calls are data-independent and run concurrently;
  FFT execution is serialised through the per-plan lock above but
  non-FFT work (sag, phase screens, glass-interval setup) overlaps.
  Measured 1.56x on the combined amp step when isolated at N=4096.
  Opt-out via new kwarg `parallel_amp=False`.  Memory guard via
  `parallel_amp_min_free_gb=48.0` auto-disables when RAM is tight
  (doubled per-call transient working set: ~2x the peak of a single
  `apply_real_lens` at the same grid size).

- **Amplitude-masked Newton inversion in `apply_real_lens_traced`.**
  New kwargs `newton_amp_mask_rel=1e-4`, `newton_mask_dilate_coarse_px=2`.
  The Newton-inverted entrance→exit spline is now evaluated only on
  coarse-grid pixels where the analytic amplitude envelope exceeds
  `newton_amp_mask_rel * amp.max()`, with the mask dilated by
  `newton_mask_dilate_coarse_px` coarse pixels so bilinear
  interpolation near mask boundaries always has real data in its
  support.  Skipped pixels get NaN and are handled by the existing
  NaN-propagation path -- identical to the ray-domain-failure
  handling already in place.  Self-disables when the mask would
  capture >95 % (overhead not worth it) or <1 % (pathological; fall
  back to full-grid Newton).  Biggest benefit on post-DOE fields
  where only the diffraction-order pixels are bright.

### Added

- **Complex-dtype awareness throughout the critical path.**  The
  `apply_real_lens`, `apply_real_lens_traced`, `apply_mirror`, and
  `angular_spectrum_propagate` functions now preserve the caller's
  complex dtype (complex64 or complex128) rather than forcing
  complex128 internally.  Module-level default
  `DEFAULT_COMPLEX_DTYPE = np.complex128` controls the fallback
  when a non-complex input is given (used by a handful of builders
  elsewhere in the library).  Runners can opt into complex64 mode
  for ~2x memory and throughput by creating their fields as
  complex64 from the start; all library functions on the hot path
  then stay at that precision end-to-end.

  To keep complex64 accurate despite the huge phase magnitudes
  (`k * z` reaches ~4e5 rad over an 80 mm air gap at
  `lambda = 1.31 um`), **kernel-phase and phase-screen arguments are
  always computed in float64 and reduced modulo 2 pi before the
  final trig cast to float32**.  Without this mitigation the naive
  complex64 ASM kernel would inject ~0.02 rad noise per Fourier
  pixel (a diffuse speckle floor at ~-80 dB); with it, the only
  remaining precision loss is the FFT's natural single-precision
  round-off.  Validated on N=512:

  | Test | Kernel phase range | c128 vs c64 rel err |
  |---|---|---|
  | ASM z=1 mm | ~5 rad | 2.2e-7 |
  | ASM z=80 mm | ~4e5 rad | 3.1e-7 (mitigation working) |
  | `apply_real_lens` singlet | ~240 rad / surface | 2.4e-7 |
  | End-to-end 2-lens chain | mixed | 4.9e-7 |

  New top-level export: `DEFAULT_COMPLEX_DTYPE`.

- **`smooth_sigma_px` kwarg on `_sample_local_tilts` (default 4.0)**
  for robustness on multi-mode inputs.  The Tier 1 input-aware ray
  launch added in 3.1.2 extracts per-pixel tilts from the local
  phase gradient; on a single-mode field (plane wave, Gaussian,
  MLA-tilted beamlet) this produces a smooth tilt field that
  correctly parametrises the forward ray trace.  On a multi-mode
  interferogram (post-DOE field with 144 diffraction orders) the
  phase gradient aliases at every fringe boundary, clips to
  `max_sin`, and injects chaotic per-pixel directions into the
  ray trace -- the `RectBivariateSpline` over the resulting
  entrance→exit map becomes high-frequency, Newton diverges, and
  the output field collapses to zero.  The new amplitude-weighted
  Gaussian smoothing ( `blur(|E|^2 * L) / blur(|E|^2)` ) low-passes
  the tilt field before clipping:

  *   Single-mode fields: tilt magnitudes preserved to ~1 % (σ=4 px
      is much smaller than the beam feature scale).
  *   Multi-mode superposition: oscillations average to mean tilt,
      which for a balanced DOE is near zero -- naturally degenerating
      to the classical collimated launch.
  *   Mixed fields: handled per-pixel via the local-neighbourhood
      average.

  Pass `smooth_sigma_px=0` to recover the pre-3.1.3 behaviour.
  Optional `multimode_diagnostic` dict parameter, populated with
  `raw_rms_L`, `smoothed_rms_L`, `smoothing_ratio` etc., for callers
  that want to log or assert on the smoothing's effect.

### Fixed

- **`apply_real_lens_traced` no longer produces a zero output field
  on multi-mode inputs at large N.**  Symptom: at N=32768 with a
  post-DOE input (12×12 Dammann orders), the Newton inversion
  diverged for all pixels -- chunk 1 took 100+ minutes iterating,
  subsequent chunks finished in seconds as every ray hit the
  fit-domain clip and was NaN'd out.  Root cause: the aliased
  per-pixel tilts from `_sample_local_tilts` fed chaotic
  `(x_out, y_out)` entries to the entrance→exit spline; the
  cubic-spline fit overfit to the high-frequency oscillations and
  its derivatives (used in Newton's Jacobian) became unusable.  Fix:
  amplitude-weighted Gaussian smoothing of the tilt field described
  above.  Validated on the plane_09 field from the TX Design 36
  production run: max |L| dropped from 0.5 (clipping active) to 0.22
  (no clipping), spline derivatives stay sensible, Newton converges
  normally.

- **Analysis pool zombie cleanup in `tx_design_study_analysis.run_all_analysis`**
  (note: this fix is in the Reverse_Symmetric_ASM tree, not the
  library itself, but is relevant to users of the analysis pattern).
  Workers previously stayed resident after the `with
  ProcessPoolExecutor` block closed -- matplotlib figure state and
  MKL thread-pool atexit handlers blocked `_process_worker`'s
  `sys.exit()` on Windows spawn, leaving multi-GB zombies pinning
  RAM.  Now uses `max_tasks_per_child=1` (forces worker OS exit
  after one task), `plt.close('all')` + `gc.collect()` in each
  worker's `finally`, and a 30 s bounded shutdown wait with
  SIGTERM/SIGKILL straggler sweep if the primary shutdown hangs.

### Added (exposed)

- `DEFAULT_COMPLEX_DTYPE` -- module-level precision default, exported
  from the top-level package.
- `NUMEXPR_AVAILABLE` -- optional-backend availability flag,
  exported from the top-level package (useful in runner scripts
  that want to gate behaviour on the fast path being available).

## [3.1.2] — 2026-04-17

### Added

- **pyFFTW-to-scipy.fft automatic fallback on allocation failure.**
  At very large grids (e.g. ``N = 32768``) and tight RAM, pyFFTW's
  per-shape aligned-buffer plan can fail to allocate (the Windows
  allocator can't find a contiguous ~16 GB block even when total
  free RAM looks sufficient).  ``_fft2`` / ``_ifft2`` now wrap the
  pyFFTW call in try/except, catch any exception, emit a one-time
  ``RuntimeWarning``, blacklist that shape for the remainder of the
  session, flush the pyFFTW plan cache, and fall through to
  ``scipy.fft`` (numerically identical to pyFFTW at 1e-14 noise,
  just ~6x slower on large grids without aligned buffers).

  Three new user-facing controls:

  * ``op.set_fft_fallback(False)`` -- disable the fallback and let
    pyFFTW errors propagate (useful to flush out genuine backend
    bugs).
  * ``op.reset_fft_backend()`` -- clear the bad-shape blacklist and
    the pyFFTW plan cache, so subsequent calls retry pyFFTW (use
    after a big one-off allocation has been freed).
  * Top-level exports: ``set_fft_fallback``, ``reset_fft_backend``.

- **Tier 1 input-aware ray launch** (new ``tilt_aware_rays`` kwarg,
  default True) in :func:`apply_real_lens_traced`.  Each ray's
  initial direction cosines ``(L, M)`` are now derived from the
  local phase gradient of ``E_in`` at its entrance position so the
  lens OPL is evaluated at the ACTUAL angle of incidence rather
  than under a blanket plane-wave assumption.  For plane-wave
  inputs this is bit-identical to the collimated launch; for MLA /
  DOE / off-axis / pre-aberrated inputs it correctly carries the
  lens-OPL-vs-angle dependence through to the exit plane.  Tilts
  are clipped to ``|sin(theta)| <= 0.5`` for numerical safety and
  low-amplitude pixels (< 0.1 % of peak) are treated as noise
  floor and launched collimated.  Cost overhead: a few-percent
  (one numpy gradient + bilinear resample), bit-exact equivalence
  on plane-wave inputs verified.

### Fixed

- **`apply_real_lens_traced` was silently discarding the input
  field's phase.**  Before this fix the output was
  ``|apply_real_lens(E_in)| * exp(i*k0*OPL_traced)`` -- the
  amplitude carried the input correctly but the phase only reflected
  the lens's ray-traced OPL applied to a synthetic plane wave.  Any
  input-field phase structure (source tilt, MLA / DOE modulation,
  off-axis wavefronts, pre-applied aberrations) was dropped.
  Symptom: tilted inputs focused on-axis; MLA-modulated inputs came
  out as featureless envelopes at downstream planes.

  New default ``preserve_input_phase=True`` keeps the full complex
  ``E_analytic`` and applies a correction that replaces the analytic
  model's lens phase with the ray-traced OPL.  Matches
  :func:`apply_real_lens`'s behaviour for the input-field part, with
  the ray-traced OPL correction on top.

  Cost: runs ``apply_real_lens`` a second time on a unit plane-wave
  reference so we can extract and subtract the analytic lens phase.
  ~20 % overhead on the total function time at large N.

  Pass ``preserve_input_phase=False`` to restore the legacy behaviour
  (useful for plane-wave-only lens-OPD measurements where the
  distinction is moot).

### UI-library fix

- **Asphere coefficient editor** (``ui/surface_editors.py``) was
  storing ``aspheric_coeffs`` as a Python ``list`` but the library's
  canonical convention (used by Zemax import,
  :func:`surface_sag_general`, and the raytracer) is a ``dict``
  ``{power: coeff}``.  UI-edited aspherics would crash with
  ``AttributeError: 'list' object has no attribute 'items'`` when
  the prescription was then simulated.  Now stores the correct dict
  form and migrates legacy list-format on load.

## [3.1.1] — 2026-04-16

### Performance

- **`apply_real_lens_traced` Newton inversion now parallel.**  The
  `n_workers` kwarg (previously a dead parameter) now dispatches the
  embarrassingly-parallel Newton-invert step to a
  `ProcessPoolExecutor`.  Each worker rebuilds the three
  `RectBivariateSpline` objects from their knot data locally, so the
  pickling cost per chunk is just the knot arrays (~200x200 floats),
  not the spline objects themselves.
  - Measured: 8.3x speedup on 16 workers on a 4M-pixel Newton
    benchmark.
  - On small grids (< 200 k coarse pixels) the function auto-falls
    back to the in-process serial path so pool startup doesn't make
    small calls slower.
  - Numerically identical to the serial path (verified `max |diff|
    = 0` on a 1 Mpx test).
  - Threading is explicitly **not** used: SciPy's
    `RectBivariateSpline.ev` does not release the GIL in current
    SciPy versions, contrary to the previous docstring claim.  The
    previous comment about thread scaling has been removed.

### Added

- **`min_coarse_samples_per_aperture` + `on_undersample` kwargs on
  `apply_real_lens_traced`.**  Subsampling guardrail: if the
  coarse Newton grid has fewer than N samples across the lens
  aperture, the cubic-spline interpolation of the wavefront aliases
  and the result is wrong.  Benchmarks showed the rule is roughly
  `RMS phase err ~ (coarse_samples_per_aperture)^-2`:

  | Coarse samples / aperture | Typical RMS phase err (lambda=1.31 um) |
  |---|---|
  | 64 | ~20 nm |
  | 32 (default threshold) | ~85 nm |
  | 16 | ~350 nm (unusable) |

  Default policy is `on_undersample='error'`: raises `ValueError`
  with the safe `ray_subsample` value computed for the current
  grid, so the user is never silently running a corrupt sim.
  `'warn'` and `'silent'` policies are available; setting
  `min_coarse_samples_per_aperture=0` disables the check.

## [3.1.0] — 2026-04-16

UX-pass companion release: progress hooks, `'real_lens_traced'`
element type for `propagate_through_system`, codegen promoted to the
public API, and a handful of UI-driven correctness fixes.

### Added (new merit term + multi-prescription optimisation)

- **`optimize.MatchIdealSystemMerit`** -- propagate a reference source
  through BOTH an idealised thin-lens architecture and the real
  prescription, then penalise the output-field mismatch.  Unlike
  `MatchIdealThinLensMerit` (which compares exit-pupil OPD to a bare
  converging sphere), this merit compares the actual complex output
  fields, so the optimizer drives the real lens toward matching the
  ideal's radiation pattern AND relative phase -- which is what the
  "replace this thin-lens system with a real one" workflow actually
  wants.  Supports four similarity metrics
  (`field_overlap`, `field_mse`, `intensity_mse`, `intensity_overlap`),
  arbitrary architectures via element lists, optional pre/post
  propagations, and a `single_lens(f)` convenience factory for the
  common case.  Also supports:
    * ``focus_search=True`` -- axial z-offset scan that decouples
      "correct focal plane" from "aberration quality".  Measured
      improvement: 93 % penalty reduction on a thick plano-convex
      whose BFL sits 6.7 mm off the ideal's target, letting the
      optimizer converge on shape rather than fighting the BFL shift.
    * ``wavelengths=[...]`` -- built-in chromatic sweep; evaluates
      at each wavelength and averages the penalty.
    * ``field_angles=[(theta_x, theta_y), ...]`` -- built-in off-axis
      sweep; applies a linear-phase carrier to the source for each
      field point.  Combines Cartesian-product-wise with
      ``wavelengths``.
- **`optimize.MultiPrescriptionParameterization`** -- holds a list of
  template prescriptions and a free-var list whose entries start with
  a prescription index.  ``design_optimize`` recognises the class
  automatically, populates ``ctx.prescriptions`` alongside the
  existing single ``ctx.prescription`` (which stays == ``[0]`` for
  backward compatibility), and passes both through to the merit
  terms.  ``MatchIdealSystemMerit``'s ``_prescription_`` placeholder
  now accepts an ``'index'`` key to select which prescription slots
  in where.  Verified with a 4f architecture: two singlets jointly
  optimised, 220 iterations, merit reduces from 0.060 -> 0.005, each
  lens settles at a distinct optimised form.

### Fixed (third-pass deep-audit findings)

- **`analysis.compute_psf`** -- default normalisation changed from
  ``'peak'`` (which made ``psf.max() == 1`` unconditionally,
  silently breaking every canonical
  ``strehl = compute_psf(abb).max() / compute_psf(ideal).max()``
  pattern) to ``'power'``.  A direct test now recovers
  Strehl = 0.906 / 0.674 / 0.411 / 0.206 at 0.05 / 0.10 / 0.15 /
  0.20 waves RMS of Z(4,0), tracking extended Marechal to ~1 %
  at small aberrations.  The old behaviour is available as
  ``normalize='peak'`` (for display) and raw ``|FFT|^2`` as
  ``normalize='none'``.  Breaking-change advisory: callers who
  relied on peak-normalised output must now opt in explicitly.
- **`propagation.angular_spectrum_propagate_tilted`** -- band-limit
  mask now tests ``|FX| < fx_max`` on the baseband (post-demod)
  frequency grid instead of ``|FX_shifted|``.  For any non-trivial
  tilt the old mask was zeroing the baseband DC mode and killing
  the propagated field.  Measured power preservation after the fix:
  input ``mean|E|^2 = 0.235``, after ``z = 0.01 m`` with tilt 0.05
  rad and bandlimit=True: ``mean|E|^2 = 0.235`` (was 3.7 x 10^-7).
- **`optimize.FocalLengthMerit` / `BackFocalLengthMerit`** -- guard
  against ``target == 0`` (collimator / afocal case).  Old code
  computed ``(efl - 0) / 0`` -> NaN and poisoned the optimizer.
  New behaviour: when ``target == 0``, penalise ``efl^2`` directly,
  driving the optimizer toward infinite EFL.
- **`optimize.ToleranceAwareMerit`** -- each Monte-Carlo trial now
  recomputes the perturbed system's BFL via ``system_abcd`` and
  scans through focus around THAT plane rather than the nominal
  BFL.  Perturbations that significantly shift the focal plane
  (e.g. large first-surface decenters) no longer produce
  artificially low Strehl from scanning off-focus.
- **`optimize.ToleranceAwareMerit`** -- form-error seed is now
  deterministic per ``(trial, surface_index)`` pair instead of
  drawn from the global RNG.  Two runs with the same base seed
  now produce identical perturbation realisations regardless of
  surrounding RNG calls.
- **`optimize`: new ``ctx_is_valid`` helper + merit-term sentinel
  guards** -- when ``system_abcd`` fails, EvaluationContext sets
  ``efl = bfl = 1e9``; merit terms that consume those now return a
  bounded penalty instead of ``(1e9 - target)^2`` which dragged the
  optimizer away from good regions.
- **`doe.makedammann2d`** -- target-pattern centering uses plain
  integer division ``(N_big - N_small) // 2`` instead of the
  Octave-port's ``ceil((N_big - N_small)/2) + 1`` offset.  For odd
  differences the old code placed the input pattern one cell to
  the left of center, breaking the binary symmetry Dammann designs
  rely on.
- **`storage.set_storage_backend('zarr')`** -- raises ``ImportError``
  immediately if zarr isn't installed instead of lazy-failing on
  the first ``append_plane`` call.
- **`codegen`** -- generated-script numeric literals now use
  ``.17e`` format (full IEEE 754 round-trip precision) instead of
  ``.6e`` which lost ~0.1 nm per value; ~0.1 um drift across a
  multi-surface prescription.
- **`elements.apply_aperture` / `apply_gaussian_aperture`** --
  accept an optional ``dy`` kwarg (defaults to ``dx``) for
  rectangular (non-square) grids.  Previously the y-coordinate
  grid was built from ``dx`` regardless, silently stretching the
  aperture along the y axis on non-square grids.
- **`optimize.RMSWavefrontMerit`** -- exposed the previously
  hard-coded ``exclude_low_order`` via a constructor kwarg.
  Default value raised from 3 to 4 (piston + 2 tilts + defocus)
  to match the "image-quality RMS after best-focus" convention.
- **`through_focus.plot_through_focus`** -- Strehl axis uses
  ``max(1.05, 1.1 * Strehl_max)`` so rare super-unity peaks stay
  visible.
- **`user_library`** -- serialise/deserialise pair now recursively
  handles ``float('inf')`` / ``float('-inf')`` anywhere in the
  prescription (previously only ``surfaces[i]['radius']`` was
  restored; infinities in ``thickness``, ``aperture_diameter``,
  ``conic``, etc. came back as the string ``'Infinity'`` and
  caused downstream ``TypeError``).
- **`elements.apply_zernike_aberration`** docstring -- explicit
  unit-conversion example for round-tripping with
  ``analysis.zernike_decompose`` (apply takes waves, decompose
  returns metres).
- **`hdf5_io`** no longer re-exports the private
  ``storage._decode_attr``.

### Fixed (second-pass deep-audit findings)

- **`rcwa.py`** — the old `rcwa_1d` built the Moharam-Gaylord
  Fourier eigendecomposition but then **threw it away** and
  renormalised ``T / sum(T)``, hiding any non-unit energy in R.
  Replaced with a clean analytical thin-phase-grating formula that
  respects energy conservation (`sum|t_m|^2 = 1` by Parseval for
  propagating orders; evanescent orders correctly carry zero).
  Docstring now states up front that this is a scalar thin-grating
  approximation, not full RCWA -- R is always zero in the thin
  regime, and a future S-matrix implementation is left as a clear
  TODO.  The function signature is unchanged; existing call sites
  keep working.
- **`detector.py::apply_detector`** — pixel binning used
  integer-truncation indexing that gave wildly non-uniform
  per-pixel sample counts (9 / 12 / 16 / 20 / 25 depending on where
  integer boundaries fell).  On a uniform field the resulting
  "std / sqrt(mean)" was ~20 instead of the ~1 a pure Poisson
  process should give, so shot-noise statistics were unreliable.
  Replaced with `scipy.ndimage.zoom`-based area-weighted
  integration.  Measured std / sqrt(mean) on a uniform high-count
  field is now 1.03 (integer pitch ratio) and 1.04 (non-integer).
- **`freeform.py::surface_sag_chebyshev`** — outside the Chebyshev
  normalisation box `[-norm_x, norm_x] * [-norm_y, norm_y]`, the
  function used to return the boundary value `T_n(+-1)`, creating
  a large step discontinuity at the domain edge that broke the ray
  tracer's Newton intersection solver.  Now zeroes the departure
  outside the domain while preserving the base conic sag.
- **`interferometry.py::phase_shift_extract`** — added
  ``convention='hardware' | 'library'`` kwarg.  The extraction
  formula's sign depends on whether the caller supplies frames
  following the Schwider/Hariharan convention ``I = a + b*cos(phi - s)``
  (what every real phase-shifting interferometer produces) or the
  opposite ``I = a + b*cos(phi + s)`` (what this library's own
  `simulate_interferogram` produces).  Default is ``'hardware'`` so
  that real-instrument data round-trips sign-correctly; pass
  ``'library'`` to round-trip a `simulate_interferogram` output.
- **`multiconfig.py::keplerian_telescope` and
  `beam_expander_prescription`** — the thin-lens separation
  ``f_obj + f_eye`` was used even though the functions build thick
  singlets, leaving the output systems non-afocal (|C| ~ 0.07/mm
  on a Keplerian 200/25 mm).  Replaced with a one-step linear
  solve on the air gap that drives the system ABCD's C element
  to exactly zero (machine precision).  Separately fixed a geometry
  bug in `beam_expander_prescription` where the eyepiece was built
  as plano-concave `[R, inf]` while using the equi-convex focal-
  length formula `R = f*(n-1)*2`, which halved the eyepiece focal
  length and gave an expansion ratio of M/2.  Eyepiece is now
  equi-shaped ``[R, -R]`` matching the formula.  `M=5` now
  delivers 5x (was 2.5x).
- **`multiconfig.py::afocal_angular_magnification`** — the
  ``is_afocal`` test used ``|B| < 1e-6``, but the afocal condition
  is ``C = 0`` (collimated in -> collimated out), not ``B = 0``
  (which is 1:1 imaging).  Test now uses ``|C| * aperture_radius <
  1e-6`` (equivalent to a sub-microradian residual output
  divergence for a typical input bundle).

### Added

#### Progress hooks (`progress.py`, new module)
- `ProgressCallback` type alias, `call_progress(cb, stage, frac, msg)`
  helper, and `ProgressScaler` for nesting sub-tasks within a parent
  budget.
- The following long-running core functions now accept an optional
  `progress=cb` keyword:
  - `apply_real_lens` (per-surface progress)
  - `apply_real_lens_traced` (amp pass + ray trace + Newton inversion)
  - `propagate_through_system` (per-element, recursively scaled into
    the lens-model sub-progress where applicable)
  - `through_focus_scan` (per-z-plane)
  - `tolerancing_sweep` (per-perturbation, each sub-run scaled to
    its slice of the overall bar)
  - `monte_carlo_tolerancing` (per-trial)
  - `design_optimize` (per merit-function evaluation; approximate
    because scipy's optimizers don't expose uniform iteration counters)
- Hooks are completely opt-in (None = no overhead) and exception-safe
  (a broken callback can never crash a simulation).  Signature is
  `(stage: str, fraction: float, message: str = '')`.
- Shared between any script that wants a progress bar and the
  optical-designer UI's wave-optics / optimizer / tolerance docks.

#### System propagation
- `propagate_through_system` gained a `'real_lens_traced'` element
  type that delegates to `apply_real_lens_traced` with optional
  `bandlimit` and `ray_subsample` per-element overrides.
- Pass-through `progress=` to `apply_real_lens` /
  `apply_real_lens_traced` per element via `ProgressScaler` windows.

#### Code generation (`codegen.py`)
- `generate_simulation_script`, `generate_script_from_zmx`, and
  `generate_script_from_txt` are now exported from the top-level
  `lumenairy` namespace.  Previously importable but
  undocumented.

### Changed

- `make_singlet` and `make_doublet` now always emit `radius_y`,
  `conic_y`, and `aspheric_coeffs_y` keys (set to `None`) so a
  prescription dict round-trips through diff-friendly tooling
  without ambiguity.
- `propagation.py` module docstring spells out the return-type
  contract (ASM + RS bare arrays; Fresnel + Fraunhofer 3-tuples).
- `apply_real_lens_traced` docstring surfaces the
  `dx \u2264 \u03bb*f / aperture` Nyquist requirement and points readers at
  `check_opd_sampling`.
- `elements.zernike` docstring corrected: it uses OSA / unit-variance
  normalization with `(n, m)` indexing, not Noll single-index.
- `create_multi_field_sources` docstring flags its
  list-of-tilted-plane-waves return shape (different from the scalar
  `create_*` helpers' `(E, x, y)` triple).

### Fixed

- `analysis.remove_wavefront_modes` accepts a `weights` keyword for
  intensity-weighted least-squares fits; vignetted / annular pupils
  no longer leak high-order content into piston/tilt/defocus.
- `raytrace.surfaces_from_prescription` plumbs the optional
  `freeform` key through to the `Surface` dataclass; freeform sags
  (XY-polynomial / Zernike / Chebyshev) are now ray-traceable, not
  wave-only.

### Removed

- `optical_table.py` + `optical_table.html` and their bidirectional
  scene/element/prescription translators.  The bundled HTML simulator
  was unreferenced from the GUI and the only Python entry points were
  not exported from `__init__.py`, so it was effectively dead code.
  Daniel L. Marks' zlib-licensed HTML application is no longer
  redistributed; see prior commits for the source.

---

## [3.0.0] — 2026-04-16

Major release: hybrid wave/ray lens model, 15+ new physics modules,
design optimizer, comprehensive validation suite.

### Added

#### Lens modelling
- `apply_real_lens_traced` — hybrid wave/ray lens model combining
  wave-optics amplitude (from `apply_real_lens`) with geometrically
  exact per-pixel ray-traced OPL phase.  Sub-nanometre OPD agreement
  with the geometric ray trace across all tested lens geometries
  (singlets, doublets, meniscus, biconcave, equi-convex).
- `surface_sag_biconic` — biconic/cylindrical/toroidal surface sag
  with independent x/y radii, conics, and aspheric coefficients.
- `apply_cylindrical_lens` — cylindrical thin-lens phase screen.
- `apply_grin_lens` — gradient-index rod lens.
- `apply_axicon` — conical phase element.
- Prescription keys `radius_y`, `conic_y`, `aspheric_coeffs_y` for
  anamorphic surfaces throughout the pipeline (ray tracer, ABCD,
  Seidel, OPD analysis).

#### Prescription builders
- `make_cylindrical` — cylindrical singlet prescription.
- `make_biconic` — biconic singlet prescription.
- `export_zemax_lens_data` — human-readable Zemax LDE text export.
- `export_zemax_zmx` — Zemax .zmx binary prescription export.

#### Design optimizer (`optimize.py`, new module)
- `DesignParameterization` — maps a flat parameter vector to a
  prescription dict for scipy optimizers.
- 18 merit term classes: `FocalLengthMerit`, `BackFocalLengthMerit`,
  `SphericalSeidelMerit`, `StrehlMerit`, `RMSWavefrontMerit`,
  `SpotSizeMerit`, `ChromaticFocalShiftMerit`, `MatchIdealThinLensMerit`,
  `MatchTargetOPDMerit`, `ZernikeCoefficientMerit`, `CompositeMerit`,
  `CallableMerit`, `MultiWavelengthMerit`, `MultiFieldMerit`,
  `MinThicknessMerit`, `MaxThicknessMerit`, `MinBackFocalLengthMerit`,
  `MaxFNumberMerit`, `ToleranceAwareMerit`.
- `design_optimize` — unified entry point supporting L-BFGS-B, SLSQP,
  trust-constr, differential_evolution, basin_hopping, dual_annealing,
  and Levenberg-Marquardt (via Householder QR).

#### Through-focus and tolerancing (`through_focus.py`, new module)
- `through_focus_scan` — propagate a field to multiple z-planes and
  collect peak intensity, Strehl, and beam metrics at each.
- `find_best_focus` — locate best focus from a through-focus scan.
- `plot_through_focus` — matplotlib visualization.
- `diffraction_limited_peak` — ASM-based reference for Strehl.
- `Perturbation`, `apply_perturbations` — structured perturbation model.
- `tolerancing_sweep` — systematic single-parameter sensitivity.
- `monte_carlo_tolerancing` — Monte Carlo tolerance analysis.

#### Analysis (`analysis.py`, expanded)
- `zernike_decompose` — Householder QR with column pivoting (gelsy),
  numerically stable for high-order and partial-pupil data.
- `zernike_reconstruct`, `zernike_polynomial`, `zernike_basis_matrix`.
- `zernike_index_to_nm`, `zernike_nm_to_index` — OSA index helpers.
- `chromatic_focal_shift` — focal shift vs wavelength.
- `polychromatic_strehl` — polychromatic Strehl ratio.
- `check_opd_sampling` — Nyquist margin calculator for OPD extraction.
- `wave_opd_1d`, `wave_opd_2d` — with focal-length warnings and
  optional reference-sphere subtraction (`f_ref`).

#### Vector diffraction (`vector_diffraction.py`, new module)
- `richards_wolf_focus` — Richards-Wolf high-NA vector focusing.
- `debye_wolf_psf` — Debye-Wolf PSF computation.

#### Partial coherence (`coherence.py`, new module)
- `koehler_image` — Koehler illumination imaging.
- `extended_source_image` — extended-source incoherent imaging.
- `mutual_coherence` — mutual coherence function.

#### Detector model (`detector.py`, new module)
- `apply_detector` — shot noise, read noise, dark current, QE,
  full-well clipping, pixel binning.
- `shack_hartmann` — Shack-Hartmann wavefront sensor simulation.

#### Thin-film coatings (`coatings.py`, new module)
- `coating_reflectance` — transfer-matrix method (TMM) for multilayer
  dielectric coatings: R, T, phase vs wavelength and angle.
- `quarter_wave_ar` — single-layer AR coating designer.
- `broadband_ar_v_coat` — two-layer V-coat AR designer.

#### Interferometry (`interferometry.py`, new module)
- `simulate_interferogram` — generate fringe patterns from OPD maps.
- `phase_shift_extract` — 4-step phase-shifting interferometry.
- `fringe_spacing` — fringe spacing calculator.

#### Freeform surfaces (`freeform.py`, new module)
- `surface_sag_xy_polynomial` — XY polynomial departure from base conic.
- `surface_sag_zernike_freeform` — Zernike polynomial freeform.
- `surface_sag_chebyshev` — Chebyshev polynomial freeform.
- `surface_sag_freeform` — unified dispatcher.

#### Ghost analysis (`ghost.py`, new module)
- `enumerate_ghost_paths` — find all double-bounce ghost paths.
- `ghost_analysis` — trace ghost paths and compute intensity.

#### RCWA (`rcwa.py`, new module)
- `rcwa_1d` — rigorous coupled-wave analysis for 1D gratings.
- `grating_efficiency_vs_wavelength` — spectral efficiency sweep.

#### Multi-configuration (`multiconfig.py`, new module)
- `Configuration` dataclass for multi-config merit evaluation.
- `multi_config_merit` — weighted merit across configurations.
- `create_zoom_configs` — zoom-system configuration builder.
- `afocal_angular_magnification` — angular magnification from ABCD.
- `beam_expander_prescription` — Galilean beam expander builder.
- `keplerian_telescope` — Keplerian telescope builder.

#### Sources (`sources.py`, expanded)
- `create_tilted_plane_wave`, `create_point_source`.
- `create_multi_field_sources` — multi-field-angle source array.
- `create_top_hat_beam`, `create_annular_beam`.
- `create_fiber_mode` — LP01 fiber mode.
- `create_led_source` — incoherent LED model.
- `create_bessel_beam` — non-diffracting Bessel beam.

#### Phase retrieval (`phase_retrieval.py`, new module)
- `gerchberg_saxton` — Gerchberg-Saxton algorithm.
- `error_reduction` — error-reduction algorithm.
- `hybrid_input_output` — hybrid input-output (HIO) algorithm.

### Changed

- **SciPy FFT default** — `USE_SCIPY_FFT = True`, `SCIPY_FFT_WORKERS = -1`.
  All wave-propagation functions now multithreaded by default (2-4x speedup).
- **`slant_correction` default reverted to `False`** — empirical validation
  showed the paraxial formula is equal-or-better for most cases because ASM
  propagation between surfaces already encodes obliquity.
- `apply_real_lens` gains `seidel_correction` option (opt-in, off by
  default) for analytic higher-order correction on doublets.

### Fixed

- **Exit-vertex OPL correction** — `apply_real_lens_traced` now transfers
  rays from the last surface's sag to the flat exit vertex plane using the
  signed parametric distance.  Previously, off-axis rays ended at
  `z = sag(h) != 0`, injecting systematic defocus (43% on doublets) or
  catastrophic sign errors (200,000x on negative meniscus lenses with
  convex rear surfaces).  Doublet focus error: 10 mm to 0.000 mm.
  Negative meniscus residual: 33,742 nm to 0.17 nm.
- **Raytrace OPL bookkeeping** — `_intersect_surface` now accumulates
  `n_medium * t` for the vertex-to-sag leg.  Previously the ray moved to
  the sag intersection without counting that path.  Singlet residuals
  dropped 17x-130x.
- **TMM coating formula** — corrected B,C matrix extraction:
  `B = M[0,0] + M[0,1]*eta_sub` (was transposed).  Quarter-wave AR now
  gives R = 0 exactly at the design wavelength.

### Validation

- `validation/real_lens_opd/` — 21-case OPD validation suite comparing
  three methods (paraxial, slant-corrected, ray-traced) against geometric
  truth.  All cases show sub-nm traced RMS.  Matching Zemax LDE + .zmx
  exports for cross-verification.

---

## [2.5.0] — Prior release

- Core ASM/Fresnel/Fraunhofer propagation.
- `apply_real_lens` thin-element phase-screen model.
- Geometric ray tracer with ABCD, spot diagrams, Seidel coefficients.
- Glass catalog (Sellmeier, refractiveindex.info integration).
- Gaussian, Hermite-Gauss, Laguerre-Gauss source models.
- Polarization (Jones calculus).
- DOE / microlens array generation.
- HDF5 field I/O.
- Plotting utilities.
