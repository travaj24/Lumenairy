# Changelog — Optical Designer (GUI)

All notable changes to the GUI application are documented here.
For core library changes, see `CHANGELOG.md`.

**Versioning note:** starting 2026-04-17 the GUI distribution tracks
the same version number as the core `lumenairy` library
(previously the GUI had its own 3.2.x track alongside the library's
3.1.x, which caused confusion -- the About dialog reads
`__version__` from the library, so users saw two different numbers
for the same release).  Historical GUI-only releases (e.g. 3.2.0,
2026-04-16) retain their original numbers below for traceability.

## [3.2.14] — 2026-04-24

### Performance perceptible in the GUI

Mirrors the core-library 3.2.14 perf pass.  The GUI does not change;
typical workflows are simply faster:

- Multi-config / wavelength sweeps + optimization loops now hit the
  ASM transfer-function `H` cache when the geometry repeats — ~1.5×
  speedup per ASM call on N=2048 grids.
- `JonesField.propagate` (used by the Wave Optics dock when the
  source is polarized) runs Ex/Ey through a single batched FFT pair
  on grids ≥ 512.
- Single-precision (`np.complex64`) toggle now exposed at the
  package top level — flip once for ~2× FFT throughput and ~2× more
  headroom on memory-tight grids.
- Aspheric-surface sag computation is now numba-fused (one threaded
  pass over the grid, no per-coefficient temporaries).  ~4.75×
  speedup on N=2048 with 5 aspheric coefficients.

## [3.2.13] — 2026-04-24

### No GUI-side changes

Validation-suite expansion in the core (~70 new physics / interop
test cases, 298 total assertions across 16 files, all PASS).  The
GUI inherits the safer regression net but has no user-facing change.

## [3.2.12] — 2026-04-24

### UI quality-of-life: keyboard, drag-drop, REPL, compact mode

- **`Ctrl+1` … `Ctrl+9`** — jump directly to workspace tab N.
- **Window title** shows current file + dirty marker (`Optical
  Designer — file.zmx*`).
- **Drag-and-drop** any `.zmx` / `.txt` / `.seq` / `.json` onto
  the window to load.
- **Permanent right-aligned status-bar metrics**: EFL, BFL, f/#,
  EPD, λ — visible on every workspace.
- **Pinned docks** across all workspaces (`View > Workspace > Pin
  Docks Across All Workspaces…`).
- **Welcome dock** — empty-state landing panel with quick-start
  buttons + recent-files list backed by `QSettings`.
- **Embedded Python REPL** dock with `model`, `np`, `plt`,
  `result`, `wave` pre-bound to the current system + latest run.
- **Workspace export/import** as `.workspace` JSON files for
  sharing custom layouts.
- **Optimizer progress badge** on the Optimize tab title
  (`Optimize • iter N`) while running.
- **Element-table polish**: right-click context menu, amber
  highlight on cells with optimization variables, search box.
- **F11 / Compact Mode** — hides menu bar + dock title bars.

## [3.2.11] — 2026-04-24

### Workspace defaults rebalanced

- Added a dedicated **Optimize** tab between Design and Analysis
  holding Optimizer + Sliders + Multi-Config + Snapshots.
- Slimmed the **Design** tab to just 2D / 3D Layout + System Data
  + Library — the docks you actually look at while building a
  layout.
- Dropped Jones Pupil from the Wave Optics tab defaults (still
  available via `View > Jones Pupil` or Manage Docks).
- Added a `defaults_revision` migration so existing users with
  saved layouts pick up the new tabs without losing customisations.

## [3.2.10] — 2026-04-24

### Top-of-window workspace tabs

A tabbed-workspace system reduces GUI clutter by grouping the 27+
analysis docks by topic.  Each tab shows only the docks relevant
to that phase of design work.

- New `ui/workspace.py` with `Workspace`, `WorkspaceBar`,
  `ManageWorkspaceDialog`, `WorkspaceManager`.
- Default workspaces: **Design**, **Analysis**, **Wave Optics**,
  **Tolerancing**, **Materials**.  Right-click any tab for Manage
  Docks / Rename / Duplicate / Delete; `+` button to add new
  workspaces; double-click to rename.
- Per-tab dock geometry preserved on switch via
  `QMainWindow.saveState()` / `restoreState()`.
- User-initiated dock visibility changes (close button, View menu
  toggle) automatically update the active workspace's dock list.
- Persistence to `QSettings('lumenairy', 'OpticalDesigner')` —
  custom workspaces survive restart.
- Wired into `main_window.py` with `_build_workspace_bar()` +
  `_init_workspaces()` + a `View > Workspace` submenu (with Reset
  to Defaults).

## [3.1.6] — 2026-04-21

### No GUI-side changes

- Core-library reliability fix: zarr storage writes now succeed on
  Windows + Python 3.14 + zarr v3 (previously crashed with
  ``FileExistsError`` on reopen).  The Optical Designer's save
  dialogs select between HDF5 and Zarr backends; before this
  patch, zarr writes could error mid-run on Windows.  No API change
  -- the GUI picks up the fix automatically.  See ``CHANGELOG.md``
  for details.

## [3.1.5] — 2026-04-20

### No GUI-side changes

- This release is a core-library bugfix for `.zmx` / `.txt` loaders
  (see `CHANGELOG.md` for details on the new
  `prescription['object_distance']` key).  The GUI does not surface
  this value directly, but prescriptions loaded via the Optical
  Designer's "Load Prescription" dialog now carry the correct
  obj-space distance in their returned dict -- downstream scripts
  and the GUI's own wave-optics preview stages benefit from the
  loader correction without any user-visible change.

## [3.1.4] — 2026-04-18

### Changed (wave-optics dock)

- **`Tilt-aware ray launch` checkbox default flipped from checked
  (True) to unchecked (False).**  Matches the core library's
  3.1.4 default flip of `apply_real_lens_traced(..., tilt_aware_rays=...)`.
  The previous default produced a reference-frame mismatch with
  the `preserve_input_phase=True` subtraction that output wrong
  fields on multi-mode inputs (post-DOE diffraction patterns,
  compound superpositions).  Existing GUI saves / sessions should
  re-run any wave-optics analysis with the new default to pick up
  the fix.  Advanced users doing rigorous off-axis characterisation
  of single-mode tilted inputs can still tick the checkbox.
- Checkbox tooltip updated to explain the new default + when to
  turn it on.

### Library side (transparent to GUI users)

See `CHANGELOG.md` for details.  Highlights: paraxial-magnification
Newton initial guess, experimental `inversion_method='backward_trace'`
opt-in for ~3x speedup on large grids, and the traced-lens
correctness fix for multi-mode inputs.

## [3.1.3] — 2026-04-17

Version-number unification release + two new controls in the
wave-optics dock exposing the core library's 3.1.3 additions.
Ships with **core library 3.1.3** (same version now -- see the
versioning note above) which is drop-in compatible with existing
GUI prescriptions.  See `CHANGELOG.md` for the full library entry.

### Added (wave-optics dock)

- **Precision selector** (Compute group): drop-down to choose between
  ``complex128`` (default, double precision) and ``complex64`` (single
  precision, half memory + ~2x FFT/phase-screen throughput).  The
  library applies its mod-2pi kernel-phase mitigation so complex64
  accuracy is bounded by FFT single-precision round-off (~-80 dB
  cumulative) rather than the phase-magnitude floor.  Tooltip lists
  the headroom tradeoff so users running deep-null / stray-light
  analysis below -60 dB know to stay at double.  The selected dtype
  propagates through the source-field allocation and all downstream
  library calls that preserve caller dtype (`apply_real_lens`,
  `apply_real_lens_traced`, `angular_spectrum_propagate`, `apply_mirror`).

- **Tilt-aware ray launch toggle** (Simulation Parameters group):
  checkbox exposing `apply_real_lens_traced(..., tilt_aware_rays=...)`.
  Defaults to True (matching the library default, and correct for the
  vast majority of inputs now that the smoothing fix makes multi-mode
  inputs robust).  Exposed primarily for A/B debugging and as an
  escape hatch if a pathological input slips past the smoothing.

### User-visible effects of the bundled library 3.1.3

In addition to the new UI controls above, the following improvements
apply automatically to every wave-optics run:

- **Wave-optics dock runs at large N (>= 16384) are faster and more
  memory-efficient.**  Each per-surface phase screen in `apply_real_lens`
  now uses a `numexpr`-fused multiply (optional dependency, falls back
  to numpy when absent) -- ~1.5-2x faster and ~3x lower peak memory at
  N=32768.  ASM propagation picks up a single-slot pyFFTW plan cache
  with in-place aligned buffers, eliminating the 30 s-TTL reallocation
  churn that previously fragmented Windows address space on multi-GB
  grids.
- **`apply_real_lens_traced` now converges correctly on multi-mode
  inputs.**  The 3.1.2 `tilt_aware_rays=True` default would silently
  zero-out the output field on post-DOE / interferometric inputs at
  large N (the per-pixel tilt extraction aliased at every fringe
  boundary, producing a chaotic entrance->exit spline that Newton
  couldn't invert).  The library now amplitude-weighted-Gaussian-smooths
  the tilt field so multi-mode inputs gracefully degenerate to the
  classical collimated launch while single-mode inputs (plane wave,
  Gaussian, MLA beamlets) keep their valid per-pixel tilts.  If you
  saw zero-output anomalies on a DOE-containing simulation, just re-run
  it in the GUI -- no prescription changes needed.
- **Optional complex64 mode** for ~2x memory / throughput at the cost
  of ~60 dB of effective cumulative dynamic range -- useful for
  design-verification sweeps at very large N where memory is the
  binding constraint.  Exposed as the Compute group's Precision
  selector in this release (see Added above).

### Fixed

- **Version line in the About dialog now reads correctly.** The GUI
  reads `__version__` from the package at runtime so it picks up the
  library bump (3.1.3) automatically -- no separate GUI-side version
  string to forget.

## [3.2.0] — 2026-04-16

Deep feature-gap pass: wired every high-leverage core-library
capability into the UI so the tool is usable for real design reviews
without dropping to Python.  Seven new analysis docks, four new
surface-form editors, a report generator, and an information-
architecture cleanup.

### Added — new docks

- **Through-focus dock** (`through_focus_dock.py`): axial Strehl /
  peak-intensity / RMS-radius / d4sigma plots with determinate
  progress, best-focus marker, CSV export.  Auto-populates its
  source field from the latest wave-optics run.
- **PSF / MTF dock** (`psf_mtf_dock.py`): log-scaled PSF + radial
  MTF plot, polychromatic-Strehl calculator across the Optimizer's
  wavelength list.  Pupil source is either the wave-optics
  exit-plane or a ray-trace-derived synthetic pupil.
- **Sensitivity dock** (`sensitivity_dock.py`): per-variable
  finite-difference d(merit)/d(var) with a horizontal bar chart
  sorted by |magnitude|.  Metric selectable (merit / RMS / EFL /
  BFL).
- **Interferometry dock** (`interferometry_dock.py`): Twyman-Green
  fringe simulator plus N-step phase-shift extraction with a
  measured-vs-truth residual plot.  Hardware / library sign
  conventions selectable.
- **Phase-retrieval dock** (`phase_retrieval_dock.py`): Gerchberg-
  Saxton + error-reduction runner with four target presets
  (Gaussian / top-hat / ring / Dammann grid), custom image loader,
  and convergence-history plot.
- **Field browser dock** (`field_browser_dock.py`): lists every
  saved plane in an HDF5/Zarr file, previews intensity + phase,
  and routes the selected plane into the Zernike, Interferometry,
  or PSF/MTF docks with one click.
- **Multi-Config dock** (`multiconfig_dock.py`): clones the current
  system into multiple configurations and drives
  `MultiPrescriptionParameterization` for joint optimisation
  (zoom steps, day/night, laser/imaging modes, ...).
- **Materials dock** (`materials_dock.py`): tabbed container that
  unifies the Glass Map (Abbe diagram) and User Library into a
  single entry point.  Original docks remain addressable from the
  View menu.

### Added — surface-form editors (`surface_editors.py`)

- **Asphere editor**: even-power coefficients up to r^20 with a
  live sag-profile preview.
- **Biconic editor**: Ry, ky overrides for anamorphic surfaces.
- **Freeform editor**: XY polynomial sag (i+j <= N) with grid UI.
- **Coating editor**: broadband/narrowband AR-coat model with
  wavelength range and target reflectance.

All four are reachable from the right-click context menu on the
surface sub-table.

### Added — optimizer upgrades

- Merit combos now include **ChromaticFocalShiftMerit**, **Match
  Ideal System** (via `MatchIdealSystemMerit.single_lens`),
  **Tolerance-aware** wrapper.
- **Wavelength / field weight editors** with photopic and cos^4
  presets.
- **Convergence plot** (merit vs iteration) drawn live inside the
  Optimizer dock; log-scale y-axis, auto-rescales.
- **Wave-merit gating**: selecting a wave merit and pressing
  "Local Optimize" redirects to the Wave Optimize path so the
  merit is actually honoured.

### Added — tolerance live histogram

- The Tolerance dock now redraws its RMS / EFL histograms every N
  trials (N = max(1, trials/40)) so the distribution forms live
  and the user can stop early once it looks converged.

### Added — snapshots compare

- "Compare selected to current" button: pops a side-by-side
  EFL / BFL / f-number delta table.

### Added — report export

- **Analysis -> Export design report (HTML)...**: one-page
  self-contained HTML with layout, spot diagram, ray-fan, Zernike
  plot, prescription table, and EFL / BFL / EPD / wavelength
  summary.  Images are embedded as base64 PNGs; file is shareable
  without separate assets.

### Added — preferences

- **Units menu** (SI vs Engineering).
- **Auto-retrace menu** (on / geometric-only / manual).
- **Error-routing policy dialog**: modal-on-error and
  status-bar-on-warn toggles go live on the diagnostics sink.

### Added — keyboard nudge

- **Shift+Up / Shift+Down** nudges the selected element's distance
  by +/-0.1 mm; **Ctrl+Shift+Up / Down** by +/-1 mm.  Works from
  anywhere in the window -- no need to click into the cell first.
  Undo-safe.

### Added — Thorlabs "find nearest part"

- **Insert -> Find nearest Thorlabs part**: ranks every catalog
  part by |dEFL| to the current system's paraxial EFL and shows
  the top 20.

### Added — empty-state CTAs (fix invisible-dependency traps)

- **Sliders dock**: when no variables are defined, shows a centred
  placeholder with a "Define variables..." button that opens the
  Optimizer's picker dialog directly.
- **Zernike dock**: adds a "From ray trace" button for fast
  geometric decomposition without requiring a prior wave-optics run.

### Changed

- **Analysis menu**: every analysis is now a "raise dock" shortcut
  rather than a dialog.  The old "Through-focus scan... future
  version" placeholder is gone.
- **Snapshots** now store the prescription alongside the state,
  enabling the new Compare workflow.
- **Wave-optics completion** auto-pushes the exit-pupil field into
  the Through-focus and PSF/MTF docks (was Zernike-only).

### Fixed

- **ProgressScaler signature**: the scaler now accepts both
  `(frac, msg)` and `(stage, frac, msg)` forms; previously being
  used as a `progress=` kwarg silently swallowed a TypeError, so
  the amp-phase progress inside `apply_real_lens_traced` was
  invisible in the UI.  Restored.

## [3.1.0] — 2026-04-16

Big usability pass driven by a UX deep-dive: undo/redo, snapshots,
diagnostics, autosave, prominent run forecast, optimizer checkbox grid,
Thorlabs catalog grouping, kerboard shortcuts, and many smaller
improvements.

### Added

#### Undo / Redo
- **Ctrl+Z / Ctrl+Y** at the window level; toolbar buttons too.
- Snapshot stack of depth 80 keeps the last ~80 mutations.  Loading a
  snapshot, importing a prescription, or running an optimizer all
  push to the stack so nothing is irrecoverable.

#### Snapshots dock
- Save the current design under a user-chosen name (Ctrl+B or the
  Snapshot button on the slider dock).
- Double-click a snapshot in the dock to restore it; restoring is
  itself undoable.
- A/B-comparison workflow without leaving the app.

#### Diagnostics dock + status-bar badge
- Replaces the scatter of silent ``except: pass`` blocks with a single
  log sink.
- Status-bar badge shows ``diag: ok`` (green) / ``diag: N new \u25CF`` (red)
  and clicking it raises the Diagnostics dock.
- The dock keeps a rolling 500-entry log with timestamps and tags.

#### Autosave + session restore
- Every system change writes ``~/.lumenairy/last_session.json``
  (debounced 1 s) so an accidental close doesn't lose the design.
- ``Edit \u2192 Restore Last Session`` brings it back; loading restores the
  full element list, source, wavelengths, EPD, and field angles.

#### Native JSON design format
- ``File \u2192 Save Design (JSON)`` (Ctrl+S) and ``File \u2192 Open Design (JSON)``.
- Self-contained, version-controlled, diffable — better than ``.zmx`` for
  shareable archived designs.
- ``File \u2192 Export Python Sim Script`` writes a runnable script via the
  new core ``codegen`` module.

#### Wave-optics dock overhaul
- **Lens-model selector**: choose between the inline ASM phase-screen
  pipeline (default, fastest), ``apply_real_lens`` (analytic, supports
  Fresnel + absorption), or ``apply_real_lens_traced`` (sub-nm OPD,
  ~10\u201330\u00d7 slower).  Routes the simulation through the chosen core
  function with full progress reporting.
- **Always-visible run forecast** strip above the Run button: lens
  model, grid, peak memory, estimated wall-clock time, disk size, with
  a colored ``[ok / HEADS-UP / CHECK BEFORE RUN]`` tag.
- **Recalibrated time/memory model** that finally agrees with measured
  runtimes for the v3.x ``apply_real_lens`` and ``apply_real_lens_traced``
  paths (calibration table in the new ``forecast_resources`` docstring).
- **Determinate progress bar** driven by the new core progress hooks:
  the bar smoothly advances through amplitude pass, ray-trace, Newton
  inversion, and field assembly.
- **Save planes: ON/OFF** segmented button next to Run \u2014 promoted
  from a buried checkbox so accidental large-N saves are harder to
  trigger.
- **Field-angle X/Y** now actually drive the source: a linear phase
  ramp ``exp(i (k_x X + k_y Y))`` is applied for every source type so
  off-axis simulations finally produce off-axis spots.

#### Optimizer dock
- **Checkbox grid dialog** replaces the two ``QInputDialog`` popups.
  One screen shows every (element, surface, parameter) with current
  values; tick what should be free, hit OK.  Bulk "Free all radii" /
  "Free all thicknesses" / "Clear all" shortcuts.
- **Wave-optimize free-variable mapping fix**: the old code mapped
  every UI ``distance`` variable to ``thicknesses[0]`` of the
  prescription regardless of which element it belonged to.  Now
  computes the correct flat thickness index per element.
- Variables and merit progress reported through the diagnostics sink
  on failure.

#### Slider dock
- **Per-slider \u00b1 range selector** (\u00b15 / 10 / 20 / 50 % for radii
  and thicknesses, \u00b10.2 / 0.5 / 1 / 2 for conics).  Pick the precision
  appropriate to each variable instead of a fixed \u00b150 %.
- **Live readout** now shows EFL, BFL, and f/# alongside the merit so
  the impact of a slider drag is visible without switching docks.
- **Snapshot button** in the toolbar saves the current parameter
  state under a name.
- Merit recomputation **debounced to 80 ms** so dragging large systems
  doesn't lock up the UI.
- ``opt_variables`` 3-tuple format is finally honoured (the old code
  unpacked them as 2-tuples and would crash when the optimizer dock
  produced ``distance`` variables).

#### Tolerance dock
- **Decenter tolerance is now actually applied** \u2014 the spinbox value
  was previously collected and discarded.  Lateral bundle offsets in
  X and Y are sampled at every trial.
- Per-trial failures route to the diagnostics sink so you can see why
  a Monte Carlo run lost trials.
- Anamorphic ``radius_y`` perturbed alongside ``radius`` (rather than
  silently snapping back to rotational symmetry).

#### Element table
- **Selection banner** above the surface detail panel makes it
  unambiguous which element\u2019s surfaces you\u2019re editing.
- **Throughput / stale indicator** in the info bar: ``[OK \u2713]`` /
  ``[STALE \u25CF]`` plus rays-alive / vignetting counts from the latest
  trace.
- **Wavelength and EPD** now use ``QDoubleSpinBox`` with debounced
  apply (no more "did my edit take?" wondering).
- **Coordinate-mode toggle** is now an unambiguous "Absolute
  coordinates [\u25A1]" checkbox-style button.
- **Group / Ungroup / Delete** buttons disable themselves when the
  current selection can\u2019t support the action.
- **Right-click on a surface** \u2192 *Propagate glass to all cemented
  faces* (handy for fixing imported doublets) or *Copy surface info to
  clipboard*.

#### Source panel
- Hides parameter rows that aren\u2019t used by the chosen source type
  (a plane wave no longer shows emitter-array fields).
- Field-angle X/Y inputs are always visible because they apply to all
  source types.
- Edits debounced via a 200 ms timer instead of firing on every
  keystroke.

#### Element insert dialogs
- Two-tier dialog with an **Advanced...** expander.  Quick path is
  unchanged (focal length + distance); advanced lets you override
  glass index, center thickness, and semi-diameter.
- **Repeat last insert** action (Ctrl+R) re-fires whichever insert
  dialog you used last.
- **Cylindrical lens** inserter now asks which axis carries the
  curvature (the old version hard-coded X).

#### Thorlabs catalog menu
- Grouped by family (LA, LB, AC, ACT, ...) with each entry labeled
  ``part   (f \u2248 NN mm)``.  Items within a family are sorted by
  focal length so you find lenses by ``f`` rather than by part number.

#### Layout views
- 2D and 3D layouts now call the core ``surface_sag_biconic`` instead
  of a hand-rolled paraxial sphere.  Conic, polynomial-aspheric, and
  biconic-Y contributions show up correctly and cannot drift from the
  ray tracer.
- 2D layout shows an empty-state hint card on first launch with
  "Insert \u2192 Lens \u2192 Plano-Convex Singlet" and shortcut tips.

#### Glass map dock
- Click-to-select a glass on the Abbe diagram, choose a target
  surface from a combo, click **Apply**.  No more guessing whether
  the click did anything.

#### Ray fan dock
- Field-curvature (the slow 21-field sweep) moved to an explicit
  **Compute** button so editing a system doesn\u2019t restart the sweep
  on every change.
- Standard matplotlib navigation toolbar (pan, zoom, save-PNG)
  added above the canvas.

#### Zernike dock
- New ``set_field`` entry point: pass a complex field and the dock
  runs ``wave_opd_2d`` with proper unwrap + reference-sphere
  subtraction before the decomposition.
- Defensive **wrap detector**: if the supplied OPD looks like raw
  wrapped phase (PV \u2264 2\u03c0) the dock prints a warning instead of
  fitting noise.
- Wave-optics dock auto-populates Zernike after a run.

#### Analysis menu
- Through-focus scan, chromatic focal shift, ray-traced Zernikes,
  spot-PNG export, Python-script export, and "Run Wave Optics now (F5)".

#### Toolbar
- Undo / Redo, Insert Lens, Insert Mirror, Run Wave Optics, Optimize
  added to the main toolbar in addition to New / Open / Retrace /
  Fit View.

#### Keyboard shortcuts
- Ctrl+Z / Ctrl+Y      undo / redo
- Ctrl+S               save as JSON
- Ctrl+L               insert plano-convex singlet
- Ctrl+Shift+L         insert achromatic doublet
- Ctrl+M               insert flat mirror
- Ctrl+R               repeat last insert
- Ctrl+D               delete element
- Ctrl+E               focus element table
- Alt+\u2191 / Alt+\u2193        move element up / down
- Ctrl+B               save snapshot
- Ctrl+T               retrace
- F5                   run wave optics
- ``Help \u2192 Keyboard Shortcuts`` lists them all.

### Changed

- All docks are now floatable: drag the title bar to pop a panel out
  into a separate window (useful on multi-monitor setups).
- **Cores stay byte-identical between the standalone library and the
  UI distribution.**  All UI changes live in ``lumenairy/ui/``;
  shared core changes (progress hooks, system element types, etc.) are
  applied to both copies.

### Fixed

- ``_populate_zernike_from_waveoptics`` no longer fails silently if the
  saved planes were renamed: falls back to the highest-z plane.
- Tolerance perturbations preserve ``radius_y``, ``conic_y``, and
  ``aspheric_coeffs`` instead of dropping them.
- Slider dock no longer misinterprets ``opt_variables`` 3-tuples
  (radius / thickness / distance / conic now all map correctly).

---

## [3.0.0] — 2026-04-16

Major release: wave-optics optimization, Zernike analysis panel,
anamorphic element support, expanded merit functions.

### Added

#### Zernike dock (`zernike_dock.py`, new)
- OSA-indexed Zernike decomposition of the exit wavefront.
- Interactive matplotlib bar chart of coefficients.
- Text summary with mode names and values.
- Configurable number of modes.

#### Wave-optics optimizer
- **Wave Optimize button** in the optimizer dock — runs
  `design_optimize` from the core library with wave-based merit terms.
- Wave merit types: Strehl ratio, RMS wavefront, chromatic focal shift,
  spot size.
- Global optimization methods: differential evolution, basin-hopping,
  dual-annealing (in addition to existing Nelder-Mead local).

#### Geometric merit types (expanded)
- RMS spot (existing).
- EFL target, BFL target — optimize toward a specific focal length.
- Seidel spherical — minimize third-order spherical aberration.
- Minimum thickness — constraint to prevent unphysical designs.
- Maximum f/# — constraint on system speed.

#### Anamorphic element support
- **Insert > Lens > Cylindrical Lens** — single-axis focusing.
- **Insert > Lens > Biconic Singlet** — independent x/y curvatures.
- **Radius Y** and **Conic Y** columns in the surface table for
  biconic surfaces.
- All existing analysis panels (spot diagram, ray fan, OPD, system
  data) handle biconic surfaces transparently.

#### Source configuration
- **Field angle X/Y** inputs on the source configuration panel for
  off-axis field point analysis.

### Changed

- **Core library updated to v3.0.0** — all new modules (vector
  diffraction, partial coherence, coatings, interferometry, freeform
  surfaces, ghost analysis, RCWA, multi-config, phase retrieval,
  through-focus/tolerancing, design optimizer) are available from the
  GUI's Python console and wave-optics dock.
- **`apply_real_lens_traced` exit-vertex fix** — the hybrid wave/ray
  lens model now focuses correctly for all lens geometries including
  cemented doublets and negative meniscus lenses.
- Removed organization name from application settings.

### Fixed

- Wave-optics dock correctly uses `apply_real_lens_traced` with the
  signed exit-vertex correction for all rear surface geometries.

---

## [2.5.0] — Prior release

- Initial GUI release.
- Element-based prescription editor with distance/tilt/decenter.
- 2D cross-section and 3D PyVista layout views.
- Spot diagram with Airy disc overlay.
- Ray fan / OPD analysis dock.
- System data (ABCD, EFL, BFL, f/#, NA).
- Glass map (interactive Abbe diagram).
- Local + global geometric optimizer.
- Live parameter sliders.
- Monte Carlo tolerance analysis.
- Wave-optics simulation panel (ASM, Fresnel, Fraunhofer, RS).
- HDF5 / Zarr output.
- Insert menu: plano-convex, biconvex, achromatic doublet, mirrors,
  DOEs, Thorlabs catalog lenses.
- Dark / Light / Midnight Blue themes.
- User library (materials, lenses, phase masks).
