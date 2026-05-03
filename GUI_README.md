# Optical Designer — GUI Application

A PySide6 desktop application for interactive optical system design, analysis,
and optimization.  Built on the `lumenairy` library.

**Version:** 3.2.14
**Author:** Andrew Traverso

Versioning note: the GUI application used to be released under its
own track (3.2.x during the 2026-04-16 feature-gap pass) but this was
found to be confusing alongside the bundled core library's separate
3.1.x versioning.  Starting 2026-04-17 the GUI distribution tracks
the same version number as the core `lumenairy` library,
so whatever `from lumenairy import __version__` returns is
also the version of the GUI bundled with it.  GUI-specific changes
continue to be tracked in `GUI_CHANGELOG.md`, now keyed on the same
versions as `CHANGELOG.md`.

## Major UI features added in 3.2.10–3.2.14

The 3.2.10–3.2.14 sequence reorganised the GUI around topic-based
**workspace tabs** at the top of the window, so the 27+ analysis
docks no longer all crowd the screen at once.

### Workspaces (3.2.10–3.2.11)
- Top-of-window tab strip groups docks by topic: **Design**,
  **Optimize**, **Analysis**, **Wave Optics**, **Tolerancing**,
  **Materials**.
- Switching tabs hides docks not in that workspace; geometry within
  a workspace is preserved on switch.
- Right-click a tab for **Manage Docks / Rename / Duplicate /
  Delete**; double-click to rename; `+` button to add a new
  workspace.
- **`View > Workspace > Pin Docks Across All Workspaces…`** marks
  docks (e.g. System Data, Element Table) as always visible.
- **Export / Import** the workspace set as a `.workspace` JSON
  file to share custom layouts between machines.
- Persists to `QSettings('lumenairy', 'OpticalDesigner')`; an
  automatic migration appends new default workspaces on upgrade
  without overwriting customisations.

### Productivity polish (3.2.12)
- **`Ctrl+1` … `Ctrl+9`** — jump directly to workspace tabs.
- **Window title** shows current file + dirty marker
  (`Optical Designer — file.zmx*`).
- **Drag-and-drop** any `.zmx`, `.txt`, `.seq`, or `.json` onto
  the window to load.
- **Permanent right-aligned status bar metrics**: EFL, BFL, f/#,
  EPD, λ — visible on every workspace.
- **Welcome dock** — empty-state landing panel with quick-start
  buttons (Open Demo, Insert Singlet, Insert Achromat, Browse
  Library) and a recent-files list backed by `QSettings`.
- **Embedded Python REPL dock** — live Python console with
  `model`, `np`, `plt`, `result`, `wave` pre-bound to the current
  system, latest geometric trace, and latest wave-optics result.
- **Optimizer progress** in the status bar plus an animated badge
  on the Optimize tab title (`Optimize • iter N`) while running.
- **Element-table polish**: right-click row context menu (Insert /
  Duplicate / Delete / Toggle Distance Variable), amber highlight
  on cells with optimization variables, search box that hides
  rows whose Name doesn't match.
- **F11 / Compact Mode** — hides menu bar and dock title bars to
  maximise working area on laptops.

### Performance (3.2.14, applies to GUI workflow)
- ASM transfer-function `H` cache: 1.5× speedup on repeat
  propagations at the same geometry (multi-config sweeps,
  optimization loops, multi-wavelength pre-views).
- Multi-slot pyFFTW plan cache: no more thrashing when calls
  oscillate between two grid shapes.
- `set_default_complex_dtype(np.complex64)` toggle: 2.18× FFT
  throughput, 2× memory headroom; explicit kernel-phase mod-2π
  folding keeps single-precision ASM accurate.
- `JonesField.propagate` runs Ex/Ey through a single batched FFT
  pair on grids ≥ 512.
- Numba-fused aspheric-coefficient sum in `surface_sag_general`:
  4.75× speedup at N=2048 with 5 aspheric coeffs.
- `apply_real_lens` refraction step memory-leaner (peak transient
  at N=8192 drops from ~5 GB to ~1.5 GB).

### Validation (3.2.13)
The bundled validation suite was expanded by ~70 new physics +
interop test cases.  16 files, 298 Harness assertions, all PASS
on every release in the 3.2.x line.

## What's new in 3.1.6

- Core library bugfix only: zarr storage reliability on Windows +
  Python 3.14 (``open_group(mode='a')`` no longer crashes when the
  store directory already exists).  No GUI-facing changes.  See
  `CHANGELOG.md` for the root cause.

## What's new in 3.1.5

- Core library bugfix only: `.zmx` / Zemax prescription-text loaders
  now preserve the obj-space distance dropped by non-refractive
  dummy surfaces between the STOP and the first active surface
  (returned as `prescription['object_distance']`).  No GUI-facing
  changes — wave-optics previews and Load-Prescription consumers
  pick up the loader correction automatically.  See `CHANGELOG.md`
  for the full rationale.

## What's new in 3.1.4

- Wave-optics dock's `Tilt-aware ray launch` checkbox now **defaults
  to off** (matching the core library's new `tilt_aware_rays=False`
  default).  This removes a reference-frame mismatch that produced
  wrong output fields on multi-mode inputs (post-DOE diffraction
  patterns); the plane-wave launch is reference-consistent for any
  input the wave model can represent.  Single-mode tilted inputs
  (rigorous off-axis characterisation) can still opt in by ticking
  the box explicitly.
- Core library upgrade includes a better Newton initial guess
  (measured paraxial magnification from the already-computed forward
  trace) and an experimental `inversion_method='backward_trace'`
  opt-in for ~3x speedup on large grids.
- See `GUI_CHANGELOG.md` and `CHANGELOG.md` for details.

## What's new in 3.1.3

## What's new in 3.2 (feature-gap pass, 2026-04-16)

Seven new analysis docks wire every high-leverage core-library capability
into the UI so the tool is usable for real design reviews without dropping
to Python:

- **Through-focus dock** — axial Strehl / peak / RMS / d4sigma plots
  with determinate progress, best-focus marker, CSV export.
- **PSF / MTF dock** — log-scaled PSF + radial MTF, polychromatic-
  Strehl across the Optimizer's wavelength list.
- **Sensitivity dock** — ranked per-variable d(merit)/d(var) bar
  chart with a selectable output metric.
- **Interferometry dock** — Twyman-Green fringe simulator + N-step
  phase-shift extraction with a residual plot.
- **Phase-retrieval dock** — Gerchberg-Saxton + error-reduction
  with four target presets and a custom image loader.
- **Field browser dock** — browse saved HDF5/Zarr planes and route
  any of them into the Zernike / Interferometry / PSF-MTF docks.
- **Multi-Config dock** — joint optimisation across multiple
  configurations via `MultiPrescriptionParameterization`.
- **Materials dock** — Glass Map + User Library unified.

Four new surface-form editors (right-click any surface):
- Asphere (even-power coefficients up to r^20, live sag preview)
- Biconic (Ry, ky anamorphic overrides)
- Freeform (XY polynomial sag, i+j <= N)
- AR coating (broadband/narrowband, wavelength range + R target)

Optimizer upgrades:
- `ChromaticFocalShiftMerit`, `MatchIdealSystemMerit`,
  `ToleranceAwareMerit` in the merit combos.
- Wavelength / field **weight editors** (photopic + cos^4 presets).
- Live **convergence plot** (merit vs iteration) inside the dock.
- Wave-merit **gating**: Local Optimize reroutes to Wave Optimize
  when a wave merit is selected.

One-page **HTML report export** from Analysis -> Export design report.

**Preferences** menu: unit system (engineering / SI), auto-retrace
mode (on / geometric-only / manual), error-routing policy.

**Keyboard nudge**: Shift+Up / Shift+Down bumps the selected
element's distance by +/-0.1 mm (+/-1 mm with Ctrl+Shift).

**Thorlabs "find nearest part"**: ranks every catalog part by dEFL
to the current system.

## What's new in 3.1 (UX deep-dive)

- **Undo / Redo** — Ctrl+Z / Ctrl+Y; depth 80; every mutation is
  recoverable.
- **Snapshots dock** — name-and-save designs for A/B comparison
  (Ctrl+B); double-click to restore, restore is itself undoable.
- **Diagnostics dock + status-bar badge** — single sink for
  non-fatal errors that previously disappeared into ``except: pass``.
  Badge flashes red when there are unread errors.
- **Autosave + session restore** — current design auto-saves to
  ``~/.lumenairy/last_session.json``; ``Edit \u2192 Restore Last
  Session`` brings it back after a crash or accidental close.
- **Native JSON design format** — ``File \u2192 Save Design`` (Ctrl+S)
  writes a self-contained JSON; ``Open Design`` reads it back.
- **Wave-optics overhaul** — lens-model selector (ASM phase-screen /
  apply_real_lens / apply_real_lens_traced), recalibrated time/memory
  forecast that finally matches measured runtimes for the v3.x
  ``apply_real_lens_traced`` path, always-visible run-forecast strip,
  determinate progress bar driven by the new core progress hooks.
- **Optimizer checkbox grid** — pick every optimization variable
  in one screen with current values shown; bulk "free all radii" /
  "free all thicknesses" shortcuts.  Wave-optimize free-variable
  mapping bug fixed.
- **Slider dock** — per-slider \u00b1 range selector (\u00b15 / 10 / 20 / 50
  %), live EFL / BFL / f/# alongside the merit, snapshot button.
- **Tolerance dock** — decenter spinbox is now actually applied
  (previously discarded).
- **Element table** — selection banner above the surface detail panel,
  STALE / OK / vignetting indicator in the info bar, debounced
  ``QDoubleSpinBox``-based wavelength + EPD inputs, unambiguous
  "Absolute coordinates" toggle, Group / Ungroup / Delete buttons
  enable themselves only when applicable, right-click on a surface
  to propagate glass to cemented faces.
- **Source panel** — irrelevant rows hidden per source type; field
  angles X/Y always visible; debounced apply.
- **Insert dialogs** — two-tier with an "Advanced..." expander for
  glass / thickness / semi-diameter overrides; cylindrical lens asks
  which axis carries curvature.
- **Repeat last insert** (Ctrl+R).
- **Thorlabs catalog grouped by family + EFL** — find lenses by
  focal length instead of part number.
- **Layout views use core sag** — 2D and 3D delegate to
  ``surface_sag_biconic`` so conic / aspheric / biconic-Y show up
  correctly.
- **Empty-state hint card** in the 2D layout on first launch.
- **Glass map dock** — explicit "Apply selected glass to: [surface]"
  control.
- **Ray fan dock** — field curvature moved to an on-demand "Compute"
  button; matplotlib navigation toolbar (pan/zoom/save-PNG).
- **Zernike dock** — ``set_field()`` API uses ``wave_opd_2d`` for
  proper unwrap + reference-sphere subtraction; wrap detector warns
  if the input looks like raw phase.
- **Analysis menu** populated: through-focus, chromatic shift,
  ray-traced Zernikes, spot PNG export, Python script export, F5 to
  run wave optics.
- **Toolbar** — undo, redo, insert lens, insert mirror, run wave
  optics, optimize.
- **Floatable docks** — drag any title bar to pop a panel out into a
  separate window (multi-monitor friendly).

## What's new in 3.0 (GUI-relevant)

- **Zernike Dock** — decompose the exit wavefront into OSA-indexed Zernike
  coefficients with an interactive bar chart and text summary.
- **Insert > Cylindrical Lens, Biconic Singlet** — new menu items for
  anamorphic elements (independent x/y radii and conics).
- **Radius Y / Conic Y columns** in the surface table for biconic surfaces.
- **Wave Optimize button** — runs the hybrid wave/ray `design_optimize`
  from the optimizer dock (Strehl, RMS wavefront, chromatic focal shift
  merit terms available alongside the existing geometric merits).
- **Geometric merit types** expanded: RMS spot, EFL target, BFL target,
  Seidel spherical, minimum thickness, maximum f/#.
- **Field angle inputs** (X, Y) on the source configuration panel.
- **Exit-vertex OPL correction** in `apply_real_lens_traced` fixes a
  systematic 43 % defocus error on cemented doublets with curved rear
  surfaces.  Traced doublet focus error: 10 mm -> 0.000 mm.
- **Core library v3.0** includes vector diffraction (Richards-Wolf), partial
  coherence, thin-film coatings, interferometry, freeform surfaces, ghost
  analysis, RCWA, multi-config/afocal mode, 8 new source types, phase
  retrieval, and more.  See `README.md` for full details.

## Quick Start

```bash
cd Optical_Propagation_Library_UI
pip install -e ".[gui]"                  # install with PySide6
python run_optical_designer.py --demo    # launch with demo lens
```

Or install and run as a command:

```bash
pip install -e ".[gui]"
optical-designer --demo
```

## Launch Options

```bash
python run_optical_designer.py                 # empty system
python run_optical_designer.py --demo          # AC254-100-C doublet
python run_optical_designer.py path/to/file.zmx   # open .zmx prescription
python run_optical_designer.py path/to/file.txt   # open prescription text
python run_optical_designer.py path/to/file.json  # open native JSON design
```

## Keyboard Shortcuts

| Action | Shortcut |
|---|---|
| New system | Ctrl+N |
| Open prescription (.zmx / .txt) | Ctrl+O |
| Save design as JSON | Ctrl+S |
| Quit | Ctrl+Q |
| Undo | Ctrl+Z |
| Redo | Ctrl+Y (Ctrl+Shift+Z on macOS) |
| Save snapshot | Ctrl+B |
| Insert plano-convex singlet | Ctrl+L |
| Insert achromatic doublet | Ctrl+Shift+L |
| Insert flat mirror | Ctrl+M |
| Repeat last insert | Ctrl+R |
| Delete element | Ctrl+D |
| Focus element table | Ctrl+E |
| Move selected element up | Alt+\u2191 |
| Move selected element down | Alt+\u2193 |
| Retrace | Ctrl+T |
| Run Wave Optics | F5 |

`Help \u2192 Keyboard Shortcuts` displays this list inside the app.

## Application Layout

```
+---------------------------------------------------------------+
|  Menu bar: File | Insert | Analysis | View | Help             |
|  Toolbar:  New | Open | Retrace | Fit View                   |
+---------------------------+-----------------------------------+
|                           |  [Spot Diagram] [System Data]     |
|  [2D Layout] [3D Layout]  |  [Ray Fan/OPD] [Glass Map]       |
|                           |  [Library]                        |
+---------------------------+-----------------------------------+
|  Element Table (with surface detail panel below)              |
|  Toolbar: Up Down Delete Group Ungroup Relative ElementView   |
+---------------------------+-----------------------------------+
|  [Optimizer] [Sliders] [Tolerance] [Wave Optics]              |
+---------------------------------------------------------------+
|  Status bar                                                   |
+---------------------------------------------------------------+
```

All panels are dockable — drag to rearrange, undock to float, close and
reopen via View menu.  Dock separators highlight on hover for easy resizing.

## Features

### Element-Based Prescription Editor

Each row in the table is an **element** (lens, mirror, DOE), not a raw
surface.  The **Distance** column is the distance from the previous element
to this one — no air-gap confusion.

- **Element view**: one row per optic (Elem#, Name, Type, Distance, Tilt, Decenter)
- **Surface view**: flat Zemax-style table showing all surfaces + air gaps + tilt/decenter
- Toggle **Relative / Absolute** coordinates (column headers change: Distance/Tilt/Decenter vs Z/Rx/Ry/X/Y)
- Click an element to see its internal surfaces in the detail panel
- **Move** elements up/down with arrow buttons (fast, single retrace)
- **Group** multiple elements into a compound optic (smart cemented interface merging)
- **Ungroup** a compound element back into individual elements

### Source Configuration

Click the Source element (row 0) to configure illumination:

| Source type | Parameters |
|---|---|
| Plane wave | Fills the entrance pupil |
| Gaussian beam | Beam diameter (1/e2), NA |
| Gaussian aperture | Soft-edge sigma |
| Point source | Object distance |
| Emitter array | Pitch, NxN, waist diameter |

### Insert Menu

| Category | Elements |
|---|---|
| **Lens** | Plano-convex, biconvex, achromatic doublet, cylindrical lens, biconic singlet (all with distance prompt) |
| **Mirror** | Flat, curved (specify focal length) |
| **DOE** | Microlens array, diffraction grating, Dammann grating |
| **Thorlabs** | All catalog lenses (AC254-050-C, LA1050-C, etc.) |
| **Source** | Plane wave, Gaussian, point source, emitter array |

All insert dialogs ask for the distance from the previous element in the
same form as the element parameters.

### Analysis Panels

| Panel | What it shows |
|---|---|
| **Spot Diagram** | Custom-painted with Airy disc overlay, RMS/GEO readout |
| **Ray Fan / OPD** | Transverse aberration, OPD fans, field curvature (matplotlib) |
| **System Data** | ABCD matrix, EFL, BFL, f/#, NA, trace statistics |
| **Glass Map** | Interactive Abbe diagram — click to select glasses |
| **Zernike** | OSA-indexed Zernike decomposition of the exit wavefront, bar chart + text |

### 2D Layout

Interactive cross-section showing lenses, glass fills, and traced rays.
Click a surface to highlight the corresponding element in the table.
Scroll to zoom, drag to pan.

### 3D Layout

Interactive PyVista/VTK rendering with:
- Orbit, zoom, pan (left/middle/scroll)
- Axes gizmo in the corner
- View snap buttons: Front, Side, Top, Iso
- Curved surface meshes and glass volumes

### Optimizer

- **Variable selection**: pick any element's distance, surface radius, thickness, or conic
- **Geometric merit types**: RMS spot, EFL target, BFL target, Seidel spherical, min thickness, max f/#
- **Wave-optics merit types**: Strehl ratio, RMS wavefront, chromatic focal shift, spot size
- **Local optimization**: Nelder-Mead (geometric) or L-BFGS-B / SLSQP / trust-constr (wave)
- **Global search**: random-restart (geometric), differential evolution, basin-hopping, dual-annealing (wave)
- Multi-wavelength, multi-field merit function
- Background thread with progress log

### Live Sliders

Drag sliders to change optimisation variables in real time.  Spot diagram,
layout, and ABCD readout update live as you drag.  Merit readout shows the
current RMS spot size.

### Tolerance Analysis

Monte Carlo perturbation of radii (%) and thicknesses (mm).  Runs in
background with progress bar.  Shows RMS and EFL distribution histograms
with median, sigma, and 95th percentile.

### Wave Optics

Full coherent propagation through the optical system:

- **Methods**: ASM, Fresnel, Fraunhofer, Rayleigh-Sommerfeld
- **Grid**: 128 to 131072 (FFT-friendly sizes: 2^a, 2^a*3, 2^a*5)
- **Backends**: NumPy FFT, pyFFTW (multi-threaded), CuPy (GPU)
- **Recommend Grid**: auto-sizes N and dx from system NA and aperture
- **Output**: save to HDF5 or Zarr, choose folder + filename separately
- **Plane selection**: checkboxes for which planes to save
- **Execution range**: start/end element dropdowns
- **Don't-save toggle**: run without saving field data (lower memory)
- **Memory limit**: 2 GB to 1 TB
- **Pre-run forecast**: memory, disk, estimated time, with warnings for
  infeasible configurations (memory exceeded, disk full, >24h runtime)
- **PSF display**: log-scale 2D plot + cross-section

### User Library

Persistent storage for materials, lenses, and phase masks across sessions:

```python
# Works from Python scripts too (no GUI needed)
import lumenairy as op

op.save_material('MyPolymer', n=1.52)
op.save_lens('My_PCX', op.make_singlet(...))
op.save_phase_mask('vortex', expression='arctan2(Y, X) * 3')
```

GUI provides a Library dock with browse/add/delete for each category.
Saved materials auto-register in `GLASS_REGISTRY` on import.

### Themes

**View > Theme**: Dark, Light, Midnight Blue

**View > Colors**: 2D background, 3D background, ray color (custom or
wavelength-based), UI highlight color

All customisations apply instantly without restart.

## Dependencies

**Required**: `numpy`, `refractiveindex`

**GUI**: `PySide6 >= 6.5`

**Recommended**: `matplotlib` (plots, wave optics), `scipy` (optimization),
`pyvista` + `pyvistaqt` (interactive 3D)

**Optional**: `pyfftw` (fast FFT), `cupy` (GPU), `h5py` (HDF5 output),
`astropy` (FITS I/O)

```bash
pip install -e ".[gui]"
pip install matplotlib scipy pyvista pyvistaqt pyfftw h5py
```

## Architecture

The GUI is a pure view/controller layer on top of the
`lumenairy` library.  The same library functions are called
by both the GUI and Python scripts:

```
PySide6 Application (ui/)
    |
    v
SystemModel  -->  build_trace_surfaces()  -->  raytrace.trace()
                                           |
    |                                      +--> raytrace.system_abcd()
    v                                      +--> raytrace.spot_rms()
Element table <-- system_changed signal
2D/3D layout  <-- trace_ready signal
Spot diagram  <-- trace_ready signal
```

The library can be used independently of the GUI:

```python
import lumenairy as op

E = op.angular_spectrum_propagate(E_in, z=0.1, wavelength=1.31e-6, dx=2e-6)
result = op.trace_prescription(rx, wavelength=1.31e-6)
op.spot_diagram(result)
```

## Project Layout

```
Optical_Propagation_Library_UI/
    README.md                    # library documentation
    GUI_README.md                # this file — GUI documentation
    LICENSE
    pyproject.toml
    requirements.txt
    run_optical_designer.py      # GUI launch script
    lumenairy/         # core library
        __init__.py              # v3.0.0
        propagation.py           # ASM, Fresnel, Fraunhofer, RS
        raytrace.py              # geometric ray tracer
        lenses.py                # apply_real_lens, apply_real_lens_traced
        optimize.py              # hybrid wave/ray design optimizer
        through_focus.py         # through-focus, tolerancing
        analysis.py              # Zernike, OPD, Strehl, MTF/OTF
        glass.py, elements.py, sources.py, ...
        coatings.py              # thin-film TMM
        interferometry.py        # interferograms, phase-shift extraction
        freeform.py              # XY polynomial, Zernike, Chebyshev surfaces
        ghost.py                 # ghost analysis
        rcwa.py                  # rigorous coupled-wave analysis
        multiconfig.py           # multi-config, afocal mode
        vector_diffraction.py    # Richards-Wolf high-NA
        coherence.py             # partial coherence
        detector.py              # detector model, Shack-Hartmann
        polarization.py          # Jones calculus
        phase_retrieval.py       # Gerchberg-Saxton, HIO
        user_library.py          # persistent material/lens/mask catalog
        ui/                      # GUI application
            model.py             # Element-based system model
            element_table.py     # Prescription editor (Radius Y, Conic Y columns)
            layout_2d.py         # 2D cross-section
            layout_3d.py         # 3D PyVista viewer
            analysis.py          # Spot diagram + system data
            rayfan_dock.py       # Ray fan / OPD / field curvature
            optimizer_dock.py    # Local + global + wave optimization
            slider_dock.py       # Live parameter sliders
            tolerance_dock.py    # Monte Carlo tolerance
            waveoptics_dock.py   # Wave-optics simulation panel
            zernike_dock.py      # Zernike decomposition panel
            glass_map_dock.py    # Abbe diagram
            library_dock.py      # User library browser
            main_window.py       # Application shell + themes
```

## License

MIT License — see `LICENSE` file.
