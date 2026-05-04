"""Real-lens OPD validation.

Compare the wave OPD produced by ``apply_real_lens`` against the
geometric OPL accumulated by the sequential ray tracer in
``lumenairy.raytrace``.  The ray tracer uses exact
vector Snell's law at each surface, so its OPL is the physically
correct reference for any thin-element phase-screen model.

For each lens case we run:
  * ``apply_real_lens(..., slant_correction=True)``   -- new default
  * ``apply_real_lens(..., slant_correction=False)``  -- old paraxial
  * ``raytrace.trace(...)``                           -- ground truth

and plot OPD(h) along a tangential pupil cut (y=0 row).

Four removal levels are produced per lens for completeness:
  * raw               -- nothing subtracted; tests absolute OPL match
  * piston            -- constant offset removed (arbitrary phase ref)
  * piston+tilt       -- linear ramp removed (pupil-centering artifacts)
  * piston+tilt+focus -- quadratic r^2 removed (isolates high-order
                         aberrations like spherical, coma, higher)

Outputs (one PNG per lens + one summary CSV + one summary Markdown)
land in ``results/``.

Usage
-----
    python run_validation.py              # run all cases
    python run_validation.py --quick      # run only a small subset
    python run_validation.py --case plano_convex_R50_BK7
    python run_validation.py --category fnum_sweep

The script is expected to take tens of minutes for the full suite
because a few fast-f/# cases require 16k or 32k grids.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Make the library importable when run as a script from the
# validation/ folder.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
if _LIB_ROOT not in sys.path:
    sys.path.insert(0, _LIB_ROOT)

import lumenairy as op
from lumenairy.lenses import apply_real_lens, apply_real_lens_traced
from lumenairy.analysis import (
    remove_wavefront_modes,
    opd_pv_rms,
    wave_opd_1d,
)
from lumenairy.prescriptions import (
    export_zemax_lens_data,
    export_zemax_zmx,
)
from lumenairy.raytrace import (
    surfaces_from_prescription,
    trace,
    _make_bundle,
    system_abcd,
)

from lens_cases import list_cases, filter_cases  # noqa: E402  (script style)

RESULTS_DIR = os.path.join(_HERE, 'results')
ZEMAX_DIR = os.path.join(_HERE, 'zemax_prescriptions')


# =========================================================================
# OPD extractors
# =========================================================================

def compute_wave_opd(prescription, wavelength, N, dx, aperture,
                     slant_correction):
    """Run a unit plane wave through ``apply_real_lens`` and extract the
    unwrapped OPD along the y=0 row.

    Returns
    -------
    x : ndarray
        Exit-plane x coordinate of each sample, in meters.
    opd : ndarray
        Optical path length (unwrapped phase / k0) at the exit plane.
        Only the in-aperture samples are returned.
    """
    pres = dict(prescription)
    pres['aperture_diameter'] = aperture

    E_in = np.ones((N, N), dtype=np.complex128)

    # apply_real_lens handles the entrance aperture internally when the
    # prescription has 'aperture_diameter'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_out = apply_real_lens(
            E_in, pres, wavelength, dx,
            bandlimit=True,
            slant_correction=slant_correction,
            fresnel=False,
            absorption=False,
        )

    # Crop to 95 % of aperture to avoid the hard-edged vignette, where
    # np.gradient of sag produces noisy normals.
    return wave_opd_1d(E_out, dx, wavelength, axis='x',
                       aperture=0.95 * aperture)


def compute_wave_opd_traced(prescription, wavelength, N, dx, aperture,
                            ray_subsample=4):
    """Wave OPD using the per-pixel ray-traced phase-screen model
    (``apply_real_lens_traced``).  Returns the same (x, opd) tuple as
    :func:`compute_wave_opd`.

    ``ray_subsample=4`` by default -- gives ~15x speedup over
    unsubsampled with < 1 nm fidelity loss on a smooth OPL surface.
    """
    pres = dict(prescription)
    pres['aperture_diameter'] = aperture
    E_in = np.ones((N, N), dtype=np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_out = apply_real_lens_traced(
            E_in, pres, wavelength, dx, bandlimit=True,
            ray_subsample=ray_subsample)
    # Use aperture=None so wave_opd_1d auto-crops to the ray-coverage
    # region (the actual exit pupil for the traced method may be
    # smaller than the entrance aperture for fast lenses).
    return wave_opd_1d(E_out, dx, wavelength, axis='x', aperture=None)


def compute_geometric_opd(prescription, wavelength, aperture, n_rays=401):
    """Trace a collimated, on-axis fan through the prescription and return
    (h_exit, OPL) at the last surface.

    Parameters
    ----------
    prescription : dict
    wavelength : float
    aperture : float
        Entrance aperture diameter [m].  Rays are sampled across this.
    n_rays : int
        Number of rays in the tangential (x) fan.

    Returns
    -------
    h_exit : ndarray
        Exit-plane x-coordinate of each surviving ray, sorted ascending.
    opl : ndarray
        OPL from input plane to exit plane for each ray.  Same length.
    """
    surfaces = surfaces_from_prescription(prescription)

    r_max = aperture / 2.0
    h_in = np.linspace(-0.95 * r_max, 0.95 * r_max, n_rays)
    z = np.zeros_like(h_in)
    rays = _make_bundle(
        x=h_in, y=z, L=z, M=z, wavelength=wavelength,
    )
    result = trace(rays, surfaces, wavelength)
    final = result.image_rays

    # Exit-vertex correction: trace() leaves rays at z = sag of the
    # last surface, but the wave model's exit field is at the flat
    # vertex plane z = 0.  Transfer each ray from sag to vertex in
    # the exit medium so OPL references a common flat plane.
    from lumenairy.glass import get_glass_index as _ggi
    n_exit = _ggi(surfaces[-1].glass_after, wavelength)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_vtx = np.where(
            final.alive & (np.abs(final.N) > 1e-30),
            -final.z / final.N, 0.0)
    final.opd = final.opd + n_exit * t_vtx
    final.x = final.x + final.L * t_vtx
    final.y = final.y + final.M * t_vtx
    final.z = np.zeros_like(final.z)

    alive = final.alive
    h_exit = final.x[alive]
    opl = final.opd[alive]

    sort_idx = np.argsort(h_exit)
    return h_exit[sort_idx], opl[sort_idx]


def paraxial_efl(prescription, wavelength):
    """Paraxial effective focal length from the system ABCD matrix."""
    surfaces = surfaces_from_prescription(prescription)
    # system_abcd returns (matrix, efl, bfl, ffl)
    _, efl, _, _ = system_abcd(surfaces, wavelength)
    return efl


# =========================================================================
# Per-case driver
# =========================================================================

def run_case(case, verbose=True):
    name = case['name']
    lam = case['wavelength']
    ap = case['aperture']
    N = case['N']
    dx = case['dx']

    if verbose:
        print(f"--- {name} ---")
        print(f"    {case['description']}")
        print(f"    aperture={ap*1e3:.2f} mm, lam={lam*1e9:.0f} nm, "
              f"N={N}, dx={dx*1e6:.2f} um, "
              f"extent={N*dx*1e3:.2f} mm")

    t0 = time.time()

    # Geometric (truth)
    h_geom, opl_geom = compute_geometric_opd(
        case['prescription'], lam, ap, n_rays=401)
    t_geom = time.time() - t0

    # Paraxial EFL (informational)
    f_nom = paraxial_efl(case['prescription'], lam)

    # Wave OPD -- analytic phase screen, slant-corrected
    t1 = time.time()
    x_wave, opl_slant = compute_wave_opd(
        case['prescription'], lam, N, dx, ap, slant_correction=True)
    t_slant = time.time() - t1

    # Wave OPD -- analytic phase screen, paraxial
    t1 = time.time()
    _, opl_par = compute_wave_opd(
        case['prescription'], lam, N, dx, ap, slant_correction=False)
    t_par = time.time() - t1

    # Wave OPD -- per-pixel ray-traced phase screen
    t1 = time.time()
    try:
        x_traced, opl_traced_raw = compute_wave_opd_traced(
            case['prescription'], lam, N, dx, ap)
        # Traced x grid may differ from analytic x grid (different exit
        # pupil due to ray bending); interpolate onto x_wave for
        # consistent comparison.
        opl_traced = np.interp(x_wave, x_traced, opl_traced_raw,
                               left=np.nan, right=np.nan)
    except Exception as e:
        print(f"    apply_real_lens_traced failed: {type(e).__name__}: {e}")
        opl_traced = np.full_like(opl_slant, np.nan)
    t_traced = time.time() - t1

    # Interpolate geometric truth onto the wave's exit-plane grid
    opl_geom_on_wave = np.interp(
        x_wave, h_geom, opl_geom, left=np.nan, right=np.nan)

    # Residuals in nm for each method
    methods = {
        'paraxial':   opl_par - opl_geom_on_wave,
        'slant':      opl_slant - opl_geom_on_wave,
        'ray-traced': opl_traced - opl_geom_on_wave,
    }
    residuals_nm = {k: v * 1e9 for k, v in methods.items()}

    removal_levels = [
        ('raw', None),
        ('piston', 'piston'),
        ('piston+tilt', 'piston,tilt'),
        ('piston+tilt+defocus', 'piston,tilt,defocus'),
    ]

    summaries = {lvl: {} for lvl, _ in removal_levels}
    for level_name, modes in removal_levels:
        for method_name, res_nm in residuals_nm.items():
            r, coeffs = remove_wavefront_modes(x_wave, res_nm, modes)
            pv, rms = opd_pv_rms(r)
            summaries[level_name][method_name] = {
                'pv_nm': pv, 'rms_nm': rms,
                'residual': r, 'coeffs': coeffs,
            }

    # Exit-plane defocus coefficient of each method, in nm/mm^2
    defocus_bias = {}
    for method_name in residuals_nm:
        c = summaries['piston+tilt+defocus'][method_name]['coeffs'] \
                 .get('defocus', np.nan)
        defocus_bias[method_name] = (c * 1e-6
                                     if np.isfinite(c) else np.nan)

    if verbose:
        print(f"    t_geom={t_geom:.2f}s  t_paraxial={t_par:.2f}s  "
              f"t_slant={t_slant:.2f}s  t_traced={t_traced:.2f}s")
        print(f"    paraxial EFL = {f_nom*1e3:.2f} mm")
        print(f"    Exit-plane defocus bias [nm/mm^2]: "
              f"paraxial={defocus_bias['paraxial']:+.3f}  "
              f"slant={defocus_bias['slant']:+.3f}  "
              f"traced={defocus_bias['ray-traced']:+.3f}")
        ptf = summaries['piston+tilt+defocus']
        print(f"    RMS<sub>pt+focus</sub> [nm]:  "
              f"paraxial={ptf['paraxial']['rms_nm']:8.2f}  "
              f"slant={ptf['slant']['rms_nm']:8.2f}  "
              f"traced={ptf['ray-traced']['rms_nm']:8.2f}  "
              f"<-- physics-check")

    return {
        'case': case,
        'f_nom': f_nom,
        'defocus_bias': defocus_bias,
        'x_wave': x_wave,
        'opl_par': opl_par,
        'opl_slant': opl_slant,
        'opl_traced': opl_traced,
        'h_geom': h_geom,
        'opl_geom': opl_geom,
        'opl_geom_on_wave': opl_geom_on_wave,
        'residuals_nm': residuals_nm,
        'summaries': summaries,
        'times': {'geom': t_geom, 'paraxial': t_par,
                  'slant': t_slant, 'traced': t_traced},
    }


# =========================================================================
# Plotting
# =========================================================================

def plot_case(result, out_path):
    case = result['case']
    x_mm = result['x_wave'] * 1e3
    h_geom_mm = result['h_geom'] * 1e3
    lam = case['wavelength']

    method_styles = {
        'paraxial':   {'color': 'C3', 'ls': '--', 'lw': 1.0,
                       'label_long': 'apply_real_lens (paraxial screen)'},
        'slant':      {'color': 'C0', 'ls': '-',  'lw': 1.0,
                       'label_long': 'apply_real_lens (slant-corrected)'},
        'ray-traced': {'color': 'C2', 'ls': '-',  'lw': 1.4,
                       'label_long': 'apply_real_lens_traced (per-pixel ray-traced phase)'},
    }

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # --- Top row: absolute OPL curves ---------------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.plot(h_geom_mm, result['opl_geom'] * 1e6, 'k-', lw=2.5,
            label='Geometric ray trace (truth)')
    for key, opl_key in [('paraxial', 'opl_par'),
                         ('slant', 'opl_slant'),
                         ('ray-traced', 'opl_traced')]:
        st = method_styles[key]
        ax.plot(x_mm, result[opl_key] * 1e6,
                color=st['color'], ls=st['ls'], lw=st['lw'],
                label=st['label_long'])
    ax.set_xlabel('exit-plane x [mm]')
    ax.set_ylabel('OPL [um]')
    bias = result['defocus_bias']
    ax.set_title(
        f"{case['description']}\n"
        f"lambda = {lam*1e9:.0f} nm, aperture = {case['aperture']*1e3:.2f} mm, "
        f"paraxial EFL = {result['f_nom']*1e3:.2f} mm | "
        f"defocus bias [nm/mm^2]: paraxial={bias['paraxial']:+.2f}, "
        f"slant={bias['slant']:+.2f}, traced={bias['ray-traced']:+.2f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)

    # --- Middle + bottom rows: four residual panels -------------------
    levels = [
        ('raw',                 gs[1, 0]),
        ('piston',              gs[1, 1]),
        ('piston+tilt',         gs[2, 0]),
        ('piston+tilt+defocus', gs[2, 1]),
    ]

    for level_name, gs_slot in levels:
        ax = fig.add_subplot(gs_slot)
        s = result['summaries'][level_name]
        for key in ('paraxial', 'slant', 'ray-traced'):
            st = method_styles[key]
            stats = s[key]
            ax.plot(x_mm, stats['residual'],
                    color=st['color'], ls=st['ls'], lw=st['lw'],
                    label=(f'{key:11s} PV={stats["pv_nm"]:.2f} nm, '
                           f'RMS={stats["rms_nm"]:.2f} nm'))
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.set_xlabel('exit-plane x [mm]')
        ax.set_ylabel('OPD residual (wave - geom) [nm]')
        ax.set_title(f'Residual -- {level_name} removed', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8, prop={'family': 'monospace'})

    fig.suptitle(f"Real-lens OPD validation: {case['name']}",
                 fontsize=13, y=0.995)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =========================================================================
# Per-case OPD curve dump (universal CSV for downstream analysis)
# =========================================================================

def write_per_case_opd_csv(result, out_dir):
    """Write the four OPD curves + raw and (piston+tilt+defocus)-removed
    residuals for one case as a tabular CSV.

    Columns:
      x_mm
        pupil-plane x coordinate, in millimetres.
      opl_{paraxial,slant,traced,geom}_nm
        absolute OPL at each x, in nanometres.  ``geom`` is the
        sequential-ray-tracer ground truth interpolated onto the wave
        grid; the other three are wave OPDs from apply_real_lens
        (paraxial, slant) and apply_real_lens_traced.
      residual_raw_{paraxial,slant,traced}_nm
        opl_method - opl_geom, no mode removal.  This is the
        physically-meaningful "wave-vs-truth" residual; the full
        defocus-bias offset is still in here.
      residual_pt+focus_{paraxial,slant,traced}_nm
        same residual after removing the best-fit piston, tilt, and
        defocus -- the "image-quality OPD" used in report.md's
        physics-check column.

    NaN rows are written as empty cells so spreadsheet importers
    handle them gracefully.

    File size: ~1-3 MB per case (21 cases * average ~1.5 MB =
    ~30 MB total committed to the repo, sized to be human-browsable
    rather than maximally compact).  Use NPZ if you need a smaller /
    Python-only format.
    """
    case = result['case']
    name = case['name']
    csv_path = os.path.join(out_dir, f'{name}_opd.csv')

    x_mm = result['x_wave'] * 1e3

    cols = {
        'x_mm':                  x_mm,
        'opl_paraxial_nm':       result['opl_par'] * 1e9,
        'opl_slant_nm':          result['opl_slant'] * 1e9,
        'opl_traced_nm':         result['opl_traced'] * 1e9,
        'opl_geom_nm':           result['opl_geom_on_wave'] * 1e9,
        'residual_raw_paraxial_nm': result['residuals_nm']['paraxial'],
        'residual_raw_slant_nm':    result['residuals_nm']['slant'],
        'residual_raw_traced_nm':   result['residuals_nm']['ray-traced'],
    }
    ptf = result['summaries']['piston+tilt+defocus']
    cols['residual_pt+focus_paraxial_nm'] = ptf['paraxial']['residual']
    cols['residual_pt+focus_slant_nm']    = ptf['slant']['residual']
    cols['residual_pt+focus_traced_nm']   = ptf['ray-traced']['residual']

    keys = list(cols.keys())
    n_rows = len(x_mm)

    def _fmt(v):
        # Empty cell for NaN -- universally readable across spreadsheets,
        # and unambiguous in CSV (vs. the literal string "nan" which
        # MATLAB and Excel handle inconsistently).
        if not np.isfinite(v):
            return ''
        return f'{v:+.6e}'

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n_rows):
            w.writerow([_fmt(cols[k][i]) for k in keys])


# =========================================================================
# Report writers
# =========================================================================

def write_report(results, out_dir):
    csv_path = os.path.join(out_dir, 'report.csv')
    md_path = os.path.join(out_dir, 'report.md')

    levels = ['raw', 'piston', 'piston+tilt', 'piston+tilt+defocus']

    method_keys = ('paraxial', 'slant', 'ray-traced')
    method_csv_labels = ('paraxial', 'slant', 'rayTraced')

    # --- CSV -----------------------------------------------------------
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(
            ['case', 'category', 'lambda_nm', 'aperture_mm',
             'f_EFL_mm', 'N', 'dx_um',
             'defocus_bias_paraxial_nm_per_mm2',
             'defocus_bias_slant_nm_per_mm2',
             'defocus_bias_rayTraced_nm_per_mm2']
            + [f'{lvl}_{m_lbl}_{stat}'
               for lvl in levels
               for m_lbl in method_csv_labels
               for stat in ('pv_nm', 'rms_nm')]
        )
        for r in results:
            c = r['case']
            row = [c['name'], c['category'], f'{c["wavelength"]*1e9:.0f}',
                   f'{c["aperture"]*1e3:.3f}',
                   f'{r["f_nom"]*1e3:.3f}', c['N'], f'{c["dx"]*1e6:.3f}']
            for mk in method_keys:
                row.append(f'{r["defocus_bias"][mk]:.4f}')
            for lvl in levels:
                for mk in method_keys:
                    s = r['summaries'][lvl][mk]
                    row.append(f'{s["pv_nm"]:.4f}')
                    row.append(f'{s["rms_nm"]:.4f}')
            w.writerow(row)

    # --- Markdown ------------------------------------------------------
    lines = []
    lines.append('# Real-lens OPD validation report\n')
    lines.append('Residuals are *wave OPD minus geometric OPL* at the '
                 'lens exit plane, in nanometers.\n')
    lines.append('## How to read this report\n')
    lines.append('After fixing an OPL-bookkeeping issue in the '
                 'sequential ray tracer (`_intersect_surface` now '
                 'accumulates the small "vertex-to-actual-sag" leg in '
                 'the appropriate medium), the wave-vs-geom comparison '
                 'is much cleaner.  Residuals decompose roughly as:\n')
    lines.append('- **Singlets at moderate to high f/#**: residuals '
                 'are tens to hundreds of nm RMS, dominated by '
                 'sampling/finite-grid effects rather than a real '
                 'physics gap.  Slant correction provides little or no '
                 'measurable benefit at this scale because the '
                 'angular-spectrum propagation between surfaces '
                 'already encodes most of the obliquity effect '
                 'naturally.\n')
    lines.append('- **Cemented doublets**: residuals remain in the '
                 'micrometre regime (hundreds of nm to tens of µm) '
                 'because the wave model treats each glass layer as a '
                 'uniform slab between vertex planes, while real rays '
                 'cross the cemented interface at z=sag(h).  This is '
                 'the genuine remaining limitation of the thin-element '
                 'phase-screen model and would require full WPM or '
                 'per-pixel ray-traced phase screens to remove.\n')
    lines.append('- **Plano-convex singlets at low f/#**: residual '
                 'grows as expected with aperture and curvature; '
                 'still well below 100 nm at f/3.\n\n')
    lines.append('Removal levels in the per-case PNG plots:\n')
    lines.append('- **raw**: absolute path-length agreement.\n')
    lines.append('- **piston**: removes the arbitrary phase reference '
                 'introduced by `np.unwrap` starting from the first '
                 'sample.\n')
    lines.append('- **piston+tilt**: also removes pupil-centering '
                 'artifacts (half-pixel grid offsets).\n')
    lines.append('- **piston+tilt+defocus**: removes the residual '
                 'wave-vs-ray quadratic mismatch.  The remaining RMS '
                 'is the high-order aberration content (spherical, '
                 'coma, higher) — this is the column reported in the '
                 'tables below.\n\n')
    lines.append('The "defocus bias" column is the coefficient of the '
                 'fitted `x²` term in the residual.  It is now small '
                 'for singlets (<10 nm/mm² typically), confirming the '
                 'wave model and ray trace agree on focal length, and '
                 'larger for cemented doublets where the uniform-slab '
                 'assumption visibly affects the local OPD curvature.\n\n')

    # Group by category
    by_cat = {}
    for r in results:
        by_cat.setdefault(r['case']['category'], []).append(r)

    for cat, rs in by_cat.items():
        lines.append(f'## Category: `{cat}`\n')
        lines.append(
            '| case | lambda [nm] | ap [mm] | EFL [mm] | '
            'RMS<sub>pt+focus</sub> [nm] paraxial | slant | '
            'ray-traced | best vs paraxial |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
        for r in rs:
            c = r['case']
            ptf = r['summaries']['piston+tilt+defocus']
            rms_p = ptf['paraxial']['rms_nm']
            rms_s = ptf['slant']['rms_nm']
            rms_t = ptf['ray-traced']['rms_nm']
            rmses = [rms_p, rms_s, rms_t]
            best = min(v for v in rmses if np.isfinite(v))
            improvement = (rms_p / best) if best > 1e-12 else float('inf')
            lines.append(
                f'| `{c["name"]}` '
                f'| {c["wavelength"]*1e9:.0f} '
                f'| {c["aperture"]*1e3:.2f} '
                f'| {r["f_nom"]*1e3:.2f} '
                f'| {rms_p:.2f} '
                f'| {rms_s:.2f} '
                f'| {rms_t:.2f} '
                f'| {improvement:.2f}x |')
        lines.append('')

    lines.append('\n---\n')
    lines.append(
        'Generated by `validation/real_lens_opd/run_validation.py`\n')

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\nReport files: {csv_path}\n               {md_path}")


# =========================================================================
# Main
# =========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', type=str, default=None,
                    help='Run only the case with this name substring.')
    ap.add_argument('--category', type=str, default=None,
                    help='Run only cases in this category.')
    ap.add_argument('--quick', action='store_true',
                    help='Run only three representative cases (fast).')
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ZEMAX_DIR, exist_ok=True)

    if args.quick:
        cases = filter_cases(name_substr='plano_convex_R50_BK7') \
              + filter_cases(name_substr='AC254_100_C') \
              + filter_cases(name_substr='fnum_sweep_f10')
        cases = cases[:3]
    elif args.case:
        cases = filter_cases(name_substr=args.case)
    elif args.category:
        cases = filter_cases(category=args.category)
    else:
        cases = list_cases()

    if not cases:
        print("No cases match the filter.  Available cases:")
        for c in list_cases():
            print(f"  [{c['category']}] {c['name']}")
        sys.exit(1)

    print("=" * 70)
    print(f"Real-lens OPD validation: {len(cases)} case(s)")
    print("=" * 70)

    results = []
    for i, case in enumerate(cases):
        print(f"\n[{i+1}/{len(cases)}]")
        try:
            r = run_case(case, verbose=True)
        except Exception as e:
            print(f"    !! FAILED: {type(e).__name__}: {e}")
            continue
        results.append(r)
        png_path = os.path.join(RESULTS_DIR, f"{case['name']}.png")
        plot_case(r, png_path)
        print(f"    saved {png_path}")

        # Dump the per-pupil-x OPD curves and residuals as CSV.
        # This is the data behind the PNG -- people who want to
        # do their own residual-fitting / pull data into Excel or
        # MATLAB or pandas can read this without any Python /
        # NumPy dependency.
        try:
            write_per_case_opd_csv(r, RESULTS_DIR)
            print(f"    OPD CSV: "
                  f"{os.path.join(RESULTS_DIR, case['name'] + '_opd.csv')}")
        except Exception as e:
            print(f"    OPD CSV failed: {type(e).__name__}: {e}")

        # Emit matching Zemax prescriptions so the same test can be
        # cross-checked in OpticStudio.  One human-readable LDE table
        # plus one best-effort .zmx per case.
        try:
            pres_for_export = dict(case['prescription'])
            pres_for_export['aperture_diameter'] = case['aperture']
            txt_path = os.path.join(ZEMAX_DIR, f"{case['name']}.txt")
            zmx_path = os.path.join(ZEMAX_DIR, f"{case['name']}.zmx")
            export_zemax_lens_data(
                pres_for_export, txt_path,
                wavelength=case['wavelength'],
                stop_surface=0,
                aperture_diameter=case['aperture'],
                back_focal_length=r['f_nom'],
                description=case['description'],
                extra_notes=[
                    f'Paraxial EFL (at this wavelength): '
                    f'{r["f_nom"]*1e3:.3f} mm',
                    f'Residual (piston+tilt+defocus) RMS (slant): '
                    f'{r["summaries"]["piston+tilt+defocus"]["slant"]["rms_nm"]:.2f} nm',
                    'Expected OPD in Zemax: compare to the '
                    'piston+tilt+defocus-removed trace; the two should '
                    'agree to the Zemax sampling limit for aberration '
                    'shape (spherical, coma, higher).',
                ])
            export_zemax_zmx(
                pres_for_export, zmx_path,
                wavelength=case['wavelength'],
                stop_surface=0,
                aperture_diameter=case['aperture'],
                back_focal_length=r['f_nom'],
                name=case['name'])
            print(f"    Zemax: {txt_path}, {zmx_path}")
        except Exception as e:
            print(f"    Zemax export failed: {type(e).__name__}: {e}")

        gc.collect()

    if results:
        write_report(results, RESULTS_DIR)
    print("\nDone.")


if __name__ == '__main__':
    main()
