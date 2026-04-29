"""Through-focus smoke test.

Runs a collimated source through a Thorlabs AC254-100-C, scans through
the nominal focal plane, reports best focus + Strehl / spot size, then
runs a small tolerancing sweep (decenter / tilt / form error) on the
first surface.

Expected output: results/through_focus_smoke.png + console table.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
if _LIB_ROOT not in sys.path:
    sys.path.insert(0, _LIB_ROOT)

import lumenairy as op  # noqa: E402
from lumenairy.prescriptions import thorlabs_lens  # noqa: E402
from lumenairy.raytrace import (  # noqa: E402
    surfaces_from_prescription, system_abcd)


def main():
    OUT_DIR = os.path.join(_HERE, 'results')
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- System ------------------------------------------------------
    lam = 1.31e-6
    N = 4096
    dx = 4e-6      # 16.4 mm extent
    # Small aperture chosen so pupil edge phase is well-sampled
    # (edge fringe period at 100 mm focal length is ~32 um for 4 mm
    # aperture -- plenty of room at dx=4 um).
    aperture = 4.0e-3

    pres_for_efl = thorlabs_lens('AC254-100-C')
    surfs = surfaces_from_prescription(pres_for_efl)
    _, efl, bfl, _ = system_abcd(surfs, lam)
    # BFL is the distance from the last surface (where apply_real_lens
    # leaves the field) to the paraxial image plane.  This is the
    # correct focal distance for through-focus scans of the exit field.
    f_nom = bfl
    print(f'Paraxial EFL at lambda={lam*1e9:.0f}nm: {efl*1e3:.2f} mm, '
          f'BFL: {bfl*1e3:.2f} mm (using BFL for through-focus)')

    pres = thorlabs_lens('AC254-100-C')
    pres['aperture_diameter'] = aperture

    # --- Source: unit plane wave through the prescription -----------
    E_source = np.ones((N, N), dtype=np.complex128)

    # --- Nominal exit field and ideal-spot peak ---------------------
    print("Computing nominal exit field ...")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_exit = op.apply_real_lens(
            E_source, pres, lam, dx,
            bandlimit=True, slant_correction=True)
    print(f"  apply_real_lens: {time.time()-t0:.1f}s")

    ideal_peak = op.diffraction_limited_peak(E_exit, lam, f_nom, dx)
    print(f"  Diffraction-limited |E|^2 peak = {ideal_peak:.4e}")

    # --- Through-focus scan -----------------------------------------
    print("\nThrough-focus scan (nominal)...")
    z_range = f_nom + np.linspace(-5e-3, 5e-3, 41)
    t0 = time.time()
    scan = op.through_focus_scan(
        E_exit, dx, lam, z_range,
        bucket_radius=20e-6, ideal_peak=ideal_peak,
        verbose=False)
    print(f"  scan: {time.time()-t0:.1f}s")

    z_best, strehl_best = op.find_best_focus(scan, 'strehl')
    z_spot, spot_min = op.find_best_focus(scan, 'spot')
    print(f"  Best Strehl: {strehl_best:.4f} at z = {z_best*1e3:.3f} mm")
    print(f"  Min D4sigma: {spot_min*1e6:.2f} um at z = {z_spot*1e3:.3f} mm")

    fig_path = os.path.join(OUT_DIR, 'through_focus_scan.png')
    op.plot_through_focus(scan, best_z=z_best, path=fig_path)
    print(f"  Saved: {fig_path}")

    # --- Tolerancing sweep ------------------------------------------
    print("\nTolerancing sweep (single-axis perturbations)...")
    perts = [
        op.Perturbation(surface_index=0, decenter=(50e-6, 0.0),
                        name='S0 decenter 50 um x'),
        op.Perturbation(surface_index=0, tilt=(1e-3, 0.0),
                        name='S0 tilt 1 mrad Tx'),
        op.Perturbation(surface_index=0, form_error_rms=100e-9,
                        random_seed=42,
                        name='S0 form error 100 nm RMS'),
        op.Perturbation(surface_index=1, decenter=(50e-6, 0.0),
                        name='S1 (interior) decenter 50 um x'),
        op.Perturbation(surface_index=2, decenter=(50e-6, 0.0),
                        name='S2 decenter 50 um x'),
    ]

    results = op.tolerancing_sweep(
        pres, lam, N, dx, E_source, perts,
        focal_length=f_nom, aperture=aperture,
        z_scan_range=(-5e-3, 5e-3), z_scan_n=21,
        bucket_radius=20e-6, verbose=True)

    # Print summary table
    print("\n Summary")
    print(f"  {'case':40s}  {'Strehl':>8s}  {'dS':>8s}  "
          f"{'spot [um]':>10s}  {'dSpot [um]':>12s}")
    for r in results:
        ds = r.get('delta_strehl', 0.0)
        dsp = r.get('delta_spot', 0.0)
        print(f"  {r['name']:40s}  {r['strehl_peak']:8.4f}  "
              f"{ds:+8.4f}  {r['d4sigma_min']*1e6:10.2f}  "
              f"{dsp*1e6:+12.3f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
