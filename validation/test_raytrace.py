"""Geometric ray tracing tests: Snell, OPL, fans, spot, through-focus,
ABCD, paraxial, chromatic shift, prescription ops.

From:
- physics_deep_test.py (Snell, OPL, through-focus, chromatic)
- physics_complex_test.py (Seidel vs ray-trace SA, grating orders)
- physics_exhaustive_test.py (spot RMS vs GEO, ray fan antisymmetric,
  make_rings count, trace_prescription, prescription summary)
- physics_extended_test.py (OPD fan, off-axis PSF shift)
- deep_audit.py (biconic raytrace, check_opd_sampling + ABCD BFL,
  through-focus on traced output, single_plane_metrics keys,
  compute_psf+through-focus, Monte Carlo tolerancing)
- hammer_test.py (ABCD vs ray-trace EFL)
"""
from __future__ import annotations

import sys

import numpy as np

from _harness import Harness

import lumenairy as op
from lumenairy.raytrace import (
    surfaces_from_prescription, system_abcd, trace, _make_bundle,
    find_paraxial_focus, seidel_coefficients, spot_rms,
    spot_geo_radius, ray_fan_data, opd_fan_data,
    trace_prescription, prescription_summary, make_rings, Surface,
    apply_doe_phase_traced, RAY_EVANESCENT, RAY_OK,
)


H = Harness('raytrace')


# ---------------------------------------------------------------------
H.section('Raytrace fundamentals')


def t_raytrace_snell():
    surfs = [Surface(radius=np.inf, glass_before='air',
                     glass_after='N-BK7',
                     semi_diameter=10e-3, thickness=10e-3)]
    angle = np.radians(10)
    rays = _make_bundle(x=np.array([0.0]), y=np.array([0.0]),
                        L=np.array([0.0]),
                        M=np.array([np.sin(angle)]),
                        wavelength=1.31e-6)
    result = trace(rays, surfs, 1.31e-6)
    M_out = result.image_rays.M[0]
    n2 = op.get_glass_index('N-BK7', 1.31e-6)
    expected_M = np.sin(np.arcsin(1.0 * np.sin(angle) / n2))
    err = abs(abs(M_out) - expected_M)
    return err < 1e-6, \
        f'M_out={M_out:.8f}, expected={expected_M:.8f}'


H.run('Raytrace: Snell law at flat interface', t_raytrace_snell)


def t_raytrace_opl_bookkeeping():
    n_glass = op.get_glass_index('N-BK7', 1.31e-6)
    d = 5e-3
    surfs = [Surface(radius=np.inf, glass_before='air',
                     glass_after='N-BK7',
                     semi_diameter=10e-3, thickness=d),
             Surface(radius=np.inf, glass_before='N-BK7',
                     glass_after='air',
                     semi_diameter=10e-3, thickness=0)]
    rays = _make_bundle(x=np.array([0.0]), y=np.array([0.0]),
                        L=np.array([0.0]), M=np.array([0.0]),
                        wavelength=1.31e-6)
    result = trace(rays, surfs, 1.31e-6)
    opl = result.image_rays.opd[0]
    expected = n_glass * d
    err = abs(opl - expected)
    return err < 1e-12, \
        f'OPL={opl:.10f}, expected={expected:.10f}'


H.run('Raytrace: on-axis OPL = n*d', t_raytrace_opl_bookkeeping)


def t_raytrace_biconic():
    N = 256; lam = 1.31e-6
    pres = op.make_biconic(50e-3, 100e-3, -60e-3, -100e-3, 4e-3,
                           'N-BK7', aperture=3e-3)
    surfs = surfaces_from_prescription(pres)
    _, efl_x, _, _ = system_abcd(surfs, lam)
    rays = _make_bundle(
        x=np.linspace(-1e-3, 1e-3, 11),
        y=np.zeros(11),
        L=np.zeros(11), M=np.zeros(11), wavelength=lam)
    res = trace(rays, surfs, lam)
    return (np.isfinite(efl_x)
            and res.image_rays.alive.sum() > 0), \
        f'EFL={efl_x*1e3:.2f}mm, alive={res.image_rays.alive.sum()}'


H.run('raytrace + biconic', t_raytrace_biconic)


# ---------------------------------------------------------------------
H.section('Seidel / paraxial / ABCD')


def t_seidel_vs_raytrace_sa():
    pres = op.make_singlet(51.5e-3, np.inf, 4.1e-3, 'N-BK7',
                           aperture=25.4e-3)
    surfs = surfaces_from_prescription(pres)
    lam = 1.31e-6
    seidel_raw = seidel_coefficients(surfs, lam)
    if isinstance(seidel_raw, tuple) and isinstance(seidel_raw[0], dict):
        S1 = float(np.sum(seidel_raw[0].get('S1', np.zeros(1))))
    else:
        S1 = 0.0
    _, _, bfl, _ = system_abcd(surfs, lam)
    surfs_to_img = surfaces_from_prescription(pres)
    surfs_to_img[-1].thickness = bfl
    surfs_to_img.append(Surface(
        radius=np.inf, semi_diameter=np.inf,
        glass_before=surfs_to_img[-1].glass_after,
        glass_after=surfs_to_img[-1].glass_after))
    rays = _make_bundle(x=np.array([0.0]), y=np.array([12e-3]),
                        L=np.array([0.0]), M=np.array([0.0]),
                        wavelength=lam)
    result = trace(rays, surfs_to_img, lam)
    img = result.image_rays
    marginal_height = abs(img.y[0]) if img.alive[0] else 0.0
    s1_positive = S1 > 0
    has_sa = marginal_height > 1e-6
    return (s1_positive and has_sa), \
        (f'S1={S1:.3e} (>0), marginal miss={marginal_height*1e6:.2f}um')


H.run('Seidel vs ray-trace: spherical aberration sign',
      t_seidel_vs_raytrace_sa)


def t_abcd_vs_raytrace_efl():
    import lumenairy.raytrace as rt
    tmpl = op.make_singlet(R1=0.05, R2=-0.05, d=4e-3, glass='N-BK7',
                           aperture=5e-3)
    wv = 1.310e-6
    surfs = rt.surfaces_from_prescription(tmpl)
    _, efl_abcd, _, _ = rt.system_abcd(surfs, wv)
    heights = np.linspace(-0.5e-3, 0.5e-3, 21)
    zeros = np.zeros_like(heights)
    rays = rt._make_bundle(x=heights, y=zeros, L=zeros, M=zeros,
                           wavelength=wv)
    res = rt.trace(rays, surfs, wv)
    final = res.image_rays
    L_vals = final.L[final.alive]
    N_vals = final.N[final.alive]
    h_in = heights[final.alive]
    slopes = L_vals / N_vals
    nz = h_in != 0
    if nz.any():
        efl_ray = float(np.mean(-h_in[nz] / slopes[nz]))
        rel_err = abs(efl_ray - efl_abcd) / abs(efl_abcd)
        return rel_err < 1e-3, \
            (f'ABCD={efl_abcd*1e3:.3f}mm  ray={efl_ray*1e3:.3f}mm  '
             f'rel={rel_err:.2e}')
    return False, 'no rays survived'


H.run('ray-trace EFL == ABCD EFL within 0.1%',
      t_abcd_vs_raytrace_efl)


def t_check_opd_sampling_with_abcd_bfl():
    pres = op.thorlabs_lens('AC254-100-C')
    pres['aperture_diameter'] = 10e-3
    surfs = surfaces_from_prescription(pres)
    _, efl, bfl, _ = system_abcd(surfs, 1.31e-6)
    samp = op.check_opd_sampling(4e-6, 1.31e-6, 10e-3, bfl, verbose=False)
    return isinstance(samp['margin'], float), \
        f'margin={samp["margin"]:.2f}'


H.run('check_opd_sampling + ABCD BFL',
      t_check_opd_sampling_with_abcd_bfl)


def t_chromatic_shift_sign():
    pres_sing = op.make_singlet(51.5e-3, np.inf, 4.1e-3, 'N-BK7',
                                aperture=10e-3)
    pres_doub = op.thorlabs_lens('AC254-100-C')
    wvs = [1.064e-6, 1.31e-6, 1.55e-6]
    _, _, shift_sing = op.chromatic_focal_shift(pres_sing, wvs)
    _, _, shift_doub = op.chromatic_focal_shift(pres_doub, wvs)
    return shift_doub < shift_sing, \
        (f'singlet shift={shift_sing*1e3:.3f}mm, '
         f'doublet shift={shift_doub*1e3:.3f}mm')


H.run('Chromatic: achromat has less focal shift than singlet',
      t_chromatic_shift_sign)


# ---------------------------------------------------------------------
H.section('Spot / ray fan / OPD fan')


def t_spot_rms_vs_geo():
    pres = op.thorlabs_lens('LA1050-C')
    surfs = surfaces_from_prescription(pres)
    bfl = find_paraxial_focus(surfs, 1.31e-6)
    surfs[-1].thickness = bfl
    surfs.append(Surface(radius=np.inf, semi_diameter=np.inf,
                         glass_before=surfs[-1].glass_after,
                         glass_after=surfs[-1].glass_after))
    rays = make_rings(10e-3, 6, 24, 0, 1.31e-6)
    result = trace(rays, surfs, 1.31e-6)
    rms, _ = spot_rms(result)
    geo = spot_geo_radius(result)
    return geo >= rms * 0.99, \
        f'RMS={rms*1e6:.2f}um, GEO={geo*1e6:.2f}um'


H.run('Spot: GEO radius >= RMS radius', t_spot_rms_vs_geo)


def t_ray_fan_symmetric():
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=10e-3)
    surfs = surfaces_from_prescription(pres)
    bfl = find_paraxial_focus(surfs, 1.31e-6)
    surfs[-1].thickness = bfl
    surfs.append(Surface(radius=np.inf, semi_diameter=np.inf,
                         glass_before=surfs[-1].glass_after,
                         glass_after=surfs[-1].glass_after))
    py, ey, px, ex = ray_fan_data(surfs, 1.31e-6, semi_aperture=5e-3,
                                  field_angle=0.0, n_rays=21)
    n = len(ey)
    err = 0
    for i in range(n//2):
        if np.isfinite(ey[i]) and np.isfinite(ey[n-1-i]):
            err = max(err, abs(ey[i] + ey[n-1-i]))
    return err < 1e-6, f'antisymmetry error = {err*1e6:.4f} um'


H.run('Ray fan: on-axis tangential is antisymmetric',
      t_ray_fan_symmetric)


def t_opd_fan_small_for_singlet():
    pres = op.make_singlet(200e-3, np.inf, 3e-3, 'N-BK7',
                           aperture=5e-3)
    surfs = surfaces_from_prescription(pres)
    py, opd_y, px, opd_x = opd_fan_data(surfs, 1.31e-6,
                                        semi_aperture=2.5e-3)
    opd_pv = np.nanmax(np.abs(opd_y))
    return opd_pv < 10, f'OPD PV = {opd_pv:.4f} waves'


H.run('OPD fan: small for well-corrected singlet',
      t_opd_fan_small_for_singlet)


def t_make_rings_count():
    rays = make_rings(5e-3, num_rings=5, rays_per_ring=12,
                      field_angle=0, wavelength=1.31e-6)
    n = rays.x.size
    expected = 5 * 12 + 1
    return n == expected, f'n_rays = {n} (expect {expected})'


H.run('make_rings: correct ray count', t_make_rings_count)


def t_trace_prescription_convenience():
    result = trace_prescription(op.thorlabs_lens('AC254-100-C'),
                                wavelength=1.31e-6, num_rings=4)
    alive = result.image_rays.alive.sum()
    return alive > 10, f'{alive} rays alive at image'


H.run('trace_prescription: returns valid result',
      t_trace_prescription_convenience)


def t_prescription_summary_text():
    import io
    import contextlib
    pres = op.thorlabs_lens('AC254-100-C')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        prescription_summary(pres, 1.31e-6)
    text = buf.getvalue()
    return len(text) > 50, f'summary length = {len(text)} chars'


H.run('Prescription summary: produces text', t_prescription_summary_text)


# ---------------------------------------------------------------------
H.section('Through-focus / single-plane / image-side')


def t_through_focus_peak_at_focus():
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=3e-3)
    surfs = surfaces_from_prescription(pres)
    _, _, bfl, _ = system_abcd(surfs, 1.31e-6)
    E_in = np.ones((256, 256), dtype=np.complex128)
    E_exit = op.apply_real_lens(E_in, pres, 1.31e-6, 16e-6)
    z = np.linspace(bfl - 5e-3, bfl + 5e-3, 21)
    scan = op.through_focus_scan(E_exit, 16e-6, 1.31e-6, z)
    z_peak = z[np.argmax(scan.peak_I)]
    err = abs(z_peak - bfl)
    return err < 2e-3, \
        (f'peak at z={z_peak*1e3:.2f}mm, BFL={bfl*1e3:.2f}mm, '
         f'err={err*1e3:.2f}mm')


H.run('Through-focus: peak near BFL', t_through_focus_peak_at_focus)


def t_through_focus_on_traced_output():
    N = 256; dx = 16e-6; lam = 1.31e-6
    pres = op.thorlabs_lens('AC254-100-C')
    pres['aperture_diameter'] = 3e-3
    E = np.ones((N, N), dtype=np.complex128)
    Et = op.apply_real_lens_traced(E, pres, lam, dx,
                                   ray_subsample=4, n_workers=1)
    ideal = op.diffraction_limited_peak(Et, lam, 85e-3, dx)
    z = np.linspace(80e-3, 90e-3, 11)
    scan = op.through_focus_scan(Et, dx, lam, z, ideal_peak=ideal)
    zb, _ = op.find_best_focus(scan, 'strehl')
    return np.isfinite(zb), f'best focus z={zb*1e3:.2f}mm'


H.run('through-focus on traced output',
      t_through_focus_on_traced_output)


def t_single_plane_metrics_keys():
    N = 256; dx = 16e-6; lam = 1.31e-6
    pres = op.make_singlet(50e-3, float('inf'), 4e-3, 'N-BK7',
                           aperture=3e-3)
    E = np.ones((N, N), dtype=np.complex128)
    E_exit = op.apply_real_lens(E, pres, lam, dx)
    m = op.single_plane_metrics(E_exit, dx, lam, bucket_radius=100e-6)
    needed = {'peak_I', 'centroid_x', 'centroid_y',
              'd4sigma_x', 'd4sigma_y', 'rms_radius',
              'power_in_bucket'}
    return needed.issubset(set(m.keys())), \
        f'keys: {sorted(m.keys())[:6]}'


H.run('single_plane_metrics keys', t_single_plane_metrics_keys)


def t_compute_psf_plus_ideal_peak():
    N = 256; dx = 16e-6; lam = 1.31e-6
    pres = op.make_singlet(50e-3, float('inf'), 4e-3, 'N-BK7',
                           aperture=3e-3)
    E = np.ones((N, N), dtype=np.complex128)
    E_exit = op.apply_real_lens(E, pres, lam, dx)
    psf, dx_psf = op.compute_psf(E_exit, lam, 100e-3, dx)
    ideal_peak = op.diffraction_limited_peak(E_exit, lam, 100e-3, dx)
    return psf.max() > 0 and ideal_peak > 0, \
        f'PSF peak={psf.max():.3e}, ideal={ideal_peak:.3e}'


H.run('compute_psf + through-focus ideal peak',
      t_compute_psf_plus_ideal_peak)


def t_offaxis_shifts_psf():
    N = 512; dx = 4e-6; lam = 1.31e-6; angle = 0.01
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=3e-3)
    surfs = surfaces_from_prescription(pres)
    _, _, bfl, _ = system_abcd(surfs, lam)
    E_on, _, _ = op.create_tilted_plane_wave(N, dx, lam, angle_y=0)
    E_on = op.apply_real_lens(E_on, pres, lam, dx)
    E_foc_on = op.angular_spectrum_propagate(E_on, bfl, lam, dx)
    _, cy_on = op.beam_centroid(E_foc_on, dx)
    E_off, _, _ = op.create_tilted_plane_wave(N, dx, lam, angle_y=angle)
    E_off = op.apply_real_lens(E_off, pres, lam, dx)
    E_foc_off = op.angular_spectrum_propagate(E_off, bfl, lam, dx)
    _, cy_off = op.beam_centroid(E_foc_off, dx)
    shift = cy_off - cy_on
    expected_shift = bfl * np.tan(angle)
    err_pct = abs(shift - expected_shift) / abs(expected_shift) * 100
    return err_pct < 60, \
        (f'measured shift={shift*1e6:.1f}um, '
         f'expected={expected_shift*1e6:.1f}um, err={err_pct:.1f}%')


H.run('Off-axis: PSF shift ~ f*tan(angle)', t_offaxis_shifts_psf)


# ---------------------------------------------------------------------
H.section('Grating / diffraction orders')


def t_grating_orders():
    N = 1024; dx = 1e-6; lam = 0.633e-6; f = 5e-3
    d = 10e-6
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    E_in = np.exp(1j * np.pi * np.sin(2 * np.pi * X / d))
    E_in = op.apply_thin_lens(E_in, f, lam, dx)
    E_focus = op.angular_spectrum_propagate(E_in, f, lam, dx)
    I = np.abs(E_focus[N//2, :])**2
    x_1 = lam * f / d
    pix_1 = int(round(x_1 / dx))
    if N//2 + pix_1 >= N or N//2 - pix_1 < 0:
        return False, f'Order at pixel {pix_1} outside grid'
    I_0 = I[N//2]
    I_p1 = I[N//2 + pix_1]
    I_m1 = I[N//2 - pix_1]
    orders_present = (I_p1 > 0.01 * I_0) and (I_m1 > 0.01 * I_0)
    return orders_present, \
        (f'I(0)={I_0:.2e}, I(+1)={I_p1:.2e}, I(-1)={I_m1:.2e}')


H.run('Grating: +-1 orders at correct position', t_grating_orders)


# ---------------------------------------------------------------------
# Additional ray-physics & interop hammer tests (3.2.13)
# ---------------------------------------------------------------------
H.section('Ray physics: conjugates / through-focus / orientation')


def t_seidel_spherical_aberration_orientation_dependence():
    """A plano-convex lens with curved side facing the incoming light
    has different spherical aberration than the reverse orientation."""
    p_a = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=20e-3)
    p_b = op.make_singlet(np.inf, -50e-3, 4e-3, 'N-BK7', aperture=20e-3)
    out_a = op.seidel_prescription(p_a, 0.587e-6)
    out_b = op.seidel_prescription(p_b, 0.587e-6)
    # seidel_prescription returns a tuple whose first element is a dict
    # of per-surface coefficient arrays; sum S1 (spherical) over surfaces.
    sa_a = float(np.sum(out_a[0]['S1']))
    sa_b = float(np.sum(out_b[0]['S1']))
    finite = np.isfinite(sa_a) and np.isfinite(sa_b)
    different = abs(sa_a - sa_b) > 0.01 * max(abs(sa_a), abs(sa_b), 1e-30)
    return finite and different, \
        f'sumS1(curved-first)={sa_a:.3e}, sumS1(flat-first)={sa_b:.3e}'


H.run('Seidel SA: depends on plano-convex orientation',
      t_seidel_spherical_aberration_orientation_dependence)


def t_through_focus_rms_minimum_near_paraxial_focus():
    """Through-focus RMS spot has its minimum near the paraxial focus."""
    from lumenairy.raytrace import refocus
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=10e-3)
    surfs = surfaces_from_prescription(pres)
    bfl = find_paraxial_focus(surfs, 0.587e-6)
    rays = make_rings(semi_aperture=4e-3, num_rings=4, rays_per_ring=8,
                      wavelength=0.587e-6)
    res = trace(rays, surfs, 0.587e-6)
    # Sweep image-plane offsets around the paraxial-focus distance.
    sweep = np.linspace(bfl - 5e-3, bfl + 5e-3, 41)
    rms = []
    for d in sweep:
        r = refocus(res, d)
        rms.append(float(spot_rms(r)[0]))
    rms = np.array(rms)
    i_min = int(np.argmin(rms))
    delta = abs(sweep[i_min] - bfl)
    return delta < 5e-3, \
        f'best plane={sweep[i_min]*1e3:.3f}mm (bfl={bfl*1e3:.2f}mm)'


H.run('Through-focus RMS minimum near paraxial focus',
      t_through_focus_rms_minimum_near_paraxial_focus)


def t_make_rings_count_matches_request():
    """make_rings returns exactly num_rings * rays_per_ring + 1 (chief)."""
    rays = make_rings(semi_aperture=5e-3, num_rings=4, rays_per_ring=8,
                      wavelength=1.31e-6, include_chief=True)
    expected = 4 * 8 + 1
    return rays.n_rays == expected, \
        f'n_rays={rays.n_rays}, expected={expected}'


H.run('make_rings: ray count = num_rings*rays_per_ring + chief',
      t_make_rings_count_matches_request)


def t_chromatic_focal_shift_consistent_with_index_change():
    """Comparing two wavelengths, the singlet focus shift should have
    the same sign as -dn (higher index -> shorter EFL)."""
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=20e-3)
    surfs = surfaces_from_prescription(pres)
    bfl_short = find_paraxial_focus(surfs, 0.486e-6)  # F-line (blue)
    bfl_long  = find_paraxial_focus(surfs, 0.656e-6)  # C-line (red)
    n_short = op.get_glass_index('N-BK7', 0.486e-6)
    n_long  = op.get_glass_index('N-BK7', 0.656e-6)
    # n_short > n_long for normal dispersion -> bfl_short < bfl_long
    sign_correct = (n_short > n_long) == (bfl_short < bfl_long)
    return sign_correct, \
        (f'n(486nm)={n_short:.4f}, n(656nm)={n_long:.4f}; '
         f'bfl(486)={bfl_short*1e3:.3f}mm, '
         f'bfl(656)={bfl_long*1e3:.3f}mm')


H.run('Chromatic shift: blue focuses shorter than red for normal dispersion',
      t_chromatic_focal_shift_consistent_with_index_change)


def t_lens_abcd_consistency_with_system_abcd():
    """For a single lens, lens_abcd should give the same EFL as
    system_abcd applied to its surfaces."""
    from lumenairy.raytrace import lens_abcd
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=20e-3)
    surfs = surfaces_from_prescription(pres)
    _, efl_sys, _, _ = system_abcd(surfs, 0.587e-6)
    info = lens_abcd(surfs, 0.587e-6)
    efl_lens = info.efl
    rel = abs(efl_sys - efl_lens) / abs(efl_sys)
    return rel < 1e-9, \
        f'system_abcd EFL={efl_sys*1e3:.6f}mm, lens_abcd EFL={efl_lens*1e3:.6f}mm'


H.run('lens_abcd EFL matches system_abcd EFL',
      t_lens_abcd_consistency_with_system_abcd)


def t_spot_rms_decreases_with_smaller_aperture():
    """Closing the aperture reduces spherical-aberration-driven spot
    RMS for a singlet (since SA scales with marginal ray height^4)."""
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=20e-3)
    surfs = surfaces_from_prescription(pres)
    rms = []
    for ap in (8e-3, 4e-3, 2e-3):
        rays = make_rings(semi_aperture=ap, num_rings=4, rays_per_ring=8,
                          wavelength=0.587e-6)
        res = trace(rays, surfs, 0.587e-6)
        rms.append(float(spot_rms(res)[0]))
    monotonic = rms[0] > rms[1] > rms[2]
    return monotonic, \
        (f'rms(ap=8mm)={rms[0]*1e6:.2f}, '
         f'rms(ap=4mm)={rms[1]*1e6:.2f}, '
         f'rms(ap=2mm)={rms[2]*1e6:.2f} um')


H.run('Spot RMS shrinks monotonically with smaller aperture',
      t_spot_rms_decreases_with_smaller_aperture)


def t_trace_prescription_returns_image_rays():
    """trace_prescription must produce a TraceResult whose image_rays
    has the requested number of rays alive."""
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=20e-3)
    res = trace_prescription(pres, semi_aperture=5e-3, num_rings=4,
                             rays_per_ring=8, wavelength=0.587e-6)
    n_alive = int(np.sum(res.image_rays.alive))
    n_tot = res.image_rays.n_rays
    return n_alive >= n_tot * 0.9, \
        f'alive={n_alive}/{n_tot}'


H.run('trace_prescription: most rays alive at image plane',
      t_trace_prescription_returns_image_rays)


# ---------------------------------------------------------------------
H.section('DOE / grating diffraction-order direction shift')


def t_doe_phase_traced_zero_order():
    """order=(0, 0) is a no-op: directions and aliveness unchanged."""
    rays = _make_bundle(x=np.linspace(-1e-3, 1e-3, 9),
                        y=np.zeros(9),
                        L=np.zeros(9), M=np.zeros(9),
                        wavelength=1.31e-6)
    out = apply_doe_phase_traced(rays, order_x=0, order_y=0,
                                 period_x=10e-6)
    same_dir = (np.allclose(out.L, rays.L)
                and np.allclose(out.M, rays.M)
                and np.allclose(out.N, rays.N))
    same_alive = bool(out.alive.all()) and (out.n_rays == rays.n_rays)
    return same_dir and same_alive, \
        f'L drift max={np.max(np.abs(out.L - rays.L)):.2e}'


H.run('DOE: zero order is a no-op', t_doe_phase_traced_zero_order)


def t_doe_phase_traced_grating_equation():
    """Single-axis grating: dL = m*lambda/period exactly; M unchanged."""
    lam = 1.31e-6
    period = 10e-6
    m = 3
    rays = _make_bundle(x=np.zeros(1), y=np.zeros(1),
                        L=np.zeros(1), M=np.zeros(1), wavelength=lam)
    out = apply_doe_phase_traced(rays, order_x=m, order_y=0,
                                 period_x=period)
    expected_L = m * lam / period
    L_err = abs(out.L[0] - expected_L)
    M_unchanged = abs(out.M[0]) < 1e-15
    norm_ok = abs(out.L[0] ** 2 + out.M[0] ** 2 + out.N[0] ** 2 - 1.0)
    return (L_err < 1e-15 and M_unchanged and norm_ok < 1e-15), \
        (f'L={out.L[0]:.8f} (expected {expected_L:.8f}), '
         f'M={out.M[0]:.2e}, |L^2+M^2+N^2-1|={norm_ok:.2e}')


H.run('DOE: 1-D grating equation dL = m*lambda/period',
      t_doe_phase_traced_grating_equation)


def t_doe_phase_traced_unit_direction_norm():
    """L^2 + M^2 + N^2 == 1 exactly for every alive ray after a 2-D
    crossed-grating order shift over a fan of input angles."""
    lam = 1.31e-6
    rays = _make_bundle(
        x=np.zeros(7), y=np.zeros(7),
        L=np.linspace(-0.1, 0.1, 7),
        M=np.linspace(-0.05, 0.05, 7),
        wavelength=lam)
    out = apply_doe_phase_traced(rays, order_x=2, order_y=-1,
                                 period_x=8e-6, period_y=12e-6)
    norm = out.L ** 2 + out.M ** 2 + out.N ** 2
    err = float(np.max(np.abs(norm[out.alive] - 1.0)))
    return err < 1e-15, f'max |norm-1| = {err:.2e}'


H.run('DOE: post-shift direction cosines remain unit-normalized',
      t_doe_phase_traced_unit_direction_norm)


def t_doe_phase_traced_evanescent():
    """Very high diffraction order: |L+dL|^2 + |M+dM|^2 > 1 should
    flag the ray alive=False with RAY_EVANESCENT."""
    lam = 1.31e-6
    period = 1.5e-6                   # period < lambda -> high orders evanescent
    rays = _make_bundle(x=np.zeros(1), y=np.zeros(1),
                        L=np.zeros(1), M=np.zeros(1), wavelength=lam)
    # m=2 -> dL = 2*1.31/1.5 = 1.747, clearly evanescent
    out = apply_doe_phase_traced(rays, order_x=2, order_y=0,
                                 period_x=period)
    dead = (not bool(out.alive[0]))
    code_ok = (out.error_code is not None
               and int(out.error_code[0]) == int(RAY_EVANESCENT))
    return dead and code_ok, \
        (f'alive={bool(out.alive[0])}, '
         f'code={int(out.error_code[0]) if out.error_code is not None else None}, '
         f'expected RAY_EVANESCENT={int(RAY_EVANESCENT)}')


H.run('DOE: evanescent orders flagged alive=False, RAY_EVANESCENT',
      t_doe_phase_traced_evanescent)


def t_doe_phase_traced_array_orders_layout():
    """Array-order input replicates bundle in order-major layout:
    out[k*n_rays:(k+1)*n_rays] is the k-th order's rays."""
    lam = 1.31e-6
    period = 20e-6
    rays = _make_bundle(x=np.linspace(-1e-3, 1e-3, 5),
                        y=np.zeros(5),
                        L=np.zeros(5), M=np.zeros(5), wavelength=lam)
    mx = np.array([0, 1, 2, -1])
    my = np.array([0, 0, 1, -2])
    out = apply_doe_phase_traced(rays, order_x=mx, order_y=my,
                                 period_x=period)
    # Length check
    if out.n_rays != len(mx) * 5:
        return False, f'len={out.n_rays}, expected {len(mx)*5}'
    # Per-order direction-cosine check (order-major layout)
    ok = True
    for k in range(len(mx)):
        sl = slice(k * 5, (k + 1) * 5)
        expected_L = mx[k] * lam / period
        expected_M = my[k] * lam / period
        if not (np.allclose(out.L[sl], expected_L)
                and np.allclose(out.M[sl], expected_M)
                and np.allclose(out.x[sl], rays.x)
                and np.allclose(out.y[sl], rays.y)):
            ok = False
            break
    return ok, f'n_rays={out.n_rays} for {len(mx)} orders x 5 rays'


H.run('DOE: order-array input replicates in order-major layout',
      t_doe_phase_traced_array_orders_layout)


def t_doe_phase_traced_freespace_shift():
    """End-to-end: launch on-axis ray bundle, apply grating shift, then
    trace through D mm of glass-free space.  Transverse landing position
    must equal D * (m*lambda/period) within paraxial tolerance."""
    lam = 1.31e-6
    period = 50e-6
    D = 30e-3                         # 30 mm of free space
    # A "free space" surface: flat, air on both sides, thickness D.
    surfs = [Surface(radius=np.inf, glass_before='air',
                     glass_after='air',
                     semi_diameter=20e-3, thickness=D),
             Surface(radius=np.inf, glass_before='air',
                     glass_after='air',
                     semi_diameter=20e-3, thickness=0)]
    rays0 = _make_bundle(x=np.zeros(1), y=np.zeros(1),
                         L=np.zeros(1), M=np.zeros(1), wavelength=lam)
    m = 4
    rays1 = apply_doe_phase_traced(rays0, order_x=m, order_y=0,
                                   period_x=period)
    res = trace(rays1, surfs, lam)
    x_landed = res.image_rays.x[0]
    # Expected (paraxial): x = D * tan(theta) = D * L / N ~ D * L for small L
    L = rays1.L[0]
    Nz = rays1.N[0]
    expected_x = D * L / Nz
    err_um = abs(x_landed - expected_x) * 1e6
    return err_um < 1e-3, \
        (f'x_landed={x_landed*1e6:.4f} um, '
         f'expected={expected_x*1e6:.4f} um, '
         f'err={err_um:.2e} um')


H.run('DOE: free-space transverse shift = distance * direction-cosine',
      t_doe_phase_traced_freespace_shift)


# ---------------------------------------------------------------------
H.section('trace() embedded grating diffraction (surface_diffraction kwarg)')


def t_trace_diff_kick_matches_apply_doe():
    """Per-surface diffraction in trace() reproduces the angular kick
    from apply_doe_phase_traced exactly."""
    pres = op.make_singlet(50e-3, -50e-3, 3e-3, 'N-BK7', aperture=10e-3)
    surfs = surfaces_from_prescription(pres)
    lam = 1.31e-6
    period = 100e-6
    bundle = _make_bundle(x=np.zeros(1), y=np.zeros(1),
                          L=np.zeros(1), M=np.zeros(1),
                          wavelength=lam)
    bundle.z = np.array([-50e-3])
    res_no = trace(bundle, surfs, lam, output_filter='last')
    res_di = trace(bundle, surfs, lam, output_filter='last',
                    surface_diffraction={1: (1, 0, period, period)})
    dL = float(res_di.image_rays.L[0]) - float(res_no.image_rays.L[0])
    expected = lam / period
    return abs(dL - expected) < 1e-9, \
        f'dL={dL:.6e}, expected={expected:.6e}'


H.run('trace + surface_diffraction: angular kick == m*lam/period',
      t_trace_diff_kick_matches_apply_doe)


def t_trace_diff_opl_includes_grating_phase():
    """The DOE OPL contribution m*lambda*x/period IS added to the OPL
    accumulator (this is the 'constant phase shift' that
    apply_doe_phase_traced docstring says callers must add manually --
    trace's surface_diffraction does it automatically)."""
    pres = op.make_singlet(50e-3, -50e-3, 3e-3, 'N-BK7', aperture=10e-3)
    surfs = surfaces_from_prescription(pres)
    lam = 1.31e-6
    period = 100e-6
    bundle = _make_bundle(x=np.array([2e-3]), y=np.array([0.0]),
                          L=np.zeros(1), M=np.zeros(1), wavelength=lam)
    bundle.z = np.array([-50e-3])
    res_no = trace(bundle, surfs, lam, output_filter='last')
    res_di = trace(bundle, surfs, lam, output_filter='last',
                    surface_diffraction={1: (1, 0, period, period)})
    dOPL = float(res_di.image_rays.opd[0]) - float(res_no.image_rays.opd[0])
    # Expect: dOPL ~ m*lambda/period * x_at_DOE.
    # x_at_DOE for ray launched at x=2 mm through 3 mm of BK7 with no
    # initial angle is approximately 2 mm minus a small refraction
    # shift; we just check magnitude is in the right ballpark.
    expected = (lam / period) * 2e-3
    rel_err = abs(dOPL - expected) / abs(expected)
    return rel_err < 0.05, \
        f'dOPL={dOPL*1e6:.3f} um, expected ~{expected*1e6:.3f} um, ' \
        f'rel_err={rel_err*100:.2f}%'


H.run('trace + surface_diffraction: OPL includes m*lam*x/period',
      t_trace_diff_opl_includes_grating_phase)


def t_trace_diff_evanescent_flagging():
    """Diffraction order whose grating equation produces L^2+M^2>1 is
    evanescent and the ray gets flagged alive=False."""
    pres = op.make_singlet(50e-3, np.inf, 1e-3, 'N-BK7', aperture=10e-3)
    surfs = surfaces_from_prescription(pres)
    lam = 1.31e-6
    period = 1.0e-6   # period < lambda forces large-angle
    bundle = _make_bundle(x=np.zeros(1), y=np.zeros(1),
                          L=np.zeros(1), M=np.zeros(1), wavelength=lam)
    bundle.z = np.array([-10e-3])
    res = trace(bundle, surfs, lam, output_filter='last',
                surface_diffraction={1: (5, 0, period, period)})
    alive = bool(res.image_rays.alive[0])
    ec = int(res.image_rays.error_code[0]) if res.image_rays.error_code is not None else -1
    return (not alive) and ec == int(RAY_EVANESCENT), \
        f'alive={alive}, error_code={ec} (expected RAY_EVANESCENT={int(RAY_EVANESCENT)})'


H.run('trace + surface_diffraction: evanescent orders flagged',
      t_trace_diff_evanescent_flagging)


if __name__ == '__main__':
    sys.exit(H.summary())
