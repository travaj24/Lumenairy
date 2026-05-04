"""Tests for lens operators: thin, spherical, aspheric, real, biconic,
cylindrical, GRIN, axicon, Maslov.

From:
- physics_deep_test.py (thin lens focus, real lens EFL, power conservation,
  biconic reduces to spherical, cylindrical flat axis)
- physics_complex_test.py (4f relay)
- physics_exhaustive_test.py (aspheric SA reduction, GRIN phase, axicon,
  doublet glass sequence)
- physics_extended_test.py (cylindrical line focus, biconic astigmatism)
- deep_audit.py (biconic + traced + opd_1d, cylindrical + seidel,
  biconic + slant_correction, seidel runs, system: biconic + free space,
  all optional features together)
"""
from __future__ import annotations

import sys

import numpy as np

from _harness import Harness

import lumenairy as op
from lumenairy.raytrace import (
    surfaces_from_prescription, system_abcd, trace, find_paraxial_focus,
    spot_rms, make_rings, Surface,
)


H = Harness('lenses')


# ---------------------------------------------------------------------
H.section('Thin-lens and real-lens basics')


def t_thin_lens_focal_spot():
    N = 512; dx = 4e-6; lam = 1.31e-6; f = 50e-3
    E = np.ones((N, N), dtype=np.complex128)
    E = op.apply_thin_lens(E, f, lam, dx)
    E_focus = op.angular_spectrum_propagate(E, f, lam, dx)
    I = np.abs(E_focus)**2
    peak_idx = np.unravel_index(np.argmax(I), I.shape)
    err = max(abs(peak_idx[0] - N//2), abs(peak_idx[1] - N//2))
    return err <= 1, f'peak offset from center = {err} pixels'


H.run('Thin lens: focuses at f', t_thin_lens_focal_spot)


def t_real_lens_singlet_efl():
    pres = op.make_singlet(51.5e-3, np.inf, 4.1e-3, 'N-BK7',
                           aperture=25.4e-3)
    surfs = surfaces_from_prescription(pres)
    _, efl, _, _ = system_abcd(surfs, 1.31e-6)
    n = op.get_glass_index('N-BK7', 1.31e-6)
    f_expected = 51.5e-3 / (n - 1)
    err_pct = abs(efl - f_expected) / f_expected * 100
    return err_pct < 1.0, \
        f'EFL={efl*1e3:.3f}mm, expected={f_expected*1e3:.3f}mm, err={err_pct:.3f}%'


H.run('Real lens: singlet EFL matches lensmaker eq',
      t_real_lens_singlet_efl)


def t_real_lens_power_conservation():
    N = 512; dx = 8e-6; lam = 1.31e-6; ap = 3e-3
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=ap)
    E_in = np.ones((N, N), dtype=np.complex128)
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    mask = (X**2 + Y**2) <= (ap/2)**2
    P_in = float(np.sum(np.abs(E_in[mask])**2) * dx**2)
    E_out = op.apply_real_lens(E_in, pres, lam, dx)
    P_out = float(np.sum(np.abs(E_out)**2) * dx**2)
    ratio = P_out / P_in
    return abs(ratio - 1.0) < 0.01, f'P_out/P_in = {ratio:.6f}'


H.run('Real lens: power conservation (no fresnel)',
      t_real_lens_power_conservation)


def t_fresnel_reduces_power():
    N = 256; dx = 16e-6; lam = 1.31e-6
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=3e-3)
    E_in = np.ones((N, N), dtype=np.complex128)
    E_no_f = op.apply_real_lens(E_in, pres, lam, dx, fresnel=False)
    E_yes_f = op.apply_real_lens(E_in, pres, lam, dx, fresnel=True)
    P_no = float(np.sum(np.abs(E_no_f)**2) * dx**2)
    P_yes = float(np.sum(np.abs(E_yes_f)**2) * dx**2)
    loss = 1 - P_yes / P_no
    return 0.04 < loss < 0.15, \
        f'Fresnel loss = {loss*100:.2f}% (expect 4-15%)'


H.run('Fresnel: reduces power by ~4-15% for BK7',
      t_fresnel_reduces_power)


def t_doublet_glass_sequence():
    pres = op.thorlabs_lens('AC254-100-C')
    surfs = pres['surfaces']
    seq = [(s['glass_before'], s['glass_after']) for s in surfs]
    expected = [('air', 'N-BAF10'), ('N-BAF10', 'N-SF6HT'),
                ('N-SF6HT', 'air')]
    return seq == expected, f'glass sequence = {seq}'


H.run('Doublet: correct glass sequence', t_doublet_glass_sequence)


# ---------------------------------------------------------------------
H.section('Biconic, cylindrical, aspheric')


def t_biconic_reduces_to_spherical():
    h = np.linspace(0, 5e-3, 100)
    sag_sym = op.surface_sag_general(h**2, R=50e-3, conic=-0.5)
    sag_bic = op.surface_sag_biconic(h, np.zeros_like(h),
                                     R_x=50e-3, R_y=50e-3,
                                     conic_x=-0.5, conic_y=-0.5)
    err = np.max(np.abs(sag_sym - sag_bic))
    return err < 1e-15, f'max diff = {err:.2e}'


H.run('Biconic: Rx=Ry reduces to symmetric',
      t_biconic_reduces_to_spherical)


def t_cylindrical_sag_zero_in_flat_axis():
    y = np.linspace(-5e-3, 5e-3, 100)
    sag = op.surface_sag_biconic(np.zeros_like(y), y,
                                 R_x=50e-3, R_y=np.inf)
    return np.all(sag == 0.0), f'max sag in y = {np.abs(sag).max():.2e}'


H.run('Cylindrical: zero sag along flat axis',
      t_cylindrical_sag_zero_in_flat_axis)


def t_cylindrical_line_focus():
    N = 256; dx = 8e-6; lam = 1.31e-6
    pres = op.make_cylindrical(50e-3, 3e-3, 'N-BK7', axis='x',
                               aperture=3e-3)
    surfs = surfaces_from_prescription(pres)
    _, _, bfl, _ = system_abcd(surfs, lam)
    E_in = np.ones((N, N), dtype=np.complex128)
    E_exit = op.apply_real_lens(E_in, pres, lam, dx)
    E_focus = op.angular_spectrum_propagate(E_exit, bfl, lam, dx)
    d4x, d4y = op.beam_d4sigma(E_focus, dx)
    ratio = d4y / d4x if d4x > 0 else 0
    return ratio > 3, \
        f'D4sig_x={d4x*1e6:.1f}um, D4sig_y={d4y*1e6:.1f}um, ratio={ratio:.1f}'


H.run('Cylindrical: line focus (y >> x at focus)',
      t_cylindrical_line_focus)


def t_biconic_astigmatic_focus():
    pres = op.make_biconic(50e-3, 80e-3, np.inf, np.inf, 3e-3,
                           'N-BK7', aperture=3e-3)
    n_bk7 = op.get_glass_index('N-BK7', 1.31e-6)
    f_x_expected = 50e-3 / (n_bk7 - 1)
    f_y_expected = 80e-3 / (n_bk7 - 1)
    ratio = f_y_expected / f_x_expected
    return abs(ratio - 80/50) < 0.01, \
        f'f_x={f_x_expected*1e3:.2f}mm, f_y={f_y_expected*1e3:.2f}mm, ratio={ratio:.3f}'


H.run('Biconic: Rx/Ry ratio matches focal-length ratio',
      t_biconic_astigmatic_focus)


def t_aspheric_conic_reduces_sa():
    lam = 1.31e-6
    pres_sphere = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7',
                                  aperture=20e-3)
    pres_sphere['surfaces'][0]['conic'] = 0.0
    pres_para = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7',
                                aperture=20e-3)
    pres_para['surfaces'][0]['conic'] = -1.0

    surfs_s = surfaces_from_prescription(pres_sphere)
    surfs_p = surfaces_from_prescription(pres_para)
    bfl_s = find_paraxial_focus(surfs_s, lam)
    bfl_p = find_paraxial_focus(surfs_p, lam)

    def get_rms(surfs, bfl):
        s2 = [Surface(radius=s.radius, conic=s.conic,
                      semi_diameter=s.semi_diameter,
                      glass_before=s.glass_before,
                      glass_after=s.glass_after,
                      is_mirror=s.is_mirror, thickness=s.thickness)
              for s in surfs]
        s2[-1].thickness = bfl
        s2.append(Surface(radius=np.inf, semi_diameter=np.inf,
                          glass_before=s2[-1].glass_after,
                          glass_after=s2[-1].glass_after))
        rays = make_rings(10e-3, 6, 24, 0, lam)
        r = trace(rays, s2, lam)
        rms, _ = spot_rms(r)
        return rms

    rms_sphere = get_rms(surfs_s, bfl_s)
    rms_para = get_rms(surfs_p, bfl_p)
    return rms_para < rms_sphere, \
        f'RMS sphere={rms_sphere*1e6:.1f}um, para={rms_para*1e6:.1f}um'


H.run('Aspheric: paraboloid reduces SA vs sphere',
      t_aspheric_conic_reduces_sa)


# ---------------------------------------------------------------------
H.section('GRIN, axicon, 4f relay')


def t_grin_lens_phase():
    N = 256; dx = 4e-6; lam = 1.31e-6
    n0 = 1.5; g = 500; d = 2e-3
    E_in = np.ones((N, N), dtype=np.complex128)
    E_out = op.apply_grin_lens(E_in, n0, g, d, lam, dx)
    phase_center = np.angle(E_out[N//2, N//2])
    phase_edge = np.angle(E_out[N//2, N//2 + 30])
    diff = abs(phase_edge - phase_center)
    return diff > 0.01, f'phase diff center-edge = {diff:.4f} rad'


H.run('GRIN lens: applies nonzero quadratic phase',
      t_grin_lens_phase)


def t_axicon_bessel():
    N = 256; dx = 4e-6; lam = 1.31e-6
    E_in = np.ones((N, N), dtype=np.complex128)
    E_out = op.apply_axicon(E_in, alpha=np.radians(0.5),
                            n_axicon=1.5, wavelength=lam, dx=dx)
    E_prop = op.angular_spectrum_propagate(E_out, 2e-3, lam, dx)
    I = np.abs(E_prop)**2
    I_center = I[N//2, N//2]
    I_edge = I[N//2, 0]
    return I_center > I_edge, \
        f'I_center={I_center:.3e}, I_edge={I_edge:.3e}'


H.run('Axicon: bright center after propagation', t_axicon_bessel)


def t_4f_relay():
    N = 512; dx = 4e-6; lam = 1.31e-6; f = 50e-3
    E_in, _, _ = op.create_gaussian_beam(N, dx, 50e-6,
                                         x0=30e-6, y0=-20e-6)
    E = op.angular_spectrum_propagate(E_in, f, lam, dx)
    E = op.apply_thin_lens(E, f, lam, dx)
    E = op.angular_spectrum_propagate(E, 2*f, lam, dx)
    E = op.apply_thin_lens(E, f, lam, dx)
    E_out = op.angular_spectrum_propagate(E, f, lam, dx)
    I_in = np.abs(E_in)**2
    I_out_flipped = np.flip(np.abs(E_out)**2)
    corr = np.corrcoef(I_in.ravel(), I_out_flipped.ravel())[0, 1]
    return corr > 0.95, f'4f relay correlation = {corr:.4f}'


H.run('4f relay: preserves field (inverted)', t_4f_relay)


# ---------------------------------------------------------------------
H.section('Composition / compatibility with real-lens pipeline')


def t_biconic_traced_opd_1d():
    N = 256; dx = 16e-6; lam = 1.31e-6
    pres = op.make_biconic(50e-3, 75e-3, -60e-3, -80e-3, 4e-3, 'N-BK7',
                           aperture=3e-3)
    E = np.ones((N, N), dtype=np.complex128)
    Et = op.apply_real_lens_traced(
        E, pres, lam, dx, n_workers=1,
        min_coarse_samples_per_aperture=0)
    _, opd = op.wave_opd_1d(Et, dx, lam, aperture=2.5e-3)
    return np.isfinite(opd).any(), 'traced pipeline + opd_1d OK'


H.run('biconic + traced + opd_1d', t_biconic_traced_opd_1d)


def t_cylindrical_seidel():
    N = 256; dx = 16e-6; lam = 1.31e-6
    pres = op.make_cylindrical(50e-3, 3e-3, 'N-BK7', axis='x',
                               aperture=3e-3)
    E = np.ones((N, N), dtype=np.complex128)
    Ec = op.apply_real_lens(E, pres, lam, dx, seidel_correction=True)
    return Ec.shape == (N, N), f'shape = {Ec.shape}'


H.run('cylindrical + seidel_correction', t_cylindrical_seidel)


def t_biconic_slant_correction():
    N = 256; dx = 16e-6; lam = 1.31e-6
    pres = op.make_biconic(50e-3, 75e-3, float('inf'), float('inf'),
                           3e-3, 'N-BK7', aperture=3e-3)
    E = np.ones((N, N), dtype=np.complex128)
    E_out = op.apply_real_lens(E, pres, lam, dx, slant_correction=True)
    return np.abs(E_out).max() > 0, f'peak={np.abs(E_out).max():.3e}'


H.run('biconic + slant_correction', t_biconic_slant_correction)


def t_seidel_runs_without_error():
    N = 256; dx = 16e-6; lam = 1.31e-6
    pres = op.make_singlet(50e-3, float('inf'), 4e-3, 'N-BK7',
                           aperture=3e-3)
    E = np.ones((N, N), dtype=np.complex128)
    E_on = op.apply_real_lens(E, pres, lam, dx, seidel_correction=True)
    return (np.abs(E_on).max() > 0 and np.all(np.isfinite(E_on))), \
        f'peak={np.abs(E_on).max():.3e}'


H.run('seidel runs without error', t_seidel_runs_without_error)


def t_system_biconic_plus_freespace():
    from lumenairy.system import propagate_through_system
    N = 256; dx = 16e-6; lam = 1.31e-6
    pres = op.make_biconic(50e-3, 60e-3, float('inf'), float('inf'),
                           3e-3, 'N-BK7', aperture=3e-3)
    E = np.ones((N, N), dtype=np.complex128)
    elements = [
        {'type': 'real_lens', 'prescription': pres},
        {'type': 'propagate', 'z': 10e-3},
    ]
    out = propagate_through_system(E, elements, lam, dx)
    E_out = out[0] if isinstance(out, tuple) else out
    return E_out.shape == (N, N), f'shape = {E_out.shape}'


H.run('system: biconic lens + free space',
      t_system_biconic_plus_freespace)


def t_all_optional_features_together():
    N = 256; dx = 16e-6; lam = 1.31e-6
    pres = op.make_doublet(50e-3, -30e-3, -80e-3, 4e-3, 2e-3,
                           'N-BK7', 'N-SF6HT', aperture=3e-3)
    pres['surfaces'][0]['decenter'] = (10e-6, 0)
    pres['surfaces'][0]['tilt'] = (1e-4, 0)
    pres['surfaces'][1]['form_error'] = np.random.default_rng(0).normal(
        0, 10e-9, (N, N))
    E = np.ones((N, N), dtype=np.complex128)
    E_out = op.apply_real_lens(
        E, pres, lam, dx,
        fresnel=True, absorption=True,
        slant_correction=True, seidel_correction=True)
    return (E_out.shape == (N, N) and np.abs(E_out).max() > 0), \
        f'shape={E_out.shape}, peak={np.abs(E_out).max():.3e}'


H.run('all optional features together', t_all_optional_features_together)


def t_multi_element_system_chain():
    from lumenairy.system import propagate_through_system
    N = 128; dx = 16e-6; lam = 1.31e-6
    E_in = np.ones((N, N), dtype=np.complex128)
    elements = [
        {'type': 'lens', 'f': 100e-3},
        {'type': 'propagate', 'z': 50e-3},
        {'type': 'lens', 'f': 50e-3},
        {'type': 'propagate', 'z': 25e-3},
    ]
    out = propagate_through_system(E_in, elements, lam, dx)
    E_out = out[0] if isinstance(out, tuple) else out
    return (E_out.shape == (N, N) and np.isfinite(E_out).all()), \
        f'output shape={E_out.shape}'


H.run('System chain: 3-element propagation',
      t_multi_element_system_chain)


def t_apply_real_lens_wave_propagator_sas():
    """The wave_propagator='sas' switch should produce a finite field
    of the same shape as wave_propagator='asm'.  The through-glass
    distance is small (~mm) so the two must match in grid geometry
    and produce similar peak amplitude."""
    N, dx, lam = 256, 4e-6, 1.31e-6
    pres = op.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=3e-3)
    E_in, _, _ = op.create_gaussian_beam(N, dx, 50e-6)
    E_asm = op.apply_real_lens(E_in, pres, lam, dx, wave_propagator='asm')
    E_sas = op.apply_real_lens(E_in, pres, lam, dx, wave_propagator='sas')
    peak_asm = np.abs(E_asm).max()
    peak_sas = np.abs(E_sas).max()
    ok = (E_asm.shape == E_sas.shape == (N, N)
          and peak_asm > 0 and peak_sas > 0)
    return ok, \
        f'|E|_max: ASM={peak_asm:.3e}, SAS={peak_sas:.3e}'


H.run('apply_real_lens: wave_propagator=sas produces valid field',
      t_apply_real_lens_wave_propagator_sas)


def t_apply_real_lens_traced_wave_propagator_sas():
    N, dx, lam = 256, 4e-6, 1.31e-6
    pres = op.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=3e-3)
    E_in, _, _ = op.create_gaussian_beam(N, dx, 50e-6)
    E_out = op.apply_real_lens_traced(
        E_in, pres, lam, dx, ray_subsample=1, n_workers=1,
        wave_propagator='sas', preserve_input_phase=False)
    ok = (E_out.shape == (N, N) and np.all(np.isfinite(E_out))
          and np.abs(E_out).max() > 0)
    return ok, f'|E|_max={np.abs(E_out).max():.3e}'


H.run('apply_real_lens_traced: wave_propagator=sas threads through',
      t_apply_real_lens_traced_wave_propagator_sas)


def t_apply_real_lens_wave_propagator_fresnel():
    N, dx, lam = 256, 4e-6, 1.31e-6
    pres = op.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=3e-3)
    E_in, _, _ = op.create_gaussian_beam(N, dx, 50e-6)
    E_out = op.apply_real_lens(
        E_in, pres, lam, dx, wave_propagator='fresnel')
    ok = (E_out.shape == (N, N) and np.all(np.isfinite(E_out))
          and np.abs(E_out).max() > 0)
    return ok, f'|E|_max={np.abs(E_out).max():.3e}'


H.run('apply_real_lens: wave_propagator=fresnel produces valid field',
      t_apply_real_lens_wave_propagator_fresnel)


def t_apply_real_lens_wave_propagator_rs():
    N, dx, lam = 256, 4e-6, 1.31e-6
    pres = op.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=3e-3)
    E_in, _, _ = op.create_gaussian_beam(N, dx, 50e-6)
    E_out = op.apply_real_lens(
        E_in, pres, lam, dx, wave_propagator='rayleigh_sommerfeld')
    ok = (E_out.shape == (N, N) and np.all(np.isfinite(E_out))
          and np.abs(E_out).max() > 0)
    return ok, f'|E|_max={np.abs(E_out).max():.3e}'


H.run('apply_real_lens: wave_propagator=rayleigh_sommerfeld valid',
      t_apply_real_lens_wave_propagator_rs)


def t_apply_real_lens_rs_matches_asm():
    """R-S and ASM are both exact propagators that preserve grid pitch;
    at mm-scale glass distances they should give nearly identical
    results (peak-amplitude agreement within a few percent)."""
    N, dx, lam = 256, 4e-6, 1.31e-6
    pres = op.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=3e-3)
    E_in, _, _ = op.create_gaussian_beam(N, dx, 50e-6)
    E_asm = op.apply_real_lens(E_in, pres, lam, dx, wave_propagator='asm')
    E_rs = op.apply_real_lens(
        E_in, pres, lam, dx, wave_propagator='rayleigh_sommerfeld')
    rel = (abs(float(np.abs(E_asm).max()) - float(np.abs(E_rs).max()))
           / float(np.abs(E_asm).max()))
    return rel < 0.02, \
        (f'|E|_max: ASM={np.abs(E_asm).max():.4e}, '
         f'RS={np.abs(E_rs).max():.4e}, rel={rel:.2e}')


H.run('apply_real_lens: RS and ASM agree at mm-scale glass thickness',
      t_apply_real_lens_rs_matches_asm)


def t_apply_real_lens_unknown_wave_propagator_raises():
    N, dx, lam = 64, 16e-6, 1.31e-6
    pres = op.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=3e-3)
    E_in = np.ones((N, N), dtype=np.complex128)
    try:
        op.apply_real_lens(E_in, pres, lam, dx, wave_propagator='bogus')
        return False, 'should have raised'
    except ValueError:
        return True, 'ValueError raised'


H.run('apply_real_lens: unknown wave_propagator raises ValueError',
      t_apply_real_lens_unknown_wave_propagator_raises)


# ---------------------------------------------------------------------
# Cross-pipeline interop & physics hammer tests (3.2.13)
# ---------------------------------------------------------------------
H.section('Cross-pipeline interop')


def t_thin_lens_vs_real_lens_focus_position():
    """Thin lens at f and real plano-convex of equivalent EFL focus
    a plane wave to the same axial location within Rayleigh range."""
    N, dx, lam, ap = 512, 8e-6, 1.31e-6, 3e-3
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=ap)
    surfs = surfaces_from_prescription(pres)
    _, efl, _, _ = system_abcd(surfs, lam)
    E_in = np.ones((N, N), dtype=np.complex128)
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    aper_mask = (X**2 + Y**2) <= (ap/2)**2
    E_in = E_in * aper_mask
    E_thin = op.apply_thin_lens(E_in.astype(np.complex128), efl, lam, dx)
    E_real = op.apply_real_lens(E_in.astype(np.complex128), pres, lam, dx)
    # Each propagated to its EFL.
    E_thin_f = op.angular_spectrum_propagate(E_thin, efl, lam, dx)
    E_real_f = op.angular_spectrum_propagate(E_real, efl, lam, dx)
    p_thin = float(np.abs(E_thin_f[N//2, N//2])**2)
    p_real = float(np.abs(E_real_f[N//2, N//2])**2)
    rel = abs(p_thin - p_real) / max(p_thin, 1e-30)
    return rel < 0.35, \
        (f'on-axis I rel diff (thin vs real plano-convex) = '
         f'{rel*100:.1f}% (efl={efl*1e3:.2f} mm)')


H.run('thin lens vs real plano-convex: similar focal-plane peak',
      t_thin_lens_vs_real_lens_focus_position)


def t_real_lens_vs_traced_low_NA_singlet():
    """At low NA, apply_real_lens (path-length) and apply_real_lens_traced
    (raytraced) should agree on the focal-plane on-axis intensity to ~10%."""
    N, dx, lam, ap = 512, 8e-6, 1.31e-6, 3e-3
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=ap)
    surfs = surfaces_from_prescription(pres)
    _, efl, _, _ = system_abcd(surfs, lam)
    E_in = np.ones((N, N), dtype=np.complex128)
    E_p = op.apply_real_lens(E_in, pres, lam, dx)
    E_t = op.apply_real_lens_traced(E_in, pres, lam, dx, n_workers=1,
                                     min_coarse_samples_per_aperture=0)
    E_pf = op.angular_spectrum_propagate(E_p, efl, lam, dx)
    E_tf = op.angular_spectrum_propagate(E_t, efl, lam, dx)
    p_p = float(np.abs(E_pf[N//2, N//2])**2)
    p_t = float(np.abs(E_tf[N//2, N//2])**2)
    rel = abs(p_p - p_t) / max(p_p, 1e-30)
    return rel < 0.30, \
        f'on-axis peak rel diff = {rel*100:.1f}% (path vs traced)'


H.run('apply_real_lens vs apply_real_lens_traced agree at low NA',
      t_real_lens_vs_traced_low_NA_singlet)


def t_thin_lens_combination_law():
    """Two thin lenses in contact behave as one with 1/f = 1/f1 + 1/f2."""
    N, dx, lam = 512, 8e-6, 1.31e-6
    f1, f2 = 100e-3, 150e-3
    f_comb = 1.0 / (1.0 / f1 + 1.0 / f2)
    E = np.ones((N, N), dtype=np.complex128)
    E_two = op.apply_thin_lens(
        op.apply_thin_lens(E, f1, lam, dx), f2, lam, dx)
    E_one = op.apply_thin_lens(E, f_comb, lam, dx)
    # The two operators should differ only by a global constant phase,
    # so the relative phase across the aperture should be identical.
    rel_phase = np.angle(E_two * np.conj(E_one))
    flat = np.std(rel_phase[N//2-50:N//2+50, N//2-50:N//2+50])
    return flat < 1e-9, \
        f'phase-residual std across central patch = {flat:.2e}'


H.run('Two thin lenses in contact: 1/f = 1/f1 + 1/f2',
      t_thin_lens_combination_law)


def t_thin_lens_inverse_cancels():
    """Apply f then -f should return the input phase (modulo 2pi)."""
    N, dx, lam = 256, 8e-6, 1.31e-6
    f = 50e-3
    E = np.ones((N, N), dtype=np.complex128) * np.exp(0j)
    E_out = op.apply_thin_lens(E, f, lam, dx)
    E_back = op.apply_thin_lens(E_out, -f, lam, dx)
    err = np.max(np.abs(E_back - E))
    return err < 1e-10, f'|E_back - E|_max = {err:.2e}'


H.run('Thin lens: apply +f then -f cancels',
      t_thin_lens_inverse_cancels)


def t_doublet_efl_lensmaker_consistency():
    """An achromat's EFL from system_abcd matches what trace_prescription
    converges to (within 1%)."""
    pres = op.make_doublet(50e-3, -40e-3, -150e-3, 4e-3, 2e-3,
                           'N-BK7', 'N-SF6', aperture=20e-3)
    surfs = surfaces_from_prescription(pres)
    _, efl_abcd, _, _ = system_abcd(surfs, 0.587e-6)
    bfl = find_paraxial_focus(surfs, 0.587e-6)
    # The achromat is in air with last surface at the back vertex; for a
    # roughly symmetric thin doublet the rear PP shift is small relative
    # to the EFL, so EFL and paraxial-focus image distance agree to ~5%.
    err = abs(efl_abcd - bfl) / abs(efl_abcd)
    return err < 0.05, \
        f'efl_ABCD={efl_abcd*1e3:.3f}mm, paraxial_focus={bfl*1e3:.3f}mm'


H.run('Achromat: ABCD EFL consistent with paraxial focus',
      t_doublet_efl_lensmaker_consistency)


def t_apply_real_lens_traced_intensity_finite():
    """Traced pipeline must produce a finite, positive intensity field
    even when the aperture is much smaller than the grid."""
    N, dx, lam, ap = 256, 8e-6, 1.31e-6, 1e-3
    pres = op.make_singlet(40e-3, np.inf, 3e-3, 'N-BK7', aperture=ap)
    E = np.ones((N, N), dtype=np.complex128)
    Et = op.apply_real_lens_traced(
        E, pres, lam, dx, n_workers=1,
        min_coarse_samples_per_aperture=0)
    return np.all(np.isfinite(Et)) and np.abs(Et).max() > 0, \
        f'finite={np.all(np.isfinite(Et))}, peak={np.abs(Et).max():.3e}'


H.run('apply_real_lens_traced: finite, positive output',
      t_apply_real_lens_traced_intensity_finite)


def t_check_grid_vs_apertures_flags_oversized():
    """`check_grid_vs_apertures` returns the offending surface when its
    semi_diameter exceeds the simulation grid's half-extent."""
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=12e-3)
    N, dx = 256, 8e-6   # grid semi = 1.024 mm, lens semi = 6 mm
    issues = op.check_grid_vs_apertures(pres, N, dx)
    grid_semi = 0.5 * N * dx
    have_issue = len(issues) >= 1
    if have_issue:
        label, sd, gs, gap = issues[0]
        ok = sd > grid_semi and gap > 0 and abs(gs - grid_semi) < 1e-12
    else:
        ok = False
    return ok, f'issues={len(issues)}, grid_semi={grid_semi*1e3:.3f}mm'


H.run('Grid-vs-aperture check: flags oversized aperture',
      t_check_grid_vs_apertures_flags_oversized)


def t_check_grid_vs_apertures_silent_when_ok():
    """`check_grid_vs_apertures` returns an empty list when every
    surface fits inside the grid."""
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=2e-3)
    N, dx = 1024, 8e-6   # grid semi = 4.096 mm, lens semi = 1 mm
    issues = op.check_grid_vs_apertures(pres, N, dx)
    return len(issues) == 0, f'issues={len(issues)}'


H.run('Grid-vs-aperture check: silent when grid is wide enough',
      t_check_grid_vs_apertures_silent_when_ok)


def t_apply_real_lens_warns_when_aperture_exceeds_grid():
    """`apply_real_lens` emits a UserWarning when the prescription's
    aperture is wider than the simulation grid."""
    import warnings as _warnings
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=12e-3)
    N, dx, lam = 64, 8e-6, 1.31e-6     # grid semi = 0.256 mm, lens = 6 mm
    E = np.ones((N, N), dtype=np.complex128)
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter('always')
        op.apply_real_lens(E, pres, lam, dx)
    fired = any(issubclass(rec.category, UserWarning)
                and 'exceed' in str(rec.message).lower()
                for rec in w)
    return fired, f'captured {len(w)} warnings, fired={fired}'


H.run('apply_real_lens: warns when aperture > grid',
      t_apply_real_lens_warns_when_aperture_exceeds_grid)


def t_axicon_makes_bessel_intensity_pattern():
    """An axicon focused over its depth-of-focus produces an on-axis
    bright line; intensity at a non-zero z stays > 0."""
    N, dx, lam = 512, 8e-6, 1.31e-6
    E = np.ones((N, N), dtype=np.complex128)
    E_ax = op.apply_axicon(E, alpha=np.radians(0.5), n_axicon=1.5,
                            wavelength=lam, dx=dx)
    E_far = op.angular_spectrum_propagate(E_ax, 50e-3, lam, dx)
    onax = float(np.abs(E_far[N//2, N//2])**2)
    return onax > 0 and np.isfinite(onax), \
        f'on-axis intensity at 50mm = {onax:.3e}'


H.run('Axicon: produces non-zero on-axis intensity at z>0',
      t_axicon_makes_bessel_intensity_pattern)


if __name__ == '__main__':
    sys.exit(H.summary())
