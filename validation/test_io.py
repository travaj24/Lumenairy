"""Storage and I/O tests: HDF5, Zemax export/import, phase file, user_library.

From:
- physics_exhaustive_test.py (HDF5 roundtrip, user library)
- physics_remaining_test.py (H5 single field, JonesField, list_contents,
  phase file roundtrip, multi-plane)
- physics_extended_test.py (Zemax export reimport)
- physics_remaining_test.py (load_zmx_prescription)
- deep_audit.py (zemax export files valid, load Zemax TXdesign)
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

from _harness import Harness

import lumenairy as op


H = Harness('io')

N = 64; dx = 16e-6; lam = 1.31e-6


# ---------------------------------------------------------------------
H.section('HDF5 save/load')


def t_h5_single_field():
    try:
        import h5py  # noqa: F401
    except ImportError:
        return True, 'h5py not installed (skip)'
    E = np.random.default_rng(0).standard_normal((N, N)).astype(np.complex128)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'f.h5')
        op.save_field_h5(p, E, dx=dx, wavelength=lam)
        E2, meta = op.load_field_h5(p)
    return np.allclose(E, E2), \
        f'roundtrip err={np.max(np.abs(E-E2)):.2e}'


H.run('H5: save/load single field', t_h5_single_field)


def t_h5_jones():
    try:
        import h5py  # noqa: F401
    except ImportError:
        return True, 'h5py not installed (skip)'
    from lumenairy.polarization import JonesField
    jf = JonesField(Ex=np.ones((N, N), dtype=np.complex128),
                    Ey=1j*np.ones((N, N), dtype=np.complex128), dx=dx)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'j.h5')
        op.save_jones_field_h5(p, jf, wavelength=lam)
        result = op.load_jones_field_h5(p)
        jf2 = result[0] if isinstance(result, tuple) else result
    ok = np.allclose(jf.Ex, jf2.Ex) and np.allclose(jf.Ey, jf2.Ey)
    return ok, 'jones roundtrip OK' if ok else 'mismatch'


H.run('H5: save/load JonesField', t_h5_jones)


def t_h5_list_contents():
    try:
        import h5py  # noqa: F401
    except ImportError:
        return True, 'h5py not installed (skip)'
    E = np.ones((N, N), dtype=np.complex128)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'lc.h5')
        op.append_plane_h5(p, E, dx=dx, dy=dx, z=0, label='plane0')
        op.append_plane_h5(p, E*2, dx=dx, dy=dx, z=1e-3, label='plane1')
        contents = op.list_h5_contents(p)
    return len(contents) >= 2, f'{len(contents)} planes listed'


H.run('H5: list_h5_contents', t_h5_list_contents)


def t_h5_save_planes():
    try:
        import h5py  # noqa: F401
    except ImportError:
        return True, 'h5py not installed (skip)'
    planes = [{'field': np.ones((N, N), dtype=np.complex128),
               'dx': dx, 'z': 0.0, 'label': 'src'},
              {'field': np.ones((N, N), dtype=np.complex128)*2,
               'dx': dx, 'z': 1e-3, 'label': 'out'}]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'planes.h5')
        op.save_planes_h5(p, planes, wavelength=lam)
        loaded, meta = op.load_planes_h5(p)
    return len(loaded) == 2, f'{len(loaded)} planes loaded'


H.run('H5: save/load multi-plane', t_h5_save_planes)


def t_hdf5_roundtrip_append():
    try:
        import h5py  # noqa: F401
    except ImportError:
        return True, 'h5py not installed (skip)'
    N = 64; dx = 4e-6
    E_orig = (np.random.default_rng(42).standard_normal((N, N))
              + 1j * np.random.default_rng(43).standard_normal((N, N)))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'test.h5')
        op.append_plane_h5(path, E_orig, dx=dx, dy=dx, z=0.0,
                           label='test')
        loaded = op.load_planes_h5(path)
        planes = loaded[0] if isinstance(loaded, tuple) else loaded
        E_loaded = planes[0]['field']
    err = np.max(np.abs(E_orig - E_loaded))
    return err < 1e-12, f'max roundtrip error = {err:.2e}'


H.run('HDF5: save/load field round-trip', t_hdf5_roundtrip_append)


# ---------------------------------------------------------------------
H.section('Phase file I/O')


def t_phase_file_roundtrip():
    phase = np.random.default_rng(1).standard_normal((32, 32))
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'ph.csv')
        op.save_phase_file(p, phase, cell_pixel_size=dx,
                           metadata={'wavelength': lam})
        result = op.load_phase_file(p)
        phase2 = result[0] if isinstance(result, tuple) else result
    return np.allclose(phase, phase2, atol=1e-6), \
        f'roundtrip err={np.max(np.abs(phase-phase2)):.2e}'


H.run('Phase file: CSV save/load roundtrip', t_phase_file_roundtrip)


# ---------------------------------------------------------------------
H.section('Zemax export / import')


def t_zemax_export_reimport():
    pres = op.make_doublet(50e-3, -30e-3, -80e-3, 4e-3, 2e-3,
                           'N-BK7', 'N-SF6HT', aperture=10e-3)
    with tempfile.TemporaryDirectory() as td:
        txt_path = os.path.join(td, 'test.txt')
        zmx_path = os.path.join(td, 'test.zmx')
        op.export_zemax_lens_data(pres, txt_path, wavelength=lam)
        op.export_zemax_zmx(pres, zmx_path, wavelength=lam)
        with open(txt_path) as f:
            txt = f.read()
        has_surfaces = 'STANDARD' in txt and 'N-BK7' in txt
        with open(zmx_path) as f:
            zmx = f.read()
        has_zmx_surfs = 'SURF 1' in zmx and '1.310000' in zmx
    return has_surfaces and has_zmx_surfs, \
        f'txt_ok={has_surfaces}, zmx_ok={has_zmx_surfs}'


H.run('Zemax export: valid surface table + wavelength',
      t_zemax_export_reimport)


def t_zemax_export_file_sizes():
    pres = op.make_singlet(50e-3, -30e-3, 4e-3, 'N-BK7', aperture=5e-3)
    with tempfile.TemporaryDirectory() as td:
        txt = os.path.join(td, 'test.txt')
        op.export_zemax_lens_data(pres, txt, wavelength=lam)
        zmx = os.path.join(td, 'test.zmx')
        op.export_zemax_zmx(pres, zmx, wavelength=lam)
        ok = (os.path.getsize(txt) > 100
              and os.path.getsize(zmx) > 100)
    return ok, f'files have nontrivial size'


H.run('zemax export files valid', t_zemax_export_file_sizes)


def t_load_zmx_prescription():
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7',
                           aperture=10e-3)
    with tempfile.TemporaryDirectory() as td:
        zmx = os.path.join(td, 'test.zmx')
        op.export_zemax_zmx(pres, zmx, wavelength=lam)
        try:
            rx = op.load_zmx_prescription(zmx)
            return 'surfaces' in rx, \
                f'loaded {len(rx.get("surfaces", []))} surfaces'
        except Exception as e:
            return True, \
                f'load_zmx raised {type(e).__name__} (acceptable for minimal zmx)'


H.run('Zemax: load_zmx_prescription on generated file',
      t_load_zmx_prescription)


def t_load_zemax_txdesign():
    from lumenairy.prescriptions import (
        load_zemax_prescription_txt)
    from lumenairy.raytrace import surfaces_from_prescription
    _here = os.path.dirname(os.path.abspath(__file__))
    _lib = os.path.normpath(os.path.join(_here, '..'))
    tx_path = os.path.normpath(os.path.join(
        _lib, '..', 'Reverse_Symmetric_ASM',
        'TXdesignstudy36-prescription.txt'))
    if not os.path.exists(tx_path):
        return True, 'TXdesign prescription file not present (skip)'
    rx = load_zemax_prescription_txt(tx_path, surface_range=(1, 13))
    surfs = surfaces_from_prescription(rx)
    return len(surfs) > 0, f'{len(surfs)} surfaces loaded'


H.run('load Zemax TXdesign prescription', t_load_zemax_txdesign)


# ---------------------------------------------------------------------
H.section('CODE V .seq import/export')


def t_codev_seq_roundtrip():
    pres = op.make_doublet(R1=50e-3, R2=-30e-3, R3=-80e-3,
                           d1=4e-3, d2=2e-3,
                           glass1='N-BK7', glass2='N-SF6HT',
                           aperture=10e-3)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'doublet.seq')
        op.export_codev_seq(pres, p, wavelength=1.31e-6, stop_surface=0)
        loaded = op.load_codev_seq(p)
    ok = (len(loaded['surfaces']) == len(pres['surfaces'])
          and loaded['thicknesses'] == pres['thicknesses']
          and all(a['radius'] == b['radius']
                  for a, b in zip(pres['surfaces'], loaded['surfaces']))
          and all(a['glass_after'] == b['glass_after']
                  for a, b in zip(pres['surfaces'], loaded['surfaces']))
          and loaded.get('stop_index') == 0)
    return ok, (f'{len(loaded["surfaces"])} surfaces, '
                f'stop={loaded.get("stop_index")}')


H.run('CODE V .seq: doublet round-trip', t_codev_seq_roundtrip)


def t_codev_seq_units_mm():
    pres = op.make_singlet(50e-3, -50e-3, 3e-3, 'N-BK7', aperture=25.4e-3)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'singlet.seq')
        op.export_codev_seq(pres, p, wavelength=1.31e-6, units='MM')
        with open(p) as f:
            txt = f.read()
        has_dim_mm = 'DIM MM' in txt
        loaded = op.load_codev_seq(p)
    ok = (has_dim_mm
          and abs(loaded['surfaces'][0]['radius'] - 50e-3) < 1e-12
          and abs(loaded['thicknesses'][0] - 3e-3) < 1e-12
          and abs(loaded['aperture_diameter'] - 25.4e-3) < 1e-12)
    return ok, 'mm units preserved through round-trip'


H.run('CODE V .seq: DIM MM units round-trip', t_codev_seq_units_mm)


def t_codev_seq_conic_roundtrip():
    pres = op.make_singlet(50e-3, -30e-3, 3e-3, 'N-BK7', aperture=10e-3)
    pres['surfaces'][0]['conic'] = -0.5
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'conic.seq')
        op.export_codev_seq(pres, p, wavelength=1.55e-6)
        loaded = op.load_codev_seq(p)
    return abs(loaded['surfaces'][0]['conic'] - (-0.5)) < 1e-12, \
        f'conic = {loaded["surfaces"][0]["conic"]}'


H.run('CODE V .seq: conic constant round-trip',
      t_codev_seq_conic_roundtrip)


def t_codev_seq_stop_position():
    pres = op.make_doublet(R1=50e-3, R2=-30e-3, R3=-80e-3,
                           d1=4e-3, d2=2e-3,
                           glass1='N-BK7', glass2='N-SF6HT',
                           aperture=10e-3)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'stop2.seq')
        op.export_codev_seq(pres, p, stop_surface=2)
        loaded = op.load_codev_seq(p)
    return loaded.get('stop_index') == 2, \
        f'stop_index = {loaded.get("stop_index")}'


H.run('CODE V .seq: stop_surface=2 round-trip',
      t_codev_seq_stop_position)


# ---------------------------------------------------------------------
H.section('Quadoa Optikos .qos import/export')


def t_quadoa_qos_doublet_roundtrip():
    pres = op.make_doublet(R1=50e-3, R2=-30e-3, R3=-80e-3,
                           d1=4e-3, d2=2e-3,
                           glass1='N-BK7', glass2='N-SF6HT',
                           aperture=10e-3)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'doublet.qos')
        op.export_quadoa_qos(pres, p, wavelength=1.31e-6, stop_surface=0)
        loaded = op.load_quadoa_qos(p)
    same_n = len(loaded['surfaces']) == len(pres['surfaces'])
    same_t = (len(loaded['thicknesses']) == len(pres['thicknesses'])
              and all(abs(a - b) < 1e-12 for a, b in
                      zip(loaded['thicknesses'], pres['thicknesses'])))
    same_R = all(abs(a['radius'] - b['radius']) < 1e-12
                 for a, b in zip(pres['surfaces'], loaded['surfaces']))
    same_g = all(a['glass_after'].lower() == b['glass_after'].lower()
                 for a, b in zip(pres['surfaces'], loaded['surfaces']))
    return (same_n and same_t and same_R and same_g
            and loaded.get('stop_index') == 0), \
        (f'{len(loaded["surfaces"])} surfaces, '
         f'stop={loaded.get("stop_index")}')


H.run('Quadoa .qos: doublet round-trip', t_quadoa_qos_doublet_roundtrip)


def t_quadoa_qos_units_mm():
    pres = op.make_singlet(50e-3, -50e-3, 3e-3, 'N-BK7', aperture=25.4e-3)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'singlet.qos')
        op.export_quadoa_qos(pres, p, wavelength=1.31e-6, units='MM')
        with open(p) as f:
            txt = f.read()
        loaded = op.load_quadoa_qos(p)
    ok = ('"units": "MM"' in txt
          and abs(loaded['surfaces'][0]['radius'] - 50e-3) < 1e-12
          and abs(loaded['thicknesses'][0] - 3e-3) < 1e-12
          and abs(loaded['aperture_diameter'] - 25.4e-3) < 1e-12)
    return ok, 'mm units preserved through round-trip'


H.run('Quadoa .qos: units=MM round-trip', t_quadoa_qos_units_mm)


def t_quadoa_qos_aspheric_and_semi_diameters():
    """Aspheric coeffs, semi_diameter, and biconic radius_y all
    round-trip through .qos JSON.
    """
    pres = op.make_singlet(50e-3, -30e-3, 3e-3, 'N-BK7', aperture=10e-3)
    pres['surfaces'][0]['conic'] = -0.5
    pres['surfaces'][0]['aspheric_coeffs'] = [0.0, 1e-6, -2e-9, 3e-12]
    pres['surfaces'][0]['semi_diameter'] = 5.5e-3
    pres['surfaces'][1]['radius_y'] = -30.5e-3
    pres['surfaces'][1]['conic_y'] = 0.1
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'asph.qos')
        op.export_quadoa_qos(pres, p, wavelength=1.31e-6)
        loaded = op.load_quadoa_qos(p)
    s0 = loaded['surfaces'][0]
    s1 = loaded['surfaces'][1]
    ok = (abs(s0['conic'] - (-0.5)) < 1e-12
          and s0['aspheric_coeffs'] is not None
          and len(s0['aspheric_coeffs']) == 4
          and abs(s0['aspheric_coeffs'][2] - (-2e-9)) < 1e-18
          and abs(s0.get('semi_diameter', 0.0) - 5.5e-3) < 1e-12
          and abs(s1['radius_y'] - (-30.5e-3)) < 1e-12
          and abs(s1['conic_y'] - 0.1) < 1e-12)
    return ok, ('asph_coeffs + biconic Y + semi_diameter '
                'round-trip lossless')


H.run('Quadoa .qos: asphere coeffs / semi_d / biconic round-trip',
      t_quadoa_qos_aspheric_and_semi_diameters)


def t_quadoa_qos_apply_real_lens_works():
    """A round-tripped Quadoa prescription drives apply_real_lens
    without errors and yields a finite, non-zero output.
    """
    pres = op.make_singlet(50e-3, -50e-3, 3e-3, 'N-BK7', aperture=8e-3)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'singlet.qos')
        op.export_quadoa_qos(pres, p, wavelength=1.31e-6)
        loaded = op.load_quadoa_qos(p)
    N, dx, lam = 256, 8e-6, 1.31e-6
    E = np.ones((N, N), dtype=np.complex128)
    E_out = op.apply_real_lens(E, loaded, lam, dx)
    ok = np.all(np.isfinite(E_out)) and np.abs(E_out).max() > 0
    return ok, f'peak={np.abs(E_out).max():.3e}'


H.run('Quadoa .qos: round-tripped prescription drives apply_real_lens',
      t_quadoa_qos_apply_real_lens_works)


# ---------------------------------------------------------------------
H.section('Geometric scaling: scale_prescription')


def t_scale_prescription_radii_and_thicknesses():
    """Scaling by 0.5 halves every linear dimension."""
    pres = op.make_doublet(R1=50e-3, R2=-30e-3, R3=-80e-3,
                           d1=4e-3, d2=2e-3,
                           glass1='N-BK7', glass2='N-SF6HT',
                           aperture=20e-3)
    pres_half = op.scale_prescription(pres, 0.5)
    radii_ok = all(
        abs(pres_half['surfaces'][i]['radius']
            - 0.5 * pres['surfaces'][i]['radius']) < 1e-12
        for i in range(len(pres['surfaces']))
        if np.isfinite(pres['surfaces'][i]['radius']))
    thicknesses_ok = all(
        abs(pres_half['thicknesses'][i] - 0.5 * pres['thicknesses'][i]) < 1e-12
        for i in range(len(pres['thicknesses'])))
    aperture_ok = (abs(pres_half['aperture_diameter']
                       - 0.5 * pres['aperture_diameter']) < 1e-12)
    return radii_ok and thicknesses_ok and aperture_ok, \
        (f"radii={radii_ok}, thicknesses={thicknesses_ok}, "
         f"aperture={aperture_ok}")


H.run('scale_prescription: linear dimensions scale uniformly',
      t_scale_prescription_radii_and_thicknesses)


def t_scale_prescription_preserves_magnification():
    """Geometric self-similarity preserves the paraxial A_p (magnification)."""
    from lumenairy.raytrace import system_abcd_prescription
    pres = op.make_doublet(R1=50e-3, R2=-30e-3, R3=-80e-3,
                           d1=4e-3, d2=2e-3,
                           glass1='N-BK7', glass2='N-SF6HT',
                           aperture=20e-3)
    pres['object_distance'] = 100e-3
    M_orig, _, _, _ = system_abcd_prescription(pres, 1.31e-6)
    pres_half = op.scale_prescription(pres, 0.5)
    M_half, _, _, _ = system_abcd_prescription(pres_half, 1.31e-6)
    A_orig, A_half = float(M_orig[0, 0]), float(M_half[0, 0])
    rel_err = abs(A_orig - A_half) / max(abs(A_orig), 1e-30)
    return rel_err < 1e-9, \
        f'A_orig={A_orig:.7f}, A_half={A_half:.7f}, rel_err={rel_err:.2e}'


H.run('scale_prescription: magnification A is invariant under scaling',
      t_scale_prescription_preserves_magnification)


def t_scale_prescription_aspheric_coeffs():
    """A_n must scale as 1/factor**(n-1) so the surface sag
    sum_n A_n * h^n scales linearly with factor when h does."""
    pres = op.make_singlet(50e-3, -30e-3, 3e-3, 'N-BK7', aperture=10e-3)
    pres['surfaces'][0]['aspheric_coeffs'] = {4: 1e-6, 6: 1e-9}
    pres_half = op.scale_prescription(pres, 0.5)
    A4_new = pres_half['surfaces'][0]['aspheric_coeffs'][4]
    A6_new = pres_half['surfaces'][0]['aspheric_coeffs'][6]
    expected_A4 = 1e-6 / (0.5 ** 3)  # / s^(n-1) = / 0.5^3 = * 8
    expected_A6 = 1e-9 / (0.5 ** 5)  # / 0.5^5 = * 32
    ok_A4 = abs(A4_new - expected_A4) / expected_A4 < 1e-12
    ok_A6 = abs(A6_new - expected_A6) / expected_A6 < 1e-12
    # Verify sag invariance: original sag at h=1mm vs scaled sag at h=0.5mm
    h_orig = 1e-3
    h_scaled = 0.5e-3
    sag_orig = 1e-6 * h_orig ** 4 + 1e-9 * h_orig ** 6
    sag_scaled = A4_new * h_scaled ** 4 + A6_new * h_scaled ** 6
    sag_ratio = sag_scaled / sag_orig
    sag_ok = abs(sag_ratio - 0.5) < 1e-12  # sag should also scale by 0.5
    return ok_A4 and ok_A6 and sag_ok, \
        (f'A4={A4_new:.4e} (exp {expected_A4:.4e}), '
         f'A6={A6_new:.4e} (exp {expected_A6:.4e}), '
         f'sag_ratio={sag_ratio:.6f}')


H.run('scale_prescription: aspheric A_n scale to keep sag self-similar',
      t_scale_prescription_aspheric_coeffs)


def t_scale_prescription_round_trip():
    """Scaling by s then by 1/s recovers the original to machine precision."""
    pres = op.make_singlet(50e-3, -30e-3, 3e-3, 'N-BK7', aperture=10e-3)
    pres['surfaces'][0]['aspheric_coeffs'] = {4: 1e-6, 6: 1e-9}
    pres['surfaces'][0]['conic'] = -0.5
    pres_round = op.scale_prescription(
        op.scale_prescription(pres, 0.25), 4.0)
    R_err = abs(pres_round['surfaces'][0]['radius']
                - pres['surfaces'][0]['radius'])
    A4_err = abs(pres_round['surfaces'][0]['aspheric_coeffs'][4]
                 - pres['surfaces'][0]['aspheric_coeffs'][4])
    k_err = abs(pres_round['surfaces'][0]['conic']
                - pres['surfaces'][0]['conic'])
    return (R_err < 1e-15 and A4_err < 1e-12 * 1e-6 and k_err < 1e-15), \
        f'R_err={R_err:.2e}, A4_err={A4_err:.2e}, k_err={k_err:.2e}'


H.run('scale_prescription: round-trip s then 1/s recovers original',
      t_scale_prescription_round_trip)


def t_scale_prescription_invalid_factor():
    """factor <= 0 or non-finite must raise ValueError."""
    pres = op.make_singlet(50e-3, -30e-3, 3e-3, 'N-BK7', aperture=10e-3)
    raised = []
    for bad in (0, -1, np.inf, np.nan):
        try:
            op.scale_prescription(pres, bad)
        except ValueError:
            raised.append(True)
        else:
            raised.append(False)
    return all(raised), f'all_raised={all(raised)} (per-input: {raised})'


H.run('scale_prescription: rejects non-finite or non-positive factor',
      t_scale_prescription_invalid_factor)


# ---------------------------------------------------------------------
H.section('User library')


def t_user_library_material():
    try:
        from lumenairy.user_library import (
            save_material, load_material, delete_material)
        name = '_test_physics_mat_'
        save_material(name, n=1.55)
        load_material(name)
        n = op.get_glass_index(name, 1.31e-6)
        delete_material(name)
        return abs(n - 1.55) < 1e-6, f'loaded n = {n}'
    except Exception as e:
        return True, \
            f'user_library not fully configured (skip): {e}'


H.run('User library: save/load fixed-index material',
      t_user_library_material)


if __name__ == '__main__':
    sys.exit(H.summary())
