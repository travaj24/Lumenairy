"""Test-lens definitions for the real-lens OPD validation.

Each case is a dict with:
    name       : short identifier (used in filenames and plot titles)
    description: human-readable label
    prescription: lens prescription dict (from make_singlet / make_doublet /
                  thorlabs_lens, or built by hand)
    aperture   : clear aperture diameter [m] -- overrides whatever is
                 in the prescription.  Usually set smaller than the
                 physical aperture to keep grid sampling tractable for
                 fast lenses.
    wavelength : vacuum wavelength [m]
    N          : grid samples (square grid)
    dx         : grid spacing [m]
    category   : 'singlet' | 'doublet' | 'meniscus' | 'fnum_sweep' | ...
                 used to group the final report.

Grid sizing rule of thumb
-------------------------
The wavefront exiting a lens of focal length ``f`` has paraxial phase
``-k*r^2/(2f)``.  The phase gradient at the edge (r = ap/2) is
``d(phi)/dr = k*(ap/2)/f``.  To keep np.unwrap happy along a 1D slice
we need dx < pi / |d(phi)/dr|, i.e.

    dx_max = lambda * f / ap

Plus grid extent must satisfy N*dx > aperture + a little guard band.
For a 10 mm aperture at 1.31 um, N*dx should be about 16 mm.
"""

from __future__ import annotations

import numpy as np

import lumenairy as op
from lumenairy.prescriptions import (
    make_singlet,
    make_doublet,
    thorlabs_lens,
)


# =========================================================================
# Helpers
# =========================================================================

def _custom_singlet(name, R1, R2, d, glass, aperture, wavelength=1.31e-6,
                    N=8192, dx=2e-6, category='singlet', description=None):
    return {
        'name': name,
        'description': description or name,
        'prescription': make_singlet(R1, R2, d, glass,
                                     aperture=aperture, name=name),
        'aperture': aperture,
        'wavelength': wavelength,
        'N': N,
        'dx': dx,
        'category': category,
    }


def _thorlabs_case(part, aperture, wavelength=1.31e-6, N=8192, dx=2e-6,
                   description=None):
    return {
        'name': part.replace('-', '_'),
        'description': description or f'Thorlabs {part}',
        'prescription': thorlabs_lens(part),
        'aperture': aperture,
        'wavelength': wavelength,
        'N': N,
        'dx': dx,
        'category': 'thorlabs',
    }


# =========================================================================
# Case list
# =========================================================================

CASES = []

# ------------------------------------------------------------------
# Singlets -- vary curvature, shape, glass
# ------------------------------------------------------------------
CASES += [
    # Plano-convex, moderate f/#
    _custom_singlet(
        'plano_convex_R50_BK7',
        R1=50e-3, R2=np.inf, d=4.0e-3, glass='N-BK7',
        aperture=10.0e-3,
        description='Plano-convex R=50 mm, N-BK7, d=4 mm (f ~ 97 mm, f/9.7)'),

    # Equi-convex, symmetric
    _custom_singlet(
        'equi_convex_R50_BK7',
        R1=50e-3, R2=-50e-3, d=5.0e-3, glass='N-BK7',
        aperture=10.0e-3,
        description='Equi-convex |R|=50 mm, N-BK7, d=5 mm (f ~ 50 mm, f/5)'),

    # Biconcave, diverging
    _custom_singlet(
        'biconcave_R50_BK7',
        R1=-50e-3, R2=50e-3, d=2.5e-3, glass='N-BK7',
        aperture=10.0e-3,
        description='Bi-concave |R|=50 mm (f ~ -50 mm, diverging)'),

    # Meniscus (bending, stresses slant correction)
    _custom_singlet(
        'meniscus_positive_BK7',
        R1=30e-3, R2=60e-3, d=3.0e-3, glass='N-BK7',
        aperture=10.0e-3,
        description='Positive meniscus R1=30, R2=60 mm (f ~ 110 mm)'),

    _custom_singlet(
        'meniscus_negative_BK7',
        R1=60e-3, R2=30e-3, d=3.0e-3, glass='N-BK7',
        aperture=10.0e-3,
        description='Negative meniscus R1=60, R2=30 mm (f ~ -110 mm)'),

    # High-index (N-SF6HT): stronger refraction at same curvature
    _custom_singlet(
        'plano_convex_R50_SF6',
        R1=50e-3, R2=np.inf, d=4.0e-3, glass='N-SF6HT',
        aperture=10.0e-3,
        description='Plano-convex R=50 mm, N-SF6HT (high index)'),

    # Thick, tests intra-glass ASM leg
    _custom_singlet(
        'plano_convex_thick_BK7',
        R1=50e-3, R2=np.inf, d=10.0e-3, glass='N-BK7',
        aperture=10.0e-3,
        description='Plano-convex R=50 mm, N-BK7, d=10 mm (thick)'),
]

# ------------------------------------------------------------------
# Thorlabs catalog
# ------------------------------------------------------------------
CASES += [
    # NB: aperture chosen <= grid extent so the wave aperture mask
    # is meaningful.  Thorlabs LA series mechanical aperture is
    # 25.4 mm; we use 12-15 mm for the validation to keep apertures
    # comfortably inside the simulation grid (16.38 mm typ).
    _thorlabs_case('LA1050-C',   aperture=15.0e-3,
                   description='LA1050-C plano-convex f=100 mm'),
    _thorlabs_case('LA1509-C',   aperture=15.0e-3,
                   description='LA1509-C plano-convex f=200 mm'),
    _thorlabs_case('LA1301-C',   aperture=15.0e-3,
                   description='LA1301-C plano-convex f=250 mm'),
    _thorlabs_case('AC254-050-C', aperture=12.0e-3, dx=1e-6, N=16384,
                   description='AC254-050-C achromat f=50 mm'),
    _thorlabs_case('AC254-100-C', aperture=15.0e-3,
                   description='AC254-100-C achromat f=100 mm'),
    _thorlabs_case('AC254-200-C', aperture=15.0e-3,
                   description='AC254-200-C achromat f=200 mm'),
]

# ------------------------------------------------------------------
# f/# sweep on plano-convex (stresses slant correction progressively)
# ------------------------------------------------------------------
# Keep glass and aperture fixed; vary R1 to change f/#.
_fnum_ap = 8.0e-3
_fnum_glass = 'N-BK7'
_fnum_d = 3.0e-3

for fnum_label, R1, dx_use, N_use in [
    ('f20', 200e-3, 2e-6, 8192),
    ('f10', 100e-3, 2e-6, 8192),
    ('f5',   50e-3, 1e-6, 16384),
    ('f3',   30e-3, 1e-6, 16384),
    ('f2',   20e-3, 0.5e-6, 32768),
]:
    CASES.append(_custom_singlet(
        f'fnum_sweep_{fnum_label}_R{int(R1*1e3)}',
        R1=R1, R2=np.inf, d=_fnum_d, glass=_fnum_glass,
        aperture=_fnum_ap,
        N=N_use, dx=dx_use,
        category='fnum_sweep',
        description=(f'Plano-convex f/# sweep: R1={R1*1e3:.0f} mm, '
                     f'ap={_fnum_ap*1e3:.0f} mm')))

# ------------------------------------------------------------------
# Wavelength sweep on AC254-100-C (achromat across its design band)
# ------------------------------------------------------------------
for lam_nm, lam_label in [
    (1064, '1064nm'),
    (1310, '1310nm'),
    (1550, '1550nm'),
]:
    CASES.append({
        'name': f'AC254_100_C_{lam_label}',
        'description': f'AC254-100-C @ {lam_nm} nm',
        'prescription': thorlabs_lens('AC254-100-C'),
        'aperture': 20.0e-3,
        'wavelength': lam_nm * 1e-9,
        'N': 8192,
        'dx': 2e-6,
        'category': 'wavelength_sweep',
    })


def list_cases():
    return CASES


def filter_cases(category=None, name_substr=None):
    out = CASES
    if category is not None:
        out = [c for c in out if c['category'] == category]
    if name_substr is not None:
        out = [c for c in out if name_substr in c['name']]
    return out
