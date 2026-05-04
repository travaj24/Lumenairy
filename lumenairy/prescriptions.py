"""
Lens prescription construction and Zemax import utilities.

Provides functions to build lens prescription dicts (singlets, doublets),
a catalog of Thorlabs stock lenses, and a parser for Zemax .zmx sequential
lens files.  All prescriptions use glass name strings rather than numeric
refractive indices so they remain wavelength-independent; indices are
resolved at runtime by the propagation engine.

Author: Andrew Traverso
"""

import os
import warnings

import numpy as np

from .glass import GLASS_REGISTRY


# ============================================================================
# Prescription builders
# ============================================================================

def make_singlet(R1, R2, d, glass, aperture=25.4e-3, name=None):
    """Build a lens prescription dict for a singlet lens.

    Parameters
    ----------
    R1 : float
        Front surface radius of curvature [m].  Use ``float('inf')`` for flat.
    R2 : float
        Back surface radius of curvature [m].  Use ``float('inf')`` for flat.
    d : float
        Center thickness [m].
    glass : str
        Glass name from ``GLASS_REGISTRY`` (e.g. ``'N-BK7'``).
    aperture : float, optional
        Clear aperture diameter [m].  Default 25.4 mm.
    name : str, optional
        Human-readable label.

    Returns
    -------
    prescription : dict
    """
    return {
        'name': name or f'Singlet ({glass})',
        'aperture_diameter': aperture,
        'surfaces': [
            {'radius': R1, 'conic': 0.0, 'aspheric_coeffs': None,
             'radius_y': None, 'conic_y': None,
             'aspheric_coeffs_y': None,
             'glass_before': 'air', 'glass_after': glass},
            {'radius': R2, 'conic': 0.0, 'aspheric_coeffs': None,
             'radius_y': None, 'conic_y': None,
             'aspheric_coeffs_y': None,
             'glass_before': glass, 'glass_after': 'air'},
        ],
        'thicknesses': [d],
    }


def make_cylindrical(R_focus, d, glass, axis='x', aperture=25.4e-3,
                     name=None):
    """Build a cylindrical-lens prescription (flat on one face, curved
    cylindrical on the other).

    A cylindrical lens focuses in one axis only.  Use ``axis='x'`` for
    a lens that focuses along the x-direction (cross-section in x has
    radius ``R_focus``, cross-section in y is flat).  For focusing in
    y use ``axis='y'``.

    Parameters
    ----------
    R_focus : float
        Radius of curvature of the focusing axis [m]; positive for
        converging.
    d : float
        Center thickness [m].
    glass : str
        Glass name.
    axis : ``'x'`` or ``'y'``
        Which axis has curvature.  The other axis is flat.
    aperture : float
        Clear aperture diameter [m].
    name : str, optional
    """
    if axis == 'x':
        R_x, R_y = R_focus, float('inf')
    elif axis == 'y':
        R_x, R_y = float('inf'), R_focus
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    return {
        'name': name or f'CylindricalLens-{axis}({glass})',
        'aperture_diameter': aperture,
        'surfaces': [
            {'radius': R_x, 'radius_y': R_y,
             'conic': 0.0, 'conic_y': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': glass},
            {'radius': float('inf'), 'radius_y': float('inf'),
             'conic': 0.0, 'conic_y': 0.0, 'aspheric_coeffs': None,
             'glass_before': glass, 'glass_after': 'air'},
        ],
        'thicknesses': [d],
    }


def make_biconic(R1_x, R1_y, R2_x, R2_y, d, glass,
                 conic1_x=0.0, conic1_y=0.0,
                 conic2_x=0.0, conic2_y=0.0,
                 aperture=25.4e-3, name=None):
    """Build a biconic (anamorphic) singlet prescription.

    Each surface has independent x- and y-axis radii and conics -- the
    sag is
    ``z(x,y) = C_x x² / (1 + sqrt(1 - (1+K_x) C_x² x²))
             + C_y y² / (1 + sqrt(1 - (1+K_y) C_y² y²))``
    with ``C = 1/R``.  Covers cylindrical, toroidal, and freeform
    biconic elements used in anamorphic imaging and beam shaping.

    Pass ``inf`` for any radius to make that axis flat.  If ``R1_y``
    (or ``R2_y``) equals the corresponding ``R1_x`` (``R2_x``) and the
    conics match, the surface reduces to rotationally-symmetric --
    in that case use :func:`make_singlet` instead for efficiency.

    Parameters
    ----------
    R1_x, R1_y : float
        Front-surface x- and y-axis radii [m].
    R2_x, R2_y : float
        Back-surface x- and y-axis radii [m].
    d : float
        Center thickness [m].
    glass : str
    conic1_x, conic1_y, conic2_x, conic2_y : float
        Per-axis conic constants (0 = spherical in that axis).
    aperture : float
    name : str, optional
    """
    return {
        'name': name or f'Biconic({glass})',
        'aperture_diameter': aperture,
        'surfaces': [
            {'radius': R1_x, 'radius_y': R1_y,
             'conic': conic1_x, 'conic_y': conic1_y,
             'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': glass},
            {'radius': R2_x, 'radius_y': R2_y,
             'conic': conic2_x, 'conic_y': conic2_y,
             'aspheric_coeffs': None,
             'glass_before': glass, 'glass_after': 'air'},
        ],
        'thicknesses': [d],
    }


def make_doublet(R1, R2, R3, d1, d2, glass1, glass2,
                 aperture=25.4e-3, name=None):
    """Build a lens prescription dict for a cemented achromatic doublet.

    Parameters
    ----------
    R1, R2, R3 : float
        Radii of curvature [m] for the three surfaces.
    d1, d2 : float
        Center thicknesses [m] of the two elements.
    glass1, glass2 : str
        Glass names from ``GLASS_REGISTRY`` for elements 1 and 2.
    aperture : float, optional
        Clear aperture diameter [m].  Default 25.4 mm.
    name : str, optional
        Human-readable label.

    Returns
    -------
    prescription : dict
    """
    return {
        'name': name or f'Doublet ({glass1}/{glass2})',
        'aperture_diameter': aperture,
        'surfaces': [
            {'radius': R1, 'conic': 0.0, 'aspheric_coeffs': None,
             'radius_y': None, 'conic_y': None,
             'aspheric_coeffs_y': None,
             'glass_before': 'air', 'glass_after': glass1},
            {'radius': R2, 'conic': 0.0, 'aspheric_coeffs': None,
             'radius_y': None, 'conic_y': None,
             'aspheric_coeffs_y': None,
             'glass_before': glass1, 'glass_after': glass2},
            {'radius': R3, 'conic': 0.0, 'aspheric_coeffs': None,
             'radius_y': None, 'conic_y': None,
             'aspheric_coeffs_y': None,
             'glass_before': glass2, 'glass_after': 'air'},
        ],
        'thicknesses': [d1, d2],
    }


# ============================================================================
# Thorlabs catalog lens presets
# ============================================================================
# Surface data from Thorlabs Zemax files.  All dimensions in meters.
# Sign convention: positive R = center of curvature to the right.

THORLABS_CATALOG = {
    # --- Plano-convex singlets (C-coated = 1050-1700 nm) ---
    'LA1050-C': {  # f=100mm, N-BK7, 1" dia
        'type': 'singlet',
        'R1': 51.5e-3, 'R2': float('inf'),
        'd': 4.1e-3, 'glass': 'N-BK7', 'aperture': 25.4e-3,
    },
    'LA1509-C': {  # f=200mm, N-BK7, 1" dia (curved side first for collimation)
        'type': 'singlet',
        'R1': 103.29e-3, 'R2': float('inf'),
        'd': 3.6e-3, 'glass': 'N-BK7', 'aperture': 25.4e-3,
    },
    'LA1301-C': {  # f=250mm, N-BK7, 1" dia
        'type': 'singlet',
        'R1': 129.2e-3, 'R2': float('inf'),
        'd': 3.4e-3, 'glass': 'N-BK7', 'aperture': 25.4e-3,
    },
    # --- Achromatic doublets (C-coated) ---
    'AC254-050-C': {  # f=50mm, 1" dia
        'type': 'doublet',
        'R1': 33.3e-3, 'R2': -24.1e-3, 'R3': -95.3e-3,
        'd1': 9.0e-3, 'd2': 3.0e-3,
        'glass1': 'N-BAF10', 'glass2': 'N-SF6HT', 'aperture': 25.4e-3,
    },
    'AC254-200-C': {  # f=200mm, 1" dia
        'type': 'doublet',
        'R1': 110.1e-3, 'R2': -80.6e-3, 'R3': -277.5e-3,
        'd1': 4.0e-3, 'd2': 2.0e-3,
        'glass1': 'N-BAF10', 'glass2': 'N-SF6HT', 'aperture': 25.4e-3,
    },
    'AC254-100-C': {  # f=100mm, 1" dia
        'type': 'doublet',
        'R1': 62.8e-3, 'R2': -46.5e-3, 'R3': -184.5e-3,
        'd1': 6.0e-3, 'd2': 2.5e-3,
        'glass1': 'N-BAF10', 'glass2': 'N-SF6HT', 'aperture': 25.4e-3,
    },
}


def thorlabs_lens(part_number):
    """Return a lens prescription dict for a Thorlabs catalog lens.

    The prescription uses glass name strings and is wavelength-independent.
    Refractive indices are resolved at runtime by ``apply_real_lens``.

    Parameters
    ----------
    part_number : str
        Thorlabs part number (e.g. ``'AC254-200-C'``, ``'LA1050-C'``).

    Returns
    -------
    prescription : dict
        Ready to pass to :func:`apply_real_lens`.
    """
    if part_number not in THORLABS_CATALOG:
        raise ValueError(f"Unknown part '{part_number}'. "
                         f"Available: {list(THORLABS_CATALOG.keys())}")

    entry = THORLABS_CATALOG[part_number]

    if entry['type'] == 'singlet':
        return make_singlet(
            R1=entry['R1'], R2=entry['R2'], d=entry['d'],
            glass=entry['glass'],
            aperture=entry['aperture'], name=part_number)

    elif entry['type'] == 'doublet':
        return make_doublet(
            R1=entry['R1'], R2=entry['R2'], R3=entry['R3'],
            d1=entry['d1'], d2=entry['d2'],
            glass1=entry['glass1'], glass2=entry['glass2'],
            aperture=entry['aperture'], name=part_number)

    else:
        raise ValueError(f"Unknown lens type: {entry['type']}")


# ============================================================================
# Zemax .zmx file parser
# ============================================================================

def load_zmx_prescription(filepath, surface_range=None, name=None):
    """Parse a Zemax .zmx text file and return a lens prescription dict.

    Reads the surface table from a Zemax sequential lens file and builds a
    prescription dict compatible with :func:`apply_real_lens`.  Handles
    standard spherical surfaces and even-aspheric surfaces (EVENASPH).

    Parameters
    ----------
    filepath : str
        Path to the .zmx file.
    surface_range : tuple of (int, int), optional
        Which Zemax surface numbers to include as lens surfaces, given as
        ``(first, last)`` inclusive.  For example, ``(2, 4)`` extracts
        surfaces 2, 3, and 4 from the Zemax file.  If *None*, all surfaces
        between the first and last glass surfaces are automatically detected.
    name : str, optional
        Human-readable label.  If *None*, derived from the filename.

    Returns
    -------
    result : dict
        ``'name'`` : str -- human-readable label

        ``'aperture_diameter'`` : float -- clear aperture [m]

        ``'surfaces'`` : list -- refracting surfaces only (for ``apply_real_lens``)

        ``'thicknesses'`` : list -- thicknesses between refracting surfaces [m]

        ``'elements'`` : list -- full element list including mirrors, each with
        ``'element_type'``: ``'surface'`` or ``'mirror'``

        ``'all_thicknesses'`` : list -- thicknesses between all elements [m]

    Notes
    -----
    The ``'surfaces'`` and ``'thicknesses'`` keys give a lens-only
    prescription that can be passed directly to :func:`apply_real_lens`.
    Mirrors and coordinate breaks are excluded from this.

    The ``'elements'`` key gives the full parsed sequence including mirrors
    (``element_type='mirror'``), which can be used with :func:`apply_mirror`
    for manual propagation through folded systems.

    Coordinate break surfaces (COORDBRK) are skipped entirely -- they
    represent geometric transforms that are not modeled by ASM.

    Conic constants are read only from dedicated ``CONI`` lines in the .zmx
    file.  The extra fields on the ``CURV`` line (which encode solve
    parameters like pickup scale factors) are ignored.

    Examples
    --------
    >>> rx = load_zmx_prescription('AC254-200-C.zmx')
    >>> E_out = apply_real_lens(E_in, rx, wavelength=1.3e-6, dx=2.1e-6)

    >>> rx = load_zmx_prescription('my_design.zmx', surface_range=(2, 5))
    """
    # Read file -- try UTF-16-LE first (Zemax default), then UTF-8
    for encoding in ('utf-16-le', 'utf-8', 'latin-1'):
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                text = f.read()
            if 'SURF' in text:
                break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        raise IOError(f"Could not read {filepath} with any supported encoding")

    # Remove BOM if present
    text = text.lstrip('\ufeff')
    lines = text.split('\n')

    # Determine unit scale factor (convert to meters)
    unit_scale = 1e-3  # default: mm
    for line in lines:
        tokens = line.strip().split()
        if tokens and tokens[0] == 'UNIT':
            unit_str = tokens[1].upper() if len(tokens) > 1 else 'MM'
            unit_map = {'MM': 1e-3, 'CM': 1e-2, 'IN': 25.4e-3, 'M': 1.0}
            unit_scale = unit_map.get(unit_str, 1e-3)
            break

    # ------------------------------------------------------------------
    # Parse surfaces
    # ------------------------------------------------------------------
    surfaces_raw = []
    current_surf = None

    for line in lines:
        stripped = line.strip()
        tokens = stripped.split()
        if not tokens:
            continue

        keyword = tokens[0]

        if keyword == 'SURF':
            if current_surf is not None:
                surfaces_raw.append(current_surf)
            current_surf = {
                'surf_num': int(tokens[1]),
                'type': 'STANDARD',
                'curvature': 0.0,
                'conic': 0.0,
                'thickness': 0.0,
                'glass': None,
                'semi_diameter': 0.0,
                'aspheric_params': {},
                'is_stop': False,
                'is_mirror': False,
                'is_coordbrk': False,
                'comment': '',
            }

        elif current_surf is not None:
            if keyword == 'TYPE':
                stype = tokens[1] if len(tokens) > 1 else 'STANDARD'
                current_surf['type'] = stype
                if stype == 'COORDBRK':
                    current_surf['is_coordbrk'] = True

            elif keyword == 'STOP':
                current_surf['is_stop'] = True

            elif keyword == 'CURV':
                # Only read the curvature value (first token after keyword).
                # Remaining fields are solve parameters (pickup source,
                # scale factor, etc.) -- NOT conic constants.
                current_surf['curvature'] = float(tokens[1])

            elif keyword == 'CONI':
                current_surf['conic'] = float(tokens[1])

            elif keyword == 'DISZ':
                if tokens[1].upper() == 'INFINITY':
                    current_surf['thickness'] = float('inf')
                else:
                    current_surf['thickness'] = float(tokens[1])

            elif keyword == 'GLAS':
                glass_name = tokens[1]
                current_surf['glass'] = glass_name
                if glass_name.upper() == 'MIRROR':
                    current_surf['is_mirror'] = True

            elif keyword == 'MIRR':
                # Some files use MIRR flag instead of GLAS MIRROR
                try:
                    if int(tokens[1]) == 1:  # 1 = reflective
                        current_surf['is_mirror'] = True
                except (ValueError, IndexError):
                    pass

            elif keyword == 'DIAM':
                current_surf['semi_diameter'] = float(tokens[1])

            elif keyword == 'PARM':
                parm_num = int(tokens[1])
                parm_val = float(tokens[2])
                if parm_val != 0.0:
                    current_surf['aspheric_params'][parm_num] = parm_val

            elif keyword == 'COMM':
                current_surf['comment'] = stripped[5:].strip().strip('"')

    # Don't forget the last surface
    if current_surf is not None:
        surfaces_raw.append(current_surf)

    # ------------------------------------------------------------------
    # Filter out coordinate breaks (non-optical surfaces)
    # ------------------------------------------------------------------
    optical_surfaces = [s for s in surfaces_raw if not s['is_coordbrk']]

    # ------------------------------------------------------------------
    # Determine which surfaces are part of the lens
    # ------------------------------------------------------------------
    if surface_range is not None:
        s_first, s_last = surface_range
        lens_surfaces = [s for s in optical_surfaces
                         if s_first <= s['surf_num'] <= s_last]
    else:
        # Auto-detect: find first and last surfaces with glass or mirror
        active = [s for s in optical_surfaces
                  if s['glass'] is not None or s['is_mirror']]
        if not active:
            raise ValueError(f"No glass/mirror surfaces found in {filepath}")
        s_first = active[0]['surf_num']
        s_last = active[-1]['surf_num'] + 1
        lens_surfaces = [s for s in optical_surfaces
                         if s_first <= s['surf_num'] <= s_last]

    if len(lens_surfaces) < 2:
        raise ValueError(
            f"Need at least 2 surfaces, got {len(lens_surfaces)} "
            f"in range ({s_first}, {s_last})")

    # ------------------------------------------------------------------
    # Object-space distance
    # ------------------------------------------------------------------
    # Zemax files typically have a chain of non-refractive surfaces
    # (OBJ plane, STOP, coordinate breaks, dummy reference planes, etc.)
    # before the first real lens surface.  These get filtered out of
    # ``lens_surfaces`` here, but their DISZ (z-thickness) values can
    # carry meaningful design geometry -- in particular, the distance
    # from the object/source plane to the first refractive surface.
    #
    # Without preserving that total, a downstream simulation that
    # propagates its own source field through the prescription will
    # implicitly place the source AT the first refractive surface,
    # collapsing the design's obj-space geometry.  For a field source
    # with finite angular spread (Gaussian beam, collimated array), this
    # causes a focal-plane defocus proportional to the dropped distance.
    #
    # Convention for ``object_distance``: the sum of DISZ values from
    # the STOP surface (treated as the TX / source plane) up to but not
    # including the first refractive surface of ``lens_surfaces``.  If
    # no STOP is present, sum from SURF 0 onward.  Non-finite DISZ
    # values (``INFINITY``) contribute 0 since Zemax uses INFINITY for
    # collimated-source configurations where the object is at infinity.
    stop_idx_in_raw = None
    for _idx, _s in enumerate(surfaces_raw):
        if _s.get('is_stop'):
            stop_idx_in_raw = _idx
            break
    # Find the index of the first lens surface in surfaces_raw.
    first_lens_surf_num = lens_surfaces[0]['surf_num']
    first_lens_idx_in_raw = next(
        (_idx for _idx, _s in enumerate(surfaces_raw)
         if _s['surf_num'] == first_lens_surf_num),
        None)
    obj_distance = 0.0
    if first_lens_idx_in_raw is not None:
        _start = stop_idx_in_raw if stop_idx_in_raw is not None else 0
        if _start < first_lens_idx_in_raw:
            for _idx in range(_start, first_lens_idx_in_raw):
                _t = surfaces_raw[_idx].get('thickness', 0.0)
                if np.isfinite(_t):
                    obj_distance += _t
    obj_distance *= unit_scale

    # ------------------------------------------------------------------
    # Build glass sequence: track current medium between surfaces
    # ------------------------------------------------------------------
    medium_between = []
    for s in lens_surfaces:
        if s['is_mirror']:
            medium_between.append(None)
        elif s['glass'] is not None and not s['is_mirror']:
            medium_between.append(s['glass'])
        else:
            medium_between.append(None)

    # ------------------------------------------------------------------
    # Build the output element list
    # ------------------------------------------------------------------
    elements = []
    for i, s in enumerate(lens_surfaces):
        # Radius from curvature (convert units)
        curv = s['curvature']
        if abs(curv) < 1e-15:
            radius = float('inf')
        else:
            radius = (1.0 / curv) * unit_scale

        # Aspheric coefficients
        asph_coeffs = None
        if s['aspheric_params']:
            asph_coeffs = {}
            for parm_num, parm_val in s['aspheric_params'].items():
                if parm_num >= 2:
                    power = 2 * parm_num
                    asph_coeffs[power] = parm_val / (unit_scale ** (power - 1))

        # Per-surface clear semi-diameter [m]
        semi_dia_m = s['semi_diameter'] * unit_scale

        if s['is_mirror']:
            elements.append({
                'element_type': 'mirror',
                'radius': radius,
                'conic': s['conic'],
                'aspheric_coeffs': asph_coeffs,
                'semi_diameter': semi_dia_m,
                'surf_num': s['surf_num'],
                'comment': s.get('comment', ''),
            })
        else:
            # Determine glass before and after this surface
            if i == 0:
                glass_before = 'air'
            else:
                glass_before = medium_between[i - 1] or 'air'
            glass_after = medium_between[i] or 'air'

            elements.append({
                'element_type': 'surface',
                'radius': radius,
                'conic': s['conic'],
                'aspheric_coeffs': asph_coeffs,
                'glass_before': glass_before,
                'glass_after': glass_after,
                'semi_diameter': semi_dia_m,
                'surf_num': s['surf_num'],
                'comment': s.get('comment', ''),
            })

    # ------------------------------------------------------------------
    # Thicknesses between consecutive lens surfaces (convert units)
    # ------------------------------------------------------------------
    thicknesses = []
    for i in range(len(lens_surfaces) - 1):
        t = lens_surfaces[i]['thickness']
        if np.isinf(t):
            t = 0.0
        thicknesses.append(t * unit_scale)

    # ------------------------------------------------------------------
    # Aperture from the stop surface or largest semi-diameter
    # ------------------------------------------------------------------
    stop_surfaces = [s for s in lens_surfaces if s['is_stop']]
    if stop_surfaces:
        aperture = stop_surfaces[0]['semi_diameter'] * 2 * unit_scale
    else:
        aperture = max(s['semi_diameter'] for s in lens_surfaces) * 2 * unit_scale

    if name is None:
        name = os.path.splitext(os.path.basename(filepath))[0]

    # ------------------------------------------------------------------
    # Build lens-only prescription (refracting surfaces only)
    # ------------------------------------------------------------------
    refr_surfaces = [e for e in elements if e['element_type'] == 'surface']
    prescription_surfaces = []
    for e in refr_surfaces:
        prescription_surfaces.append({
            'radius': e['radius'],
            'conic': e['conic'],
            'aspheric_coeffs': e['aspheric_coeffs'],
            'glass_before': e['glass_before'],
            'glass_after': e['glass_after'],
        })

    # Thicknesses for the lens-only prescription (between refracting surfaces)
    refr_indices = [i for i, e in enumerate(elements)
                    if e['element_type'] == 'surface']
    lens_thicknesses = []
    for j in range(len(refr_indices) - 1):
        # Sum thicknesses between consecutive refracting surfaces
        idx_start = refr_indices[j]
        idx_end = refr_indices[j + 1]
        total_t = 0
        for k in range(idx_start, idx_end):
            if k < len(thicknesses):
                total_t += thicknesses[k]
        lens_thicknesses.append(total_t)

    # ------------------------------------------------------------------
    # Warn about unknown glasses
    # ------------------------------------------------------------------
    unknown_glasses = set()
    for e in elements:
        if e['element_type'] == 'surface':
            for g in (e['glass_before'], e['glass_after']):
                if g != 'air' and g not in GLASS_REGISTRY:
                    unknown_glasses.add(g)
    if unknown_glasses:
        warnings.warn(
            f"Glasses not in GLASS_REGISTRY: {unknown_glasses}. "
            f"Add them before calling apply_real_lens. Example:\n"
            f"  GLASS_REGISTRY['GLASS_NAME'] = ('specs', 'CATALOG', 'PAGE')\n"
            f"Browse refractiveindex.info to find the correct path.")

    # ------------------------------------------------------------------
    # Coordinate breaks
    # ------------------------------------------------------------------
    # Extract decenter/tilt parameters from every COORDBRK surface in
    # the raw surface list.  Zemax PARM 1-6 on a COORDBRK surface are:
    #   PARM 1:  Decenter X [lens units, e.g. mm]
    #   PARM 2:  Decenter Y [lens units]
    #   PARM 3:  Tilt X (rotation about x-axis) [degrees]
    #   PARM 4:  Tilt Y (rotation about y-axis) [degrees]
    #   PARM 5:  Tilt Z (rotation about z-axis, a.k.a. roll) [degrees]
    #   PARM 6:  Order  (0 = decenter then tilt, 1 = tilt then decenter)
    #
    # Decenters are converted to meters (multiplied by ``unit_scale``);
    # tilts remain in degrees.  The loader preserves COORDBRK DISZ
    # thickness via the usual ``all_thicknesses`` path (unchanged here).
    #
    # Downstream callers (wave-optics simulations) should iterate this
    # list and apply each break at its z-position in the propagation
    # chain -- see tx_design_study_sim.py's _apply_coord_break for an
    # example implementation.
    coord_breaks = []
    for s in surfaces_raw:
        if not s.get('is_coordbrk'):
            continue
        parms = s.get('aspheric_params', {})
        cb = {
            'surf_num': s['surf_num'],
            'decenter_x_m': float(parms.get(1, 0.0)) * unit_scale,
            'decenter_y_m': float(parms.get(2, 0.0)) * unit_scale,
            'tilt_x_deg':   float(parms.get(3, 0.0)),
            'tilt_y_deg':   float(parms.get(4, 0.0)),
            'tilt_z_deg':   float(parms.get(5, 0.0)),
            'order': int(float(parms.get(6, 0.0)) or 0),
            'thickness_m':  float(s.get('thickness', 0.0) or 0.0) * (
                unit_scale if np.isfinite(s.get('thickness', 0.0))
                else 1.0),
        }
        coord_breaks.append(cb)

    return {
        'name': name,
        'aperture_diameter': aperture,
        # Lens-only prescription (for apply_real_lens)
        'surfaces': prescription_surfaces,
        'thicknesses': lens_thicknesses,
        # Full element list including mirrors (for manual use)
        'elements': elements,
        'all_thicknesses': thicknesses,
        # Distance from the stop / source plane to the first refractive
        # surface.  Non-zero when the .zmx has dummy surfaces between
        # the object and the first lens; 0 when the first lens is the
        # first surface after the object.  Callers doing wave-optics
        # propagation should apply this as free space between their
        # source (or post-MLA) field and the first lens event.
        'object_distance': obj_distance,
        # List of coordinate breaks (decenters / tilts).  Each entry is
        # a dict with keys ``surf_num``, ``decenter_x_m``,
        # ``decenter_y_m``, ``tilt_x_deg``, ``tilt_y_deg``,
        # ``tilt_z_deg``, ``order``, ``thickness_m``.  Empty list when
        # the prescription has no COORDBRK surfaces.  Sorted in
        # .zmx surface order.
        'coord_breaks': coord_breaks,
    }


# ---------------------------------------------------------------------------
# Zemax Prescription Data text export parser
# ---------------------------------------------------------------------------

def load_zemax_prescription_txt(filepath, surface_range=None, name=None):
    """
    Parse a Zemax "Prescription Data" text export and return a lens prescription.

    Zemax's *Analyze -> Reports -> Prescription Data* command exports a
    tab-separated text report containing the full surface table plus
    system parameters (wavelength, units, focal length, etc.).  This
    parser reads that format and produces the same output structure as
    :func:`load_zmx_prescription` so the two loaders are interchangeable.

    The file is typically UTF-16 encoded (both BOM-marked UTF-16 and
    UTF-8 are tried automatically).

    Parameters
    ----------
    filepath : str
        Path to the prescription text file.
    surface_range : tuple of (int, int), optional
        Which Zemax surface numbers to include as lens surfaces,
        inclusive on both ends.  If None, auto-detect the first and last
        surfaces with glass or mirror.
    name : str, optional
        Human-readable label.  If None, derived from the filename.

    Returns
    -------
    prescription : dict
        Dictionary with keys:

        - ``'name'``             : human-readable label
        - ``'aperture_diameter'``: clear aperture [m]
        - ``'surfaces'``         : refracting surfaces only (for apply_real_lens)
        - ``'thicknesses'``      : thicknesses between refracting surfaces [m]
        - ``'elements'``         : full element list including mirrors
        - ``'all_thicknesses'``  : thicknesses between all elements [m]
        - ``'wavelength'``       : primary wavelength [m] (if found in header)
        - ``'units'``            : lens unit string from the header

    Notes
    -----
    Supported column format (tab-separated, one row per surface)::

        Surf  Type   Radius  Thickness  Glass  Clear Diam  Chip Zone  Mech Diam  Conic  Comment

    Surfaces tagged as ``COORDBRK`` are filtered out (they represent
    geometric transforms).  ``MIRROR`` surfaces are tagged as mirror
    elements in the ``'elements'`` list but excluded from the lens-only
    ``'surfaces'`` list.  ``DGRATING`` surfaces are treated as flat
    optical surfaces (their diffractive behavior is not modeled here).

    Unlike ``.zmx`` files, prescription text reports give the radius
    directly (not as curvature), so there are no pickup-solve parameters
    to worry about.  The "Conic" column is read directly.

    Radii, thicknesses, and diameters are converted from the report's
    native units (Millimeters by default) to meters.

    Examples
    --------
    >>> rx = load_zemax_prescription_txt('TXdesign-prescription.txt')
    >>> print(f"Found {len(rx['elements'])} elements")
    >>> print(f"Wavelength: {rx.get('wavelength', 0)*1e9:.0f} nm")
    """
    # Try encodings in order: UTF-16 (with BOM, most common), UTF-16-LE,
    # UTF-8, and finally latin-1 as a fallback.
    text = None
    for encoding in ('utf-16', 'utf-16-le', 'utf-8', 'latin-1'):
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                candidate = f.read()
            if 'SURFACE DATA SUMMARY' in candidate:
                text = candidate
                break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if text is None:
        raise IOError(
            f"Could not read {filepath} with any supported encoding "
            f"(tried utf-16, utf-16-le, utf-8, latin-1)."
        )

    text = text.lstrip('\ufeff')  # strip BOM if present

    # ---------------------------------------------------------------
    # Parse header metadata (wavelength, units, stop radius, etc.)
    # ---------------------------------------------------------------
    wavelength_m = None
    unit_scale = 1e-3        # default: millimeters
    unit_name = 'Millimeters'
    for line in text.split('\n'):
        s = line.strip()
        if not s or ':' not in s:
            continue
        # "Primary Wavelength [µm] :  1.31"
        if 'Primary Wavelength' in s:
            try:
                val = s.split(':', 1)[1].strip().split()[0]
                wavelength_m = float(val) * 1e-6  # µm -> m
            except (ValueError, IndexError):
                pass
        # "Lens Units              :   Millimeters"
        elif 'Lens Units' in s:
            try:
                unit_name = s.split(':', 1)[1].strip()
                unit_map = {
                    'Millimeters': 1e-3,
                    'Centimeters': 1e-2,
                    'Meters': 1.0,
                    'Inches': 25.4e-3,
                }
                unit_scale = unit_map.get(unit_name, 1e-3)
            except (ValueError, IndexError):
                pass

    # ---------------------------------------------------------------
    # Locate the SURFACE DATA SUMMARY table
    # ---------------------------------------------------------------
    start = text.find('SURFACE DATA SUMMARY')
    if start < 0:
        raise ValueError(
            f"{filepath} does not contain a 'SURFACE DATA SUMMARY' section."
        )
    # End of table: the next "SURFACE DATA DETAIL" or "EDGE THICKNESS"
    end_markers = ('SURFACE DATA DETAIL', 'EDGE THICKNESS DATA',
                   'MULTI-CONFIGURATION DATA')
    end = len(text)
    for marker in end_markers:
        idx = text.find(marker, start + 1)
        if idx > 0 and idx < end:
            end = idx
    table = text[start:end]

    # Locate the column header row and parse rows after it
    lines = table.split('\n')
    header_idx = None
    for i, line in enumerate(lines):
        if 'Surf' in line and 'Type' in line and 'Radius' in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find column header row in surface table.")

    # ---------------------------------------------------------------
    # Parse each surface row
    # ---------------------------------------------------------------
    # Columns (tab-separated):
    #   Surf, Type, Radius, Thickness, Glass, Clear Diam, Chip Zone,
    #   Mech Diam, Conic, Comment
    surfaces_raw = []
    last_surf_num = -1

    for raw in lines[header_idx + 1:]:
        line = raw.rstrip()
        if not line.strip():
            continue

        fields = [f.strip() for f in line.split('\t')]
        if len(fields) < 9:
            # Not a surface row (blank separator, continuation, etc.)
            continue

        surf_label = fields[0]
        type_str = fields[1]
        radius_str = fields[2]
        thickness_str = fields[3]
        glass_str = fields[4] if len(fields) > 4 else ''
        clear_diam_str = fields[5] if len(fields) > 5 else ''
        conic_str = fields[8] if len(fields) > 8 else '0'
        comment = fields[9] if len(fields) > 9 else ''

        # Map OBJ / STO / IMA / numeric
        if surf_label == 'OBJ':
            surf_num = 0
        elif surf_label == 'STO':
            surf_num = last_surf_num + 1
        elif surf_label == 'IMA':
            surf_num = last_surf_num + 1
        else:
            try:
                surf_num = int(surf_label)
            except ValueError:
                continue  # skip malformed rows
        last_surf_num = max(last_surf_num, surf_num)

        # Parse numeric fields (handle "Infinity" and "-" placeholders)
        def _parse_float(s, default=0.0):
            s = s.strip()
            if not s or s == '-':
                return default
            if s.lower() in ('infinity', 'inf'):
                return float('inf')
            try:
                return float(s)
            except ValueError:
                return default

        radius = _parse_float(radius_str, float('inf'))
        thickness = _parse_float(thickness_str, 0.0)
        # The "Clear Diam" column in the prescription text report is the
        # full diameter, not semi-diameter.  Divide by 2 so the internal
        # representation matches the .zmx parser (which reads DIAM as
        # semi-diameter directly).
        semi_diameter = _parse_float(clear_diam_str, 0.0) / 2.0
        conic = _parse_float(conic_str, 0.0)

        glass = glass_str if glass_str else None
        is_mirror = glass is not None and glass.upper() == 'MIRROR'
        is_coordbrk = type_str.upper() == 'COORDBRK'
        is_stop = surf_label == 'STO'

        # Convert to meters
        if not np.isinf(radius):
            radius = radius * unit_scale
        thickness = thickness * unit_scale
        semi_diameter = semi_diameter * unit_scale

        surfaces_raw.append({
            'surf_num': surf_num,
            'surf_label': surf_label,
            'type': type_str,
            'radius': radius,
            'conic': conic,
            'thickness': thickness,
            'glass': glass,
            'semi_diameter': semi_diameter,
            'aspheric_params': {},
            'is_stop': is_stop,
            'is_mirror': is_mirror,
            'is_coordbrk': is_coordbrk,
            'comment': comment,
        })

    # ---------------------------------------------------------------
    # Filter out coordinate breaks (non-optical) and pick lens surfaces
    # ---------------------------------------------------------------
    optical_surfaces = [s for s in surfaces_raw if not s['is_coordbrk']]

    if surface_range is not None:
        s_first, s_last = surface_range
        lens_surfaces = [s for s in optical_surfaces
                         if s_first <= s['surf_num'] <= s_last]
    else:
        active = [s for s in optical_surfaces
                  if s['glass'] is not None or s['is_mirror']]
        if not active:
            raise ValueError(f"No glass/mirror surfaces found in {filepath}")
        s_first = active[0]['surf_num']
        s_last = active[-1]['surf_num'] + 1
        lens_surfaces = [s for s in optical_surfaces
                         if s_first <= s['surf_num'] <= s_last]

    if len(lens_surfaces) < 2:
        raise ValueError(
            f"Need at least 2 surfaces, got {len(lens_surfaces)} "
            f"in range ({s_first}, {s_last})"
        )

    # Object-space distance (see load_zmx_prescription for rationale).
    # Sum ``thickness`` values from the STOP surface (treated as the
    # source plane) up to but not including the first refractive
    # surface.  If no STOP is present, sum from SURF 0 onward.
    stop_idx_in_raw = None
    for _idx, _s in enumerate(surfaces_raw):
        if _s.get('is_stop'):
            stop_idx_in_raw = _idx
            break
    first_lens_surf_num = lens_surfaces[0]['surf_num']
    first_lens_idx_in_raw = next(
        (_idx for _idx, _s in enumerate(surfaces_raw)
         if _s['surf_num'] == first_lens_surf_num),
        None)
    obj_distance = 0.0
    if first_lens_idx_in_raw is not None:
        _start = stop_idx_in_raw if stop_idx_in_raw is not None else 0
        if _start < first_lens_idx_in_raw:
            for _idx in range(_start, first_lens_idx_in_raw):
                _t = surfaces_raw[_idx].get('thickness', 0.0)
                if np.isfinite(_t):
                    obj_distance += _t
    # Note: .txt-loader thicknesses are already in meters (no unit_scale).

    # Track the glass medium between each consecutive surface pair
    medium_between = []
    for s in lens_surfaces:
        if s['is_mirror']:
            medium_between.append(None)
        elif s['glass'] is not None and not s['is_mirror']:
            medium_between.append(s['glass'])
        else:
            medium_between.append(None)

    # Build the output element list
    elements = []
    for i, s in enumerate(lens_surfaces):
        if s['is_mirror']:
            elements.append({
                'element_type': 'mirror',
                'radius': s['radius'],
                'conic': s['conic'],
                'aspheric_coeffs': None,
                'semi_diameter': s['semi_diameter'],
                'surf_num': s['surf_num'],
                'comment': s.get('comment', ''),
            })
        else:
            if i == 0:
                glass_before = 'air'
            else:
                glass_before = medium_between[i - 1] or 'air'
            glass_after = medium_between[i] or 'air'

            elements.append({
                'element_type': 'surface',
                'radius': s['radius'],
                'conic': s['conic'],
                'aspheric_coeffs': None,
                'glass_before': glass_before,
                'glass_after': glass_after,
                'semi_diameter': s['semi_diameter'],
                'surf_num': s['surf_num'],
                'comment': s.get('comment', ''),
            })

    # All-element thicknesses (one fewer than elements)
    thicknesses = []
    for i in range(len(lens_surfaces) - 1):
        t = lens_surfaces[i]['thickness']
        if np.isinf(t):
            t = 0.0
        thicknesses.append(t)

    # Aperture from the stop surface or largest semi-diameter
    stop_surfaces = [s for s in lens_surfaces if s['is_stop']]
    if stop_surfaces:
        aperture = stop_surfaces[0]['semi_diameter'] * 2
    else:
        aperture = max(s['semi_diameter'] for s in lens_surfaces) * 2

    if name is None:
        name = os.path.splitext(os.path.basename(filepath))[0]

    # Build the lens-only prescription (refracting surfaces only)
    refr_surfaces = [e for e in elements if e['element_type'] == 'surface']
    prescription_surfaces = [
        {
            'radius': e['radius'],
            'conic': e['conic'],
            'aspheric_coeffs': e['aspheric_coeffs'],
            'glass_before': e['glass_before'],
            'glass_after': e['glass_after'],
        }
        for e in refr_surfaces
    ]

    # Thicknesses between refracting surfaces only
    refr_indices = [i for i, e in enumerate(elements) if e['element_type'] == 'surface']
    lens_thicknesses = []
    for j in range(len(refr_indices) - 1):
        idx_start = refr_indices[j]
        idx_end = refr_indices[j + 1]
        total_t = sum(thicknesses[k] for k in range(idx_start, idx_end)
                      if k < len(thicknesses))
        lens_thicknesses.append(total_t)

    # Warn about unknown glasses
    unknown_glasses = set()
    for e in elements:
        if e['element_type'] == 'surface':
            for g in (e['glass_before'], e['glass_after']):
                if g != 'air' and g not in GLASS_REGISTRY:
                    unknown_glasses.add(g)
    if unknown_glasses:
        warnings.warn(
            f"Glasses not in GLASS_REGISTRY: {unknown_glasses}. "
            f"Add them before calling apply_real_lens. Example:\n"
            f"  GLASS_REGISTRY['GLASS_NAME'] = ('specs', 'CATALOG', 'PAGE')\n"
            f"Browse refractiveindex.info to find the correct path."
        )

    return {
        'name': name,
        'aperture_diameter': aperture,
        # Lens-only prescription (for apply_real_lens)
        'surfaces': prescription_surfaces,
        'thicknesses': lens_thicknesses,
        # Full element list including mirrors (for manual use)
        'elements': elements,
        'all_thicknesses': thicknesses,
        # Distance from the stop / source plane to the first refractive
        # surface.  See load_zmx_prescription for rationale.
        'object_distance': obj_distance,
        # Metadata from header
        'wavelength': wavelength_m,
        'units': unit_name,
    }


# ============================================================================
# Zemax export
# ============================================================================
#
# These helpers write a lens prescription out in two forms that are
# useful for cross-verifying wave simulations against Zemax
# OpticStudio:
#
#   1. A human-readable LDE-style text table that can be typed (or
#      column-copy-pasted) into the Zemax Lens Data Editor.
#
#   2. A minimal ``.zmx`` sequential file that Zemax can import with
#      File > Open.  The generated file is intentionally minimal: it
#      defines only the surface table, wavelength, aperture and field,
#      using Zemax defaults for everything else.  After loading you may
#      want to verify the APERTURE settings (Clear Semi-Diameter
#      floating vs. fixed) and the STOP location to match your
#      experimental conditions.
#
# Sign convention matches Zemax's default (and our library's): positive
# radius of curvature means the centre of curvature lies to the right
# of the surface vertex.


def export_zemax_lens_data(prescription, path, wavelength=1.31e-6,
                           stop_surface=0, aperture_diameter=None,
                           back_focal_length=None,
                           description=None, extra_notes=None):
    """Write a human-readable Zemax-LDE-style text table for a lens
    prescription.

    The resulting file is easy to eyeball and can be transcribed into
    Zemax OpticStudio by hand.  For direct import, see
    :func:`export_zemax_zmx`.

    Parameters
    ----------
    prescription : dict
        Prescription dict with keys ``'surfaces'`` and ``'thicknesses'``
        (see :func:`make_singlet`).
    path : str
        Output file path (``.txt`` recommended).
    wavelength : float, default 1.31e-6
        Primary wavelength [m] to record in the file header.
    stop_surface : int, default 0
        Zero-based index of the aperture stop within the refracting
        surface list.
    aperture_diameter : float, optional
        Clear aperture diameter [m].  Falls back to
        ``prescription.get('aperture_diameter')``.
    back_focal_length : float, optional
        BFL [m] to insert between the last refracting surface and the
        image plane.  If ``None``, ``0.0`` is written (user should set
        by eye in Zemax).
    description : str, optional
        Free-form description written at the top of the file.
    extra_notes : list of str, optional
        Additional lines appended to the header as comments.

    Notes
    -----
    Column units: radii and thicknesses in *millimeters*, diameters in
    *millimeters*, conic dimensionless.  Glass strings are written as
    they appear in the prescription.  Infinite radii are rendered as
    ``Infinity`` (matching Zemax's text convention).
    """
    surfaces = prescription['surfaces']
    thicknesses = prescription['thicknesses']
    if aperture_diameter is None:
        aperture_diameter = prescription.get('aperture_diameter', 25.4e-3)
    semi_dia_mm = 0.5 * aperture_diameter * 1e3
    bfl_mm = (back_focal_length * 1e3) if back_focal_length else 0.0
    name = prescription.get('name', os.path.splitext(
        os.path.basename(path))[0])

    def _fmt_radius(R):
        return 'Infinity' if (R is None or np.isinf(R)) else f'{R*1e3:.6f}'

    lines = []
    lines.append(f'# Zemax-compatible lens data for: {name}')
    if description:
        lines.append(f'# Description: {description}')
    lines.append('#')
    lines.append('# Test conditions')
    lines.append(f'#   Primary wavelength: {wavelength*1e6:.4f} um')
    lines.append(f'#   Source: collimated on-axis plane wave')
    lines.append(f'#   Aperture: clear semi-diameter = '
                 f'{semi_dia_mm:.4f} mm (diameter {aperture_diameter*1e3:.4f} mm)')
    lines.append(f'#   Stop surface index: {stop_surface + 1}')
    if extra_notes:
        for note in extra_notes:
            lines.append(f'#   {note}')
    lines.append('#')
    lines.append('# Paste into the Zemax Lens Data Editor (Sequential mode)')
    lines.append('# Columns: SURF | TYPE | RADIUS [mm] | THICKNESS [mm] '
                 '| MATERIAL | SEMI-DIA [mm] | CONIC | COMMENT')
    lines.append('#')
    # Header row for the table
    lines.append(
        '# {0:4s} {1:11s} {2:>16s} {3:>16s} {4:>10s} {5:>10s} {6:>8s}  {7}'
        .format('SURF', 'TYPE', 'RADIUS', 'THICKNESS',
                'MATERIAL', 'SEMI-DIA', 'CONIC', 'COMMENT'))

    # Object surface
    lines.append(
        '  {0:4s} {1:11s} {2:>16s} {3:>16s} {4:>10s} {5:>10s} {6:>8s}  {7}'
        .format('OBJ', 'STANDARD', 'Infinity', 'Infinity',
                '--', '0.000', '0.000', 'Object at infinity'))

    # Refracting surfaces
    for i, surf in enumerate(surfaces):
        label = 'STO' if i == stop_surface else str(i + 1)
        stop_mark = ' * ' if i == stop_surface else '   '
        rad = _fmt_radius(surf.get('radius'))
        # Thickness after this surface: between-element spacing, or
        # BFL after the last surface.
        if i < len(thicknesses):
            t_mm = thicknesses[i] * 1e3
        else:
            t_mm = bfl_mm
        t_str = f'{t_mm:.6f}'
        glass = surf.get('glass_after', '')
        if not glass or glass.lower() in ('air', ''):
            glass = '--'
        conic = surf.get('conic', 0.0) or 0.0
        comment = surf.get('comment', '')
        lines.append(
            '{mark}{surf:4s} {tp:11s} {rad:>16s} {th:>16s} '
            '{gl:>10s} {sd:>10.4f} {con:>8.4f}  {cm}'
            .format(mark=stop_mark, surf=label, tp='STANDARD',
                    rad=rad, th=t_str, gl=glass, sd=semi_dia_mm,
                    con=float(conic), cm=comment or f'surface {i+1}'))

    # Image plane
    lines.append(
        '   {0:4s} {1:11s} {2:>16s} {3:>16s} {4:>10s} {5:>10.4f} {6:>8.4f}  {7}'
        .format('IMA', 'STANDARD', 'Infinity', '0.000000',
                '--', 0.0, 0.0, 'Image plane'))

    lines.append('#')
    lines.append('# Legend: "*" marks the aperture stop.')
    lines.append('# To verify OPD in Zemax: Analysis > Wavefront > '
                 'Wavefront Map, or Analysis > Aberrations > Optical '
                 'Path Difference (OPD fan).')
    lines.append('# Remember to set Aperture Type -> "Float By Stop '
                 'Size" and the primary wavelength to match the value '
                 'listed above.')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def export_zemax_zmx(prescription, path, wavelength=1.31e-6,
                     stop_surface=0, aperture_diameter=None,
                     back_focal_length=None, name=None):
    """Write a minimal Zemax OpticStudio ``.zmx`` sequential file for a
    prescription.

    The produced file contains surface data, one wavelength, and an
    on-axis field.  Other Zemax settings use defaults; you may need to
    tweak glass catalogs or aperture conventions after opening the file.

    Parameters
    ----------
    prescription : dict
        As for :func:`export_zemax_lens_data`.
    path : str
        Output ``.zmx`` file path.
    wavelength : float
        Wavelength [m] recorded as the primary.
    stop_surface : int
        Zero-based index of the aperture stop among refracting surfaces.
    aperture_diameter : float, optional
        Entrance pupil diameter in meters; falls back to
        ``prescription['aperture_diameter']``.
    back_focal_length : float, optional
        BFL [m] between the last refracting surface and the image plane.
        Defaults to zero (user must adjust).
    name : str, optional
        Lens name recorded in the file header.

    Notes
    -----
    Zemax's ``.zmx`` format has evolved over versions; this writer
    targets a format accepted by recent OpticStudio releases for
    sequential systems.  If your version refuses the file, start a new
    session in Zemax and manually enter the rows from
    :func:`export_zemax_lens_data` instead.
    """
    surfaces = prescription['surfaces']
    thicknesses = prescription['thicknesses']
    if aperture_diameter is None:
        aperture_diameter = prescription.get('aperture_diameter', 25.4e-3)
    bfl = back_focal_length or 0.0

    name = name or prescription.get('name',
        os.path.splitext(os.path.basename(path))[0])
    # Zemax EPD is in mm
    epd_mm = aperture_diameter * 1e3
    wvl_um = wavelength * 1e6

    lines = []
    lines.append('VERS 210000 0 123 0 0')
    lines.append('MODE SEQ')
    lines.append(f'NAME {name}')
    lines.append('UNIT MM X W X CM MR CPMM')
    lines.append(f'ENPD {epd_mm:.8f}')
    lines.append('ENVD 2.0e+01 1 0')
    lines.append('GFAC 0 0')
    lines.append('GCAT SCHOTT MISC')
    lines.append('RAIM 0 0 1 1 0 0 0 0 0')
    lines.append('PUSH 0 0 0 0 0 0')
    lines.append('SDMA 0 1 0')
    lines.append('FTYP 0 0 1 1 0 0 0')
    lines.append('ROPD 2')
    lines.append('PICB 1')
    lines.append('XFLN 0')
    lines.append('YFLN 0')
    lines.append('FWGN 1')
    lines.append('VDXN 0')
    lines.append('VDYN 0')
    lines.append('VCXN 0')
    lines.append('VCYN 0')
    lines.append('VANN 0')
    lines.append(f'WAVM 1 {wvl_um:.6f} 1.0')
    lines.append('PWAV 1')

    def _zemax_curv(R):
        return 0.0 if (R is None or np.isinf(R)) else 1.0 / (R * 1e3)

    def _zemax_disz(t_m):
        return t_m * 1e3

    semi_dia_mm = 0.5 * aperture_diameter * 1e3

    # SURF 0: object at infinity
    lines.append('SURF 0')
    lines.append('  TYPE STANDARD')
    lines.append('  CURV 0 0 0 0 0 ""')
    lines.append('  DISZ INFINITY')
    lines.append(f'  DIAM {semi_dia_mm:.6f} 0 0 0 1 ""')

    # Refracting surfaces
    for i, surf in enumerate(surfaces):
        idx = i + 1
        R = surf.get('radius')
        conic = surf.get('conic', 0.0) or 0.0
        glass = surf.get('glass_after', '')
        if glass and glass.lower() not in ('air', ''):
            glass_line = f'  GLAS {glass} 0 0 1.5 50.0 0 0 0 0 0 0'
        else:
            glass_line = None

        t_m = thicknesses[i] if i < len(thicknesses) else bfl
        disz_val = _zemax_disz(t_m)
        curv_val = _zemax_curv(R)

        lines.append(f'SURF {idx}')
        lines.append('  TYPE STANDARD')
        if i == stop_surface:
            lines.append('  STOP')
        lines.append(f'  CURV {curv_val:.10f} 0 0 0 0 ""')
        lines.append(f'  DISZ {disz_val:.8f}')
        if conic != 0.0:
            lines.append(f'  CONI {conic:.6f}')
        if glass_line is not None:
            lines.append(glass_line)
        lines.append(f'  DIAM {semi_dia_mm:.6f} 0 0 0 1 ""')

    # Image surface at BFL after last refracting surface
    last_idx = len(surfaces) + 1
    lines.append(f'SURF {last_idx}')
    lines.append('  TYPE STANDARD')
    lines.append('  CURV 0 0 0 0 0 ""')
    lines.append('  DISZ 0.0')
    lines.append(f'  DIAM {semi_dia_mm:.6f} 0 0 0 1 ""')

    lines.append('BLNK ')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# ============================================================================
# CODE V .seq (sequence) file I/O
# ============================================================================


def export_codev_seq(prescription, path, wavelength=1.31e-6,
                     stop_surface=0, aperture_diameter=None,
                     back_focal_length=None, name=None, units='M'):
    """Write a CODE V sequential ``.seq`` file for a prescription.

    The produced file uses the canonical CODE V ``LEN NEW`` / ``DIM`` /
    ``WL`` header followed by one surface block per refracting surface
    (``RDY`` / ``THI`` / ``GLA`` / optional ``CON`` / ``STO``) and an
    image plane at the end.

    Parameters
    ----------
    prescription : dict
        Same format used by :func:`apply_real_lens` and
        :func:`export_zemax_zmx`: ``{'surfaces': [...],
        'thicknesses': [...], 'aperture_diameter': ...}``.
    path : str
        Output ``.seq`` file path.
    wavelength : float
        Wavelength [m] to write in the header.
    stop_surface : int
        Zero-based index of the stop among refracting surfaces.
    aperture_diameter : float, optional
        Entrance-pupil diameter in meters; falls back to
        ``prescription['aperture_diameter']``.
    back_focal_length : float, optional
        BFL [m] from last refracting surface to image plane; defaults to 0.
    name : str, optional
        Lens name written as a ``! title`` comment.
    units : {'M', 'MM', 'IN'}, default 'M'
        Physical-length unit written on the ``DIM`` line.  Library
        prescriptions are SI meters; 'MM'/'IN' trigger conversion on
        write (useful for handing files to CODE V users).

    Notes
    -----
    CODE V's full ``.seq`` syntax covers several hundred commands
    (freeforms, multi-configurations, diffractive surfaces,
    zoom positions, tolerances, user-defined surfaces, ...).  This
    writer emits the common subset: spherical + conic surfaces,
    one wavelength, circular aperture, on-axis field.  Round-trips
    cleanly with :func:`load_codev_seq`; more exotic surface types
    must be hand-edited.
    """
    surfaces = prescription['surfaces']
    thicknesses = prescription['thicknesses']
    if aperture_diameter is None:
        aperture_diameter = prescription.get('aperture_diameter', 25.4e-3)
    bfl = back_focal_length or 0.0

    name = name or prescription.get('name',
        os.path.splitext(os.path.basename(path))[0])

    units = str(units).upper()
    if units not in ('M', 'MM', 'IN'):
        raise ValueError(
            f"export_codev_seq: units must be M, MM, or IN (got {units!r})")
    scale = {'M': 1.0, 'MM': 1e3, 'IN': 1.0 / 0.0254}[units]

    def _fmt(v):
        if v is None or np.isinf(v):
            return 'INFINITY'
        return f'{v * scale:.8f}'

    wvl_nm = wavelength * 1e9

    lines = []
    lines.append(f'! Generated by lumenairy (export_codev_seq)')
    lines.append(f'! {name}')
    lines.append('LEN NEW')
    lines.append(f'DIM {units}')
    lines.append(f'WL {wvl_nm:.4f}')
    lines.append('REF 1')
    lines.append(
        f'APE F1 CIR R {aperture_diameter * 0.5 * scale:.6f}')

    # CODE V .seq surface blocks.  We write object (S0), refracting
    # surfaces (S1..Sn), and image plane.  Thicknesses in library
    # prescriptions are gap-after-surface; the same convention holds for
    # CODE V's THI.
    lines.append('!')
    lines.append('! Object surface')
    lines.append('SO')
    lines.append('  RDY INFINITY')
    lines.append('  THI INFINITY')
    lines.append('')

    for i, surf in enumerate(surfaces):
        R = surf.get('radius')
        conic = surf.get('conic', 0.0) or 0.0
        glass = surf.get('glass_after', '')
        t_m = thicknesses[i] if i < len(thicknesses) else bfl

        lines.append(f'! Surface {i + 1}')
        lines.append(f'S{i + 1}')
        if i == stop_surface:
            lines.append('  STO')
        lines.append(f'  RDY {_fmt(R)}')
        lines.append(f'  THI {_fmt(t_m)}')
        if glass and glass.lower() not in ('air', ''):
            lines.append(f'  GLA {glass}')
        if conic != 0.0:
            lines.append(f'  CON {conic:.8f}')
        lines.append('')

    # Image surface
    lines.append('! Image surface')
    lines.append(f'SI')
    lines.append('  RDY INFINITY')
    lines.append(f'  THI {_fmt(bfl)}')
    lines.append('')

    lines.append('GO')
    lines.append('END')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def load_codev_seq(filepath, name=None):
    """Parse a CODE V ``.seq`` file into a library prescription dict.

    Reads the common subset emitted by :func:`export_codev_seq` plus
    several conventional variants:

    * ``DIM M`` / ``DIM MM`` / ``DIM IN`` — unit handling
    * ``WL <nm>`` — wavelength (not round-tripped, but stored in the
      returned dict under key ``wavelength``)
    * ``S<i>`` headers or ``SO`` / ``SI`` markers
    * ``RDY <val>`` / ``CUY <c>`` — radius or curvature (``CUY`` is
      converted via ``R = 1/c``)
    * ``THI <val>`` — thickness
    * ``GLA <name>`` — glass (``GLA AIR`` or ``GLA`` omitted = air)
    * ``CON <k>`` — conic constant
    * ``STO`` — aperture stop surface
    * ``APE F1 CIR R <r>`` — aperture radius

    Returns a dict of the same shape as :func:`make_singlet`:
    ``{'name', 'aperture_diameter', 'surfaces', 'thicknesses',
       'wavelength', 'stop_index'}``.

    Unknown commands are silently ignored so that files written by
    CODE V with extra boilerplate (tolerancing directives, zoom
    positions, spot-diagram commands, ...) still parse.
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()

    # Strip comments (lines starting with '!') and blank lines.
    lines_iter = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s or s.startswith('!'):
            continue
        lines_iter.append(s)

    units = 'M'
    wavelength_m = None
    aperture_diameter_m = None

    # Surface blocks as we encounter them.  `current` is a dict mutated
    # while inside S<i> / SO / SI; flushed to `surfaces` on the next
    # surface header or END.
    surfaces_raw = []
    current = None
    stop_index_raw = None

    def _unit_to_meters(v):
        if v is None:
            return None
        return {'M': 1.0, 'MM': 1e-3,
                'IN': 0.0254}[units] * float(v)

    def _flush():
        nonlocal current
        if current is not None and current.get('kind') == 'refracting':
            surfaces_raw.append(current)
        current = None

    i = 0
    while i < len(lines_iter):
        tokens = lines_iter[i].split()
        cmd = tokens[0].upper()

        if cmd == 'LEN' or cmd == 'GO' or cmd == 'END' or cmd == 'REF':
            if cmd == 'END':
                _flush()
                break
            i += 1
            continue

        if cmd == 'DIM':
            if len(tokens) >= 2:
                unit_tok = tokens[1].upper()
                if unit_tok in ('M', 'MM', 'IN'):
                    units = unit_tok
            i += 1
            continue

        if cmd == 'WL':
            if len(tokens) >= 2:
                try:
                    wavelength_m = float(tokens[1]) * 1e-9
                except ValueError:
                    pass
            i += 1
            continue

        if cmd == 'APE':
            # APE F1 CIR R <radius>  or  APE CIR R <radius>
            for j, tk in enumerate(tokens):
                if tk.upper() == 'R' and j + 1 < len(tokens):
                    try:
                        r = float(tokens[j + 1])
                        aperture_diameter_m = 2.0 * _unit_to_meters(r)
                    except ValueError:
                        pass
                    break
            i += 1
            continue

        # Surface headers.  Accept S<int>, SO, SI, SIM.
        if cmd == 'SO':
            _flush()
            current = {'kind': 'object'}
            i += 1
            continue
        if cmd in ('SI', 'SIM'):
            _flush()
            current = {'kind': 'image'}
            i += 1
            continue
        if cmd.startswith('S') and len(cmd) > 1 and cmd[1:].isdigit():
            _flush()
            current = {'kind': 'refracting',
                       'index': int(cmd[1:]),
                       'radius': None, 'conic': 0.0,
                       'thickness': 0.0, 'glass': 'air',
                       'is_stop': False}
            i += 1
            continue

        # Commands that apply to the current surface block
        if current is None:
            i += 1
            continue

        if cmd == 'STO':
            if current.get('kind') == 'refracting':
                current['is_stop'] = True
                stop_index_raw = current['index']
            i += 1
            continue

        if cmd == 'RDY':
            if len(tokens) >= 2:
                val = tokens[1]
                if val.upper() == 'INFINITY' or val.upper() == 'INF':
                    current['radius'] = float('inf')
                else:
                    try:
                        current['radius'] = _unit_to_meters(float(val))
                    except ValueError:
                        pass
            i += 1
            continue

        if cmd == 'CUY':
            if len(tokens) >= 2:
                try:
                    c = float(tokens[1])
                    current['radius'] = (float('inf') if c == 0
                                         else 1.0 / c / {
                        'M': 1.0, 'MM': 1e3,
                        'IN': 1.0 / 0.0254}[units])
                except ValueError:
                    pass
            i += 1
            continue

        if cmd == 'THI':
            if len(tokens) >= 2:
                val = tokens[1]
                if val.upper() == 'INFINITY' or val.upper() == 'INF':
                    current['thickness'] = float('inf')
                else:
                    try:
                        current['thickness'] = _unit_to_meters(float(val))
                    except ValueError:
                        pass
            i += 1
            continue

        if cmd == 'GLA':
            if len(tokens) >= 2:
                g = tokens[1]
                current['glass'] = (
                    'air' if g.upper() in ('AIR', '') else g)
            i += 1
            continue

        if cmd == 'CON':
            if len(tokens) >= 2:
                try:
                    current['conic'] = float(tokens[1])
                except ValueError:
                    pass
            i += 1
            continue

        # Unknown directive — ignore
        i += 1

    _flush()

    if not surfaces_raw:
        raise ValueError(
            f"load_codev_seq: no refracting surfaces found in "
            f"{filepath!r}.  Check the file is a valid CODE V .seq.")

    # Build the library prescription
    surfaces = []
    thicknesses = []
    prev_glass = 'air'
    # The last refracting surface's thickness goes to the image plane,
    # not the prescription's `thicknesses` list (library convention is
    # that `thicknesses[i]` is the gap AFTER surface i, stopping at the
    # last refracting surface; the final image-plane gap is the BFL and
    # not part of the prescription).
    for i, s in enumerate(surfaces_raw):
        surf_dict = {
            'radius': s.get('radius', float('inf')),
            'conic': s.get('conic', 0.0),
            'aspheric_coeffs': None,
            'radius_y': None, 'conic_y': None,
            'aspheric_coeffs_y': None,
            'glass_before': prev_glass,
            'glass_after': s.get('glass', 'air'),
        }
        surfaces.append(surf_dict)
        if i < len(surfaces_raw) - 1:
            thicknesses.append(s.get('thickness', 0.0))
        prev_glass = s.get('glass', 'air')

    # Figure out the stop index (0-based among refracting surfaces).
    stop_index = None
    for i, s in enumerate(surfaces_raw):
        if s.get('is_stop'):
            stop_index = i
            break

    result = {
        'name': name or os.path.splitext(os.path.basename(filepath))[0],
        'aperture_diameter': aperture_diameter_m or 25.4e-3,
        'surfaces': surfaces,
        'thicknesses': thicknesses,
    }
    if wavelength_m is not None:
        result['wavelength'] = wavelength_m
    if stop_index is not None:
        result['stop_index'] = stop_index
    return result


# ============================================================================
# Quadoa Optikos .qos (JSON) file I/O -- best-effort
# ============================================================================
#
# Quadoa Optikos uses a JSON-based ``.qos`` system file.  The full schema
# is not publicly documented at the level of every field, so the
# exporter below writes a self-defined JSON layout that captures every
# field a lumenairy prescription holds (surfaces, glasses, thicknesses,
# aperture, conics, asphere coefficients, biconic Y-axis radii, stop
# index, wavelength, name, units, semi-diameters).  Importer round-trips
# this layout exactly.  When Quadoa publishes a stable schema -- or when
# users supply a reference ``.qos`` -- this can be tightened.  The
# JSON-writer side is intentionally schema-versioned so future readers
# can detect the layout.
# ============================================================================

QUADOA_SCHEMA_VERSION = '1.0'


def _quadoa_serialize_radius(R, scale):
    if R is None or not np.isfinite(R):
        return None
    return float(R) * scale


def _quadoa_serialize_aspheric(coeffs):
    if coeffs is None:
        return None
    return [float(c) for c in coeffs]


def export_quadoa_qos(prescription, path, wavelength=1.31e-6,
                      stop_surface=0, aperture_diameter=None,
                      back_focal_length=None, name=None, units='M'):
    """Write a Quadoa Optikos-style ``.qos`` JSON system file.

    Quadoa's native file format is JSON-based.  The official schema is
    not fully publicly documented, so this writer emits a self-defined
    JSON layout (schema version :data:`QUADOA_SCHEMA_VERSION`) that
    captures every field a lumenairy prescription carries -- surfaces,
    radii (incl. biconic Y), conics, asphere coefficients, glasses,
    thicknesses, semi-diameters, aperture, stop index, wavelength,
    and units.  :func:`load_quadoa_qos` reads this layout back losslessly.

    .. warning::
        Quadoa-readability of the produced file is **not yet verified**.
        For pure round-tripping inside lumenairy this is exact; for
        external interchange with Quadoa Optikos itself, validate
        against a known-good reference ``.qos`` first.

    Parameters
    ----------
    prescription : dict
        Same format used by :func:`apply_real_lens`,
        :func:`export_zemax_zmx`, and :func:`export_codev_seq`.
    path : str
        Output ``.qos`` file path.
    wavelength : float
        Reference wavelength [m].
    stop_surface : int
        Zero-based index of the stop among refracting surfaces.
    aperture_diameter : float, optional
        Entrance-pupil diameter [m]; falls back to
        ``prescription['aperture_diameter']``.
    back_focal_length : float, optional
        BFL [m] from last surface to image plane.
    name : str, optional
        System name written into the JSON header.
    units : {'M', 'MM', 'IN'}, default 'M'
        Length units written in the header (file body is rescaled
        on write to preserve the chosen unit).

    See Also
    --------
    load_quadoa_qos
    export_zemax_zmx, export_codev_seq
    """
    import json

    surfaces = prescription['surfaces']
    thicknesses = prescription['thicknesses']
    if aperture_diameter is None:
        aperture_diameter = prescription.get('aperture_diameter', 25.4e-3)
    bfl = back_focal_length or 0.0

    name = name or prescription.get('name',
        os.path.splitext(os.path.basename(path))[0])

    units = str(units).upper()
    if units not in ('M', 'MM', 'IN'):
        raise ValueError(
            f"export_quadoa_qos: units must be M, MM, or IN (got {units!r})")
    scale = {'M': 1.0, 'MM': 1e3, 'IN': 1.0 / 0.0254}[units]

    surf_list = []
    for i, surf in enumerate(surfaces):
        t_m = thicknesses[i] if i < len(thicknesses) else bfl
        sd = surf.get('semi_diameter')
        entry = {
            'index': i,
            'radius': _quadoa_serialize_radius(
                surf.get('radius'), scale),
            'radius_y': _quadoa_serialize_radius(
                surf.get('radius_y'), scale),
            'conic': float(surf.get('conic', 0.0) or 0.0),
            'conic_y': (None if surf.get('conic_y') is None
                        else float(surf['conic_y'])),
            'aspheric_coeffs': _quadoa_serialize_aspheric(
                surf.get('aspheric_coeffs')),
            'aspheric_coeffs_y': _quadoa_serialize_aspheric(
                surf.get('aspheric_coeffs_y')),
            'glass_before': surf.get('glass_before', 'air'),
            'glass_after': surf.get('glass_after', 'air'),
            'thickness': float(t_m) * scale,
            'is_stop': bool(i == stop_surface),
            'semi_diameter': (None if sd is None or not np.isfinite(sd)
                              else float(sd) * scale),
            'comment': surf.get('comment', ''),
        }
        surf_list.append(entry)

    doc = {
        'format': 'quadoa-optikos-system',
        'schema_version': QUADOA_SCHEMA_VERSION,
        'generated_by': 'lumenairy.export_quadoa_qos',
        'name': name,
        'units': units,
        'wavelength_nm': float(wavelength) * 1e9,
        'aperture_diameter': float(aperture_diameter) * scale,
        'back_focal_length': float(bfl) * scale,
        'stop_surface': int(stop_surface),
        'surfaces': surf_list,
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2)
        f.write('\n')


def load_quadoa_qos(filepath, name=None):
    """Parse a Quadoa Optikos-style ``.qos`` JSON file into a
    lumenairy prescription dict.

    Round-trips losslessly with :func:`export_quadoa_qos` (schema
    version :data:`QUADOA_SCHEMA_VERSION`).  If the file is missing
    the ``format`` / ``schema_version`` header but otherwise looks
    JSON-like with a ``surfaces`` array, the parser falls back to
    a permissive read; unknown fields are preserved on each surface
    under ``surf['_extras']`` so callers can inspect them without
    losing information.

    Parameters
    ----------
    filepath : str
    name : str, optional
        Override for the prescription name; defaults to the JSON
        ``name`` field or the file stem.

    Returns
    -------
    dict
        Standard lumenairy prescription dict
        (``{'name', 'aperture_diameter', 'surfaces', 'thicknesses',
            'wavelength', 'stop_index', ...}``).

    See Also
    --------
    export_quadoa_qos
    """
    import json

    with open(filepath, 'r', encoding='utf-8') as f:
        doc = json.load(f)

    if not isinstance(doc, dict) or 'surfaces' not in doc:
        raise ValueError(
            f"load_quadoa_qos: {filepath!r} is not a recognisable "
            f"Quadoa-style JSON system file (no 'surfaces' array).")

    units = str(doc.get('units', 'M')).upper()
    if units not in ('M', 'MM', 'IN'):
        warnings.warn(
            f"load_quadoa_qos: unknown units {units!r}, assuming meters",
            UserWarning, stacklevel=2)
        units = 'M'
    inv_scale = {'M': 1.0, 'MM': 1e-3, 'IN': 0.0254}[units]

    def _radius_in(v):
        if v is None:
            return float('inf')
        return float(v) * inv_scale

    raw = doc['surfaces']
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"load_quadoa_qos: {filepath!r} has empty 'surfaces' list.")

    surfaces = []
    thicknesses = []
    stop_index = None
    semi_diameters = []
    known_keys = {
        'index', 'radius', 'radius_y', 'conic', 'conic_y',
        'aspheric_coeffs', 'aspheric_coeffs_y',
        'glass_before', 'glass_after',
        'thickness', 'is_stop', 'semi_diameter', 'comment',
    }
    for i, s in enumerate(raw):
        if not isinstance(s, dict):
            raise ValueError(
                f"load_quadoa_qos: surface {i} is not a JSON object.")
        sd = s.get('semi_diameter')
        surf = {
            'radius': _radius_in(s.get('radius')),
            'conic': float(s.get('conic', 0.0) or 0.0),
            'aspheric_coeffs': (None if s.get('aspheric_coeffs') is None
                                else list(s['aspheric_coeffs'])),
            'radius_y': (None if s.get('radius_y') is None
                         else _radius_in(s['radius_y'])),
            'conic_y': (None if s.get('conic_y') is None
                        else float(s['conic_y'])),
            'aspheric_coeffs_y': (None if s.get('aspheric_coeffs_y') is None
                                  else list(s['aspheric_coeffs_y'])),
            'glass_before': s.get('glass_before', 'air'),
            'glass_after': s.get('glass_after', 'air'),
        }
        if sd is not None:
            surf['semi_diameter'] = float(sd) * inv_scale
            semi_diameters.append(surf['semi_diameter'])
        if s.get('comment'):
            surf['comment'] = s['comment']
        extras = {k: v for k, v in s.items() if k not in known_keys}
        if extras:
            surf['_extras'] = extras
        surfaces.append(surf)
        if i < len(raw) - 1:
            thicknesses.append(float(s.get('thickness', 0.0)) * inv_scale)
        if s.get('is_stop'):
            stop_index = i

    aperture_diameter = doc.get('aperture_diameter')
    aperture_m = (
        25.4e-3 if aperture_diameter is None
        else float(aperture_diameter) * inv_scale)

    result = {
        'name': name or doc.get('name')
            or os.path.splitext(os.path.basename(filepath))[0],
        'aperture_diameter': aperture_m,
        'surfaces': surfaces,
        'thicknesses': thicknesses,
    }
    if 'wavelength_nm' in doc:
        result['wavelength'] = float(doc['wavelength_nm']) * 1e-9
    if stop_index is None and 'stop_surface' in doc:
        try:
            stop_index = int(doc['stop_surface'])
        except (TypeError, ValueError):
            stop_index = None
    if stop_index is not None:
        result['stop_index'] = stop_index
    if semi_diameters:
        result['has_semi_diameters'] = True
    return result
