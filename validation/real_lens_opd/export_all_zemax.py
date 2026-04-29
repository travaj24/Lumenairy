"""Generate Zemax prescription files for every case, without rerunning
the wave simulation.  Uses paraxial EFL from the geometric trace.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
if _LIB_ROOT not in sys.path:
    sys.path.insert(0, _LIB_ROOT)

from lumenairy.prescriptions import (
    export_zemax_lens_data, export_zemax_zmx,
)
from lumenairy.raytrace import (
    surfaces_from_prescription, system_abcd,
)

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from lens_cases import list_cases


OUT_DIR = os.path.join(_HERE, 'zemax_prescriptions')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = list_cases()
    print(f"Exporting Zemax files for {len(cases)} cases ...")
    for case in cases:
        pres = dict(case['prescription'])
        pres['aperture_diameter'] = case['aperture']
        surfs = surfaces_from_prescription(pres)
        _, efl, bfl, _ = system_abcd(surfs, case['wavelength'])

        txt_path = os.path.join(OUT_DIR, f"{case['name']}.txt")
        zmx_path = os.path.join(OUT_DIR, f"{case['name']}.zmx")
        export_zemax_lens_data(
            pres, txt_path,
            wavelength=case['wavelength'],
            stop_surface=0,
            aperture_diameter=case['aperture'],
            back_focal_length=bfl,
            description=case['description'],
            extra_notes=[
                f'Paraxial EFL at this wavelength: {efl*1e3:.4f} mm',
                f'Paraxial BFL at this wavelength: {bfl*1e3:.4f} mm',
                'Use Analysis > Aberrations > OPD fan or > Wavefront '
                'Map to compare against piston+tilt+defocus-removed '
                'residual in the wave validation.',
            ])
        export_zemax_zmx(
            pres, zmx_path,
            wavelength=case['wavelength'],
            stop_surface=0,
            aperture_diameter=case['aperture'],
            back_focal_length=bfl,
            name=case['name'])
        print(f"  {case['name']}  (EFL={efl*1e3:.2f} mm, BFL={bfl*1e3:.2f} mm)")

    # Also emit an index file
    idx_path = os.path.join(OUT_DIR, 'INDEX.md')
    with open(idx_path, 'w', encoding='utf-8') as f:
        f.write('# Zemax-compatible prescriptions for validation cases\n\n')
        f.write('Each case has two files:\n')
        f.write('- `<name>.txt` -- human-readable lens-data table for '
                'manual entry into Zemax\'s Lens Data Editor.\n')
        f.write('- `<name>.zmx` -- minimal Zemax sequential file for '
                'File > Open.  After loading, verify the wavelength, '
                'aperture type, and stop index match what is written '
                'in the `.txt` header.\n\n')
        f.write('| Case | Wavelength [um] | Aperture [mm] | EFL [mm] | BFL [mm] |\n')
        f.write('|---|---:|---:|---:|---:|\n')
        for case in cases:
            pres = dict(case['prescription'])
            pres['aperture_diameter'] = case['aperture']
            surfs = surfaces_from_prescription(pres)
            _, efl, bfl, _ = system_abcd(surfs, case['wavelength'])
            f.write(
                f"| `{case['name']}` | {case['wavelength']*1e6:.4f} "
                f"| {case['aperture']*1e3:.2f} | {efl*1e3:.2f} "
                f"| {bfl*1e3:.2f} |\n")
    print(f"\nIndex: {idx_path}")


if __name__ == '__main__':
    main()
