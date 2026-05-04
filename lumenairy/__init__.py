"""
lumenairy — Coherent Optical Field Propagation Library
================================================================

A comprehensive library for simulating coherent optical beam propagation
using the Angular Spectrum Method (ASM) and related techniques.

Convention: exp(-i*omega*t) time dependence throughout.
Units: SI meters for all spatial quantities.

Usage::

    from lumenairy import angular_spectrum_propagate, apply_thin_lens
    # or
    import lumenairy as op
    E_out = op.angular_spectrum_propagate(E_in, z, wavelength, dx)

All public functions are available directly from the package namespace.
For more granular imports, use the submodules::

    from lumenairy.propagation import angular_spectrum_propagate
    from lumenairy.lenses import apply_real_lens
    from lumenairy.glass import get_glass_index, GLASS_REGISTRY

Author: Andrew Traverso
"""

# ── Propagation ──────────────────────────────────────────────────────────
from .propagation import (
    angular_spectrum_propagate,
    angular_spectrum_propagate_tilted,
    scalable_angular_spectrum_propagate,
    fresnel_propagate,
    fraunhofer_propagate,
    set_default_complex_dtype,
    get_default_complex_dtype,
    set_asm_cache_size,
    set_fft_plan_cache_size,
    clear_asm_caches,
    reset_fft_backend,
    rayleigh_sommerfeld_propagate,
    resample_field,
    # FFT backend configuration
    PYFFTW_AVAILABLE,
    CUPY_AVAILABLE,
    set_fft_threads,
    set_fft_fallback,
    reset_fft_backend,
    # Precision configuration (complex64 vs complex128)
    DEFAULT_COMPLEX_DTYPE,
)

# ── Optional optimisation-backend availability flags ────────────────────
# Users / runners can inspect these before toggling tunables that
# depend on the optional backends (e.g. the numexpr-fused phase-screen
# path inside apply_real_lens).  Truthy if the package is importable
# in the current environment.
from .lenses import NUMEXPR_AVAILABLE

# ── Backend / runtime helpers ────────────────────────────────────────────
from ._backends import available_cpus

# ── Lenses ───────────────────────────────────────────────────────────────
from .lenses import (
    apply_thin_lens,
    apply_spherical_lens,
    apply_aspheric_lens,
    apply_real_lens,
    apply_real_lens_traced,
    surface_sag_general,
    surface_sag_biconic,
    apply_cylindrical_lens,
    apply_grin_lens,
    apply_axicon,
    check_grid_vs_apertures,
)
from .lenses import apply_real_lens_maslov

# ── Glass catalog ────────────────────────────────────────────────────────
from .glass import (
    get_glass_index,
    get_glass_index_complex,
    GLASS_REGISTRY,
)

# ── Optical elements ─────────────────────────────────────────────────────
from .elements import (
    apply_mirror,
    apply_aperture,
    apply_gaussian_aperture,
    apply_mask,
    zernike,
    apply_zernike_aberration,
    generate_turbulence_screen,
)

# ── Sources ──────────────────────────────────────────────────────────────
from .sources import (
    create_gaussian_beam,
    create_hermite_gauss,
    create_laguerre_gauss,
    hermite_physicist,
    laguerre_generalized,
)

# ── Beam analysis ────────────────────────────────────────────────────────
from .analysis import (
    beam_centroid,
    beam_d4sigma,
    beam_power,
    strehl_ratio,
    check_sampling_conditions,
    compute_psf,
    compute_otf,
    compute_mtf,
    mtf_radial,
    remove_wavefront_modes,
    opd_pv_rms,
    wave_opd_1d,
    wave_opd_2d,
    check_opd_sampling,
    chromatic_focal_shift,
    polychromatic_strehl,
    radial_power_bands,
    zernike_polynomial,
    zernike_basis_matrix,
    zernike_decompose,
    zernike_reconstruct,
    zernike_index_to_nm,
    zernike_nm_to_index,
)

# ── Off-axis + extended source helpers ─────────────────────────────────
from .sources import (
    create_tilted_plane_wave,
    create_point_source,
    create_multi_field_sources,
    create_top_hat_beam,
    create_annular_beam,
    create_fiber_mode,
    create_led_source,
    create_bessel_beam,
)

# ── High-NA vector diffraction (Richards-Wolf) ─────────────────────────
from .vector_diffraction import (
    richards_wolf_focus,
    debye_wolf_psf,
)

# ── Partial coherence / extended-source imaging ────────────────────────
from .coherence import (
    koehler_image,
    extended_source_image,
    mutual_coherence,
)

# ── Detector model / wavefront sensing ─────────────────────────────────
from .detector import (
    apply_detector,
    shack_hartmann,
)

# ── Thin-film coatings ────────────────────────────────────────────────
from .coatings import (
    coating_reflectance,
    quarter_wave_ar,
    broadband_ar_v_coat,
)

# ── Interferometry ────────────────────────────────────────────────────
from .interferometry import (
    simulate_interferogram,
    phase_shift_extract,
    fringe_spacing,
)

# ── Freeform surfaces ─────────────────────────────────────────────────
from .freeform import (
    surface_sag_xy_polynomial,
    surface_sag_zernike_freeform,
    surface_sag_chebyshev,
    surface_sag_freeform,
)

# ── Ghost analysis ────────────────────────────────────────────────────
from .ghost import (
    enumerate_ghost_paths,
    ghost_analysis,
)

# ── BSDF surface scatter (stray-light analysis) ─────────────────────────
from .bsdf import (
    BSDFModel,
    LambertianBSDF,
    GaussianBSDF,
    HarveyShackBSDF,
    make_bsdf,
    sample_scatter_rays,
)

# ── RCWA (rigorous coupled-wave analysis) ─────────────────────────────
from .rcwa import (
    rcwa_1d,
    grating_efficiency_vs_wavelength,
)

# ── Multi-configuration + afocal mode ─────────────────────────────────
from .multiconfig import (
    Configuration,
    multi_config_merit,
    create_zoom_configs,
    afocal_angular_magnification,
    beam_expander_prescription,
    keplerian_telescope,
)

# ── Through-focus / tolerancing ─────────────────────────────────────────
from .through_focus import (
    single_plane_metrics,
    diffraction_limited_peak,
    through_focus_scan,
    find_best_focus,
    plot_through_focus,
    ThroughFocusResult,
    Perturbation,
    apply_perturbations,
    tolerancing_sweep,
    monte_carlo_tolerancing,
)

# ── Hybrid wave/ray design optimization ────────────────────────────────
from .optimize import (
    DesignParameterization,
    MultiPrescriptionParameterization,
    MeritTerm,
    FocalLengthMerit,
    BackFocalLengthMerit,
    SphericalSeidelMerit,
    StrehlMerit,
    RMSWavefrontMerit,
    SpotSizeMerit,
    ChromaticFocalShiftMerit,
    MatchIdealThinLensMerit,
    MatchIdealSystemMerit,
    MatchTargetOPDMerit,
    ZernikeCoefficientMerit,
    LGAberrationMerit,
    CompositeMerit,
    CallableMerit,
    MultiWavelengthMerit,
    MultiFieldMerit,
    MinThicknessMerit,
    MaxThicknessMerit,
    MinBackFocalLengthMerit,
    MaxFNumberMerit,
    ToleranceAwareMerit,
    EvaluationContext,
    DesignResult,
    design_optimize,
)

# ── Phase-space asymptotic propagator + LG aberration tensor ────────────
from .asymptotic import (
    CanonicalPolyFit,
    AberrationTensorResult,
    fit_canonical_polynomials,
    aberration_tensor,
    propagate_modal_asymptotic,
    solve_envelope_stationary,
    lg_polynomial,
    hg_polynomial,
    evaluate_lg_mode,
    evaluate_hg_mode,
    decompose_lg,
    decompose_hg,
    lg_seidel_label,
    gaussian_moment_2d,
    gaussian_moment_table_2d,
)

# ── DOE / Gratings / Phase I/O ──────────────────────────────────────────
from .doe import (
    create_periodic_phase_mask,
    create_microlens_array,
    makedammann2d,
    load_phase_file,
    save_phase_file,
    load_fits_field,
    save_fits_field,
)

# ── Phase retrieval ─────────────────────────────────────────────────────
from .phase_retrieval import (
    gerchberg_saxton,
    error_reduction,
    hybrid_input_output,
)

# ── Code generation: prescription -> standalone simulation script ───────
from .codegen import (
    generate_simulation_script,
    generate_script_from_zmx,
    generate_script_from_txt,
)

# ── Progress reporting (opt-in hook for long-running functions) ─────────
from .progress import ProgressCallback, ProgressScaler, call_progress

# ── Polarization / Jones calculus ───────────────────────────────────────
from .polarization import (
    JonesField,
    apply_jones_matrix,
    apply_polarizer,
    apply_waveplate,
    apply_half_wave_plate,
    apply_quarter_wave_plate,
    apply_rotator,
    create_linear_polarized,
    create_circular_polarized,
    create_elliptical_polarized,
    stokes_parameters,
    degree_of_polarization,
    polarization_ellipse,
)

# ── Lens prescriptions ──────────────────────────────────────────────────
from .prescriptions import (
    make_singlet,
    make_doublet,
    make_cylindrical,
    make_biconic,
    thorlabs_lens,
    load_zmx_prescription,
    load_zemax_prescription_txt,
    export_zemax_lens_data,
    export_zemax_zmx,
    load_codev_seq,
    export_codev_seq,
    export_quadoa_qos,
    load_quadoa_qos,
    QUADOA_SCHEMA_VERSION,
    THORLABS_CATALOG,
)

# ── Geometric ray tracing ────────────────────────────────────────────────
from .raytrace import (
    RayBundle,
    Surface,
    TraceResult,
    trace,
    surfaces_from_prescription,
    make_ray,
    make_fan,
    make_ring,
    make_grid,
    make_rings,
    apply_doe_phase_traced,
    trace_prescription,
    system_abcd,
    system_abcd_prescription,
    seidel_coefficients,
    seidel_prescription,
    spot_rms,
    spot_geo_radius,
    spot_diagram,
    ray_fan_data,
    ray_fan_plot,
    ray_fan_plot_prescription,
    opd_fan_data,
    through_focus_rms,
    refocus,
    find_stop,
    compute_pupils,
    lens_abcd,
    find_lenses,
    LensInfo,
    PupilInfo,
    RAY_OK,
    RAY_TIR,
    RAY_APERTURE,
    RAY_MISSED_SURFACE,
    RAY_NAN,
    RAY_EVANESCENT,
    find_paraxial_focus,
    trace_summary,
    prescription_summary,
    surfaces_from_elements,
    raytrace_system,
)

# ── System propagation ──────────────────────────────────────────────────
from .system import propagate_through_system

# ── Storage (unified HDF5 / Zarr) ────────────────────────────────────────
from .storage import (
    # Unified dispatch API
    set_storage_backend,
    get_storage_backend,
    default_extension,
    append_plane,
    load_planes,
    list_planes,
    load_plane_by_label,
    load_plane_slice,
    write_sim_metadata,
    read_sim_metadata,
    # Backwards-compatible aliases
    list_planes_store,
    load_plane_by_label_store,
    load_plane_slice_store,
    write_metadata,
    read_metadata,
    # HDF5-specific functions
    save_field_h5,
    load_field_h5,
    save_planes_h5,
    load_planes_h5,
    save_jones_field_h5,
    load_jones_field_h5,
    append_plane_h5,
    list_h5_contents,
    TempFieldStore,
)

# ── Memory-aware batching helpers ───────────────────────────────────────
from .memory import (
    available_memory_bytes,
    total_memory_bytes,
    memory_info,
    bytes_per_element,
    array_bytes,
    estimate_op_memory,
    pick_batch_size,
    should_split,
    format_bytes,
    print_memory_report,
    get_ram_budget,
    set_max_ram,
)

# ── Plotting utilities ─────────────────────────────────────────────────
from .plotting import (
    plot_intensity,
    plot_phase,
    plot_field,
    plot_amplitude_phase,
    plot_cross_section,
    plot_planes_grid,
    plot_psf,
    plot_mtf,
    plot_stokes,
    plot_polarization_ellipses,
    plot_beam_profile,
    plot_jones_pupil,
    compute_jones_pupil,
)

__version__ = "3.3.1"

__all__ = [
    # Propagation
    'angular_spectrum_propagate',
    'angular_spectrum_propagate_tilted',
    'scalable_angular_spectrum_propagate',
    'fresnel_propagate',
    'fraunhofer_propagate',
    'rayleigh_sommerfeld_propagate',
    'resample_field',
    'set_default_complex_dtype',
    'get_default_complex_dtype',
    'set_asm_cache_size',
    'set_fft_plan_cache_size',
    'clear_asm_caches',
    'reset_fft_backend',
    # Lenses
    'apply_thin_lens',
    'apply_spherical_lens',
    'apply_aspheric_lens',
    'apply_real_lens',
    'apply_real_lens_traced',
    'apply_real_lens_maslov',
    'surface_sag_general',
    'surface_sag_biconic',
    'make_cylindrical',
    'make_biconic',
    'apply_cylindrical_lens',
    'apply_grin_lens',
    'apply_axicon',
    'check_grid_vs_apertures',
    # Glass
    'get_glass_index',
    'get_glass_index_complex',
    'GLASS_REGISTRY',
    # Elements
    'apply_mirror',
    'apply_aperture',
    'apply_gaussian_aperture',
    'apply_mask',
    'zernike',
    'apply_zernike_aberration',
    'generate_turbulence_screen',
    # Sources
    'create_gaussian_beam',
    'create_hermite_gauss',
    'create_laguerre_gauss',
    'hermite_physicist',
    'laguerre_generalized',
    'create_tilted_plane_wave',
    'create_point_source',
    'create_multi_field_sources',
    'create_top_hat_beam',
    'create_annular_beam',
    'create_fiber_mode',
    'create_led_source',
    'create_bessel_beam',
    # Analysis
    'beam_centroid',
    'beam_d4sigma',
    'beam_power',
    'strehl_ratio',
    'check_sampling_conditions',
    'compute_psf',
    'compute_otf',
    'compute_mtf',
    'mtf_radial',
    'remove_wavefront_modes',
    'opd_pv_rms',
    'wave_opd_1d',
    'wave_opd_2d',
    'check_opd_sampling',
    'chromatic_focal_shift',
    'polychromatic_strehl',
    'radial_power_bands',
    'zernike_polynomial',
    'zernike_basis_matrix',
    'zernike_decompose',
    'zernike_reconstruct',
    'zernike_index_to_nm',
    'zernike_nm_to_index',
    # Geometric ray tracing
    'RayBundle',
    'Surface',
    'TraceResult',
    'trace',
    'surfaces_from_prescription',
    'make_ray',
    'make_fan',
    'make_ring',
    'make_grid',
    'make_rings',
    'apply_doe_phase_traced',
    'trace_prescription',
    'system_abcd',
    'system_abcd_prescription',
    'seidel_coefficients',
    'seidel_prescription',
    'spot_rms',
    'spot_geo_radius',
    'spot_diagram',
    'ray_fan_data',
    'ray_fan_plot',
    'ray_fan_plot_prescription',
    'opd_fan_data',
    'through_focus_rms',
    'refocus',
    'find_stop',
    'compute_pupils',
    'lens_abcd',
    'find_lenses',
    'LensInfo',
    'PupilInfo',
    'RAY_OK',
    'RAY_TIR',
    'RAY_APERTURE',
    'RAY_MISSED_SURFACE',
    'RAY_NAN',
    'RAY_EVANESCENT',
    'find_paraxial_focus',
    'trace_summary',
    'prescription_summary',
    'surfaces_from_elements',
    'raytrace_system',
    # Through-focus / tolerancing
    'single_plane_metrics',
    'diffraction_limited_peak',
    'through_focus_scan',
    'find_best_focus',
    'plot_through_focus',
    'ThroughFocusResult',
    'Perturbation',
    'apply_perturbations',
    'tolerancing_sweep',
    'monte_carlo_tolerancing',
    # Hybrid wave/ray design optimization
    'DesignParameterization',
    'MultiPrescriptionParameterization',
    'MeritTerm',
    'FocalLengthMerit',
    'BackFocalLengthMerit',
    'SphericalSeidelMerit',
    'StrehlMerit',
    'RMSWavefrontMerit',
    'SpotSizeMerit',
    'ChromaticFocalShiftMerit',
    'MatchIdealThinLensMerit',
    'MatchIdealSystemMerit',
    'MatchTargetOPDMerit',
    'ZernikeCoefficientMerit',
    'LGAberrationMerit',
    'CompositeMerit',
    'CallableMerit',
    'MultiWavelengthMerit',
    'MultiFieldMerit',
    'MinThicknessMerit',
    'MaxThicknessMerit',
    'MinBackFocalLengthMerit',
    'MaxFNumberMerit',
    'ToleranceAwareMerit',
    'EvaluationContext',
    'DesignResult',
    'design_optimize',
    # Phase-space asymptotic propagator + LG aberration tensor
    'CanonicalPolyFit',
    'AberrationTensorResult',
    'fit_canonical_polynomials',
    'aberration_tensor',
    'propagate_modal_asymptotic',
    'solve_envelope_stationary',
    'lg_polynomial',
    'hg_polynomial',
    'evaluate_lg_mode',
    'evaluate_hg_mode',
    'decompose_lg',
    'decompose_hg',
    'lg_seidel_label',
    'gaussian_moment_2d',
    'gaussian_moment_table_2d',
    # Vector diffraction
    'richards_wolf_focus',
    'debye_wolf_psf',
    # Partial coherence
    'koehler_image',
    'extended_source_image',
    'mutual_coherence',
    # Detector / wavefront sensing
    'apply_detector',
    'shack_hartmann',
    # Thin-film coatings
    'coating_reflectance',
    'quarter_wave_ar',
    'broadband_ar_v_coat',
    # Interferometry
    'simulate_interferogram',
    'phase_shift_extract',
    'fringe_spacing',
    # Freeform surfaces
    'surface_sag_xy_polynomial',
    'surface_sag_zernike_freeform',
    'surface_sag_chebyshev',
    'surface_sag_freeform',
    # Ghost analysis
    'enumerate_ghost_paths',
    'ghost_analysis',
    # BSDF surface scatter
    'BSDFModel',
    'LambertianBSDF',
    'GaussianBSDF',
    'HarveyShackBSDF',
    'make_bsdf',
    'sample_scatter_rays',
    # RCWA
    'rcwa_1d',
    'grating_efficiency_vs_wavelength',
    # Multi-configuration / afocal
    'Configuration',
    'multi_config_merit',
    'create_zoom_configs',
    'afocal_angular_magnification',
    'beam_expander_prescription',
    'keplerian_telescope',
    # DOE / I/O
    'create_periodic_phase_mask',
    'create_microlens_array',
    'makedammann2d',
    'load_phase_file',
    'save_phase_file',
    'load_fits_field',
    'save_fits_field',
    # Phase retrieval
    'gerchberg_saxton',
    'error_reduction',
    'hybrid_input_output',
    # Code generation
    'generate_simulation_script',
    'generate_script_from_zmx',
    'generate_script_from_txt',
    # Progress reporting
    'ProgressCallback',
    'ProgressScaler',
    'call_progress',
    # Polarization / Jones calculus
    'JonesField',
    'apply_jones_matrix',
    'apply_polarizer',
    'apply_waveplate',
    'apply_half_wave_plate',
    'apply_quarter_wave_plate',
    'apply_rotator',
    'create_linear_polarized',
    'create_circular_polarized',
    'create_elliptical_polarized',
    'stokes_parameters',
    'degree_of_polarization',
    'polarization_ellipse',
    # Prescriptions
    'export_zemax_lens_data',
    'export_zemax_zmx',
    'load_codev_seq',
    'export_codev_seq',
    'export_quadoa_qos',
    'load_quadoa_qos',
    'QUADOA_SCHEMA_VERSION',
    'make_singlet',
    'make_doublet',
    'thorlabs_lens',
    'load_zmx_prescription',
    'load_zemax_prescription_txt',
    'THORLABS_CATALOG',
    # System
    'propagate_through_system',
    # HDF5 I/O
    'save_field_h5',
    'load_field_h5',
    'save_planes_h5',
    'load_planes_h5',
    'save_jones_field_h5',
    'load_jones_field_h5',
    'append_plane_h5',
    'list_h5_contents',
    'list_planes',
    'load_plane_by_label',
    'load_plane_slice',
    'TempFieldStore',
    'write_sim_metadata',
    'read_sim_metadata',
    # Unified storage (HDF5 / Zarr)
    'set_storage_backend',
    'get_storage_backend',
    'default_extension',
    'append_plane',
    'load_planes',
    'list_planes_store',
    'load_plane_by_label_store',
    'load_plane_slice_store',
    'write_metadata',
    'read_metadata',
    # Memory-aware batching
    'available_memory_bytes',
    'total_memory_bytes',
    'memory_info',
    'bytes_per_element',
    'array_bytes',
    'estimate_op_memory',
    'pick_batch_size',
    'should_split',
    'format_bytes',
    'print_memory_report',
    'get_ram_budget',
    'set_max_ram',
    # Plotting
    'plot_intensity',
    'plot_phase',
    'plot_field',
    'plot_amplitude_phase',
    'plot_cross_section',
    'plot_planes_grid',
    'plot_psf',
    'plot_mtf',
    'plot_stokes',
    'plot_polarization_ellipses',
    'plot_beam_profile',
    'plot_jones_pupil',
    'compute_jones_pupil',
    # Backend info
    'PYFFTW_AVAILABLE',
    'CUPY_AVAILABLE',
    'NUMEXPR_AVAILABLE',
    'available_cpus',
    'set_fft_threads',
    'set_fft_fallback',
    'reset_fft_backend',
    # Precision configuration
    'DEFAULT_COMPLEX_DTYPE',
]
