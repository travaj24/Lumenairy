"""
Core Optical Propagation Module
================================

Provides exact and approximate free-space propagation of coherent optical
fields on discrete 2-D grids.

Methods implemented:
    - Angular Spectrum Method (ASM) -- exact, band-limited
    - Tilted / off-axis ASM        -- ASM with carrier frequency removal
    - Single-FFT Fresnel           -- paraxial, changes output grid spacing
    - Fraunhofer (far-field)       -- single FFT, large-z limit
    - Rayleigh-Sommerfeld          -- convolution with Green's function

Convention
----------
Time dependence:  exp(-i*omega*t)  throughout.
Units:            SI meters for all spatial quantities.

Return-type contract
--------------------
Propagators that **preserve the grid spacing** (ASM, tilted ASM,
Rayleigh-Sommerfeld) return the field as a bare ``ndarray``::

    E_out = angular_spectrum_propagate(E, z, lam, dx)
    E_out = rayleigh_sommerfeld_propagate(E, z, lam, dx)

Propagators that **change the grid spacing** (Fresnel, Fraunhofer)
return a 3-tuple ``(E_out, dx_out, dy_out)`` so callers can resample
or update their pixel pitch::

    E_out, dx_out, dy_out = fresnel_propagate(E, z, lam, dx)
    E_out, dx_out, dy_out = fraunhofer_propagate(E, z, lam, dx)

This split is intentional and stable -- code that treats the bare
return as iterable will fail loudly rather than silently miscompute.

Backends
--------
    NumPy   -- CPU, always available (default)
    CuPy    -- GPU, auto-detected at import time
    pyFFTW  -- multi-threaded CPU FFT, opt-in via USE_PYFFTW flag

Author:  Andrew Traverso
"""

import threading
from collections import OrderedDict

import numpy as np

# ============================================================================
# Optional backend imports
# ============================================================================

# GPU acceleration via CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    # Sentinel so ``xp is cp`` / ``isinstance(..., cp.ndarray)`` checks
    # below don't NameError when cupy isn't installed.
    cp = None
    CUPY_AVAILABLE = False


def _is_cupy_array(x):
    """
    Reliable CuPy array check.

    Historically this module used ``hasattr(x, 'device')`` as a duck-type
    test for a CuPy device array.  That broke in NumPy 2.x: ``ndarray``
    now exposes ``.device`` as part of the Python Array API standard, so
    every NumPy array falsely tests as a CuPy array and gets routed
    through the (unusable without CUDA) CuPy FFT path.  Use ``isinstance``
    against the real CuPy type instead.
    """
    if not CUPY_AVAILABLE:
        return False
    return isinstance(x, cp.ndarray)

# Multi-threaded FFT via pyFFTW
try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(30.0)
    PYFFTW_AVAILABLE = True
except ImportError:
    PYFFTW_AVAILABLE = False

# SciPy FFT (multi-threaded via workers parameter, always available with scipy)
try:
    import scipy.fft as _scipy_fft
    SCIPY_FFT_AVAILABLE = True
except ImportError:
    SCIPY_FFT_AVAILABLE = False

# Affinity-aware CPU count -- respects cgroups / taskset / Python 3.13+
# process_cpu_count so we don't oversubscribe a restricted machine.
from ._backends import available_cpus as _available_cpus


# ============================================================================
# FFT backend configuration
# ============================================================================

# Number of threads for pyFFTW.  Initialised once at import time from
# :func:`lumenairy._backends.available_cpus`; pass a positive
# int to override.  ``0`` falls back to pyFFTW's own default (all cores
# as seen by libfftw3 regardless of affinity).
FFTW_THREADS = _available_cpus()

# Master switch -- route CPU FFTs through pyFFTW when the library is
# installed.  On a 4096x4096 complex128 FFT, pyFFTW + 24 threads is
# measured ~10x faster than NumPy's single-threaded pocketfft and
# ~6x faster than SciPy's ``workers=-1`` pocketfft.  Plan caching is
# enabled above, so repeated calls at the same shape hit a cached plan
# and don't re-plan.
#
# Memory overhead: pyFFTW allocates an aligned workspace per (shape,
# dtype) pair.  On typical optics grids (<= 8192) this is a few hundred
# MB; set ``USE_PYFFTW = False`` if memory is tighter than throughput.
USE_PYFFTW = PYFFTW_AVAILABLE

# Master switch -- set True to route CPU FFTs through scipy.fft.
# Used when pyFFTW is unavailable.  scipy.fft supports multi-threading
# via the ``workers`` parameter and is always available with scipy.
USE_SCIPY_FFT = True

# Number of threads for scipy.fft (-1 = all available cores).  Memory
# usage is unchanged by threading -- pocketfft splits a single FFT
# across cores sharing the same input/output buffers, not by
# allocating per-thread copies.
SCIPY_FFT_WORKERS = -1

# Minimum grid dimension (along axis-0) before pyFFTW is invoked.
# Below this size the single-thread dispatch avoids the planning /
# buffer-alignment overhead that dominates for tiny FFTs.  256 is
# empirically where pyFFTW + plan cache starts to beat NumPy.
FFTW_MIN_SIZE = 256

# Shapes that have failed pyFFTW allocation once in this process.  On
# subsequent calls we skip straight to the scipy fallback for those
# shapes rather than eating the allocation failure repeatedly and
# thrashing the plan cache.  Gets reset when the user explicitly
# flushes via ``reset_fft_backend()``.
_PYFFTW_BAD_SHAPES: set[tuple] = set()

# Single toggle: when True, ``_fft2`` / ``_ifft2`` wrap the pyFFTW call
# in try/except and fall back to scipy.fft (or numpy.fft) on any
# exception.  Allocation failures on large grids (contiguous aligned
# buffer can't fit in free RAM) are the common case, but the
# wrapper also catches plan-cache eviction races and thread-pool
# exhaustion.  Disable via ``set_fft_fallback(False)`` if you want
# pyFFTW errors to propagate (e.g. to detect genuine bugs in the
# backend rather than silently degrading performance).
PYFFTW_FALLBACK_ON_ERROR = True

# Default complex dtype for functions that need to allocate a fresh
# complex array when the caller passes a real input.  Functions that
# operate on an input complex field (e.g. :func:`apply_real_lens`,
# :func:`angular_spectrum_propagate`) infer the target dtype from
# ``E_in.dtype`` and only fall back to this default when the input is
# not complex.  Flip to ``np.complex64`` for ~2x memory + throughput
# at the cost of ~80 dB (not ~140 dB) effective cumulative dynamic
# range; the in-library kernel-phase and phase-screen mitigations
# keep ASM and lens accuracy at single-precision-FFT noise-floor
# levels rather than degrading further with phase magnitude.
DEFAULT_COMPLEX_DTYPE = np.complex128


def set_default_complex_dtype(dtype) -> None:
    """Set the default complex precision used when functions need to
    allocate a fresh complex array (e.g. when callers pass real-valued
    inputs).  Functions that operate on already-complex inputs preserve
    the caller's dtype, so this affects only the "no input dtype"
    paths and any code that deliberately reads ``DEFAULT_COMPLEX_DTYPE``.

    Pass ``np.complex64`` for ~1.6x FFT throughput and ~2x memory
    headroom.  The library carries explicit kernel-phase mod-2pi
    folding everywhere accuracy depends on the absolute kernel
    argument, so single-precision propagation stays at the float32
    noise floor (~80 dB cumulative dynamic range) rather than
    degrading further with phase magnitude.

    Pass ``np.complex128`` to revert (this is the library default).
    """
    global DEFAULT_COMPLEX_DTYPE
    dt = np.dtype(dtype)
    if dt not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError(
            f"set_default_complex_dtype: dtype must be np.complex64 or "
            f"np.complex128, got {dt!r}.")
    DEFAULT_COMPLEX_DTYPE = dt
    # Switching precision invalidates the H cache (different storage
    # dtype keys all the cached entries); the freq-grid / band-limit
    # caches are dtype-independent so leave them alone.
    with _ASM_CACHE_LOCK:
        _H_CACHE.clear()


def get_default_complex_dtype():
    """Return the currently-configured default complex dtype."""
    return DEFAULT_COMPLEX_DTYPE


# ----------------------------------------------------------------------------
# Multi-slot pyFFTW plan cache (3.2.14)
# ----------------------------------------------------------------------------
# Earlier the cache held *one* plan per direction (forward / inverse).
# That worked when a single call site dominated, but optimization
# loops, JonesField (Ex/Ey at one shape, then a 3D batch shape),
# Maslov (mixes the input grid and per-axis 1-D FFTs), and any code
# that propagates at multiple sizes thrashes the single slot --
# every call between two shapes has to reallocate the bound buffer
# and re-plan.  An LRU dict keyed by ``(direction, shape, dtype,
# threads)`` lets several recently-used plans stay resident, with
# bounded memory because old entries fall out the back of the LRU.
#
# Entry layout: OrderedDict[key] = (plan, buf, lock).  ``buf`` is
# bound as both input and output of ``plan`` (in-place); the lock
# serialises concurrent execution on the shared buffer.
_PYFFTW_PLAN_CACHE: 'OrderedDict[tuple, tuple]' = OrderedDict()
_PYFFTW_PLAN_CACHE_SIZE = 8       # # of plans to keep resident
_PYFFTW_PLAN_LOCK = threading.Lock()


def reset_fft_backend():
    """Clear the pyFFTW plan cache and the bad-shape blacklist.

    Useful after a transient memory crunch has passed (e.g. one big
    allocation has since been freed) so subsequent large FFTs can
    retry the fast pyFFTW path instead of staying pinned to the
    scipy fallback.  Also drops every cached plan, freeing all
    aligned workspaces.
    """
    global _PYFFTW_BAD_SHAPES
    _PYFFTW_BAD_SHAPES = set()
    with _PYFFTW_PLAN_LOCK:
        _PYFFTW_PLAN_CACHE.clear()
    # Also flush the H / kxky / bandlimit caches -- they are sized
    # for "what was just being computed" so dropping them on backend
    # reset matches the user's mental model.
    clear_asm_caches()
    if PYFFTW_AVAILABLE:
        try:
            pyfftw.interfaces.cache.disable()
            pyfftw.interfaces.cache.enable()
            pyfftw.interfaces.cache.set_keepalive_time(30.0)
        except Exception:
            pass


def set_fft_plan_cache_size(n):
    """Set the maximum number of pyFFTW plans kept resident.  Default
    is 8.  Pass ``1`` to mimic the legacy single-slot behaviour."""
    global _PYFFTW_PLAN_CACHE_SIZE
    _PYFFTW_PLAN_CACHE_SIZE = max(1, int(n))
    with _PYFFTW_PLAN_LOCK:
        while len(_PYFFTW_PLAN_CACHE) > _PYFFTW_PLAN_CACHE_SIZE:
            _PYFFTW_PLAN_CACHE.popitem(last=False)


def _get_or_make_plan(direction, shape, dtype, threads):
    """Return a cached in-place pyFFTW plan for ``direction`` ('fwd' or
    'inv') at the requested ``shape`` / ``dtype`` / ``threads``.

    Multi-slot LRU: each (direction, shape, dtype, threads) combination
    has its own resident plan.  On hit the entry is moved to the
    front of the LRU; on miss a fresh plan + aligned buffer is built
    and the oldest entry is evicted if the cache is full.

    Returns
    -------
    plan : pyfftw.FFTW
        Bound to ``buf`` as both input and output (in-place transform).
    buf : ndarray
        Aligned workspace.  Callers must ``np.copyto(buf, data)``
        before executing and ``buf.copy()`` the result out before
        the next call on the same plan clobbers ``buf``.
    lock : threading.Lock
        Per-plan lock; serialises concurrent execution on the shared
        buffer.
    """
    shape_t = tuple(int(s) for s in shape)
    dt = np.dtype(dtype)
    key = (str(direction), shape_t, dt.str, int(threads))

    with _PYFFTW_PLAN_LOCK:
        if key in _PYFFTW_PLAN_CACHE:
            entry = _PYFFTW_PLAN_CACHE[key]
            _PYFFTW_PLAN_CACHE.move_to_end(key)
            plan, buf, lock = entry
            if buf.shape == shape_t and buf.dtype == dt:
                return plan, buf, lock
            # Buffer mutated under us (rare; defensive); fall through
            # and rebuild.
            del _PYFFTW_PLAN_CACHE[key]

    # Fresh allocation + plan.  FFTW_ESTIMATE keeps planning cost
    # negligible; speedup vs scipy.fft comes from buffer reuse and
    # the threaded kernel.  pyfftw.FFTW.__call__ on the same buf is
    # not thread-safe -- the per-plan lock guards that.
    buf = pyfftw.empty_aligned(shape_t, dtype=dt)
    direction_flag = 'FFTW_FORWARD' if direction == 'fwd' else 'FFTW_BACKWARD'
    # Choose FFT axes: for 2-D shapes use (0, 1); for higher-D shapes
    # (e.g. batched (B, Ny, Nx) JonesField propagation) FFT only the
    # last two axes.
    if len(shape_t) <= 2:
        axes = (0, 1)
    else:
        axes = (len(shape_t) - 2, len(shape_t) - 1)
    plan = pyfftw.FFTW(
        buf, buf,
        axes=axes,
        direction=direction_flag,
        flags=('FFTW_ESTIMATE',),
        threads=max(1, int(threads)),
    )
    lock = threading.Lock()
    with _PYFFTW_PLAN_LOCK:
        _PYFFTW_PLAN_CACHE[key] = (plan, buf, lock)
        while len(_PYFFTW_PLAN_CACHE) > _PYFFTW_PLAN_CACHE_SIZE:
            _PYFFTW_PLAN_CACHE.popitem(last=False)
    return plan, buf, lock


# ----------------------------------------------------------------------------
# ASM transfer-function and frequency-grid caches (3.2.14)
# ----------------------------------------------------------------------------
# ``angular_spectrum_propagate`` builds three scalar/vector grids on
# every call: kx_sq[Nx], ky_sq[Ny], the band-limit masks bl_x[Nx] /
# bl_y[Ny], and finally the (Ny x Nx) transfer function H = exp(1j *
# kz * z) (band-limited).  The caches below short-circuit those
# rebuilds when the geometry repeats:
#
# * ``_FREQ_GRID_CACHE``   -- (kx_sq, ky_sq) keyed by (Ny, Nx, dy, dx,
#                              cdtype) so any two ASM calls on the
#                              same grid share the spatial-frequency
#                              vectors regardless of z / wavelength.
# * ``_BANDLIMIT_CACHE``   -- (bl_x, bl_y) keyed by the additional
#                              (lambda, |z|) pair (the fx_max, fy_max
#                              cutoffs depend on those).
# * ``_H_CACHE``           -- the full transfer function H, keyed by
#                              (Ny, Nx, dy, dx, lambda, z, bandlimit,
#                              cdtype).  Hits avoid the entire
#                              chunked construction loop -- typically
#                              30-50% of ASM call time on 2k+ grids.
#
# All three are bounded LRU dicts so peak memory stays controlled
# even if user code propagates at many distinct (z, lambda) pairs.

_FREQ_GRID_CACHE: 'OrderedDict[tuple, tuple]' = OrderedDict()
_FREQ_GRID_CACHE_SIZE = 16

_BANDLIMIT_CACHE: 'OrderedDict[tuple, tuple]' = OrderedDict()
_BANDLIMIT_CACHE_SIZE = 16

_H_CACHE: 'OrderedDict[tuple, np.ndarray]' = OrderedDict()
_H_CACHE_SIZE = 8

# 3.2.14.1: Per-entry size cap.  At N=32768 each H is 16 GB
# complex128; without this cap, an 8-entry cache can hold up to
# 128 GB of transfer functions and starve apply_real_lens of the
# few GB it needs for its own sag intermediates.  Above this
# threshold the cache transparently skips storage (lookups still
# work for any small entries already cached).
_H_CACHE_MAX_BYTES_PER_ENTRY = 2 * 1024 * 1024 * 1024   # 2 GB
# Total budget across all entries -- the cache evicts the oldest
# entries until total bytes fits this bound, regardless of count.
# Set generous enough that typical N <= 4k workflows fill up the
# count limit before the bytes limit; at larger N the bytes limit
# kicks in first.
_H_CACHE_MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024       # 8 GB

_ASM_CACHE_LOCK = threading.Lock()


def clear_asm_caches():
    """Drop the H, frequency-grid, and band-limit caches (Tier 1.1
    + Tier 3 of the 3.2.14 perf pass)."""
    with _ASM_CACHE_LOCK:
        _FREQ_GRID_CACHE.clear()
        _BANDLIMIT_CACHE.clear()
        _H_CACHE.clear()


def set_asm_cache_size(h_cache=None, freq_cache=None, bandlimit_cache=None,
                       h_max_bytes_per_entry=None, h_max_total_bytes=None):
    """Tune the per-cache LRU bounds.  Pass ``None`` to leave a
    bound unchanged.

    Parameters
    ----------
    h_cache, freq_cache, bandlimit_cache : int, optional
        Maximum number of entries.
    h_max_bytes_per_entry : int, optional
        Per-entry size cap for the H cache.  An H above this
        threshold is not stored.  Default 2 GB; at N=32768 each H
        is 16 GB so the cache transparently skips storage.
    h_max_total_bytes : int, optional
        Total bytes budget across all H entries.  Oldest entries
        are evicted until total bytes fit.  Default 8 GB.
    """
    global _H_CACHE_SIZE, _FREQ_GRID_CACHE_SIZE, _BANDLIMIT_CACHE_SIZE
    global _H_CACHE_MAX_BYTES_PER_ENTRY, _H_CACHE_MAX_TOTAL_BYTES
    with _ASM_CACHE_LOCK:
        if h_cache is not None:
            _H_CACHE_SIZE = max(1, int(h_cache))
        if h_max_bytes_per_entry is not None:
            _H_CACHE_MAX_BYTES_PER_ENTRY = int(h_max_bytes_per_entry)
        if h_max_total_bytes is not None:
            _H_CACHE_MAX_TOTAL_BYTES = int(h_max_total_bytes)
        # Apply count + bytes bounds together.
        total = sum(int(getattr(v, 'nbytes', 0)) for v in _H_CACHE.values())
        while (len(_H_CACHE) > _H_CACHE_SIZE
               or total > _H_CACHE_MAX_TOTAL_BYTES):
            try:
                _, dropped = _H_CACHE.popitem(last=False)
                total -= int(getattr(dropped, 'nbytes', 0))
            except KeyError:
                break
        if freq_cache is not None:
            _FREQ_GRID_CACHE_SIZE = max(1, int(freq_cache))
            while len(_FREQ_GRID_CACHE) > _FREQ_GRID_CACHE_SIZE:
                _FREQ_GRID_CACHE.popitem(last=False)
        if bandlimit_cache is not None:
            _BANDLIMIT_CACHE_SIZE = max(1, int(bandlimit_cache))
            while len(_BANDLIMIT_CACHE) > _BANDLIMIT_CACHE_SIZE:
                _BANDLIMIT_CACHE.popitem(last=False)


def _get_or_make_freq_grids(Ny, Nx, dy, dx, xp_is_numpy):
    """Cached (kx_sq, ky_sq) 1-D float64 grids for the current shape /
    pixel pitch.  CuPy callers skip the cache (device arrays don't
    survive a host-side dict)."""
    if not xp_is_numpy:
        kx_sq = (2 * np.pi * (cp.arange(Nx) - Nx / 2) / (Nx * dx)) ** 2
        ky_sq = (2 * np.pi * (cp.arange(Ny) - Ny / 2) / (Ny * dy)) ** 2
        # Note: matches the integer arithmetic of the legacy code,
        # which used `(arange(N) - N/2) * (1/(N*dx))` then squared.
        return kx_sq, ky_sq
    key = (int(Ny), int(Nx), float(dy), float(dx))
    with _ASM_CACHE_LOCK:
        if key in _FREQ_GRID_CACHE:
            _FREQ_GRID_CACHE.move_to_end(key)
            return _FREQ_GRID_CACHE[key]
    dfx = 1.0 / (Nx * dx)
    dfy = 1.0 / (Ny * dy)
    fx = (np.arange(Nx) - Nx / 2) * dfx
    fy = (np.arange(Ny) - Ny / 2) * dfy
    kx_sq = (2 * np.pi * fx) ** 2
    ky_sq = (2 * np.pi * fy) ** 2
    with _ASM_CACHE_LOCK:
        _FREQ_GRID_CACHE[key] = (kx_sq, ky_sq)
        while len(_FREQ_GRID_CACHE) > _FREQ_GRID_CACHE_SIZE:
            _FREQ_GRID_CACHE.popitem(last=False)
    return kx_sq, ky_sq


def _get_or_make_bandlimit(Ny, Nx, dy, dx, wavelength, abs_z, xp_is_numpy):
    """Cached 1-D band-limit masks.  Both axes share a single key."""
    if abs_z == 0:
        return None, None
    if not xp_is_numpy:
        Lx = Nx * dx
        Ly = Ny * dy
        fx_max = Lx / (2 * wavelength * abs_z)
        fy_max = Ly / (2 * wavelength * abs_z)
        fx = (cp.arange(Nx) - Nx / 2) / (Nx * dx)
        fy = (cp.arange(Ny) - Ny / 2) / (Ny * dy)
        return cp.abs(fx) < fx_max, cp.abs(fy) < fy_max
    key = (int(Ny), int(Nx), float(dy), float(dx),
           float(wavelength), float(abs_z))
    with _ASM_CACHE_LOCK:
        if key in _BANDLIMIT_CACHE:
            _BANDLIMIT_CACHE.move_to_end(key)
            return _BANDLIMIT_CACHE[key]
    Lx = Nx * dx
    Ly = Ny * dy
    fx_max = Lx / (2 * wavelength * abs_z)
    fy_max = Ly / (2 * wavelength * abs_z)
    fx = (np.arange(Nx) - Nx / 2) / (Nx * dx)
    fy = (np.arange(Ny) - Ny / 2) / (Ny * dy)
    bl_x = np.abs(fx) < fx_max
    bl_y = np.abs(fy) < fy_max
    with _ASM_CACHE_LOCK:
        _BANDLIMIT_CACHE[key] = (bl_x, bl_y)
        while len(_BANDLIMIT_CACHE) > _BANDLIMIT_CACHE_SIZE:
            _BANDLIMIT_CACHE.popitem(last=False)
    return bl_x, bl_y


def _h_cache_lookup(key):
    with _ASM_CACHE_LOCK:
        if key in _H_CACHE:
            _H_CACHE.move_to_end(key)
            return _H_CACHE[key]
    return None


def _h_cache_store(key, H):
    """Store H in the cache, honoring both count and byte bounds.

    Per-entry cap (`_H_CACHE_MAX_BYTES_PER_ENTRY`): if H is bigger
    than this, skip storage entirely.  At N=32768 a complex128 H is
    16 GB; caching it would starve every other allocator on the
    machine.  Lookups for that key will miss, the kernel rebuilds
    H, and the caller still works correctly -- just without the
    speedup (which the cache could never have meaningfully delivered
    on a memory-tight grid anyway).

    Total cap (`_H_CACHE_MAX_TOTAL_BYTES`): after insertion, evict
    oldest entries until total bytes fits.  Keeps the cache useful
    on intermediate grids (e.g. N=4096 at ~256 MB per H, holds 32+
    entries within the budget) without ballooning at large N.
    """
    try:
        h_bytes = int(H.nbytes)
    except Exception:
        h_bytes = 0
    if h_bytes > _H_CACHE_MAX_BYTES_PER_ENTRY:
        return  # too big to cache; lookups will miss + rebuild
    with _ASM_CACHE_LOCK:
        _H_CACHE[key] = H
        # Drop oldest entries until count and total-bytes fit.
        total = sum(int(getattr(v, 'nbytes', 0)) for v in _H_CACHE.values())
        while (len(_H_CACHE) > _H_CACHE_SIZE
               or total > _H_CACHE_MAX_TOTAL_BYTES):
            try:
                _, dropped = _H_CACHE.popitem(last=False)
                total -= int(getattr(dropped, 'nbytes', 0))
            except KeyError:
                break


def set_fft_fallback(enabled: bool) -> None:
    """Enable / disable the automatic pyFFTW -> scipy.fft fallback on
    allocation or runtime errors."""
    global PYFFTW_FALLBACK_ON_ERROR
    PYFFTW_FALLBACK_ON_ERROR = bool(enabled)


def set_fft_threads(n):
    """Override the thread count used by the pyFFTW / scipy.fft path.

    Pass a positive int to pin to that many threads (useful inside a
    process pool where you don't want each worker to spin up its own
    thread farm -- ``set_fft_threads(1)`` gives each worker a
    single-threaded FFT and avoids oversubscription).  Pass ``0`` or
    ``None`` to restore the affinity-aware default from
    :func:`lumenairy._backends.available_cpus`.
    """
    global FFTW_THREADS, SCIPY_FFT_WORKERS
    if n is None or n == 0:
        FFTW_THREADS = _available_cpus()
        SCIPY_FFT_WORKERS = -1
    else:
        FFTW_THREADS = max(1, int(n))
        SCIPY_FFT_WORKERS = max(1, int(n))


# ============================================================================
# FFT helper functions
# ============================================================================

def _scipy_or_numpy_fft2(x):
    """Used by the fallback path (and by small-grid calls)."""
    if USE_SCIPY_FFT and SCIPY_FFT_AVAILABLE:
        return _scipy_fft.fft2(x, workers=SCIPY_FFT_WORKERS)
    return np.fft.fft2(x)


def _scipy_or_numpy_ifft2(x):
    if USE_SCIPY_FFT and SCIPY_FFT_AVAILABLE:
        return _scipy_fft.ifft2(x, workers=SCIPY_FFT_WORKERS)
    return np.fft.ifft2(x)


def _handle_pyfftw_failure(x, op_name, exc):
    """Record a bad shape and emit a one-time warning.

    Called from ``_fft2`` / ``_ifft2`` when a pyFFTW call raises (most
    commonly ``MemoryError`` from an inability to allocate the
    aligned contiguous plan buffer on large grids with tight RAM).
    """
    shape = tuple(x.shape)
    was_new = shape not in _PYFFTW_BAD_SHAPES
    _PYFFTW_BAD_SHAPES.add(shape)
    if was_new:
        import sys, warnings
        # Flush pyFFTW's plan cache so the failed buffers are freed
        # and subsequent SMALLER calls have room.  We keep caching
        # enabled so that unaffected shapes still get plan reuse.
        try:
            pyfftw.interfaces.cache.disable()
            pyfftw.interfaces.cache.enable()
            pyfftw.interfaces.cache.set_keepalive_time(30.0)
        except Exception:
            pass
        warnings.warn(
            f'pyFFTW {op_name} failed on shape {shape}: '
            f'{type(exc).__name__}: {exc}.  Falling back to '
            f'scipy.fft for this shape.  (Likely cause: aligned '
            f'contiguous buffer allocation failed under memory '
            f'pressure.)  Call '
            f'lumenairy.propagation.reset_fft_backend() '
            f'after the large allocation is freed to re-enable '
            f'pyFFTW for future calls at this shape.',
            RuntimeWarning, stacklevel=3)


def _fft2(x):
    """
    2-D FFT dispatcher.

    Priority order:
        1. CuPy (if input is a CuPy array)
        2. pyFFTW via the single-slot cached plan (if USE_PYFFTW, array
           large enough, shape not in the bad-shape blacklist).  Hits
           :func:`_get_or_make_plan` which reuses an existing plan
           when ``(shape, dtype, threads)`` matches exactly, otherwise
           reallocates.
        3. SciPy FFT (if USE_SCIPY_FFT)
        4. NumPy FFT (fallback)

    pyFFTW calls are wrapped in try/except so a failed aligned-buffer
    allocation (common on very large grids under memory pressure)
    falls back to scipy.fft rather than propagating the error.  See
    :data:`PYFFTW_FALLBACK_ON_ERROR`.
    """
    if _is_cupy_array(x):
        return cp.fft.fft2(x)
    shape = tuple(x.shape)
    if (USE_PYFFTW and PYFFTW_AVAILABLE
            and x.shape[0] >= FFTW_MIN_SIZE
            and shape not in _PYFFTW_BAD_SHAPES):
        threads = FFTW_THREADS if FFTW_THREADS > 0 else 1
        try:
            plan, buf, lock = _get_or_make_plan('fwd', shape, x.dtype, threads)
            # Hold the lock across copy-in, execute, and copy-out so
            # a concurrent ``_fft2`` / ``_ifft2`` caller can't race
            # on the shared bound buffer.
            with lock:
                np.copyto(buf, x, casting='no')
                plan()
                # Copy out so the next _fft2/_ifft2 call (which will
                # overwrite ``buf``) can't corrupt the caller's result.
                return buf.copy()
        except Exception as e:
            if not PYFFTW_FALLBACK_ON_ERROR:
                raise
            _handle_pyfftw_failure(x, 'fft2', e)
            # fall through to scipy/numpy
    return _scipy_or_numpy_fft2(x)


def _ifft2(x):
    """
    2-D inverse FFT dispatcher.

    Same priority as :func:`_fft2`.  Uses a separate single-slot
    cached plan for the inverse direction; ``pyfftw.FFTW`` with
    ``direction='FFTW_BACKWARD'`` normalises by ``N`` by default,
    matching ``numpy.fft.ifft2`` semantics.
    """
    if _is_cupy_array(x):
        return cp.fft.ifft2(x)
    shape = tuple(x.shape)
    if (USE_PYFFTW and PYFFTW_AVAILABLE
            and x.shape[0] >= FFTW_MIN_SIZE
            and shape not in _PYFFTW_BAD_SHAPES):
        threads = FFTW_THREADS if FFTW_THREADS > 0 else 1
        try:
            plan, buf, lock = _get_or_make_plan('inv', shape, x.dtype, threads)
            with lock:
                np.copyto(buf, x, casting='no')
                plan()
                return buf.copy()
        except Exception as e:
            if not PYFFTW_FALLBACK_ON_ERROR:
                raise
            _handle_pyfftw_failure(x, 'ifft2', e)
    return _scipy_or_numpy_ifft2(x)


# ============================================================================
# Angular Spectrum Method (ASM) -- exact, band-limited
# ============================================================================

def angular_spectrum_propagate(
    E_in,
    z,
    wavelength,
    dx,
    dy=None,
    bandlimit=True,
    return_transfer_function=False,
    use_gpu=False,
    verbose=False
):
    """
    Propagate an optical field using the Angular Spectrum Method (ASM).

    This function propagates a 2-D complex electric field through free space
    using the exact transfer function (no paraxial approximation).

    Parameters
    ----------
    E_in : ndarray (complex)
        Input electric field, shape (Ny, Nx).  Can be a NumPy or CuPy array.

    z : float
        Propagation distance in meters.
        Positive z = forward propagation (away from source).
        Negative z = backward propagation (toward source).

    wavelength : float
        Optical wavelength in meters (e.g. 1.31e-6 for 1310 nm).

    dx : float
        Grid spacing in x-direction in meters (e.g. 1e-6 for 1 um).

    dy : float, optional
        Grid spacing in y-direction in meters.  If None, assumes dy = dx.

    bandlimit : bool, default True
        If True, applies band-limiting to suppress Fresnel aliasing.
        The band-limit cutoff per axis is:  f_max = L / (2 * lambda * |z|).
        Recommended for large propagation distances.

    return_transfer_function : bool, default False
        If True, also returns the transfer function H.

    use_gpu : bool, default False
        If True and CuPy is available, performs computation on GPU.
        If *E_in* is already a CuPy array, GPU is used automatically.

    verbose : bool, default False
        If True, prints diagnostic information.

    Returns
    -------
    E_out : ndarray (complex)
        Propagated electric field, same shape and array type as *E_in*.

    H : ndarray (complex), optional
        Transfer function (only returned when *return_transfer_function=True*).

    Notes
    -----
    Sampling requirements for accurate results:

    1. ``dx < lambda / 2`` -- Nyquist for propagating waves.
    2. ``L > 2 * lambda * z / d_min`` -- avoids Fresnel aliasing, where
       L = N * dx is the grid extent and d_min is the smallest feature size
       to be resolved.

    Memory: approximately 3x the size of the input array (E_in, E_fft, H).

    Examples
    --------
    >>> import numpy as np
    >>> from lumenairy.propagation import angular_spectrum_propagate
    >>>
    >>> N = 512
    >>> dx = 1e-6                    # 1 um grid spacing
    >>> wavelength = 1.31e-6         # 1310 nm
    >>> x = (np.arange(N) - N/2) * dx
    >>> X, Y = np.meshgrid(x, x)
    >>> sigma = 10e-6                # 10 um beam waist
    >>> E_in = np.exp(-(X**2 + Y**2) / (2 * sigma**2)).astype(complex)
    >>>
    >>> E_out = angular_spectrum_propagate(E_in, z=1e-3,
    ...                                    wavelength=wavelength, dx=dx)
    >>> print(f"Input power:  {np.sum(np.abs(E_in)**2):.4f}")
    >>> print(f"Output power: {np.sum(np.abs(E_out)**2):.4f}")

    References
    ----------
    [1] Goodman, J.W. "Introduction to Fourier Optics" (3rd ed.), Ch. 3-4.
    [2] Matsushima, K. and Shimobaba, T. (2009). "Band-limited angular
        spectrum method for numerical simulation of free-space propagation
        in far and near fields." Opt. Express 17(22): 19662-19673.
    """

    # -- array library selection (NumPy vs. CuPy) ---------------------------
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        xp = cp
        if not _is_cupy_array(E_in):
            E_in = cp.asarray(E_in)
    else:
        xp = np
        if _is_cupy_array(E_in):
            E_in = E_in.get()  # CuPy -> NumPy when GPU not requested

    Ny, Nx = E_in.shape

    if dy is None:
        dy = dx

    # -- wave parameters -----------------------------------------------------
    k = 2 * np.pi / wavelength

    # Target complex dtype for the transfer function and the output.
    # Inferred from E_in so the caller controls precision by the dtype of
    # the field they pass in.  Non-complex input (e.g. float arrays used
    # in examples) falls back to DEFAULT_COMPLEX_DTYPE.
    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(DEFAULT_COMPLEX_DTYPE)
    target_fdtype = np.float32 if target_cdtype == np.complex64 else np.float64

    # ── 3.2.14 H cache ───────────────────────────────────────────────
    # Geometry signature.  Hits return the previously-built H without
    # re-running the chunked kernel construction (~30-50% of total
    # ASM time on 2k+ grids).  CuPy device arrays are kept out of the
    # cache (host-side dict can't safely retain device pointers across
    # context lifetimes).
    h_key = None
    H = None
    if xp is np:
        h_key = (int(Ny), int(Nx), float(dy), float(dx),
                 float(wavelength), float(z), bool(bandlimit),
                 np.dtype(target_cdtype).str)
        H = _h_cache_lookup(h_key)

    if H is None:
        # Spatial-frequency squared vectors (cached on numpy path).
        kx_sq, ky_sq = _get_or_make_freq_grids(Ny, Nx, dy, dx, xp is np)
        if bandlimit and z != 0:
            bl_x, bl_y = _get_or_make_bandlimit(
                Ny, Nx, dy, dx, wavelength, abs(z), xp is np)
        else:
            bl_x = bl_y = None

        # Chunked H construction, sized to fit a small slice of RAM.
        from .memory import get_ram_budget
        ram = get_ram_budget()
        row_cost = 3 * Nx * 16   # bytes per row of workspace (complex128)
        if row_cost > 0:
            max_chunk = max(1, int(ram * 0.1 / row_cost))
        else:
            max_chunk = Ny
        chunk = min(Ny, max_chunk)

        H = xp.empty((Ny, Nx), dtype=target_cdtype)
        kept_count = 0
        for j0 in range(0, Ny, chunk):
            j1 = min(Ny, j0 + chunk)
            # kz_sq is float64 regardless of target dtype to keep the
            # huge kernel argument (kz * z up to ~1e6 rad) accurate.
            kz_sq_c = k**2 - kx_sq[None, :] - ky_sq[j0:j1, None]
            prop = kz_sq_c > 0
            kz_c = xp.where(prop, xp.sqrt(xp.maximum(kz_sq_c, 0)), 0)
            if target_cdtype == np.complex128:
                H_c = xp.where(prop, xp.exp(1j * kz_c * z), 0)
            else:
                # complex64 path: fold phase mod 2*pi in float64
                # BEFORE casting to float32 so the float32 precision
                # floor doesn't inject speckle-like noise.
                phase = xp.mod(kz_c * z, 2.0 * np.pi)
                c = xp.cos(phase).astype(target_fdtype)
                s = xp.sin(phase).astype(target_fdtype)
                H_c = xp.empty((j1 - j0, Nx), dtype=target_cdtype)
                H_c.real[:] = xp.where(prop, c, target_fdtype(0))
                H_c.imag[:] = xp.where(prop, s, target_fdtype(0))
            if bl_x is not None:
                bl_mask = bl_x[None, :] & bl_y[j0:j1, None]
                H_c *= bl_mask
                if verbose:
                    kept_count += int(xp.sum(bl_mask))
            H[j0:j1, :] = H_c

        if verbose and bl_x is not None:
            kept_frac = kept_count / (Nx * Ny)
            print(f"  Band-limiting: keeping {kept_frac*100:.1f}% of spectrum")
        if verbose:
            print(f"  ASM propagation: z = {z*1e3:.3f} mm  "
                  f"(H cache miss, built in {chunk}-row chunks)")
            print(f"  Grid: {Ny}x{Nx}, dx={dx*1e6:.3f} um, dy={dy*1e6:.3f} um")
            print(f"  Wavelength: {wavelength*1e9:.1f} nm")
        # Store under the numpy key only.  The cached H is read-only
        # in normal use; we don't deep-copy on lookup, so callers must
        # not mutate it in place.
        if h_key is not None:
            _h_cache_store(h_key, H)
    elif verbose:
        print(f"  ASM propagation: z = {z*1e3:.3f} mm  (H cache HIT)")

    # -- propagate: E_out = IFFT{ FFT{E_in} * H } ---------------------------
    if xp is np:
        E_fft = np.fft.fftshift(_fft2(np.fft.ifftshift(E_in)))
        E_out = np.fft.fftshift(_ifft2(np.fft.ifftshift(E_fft * H)))
    else:
        E_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(E_in)))
        E_out = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(E_fft * H)))

    if return_transfer_function:
        return E_out, H
    else:
        return E_out


def angular_spectrum_propagate_batch(E_stack, z, wavelength, dx,
                                      dy=None, bandlimit=True,
                                      use_gpu=False):
    """ASM propagation of a stack of fields ``(B, Ny, Nx)`` in one
    fused FFT pair (3.2.14).

    All ``B`` fields share the same grid + wavelength + propagation
    distance, so the transfer function ``H`` is built once (reusing
    the H cache) and broadcast across the batch.  Two batched FFTs
    (forward + inverse, axes ``(-2, -1)``) replace ``2*B`` separate
    2-D FFTs, which on JonesField (Ex, Ey) is ~30-60% wall-clock
    faster than calling :func:`angular_spectrum_propagate` per
    component.

    Parameters
    ----------
    E_stack : ndarray, complex, shape (B, Ny, Nx)
        Input field stack.  ``B`` must be at least 1.
    z, wavelength, dx, dy, bandlimit, use_gpu
        Same semantics as :func:`angular_spectrum_propagate`.

    Returns
    -------
    E_out : ndarray, complex, shape (B, Ny, Nx)
        Propagated stack, same dtype + array library as input.
    """
    if E_stack.ndim != 3:
        raise ValueError(
            f"angular_spectrum_propagate_batch: input must be 3-D "
            f"(B, Ny, Nx), got shape {E_stack.shape}.")

    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_stack)):
        xp = cp
        if not _is_cupy_array(E_stack):
            E_stack = cp.asarray(E_stack)
    else:
        xp = np
        if _is_cupy_array(E_stack):
            E_stack = E_stack.get()

    B, Ny, Nx = E_stack.shape
    if dy is None:
        dy = dx

    if xp.iscomplexobj(E_stack):
        target_cdtype = E_stack.dtype
    else:
        target_cdtype = np.dtype(DEFAULT_COMPLEX_DTYPE)
        E_stack = E_stack.astype(target_cdtype)

    # Reuse the H cache from the scalar propagator: build H by
    # delegating to the scalar function on a tiny ``Ny x Nx`` field
    # of the right dtype with ``return_transfer_function=True``.  H
    # is read-only after construction so it is safe to reuse across
    # the batch.
    _proxy = xp.empty((Ny, Nx), dtype=target_cdtype)
    _, H = angular_spectrum_propagate(
        _proxy, z, wavelength, dx, dy=dy, bandlimit=bandlimit,
        return_transfer_function=True, use_gpu=(xp is not np),
    )

    # Single batched FFT pair across the last two axes.  pyFFTW's
    # multi-slot plan cache (also new in 3.2.14) keys on the full
    # shape including the batch dimension, so a 3-D plan is built on
    # the first call and reused thereafter.  The numpy / scipy
    # fallback paths handle 3-D input natively via ``fft2`` over the
    # last two axes.
    if xp is np:
        # Use scipy.fft for ND batched (workers parameter), pyFFTW
        # plan cache picks up the (B, Ny, Nx) shape automatically via
        # ``_fft2`` if the array is large enough.
        E_fft = xp.fft.fftshift(
            _fft2_nd(xp.fft.ifftshift(E_stack, axes=(-2, -1))),
            axes=(-2, -1))
        E_out = xp.fft.fftshift(
            _ifft2_nd(xp.fft.ifftshift(E_fft * H[None, :, :],
                                        axes=(-2, -1))),
            axes=(-2, -1))
    else:
        E_fft = xp.fft.fftshift(
            xp.fft.fft2(xp.fft.ifftshift(E_stack, axes=(-2, -1)),
                        axes=(-2, -1)),
            axes=(-2, -1))
        E_out = xp.fft.fftshift(
            xp.fft.ifft2(xp.fft.ifftshift(E_fft * H[None, :, :],
                                            axes=(-2, -1)),
                          axes=(-2, -1)),
            axes=(-2, -1))
    return E_out


def _fft2_nd(x):
    """N-D forward 2-D FFT over the last two axes.  Uses pyFFTW for
    contiguous (B, Ny, Nx) shapes when large enough; falls back to
    scipy.fft / numpy for everything else."""
    if _is_cupy_array(x):
        return cp.fft.fft2(x, axes=(-2, -1))
    shape = tuple(x.shape)
    if (USE_PYFFTW and PYFFTW_AVAILABLE and len(shape) >= 2
            and shape[-2] >= FFTW_MIN_SIZE
            and shape not in _PYFFTW_BAD_SHAPES):
        threads = FFTW_THREADS if FFTW_THREADS > 0 else 1
        try:
            plan, buf, lock = _get_or_make_plan(
                'fwd', shape, x.dtype, threads)
            with lock:
                np.copyto(buf, x, casting='no')
                plan()
                return buf.copy()
        except Exception as e:
            if not PYFFTW_FALLBACK_ON_ERROR:
                raise
            _handle_pyfftw_failure(x, 'fft2_nd', e)
    if USE_SCIPY_FFT and SCIPY_FFT_AVAILABLE:
        return _scipy_fft.fft2(x, axes=(-2, -1), workers=SCIPY_FFT_WORKERS)
    return np.fft.fft2(x, axes=(-2, -1))


def _ifft2_nd(x):
    if _is_cupy_array(x):
        return cp.fft.ifft2(x, axes=(-2, -1))
    shape = tuple(x.shape)
    if (USE_PYFFTW and PYFFTW_AVAILABLE and len(shape) >= 2
            and shape[-2] >= FFTW_MIN_SIZE
            and shape not in _PYFFTW_BAD_SHAPES):
        threads = FFTW_THREADS if FFTW_THREADS > 0 else 1
        try:
            plan, buf, lock = _get_or_make_plan(
                'inv', shape, x.dtype, threads)
            with lock:
                np.copyto(buf, x, casting='no')
                plan()
                return buf.copy()
        except Exception as e:
            if not PYFFTW_FALLBACK_ON_ERROR:
                raise
            _handle_pyfftw_failure(x, 'ifft2_nd', e)
    if USE_SCIPY_FFT and SCIPY_FFT_AVAILABLE:
        return _scipy_fft.ifft2(x, axes=(-2, -1), workers=SCIPY_FFT_WORKERS)
    return np.fft.ifft2(x, axes=(-2, -1))


# ============================================================================
# Tilted / off-axis ASM propagation
# ============================================================================

def angular_spectrum_propagate_tilted(E_in, z, wavelength, dx, dy=None,
                                      tilt_x=0.0, tilt_y=0.0, bandlimit=True):
    """
    ASM propagation with a carrier tilt (off-axis propagation).

    Propagates the field while accounting for a mean propagation direction
    that is tilted relative to the optical axis.  This is useful for:

    - Beams arriving at an angle
    - Propagation after a prism or wedge
    - Off-axis portions of a wide-field system

    The tilt is handled by shifting the frequency-domain transfer function,
    which is equivalent to propagating the field in a tilted reference frame.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input electric field.

    z : float
        Propagation distance [m] along the tilted axis.

    wavelength : float
        Optical wavelength [m].

    dx : float
        Grid spacing in x [m].

    dy : float, optional
        Grid spacing in y [m].  Defaults to dx.

    tilt_x, tilt_y : float, default 0.0
        Tilt angles [radians] of the propagation direction relative to the
        z-axis.  The beam propagates at angle (tilt_x, tilt_y) from the
        optical axis.

    bandlimit : bool, default True
        Apply band-limiting to avoid aliasing.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
        Propagated electric field.

    Notes
    -----
    The method removes the carrier frequency (tilt) before propagation,
    then restores it afterwards.  This keeps the field well-centred on
    the grid even for large tilt angles, avoiding grid walk-off.

    The carrier spatial frequencies are::

        fx0 = sin(tilt_x) / wavelength
        fy0 = sin(tilt_y) / wavelength

    The field is demodulated as::

        E_demod = E_in * exp(-i * 2*pi * (fx0*X + fy0*Y))

    propagated with a shifted transfer function, then remodulated::

        E_out = E_prop * exp(+i * 2*pi * (fx0*X + fy0*Y))

    For ``tilt_x = tilt_y = 0`` this reduces to standard ASM propagation.
    """
    if dy is None:
        dy = dx

    Ny, Nx = E_in.shape

    # -- carrier spatial frequencies from tilt angles ------------------------
    fx0 = np.sin(tilt_x) / wavelength
    fy0 = np.sin(tilt_y) / wavelength

    # Shortcut: no tilt -> fall back to standard ASM
    if abs(fx0) < 1e-15 and abs(fy0) < 1e-15:
        return angular_spectrum_propagate(E_in, z, wavelength, dx, dy,
                                          bandlimit=bandlimit)

    # -- spatial coordinate grids --------------------------------------------
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)

    # -- demodulate: remove carrier tilt -------------------------------------
    carrier = np.exp(-1j * 2 * np.pi * (fx0 * X + fy0 * Y))
    E_demod = E_in * carrier

    # -- shifted transfer function -------------------------------------------
    # kz is evaluated at (fx + fx0, fy + fy0) so the baseband field
    # propagates with the correct kz for each plane-wave component.
    k = 2 * np.pi / wavelength
    dfx = 1.0 / (Nx * dx)
    dfy = 1.0 / (Ny * dy)
    fx = (np.arange(Nx) - Nx / 2) * dfx
    fy = (np.arange(Ny) - Ny / 2) * dfy
    FX, FY = np.meshgrid(fx, fy)

    FX_shifted = FX + fx0
    FY_shifted = FY + fy0
    kx = 2 * np.pi * FX_shifted
    ky = 2 * np.pi * FY_shifted

    kz_sq = k**2 - kx**2 - ky**2
    kz = np.where(kz_sq > 0, np.sqrt(np.maximum(kz_sq, 0)), 0)
    H = np.exp(1j * kz * z)
    H = np.where(kz_sq > 0, H, 0)

    # -- band-limiting on the BASEBAND (demodulated) spectrum ---------------
    # The Matsushima-Shimobaba criterion bounds the frequency content of the
    # field AS SAMPLED on the grid; after demodulation the tilted beam's
    # energy lives around FX = 0 (baseband), so the mask must be centred
    # there.  Applying it to ``FX_shifted`` (carrier-plus-baseband) kills
    # the baseband DC mode for any meaningful tilt and zeros the whole
    # propagated field.
    if bandlimit and z != 0:
        Lx = Nx * dx
        Ly = Ny * dy
        fx_max = Lx / (2 * wavelength * abs(z))
        fy_max = Ly / (2 * wavelength * abs(z))
        H = np.where((np.abs(FX) < fx_max) &
                      (np.abs(FY) < fy_max), H, 0)

    # -- propagate baseband with shifted transfer function -------------------
    E_fft = np.fft.fftshift(_fft2(np.fft.ifftshift(E_demod)))
    E_prop = np.fft.fftshift(_ifft2(np.fft.ifftshift(E_fft * H)))

    # -- remodulate: restore carrier tilt ------------------------------------
    E_out = E_prop * np.conj(carrier)

    return E_out


# ============================================================================
# Single-FFT Fresnel propagation
# ============================================================================

def fresnel_propagate(E_in, z, wavelength, dx, dy=None):
    """
    Propagate a field using the single-FFT Fresnel method.

    This is the Fresnel (paraxial) approximation to diffraction.  It uses a
    single FFT and is faster than ASM for long propagation distances, but
    **changes the grid spacing** in the output plane.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input field.

    z : float
        Propagation distance [m].

    wavelength : float
        Wavelength [m].

    dx : float
        Input grid spacing in x [m].

    dy : float, optional
        Input grid spacing in y [m].  Defaults to dx.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
        Output field.

    dx_out : float
        Output grid spacing in x [m].

    dy_out : float
        Output grid spacing in y [m].

    Notes
    -----
    The output grid spacing is::

        dx_out = wavelength * |z| / (Nx * dx)

    This method is valid when the Fresnel number is moderate::

        N_F = a^2 / (lambda * z) ~ 1

    where *a* is the beam / aperture radius.

    For very short distances (large Fresnel number), use ASM instead.
    For very long distances (small Fresnel number), this becomes equivalent
    to the Fraunhofer approximation.
    """
    if dy is None:
        dy = dx

    Ny, Nx = E_in.shape
    k = 2 * np.pi / wavelength

    # -- input coordinates ---------------------------------------------------
    x1 = (np.arange(Nx) - Nx / 2) * dx
    y1 = (np.arange(Ny) - Ny / 2) * dy
    X1, Y1 = np.meshgrid(x1, y1)

    # -- output grid spacing (changes with z) --------------------------------
    dx_out = wavelength * abs(z) / (Nx * dx)
    dy_out = wavelength * abs(z) / (Ny * dy)

    # -- output coordinates --------------------------------------------------
    x2 = (np.arange(Nx) - Nx / 2) * dx_out
    y2 = (np.arange(Ny) - Ny / 2) * dy_out
    X2, Y2 = np.meshgrid(x2, y2)

    # -- quadratic phase in input plane --------------------------------------
    E_mod = E_in * np.exp(1j * k / (2 * z) * (X1**2 + Y1**2))

    # -- FFT -----------------------------------------------------------------
    E_fft = np.fft.fftshift(_fft2(np.fft.ifftshift(E_mod)))

    # -- quadratic phase in output plane + prefactor -------------------------
    prefactor = (np.exp(1j * k * z) / (1j * wavelength * z)
                 * np.exp(1j * k / (2 * z) * (X2**2 + Y2**2))
                 * dx * dy)

    E_out = prefactor * E_fft

    return E_out, dx_out, dy_out


# =============================================================================
# FRAUNHOFER (FAR-FIELD) PROPAGATION
# =============================================================================

def resample_field(E_in, dx_in, dx_out, N_out=None, order=3):
    """
    Resample a complex optical field from one grid spacing to another.

    This is the bridge function for switching between propagation methods
    that use different grid spacings (e.g. Fresnel output -> ASM input,
    or vice versa).  Both amplitude and phase are interpolated using
    scipy's map_coordinates.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input field on a grid with spacing ``dx_in``.
    dx_in : float
        Input grid spacing [m].
    dx_out : float
        Desired output grid spacing [m].
    N_out : int or None
        Output grid size.  If ``None``, chosen so the output covers the
        same physical extent as the input: ``N_out = round(N_in * dx_in / dx_out)``.
    order : int, default 3
        Interpolation order (1=linear, 3=cubic, 5=quintic).

    Returns
    -------
    E_out : ndarray (complex, N_out x N_out)
        Resampled field on the new grid.
    dx_out : float
        The output grid spacing (same as the input parameter, returned
        for convenience so callers can chain: ``E, dx = resample_field(...)``).

    Notes
    -----
    - Interpolation introduces a small error proportional to (dx_out/feature_size)^order.
      For order=3 (cubic), this is < 0.1% when features are sampled at >= 4 pixels.
    - For downsampling (dx_out > dx_in), consider anti-alias filtering first.
    - The field is assumed to be on a centered grid: x = (arange(N) - N/2) * dx.
    """
    from scipy.ndimage import map_coordinates

    Ny_in, Nx_in = E_in.shape
    if N_out is None:
        Nx_out = int(round(Nx_in * dx_in / dx_out))
        Ny_out = int(round(Ny_in * dx_in / dx_out))
    else:
        Nx_out = Ny_out = int(N_out)

    # Output coordinates in input-pixel units.
    # Input grid:  x_in[i]  = (i - Nx_in/2)  * dx_in
    # Output grid: x_out[j] = (j - Nx_out/2) * dx_out
    # Map: i = x_out / dx_in + Nx_in/2 = (j - Nx_out/2) * dx_out/dx_in + Nx_in/2
    scale = dx_out / dx_in
    jx = np.arange(Nx_out)
    jy = np.arange(Ny_out)
    ix = (jx - Nx_out / 2) * scale + Nx_in / 2
    iy = (jy - Ny_out / 2) * scale + Ny_in / 2
    IX, IY = np.meshgrid(ix, iy)
    coords = np.array([IY.ravel(), IX.ravel()])

    # Interpolate real and imaginary parts separately
    real_out = map_coordinates(E_in.real, coords, order=order, mode='constant', cval=0.0)
    imag_out = map_coordinates(E_in.imag, coords, order=order, mode='constant', cval=0.0)
    E_out = (real_out + 1j * imag_out).reshape(Ny_out, Nx_out)

    return E_out, dx_out


def fraunhofer_propagate(E_in, z, wavelength, dx, dy=None):
    """
    Propagate a field to the Fraunhofer (far-field) diffraction pattern.

    This is the far-field limit of the Fresnel propagator, valid when the
    Fresnel number is small:

        N_F = a^2 / (lambda * z) << 1

    where ``a`` is the characteristic aperture/beam radius. In practice this
    means ``z`` must be large compared to ``a^2 / lambda``. For smaller
    distances, use :func:`fresnel_propagate` or :func:`angular_spectrum_propagate`.

    Mathematically, the Fraunhofer integral reduces to a single Fourier
    transform of the input field with a quadratic phase and scaling prefactor::

        E(x2, y2) = [exp(i*k*z) / (i*lambda*z)]
                    * exp(i*k/(2z) * (x2^2 + y2^2))
                    * FFT{E(x1, y1)} * dx*dy

    Parameters
    ----------
    E_in : ndarray (complex, N×N)
        Input field.
    z : float
        Propagation distance [m].
    wavelength : float
        Free-space wavelength [m].
    dx : float
        Input grid spacing in x [m].
    dy : float, optional
        Input grid spacing in y [m]. Defaults to dx.

    Returns
    -------
    E_out : ndarray (complex, N×N)
        Field in the far-field plane.
    dx_out : float
        Output grid spacing in x [m] = wavelength * |z| / (N * dx).
    dy_out : float
        Output grid spacing in y [m] = wavelength * |z| / (N * dy).

    Notes
    -----
    The output grid spacing is the same as :func:`fresnel_propagate`:

        dx_out = wavelength * |z| / (N * dx)

    The difference from Fresnel is that Fraunhofer drops the input-plane
    quadratic phase (assumed to be negligible at large z), so there is only
    one FFT and one scalar multiplication. It is slightly faster and more
    numerically stable than Fresnel at large distances.

    For focal-plane computation of a converging beam (e.g. after a lens),
    Fraunhofer is the standard approach: place the input field at the lens,
    set z = focal length, and the output is the focal-plane field.
    """
    if dy is None:
        dy = dx

    Ny, Nx = E_in.shape
    k = 2 * np.pi / wavelength

    # Output grid spacing
    dx_out = wavelength * abs(z) / (Nx * dx)
    dy_out = wavelength * abs(z) / (Ny * dy)

    # Output coordinates
    x2 = (np.arange(Nx) - Nx / 2) * dx_out
    y2 = (np.arange(Ny) - Ny / 2) * dy_out
    X2, Y2 = np.meshgrid(x2, y2)

    # Single FFT of the input field
    E_fft = np.fft.fftshift(_fft2(np.fft.ifftshift(E_in)))

    # Output quadratic phase + prefactor
    prefactor = (np.exp(1j * k * z) / (1j * wavelength * z)
                 * np.exp(1j * k / (2 * z) * (X2**2 + Y2**2))
                 * dx * dy)

    E_out = prefactor * E_fft

    return E_out, dx_out, dy_out


# ============================================================================
# Rayleigh-Sommerfeld propagation
# ============================================================================

def rayleigh_sommerfeld_propagate(
    E_in,
    z,
    wavelength,
    dx,
    dy=None,
    use_gpu=False,
    verbose=False
):
    """
    Propagate an optical field using the Rayleigh-Sommerfeld convolution.

    This method computes the first Rayleigh-Sommerfeld solution by
    convolving the input field with the free-space impulse response
    (Green's function).  Unlike the ASM transfer-function approach,
    the RS convolution constructs the propagation kernel in the
    *spatial* domain and performs the convolution via FFT, which
    naturally captures near-field diffraction effects without the
    band-limiting approximation used in ASM.

    The impulse response is:

        h(x, y, z) = (1 / 2pi) * (z / r^2) * (ik - 1/r) * exp(ikr)

    where ``r = sqrt(x^2 + y^2 + z^2)`` and ``k = 2*pi / lambda``.

    The convolution is computed as::

        E_out = IFFT{ FFT{E_in} * FFT{h} }

    using zero-padded arrays (2N x 2N) to avoid circular convolution
    artifacts.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input electric field.
    z : float
        Propagation distance [m].  Positive = forward.
    wavelength : float
        Free-space wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to dx.
    use_gpu : bool, default False
        Use CuPy GPU acceleration if available.
    verbose : bool, default False
        Print diagnostic info.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
        Propagated field (same shape as input).

    Notes
    -----
    **When to use RS instead of ASM:**

    - Near-field propagation (z ~ a few wavelengths) where ASM's
      band-limiting can suppress valid high-frequency content.
    - Validation / cross-check against ASM results.
    - Situations where the exact Green's function is preferred over
      the plane-wave decomposition.

    **Computational cost:** ~4x ASM due to zero-padding (2N FFTs
    instead of N FFTs) and the spatial-domain kernel construction.

    **Memory:** ~6x input array size (padded E, padded h, FFTs).

    At large distances (z >> a^2 / lambda), RS and ASM give identical
    results.  For intermediate distances they agree to machine precision
    when ASM uses no band-limiting (``bandlimit=False``).

    References
    ----------
    [1] Goodman, J.W. "Introduction to Fourier Optics" (3rd ed.),
        Section 3.5: Rayleigh-Sommerfeld Diffraction Theory.
    [2] Shen, F. and Wang, A. (2006). "Fast-Fourier-transform based
        numerical integration method for the Rayleigh-Sommerfeld
        diffraction formula." Appl. Opt. 45(6): 1102-1110.

    Examples
    --------
    >>> import numpy as np
    >>> from lumenairy.propagation import rayleigh_sommerfeld_propagate
    >>>
    >>> N = 512; dx = 1e-6; wv = 0.633e-6
    >>> x = (np.arange(N) - N/2) * dx
    >>> X, Y = np.meshgrid(x, x)
    >>> E_in = (np.sqrt(X**2 + Y**2) < 50e-6).astype(complex)  # circular aperture
    >>>
    >>> E_out = rayleigh_sommerfeld_propagate(E_in, z=1e-3, wavelength=wv, dx=dx)
    """

    # -- array library selection -----------------------------------------------
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        xp = cp
        if not _is_cupy_array(E_in):
            E_in = cp.asarray(E_in)
    else:
        xp = np
        if _is_cupy_array(E_in):
            E_in = E_in.get()

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx

    k = 2 * np.pi / wavelength

    # -- zero-pad to avoid circular convolution --------------------------------
    # Use the transfer-function approach instead of direct convolution:
    # compute h in spatial domain, FFT it, multiply with FFT of padded E.
    # The input field is centred in the padded array.
    Ny2 = 2 * Ny
    Nx2 = 2 * Nx

    E_padded = xp.zeros((Ny2, Nx2), dtype=xp.complex128)
    # Centre the input field in the padded array
    y0 = Ny // 2
    x0 = Nx // 2
    E_padded[y0:y0 + Ny, x0:x0 + Nx] = E_in

    # -- build the RS impulse response h(x, y, z) on the padded grid -----------
    # Centred at (0, 0) for proper convolution alignment
    x = (xp.arange(Nx2) - Nx2 / 2) * dx
    y = (xp.arange(Ny2) - Ny2 / 2) * dy
    X, Y = xp.meshgrid(x, y)
    r = xp.sqrt(X ** 2 + Y ** 2 + z ** 2)

    # Rayleigh-Sommerfeld impulse response (first kind):
    #   h = (1/2pi) * (z/r^2) * (ik - 1/r) * exp(ikr)
    h = (z / (2 * np.pi * r ** 2)) * xp.exp(1j * k * r) * (1j * k - 1.0 / r)
    h *= dx * dy  # pixel area for discrete convolution

    if verbose:
        print(f"  RS propagation: z = {z*1e3:.3f} mm")
        print(f"  Grid: {Ny}x{Nx} -> padded {Ny2}x{Nx2}")
        print(f"  Wavelength: {wavelength*1e9:.1f} nm")
        print(f"  Kernel max |h|: {float(xp.max(xp.abs(h))):.4e}")

    # -- convolve via FFT ------------------------------------------------------
    # ifftshift the kernel so its origin is at array index (0,0) for FFT
    if xp is np:
        H = _fft2(np.fft.ifftshift(h))
        E_fft = _fft2(E_padded)
        E_conv = _ifft2(E_fft * H)
    else:
        H = xp.fft.fft2(xp.fft.ifftshift(h))
        E_fft = xp.fft.fft2(E_padded)
        E_conv = xp.fft.ifft2(E_fft * H)

    # -- extract the valid region (same location as input was placed) ----------
    E_out = E_conv[y0:y0 + Ny, x0:x0 + Nx]

    return E_out


# =============================================================================
# SCALABLE ANGULAR SPECTRUM (SAS) — Heintzmann / Loetgering / Wechsler, 2023
# =============================================================================


def scalable_angular_spectrum_propagate(
    E_in,
    z,
    wavelength,
    dx,
    pad=2,
    skip_final_phase=False,
    use_gpu=False,
    verbose=False,
):
    """
    Scalable-angular-spectrum propagator with variable output pitch.

    Implements the Heintzmann-Loetgering-Wechsler 2023 three-FFT kernel:
    apply an ASM-minus-Fresnel precompensation phase in the spatial-frequency
    domain, then a Fresnel chirp + single FFT (+ optional final quadratic
    phase).  The output grid has pixel pitch
    ``dx_out = lambda * z / (pad * N * dx)`` which can be much larger than
    the input pitch, letting one propagate over distances where a standard
    angular-spectrum call would need impractically many samples to span the
    geometric cone of the beam.

    The kernel is exact up to the ASM-vs-Fresnel band-limit cutoff ``W``
    baked into the precompensation; beyond that cutoff the method gracefully
    reduces to a zeroed transfer function (high-NA components are dropped).
    A closed-form ``z_limit`` from the paper bounds the propagation distance
    for which the method remains valid at the input sampling; we warn (not
    raise) when ``z > z_limit`` so the caller can still experiment.

    Parameters
    ----------
    E_in : ndarray (complex, N x N)
        Input field.  Must be square (Ny == Nx = N).  NumPy or CuPy.

    z : float
        Propagation distance [m].  Positive = forward.

    wavelength : float
        Wavelength [m].

    dx : float
        Input grid pitch [m].  Input extent is ``L = N * dx``.

    pad : int, default 2
        Zero-padding factor applied before the SAS kernel.  The reference
        implementation uses 2; larger values reduce aliasing further at the
        cost of more compute.  Output is cropped back to ``N x N`` after
        the kernel runs.

    skip_final_phase : bool, default False
        If True, skip the final post-FFT quadratic phase.  The resulting
        complex field has the correct *intensity* but not the correct phase
        at the output plane.  Equivalent to the paper's
        ``skip_final_phase=True`` mode; cheaper by one N^2 multiply.

    use_gpu : bool, default False
        Run on CuPy when available.  Like the other propagators, if
        ``E_in`` is already a CuPy array this is honoured automatically.

    verbose : bool, default False
        Print grid, pitch, and band-limit diagnostics.

    Returns
    -------
    E_out : ndarray (complex, N x N)
        Propagated field on a grid of pitch ``dx_out``.

    dx_out : float
        Output grid pitch = ``wavelength * z / (pad * N * dx)``.

    dy_out : float
        Output grid pitch in y (equal to ``dx_out`` for square input).

    Notes
    -----
    Choice of propagator by regime (for free-space diffraction of an N x N
    field, extent L, wavelength lam, distance z):

    * ``z << L^2 / (N * lam)``  — use :func:`angular_spectrum_propagate`
      (Fresnel number large, output pitch = input pitch).
    * ``z ~ L^2 / (N * lam)``   — either ASM or SAS work; SAS gives a
      better-scaled output grid if the beam has diverged past the input
      window.
    * ``z >> L^2 / (N * lam)``  — use SAS.  Plain ASM needs a much larger
      N to avoid aliasing; pure Fresnel loses the phase accuracy that SAS
      recovers through its precompensation term.
    * ``z -> infinity``         — :func:`fraunhofer_propagate`.

    Sampling assumption: the input field is centred in its array and the
    returned array is centred (fftshift applied to the SAS output).  This
    differs from the reference notebook which returns FFT-natural order.

    References
    ----------
    [1] Heintzmann, R.; Loetgering, L.; Wechsler, F. (2023).  "Scalable
        angular spectrum propagation".  *Optica* 10(11): 1407-1416.
        doi:10.1364/OPTICA.497809
    [2] Reference PyTorch implementation:
        https://github.com/bionanoimaging/Scalable-Angular-Spectrum-Method-SAS
    """
    # -- array library selection (NumPy vs. CuPy) ---------------------------
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        xp = cp
        if not _is_cupy_array(E_in):
            E_in = cp.asarray(E_in)
    else:
        xp = np
        if _is_cupy_array(E_in):
            E_in = E_in.get()

    # -- validate input ------------------------------------------------------
    if E_in.ndim != 2:
        raise ValueError(
            f"scalable_angular_spectrum_propagate: expected 2-D field, "
            f"got shape {E_in.shape}.")
    Ny, Nx = E_in.shape
    if Ny != Nx:
        raise ValueError(
            f"scalable_angular_spectrum_propagate: input must be square "
            f"(got {Ny}x{Nx}).  The SAS kernel is derived for a single N.")
    N = Nx
    L = N * dx
    pad = int(pad)
    if pad < 1:
        raise ValueError(f"pad must be >= 1, got {pad}")

    # -- closed-form z-limit from Heintzmann et al. (2023) -------------------
    # Beyond z_limit the band-limit filter W kills the ASM-like components
    # that the precompensation phase is meant to correct, and SAS reduces to
    # plain Fresnel with the usual far-field error.
    lam = wavelength
    s = L ** 2 / (8 * L ** 2 + N ** 2 * lam ** 2)
    denom = lam * (-1.0 + 2.0 * np.sqrt(2.0) * np.sqrt(s))
    if abs(denom) < 1e-30:
        z_limit = float("inf")
    else:
        z_limit = float(
            -4.0 * L * np.sqrt(8.0 * L ** 2 / N ** 2 + lam ** 2)
            * np.sqrt(s) / denom)
    if z > z_limit > 0 and verbose:
        print(f"  SAS: z = {z*1e3:.2f} mm exceeds z_limit = "
              f"{z_limit*1e3:.2f} mm; accuracy may degrade.")

    # -- padded grid ---------------------------------------------------------
    L_new = pad * L
    N_new = pad * N

    # -- choose precision from input dtype (matches angular_spectrum_propagate)
    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(DEFAULT_COMPLEX_DTYPE)
    target_fdtype = (np.float32
                     if target_cdtype == np.complex64 else np.float64)

    # -- zero-pad the input, centred ----------------------------------------
    psi_p = xp.zeros((N_new, N_new), dtype=target_cdtype)
    as1 = (N + 1) // 2
    psi_p[as1:as1 + N, as1:as1 + N] = E_in

    # -- spatial-frequency axes (natural FFT order) --------------------------
    #   fftfreq(N_new, d=L_new/N_new) = fftfreq(N_new, d=dx)
    f_x = xp.fft.fftfreq(N_new, d=dx).astype(target_fdtype)
    f_y = f_x  # square grid

    # -- band-limit W: ASM-vs-Fresnel validity region ------------------------
    # Paper eq. (12): the precompensation is valid wherever both
    # inequalities hold, else drop the mode.
    two_z = 2.0 * z if z != 0 else 1e-30
    cx = lam * f_x[None, :]
    cy = lam * f_y[:, None]
    tx = L_new / two_z + xp.abs(cx)
    ty = L_new / two_z + xp.abs(cy)
    W = ((cx ** 2 * (1.0 + tx ** 2) / tx ** 2 + cy ** 2 <= 1.0)
         & (cy ** 2 * (1.0 + ty ** 2) / ty ** 2 + cx ** 2 <= 1.0))

    # -- ASM-minus-Fresnel precompensation phase -----------------------------
    #   H_AS  = sqrt(1 - (lam*fx)^2 - (lam*fy)^2)
    #   H_Fr  = 1 - ((lam*fx)^2 + (lam*fy)^2) / 2
    #   delta_H = W * exp( i * k * z * (H_AS - H_Fr) )
    k = 2 * np.pi / lam
    h_AS = xp.sqrt((1.0 + 0j) - cx ** 2 - cy ** 2)
    h_Fr = 1.0 - 0.5 * (cx ** 2 + cy ** 2)
    delta_H = W * xp.exp(1j * k * z * (h_AS - h_Fr))
    delta_H = delta_H.astype(target_cdtype, copy=False)

    # -- apply precompensation in frequency space ---------------------------
    # The reference uses ifftshift(psi_p) then fft2, i.e. treat the centred
    # array as if its zero-pixel is at the centre.  Match exactly.
    psi_precomp = xp.fft.ifft2(
        xp.fft.fft2(xp.fft.ifftshift(psi_p)) * delta_H)

    # -- Fresnel chirp + single FFT -----------------------------------------
    # x,y arrays are in natural FFT order (ifftshifted centred coords).
    coord_centred = xp.linspace(
        -L_new / 2, L_new / 2, N_new, endpoint=False,
        dtype=target_fdtype)
    coord_nat = xp.fft.ifftshift(coord_centred)
    x = coord_nat[None, :]
    y = coord_nat[:, None]

    H1 = xp.exp(1j * k / (2.0 * z) * (x ** 2 + y ** 2))
    # Fresnel-style amplitude prefactor so the output is the physical
    # diffracted field (not a raw DFT sample).  This matches the
    # normalization used by fresnel_propagate in this library and is
    # absent from the reference PyTorch notebook.
    amp_pref = (dx * dx) / (1j * lam * z)
    if skip_final_phase:
        psi_p_final = amp_pref * xp.fft.fftshift(
            xp.fft.fft2(H1 * psi_precomp))
    else:
        dq = lam * z / L_new  # output pitch on padded grid
        Q = dq * N_new        # full extent of padded output grid
        q_centred = xp.linspace(
            -Q / 2, Q / 2, N_new, endpoint=False, dtype=target_fdtype)
        q_nat = xp.fft.ifftshift(q_centred)
        qx = q_nat[None, :]
        qy = q_nat[:, None]
        H2 = xp.exp(1j * k * z) * xp.exp(
            1j * k / (2.0 * z) * (qx ** 2 + qy ** 2))
        psi_p_final = amp_pref * xp.fft.fftshift(
            H2 * xp.fft.fft2(H1 * psi_precomp))

    # -- crop back to original N x N ----------------------------------------
    E_out = psi_p_final[as1:as1 + N, as1:as1 + N]

    # -- output grid pitch ---------------------------------------------------
    dx_out = lam * z / (pad * N * dx)

    if verbose:
        print(f"  SAS propagation: z = {z*1e3:.3f} mm")
        print(f"  Input  grid: {N}x{N}  pitch {dx*1e6:.3f} um  "
              f"extent {L*1e3:.3f} mm")
        print(f"  Output grid: {N}x{N}  pitch {dx_out*1e6:.3f} um  "
              f"extent {N*dx_out*1e3:.3f} mm  "
              f"(zoom {dx_out/dx:.2f}x)")
        kept = float(xp.sum(W)) / (N_new * N_new)
        print(f"  Band-limit kept: {kept*100:.1f}% of SAS spectrum")
        if z_limit > 0:
            print(f"  z_limit from paper: {z_limit*1e3:.2f} mm")

    return E_out, dx_out, dx_out
