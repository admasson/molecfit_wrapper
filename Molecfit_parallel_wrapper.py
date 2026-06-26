"""
Minimal Molecfit wrapper for HARPS, ESPRESSO, NIRPS, CARMENES VIS/NIR.

What this script does:
1) Reads a CSV list of input spectra (one row per spectrum).
2) Converts each FITS spectrum into Molecfit ASCII input (WAVE, FLUX, ERR, QUAL).
3) Builds spectrum-specific Molecfit parameter files from a generic template.
4) Runs molecfit + calctrans.
5) Saves TAC (transmission corrected models) outputs and a compact run summary.

use along "wrapper_user_guide.ipynb" to generate the proper s1d files format for each instrument.

-------------------------------------------------------------------------------
Author  : A. Masson (amasson@cab.inta-csic.es - amasson.atro@gmail.com)
Date    : 2026
-------------------------------------------------------------------------------

Credits:
- ESO for the Molecfit software.
- J. den Brok for the original molecfit wrapper for PALOMAR/DuPont/Keck,
  upon which this script is built:
  https://github.com/jdenbrok/molecfit_wrapper/tree/main
"""

import csv
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import constants as const


SPEED_OF_LIGHT_M_S = const.c.value
RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
DATE_STRING = time.strftime("_%d_%m_%Y")


# ============================================================
# Wrapper data/folder structure (quick reference)
# ============================================================
# Expected repository layout for this script:
#
# - Molecfit_parallel_wrapper.py
#     Main executable wrapper.
#
# - wrapper_user_guide.ipynb
#     Interactive interface notebook to prepare S1D files/CSVs and inspect outputs.
#
# - Parameter_Files/
#     Static configuration assets used by wrapper:
#     - generic_parfile.par: Molecfit template with placeholders.
#     - tellurics_include_*.dat: fit windows to include.
#     - tellurics_exclude_*.dat: windows to exclude.
#
# - Automated_Program/
#     Runtime working directory used by wrapper:
#     - input CSVs (path,berv) that define spectra to process.
#     - temporary files generated per spectrum (temp/par/include/exclude/ascii).
#     - optional error_spectra.csv if some files fail.
#
# - s1d_carmenes_nir/ and s1d_carmenes_vis/
#     Example input S1D FITS products prepared by notebook cells.
#
# - Output/
#     Per-spectrum Molecfit run folders created by this wrapper.
#     Naming pattern: output{index}_{channel}_{timestamp}/
#
# - Final_Results_Molecfit/
#     Consolidated final products copied by wrapper:
#     - *_TAC.dat per input spectrum.
#     - one global result_listing_*.csv summary.
#
# - backup/
#     Archived legacy assets not used in current clean workflow.


# ============================================================
# User configuration
# ============================================================
# Template par file shipped with Molecfit wrapper setup.
# This file contains placeholders (#path_to_fits, #output_dir, etc.)
# that are replaced for each spectrum before running molecfit.
PATH_TO_GEN_PAR = "/home/amasson/data/molecfit_wrapper/Parameter_Files/generic_parfile.par"

# Input CSV to process. Expected columns:
# - path: absolute or relative path to the input FITS file
# - berv: barycentric Earth radial velocity in m/s (needed when berv_mode=csv_bary_to_topo)
PATH_TO_LIST = "/home/amasson/data/molecfit_wrapper/Automated_Program/noKepler-91b_CARMENES_02-07-2019_CARMENES_NIR_S1D.csv"

# Supported: HARPS, ESPRESSO, NIRPS, CARMENES_VIS, CARMENES_NIR
CONFIGURED_CHANNEL_TYPE = "CARMENES_NIR"

PATH_TO_MOLECFIT_BIN = "/home/amasson/data/molecfit/bin/"
# Working folder where temporary ASCII/par files are written during a run.
PATH_TO_PROGRAM = "/home/amasson/data/molecfit_wrapper/Automated_Program"
# Prefix used to create per-spectrum run folders:
# output{index}_{CONFIGURED_CHANNEL_TYPE}_{RUN_TIMESTAMP}
PATH_TO_RESULTS_PREFIX = "/home/amasson/data/molecfit_wrapper/Output/output"
# Folder with include/exclude telluric windows for each instrument.
PATH_TO_TELLURICS_DIR = "/home/amasson/data/molecfit_wrapper/Parameter_Files"
# Final consolidated output folder where TAC tables + summary CSV are copied.
PATH_TO_FINAL_RESULTS = (
    "/home/amasson/data/molecfit_wrapper/Final_Results_Molecfit/Final_Results" + DATE_STRING
)

# If False: keep Gaussian width fixed to seeded value from resolving power. If False: let Molecfit fit the instrumental resolution kernel.
FIT_RESOLUTION_IN_MOLECFIT = True

# If True and justified by sampling variability, allow linear variable kernel in the molecfit fit.
ALLOW_VARIABLE_KERNEL = True

# Number of parallel worker processes used to process CSV rows.
# - Set to 1 for sequential mode (easier debugging).
# - Set to >1 to process multiple spectra concurrently.
N_PARALLEL_PROCESSES = 4


# ------------------------------------------------------------
# CHANNEL_CONFIGS guide
# ------------------------------------------------------------
# This dictionary is the instrument abstraction layer.
# Each instrument entry tells the wrapper how to:
# - read the spectrum from FITS
# - convert wavelength frame/units consistently
# - choose molecules and fitting windows
# - fetch observatory/weather metadata from header keywords
#
# Field meanings:
# - fits_level: FITS HDU index containing spectrum data
# - vac_air: wavelength medium expected by Molecfit ('vac' or 'air')
# - wave_source: source strategy for wavelength extraction: computing spectral bins from a linear formula and header keywords, or grab wavelength solutions directly from the FITS file.
# - error_mode: how ERR is built ('from_column' or 'unity')
# - resolving_power: instrument resolving power R used for kernel seed
# - berv_mode:
#     - 'csv_bary_to_topo' means CSV must provide berv [m/s] and we de-redshift wavelengths to shift them from barycentric restframe back to Earth restframe
#     - 'none' means no BERV correction is applied in wrapper: we assume data are already in Earth restframe
# - list_molec: molecules included in atmospheric model
# - fit_molec: fit flag per molecule (1 fit, 0 keep fixed)
# - relcol: initial relative molecular column guesses
# - include_file / exclude_file: wavelength windows (in micron) used to constrain fit
# - header_keys: mapping of required physical quantities to FITS keywords
#
# If you add a new instrument, copy one block, then adapt all keys above.

CHANNEL_CONFIGS = {
    "HARPS": {
        # Data layout / spectral physics
        "fits_level": 0,
        "vac_air": "air",
        "wave_source": "HEADER_LINEAR_WCS",
        "error_mode": "unity",
        "resolving_power": 115000.0,
        "berv_mode": "csv_bary_to_topo",

        # Molecules to include and initial fit state
        "list_molec": "H2O O2",
        "fit_molec": "1 1",
        "relcol": "1.0 1.0",

        # Fit windows
        "include_file": PATH_TO_TELLURICS_DIR + "/tellurics_include_harps.dat",
        "exclude_file": PATH_TO_TELLURICS_DIR + "/tellurics_exclude_harps.dat",

        # Header key mapping for atmospheric and site parameters
        "header_keys": {
            "obsdate": "MJD-OBS",
            "utc": "UTC",
            "telalt": "HIERARCH ESO TEL ALT",
            "geoelev": "HIERARCH ESO TEL GEOELEV",
            "longitude": "HIERARCH ESO TEL GEOLON",
            "latitude": "HIERARCH ESO TEL GEOLAT",
            "temp": "HIERARCH ESO TEL AMBI TEMP",
            "pres_start": "HIERARCH ESO TEL AMBI PRES START",
            "pres_end": "HIERARCH ESO TEL AMBI PRES END",
            "rhum": "HIERARCH ESO TEL AMBI RHUM",
            "m1temp": "HIERARCH ESO TEL TH M1 TEMP",
        },
    },
    "ESPRESSO": {
        # Data layout / spectral physics
        "fits_level": 1,
        "vac_air": "vac",
        "wave_source": "WAVE_TABLE_ROW0",
        "error_mode": "from_column",
        "resolving_power": 140000.0,
        "berv_mode": "csv_bary_to_topo",

        # Molecules to include and initial fit state
        "list_molec": "H2O O2",
        "fit_molec": "1 1",
        "relcol": "1.0 1.0",

        # Fit windows
        "include_file": PATH_TO_TELLURICS_DIR + "/tellurics_include_harps.dat",
        "exclude_file": PATH_TO_TELLURICS_DIR + "/tellurics_exclude_harps.dat",

        # Header key mapping for atmospheric and site parameters
        "header_keys": {
            "obsdate": "MJD-OBS",
            "utc": "UTC",
            "telalt": "HIERARCH ESO TEL ALT",
            "geoelev": "HIERARCH ESO TEL GEOELEV",
            "longitude": "HIERARCH ESO TEL GEOLON",
            "latitude": "HIERARCH ESO TEL GEOLAT",
            "temp": "HIERARCH ESO TEL AMBI TEMP",
            "pres_start": "HIERARCH ESO TEL AMBI PRES START",
            "pres_end": "HIERARCH ESO TEL AMBI PRES END",
            "rhum": "HIERARCH ESO TEL AMBI RHUM",
            "m1temp": "HIERARCH ESO TEL TH M1 TEMP",
        },
    },
    "NIRPS": {
        # Data layout / spectral physics
        "fits_level": 1,
        "vac_air": "vac",
        "wave_source": "WAVE_TABLE_ROW0",
        "error_mode": "from_column",
        "resolving_power": 80000.0,
        "berv_mode": "csv_bary_to_topo",

        # Molecules to include and initial fit state
        "list_molec": "H2O CO2 CH4",
        "fit_molec": "1 1 1",
        "relcol": "1.0 1.0 1.0",

        # Fit windows
        "include_file": PATH_TO_TELLURICS_DIR + "/tellurics_include_nirps.dat",
        "exclude_file": PATH_TO_TELLURICS_DIR + "/tellurics_exclude_nirps.dat",

        # Header key mapping for atmospheric and site parameters
        "header_keys": {
            "obsdate": "MJD-OBS",
            "utc": "UTC",
            "telalt": "HIERARCH ESO TEL ALT",
            "geoelev": "HIERARCH ESO TEL GEOELEV",
            "longitude": "HIERARCH ESO TEL GEOLON",
            "latitude": "HIERARCH ESO TEL GEOLAT",
            "temp": "HIERARCH ESO TEL AMBI TEMP",
            "pres_start": "HIERARCH ESO TEL AMBI PRES START",
            "pres_end": "HIERARCH ESO TEL AMBI PRES END",
            "rhum": "HIERARCH ESO TEL AMBI RHUM",
            "m1temp": "HIERARCH ESO TEL AMBI TEMP",
        },
    },
    "CARMENES_VIS": {
        # Data layout / spectral physics
        "fits_level": 1,
        "vac_air": "vac",
        "wave_source": "WAVE_TABLE_ROW0",
        "error_mode": "from_column",
        "resolving_power": 94600.0,
        "berv_mode": "none",

        # Molecules to include and initial fit state
        "list_molec": "H2O O2",
        "fit_molec": "1 1",
        "relcol": "1.0 1.0",

        # Fit windows
        "include_file": PATH_TO_TELLURICS_DIR + "/tellurics_include_carmenes_vis.dat",
        "exclude_file": PATH_TO_TELLURICS_DIR + "/tellurics_exclude_carmenes_vis.dat",

        # Header key mapping for atmospheric and site parameters
        "header_keys": {
            "obsdate": "MJD-OBS",
            "utc": "HIERARCH CAHA INS VIS CCD UT",
            "telalt": "HIERARCH CAHA TEL POS EL_START",
            "geoelev": "HIERARCH CAHA TEL GEOELEV",
            "longitude": "HIERARCH CAHA TEL GEOLON",
            "latitude": "HIERARCH CAHA TEL GEOLAT",
            "temp": "HIERARCH CAHA GEN AMBI TEMPERATURE",
            "pres_start": "HIERARCH CAHA GEN AMBI PRESSURE",
            "pres_end": "HIERARCH CAHA GEN AMBI PRESSURE",
            "rhum": "HIERARCH CAHA GEN AMBI RHUM",
            "m1temp": "HIERARCH CAHA GEN AMBI TEMPERATURE",
        },
    },
    "CARMENES_NIR": {
        # Data layout / spectral physics
        "fits_level": 1,
        "vac_air": "vac",
        "wave_source": "WAVE_TABLE_ROW0",
        "error_mode": "from_column",
        "resolving_power": 80400.0,
        "berv_mode": "none",

        # Molecules to include and initial fit state
        "list_molec": "H2O CO2 CH4",
        "fit_molec": "1 1 1",
        "relcol": "1.0 1.0 1.0",

        # Fit windows
        "include_file": PATH_TO_TELLURICS_DIR + "/tellurics_include_carmenes_nir.dat",
        "exclude_file": PATH_TO_TELLURICS_DIR + "/tellurics_exclude_carmenes_nir.dat",

        # Header key mapping for atmospheric and site parameters
        "header_keys": {
            "obsdate": "MJD-OBS",
            "utc": "UT",
            "telalt": "HIERARCH CAHA TEL POS EL_START",
            "geoelev": "HIERARCH CAHA TEL GEOELEV",
            "longitude": "HIERARCH CAHA TEL GEOLON",
            "latitude": "HIERARCH CAHA TEL GEOLAT",
            "temp": "HIERARCH CAHA GEN AMBI TEMPERATURE",
            "pres_start": "HIERARCH CAHA GEN AMBI PRESSURE",
            "pres_end": "HIERARCH CAHA GEN AMBI PRESSURE",
            "rhum": "HIERARCH CAHA GEN AMBI RHUM",
            "m1temp": "HIERARCH CAHA GEN AMBI TEMPERATURE",
        },
    },
}


def get_channel_config():
    """Return the instrument configuration selected by CONFIGURED_CHANNEL_TYPE.

    Raises:
        ValueError: if CONFIGURED_CHANNEL_TYPE is not one of CHANNEL_CONFIGS keys.
    """
    tag = str(CONFIGURED_CHANNEL_TYPE).strip().upper()
    if tag not in CHANNEL_CONFIGS:
        raise ValueError(f"Unsupported CONFIGURED_CHANNEL_TYPE={CONFIGURED_CHANNEL_TYPE}")
    return CHANNEL_CONFIGS[tag]


def read_input_csv(path_to_csv):
    """Read and normalize the input CSV schema.

    The parser is intentionally permissive for first-column naming:
    - It accepts "path" or "file" (case-insensitive after normalization).
    - If neither exists, it falls back to the first column.

    Parameters:
        path_to_csv (str): path to CSV file.

    Returns:
        pd.DataFrame: DataFrame with normalized columns:
            - path (str)
            - berv (float, NaN if missing)
    """
    if not path_to_csv.lower().endswith(".csv"):
        raise ValueError("Input list must be a CSV with at least a path/file column.")

    raw = pd.read_csv(path_to_csv)
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    path_col = None
    for candidate in ["path", "file"]:
        if candidate in raw.columns:
            path_col = candidate
            break
    if path_col is None:
        path_col = raw.columns[0]

    out = pd.DataFrame()
    out["path"] = raw[path_col].astype(str)
    out["berv"] = pd.to_numeric(raw["berv"], errors="coerce") if "berv" in raw.columns else np.nan
    return out


def first_header_value(path_to_file, keys):
    """Return the first matching header value among candidate FITS keywords.

    Parameters:
        path_to_file (str): FITS file path.
        keys (list[str]): ordered list of header keywords to try.
    """
    header = fits.getheader(path_to_file, 0)
    for key in keys:
        if key in header and header[key] is not None:
            return float(header[key])
    raise KeyError(f"Missing header keywords {keys} in {path_to_file}")


def flatten_table_column(column):
    """Convert a FITS table column into a 1D float array.

    Handles common layouts seen in spectroscopic tables:
    - variable-length (dtype=object): use first row payload
    - one-row 2D arrays: unwrap first row
    - scalar values: convert to 1-element array
    """
    arr = np.asarray(column)

    if arr.dtype == object and arr.size > 0:
        # Typical for variable-length table columns.
        first = np.asarray(arr[0], dtype=float).ravel()
        return first

    if arr.ndim == 0:
        return np.asarray([float(arr)], dtype=float)

    if arr.ndim >= 2 and arr.shape[0] == 1:
        return np.asarray(arr[0], dtype=float).ravel()

    return np.asarray(arr, dtype=float).ravel()


def get_table_column(hdul, ext_idx, candidates):
    """Fetch one named column from a FITS BinTable extension.

    Parameters:
        hdul: opened astropy HDUList
        ext_idx (int): extension index containing the table
        candidates (list[str]): accepted aliases for the same quantity
    """
    data = hdul[ext_idx].data
    names = getattr(data, "names", None)
    if names is None:
        raise ValueError(f"HDU {ext_idx} is not a table")

    by_upper = {str(name).upper(): name for name in names}
    for cand in candidates:
        key = str(cand).upper()
        if key in by_upper:
            return flatten_table_column(data[by_upper[key]])

    raise KeyError(f"Missing any of columns {candidates} in HDU {ext_idx}")


def read_spectrum(path_to_file, channel_config):
    """Read wavelength, flux, and error arrays from one FITS spectrum.

    Behavior depends on channel_config:
    - fits_level=0 (e.g. HARPS): wavelength from linear WCS header
    - fits_level=1 (tables): WAVE/FLUX/ERR-like columns

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (wave, flux, err)
    """
    fits_level = int(channel_config["fits_level"])

    with fits.open(path_to_file) as hdul:
        if fits_level == 0:
            header = hdul[0].header
            wave = (
                header["CRVAL1"]
                + np.arange(header["NAXIS1"], dtype=float) * header["CDELT1"]
            )
            flux = np.asarray(hdul[0].data, dtype=float).ravel()
            err = np.ones_like(flux, dtype=float)
        else:
            wave = get_table_column(hdul, fits_level, ["WAVE", "WAVELENGTH", "LAMBDA"])
            flux = get_table_column(hdul, fits_level, ["FLUX", "SPEC", "SPECTRUM"])

            if str(channel_config["error_mode"]).lower() == "from_column":
                err = get_table_column(hdul, fits_level, ["ERR", "ERROR", "SIG", "SIGMA", "E_FLUX"])
            else:
                err = np.ones_like(flux, dtype=float)

    # Molecfit expects aligned wavelength/flux/error vectors.
    if not (wave.size == flux.size == err.size):
        raise ValueError(
            f"Array size mismatch in {path_to_file}: WAVE={wave.size}, FLUX={flux.size}, ERR={err.size}"
        )

    return wave, flux, err


def sanitize_spectrum(wave_values, flux_values, err_values):
    """Clean one spectrum and build Molecfit QUAL mask.

    Steps:
    1) drop non-finite wavelengths
    2) replace invalid flux/error with robust fallback values
    3) mark replaced points with QUAL=1 (mask), keep valid points QUAL=0
    4) sort by wavelength

    Returns:
        wave, flux_clean, err_clean, qual
    """
    wave = np.asarray(wave_values, dtype=float).ravel()
    flux = np.asarray(flux_values, dtype=float).ravel()
    err = np.asarray(err_values, dtype=float).ravel()

    finite_wave = np.isfinite(wave)
    wave = wave[finite_wave]
    flux = flux[finite_wave]
    err = err[finite_wave]

    # Prevent unstable runs on near-empty arrays.
    if wave.size < 16:
        raise ValueError("Too few valid wavelength samples")

    bad_flux = ~np.isfinite(flux)
    bad_err = (~np.isfinite(err)) | (err <= 0)
    bad = bad_flux | bad_err

    good_flux = flux[~bad_flux]
    good_err = err[(~bad_err) & np.isfinite(err) & (err > 0)]
    flux_fill = float(np.median(good_flux)) if good_flux.size else 1.0
    err_fill = float(np.median(good_err)) if good_err.size else 1e-3

    flux_clean = flux.copy()
    err_clean = err.copy()
    flux_clean[bad] = flux_fill
    err_clean[bad] = max(err_fill, 1e-12)

    # Molecfit convention: QUAL=0 good, QUAL>0 masked.
    qual = np.zeros(wave.size, dtype=int)
    qual[bad] = 1

    order = np.argsort(wave)
    return wave[order], flux_clean[order], err_clean[order], qual[order]


def infer_wlgtomicron(wave_values):
    """Infer conversion factor to microns from wavelength value range.

    Returns:
        float: multiplier converting native wavelength unit to micron.
    """
    arr = np.asarray(wave_values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Cannot infer wavelength unit from empty array")

    if np.nanmax(finite) < 10:
        return 1.0
    if 300 < np.nanmin(finite) and np.nanmax(finite) < 2700:
        return 1e-3
    if 2700 < np.nanmin(finite) and np.nanmax(finite) < 40000:
        return 1e-4

    raise ValueError("Unsupported wavelength unit range")


def write_ascii_with_qual(path_to_ascii, wave, flux, err, qual):
    """Write Molecfit-compatible ASCII input table.

    Column order is fixed by generic par file:
    WAVE, FLUX, ERR, QUAL
    """
    table = np.column_stack([wave, flux, err, qual.astype(int)])
    np.savetxt(path_to_ascii, table)


def load_windows(path_to_file, allow_empty=False):
    """Load telluric include/exclude windows from ASCII file.

    Expected format: two numeric columns per non-comment line -> lo hi (micron).
    """
    if not os.path.exists(path_to_file):
        raise FileNotFoundError(path_to_file)

    windows = []
    with open(path_to_file, "r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if s == "" or s.startswith("#"):
                continue
            lo, hi = s.split()[:2]
            lo = float(lo)
            hi = float(hi)
            if hi > lo:
                windows.append([lo, hi])

    if len(windows) == 0 and not allow_empty:
        raise ValueError(f"No windows found in {path_to_file}")

    return windows


def clip_windows(windows, wave_min_um, wave_max_um):
    """Clip configured windows to the actual spectrum wavelength span."""
    clipped = []
    for lo, hi in windows:
        clo = max(float(lo), float(wave_min_um))
        chi = min(float(hi), float(wave_max_um))
        if chi - clo > 1e-4:
            clipped.append([clo, chi])
    return clipped


def estimate_fwhm_pixels(wave_um, resolving_power):
    """Seed Gaussian FWHM in pixels from instrument resolving power.

    Uses relation:
        FWHM_pix ~ lambda / (R * dlambda_per_pixel)

    Also returns a boolean flag to activate variable kernel when wavelength
    sampling strongly varies over a broad interval.
    """
    dlam = np.diff(wave_um)
    dlam = dlam[np.isfinite(dlam) & (dlam > 0)]
    if dlam.size == 0:
        raise ValueError("Cannot estimate FWHM from degenerate wavelength sampling")

    step = float(np.median(dlam))
    lam0 = float(np.nanmedian(wave_um))
    fwhm_pix = lam0 / (float(resolving_power) * step)
    if not np.isfinite(fwhm_pix) or fwhm_pix <= 0:
        raise ValueError("Invalid seeded FWHM in pixels")

    dyn_ratio = float(np.percentile(dlam, 90) / np.percentile(dlam, 10)) if dlam.size > 20 else 1.0
    use_variable_kernel = bool((wave_um.max() - wave_um.min() > 0.2) and (dyn_ratio > 1.15))
    return fwhm_pix, use_variable_kernel


def replace_placeholder(file_path, pattern, value):
    """Simple placeholder substitution in text files."""
    with open(file_path, "r", encoding="utf-8") as fin:
        text = fin.read()
    text = text.replace(pattern, str(value))
    with open(file_path, "w", encoding="utf-8") as fout:
        fout.write(text)


def set_par_value(file_path, key, value):
    """Set or append a key:value entry in a Molecfit par file."""
    with open(file_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    prefix = f"{key}:"
    new_line = f"{prefix} {value}\n"

    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith(prefix):
            lines[i] = new_line
            replaced = True
            break

    if not replaced:
        lines.append(new_line)

    with open(file_path, "w", encoding="utf-8") as fout:
        fout.writelines(lines)


def get_weather_data(path_to_file, channel_config):
    """Read meteorological/site quantities from instrument-specific header keys.

    Returns:
        tuple[float, float, float, float]: (temp, m1temp, pressure, rhum)
    """
    keys = channel_config["header_keys"]
    temp = first_header_value(path_to_file, [keys["temp"]])
    m1temp = first_header_value(path_to_file, [keys["m1temp"]])
    p0 = first_header_value(path_to_file, [keys["pres_start"]])
    p1 = first_header_value(path_to_file, [keys["pres_end"]])
    rhum = first_header_value(path_to_file, [keys["rhum"]])
    return float(temp), float(m1temp), float(0.5 * (p0 + p1)), float(rhum)


def parse_fit_res(res_path):
    """Parse minimal diagnostics from Spectrum_*_fit.res output.

    Extracted values are kept intentionally short for run summary logging.
    """
    status = np.nan
    reduced_chi2 = np.nan
    fwhm = np.nan

    with open(res_path, "r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if s.startswith("Status:"):
                status = float(s.split(":", 1)[1].strip())
            elif s.startswith("Reduced chi2:"):
                reduced_chi2 = float(s.split(":", 1)[1].strip())
            elif s.startswith("FWHM of Gaussian in pixels:"):
                fwhm = float(s.split(":", 1)[1].split("+-")[0].strip())

    return status, reduced_chi2, fwhm


def run_one_spectrum(index, row, channel_config, template_copy_path):
    """Process one spectrum end-to-end (prepare input, run Molecfit, collect outputs).

    Parameters:
        index (int): row index in CSV, also used in temporary/output filenames.
        row (pd.Series or dict): must contain at least row['path'], optionally row['berv'].
        channel_config (dict): instrument settings from CHANNEL_CONFIGS.
        template_copy_path (str): path to copied generic par template in PATH_TO_PROGRAM.

    Returns:
        dict: compact run summary for this spectrum.
    """
    # STEP 1/11: identify current input file and create destination folder.
    path_to_fits = str(row["path"])
    print(f"[INFO] Preparing {os.path.basename(path_to_fits)}")

    # Keep one output folder per spectrum so debugging is straightforward.
    output_dir = f"{PATH_TO_RESULTS_PREFIX}{index}_{CONFIGURED_CHANNEL_TYPE}_{RUN_TIMESTAMP}"
    os.makedirs(output_dir, exist_ok=True)

    # STEP 2/11: load WAVE, FLUX, ERR arrays from FITS using channel rules.
    wave, flux, err = read_spectrum(path_to_fits, channel_config)

    # STEP 3/11: optionally convert barycentric wavelengths to topocentric frame.
    berv_mode = str(channel_config.get("berv_mode", "none")).lower()
    if berv_mode == "csv_bary_to_topo":
        berv = row["berv"]
        if pd.isna(berv):
            raise ValueError(f"Missing berv in CSV for {path_to_fits}")
        wave = wave / (1.0 + float(berv) / SPEED_OF_LIGHT_M_S)

    # STEP 4/11: sanitize arrays and build QUAL mask expected by Molecfit.
    wave, flux, err, qual = sanitize_spectrum(wave, flux, err)

    # STEP 5/11: infer wavelength unit and convert to micron for window clipping.
    wlgtomicron = infer_wlgtomicron(wave)
    wave_um = wave * wlgtomicron

    # STEP 6/11: load and clip include/exclude windows to this spectrum span.
    # include windows define where fit is constrained; exclude windows mask regions.
    include_windows = load_windows(channel_config["include_file"], allow_empty=False)
    exclude_windows = load_windows(channel_config["exclude_file"], allow_empty=True)

    include_clipped = clip_windows(include_windows, wave_um.min(), wave_um.max())
    exclude_clipped = clip_windows(exclude_windows, wave_um.min(), wave_um.max())

    if len(include_clipped) == 0:
        raise ValueError("No telluric fitting windows overlap this spectrum")

    # STEP 7/11: create temporary working files (ASCII input + par + range files).
    ascii_input = os.path.join(PATH_TO_PROGRAM, f"temp_input_{index}.dat")
    include_file = os.path.join(PATH_TO_PROGRAM, f"include_{index}.dat")
    exclude_file = os.path.join(PATH_TO_PROGRAM, f"exclude_{index}.dat")
    par_file = os.path.join(PATH_TO_PROGRAM, f"temp_{index}.par")
    par_file_calctrans = os.path.join(PATH_TO_PROGRAM, f"temp_{index}_calctrans.par")

    write_ascii_with_qual(ascii_input, wave, flux, err, qual)
    np.savetxt(include_file, np.asarray(include_clipped, dtype=float))
    np.savetxt(exclude_file, np.asarray(exclude_clipped, dtype=float))

    shutil.copy(template_copy_path, par_file)

    # STEP 8/11: instantiate a spectrum-specific par file from generic template.
    replace_placeholder(par_file, "#path_to_fits", ascii_input)
    replace_placeholder(par_file, "#wlgtomicron", str(wlgtomicron))
    replace_placeholder(par_file, "#wrange_include", include_file)
    replace_placeholder(par_file, "#wrange_exclude", exclude_file)
    replace_placeholder(par_file, "#prange_exclude", "none")
    replace_placeholder(par_file, "#output_dir", output_dir)
    replace_placeholder(par_file, "#output_name", f"Spectrum_{index}")

    # STEP 9/11: inject instrument model settings + observing/weather metadata.
    # These values override default par-file keywords and avoid ambiguous header parsing.
    set_par_value(par_file, "vac_air", channel_config["vac_air"])
    set_par_value(par_file, "list_molec", channel_config["list_molec"])
    set_par_value(par_file, "fit_molec", channel_config["fit_molec"])
    set_par_value(par_file, "relcol", channel_config["relcol"])

    # Header-driven observational metadata required by atmospheric model.
    keys = channel_config["header_keys"]
    set_par_value(par_file, "obsdate", first_header_value(path_to_fits, [keys["obsdate"]]))
    set_par_value(par_file, "obsdate_key", "NONE")
    set_par_value(par_file, "utc", first_header_value(path_to_fits, [keys["utc"]]))
    set_par_value(par_file, "utc_key", "NONE")
    set_par_value(par_file, "telalt", first_header_value(path_to_fits, [keys["telalt"]]))
    set_par_value(par_file, "telalt_key", "NONE")
    set_par_value(par_file, "geoelev", first_header_value(path_to_fits, [keys["geoelev"]]))
    set_par_value(par_file, "geoelev_key", "NONE")
    set_par_value(par_file, "longitude", first_header_value(path_to_fits, [keys["longitude"]]))
    set_par_value(par_file, "longitude_key", "NONE")
    set_par_value(par_file, "latitude", first_header_value(path_to_fits, [keys["latitude"]]))
    set_par_value(par_file, "latitude_key", "NONE")

    # Weather terms (site pressure/humidity/temperature and M1 temperature).
    temp, m1temp, pressure, rhum = get_weather_data(path_to_fits, channel_config)
    set_par_value(par_file, "temp", temp)
    set_par_value(par_file, "temp_key", "NONE")
    set_par_value(par_file, "pres", pressure)
    set_par_value(par_file, "pres_key", "NONE")
    set_par_value(par_file, "rhum", rhum)
    set_par_value(par_file, "rhum_key", "NONE")
    set_par_value(par_file, "m1temp", m1temp)
    set_par_value(par_file, "m1temp_key", "NONE")

    # Continue STEP 9/11: seed resolution kernel terms for stable convergence.
    seeded_fwhm, var_kernel = estimate_fwhm_pixels(wave_um, channel_config["resolving_power"])
    set_par_value(par_file, "res_gauss", seeded_fwhm)
    set_par_value(par_file, "fit_res_gauss", int(bool(FIT_RESOLUTION_IN_MOLECFIT)))
    set_par_value(par_file, "varkern", int(bool(ALLOW_VARIABLE_KERNEL and var_kernel)))

    # STEP 10/11: run molecfit (fit step), then calctrans (full-range correction step).
    cmd_molecfit = os.path.join(PATH_TO_MOLECFIT_BIN, "molecfit") + f" {par_file}"
    if os.system(cmd_molecfit) != 0:
        raise RuntimeError(f"molecfit failed for {path_to_fits}")

    # Calctrans uses fitted atmosphere but removes fitting-range restrictions.
    shutil.copy(par_file, par_file_calctrans)
    set_par_value(par_file_calctrans, "wrange_include", "none")
    set_par_value(par_file_calctrans, "wrange_exclude", "none")
    set_par_value(par_file_calctrans, "prange_exclude", "none")

    cmd_calctrans = os.path.join(PATH_TO_MOLECFIT_BIN, "calctrans") + f" {par_file_calctrans}"
    if os.system(cmd_calctrans) != 0:
        raise RuntimeError(f"calctrans failed for {path_to_fits}")

    # STEP 11/11: collect outputs, parse diagnostics, and cleanup temp files.
    tac_source = os.path.join(output_dir, f"Spectrum_{index}_TAC.dat")
    tac_target = os.path.join(
        PATH_TO_FINAL_RESULTS,
        os.path.basename(path_to_fits).replace(".fits", "_TAC.dat"),
    )
    shutil.copy(tac_source, tac_target)

    res_path = os.path.join(output_dir, f"Spectrum_{index}_fit.res")
    status, reduced_chi2, fit_fwhm = parse_fit_res(res_path)

    # Cleanup temporary files produced for this spectrum.
    for tmp in [ascii_input, include_file, exclude_file, par_file, par_file_calctrans]:
        if os.path.exists(tmp):
            os.remove(tmp)

    return {
        "index": index,
        "input_fits": path_to_fits,
        "output_dir": output_dir,
        "status": status,
        "reduced_chi2": reduced_chi2,
        "fwhm_pixels": fit_fwhm,
        "tac_file": tac_target,
    }


def main():
    """Entry point: load config, iterate spectra, and write run summaries."""
    # STEP A: resolve instrument behavior from CONFIGURED_CHANNEL_TYPE.
    channel_config = get_channel_config()

    # STEP B: load and normalize CSV schema into (path, berv).
    spectra = read_input_csv(PATH_TO_LIST)

    os.makedirs(PATH_TO_PROGRAM, exist_ok=True)
    os.makedirs(PATH_TO_FINAL_RESULTS, exist_ok=True)

    # STEP C: copy template locally so original generic file always stays untouched.
    template_copy = os.path.join(PATH_TO_PROGRAM, os.path.basename(PATH_TO_GEN_PAR))
    shutil.copy(PATH_TO_GEN_PAR, template_copy)

    summary_rows = []
    errors = []

    # STEP D: process each spectrum and collect success/failure summaries.
    # Use configurable parallel workers when N_PARALLEL_PROCESSES > 1.
    if int(N_PARALLEL_PROCESSES) <= 1:
        for idx, row in spectra.iterrows():
            try:
                result = run_one_spectrum(idx, row, channel_config, template_copy)
                summary_rows.append(result)
                print(
                    f"[OK] Spectrum_{idx}: status={result['status']}, "
                    f"chi2={result['reduced_chi2']:.4g}, fwhm={result['fwhm_pixels']:.4g} px"
                )
            except Exception as exc:
                errors.append({"index": int(idx), "path": str(row["path"]), "error": str(exc)})
                print(f"[ERROR] Spectrum_{idx}: {exc}")
    else:
        print(f"[INFO] Parallel mode enabled with {int(N_PARALLEL_PROCESSES)} workers")

        futures = {}
        with ProcessPoolExecutor(max_workers=int(N_PARALLEL_PROCESSES)) as executor:
            for idx, row in spectra.iterrows():
                # Pass plain dicts to worker processes for robust pickling.
                row_dict = row.to_dict()
                fut = executor.submit(run_one_spectrum, int(idx), row_dict, channel_config, template_copy)
                futures[fut] = {"index": int(idx), "path": str(row_dict.get("path", ""))}

            for fut in as_completed(futures):
                meta = futures[fut]
                idx = meta["index"]
                path = meta["path"]
                try:
                    result = fut.result()
                    summary_rows.append(result)
                    print(
                        f"[OK] Spectrum_{idx}: status={result['status']}, "
                        f"chi2={result['reduced_chi2']:.4g}, fwhm={result['fwhm_pixels']:.4g} px"
                    )
                except Exception as exc:
                    errors.append({"index": idx, "path": path, "error": str(exc)})
                    print(f"[ERROR] Spectrum_{idx}: {exc}")

    # Keep deterministic summary/error ordering regardless of parallel completion order.
    summary_rows = sorted(summary_rows, key=lambda x: x.get("index", -1))
    errors = sorted(errors, key=lambda x: x.get("index", -1))

    if os.path.exists(template_copy):
        os.remove(template_copy)

    # STEP E: write global run summary CSV.
    summary_csv = os.path.join(PATH_TO_FINAL_RESULTS, f"result_listing_{DATE_STRING}.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "index",
                "input_fits",
                "output_dir",
                "status",
                "reduced_chi2",
                "fwhm_pixels",
                "tac_file",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # STEP F: persist failures separately so successful outputs remain usable.
    if errors:
        error_csv = os.path.join(PATH_TO_PROGRAM, "error_spectra.csv")
        with open(error_csv, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=["index", "path", "error"])
            writer.writeheader()
            writer.writerows(errors)
        print(f"[WARN] {len(errors)} spectra failed. Details: {error_csv}")

    print(f"[INFO] Done. Summary: {summary_csv}")


if __name__ == "__main__":
    main()
