"""
The following Code constitutes a Python wrapper to prepare data in order for molecfit to be able to perform a decent telluric absorption correction.
Molecfit is a software tool developed by ESO (Smette et al. 2015, Kausch et al. 2015):
https://www.eso.org/sci/software/pipelines/skytools/molecfit
"""

# Heavily adapted from original script by:
__author__ = "J. den Brok"
__version__ = "v1.2.1"
__email__ = "jdenbrok@astro.uni-bonn.de"



import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.time import Time
from scipy.interpolate import interp1d
import os, shutil
from tempfile import mkstemp
from os import fdopen, remove
import time
from joblib import Parallel, delayed
import multiprocessing
import csv
import pandas as pd
import mpmath as mpm
import glob


SPEED_OF_LIGHT_M_S = 299792458.0


#string that contains the current date
#useful to keep an order in the directory sorting
date_string = time.strftime("_%d_%m_%Y")
run_timestamp = time.strftime("%Y%m%d_%H%M%S")
molecfit_version = "Release v1.5.9"

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Change following parameters
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#1) <path to generic parameter file(s)>
path_to_gen_par = "/home/amasson/data/molecfit_wrapper/Parameter_Files/generic_parfile.par"
#name of the template parameter file
name_gen_par = "/"+path_to_gen_par.split("/")[-1]

#2) <path to list of spectra file>
path_to_list = "/home/amasson/data/molecfit_wrapper/Automated_Program/Kepler-91b_CARMENES_02-07-2019_CARMENES_NIR.csv"
# Explicit wrapper setup for one homogeneous input CSV.
# Supported values: HARPS, NIRPS, ESPRESSO, CARMENES_VIS, CARMENES_NIR
configured_channel_type = "CARMENES_NIR"

#3) <path to where the software tools >
path_to_molecfit = "/home/amasson/data/molecfit/bin/"

#4) <path to directory where Program is>
path_to_program = "/home/amasson/data/molecfit_wrapper/Automated_Program"

path_to_results = "/home/amasson/data/molecfit_wrapper/Output/output"
path_to_tellurics_dir = "/home/amasson/data/molecfit_wrapper/Parameter_Files"

#<path to file with the telluric region selected for plotting>
path_to_telluric_plotting_reg = "/home/amasson/data/molecfit_wrapper/Parameter_Files/tellurics_plot_reg.dat"


CHANNEL_CONFIGS = {
    "HARPS": {
        "fits_level": 0,
        "vac_air": "air",
        "wave_source": "WAVE",
        "error_mode": "unity",
        "resolving_power": 115000.0,
        "berv_mode": "csv_bary_to_topo",
        "list_molec": "H2O O2",
        "fit_molec": "1 1",
        "relcol": "1.0 1.0",
        "include_file": path_to_tellurics_dir + "/tellurics_include_harps.dat",
        "exclude_file": path_to_tellurics_dir + "/tellurics_exclude_harps.dat",
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
        "fits_level": 1,
        "vac_air": "vac",
        "wave_source": "WAVE_TABLE_ROW0",
        "error_mode": "from_column",
        "resolving_power": 80000.0,
        "berv_mode": "csv_bary_to_topo",
        "list_molec": "H2O CO2 CH4",
        "fit_molec": "1 1 1",
        "relcol": "1.0 1.0 1.0",
        "include_file": path_to_tellurics_dir + "/tellurics_include_nirps.dat",
        "exclude_file": path_to_tellurics_dir + "/tellurics_exclude_nirps.dat",
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
    "ESPRESSO": {
        "fits_level": 1,
        "vac_air": "vac",
        "wave_source": "WAVE",
        "error_mode": "from_column",
        "resolving_power": 140000.0,
        "berv_mode": "csv_bary_to_topo",
        "list_molec": "H2O O2",
        "fit_molec": "1 1",
        "relcol": "1.0 1.0",
        "include_file": path_to_tellurics_dir + "/tellurics_include_harps.dat",
        "exclude_file": path_to_tellurics_dir + "/tellurics_exclude_harps.dat",
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
    "CARMENES_VIS": {
        "fits_level": 1,
        "vac_air": "vac",
        "wave_source": "WAVE",
        "error_mode": "from_column",
        "resolving_power": 94600.0,
        "berv_mode": "none",
        "list_molec": "H2O O2",
        "fit_molec": "1 1",
        "relcol": "1.0 1.0",
        "include_file": path_to_tellurics_dir + "/tellurics_include_carmenes_vis.dat",
        "exclude_file": path_to_tellurics_dir + "/tellurics_exclude_carmenes_vis.dat",
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
        "fits_level": 1,
        "vac_air": "vac",
        "wave_source": "WAVE",
        "error_mode": "from_column",
        "resolving_power": 80400.0,
        "berv_mode": "none",
        "list_molec": "H2O CO2 CH4",
        "fit_molec": "1 1 1",
        "relcol": "1.0 1.0 1.0",
        "include_file": path_to_tellurics_dir + "/tellurics_include_carmenes_nir.dat",
        "exclude_file": path_to_tellurics_dir + "/tellurics_exclude_carmenes_nir.dat",
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

# Convolution behavior for molecfit spectral resolution kernel.
# If False, keep kernel fixed to the wavelength-space estimate from instrument R.
fit_resolution_in_molecfit = True

# Enable only when you explicitly want a wavelength-variable kernel.
allow_variable_kernel = True


#<path to final results>
#this directory will be created
path_to_final_results = "/home/amasson/data/molecfit_wrapper/Final_Results_Molecfit/Final_Results"+date_string

#path to the LSF fiel aht will be creayed if OWN_LSF True
path_kernel_file = path_to_program + "/own_input_kernel.dat"


if ".csv" in path_to_list:
    # Read CSV positionally so both of these are accepted:
    # path
    # /some/file.fits
    # and
    # path,berv
    # /some/file.fits,-20383.5
    # This also tolerates a malformed one-column header like "file" followed by
    # two-column data rows produced in ad-hoc notebooks.
    list_of_spectra = pd.read_csv(
        path_to_list,
        header=0,
        names=["path", "berv"],
        usecols=[0, 1],
    )
    list_of_spectra["path"] = list_of_spectra["path"].astype(str)
else:
    list_of_spectra = np.genfromtxt(path_to_list,dtype=["U150"],names=["path"])



shutil.copy(path_to_gen_par,path_to_program)


def _get_csv_berv_m_per_s(i, path_to_file):
    if not hasattr(list_of_spectra, "columns") or "berv" not in list_of_spectra.columns:
        raise ValueError(
            "input requires a 'berv' column in the CSV (m/s) for file: {}".format(path_to_file)
        )

    berv_value = list_of_spectra["berv"].iloc[i]
    if pd.isna(berv_value):
        raise ValueError(
            "Missing berv value in CSV for file: {}".format(path_to_file)
        )

    return float(berv_value)


def _get_header_berv_m_per_s(path_to_file):
    """
    Read BERV from header and return m/s.
    ESO QC BERV is typically stored in km/s.
    """
    header = fits.getheader(path_to_file, 0)

    # (keyword, multiplicative factor to m/s)
    key_candidates = [
        ("HIERARCH ESO QC BERV", 1000.0),
        ("ESO QC BERV", 1000.0),
        ("BERV", 1000.0),
    ]

    for key, scale_to_m_per_s in key_candidates:
        if key in header and header[key] is not None:
            return float(header[key]) * float(scale_to_m_per_s)

    raise KeyError(
        "Missing BERV keyword in header for file: {}. Tried: {}".format(
            path_to_file,
            ", ".join([k for k, _ in key_candidates]),
        )
    )


def replace(file_path, file_path_new,pattern, subst):
    """
    :param file_path: path to the file being changed
    :param file_path_new: name of the new file
    :param pattern: string to be replaced
    :param subst: string that replaces 
    """
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    if  file_path== file_path_new:
        remove(file_path)
    #Move new file
    shutil.move(abs_path, file_path_new)


def _set_parfile_value(file_path, key, value):
    """
    Update or append a key:value entry in a molecfit par file.
    """
    with open(file_path, "r") as fin:
        lines = fin.readlines()

    prefix = key + ":"
    new_line = "{} {}\n".format(prefix, value)
    replaced = False
    for idx, line in enumerate(lines):
        if line.strip().startswith(prefix):
            lines[idx] = new_line
            replaced = True
            break

    if not replaced:
        lines.append(new_line)

    with open(file_path, "w") as fout:
        fout.writelines(lines)

def _first_header_value(path_to_file, keys):
    header = fits.getheader(path_to_file, 0)
    for key in keys:
        if key in header and header[key] is not None:
            return float(header[key])
    raise KeyError(
        "Missing required header keyword(s) {} in file: {}".format(keys, path_to_file)
    )

def _first_header_raw(path_to_file, keys):
    header = fits.getheader(path_to_file, 0)
    for key in keys:
        if key in header and header[key] is not None:
            return header[key]
    raise KeyError(
        "Missing required header keyword(s) {} in file: {}".format(keys, path_to_file)
    )

def _get_channel_config():
    channel_type = str(configured_channel_type).strip().upper()
    if channel_type not in CHANNEL_CONFIGS:
        raise ValueError(
            "Invalid configured_channel_type '{}'. Supported values are: {}".format(
                configured_channel_type,
                ", ".join(sorted(CHANNEL_CONFIGS.keys())),
            )
        )
    return CHANNEL_CONFIGS[channel_type]


def _get_channel_header_value(path_to_file, channel_config, field_name):
    header_keys = channel_config.get("header_keys")
    if not isinstance(header_keys, dict):
        raise ValueError("Missing 'header_keys' mapping in channel configuration")
    if field_name not in header_keys:
        raise ValueError(
            "Missing header key mapping for '{}' in configured channel".format(field_name)
        )

    key = header_keys[field_name]
    return _first_header_value(path_to_file, [key])


def _build_output_dir_for_spectrum(index, path_to_file):
    family = str(configured_channel_type).strip().upper()
    return f"{path_to_results}{index}_{family}_{run_timestamp}"


def _load_telluric_windows(path_to_windows, allow_empty=False):
    """
    Load telluric windows from ASCII file.
    Accepted formats:
    - 2-column rows: lo hi
    - legacy single-row pairs: lo1 hi1 lo2 hi2 ...
    """
    if path_to_windows is None:
        raise ValueError("Telluric windows file path is not set")
    if not os.path.exists(path_to_windows):
        raise FileNotFoundError("Telluric windows file not found: {}".format(path_to_windows))

    windows = []
    with open(path_to_windows, "r") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 2:
                raise ValueError("Invalid telluric window line '{}' in {}".format(stripped, path_to_windows))
            lo = float(parts[0])
            hi = float(parts[1])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                raise ValueError("Invalid telluric window '{}' '{}' in {}".format(lo, hi, path_to_windows))
            windows.append([lo, hi])

    if len(windows) == 0 and not allow_empty:
        raise ValueError("Telluric windows file is empty: {}".format(path_to_windows))

    return windows


def _get_fit_window_templates(channel_config):
    include_path = channel_config["include_file"]
    exclude_path = channel_config["exclude_file"]
    include_windows = _load_telluric_windows(include_path, allow_empty=False)
    exclude_windows = _load_telluric_windows(exclude_path, allow_empty=True)
    return include_windows, exclude_windows

def get_header_info(path_to_file, expression):
    """
    :param path_to_file: path to the fits file
    :param expression: expression pressent in header
    :return value: header value returned
    """
    hdul = fits.open(path_to_file)
    result_value = hdul[0].header[expression]
    hdul.close()
    
    return result_value

def _wavelength_from_wcs_header(header):
    """
    Build a wavelength array from linear WCS keywords in the primary header.
    Expects CRVAL1 with either CDELT1 or CD1_1 and NAXIS1.
    """
    crval = header["CRVAL1"]
    cdelt = header["CDELT1"]
    npix  = header["NAXIS1"]

    return(crval + np.arange(npix) * cdelt)

def _wavelength_from_harps_drs_header(header, data_len=None):
    """
    Build HARPS wavelengths from linear solution that is read in the S1D header
    The corresponding wavelength is in AIR and, by default, in BARYCENTRIC RF for raw S1D data
    """
    crval = header["CRVAL1"]
    cdelt = header["CDELT1"]
    npix  = header["NAXIS1"]

    return(crval + np.arange(npix) * cdelt)


def _find_column_in_hdul(hdul, candidate_names):
    """
    Search all HDUs for a table column matching one of candidate names.
    Matching is case-insensitive and returns the first hit.
    """
    wanted = {str(name).upper() for name in candidate_names}
    for hdu in hdul:
        data = getattr(hdu, "data", None)
        names = getattr(data, "names", None)
        if not names:
            continue
        # Keep original name mapping so we can index with the exact column key.
        by_upper = {str(col).upper(): col for col in names}
        for key in wanted:
            if key in by_upper:
                return data[by_upper[key]]
    return None

def _estimate_bin_width_from_wave(waves):
    """
    Robust bin-width estimator from wavelength sampling.
    """
    waves = np.asarray(waves, dtype=float)
    if waves.size < 2:
        raise ValueError("Cannot estimate bin width from fewer than 2 wavelength points")
    diffs = np.diff(waves)
    finite = diffs[np.isfinite(diffs)]
    finite = finite[np.abs(finite) > 0]
    if finite.size == 0:
        raise ValueError("Cannot estimate bin width from degenerate wavelength array")
    return float(np.median(np.abs(finite)))

def _get_resolution_and_binwidth(path_to_file, waves=None):
    """
    Get spectral resolution and bin width with broad header support.
    Requires explicit header values; does not use inferred fallbacks.
    """
    resolution = _first_header_value(path_to_file, [
        "SPEC_RES",
        "SPECRES",
        "HIERARCH CARACAL RESOLUTION",
        "HIERARCH CAHA INS SPEC RES",
        "HIERARCH ESO DRS SPEC RES",
        "HIERARCH ESO INS SPEC RES",
        "HIERARCH ESO INS RESOL",
    ])

    bin_width = _first_header_value(path_to_file, [
        "SPEC_BIN",
        "CDELT1",
        "CD1_1",
        "HIERARCH CARACAL SPEC BIN",
    ])

    return float(resolution), float(bin_width)

def get_data(path_to_file, expression, fits_level = 1):
    """
    :param path_to_file: path to the fits file
    :param expression: expression pressent in header
    :param fits_level: level of the data tabel in the fits file
    :return value: header value returned
    """
    hdul = fits.open(path_to_file)
    expr_upper = str(expression).upper()

    if fits_level >= len(hdul):
        hdul.close()
        raise ValueError(
            "Requested FITS extension {} not present in file {}".format(
                fits_level,
                path_to_file,
            )
        )

    if expr_upper in ("WAVE_TABLE_ROW0", "WAVE_ROW0"):
        ext_data = hdul[fits_level].data
        ext_names = getattr(ext_data, "names", None)
        if ext_names is None:
            hdul.close()
            raise ValueError(
                "Requested {} but FITS extension {} is not a table in {}".format(
                    expr_upper,
                    fits_level,
                    path_to_file,
                )
            )

        by_upper = {str(col).upper(): col for col in ext_names}
        if "WAVE" not in by_upper:
            hdul.close()
            raise ValueError(
                "Requested {} but column 'WAVE' is missing in extension {} of {}".format(
                    expr_upper,
                    fits_level,
                    path_to_file,
                )
            )

        wave_column = np.asarray(ext_data[by_upper["WAVE"]])
        hdul.close()
        if wave_column.ndim == 0:
            raise ValueError("Invalid scalar WAVE column in {}".format(path_to_file))
        if wave_column.shape[0] < 1:
            raise ValueError("Empty WAVE column in {}".format(path_to_file))
        return np.asarray(wave_column[0], dtype=float)

    ext_data = hdul[fits_level].data
    ext_names = getattr(ext_data, "names", None)
    if ext_names is not None:
        by_upper = {str(col).upper(): col for col in ext_names}
        if expr_upper in by_upper:
            result_data = ext_data[by_upper[expr_upper]]
            hdul.close()
            if len(result_data) == 1:
                return result_data[0]
            return result_data

    if expr_upper == "WAVE":
        alt_wave = _find_column_in_hdul(
            hdul,
            ["WAVE", "WAVE_AIR", "WAVE_VAC", "WAVELENGTH", "LAMBDA", "LAMBDA_AIR", "LAMBDA_VAC", "WVL"],
        )
        if alt_wave is not None:
            hdul.close()
            return alt_wave

        hdr = hdul[0].header
        if "CRVAL1" in hdr and "CDELT1" in hdr and "NAXIS1" in hdr:
            out = _wavelength_from_wcs_header(hdr)
            hdul.close()
            return out

        hdul.close()
        raise ValueError("No wavelength solution found in file: {}".format(path_to_file))

    if expr_upper == "FLUX":
        alt_flux = _find_column_in_hdul(
            hdul,
            ["FLUX", "SPEC", "SPECTRUM", "SCIENCE", "FLUX_CAL"],
        )
        if alt_flux is not None:
            hdul.close()
            return alt_flux

        if hdul[0].data is not None:
            out = np.asarray(hdul[0].data)
            hdul.close()
            return out

        hdul.close()
        raise ValueError("No FLUX column/data found in file: {}".format(path_to_file))

    if expr_upper in ("ERR", "ERROR"):
        alt_err = _find_column_in_hdul(
            hdul,
            ["ERR", "ERROR", "SIG", "SIGMA", "E_FLUX", "FLUX_ERR"],
        )
        if alt_err is not None:
            hdul.close()
            return alt_err
        hdul.close()
        raise ValueError("No ERR column found in file: {}".format(path_to_file))

    hdul.close()
    raise ValueError("Unsupported expression '{}' for file {}".format(expression, path_to_file))

def get_FWHM(path_to_fits,min_wlg, max_wlg,wvlng_to_mic):
    """
    Function that determines the FWHM for gaussian kernels (or the kernel in general)
    :param path_to_fits: path to the fits file
    :param min_wlg: the lower wavelength bound in microns
    :param max_wlg: the upper wavelength bound in microns
    :param wvlng_to_mic: needed for Xshooter, the conversion factor to microns
    
    :return FWHM: the FWHM in pixels of the central wavelength
    :return var: if long spectrum, do variable with wavelength
    """
    channel_config = _get_channel_config()
    resolving_power = float(channel_config["resolving_power"])
    if resolving_power <= 0:
        raise ValueError("Invalid resolving_power '{}' in channel configuration".format(resolving_power))

    fits_level = int(channel_config["fits_level"])
    waves_for_bin = np.asarray(get_data(path_to_fits, "WAVE", fits_level=fits_level), dtype=float).ravel()

    # Compute FWHM in pixel space directly from local wavelength sampling:
    # FWHM_pix(lambda) = lambda / (R * d_lambda_per_pixel)
    waves_um = waves_for_bin * float(wvlng_to_mic)
    dlam = np.gradient(waves_um)

    valid = np.isfinite(waves_um) & np.isfinite(dlam) & (dlam > 0)
    valid &= (waves_um >= float(min_wlg)) & (waves_um <= float(max_wlg))
    if np.sum(valid) < 10:
        raise ValueError("Too few valid wavelength samples to estimate instrumental FWHM")

    fwhm_pix = waves_um[valid] / (resolving_power * dlam[valid])
    finite_fwhm = fwhm_pix[np.isfinite(fwhm_pix) & (fwhm_pix > 0)]
    if finite_fwhm.size == 0:
        raise ValueError("Instrumental FWHM estimate is empty or non-physical")

    FWHM = float(np.median(finite_fwhm))

    # Turn on variable kernel only if strongly justified by sampling/FWHM drift.
    p10 = float(np.percentile(finite_fwhm, 10))
    p90 = float(np.percentile(finite_fwhm, 90))
    dynamic_ratio = (p90 / p10) if p10 > 0 else 1.0
    var = bool((max_wlg - min_wlg > 0.2) and (dynamic_ratio > 1.15))

    return FWHM, var

tell_for_plot = np.loadtxt(path_to_telluric_plotting_reg)

def get_inclusion_region(wave_min=0, wave_max=4, include_windows=None, instrument_family=None):
    """
    :param wave_min: lower bound of spectrum, in Microns
    :param wave_max: upper bound of spectrum, in Microns
    """
    inclusion_region = []

    if include_windows is None:
        raise ValueError("include_windows must be provided explicitly for instrument '{}'".format(instrument_family))

    # Keep only telluric windows fully inside the observed wavelength range.
    for lo, hi in include_windows:
        if lo >= wave_min and hi <= wave_max:
            inclusion_region.append([lo, hi])

    acceptable = len(inclusion_region) >= 1
    return inclusion_region, acceptable



def get_wlgtomicron(path_to_file, telescope):
    """
    Multiplicative factor to convert wavelength to micron
    :param telescope: if true, we are dealing with an ASCII file
    """
    if telescope:
        waves = np.genfromtxt(path_to_file.replace(".fits",".dat"))[:,0]
        
    else:
        channel_config = _get_channel_config()
        fits_level = int(channel_config["fits_level"])
        waves = get_data(path_to_file,"WAVE", fits_level=fits_level)
    
    if all(i < 10 for i in waves):
        #in this case, the wavelengths are in Microns 
        return 1.
    elif all(i>300 and i < 2700 for i in waves):
        #in this case the wavelengths are in nm  
        return 0.001
    elif all(i>2700 and i < 40000 for i in waves):
        #in this case the wavelength units are Angstroms
        return 0.0001
    else:
        raise ValueError("Wavelength unit could not be determined from data in {}".format(path_to_file))

def save_plot(path_to_tac_file, wlg_to_microns, path_to_output, fitting_range, index):
    """
    Plot raw vs TAC flux from ASCII output table.
    """
    wlg_to_angstrom = wlg_to_microns / 1e-4
    file_name = path_to_tac_file.split("/")[-1]
    name_plot = file_name.replace(".fits", ".pdf")

    tac_dat = os.path.join(path_to_output, file_name.replace(".fits", "_TAC.dat"))
    data_output = np.genfromtxt(tac_dat)
    waves_obj = np.asarray(data_output[:, 0], dtype=float)
    flux_obj = np.asarray(data_output[:, 1], dtype=float)
    flux_obj_tac = np.asarray(data_output[:, 4] if data_output.shape[1] > 5 else data_output[:, 3], dtype=float)

    plot_range = []
    for idx in range(len(tell_for_plot)):
        lo = tell_for_plot[idx, 0] / wlg_to_microns
        hi = tell_for_plot[idx, 1] / wlg_to_microns
        if lo > waves_obj[0] and hi < waves_obj[-1]:
            plot_range.append(tell_for_plot[idx, :])

    plot_range = np.array(plot_range)
    n_plots = len(plot_range)

    if n_plots == 0:
        plt.figure(figsize=(15, 4))
        plt.plot(waves_obj * wlg_to_angstrom, flux_obj, label="raw", color="black", lw=1, alpha=.6)
        plt.plot(waves_obj * wlg_to_angstrom, flux_obj_tac, label="tac", color="red", lw=1)
        plt.legend()
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"f$_\lambda$")
        plt.savefig(path_to_final_results + "/" + name_plot)
        return

    plt.figure(figsize=(15, 8))
    plt.subplot(2, 2, 1)
    plt.title(file_name)
    plt.plot(waves_obj * wlg_to_angstrom, flux_obj, label="raw", color="black", lw=1, alpha=.6)
    plt.plot(waves_obj * wlg_to_angstrom, flux_obj_tac, label="tac", color="red", lw=1)
    plt.legend()
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"f$_\lambda$")
    for idx in range(n_plots):
        plot_indices = np.where(np.logical_and(
            waves_obj > plot_range[idx, 0] / wlg_to_microns,
            waves_obj < plot_range[idx, 1] / wlg_to_microns,
        ))
        plt.subplot(2, 2, idx + 2)
        plt.plot(waves_obj[plot_indices] * wlg_to_angstrom, flux_obj[plot_indices], label="raw", color="black", lw=1, alpha=.6)
        plt.plot(waves_obj[plot_indices] * wlg_to_angstrom, flux_obj_tac[plot_indices], label="tac", color="red", lw=1)
        plt.legend()
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"f$_\lambda$")
    plt.tight_layout()
    plt.savefig(path_to_final_results + "/" + name_plot)


if not os.path.exists(path_to_final_results):
    os.makedirs(path_to_final_results)
def save_result(path_to_file, path_to_output, output_index=None):
    """
    :param path_to_file: The path give in the list of spectra
    :param path_to_output: path to the directory where the output is saves
    """
    file_name = path_to_file.split("/")[-1]
    
    name_results_table = file_name.replace(".fits", "_TAC.dat")
    source_table_candidates = [name_results_table]
    if output_index is not None:
        source_table_candidates.append("Spectrum_" + str(output_index) + "_TAC.dat")

    source_table = None
    for candidate in source_table_candidates:
        candidate_path = os.path.join(path_to_output, candidate)
        if os.path.exists(candidate_path):
            source_table = candidate_path
            break

    if source_table is None:
        raise FileNotFoundError(
            "No TAC .dat output found in {} (tried {})".format(
                path_to_output,
                ", ".join(source_table_candidates),
            )
        )

    dest = os.path.join(path_to_final_results, name_results_table)
    shutil.copy(source_table, dest)
    return dest


def get_FWHM_result(output_dir, output_name):
    """
    A function that reads the FWHM result in the .res file
    
    :param output_dir: directory, where the output is saved
    :param output_name: name of the outputted files
    """
    
    FWHM = None
    with open(output_dir+"/"+output_name+"_fit.res") as f:
        for line in f:
            if "FWHM of Gaussian in pixels: " in line:
                FWHM = line.split(":")[-1].strip()
    if FWHM is None:
        raise ValueError("FWHM of Gaussian in pixels not found in {}_fit.res".format(output_name))
    return FWHM
def get_results(output_dir, output_name):
    """
    A function that reads the reduced chi2 and wavelength solutions result in the .res file
    
    :param output_dir: directory, where the output is saved
    :param output_name: name of the outputted files
    """
    with open(output_dir+"/"+output_name+"_fit.res") as f:
        red_chi2 = None
        wvlg_sol_1 = None
        wvlg_sol_2 = None
        
        for line in f:
            if "Reduced chi2:" in line:
                red_chi2 = line.split(" ")[-1]
            elif "Chip 1, coef 0:" in line:
                wvlg_sol_1 = line.split(" ")[-3]+line.split(" ")[-2]+line.split(" ")[-1]
            elif "Chip 1, coef 1:" in line:
                wvlg_sol_2 = line.split(" ")[-3]+line.split(" ")[-2]+line.split(" ")[-1]

    missing = []
    if red_chi2 is None:
        missing.append("Reduced chi2")
    if wvlg_sol_1 is None:
        missing.append("Chip 1, coef 0")
    if wvlg_sol_2 is None:
        missing.append("Chip 1, coef 1")
    if missing:
        raise ValueError("Missing fit diagnostics in {}_fit.res: {}".format(output_name, ", ".join(missing)))

    return red_chi2,wvlg_sol_1,wvlg_sol_2

def get_signal_to_noise(waves_target, flux_target,bin_size=20):
    """
    :param bin_size: number of elements to be grouped to one
    """
    
    #if bin_size_too_large
    if bin_size>=len(waves_target)//2:
        bin_size = len(waves_target)//2
    
    #prepare the bined array length
    n_elm = len(waves_target)//bin_size
    if len(waves_target)%bin_size!=0:
        n_elm+=1
        
    #inizialize the array:
    moving_averaged_flux = np.zeros(n_elm+2)
    waves_averaged = np.zeros(n_elm+2)
    std_dev = np.zeros(n_elm+2)
    #plus 2, because I add the begining and end of the original array in order for the interpolation to work
    
    for i in range(n_elm-1):
        moving_averaged_flux[i+1] = np.median(flux_target[bin_size*i:bin_size*i+bin_size])
        waves_averaged[i+1] = np.median(waves_target[bin_size*i:bin_size*i+bin_size])
        std_dev[i+1] = np.std(flux_target[bin_size*i:bin_size*i+bin_size])
    moving_averaged_flux[-2] = np.mean(flux_target[bin_size*(n_elm-1):])
    waves_averaged[-2] = np.mean(waves_target[bin_size*(n_elm-1):])
    
    #initialize the end elements:
    moving_averaged_flux[0] = flux_target[0]
    moving_averaged_flux[-1] = flux_target[-1]
    waves_averaged[0] = waves_target[0]
    waves_averaged[-1] = waves_target[-1]
    
    #interpolate
    f_interp = interp1d(waves_averaged,moving_averaged_flux)
    f_std = interp1d(waves_averaged,std_dev)
    
    #return signal to noise given by : mean(mu)/mean(sigma)
    return np.mean(f_interp(waves_target))/np.mean(f_std(waves_target))


def get_weather_data(path_to_file, channel_config):
    """
    Read weather metadata from strict channel-specific header keys.
    :return temp, pres, hum:
    Raises ValueError if any required keyword is missing, so that unsupported
    file formats are caught explicitly rather than silently using wrong values.
    """
    temp = _get_channel_header_value(path_to_file, channel_config, "temp")
    pres_start = _get_channel_header_value(path_to_file, channel_config, "pres_start")
    pres_end = _get_channel_header_value(path_to_file, channel_config, "pres_end")
    pres = 0.5 * (pres_start + pres_end)

    rhum = _get_channel_header_value(path_to_file, channel_config, "rhum")
    mirror_temp = _get_channel_header_value(path_to_file, channel_config, "m1temp")

    return float(temp), float(mirror_temp), float(pres), float(rhum)
    

def _sanitize_spectrum_for_molecfit(wave_values, flux_values, flux_err_values):
    """
    Prepare spectrum arrays for molecfit ASCII input.
    Strategy:
    - Drop rows with invalid wavelength (cannot be represented safely).
    - Keep rows with invalid flux/err but flag them with QUAL=1 so molecfit
      can ignore them while still parsing a fully finite table.
    """
    wave = np.asarray(wave_values, dtype=float).ravel()
    flux = np.asarray(flux_values, dtype=float).ravel()
    ferr = np.asarray(flux_err_values, dtype=float).ravel()

    if not (wave.size == flux.size == ferr.size):
        raise ValueError(
            "WAVE/FLUX/ERR size mismatch: {} / {} / {}".format(
                wave.size, flux.size, ferr.size
            )
        )

    finite_wave = np.isfinite(wave)
    n_drop_wave = int(np.sum(~finite_wave))
    if np.sum(finite_wave) < 10:
        raise ValueError("Too few valid wavelength samples after filtering")

    wave = wave[finite_wave]
    flux = flux[finite_wave]
    ferr = ferr[finite_wave]

    bad_flux = ~np.isfinite(flux)
    bad_err = (~np.isfinite(ferr)) | (ferr <= 0)
    bad_any = bad_flux | bad_err

    # Build robust finite placeholders for flagged rows.
    good_flux = flux[~bad_flux]
    good_err = ferr[(~bad_err) & np.isfinite(ferr) & (ferr > 0)]
    flux_fill = float(np.median(good_flux)) if good_flux.size else 1.0
    err_fill = float(np.median(good_err)) if good_err.size else 1e-3
    err_fill = max(err_fill, 1e-12)

    flux_clean = flux.copy()
    err_clean = ferr.copy()
    flux_clean[bad_any] = flux_fill
    err_clean[bad_any] = err_fill

    # Molecfit convention: QUAL=0 good, QUAL>0 masked/bad.
    qual = np.zeros(wave.size, dtype=int)
    qual[bad_any] = 1

    stats = {
        "n_total": int(wave_values.__len__() if hasattr(wave_values, "__len__") else wave.size),
        "n_drop_wave": n_drop_wave,
        "n_flagged": int(np.sum(bad_any)),
        "n_kept": int(wave.size),
    }
    return wave, flux_clean, err_clean, qual, stats


# def _normalize_spectrum_for_molecfit(wave, flux, ferr, qual_mask, window_pixels=401, percentile=85.0):
#     """
#     Robust continuum normalization for 1D spectra.
#     Uses a rolling upper-percentile continuum estimate and avoids bad pixels.
#     """
#     wave = np.asarray(wave, dtype=float).ravel()
#     flux = np.asarray(flux, dtype=float).ravel()
#     ferr = np.asarray(ferr, dtype=float).ravel()
#     qual = np.asarray(qual_mask, dtype=int).ravel()

#     if not (wave.size == flux.size == ferr.size == qual.size):
#         raise ValueError("Normalization input arrays have mismatched sizes")

#     good = np.isfinite(flux) & np.isfinite(ferr) & (ferr > 0) & (qual == 0)
#     if np.sum(good) < 30:
#         # Not enough good points; keep original scale.
#         cont = np.ones_like(flux, dtype=float)
#         return flux, ferr, cont

#     # Ensure odd window >= 11.
#     w = int(max(11, window_pixels))
#     if w % 2 == 0:
#         w += 1

#     s = pd.Series(flux)
#     q = float(percentile) / 100.0
#     cont = s.rolling(window=w, center=True, min_periods=max(5, w // 6)).quantile(q).to_numpy()

#     # Fill edges/gaps in continuum estimate.
#     finite_cont = np.isfinite(cont) & (cont > 0)
#     if np.sum(finite_cont) < 10:
#         med_flux = float(np.median(flux[good]))
#         cont = np.full_like(flux, med_flux if med_flux > 0 else 1.0, dtype=float)
#     else:
#         x = np.arange(flux.size, dtype=float)
#         cont = np.interp(x, x[finite_cont], cont[finite_cont])
#         med_flux = float(np.median(flux[good]))
#         floor = max(1e-20, 0.02 * med_flux)
#         cont = np.maximum(cont, floor)

#     flux_n = flux / cont
#     ferr_n = ferr / cont
#     ferr_n = np.maximum(ferr_n, 1e-12)
#     return flux_n, ferr_n, cont


def save_array_as_ASCII_with_qual(path_to_fits_file, wave_values, flux_values, flux_err, qual_values):
    """
    Save molecfit ASCII table with explicit QUAL mask column.
    """
    table = np.zeros((len(wave_values), 4))
    table[:, 0] = wave_values
    table[:, 1] = flux_values
    table[:, 2] = flux_err
    table[:, 3] = np.asarray(qual_values, dtype=int)
    np.savetxt(path_to_fits_file.replace(".fits", ".dat"), table)

list_of_spectra_errors = []
list_of_fitting_results = [["Name", "Name TAC","FWHM", "S/N raw","S/N tac","reduced chi2", "Quality","Wvlg solution 1","Wvlg solution 2"]]

def invoke_molecfit(i):
    if hasattr(list_of_spectra, "iloc"):
        path = str(list_of_spectra["path"].iloc[i])
    else:
        path = list_of_spectra["path"][i]

    print("[INFO]\t Preparing " + path.split("/")[-1])
    output_dir_i = _build_output_dir_for_spectrum(i, path)

    channel_config = _get_channel_config()
    configured_family = str(configured_channel_type).strip().upper()
    instrument_header = str(_first_header_raw(path, ["INSTRUME"])).strip().upper()
    instrument_to_family = {
        "HARPS": "HARPS",
        "NIRPS": "NIRPS",
        "ESPRESSO": "ESPRESSO",
        "CARMENES": "CARMENES_VIS",  # CARMENES family requires explicit VIS/NIR selection in config.
    }
    expected_family = instrument_to_family.get(instrument_header)
    if expected_family is not None and configured_family != expected_family and not (
        instrument_header == "CARMENES" and configured_family in ("CARMENES_VIS", "CARMENES_NIR")
    ):
        raise ValueError(
            "Configured channel '{}' does not match INSTRUME='{}' for file '{}'. "
            "Set configured_channel_type accordingly.".format(
                configured_family,
                instrument_header,
                path,
            )
        )

    wave_expression = channel_config["wave_source"]
    fits_level = int(channel_config["fits_level"])
    error_mode = str(channel_config["error_mode"]).strip().lower()
    wavelengths_array = np.asarray(get_data(path, wave_expression, fits_level=fits_level), dtype=float)
    flux_array = np.asarray(get_data(path, "FLUX", fits_level=fits_level), dtype=float)
    if error_mode == "from_column":
        flux_error_array = np.asarray(get_data(path, "ERR", fits_level=fits_level), dtype=float)
    elif error_mode == "unity":
        flux_error_array = np.ones_like(flux_array, dtype=float)
        print("[INFO]\t Using unity error vector for this channel configuration.")
    else:
        raise ValueError("Unsupported error_mode '{}' for configured channel.".format(error_mode))

    berv_mode = str(channel_config.get("berv_mode", "none")).strip().lower()
    if berv_mode == "csv_bary_to_topo":
        print("[INFO]\t Applying CSV BERV correction (barycentric -> topocentric).")
        berv_m_per_s = _get_csv_berv_m_per_s(i, path)
        wavelengths_array = wavelengths_array / (1.0 + berv_m_per_s / SPEED_OF_LIGHT_M_S)
    elif berv_mode == "header_bary_to_topo":
        print("[INFO]\t Applying header BERV correction (barycentric -> topocentric).")
        berv_m_per_s = _get_header_berv_m_per_s(path)
        wavelengths_array = wavelengths_array / (1.0 + berv_m_per_s / SPEED_OF_LIGHT_M_S)
    elif berv_mode == "none":
        pass
    else:
        raise ValueError("Unsupported berv_mode '{}' for configured channel.".format(berv_mode))

    wave_clean, flux_clean, err_clean, qual_mask, qc_stats = _sanitize_spectrum_for_molecfit(
        wavelengths_array,
        flux_array,
        flux_error_array,
    )

    if qc_stats["n_drop_wave"] > 0 or qc_stats["n_flagged"] > 0:
        print(
            "[INFO]\t Input cleaning for {}: dropped {} invalid-wave rows, flagged {} bad flux/err rows.".format(
                os.path.basename(path),
                qc_stats["n_drop_wave"],
                qc_stats["n_flagged"],
            )
        )

    flux_for_fit = flux_clean
    err_for_fit = err_clean

    save_array_as_ASCII_with_qual(path, wave_clean, flux_for_fit, err_for_fit, qual_mask)
    wlgtomicron = get_wlgtomicron(path, True)
    lambda_min = float(np.nanmin(wave_clean)) * wlgtomicron
    lambda_max = float(np.nanmax(wave_clean)) * wlgtomicron

    name_temp_par = "/temp_" + str(i) + ".par"
    temp_par_path = path_to_program + name_temp_par
    replace(
        path_to_program + name_gen_par,
        temp_par_path,
        "#path_to_fits",
        path.replace(".fits", ".dat"),
    )
    _set_parfile_value(temp_par_path, "vac_air", channel_config["vac_air"])
    _set_parfile_value(temp_par_path, "list_molec", channel_config["list_molec"])
    _set_parfile_value(temp_par_path, "fit_molec", channel_config["fit_molec"])
    _set_parfile_value(temp_par_path, "relcol", channel_config["relcol"])
    replace(temp_par_path, temp_par_path, "#wlgtomicron", str(wlgtomicron))

    exclude_file_path = path_to_program + "/exclude_" + str(i) + ".dat"
    include_file_path = path_to_program + "/include_" + str(i) + ".dat"
    replace(temp_par_path, temp_par_path, "#wrange_include", include_file_path)
    replace(temp_par_path, temp_par_path, "#wrange_exclude", exclude_file_path)
    replace(temp_par_path, temp_par_path, "#prange_exclude", "none")
    replace(temp_par_path, temp_par_path, "#output_dir", output_dir_i)
    replace(temp_par_path, temp_par_path, "#output_name", "Spectrum_" + str(i))

    # Inject weather/site parameters from strict channel-specific headers.
    mjd_obs = _get_channel_header_value(path, channel_config, "obsdate")
    replace(temp_par_path, temp_par_path, "obsdate#:", "obsdate: " + str(mjd_obs))
    replace(temp_par_path, temp_par_path, "obsdate_key: MJD-OBS", "obsdate_key: NONE")

    utc_sec = _get_channel_header_value(path, channel_config, "utc")
    replace(temp_par_path, temp_par_path, "utc#:", "utc: " + str(utc_sec))
    replace(temp_par_path, temp_par_path, "utc_key: UTC", "utc_key: NONE")

    telalt = _get_channel_header_value(path, channel_config, "telalt")
    replace(temp_par_path, temp_par_path, "telalt#:", "telalt: " + str(telalt))
    replace(temp_par_path, temp_par_path, "telalt_key: ESO TEL ALT", "telalt_key: NONE")

    geoelev = _get_channel_header_value(path, channel_config, "geoelev")
    replace(temp_par_path, temp_par_path, "geoelev#:", "geoelev: " + str(geoelev))
    replace(temp_par_path, temp_par_path, "geoelev_key: ESO TEL GEOELEV", "geoelev_key: NONE")

    longitude = _get_channel_header_value(path, channel_config, "longitude")
    replace(temp_par_path, temp_par_path, "longitude#:", "longitude: " + str(longitude))
    replace(temp_par_path, temp_par_path, "longitude_key: ESO TEL GEOLON", "longitude_key: NONE")

    latitude = _get_channel_header_value(path, channel_config, "latitude")
    replace(temp_par_path, temp_par_path, "latitude#:", "latitude: " + str(latitude))
    replace(temp_par_path, temp_par_path, "latitude_key: ESO TEL GEOLAT", "latitude_key: NONE")

    temp, mir_temp, pressure, humid = get_weather_data(path, channel_config)
    replace(temp_par_path, temp_par_path, "temp#:", "temp: " + str(temp))
    replace(temp_par_path, temp_par_path, "temp_key: ESO TEL AMBI TEMP", "temp_key: NONE")
    replace(temp_par_path, temp_par_path, "pres#:", "pres: " + str(pressure))
    replace(temp_par_path, temp_par_path, "pres_key: ESO TEL AMBI PRES START", "pres_key: NONE")
    replace(temp_par_path, temp_par_path, "rhum#:", "rhum: " + str(humid))
    replace(temp_par_path, temp_par_path, "rhum_key: ESO TEL AMBI RHUM", "rhum_key: NONE")
    replace(temp_par_path, temp_par_path, "m1temp#:", "m1temp: " + str(mir_temp))
    replace(temp_par_path, temp_par_path, "m1temp_key: ESO TEL TH M1 TEMP", "m1temp_key: NONE")

    include_windows_template, exclude_windows_template = _get_fit_window_templates(channel_config)

    Exluding = []
    for lo, hi in exclude_windows_template:
        if lo >= lambda_min and hi <= lambda_max:
            Exluding.append([lo, hi])

    np.savetxt(exclude_file_path, Exluding)

    Including = get_inclusion_region(
        lambda_min,
        lambda_max,
        include_windows=include_windows_template,
        instrument_family=str(configured_channel_type).strip().upper(),
    )

    if not Including[1]:
        raise ValueError("Inclusion region too small for file: {}".format(path))
    np.savetxt(include_file_path, Including[0])

    res_FWHM, var_wlg = get_FWHM(path, lambda_min, lambda_max, wlgtomicron)
    print("[INFO]\t Resolution seed (pixels) for {}: {:.4f} (fit_res_gauss={}, varkern={})".format(
        os.path.basename(path),
        float(res_FWHM),
        int(bool(fit_resolution_in_molecfit)),
        int(bool(allow_variable_kernel and var_wlg)),
    ))
    replace(temp_par_path, temp_par_path, "res_gauss: #res_gauss", "res_gauss: " + str(res_FWHM))
    if not fit_resolution_in_molecfit:
        replace(temp_par_path, temp_par_path, "fit_res_gauss: 1", "fit_res_gauss: 0")
    _set_parfile_value(temp_par_path, "varkern", str(int(bool(allow_variable_kernel and var_wlg))))

    print("[INFO]\t Running Molecfit")
    molecfit_cmd = path_to_molecfit + "molecfit " + temp_par_path
    if os.system(molecfit_cmd) != 0:
        raise RuntimeError("molecfit failed for {}".format(path))

    # Run calctrans on full spectral range: keep the fit constrained by
    # wrange_include/wrange_exclude, but do not limit transmission output.
    name_temp_par_calctrans = "/temp_" + str(i) + "_calctrans.par"
    temp_calctrans_path = path_to_program + name_temp_par_calctrans
    shutil.copy(temp_par_path, temp_calctrans_path)
    replace(temp_calctrans_path, temp_calctrans_path,
        "wrange_include: "+include_file_path, "wrange_include: none")
    replace(temp_calctrans_path, temp_calctrans_path,
        "wrange_exclude: "+exclude_file_path, "wrange_exclude: none")
    replace(temp_calctrans_path, temp_calctrans_path,
        "prange_exclude: none", "prange_exclude: none")
    calctrans_cmd = path_to_molecfit + "calctrans " + temp_calctrans_path
    if os.system(calctrans_cmd) != 0:
        raise RuntimeError("calctrans failed for {}".format(path))

    ###
    ### Save the results in a seperate directory
    ###
    
    save_result(path, output_dir_i, output_index=i)

    # Persist fit-range files in output folder for notebook diagnostics.
    if os.path.exists(include_file_path):
        shutil.copy(include_file_path, output_dir_i + "/include_" + str(i) + ".dat")
    if os.path.exists(exclude_file_path):
        shutil.copy(exclude_file_path, output_dir_i + "/exclude_" + str(i) + ".dat")

    path_final_file = output_dir_i + "/" + path.split("/")[-1].replace(".fits", "_TAC.dat")
    final_table = np.genfromtxt(path_final_file)
    waves_targ = final_table[:, 0]
    flux_targ = final_table[:, 1]
    flux_targ_tac = final_table[:, 4] if final_table.shape[1] > 5 else final_table[:, 3]

    s_n_raw = get_signal_to_noise(waves_targ, flux_targ)
    s_n_tac = get_signal_to_noise(waves_targ, flux_targ_tac)
    
    # Legacy master-transmission quality scoring removed.
    # -1 means "not evaluated".
    quality_label = -1
        
    
    ##get other parameters
    red_chi2, wvlg_sol1, wavlg_sol2 = get_results(output_dir_i,"Spectrum_"+str(i))
    FWHM_result = get_FWHM_result(output_dir_i, "Spectrum_" + str(i))
    
    returner = [
        path.split("/")[-1].replace(".fits", " "),
        path.split("/")[-1].replace(".fits", "_TAC.dat"),
        FWHM_result,
        s_n_raw,
        s_n_tac,
        red_chi2,
        quality_label,
        wvlg_sol1,
        wavlg_sol2,
    ]

    remove(temp_par_path)
    if os.path.exists(temp_calctrans_path):
        remove(temp_calctrans_path)
    remove(exclude_file_path)
    remove(include_file_path)

    return True, returner, [path, wlgtomicron, output_dir_i, Including[0], i]


if __name__ == '__main__':
    num_cores = multiprocessing.cpu_count()
    
    #results = Parallel(n_jobs=1)(delayed(invoke_molecfit)(iterator) for iterator in range(12,len(list_of_spectra["path"])))
    results = Parallel(n_jobs=1)(delayed(invoke_molecfit)(iterator) for iterator in range(len(list_of_spectra["path"])))
    for j in range(len(results)):
        if results[j][0]:
            save_plot(*results[j][2])
            list_of_fitting_results.append(results[j][1])
        elif not results[j][0]:
            list_of_spectra_errors.append(results[j][1])
    if len(list_of_spectra_errors) != 0:
        print("For some files, an error occured")
        np.savetxt(path_to_program+"/error_spectra.dat",list_of_spectra_errors,fmt="%s")
    
    
    #remove temporary par file:
    remove(path_to_program+name_gen_par)
    
    #save the final result listing
    csvfile = path_to_final_results+"/result_listing_"+date_string+".csv"
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(list_of_fitting_results)
