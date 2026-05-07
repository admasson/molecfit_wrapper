"""
The following Code constitutes a Python wrapper to prepare data in order for molecfit to be able to perform a decent telluric absorption correction.
Molecfit is a software tool developed by ESO (Smette et al. 2015, Kausch et al. 2015):
https://www.eso.org/sci/software/pipelines/skytools/molecfit
"""

__author__ = "J. den Brok"
__version__ = "v1.2.1"
__email__ = "jdenbrok@astro.uni-bonn.de"



import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.time import Time
from scipy.interpolate import interp1d
from scipy.optimize import minimize
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

try:
    from PyAstronomy import pyasl
except Exception as e:
    raise ImportError("PyAstronomy is required but could not be imported") from e


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
path_to_list = "/home/amasson/data/molecfit_wrapper/Automated_Program/TOI-969b_HARPS_03-03-2024.csv"

#3) <path to where the software tools >
path_to_molecfit = "/home/amasson/data/molecfit/bin/"

#4) <path to directory where Program is>
path_to_program = "/home/amasson/data/molecfit_wrapper/Automated_Program"

path_to_results = "/home/amasson/data/molecfit_wrapper/Output/output"

#<path to file with inclusion regions>
path_to_tellurics = "/home/amasson/data/molecfit_wrapper/Parameter_Files/tellurics_regions.dat"
path_to_tellurics_dir = "/home/amasson/data/molecfit_wrapper/Parameter_Files"

# Dedicated fit windows per instrument/channel (in microns).
path_to_telluric_include_by_instrument = {
    "HARPS": path_to_tellurics_dir + "/tellurics_include_harps.dat",
    "NIRPS": path_to_tellurics_dir + "/tellurics_include_nirps.dat",
    "CARMENES_VIS": path_to_tellurics_dir + "/tellurics_include_carmenes_vis.dat",
    "CARMENES_NIR": path_to_tellurics_dir + "/tellurics_include_carmenes_nir.dat",
}

# Dedicated exclusion ranges per instrument/channel. Keep files empty if unused.
path_to_telluric_exclude_by_instrument = {
    "HARPS": path_to_tellurics_dir + "/tellurics_exclude_harps.dat",
    "NIRPS": path_to_tellurics_dir + "/tellurics_exclude_nirps.dat",
    "CARMENES_VIS": path_to_tellurics_dir + "/tellurics_exclude_carmenes_vis.dat",
    "CARMENES_NIR": path_to_tellurics_dir + "/tellurics_exclude_carmenes_nir.dat",
}

#<path to file with the telluric region selected for plotting>
path_to_telluric_plotting_reg = "/home/amasson/data/molecfit_wrapper/Parameter_Files/tellurics_plot_reg.dat"

#<path to file with emission lines>
path_to_ems_lines = "/home/amasson/data/molecfit_wrapper/Parameter_Files/Emission_Lines.dat"


# #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# # Optional, only necessary if Palomar or Du Pont
# #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# #<path to the palomar wearther data + first part of filename>
# path_p60 = "/home/data/molecfit_wrapper/Parameter_Files/weather_data/p60weather.txt"
# path_p200 = "/home/data/molecfit_wrapper/Parameter_Files/weather_data/p200weather.txt"

# #<path to the DuPont wearther data >
# path_DuPont_weather = "/home/data/molecfit_wrapper/Parameter_Files/weather_data/weather_data_all.txt"
# #<path to Master Transmission File>
# path_master_trans = "/home/data/molecfit_wrapper/Parameter_Files/Master_Transmission.txt"

# Optional weather files for Palomar/DuPont-specific modes.
path_p60 = None
path_p200 = None
path_DuPont_weather = None

# Quality-check master transmission templates.
# Set to None to disable global quality check.
path_master_trans = "/home/data/molecfit_wrapper/Parameter_Files/Master_Transmission.txt"

# Optional per-instrument templates. Use None to skip quality check for that case.
path_master_trans_by_instrument = {
    "HARPS": None,
    "NIRPS": None,
    "CARMENES_VIS": None,
    "CARMENES_NIR": None,
}

# Optional fallback resolving powers when header resolution keywords are absent.
# Values are typical defaults and can be tuned per dataset.
default_resolution_by_instrument = {
    "HARPS": 115000.0,
    "NIRPS": 80000.0,
    "CARMENES_VIS": 94600.0,
    "CARMENES_NIR": 80400.0,
}

# Molecule setup per instrument/channel.
molecfit_molecules_by_instrument = {
    "HARPS": {
        "list_molec": "H2O O2",
        "fit_molec": "1 1",
        "relcol": "1.0 1.0",
    },
    "NIRPS": {
        "list_molec": "H2O CO2 CH4",
        "fit_molec": "1 1 1",
        "relcol": "1.0 1.0 1.0",
    },
    "CARMENES_VIS": {
        "list_molec": "H2O O2",
        "fit_molec": "1 1",
        "relcol": "1.0 1.0",
    },
    "CARMENES_NIR": {
        "list_molec": "H2O CO2 CH4",
        "fit_molec": "1 1 1",
        "relcol": "1.0 1.0 1.0",
    },
}

# Convolution behavior for molecfit spectral resolution kernel.
# If False, keep kernel fixed to the wavelength-space estimate from instrument R.
fit_resolution_in_molecfit = False

# Enable only when you explicitly want a wavelength-variable kernel.
allow_variable_kernel = True

# Optional flux pre-normalization before running molecfit.
# This helps when S1D continuum shape is not compatible with low-order
# continuum fitting in the parfile.
normalize_s1d_before_molecfit = True
normalization_window_pixels = 401
normalization_percentile = 85.0


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
    print('h')
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
    missing = []
    for key in keys:
        try:
            value = get_header_info(path_to_file, key)
            if value is not None:
                return float(value)
        except KeyError:
            missing.append(key)
            continue
    raise KeyError(
        "Missing required header keyword(s) {} in file: {}".format(missing, path_to_file)
    )

def _first_header_raw(path_to_file, keys):
    missing = []
    for key in keys:
        try:
            value = get_header_info(path_to_file, key)
            if value is not None:
                return value
        except KeyError:
            missing.append(key)
            continue
    raise KeyError(
        "Missing required header keyword(s) {} in file: {}".format(missing, path_to_file)
    )

def _detect_instrument_family(path_to_file):
    instr = str(_first_header_raw(path_to_file, ["INSTRUME"])).upper()
    subsys = str(_first_header_raw(path_to_file, ["SUBSYS"])).upper()
    if "HARPS" in instr:
        return "HARPS"
    if "NIRPS" in instr:
        return "NIRPS"
    if "CARMENES" in instr:
        if "VIS" in subsys:
            return "CARMENES_VIS"
        if "NIR" in subsys:
            return "CARMENES_NIR"
        raise ValueError(
            "CARMENES detected but SUBSYS keyword ('{}') is neither VIS nor NIR "
            "in file: {}".format(subsys, path_to_file)
        )
    raise ValueError(
        "Unsupported instrument '{}' in file: {}. "
        "Only HARPS, NIRPS, and CARMENES (VIS/NIR) are supported.".format(instr, path_to_file)
    )

def _get_master_transmission_path(path_to_file):
    family = _detect_instrument_family(path_to_file)
    if family not in path_master_trans_by_instrument:
        raise ValueError(
            "No master transmission template configured for instrument family '{}' "
            "(file: {}). Add an entry to path_master_trans_by_instrument.".format(family, path_to_file)
        )
    return path_master_trans_by_instrument[family]


def _build_output_dir_for_spectrum(index, path_to_file):
    family = _detect_instrument_family(path_to_file)
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

    try:
        arr = np.loadtxt(path_to_windows, comments="#")
    except Exception:
        raise ValueError("Could not parse telluric windows file: {}".format(path_to_windows))

    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        if allow_empty:
            return []
        raise ValueError("Telluric windows file is empty: {}".format(path_to_windows))

    if arr.ndim == 1:
        if arr.size % 2 != 0:
            raise ValueError("Invalid telluric window file '{}' (odd number of values)".format(path_to_windows))
        arr = arr.reshape(-1, 2)
    elif arr.ndim == 2:
        if arr.shape[1] == 2:
            arr = arr
        elif arr.shape[0] == 1 and arr.shape[1] % 2 == 0:
            arr = arr.reshape(-1, 2)
        else:
            raise ValueError("Invalid telluric window file '{}' (unsupported shape {})".format(path_to_windows, arr.shape))
    else:
        raise ValueError("Invalid telluric window file '{}'".format(path_to_windows))

    windows = []
    for lo, hi in arr:
        if np.isfinite(lo) and np.isfinite(hi) and (hi > lo):
            windows.append([float(lo), float(hi)])
    return windows


def _get_fit_window_templates(instrument_family):
    if instrument_family not in path_to_telluric_include_by_instrument:
        raise ValueError("No include-window file configured for instrument family '{}'".format(instrument_family))
    if instrument_family not in path_to_telluric_exclude_by_instrument:
        raise ValueError("No exclude-window file configured for instrument family '{}'".format(instrument_family))

    include_path = path_to_telluric_include_by_instrument[instrument_family]
    exclude_path = path_to_telluric_exclude_by_instrument[instrument_family]
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
    try:
        # Try to read the requested expression from the configured data extension.
        result_data = hdul[fits_level].data[expression]

        # SOAR-like spectra can store only a plain array instead of named columns.
        if np.isscalar(result_data):
            result_data = hdul[fits_level].data

        if len(result_data)==1:
            return result_data[0]
        else:
            return result_data
    except Exception:
        expr_upper = str(expression).upper()

        # Generic table fallbacks (useful for CARMENES/CARACAL naming).
        if expr_upper == "WAVE":
            alt_wave = _find_column_in_hdul(
                hdul,
                ["WAVE", "WAVE_AIR", "WAVE_VAC", "WAVELENGTH", "LAMBDA", "LAMBDA_AIR", "LAMBDA_VAC", "WVL"],
            )
            if alt_wave is not None:
                return alt_wave

        if expr_upper == "FLUX":
            alt_flux = _find_column_in_hdul(
                hdul,
                ["FLUX", "SPEC", "SPECTRUM", "SCIENCE", "FLUX_CAL"],
            )
            if alt_flux is not None:
                return alt_flux

        if expr_upper in ("ERR", "ERROR"):
            alt_err = _find_column_in_hdul(
                hdul,
                ["ERR", "ERROR", "SIG", "SIGMA", "E_FLUX", "FLUX_ERR"],
            )
            if alt_err is not None:
                return alt_err

        # Wavelength array from primary-HDU header.
        # HARPS and NIRPS S1D files carry a linear WCS (CRVAL1/CDELT1/NAXIS1);
        # For other instruments the dimensionality determines the strategy.
        if expr_upper == "WAVE":
            hdr = hdul[0].header
            instr = str(hdr["INSTRUME"]).upper()
            if "HARPS" in instr or "NIRPS" in instr:
                return _wavelength_from_wcs_header(hdr)
            naxis = hdr["NAXIS"]
            naxis2 = hdr.get("NAXIS2")
            if naxis2 is None:
                raise KeyError("NAXIS2 keyword is required for non-HARPS/NIRPS wavelength reconstruction")
            is_1d = (naxis == 1) or (naxis2 == 1) or (naxis2 == 0)
            if is_1d:
                return _wavelength_from_wcs_header(hdr)
            elif naxis == 2:
                data_len = None
                if hdul[0].data is not None:
                    data_len = np.asarray(hdul[0].data).size
                return _wavelength_from_harps_drs_header(hdr, data_len=data_len)
            else:
                raise ValueError(
                    "Unsupported FITS image geometry (NAXIS={}, NAXIS2={}) for wavelength "
                    "reconstruction in file: {}".format(naxis, naxis2, path_to_file)
                )

        # Flux from 1D primary-image spectra (HARPS/NIRPS S1D only).
        if expr_upper == "FLUX" and hdul[0].data is not None:
            instr = str(hdul[0].header["INSTRUME"]).upper()
            if "HARPS" in instr or "NIRPS" in instr:
                return np.array(hdul[0].data)
            raise ValueError(
                "Cannot read FLUX from primary image HDU for unsupported instrument '{}' "
                "in file: {}".format(instr, path_to_file)
            )

        raise
    finally:
        hdul.close()

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def get_n(sigm):
    sample = np.arange(0,1000)
    gaussian_sample = gaussian(sample,0,sigm)
    n_return = np.where(gaussian_sample<1e-20)[0][0]
    return n_return
    
def create_LSF_file(filepath, telescope):
    """
    :param filepath: path to the input fits or ASCII file
    """
    #the wavelengths of the spectrum
    if telescope:
        waves_target = np.genfromtxt(filepath.replace(".fits",".dat"))[:,0]
        resol = 7400/float(get_header_info(filepath,"SKYFWHM"))
        bin_width = get_header_info(filepath,"CDELT1")
    else:    
        waves_target = get_data(filepath,"WAVE")
        resol, bin_width = _get_resolution_and_binwidth(filepath, waves=waves_target)
    
    
    FWHMs = np.round(waves_target/resol/bin_width,4)
    sigmas = FWHMs / (2*np.sqrt(2*np.log(2)))
    
    n = 2*get_n(sigmas[-1])

    # want uneven number of 
    if n%2==0:
        n+=1
        
    LSF_table = np.zeros((len(waves_target),n))
    LSF_size = np.arange(-n//2+1, n//2+1,1)#check how to floor the lower bound
    
    for i in range(len(waves_target)):
        LSF_table[i,:]=gaussian(LSF_size,0,sigmas[i])
        
    
   
    with open(path_kernel_file, 'wb') as fout:
        np.savetxt(fout,LSF_table, fmt='%.2e')
        
        
    return FWHMs[len(FWHMs)//2]


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
    waves_for_bin = np.asarray(get_data(path_to_fits, "WAVE"), dtype=float).ravel()
    resol, _ = _get_resolution_and_binwidth(path_to_fits, waves=waves_for_bin)

    # Compute FWHM in pixel space directly from local wavelength sampling:
    # FWHM_pix(lambda) = lambda / (R * d_lambda_per_pixel)
    waves_um = waves_for_bin * float(wvlng_to_mic)
    dlam = np.gradient(waves_um)

    valid = np.isfinite(waves_um) & np.isfinite(dlam) & (dlam > 0)
    valid &= (waves_um >= float(min_wlg)) & (waves_um <= float(max_wlg))
    if np.sum(valid) < 10:
        raise ValueError("Too few valid wavelength samples to estimate instrumental FWHM")

    fwhm_pix = waves_um[valid] / (float(resol) * dlam[valid])
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
        waves = get_data(path_to_file,"WAVE")
    
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

def save_plot(path_to_tac_file,wlg_to_microns, palomar, path_to_output,fitting_range,index):
    """
    :param path_to_tac_file: path to the telluric corrected fits (or ASCII) file 
    :param wlg_to_microns: conversionfactor to 
    :param fitting_range: the region where the fit is done
    :param index: the index of the spectrum (the i in the forloop)
    """
    wlg_to_angstrom = wlg_to_microns/1e-4
    if palomar:
        #prepare the file string to lead to the TAC file
        file_name = path_to_tac_file.split("/")[-1]
        name_results = file_name.replace(".fits", "_TAC.dat")
        name_plot = file_name.replace(".fits", ".pdf")
        data_output = np.genfromtxt(path_to_output+"/"+name_results)
        #import the data
        waves_obj = data_output[:,0]
        flux_obj = data_output[:,1]
        if np.shape(data_output)[1]>5:
            flux_obj_tac = data_output[:,4]
        else:
            flux_obj_tac = data_output[:,3]
    else:
        #prepare the file string to lead to the TAC file
        file_name = path_to_tac_file.split("/")[-1]
        name_results = file_name.replace(".fits", "_TAC.fits")
        name_plot = file_name.replace(".fits", ".pdf")
        
        #import the data
        waves_obj = get_data(path_to_final_results+"/"+name_results,"WAVE")
        flux_obj = get_data(path_to_final_results+"/"+name_results,"FLUX")
        flux_obj_tac = get_data(path_to_final_results+"/"+name_results,"tacflux")
        
        #take care, that for VIS the wavelenthregion at boundary not included:
        inices = np.where(waves_obj>0.56/wlg_to_microns)
        waves_obj = waves_obj[inices]
        flux_obj = flux_obj[inices]
        flux_obj_tac = flux_obj_tac[inices]
        
    # Import the telluric fit for inspection.
    # Depending on molecfit output settings, fit arrays can be in
    # Spectrum_i_fit.fits or Spectrum_i_fit.res.fits.
    waves_fit = None
    flux_fit = None
    flux_fit_model = None
    fit_candidates = [
        os.path.join(path_to_output, "Spectrum_"+str(index)+"_fit.fits"),
        os.path.join(path_to_output, "Spectrum_"+str(index)+"_fit.res.fits"),
    ]
    if not any(os.path.exists(p) for p in fit_candidates):
        fit_candidates.extend(sorted(glob.glob(os.path.join(path_to_output, "*fit*.fits"))))

    for fit_path in fit_candidates:
        if not os.path.exists(fit_path):
            continue
        try:
            hdul_fit = fits.open(fit_path)
            for hdu in hdul_fit:
                data = getattr(hdu, "data", None)
                names = getattr(data, "names", None)
                if not names:
                    continue
                by_upper = {str(n).upper(): n for n in names}
                w_key = by_upper.get("LAMBDA", by_upper.get("WAVE", by_upper.get("WAVELENGTH")))
                f_key = by_upper.get("FLUX", by_upper.get("SPEC", by_upper.get("SPECTRUM")))
                m_key = by_upper.get("MFLUX", by_upper.get("MODEL", by_upper.get("BESTFIT", by_upper.get("FIT"))))
                if w_key is not None and f_key is not None and m_key is not None:
                    waves_fit = np.asarray(data[w_key], dtype=float)
                    flux_fit = np.asarray(data[f_key], dtype=float)
                    flux_fit_model = np.asarray(data[m_key], dtype=float)
                    break
            hdul_fit.close()
            if waves_fit is not None:
                break
        except Exception as e:
            if 'hdul_fit' in locals() and hdul_fit is not None:
                hdul_fit.close()
            raise RuntimeError("Could not read fit diagnostics file '{}'".format(fit_path)) from e
    
    ###
    ### Plot the telluric correction
    ###
    #determine number of subplots needed:
    plot_range = []
    for i in range(len(tell_for_plot)):
        if tell_for_plot[i,0]/wlg_to_microns>waves_obj[0] and tell_for_plot[i,1]/wlg_to_microns<waves_obj[-1]:
            plot_range.append(tell_for_plot[i,:])
    
    plot_range = np.array(plot_range)
    n_plots = len(plot_range)
    
    if n_plots == 0:
        #plot the telluric correction
        plt.figure(figsize=(15,4))
        plt.plot(waves_obj*wlg_to_angstrom,flux_obj, label="raw", color = "black", lw = 1, alpha = .6)
        plt.plot(waves_obj*wlg_to_angstrom,flux_obj_tac, label="tac", color = "red", lw = 1)
        plt.legend()
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"f$_\lambda$")
        plt.savefig(path_to_final_results+"/"+name_plot)
    else:
        #plot the telluric correction
        fig = plt.figure(figsize=(15,8))
        plt.subplot(2, 2, 1)
        plt.title(file_name)
        plt.plot(waves_obj*wlg_to_angstrom,flux_obj,label="raw", color = "black", lw = 1, alpha = .6)
        plt.plot(waves_obj*wlg_to_angstrom,flux_obj_tac,label="tac", color ="red", lw = 1)
        plt.legend()
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"f$_\lambda$")
        for j in range(n_plots):
            plot_indices = np.where(np.logical_and(waves_obj>plot_range[j,0]/wlg_to_microns,waves_obj<plot_range[j,1]/wlg_to_microns))
            plt.subplot(2, 2, j+2)
            plt.plot(waves_obj[plot_indices]*wlg_to_angstrom,flux_obj[plot_indices],label="raw", color = "black", lw = 1, alpha = .6)
            plt.plot(waves_obj[plot_indices]*wlg_to_angstrom,flux_obj_tac[plot_indices],label="tac", color = "red", lw = 1)
            plt.legend()
            plt.xlabel(r"$\lambda$")
            plt.ylabel(r"f$_\lambda$")
        plt.tight_layout()
        plt.savefig(path_to_final_results+"/"+name_plot)
        
    ###
    ### Plot the Fitting region
    ###
    if waves_fit is None or flux_fit is None or flux_fit_model is None:
        print("[WARN]\t Fit model arrays not found in output directory; skipping fit-range plot for", file_name)
        return

    n_fit = len(fitting_range)
    
    plot_row = 2
    if n_fit==5 or n_fit == 6:
        plot_row = 3
    elif n_fit==7 or n_fit == 8:
        plot_row = 5
    
    fig = plt.figure(figsize=(15,8))
    plt.title(file_name+" Fit")
    for i in range(n_fit):
        range_to_plot=np.where(np.logical_and(waves_fit>fitting_range[i][0]-0.005,waves_fit<fitting_range[i][1]+0.005))
        plt.subplot(plot_row, 2, i+1)
        plt.plot(waves_obj[range_to_plot]*wlg_to_angstrom,flux_fit[range_to_plot],label="raw", color = "black", lw = 1)
        plt.plot(waves_obj[range_to_plot]*wlg_to_angstrom,flux_fit_model[range_to_plot],label="model", color = "red", lw = 1)
        plt.legend()
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"f$_\lambda$")
        plt.legend()
    plt.tight_layout()
    plt.savefig(path_to_final_results+"/"+name_plot.replace(".pdf","_Fit.pdf"))


if not os.path.exists(path_to_final_results):
    os.makedirs(path_to_final_results)
def save_result(path_to_file,path_to_output, SOAR_tel = False,Pal_no_err = False, output_index=None):
    """
    :param path_to_file: The path give in the list of spectra
    :param path_to_output: path to the directory where the output is saves
    """
    file_name = path_to_file.split("/")[-1]
    
    name_results = file_name.replace(".fits", "_TAC.fits")
    source_fits_candidates = [name_results]
    if output_index is not None:
        source_fits_candidates.append("Spectrum_"+str(output_index)+"_TAC.fits")

    source_fits = None
    for candidate in source_fits_candidates:
        candidate_path = os.path.join(path_to_output, candidate)
        if os.path.exists(candidate_path):
            source_fits = candidate_path
            break

    if source_fits is not None:
        shutil.copy(source_fits, os.path.join(path_to_final_results, name_results))
    else:
        #in this case, we must add the table to the fits file
        name_results_table = file_name.replace(".fits", "_TAC.dat")
        source_table_candidates = [name_results_table]
        if output_index is not None:
            source_table_candidates.append("Spectrum_"+str(output_index)+"_TAC.dat")

        source_table = None
        for candidate in source_table_candidates:
            candidate_path = os.path.join(path_to_output, candidate)
            if os.path.exists(candidate_path):
                source_table = candidate_path
                break

        if source_table is None:
            raise FileNotFoundError(
                "No TAC output found in {} (tried {})".format(
                    path_to_output,
                    ", ".join(source_fits_candidates + source_table_candidates),
                )
            )

        name_results_tac = file_name.replace(".fits", "_TAC.fits")
        
        hdul_file = fits.open(path_to_file)
        table_tac = np.genfromtxt(source_table)
        header = hdul_file[0].header
        
            
        #append the tac fluxes table to the fits file
        if SOAR_tel or Pal_no_err:
            tac_flux = table_tac[:,3]
            
            try:
                hdul_file[0].data = np.vstack([hdul_file[0].data, tac_flux])
            except ValueError:
                hdul_file[0].data = np.vstack([hdul_file[0].data, np.array([[tac_flux]])])
        else:
            tac_flux = table_tac[:,4]
            if hdul_file[0].data is None:
                # NIRPS S1D-like products can have NAXIS=0 in primary HDU.
                # Build a compact primary image with [wave, flux, tac, tac_err].
                wave = table_tac[:,0]
                flux = table_tac[:,1]
                tac_flux_err = table_tac[:,5]
                hdul_file[0].data = np.vstack([wave, flux, tac_flux, tac_flux_err])
            elif len(np.shape(hdul_file[0].data))>2:
                hdul_file[0].data = np.vstack([hdul_file[0].data[0][0],hdul_file[0].data[1][0], tac_flux])
                tac_flux_err = table_tac[:,5]
                hdul_file[0].data = np.vstack([hdul_file[0].data,tac_flux_err])
            else:
                hdul_file[0].data = np.vstack([hdul_file[0].data, tac_flux])
                tac_flux_err = table_tac[:,5]
                hdul_file[0].data = np.vstack([hdul_file[0].data,tac_flux_err])
        #save the fits file
        hdul_file[0].header["BUNITTAC"] = "erg/s/cm2/AA"
        hdul_file[0].header["TAC_INFO"] = "TAC performed "+time.strftime("%m/%d/%Y")+", Molecfit "+molecfit_version
        hdul_file.writeto(path_to_final_results+"/"+name_results_tac, overwrite = True,output_verify='ignore')
        hdul_file.close()


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
                FWHM = line.split(" ")[-3]
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


def get_wstart(ref, wave_ref, wave_per_pixel):
    """
    :param ref:
    :param wave_ref:
    """

    return wave_ref - ((ref-1)*wave_per_pixel)



def get_wavelength(start_wave, wave_per_pixel, size):

    return np.array([start_wave + i*wave_per_pixel for i in range(size)])

def get_telalt(path_to_fits_file):
    """
    calculates Telescope altitude angle in deg
    """
    Angle_string = get_header_info(path_to_fits_file,"ANGLE").split(" ")

    # a few Palomar spectra where not nicely formated
    if len(Angle_string) == 1:
        deg =Angle_string[0].split("deg")[0]
        minute = Angle_string[0].split("deg")[-1].replace("min","")
        telalt = float(deg)+float(minute)/60
    else:
        if "deg" in Angle_string[0]:
            Angle_string[0] = Angle_string[0].replace("deg","")

        if "min" in Angle_string[1]:
            Angle_string[1] = Angle_string[1].replace("min","")
            telalt = float(Angle_string[0])+float(Angle_string[1])/60
        else:
            telalt = float(Angle_string[0])+float(Angle_string[2])/60
    return telalt

def round_number(x, base=3):
    """
    rounding function for the get_weather_data function
    """
   
    rounded = int(base * round(float(x)/base))
    if rounded ==0 or rounded ==3 or rounded ==6 or rounded==9:
        rounded = "0" + str(rounded)
    else:
        rounded = str(rounded)
    return rounded

def get_weather_data(path_to_file):
    """
    Read weather metadata from generic ESO-like header keys.
    :return temp, pres, hum:
    Raises ValueError if any required keyword is missing, so that unsupported
    file formats are caught explicitly rather than silently using wrong values.
    """
    temp = _first_header_value(path_to_file, [
        "HIERARCH ESO TEL AMBI TEMP", "ESO TEL AMBI TEMP", "TEL AMBI TEMP",
        "HIERARCH CAHA GEN AMBI TEMPERATURE"
    ])

    pres_start = _first_header_value(path_to_file, [
        "HIERARCH ESO TEL AMBI PRES START", "ESO TEL AMBI PRES START", "TEL AMBI PRES START"
    ])
    pres_end = _first_header_value(path_to_file, [
        "HIERARCH ESO TEL AMBI PRES END", "ESO TEL AMBI PRES END", "TEL AMBI PRES END"
    ])
    pres = 0.5 * (pres_start + pres_end)

    rhum = _first_header_value(path_to_file, [
        "HIERARCH ESO TEL AMBI RHUM", "ESO TEL AMBI RHUM", "TEL AMBI RHUM",
        "HIERARCH CAHA GEN AMBI RHUM"
    ])

    m1 = _first_header_value(path_to_file, [
        "HIERARCH ESO TEL TH M1 TEMP", "ESO TEL TH M1 TEMP", "TEL TH M1 TEMP"
    ])
    mirror_temp = m1

    return float(temp), float(mirror_temp), float(pres), float(rhum)
    

O_2_region = [[680,700],[750,780]]
H_2O_region = [[645,660], [715,740],[785,855], [885,1000]]


def residual_scaling(scale_factor,wave_target, trans_target,wave_mas, trans_mas):
    """
    function that finds the scale factor such that best matches master
    :param scale_factor: the factor by which the target spectrum has to be multiplied to best fit the generic, master 
    transmission spetcrum
    :param wave_target: array containing the wavelengths of the target (in nm)
    :param trans_target: array containing the transmission values
    :param wave_mas: array containing the wavelengths of the master transmission spectrum (in nm)
    :param trans_mas:  array containing the transmission values of the master transmission spectrum
    """
    #interpolate the master transmission. The target spectrum might be of different length 
    inetrp_mas = interp1d(wave_mas,trans_mas, bounds_error = False,fill_value="extrapolate")
    
    #masked = np.where(np.logical_or(wave_target<680, wave_target>780, np.logical_and(wave_target>720, wave_target<750)))
    
    #scale the target transmission
    scaled = scale_factor *(trans_target-np.ones(len(trans_target))) + np.ones(len(trans_target))
    
    #the residual: difference between the two spectra
    resid = sum(abs(scaled-inetrp_mas(wave_target)))
    
    return resid


def scales(waves_target, waves_master, trans_target, trans_master):
    """
    :param wave_target: array containing the wavelengths of the target (in nm)
    :param trans_target: array containing the transmission values
    :param waves_master: array containing the wavelengths of the master transmission spectrum (in nm)
    :param trans_master:  array containing the transmission values of the master transmission spectrum
    """
    #have two main different absorption lines. Need to be looked at separatly, as they scale differently
    O_2_scales = []
    H2O_scales = []
    
    #the O_2 region
    for i in range(len(O_2_region)):
        wave_min = O_2_region[i][0]
        wave_max = O_2_region[i][1]
        
    
        
        index_range = np.where(np.logical_and(waves_target>wave_min, waves_target<wave_max))
        index_range_master = np.where(np.logical_and(waves_master>wave_min, waves_master<wave_max))
        if len(index_range[0]) == 0 or len(index_range_master[0]) == 0:
            continue
        
        O_2_scales.append(minimize(residual_scaling,x0=1, args=(waves_target[index_range], trans_target[index_range],\
                                      waves_master[index_range_master],trans_master[index_range_master])).x)
    
    #the H2_O region
    for i in range(len(H_2O_region)):
        wave_min = H_2O_region[i][0]
        wave_max = H_2O_region[i][1]
        
        index_range = np.where(np.logical_and(waves_target>wave_min, waves_target<wave_max))
        index_range_master = np.where(np.logical_and(waves_master>wave_min, waves_master<wave_max))
        if len(index_range[0]) == 0 or len(index_range_master[0]) == 0:
            continue
        H2O_scales.append(minimize(residual_scaling,x0=1, args=(waves_target[index_range], trans_target[index_range],\
                                      waves_master[index_range_master],trans_master[index_range_master])).x)
    
    return np.array(O_2_scales), np.array(H2O_scales)


def get_trans_spectrum(path_to, use_ascii_input=False):
    """
    :param path: path to the fits file
    :return waves: wavelength array
    :return trans_spec: transmission array
    """
    #import of the data
    
    #different procedure for Palomar or 
    if use_ascii_input:
        data_output = np.genfromtxt(path_to)
        flux = data_output[:,1]
        if np.shape(data_output)[1]>5:
            tac_flux=data_output[:,4]
        else:
            tac_flux=data_output[:,3]
        waves = data_output[:,0]
      
    else:
        hdul = fits.open(path_to)
        waves = hdul[1].data["WAVE"]
        flux = hdul[1].data["FLUX"]
        tac_flux = hdul[1].data["tacflux"]
        hdul.close()
    
    #preparation of the data
    trans_spec = np.zeros(len(flux))
    for i in range(len(flux)):
        if tac_flux[i]!=0.:
            trans_spec[i] = flux[i]/tac_flux[i]
        elif flux[i] == tac_flux[i]:
            trans_spec[i] = 1.
        else:
            trans_spec[i] = 0.
            
    return waves, trans_spec

def test_region(scalings):
    """
    Function that checks, whether the different telluric regions do not have to large variances
    """
    accept = True
    scale_median = np.median(scalings)
    
    for scaling in scalings:
        if scale_median < scaling-0.07 or scale_median > scaling+0.07:
            accept = False
    if len(scalings)==0:
        accept = True
    return accept


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


def _normalize_spectrum_for_molecfit(wave, flux, ferr, qual_mask, window_pixels=401, percentile=85.0):
    """
    Robust continuum normalization for 1D spectra.
    Uses a rolling upper-percentile continuum estimate and avoids bad pixels.
    """
    wave = np.asarray(wave, dtype=float).ravel()
    flux = np.asarray(flux, dtype=float).ravel()
    ferr = np.asarray(ferr, dtype=float).ravel()
    qual = np.asarray(qual_mask, dtype=int).ravel()

    if not (wave.size == flux.size == ferr.size == qual.size):
        raise ValueError("Normalization input arrays have mismatched sizes")

    good = np.isfinite(flux) & np.isfinite(ferr) & (ferr > 0) & (qual == 0)
    if np.sum(good) < 30:
        # Not enough good points; keep original scale.
        cont = np.ones_like(flux, dtype=float)
        return flux, ferr, cont

    # Ensure odd window >= 11.
    w = int(max(11, window_pixels))
    if w % 2 == 0:
        w += 1

    s = pd.Series(flux)
    q = float(percentile) / 100.0
    cont = s.rolling(window=w, center=True, min_periods=max(5, w // 6)).quantile(q).to_numpy()

    # Fill edges/gaps in continuum estimate.
    finite_cont = np.isfinite(cont) & (cont > 0)
    if np.sum(finite_cont) < 10:
        med_flux = float(np.median(flux[good]))
        cont = np.full_like(flux, med_flux if med_flux > 0 else 1.0, dtype=float)
    else:
        x = np.arange(flux.size, dtype=float)
        cont = np.interp(x, x[finite_cont], cont[finite_cont])
        med_flux = float(np.median(flux[good]))
        floor = max(1e-20, 0.02 * med_flux)
        cont = np.maximum(cont, floor)

    flux_n = flux / cont
    ferr_n = ferr / cont
    ferr_n = np.maximum(ferr_n, 1e-12)
    return flux_n, ferr_n, cont


def save_array_as_ASCII(path_to_fits_file, wave_values, flux_values, flux_err):
    """
    :param path_to_fits_file: path to the fits file 
    :param wave_values: list/array containing the wavelength values
    :param flux_values: list/array containing the flux values
    """
    #save data into a single array (4 columns: WAVE FLUX ERR QUAL)
    table = np.zeros((len(wave_values),4))
    table[:,0] = wave_values
    table[:,1] = flux_values
    table[:,2] = flux_err
    # col 3 is QUAL: 0 = good pixel (molecfit convention)
    
    #save the array
    np.savetxt(path_to_fits_file.replace(".fits",".dat"),table)


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

def save_array_as_ASCII_no_err(path_to_fits_file, wave_values, flux_values):
    """
    :param path_to_fits_file: path to the fits file
    :param wave_values: list/array containing the wavelength values
    :param flux_values: list/array containing the flux values
    """
    # save data into a single array
    table = np.zeros((len(wave_values), 3))
    table[:, 0] = wave_values
    table[:, 1] = flux_values
    
    #save the array
    np.savetxt(path_to_fits_file.replace(".fits",".dat"),table)

def nonlinearwave(nwave, specstr, verbose=False):
    """Compute non-linear wavelengths from multispec string
    
    Returns wavelength array and dispersion fields.
    Raises a ValueError if it can't understand the dispersion string.
    """

    fields = specstr.split()
    if int(fields[2]) != 2:
        raise ValueError('Not nonlinear dispersion: dtype=' + fields[2])
    if len(fields) < 12:
        raise ValueError('Bad spectrum format (only %d fields)' % len(fields))
    wt = float(fields[9])
    w0 = float(fields[10])
    ftype = int(fields[11])
    if ftype == 3:

        # cubic spline

        if len(fields) < 15:
            raise ValueError('Bad spline format (only %d fields)' % len(fields))
        npieces = int(fields[12])
        pmin = float(fields[13])
        pmax = float(fields[14])
        if verbose:
            print('Dispersion is order-{}d cubic spline'.format(npieces))
        if len(fields) != 15 + npieces + 3:
            raise ValueError('Bad order-%d spline format (%d fields)' % (npieces, len(fields)))
        coeff = np.asarray(fields[15:], dtype=float)
        # normalized x coordinates
        s = (np.arange(nwave, dtype=float) + 1 - pmin) / (pmax - pmin) * npieces
        j = s.astype(int).clip(0, npieces - 1)
        a = (j + 1) - s
        b = s - j
        x0 = a ** 3
        x1 = 1 + 3 * a * (1 + a * b)
        x2 = 1 + 3 * b * (1 + a * b)
        x3 = b ** 3
        wave = coeff[j] * x0 + coeff[j + 1] * x1 + coeff[j + 2] * x2 + coeff[j + 3] * x3

    elif ftype == 1 or ftype == 2:

        # chebyshev or legendre polynomial
        # legendre not tested yet

        if len(fields) < 15:
            raise ValueError('Bad polynomial format (only %d fields)' % len(fields))
        order = int(fields[12])
        pmin = float(fields[13])
        pmax = float(fields[14])
        if verbose:
            if ftype == 1:
                print('Dispersion is order-{}d Chebyshev polynomial'.format(order))
            else:
                print('Dispersion is order-{}d Legendre polynomial (NEEDS TEST)'.format(order))
        if len(fields) != 15 + order:
            # raise ValueError('Bad order-%d polynomial format (%d fields)' % (order, len(fields)))
            if verbose:
                print('Bad order-{}d polynomial format ({}d fields)'.format(order, len(fields)))
                print("Changing order from {} to {}".format(order, len(fields) - 15))
            order = len(fields) - 15
        coeff = np.asarray(fields[15:], dtype=float)
        # normalized x coordinates
        pmiddle = (pmax + pmin) / 2
        prange = pmax - pmin
        x = (np.arange(nwave, dtype=float) + 1 - pmiddle) / (prange / 2)
        p0 = np.ones(nwave, dtype=float)
        p1 = x
        wave = p0 * coeff[0] + p1 * coeff[1]
        for i in range(2, order):
            if ftype == 1:
                # chebyshev
                p2 = 2 * x * p1 - p0
            else:
                # legendre
                p2 = ((2 * i - 1) * x * p1 - (i - 1) * p0) / i
            wave = wave + p2 * coeff[i]
            p0 = p1
            p1 = p2

    else:
        raise ValueError('Cannot handle dispersion function of type %d' % ftype)

    return wave, fields


def readmultispec(fitsfile, reform=True, quiet=False):
    """Read IRAF echelle spectrum in multispec format from a FITS file
    
    Can read most multispec formats including linear, log, cubic spline,
    Chebyshev or Legendre dispersion spectra
    
    If reform is true, a single spectrum dimensioned 4,1,NWAVE is returned
    as 4,NWAVE (this is the default.)  If reform is false, it is returned as
    a 3-D array.
    """

    fh = fits.open(fitsfile)
    try:
        header = fh[0].header
        flux = fh[0].data
    finally:
        fh.close()
    temp = flux.shape
   
    nwave = temp[-1]
    if len(temp) == 1:
        nspec = 1
    else:
        nspec = temp[-2]

    # first try linear dispersion
    has_linear_keywords = all(k in header for k in ['crval1', 'crpix1', 'cd1_1', 'ctype1'])
    if has_linear_keywords:
        crval1 = header['crval1']
        crpix1 = header['crpix1']
        cd1_1 = header['cd1_1']
        ctype1 = header['ctype1']
        if ctype1.strip() == 'LINEAR':
            wavelen = np.zeros((nspec, nwave), dtype=float)
            ww = (np.arange(nwave, dtype=float) + 1 - crpix1) * cd1_1 + crval1
            for i in range(nspec):
                wavelen[i, :] = ww
            # handle log spacing too
            dcflag = header['dc-flag']
            if dcflag == 1:
                wavelen = 10.0 ** wavelen
                if not quiet:
                    print('Dispersion is linear in log wavelength')
            elif dcflag == 0:
                if not quiet:
                    print('Dispersion is linear')
            else:
                raise ValueError('Dispersion not linear or log (DC-FLAG=%s)' % dcflag)

            if nspec == 1 and reform:
                # get rid of unity dimensions
                flux = np.squeeze(flux)
                wavelen.shape = (nwave,)
            return {'flux': flux, 'wavelen': wavelen, 'header': header, 'wavefields': None}

    # get wavelength parameters from multispec keywords
    try:
        wat2 = header['wat2_*']
        count = len(wat2)
    except KeyError:
        raise ValueError('Cannot decipher header, need either WAT2_ or CRVAL keywords')

    # concatenate them all together into one big string
    watstr = []
    for i in range(len(wat2)):
        # hack to fix the fact that older pyfits versions (< 3.1)
        # strip trailing blanks from string values in an apparently
        # irrecoverable way
        # v = wat2[i].value
        v = wat2[i]
        v = v + (" " * (68 - len(v)))  # restore trailing blanks
        watstr.append(v)
    watstr = ''.join(watstr)
    
    # find all the spec#="..." strings
    specstr = [''] * nspec
    for i in range(nspec):
        sname = 'spec' + str(i + 1)
        p1 = watstr.find(sname)
        p2 = watstr.find('"', p1)
        p3 = watstr.find('"', p2 + 1)
        if p1 < 0 or p1 < 0 or p3 < 0:
            raise ValueError('Cannot find ' + sname + ' in WAT2_* keyword')
        specstr[i] = watstr[p2 + 1:p3]

    wparms = np.zeros((nspec, 9), dtype=float)
    w1 = np.zeros(9, dtype=float)
    for i in range(nspec):
        w1 = np.asarray(specstr[i].split(), dtype=float)
        wparms[i, :] = w1[:9]
        if w1[2] == -1:
            raise ValueError('Spectrum %d has no wavelength calibration (type=%d)' %
                             (i + 1, w1[2]))
            # elif w1[6] != 0:
            #    raise ValueError('Spectrum %d has non-zero redshift (z=%f)' % (i+1,w1[6]))

    wavelen = np.zeros((nspec, nwave), dtype=float)
    wavefields = [None] * nspec
    for i in range(nspec):
        # if i in skipped_orders:
        #    continue
        verbose = (not quiet) and (i == 0)
        if wparms[i, 2] == 0 or wparms[i, 2] == 1 or wparms[i, 2] == 2:
            # simple linear or log spacing
            wavelen[i, :] = np.arange(nwave, dtype=float) * wparms[i, 4] + wparms[i, 3]
            if wparms[i, 2] == 1:
                wavelen[i, :] = 10.0 ** wavelen[i, :]
                if verbose:
                    print('Dispersion is linear in log wavelength')
            elif verbose:
                print('Dispersion is linear')
        else:
            # non-linear wavelengths
            wavelen[i, :], wavefields[i] = nonlinearwave(nwave, specstr[i],
                                                         verbose=verbose)
        wavelen *= 1.0 + wparms[i, 6]
        if verbose:
            print("Correcting for redshift: z={}".format(wparms[i, 6]))
    if nspec == 1 and reform:
        # get rid of unity dimensions
        flux = np.squeeze(flux)
        wavelen.shape = (nwave,)
    return {'flux': flux, 'wavelen': wavelen, 'header': header, 'wavefields': wavefields}


list_of_spectra_errors = []
list_of_fitting_results = [["Name", "Name TAC","FWHM", "S/N raw","S/N tac","reduced chi2", "Quality","Wvlg solution 1","Wvlg solution 2"]]

def invoke_molecfit(i):
    
        
    if hasattr(list_of_spectra, "iloc"):
        path = str(list_of_spectra["path"].iloc[i])
    else:
        path = list_of_spectra["path"][i]

    """
    if i < 2:
        Completion_successfull = False
        #skip the rest of the for loop
        print(path,", exit 1")
        return Completion_successfull , path+" #Problem: Fileformat not known"
    """    
    
    print("[INFO]\t Preparing "+path.split("/")[-1])
    output_dir_i = _build_output_dir_for_spectrum(i, path)
    ###
    ### get spectrum information
    ### Goal: identify telescope and get wavelength and flux data
    ###
    try:
        instrument_family = _detect_instrument_family(path)
        if instrument_family in ["NIRPS"]:
            print('Assuming spectra are defined in AIR and BARYCENTRIC RF: moving from BARYCENTRIC to EARTH, staying in AIR')
            # NIRPS S1D products store explicit WAVE/WAVE_AIR columns in the
            # SPECTRUM table and are delivered in the barycentric frame.
            wavelengths_array = np.asarray(get_data(path, "WAVE_AIR"), dtype=float)
            berv_m_per_s = _get_csv_berv_m_per_s(i, path)
            wavelengths_array = wavelengths_array / (1.0 + berv_m_per_s / SPEED_OF_LIGHT_M_S)
        elif instrument_family in ['HARPS']:
            print('Assuming spectra are defined in AIR and BARYCENTRIC RF: moving from BARYCENTRIC to EARTH, staying in AIR')
            # NIRPS S1D products store explicit WAVE/WAVE_AIR columns in the
            # SPECTRUM table and are delivered in the barycentric frame.
            wavelengths_array = np.asarray(get_data(path, "WAVE"), dtype=float)
            berv_m_per_s = _get_csv_berv_m_per_s(i, path)
            wavelengths_array = wavelengths_array / (1.0 + berv_m_per_s / SPEED_OF_LIGHT_M_S)


        else:
            wavelengths_array = np.asarray(get_data(path, "WAVE"), dtype=float)
        flux_array = np.asarray(get_data(path, "FLUX"), dtype=float)
        flux_error_array = np.asarray(get_data(path, "ERR"), dtype=float)
        lambda_min = float(np.nanmin(wavelengths_array))
        lambda_max = float(np.nanmax(wavelengths_array))
    except Exception as e:
        print("[DEBUG] invoke_molecfit generic spectrum parsing failed:", e)
        Completion_successfull = False
        return Completion_successfull, path+" #Problem: Fileformat not known"

    # Always provide an explicit ASCII table to molecfit for robust WAVE/FLUX/ERR/QUAL handling.
    try:
        wave_clean, flux_clean, err_clean, qual_mask, qc_stats = _sanitize_spectrum_for_molecfit(
            wavelengths_array,
            flux_array,
            flux_error_array,
        )
    except Exception as e:
        Completion_successfull = False
        return Completion_successfull, path + " #Problem: invalid WAVE/FLUX/ERR values ({}).".format(e)

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
    if normalize_s1d_before_molecfit:
        flux_for_fit, err_for_fit, continuum_used = _normalize_spectrum_for_molecfit(
            wave_clean,
            flux_clean,
            err_clean,
            qual_mask,
            window_pixels=normalization_window_pixels,
            percentile=normalization_percentile,
        )
        # also ensure all values are strictly positive
        flux_for_fit += np.nanmax(np.abs(flux_for_fit))+1.0
        print(
            "[INFO]\t Pre-normalized {} for molecfit (window={}, percentile={}%).".format(
                os.path.basename(path),
                int(normalization_window_pixels),
                float(normalization_percentile),
            )
        )

    save_array_as_ASCII_with_qual(path, wave_clean, flux_for_fit, err_for_fit, qual_mask)
    use_ascii_input = True

    #get the wavelength units
    wlgtomicron = get_wlgtomicron(path, use_ascii_input)
                            
    #convert wavelengths to microns
    lambda_min = float(np.nanmin(wave_clean)) * wlgtomicron
    lambda_max = float(np.nanmax(wave_clean)) * wlgtomicron
    
    ###
    ### Name for the generic template paramete file
    ###
    #name of the temporary arameter file
    name_temp_par = "/temp_"+str(i)+".par"
    ###
    ### Input Paramteres
    ###

    #replace the input spectrum
    #Notice: this creates a new copy of the parameter file
        
    if use_ascii_input:
        replace(path_to_program+name_gen_par,path_to_program+name_temp_par, "#path_to_fits", \
            path.replace(".fits",".dat"))
    else:
        replace(path_to_program+name_gen_par,path_to_program+name_temp_par, "#path_to_fits", path)

    # Apply molecule setup for the current instrument/channel.
    if instrument_family not in molecfit_molecules_by_instrument:
        raise ValueError("No molecule configuration for instrument family '{}'".format(instrument_family))
    molec_cfg = molecfit_molecules_by_instrument[instrument_family]
    _set_parfile_value(path_to_program+name_temp_par, "list_molec", molec_cfg["list_molec"])
    _set_parfile_value(path_to_program+name_temp_par, "fit_molec", molec_cfg["fit_molec"])
    _set_parfile_value(path_to_program+name_temp_par, "relcol", molec_cfg["relcol"])
    #replace the inclusion range
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "#wrange_include", \
    path_to_program+"/include_"+str(i)+".dat")
    
    #replace the exclusion range
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "#wrange_exclude", \
            path_to_program+"/exclude_"+str(i)+".dat")
        
    #replace the exclusion range pixels
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "#prange_exclude", "none")
        
    #replace the wlgtomicron
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "#wlgtomicron", \
            str(wlgtomicron))
            
            
            
    ###
    ### Output Parameters
    ###
            
    #define the output directory
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "#output_dir", \
            output_dir_i)
            
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "#output_name", \
            "Spectrum_"+str(i))

    # Inject weather/site parameters from generic ESO-like headers.
    mjd_obs = _first_header_value(path, ["MJD-OBS", "HIERARCH CARACAL MJD-OBS"])
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "obsdate#:", "obsdate: "+str(mjd_obs))
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "obsdate_key: MJD-OBS", "obsdate_key: NONE")

    utc_sec = _first_header_value(path, ["UTC"])
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "utc#:", "utc: "+str(utc_sec))
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "utc_key: UTC", "utc_key: NONE")

    telalt = _first_header_value(path, ["HIERARCH ESO TEL ALT", "HIERARCH CAHA TEL POS EL_START"])
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "telalt#:", "telalt: "+str(telalt))
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "telalt_key: ESO TEL ALT", "telalt_key: NONE")

    geoelev = _first_header_value(path, ["HIERARCH ESO TEL GEOELEV", "HIERARCH CAHA TEL GEOELEV"])
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "geoelev#:", "geoelev: "+str(geoelev))
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "geoelev_key: ESO TEL GEOELEV", "geoelev_key: NONE")

    longitude = _first_header_value(path, ["HIERARCH ESO TEL GEOLON", "HIERARCH CAHA TEL GEOLON"])
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "longitude#:", "longitude: "+str(longitude))
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "longitude_key: ESO TEL GEOLON", "longitude_key: NONE")

    latitude = _first_header_value(path, ["HIERARCH ESO TEL GEOLAT", "HIERARCH CAHA TEL GEOLAT"])
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "latitude#:", "latitude: "+str(latitude))
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "latitude_key: ESO TEL GEOLAT", "latitude_key: NONE")

    temp, mir_temp, pressure, humid = get_weather_data(path)
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "temp#:", "temp: "+str(temp))
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "temp_key: ESO TEL AMBI TEMP", "temp_key: NONE")
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "pres#:", "pres: "+str(pressure))
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "pres_key: ESO TEL AMBI PRES START", "pres_key: NONE")
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "rhum#:", "rhum: "+str(humid))
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "rhum_key: ESO TEL AMBI RHUM", "rhum_key: NONE")
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "m1temp#:", "m1temp: "+str(mir_temp))
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "m1temp_key: ESO TEL TH M1 TEMP", "m1temp_key: NONE")
        
        
    ###
    ###Alter the exclusion & inclusion ASCII files
    ###
    include_windows_template, exclude_windows_template = _get_fit_window_templates(instrument_family)

    Exluding = []
    """
    file containing the exclusion region
    (regions of intrinsic emission from science
    target)
    """
    exclude_file_path = path_to_program+"/exclude_"+str(i)+".dat"
    include_file_path = path_to_program+"/include_"+str(i)+".dat"

    # Keep only exclusion windows fully inside the observed wavelength range.
    for lo, hi in exclude_windows_template:
        if lo >= lambda_min and hi <= lambda_max:
            Exluding.append([lo, hi])

    np.savetxt(exclude_file_path,Exluding)
    
    Including = get_inclusion_region(
        lambda_min,
        lambda_max,
        include_windows=include_windows_template,
        instrument_family=instrument_family,
    )
    
    
    """
    file containing he inclusion region
    (regions of telluric absorption in
    the spectrum)
    """
    
    #check, if the inclusion file has enough regions, else write down the file and look at it manually
    
    if Including[1]: 
        np.savetxt(include_file_path,Including[0])
        
    else:
        Completion_successfull = False
        return Completion_successfull , path+"#Problem: Inclusion Region too small"
    ###
    ### Include own LSF File
    ###
    Own_LSF_input = False
    
    if Own_LSF_input:
        path_kernel_file = path_to_program+"/own_input_kernel.dat"
        middle_FWHM = create_LSF_file(path, False)
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "kernel_file: none", \
                "kernel_file: "+path_kernel_file)
    else:
        #replace the resolution
        try:
            res_FWHM, var_wlg = get_FWHM(path,lambda_min,lambda_max,wlgtomicron)
        except Exception as e:
            Completion_successfull = False
            #skip the rest of the for loop
            remove(path_to_program+name_temp_par)
            remove(exclude_file_path)
            remove(include_file_path)
            return Completion_successfull , path+f" #Problem: {e}"

        print("[INFO]\t Resolution seed (pixels) for {}: {:.4f} (fit_res_gauss={}, varkern={})".format(
            os.path.basename(path),
            float(res_FWHM),
            int(bool(fit_resolution_in_molecfit)),
            int(bool(allow_variable_kernel and var_wlg)),
        ))
        
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "res_gauss: #res_gauss", \
                "res_gauss: "+str(res_FWHM))

        if not fit_resolution_in_molecfit:
            replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "fit_res_gauss: 1", "fit_res_gauss: 0")

        _set_parfile_value(
            path_to_program+name_temp_par,
            "varkern",
            str(int(bool(allow_variable_kernel and var_wlg))),
        )
    ###
    ### Invoke Molecfit
    ###
    
    print("[INFO]\t Running Molecfit")
    os.system(path_to_molecfit+"molecfit " + path_to_program +name_temp_par)

    # Run calctrans on full spectral range: keep the fit constrained by
    # wrange_include/wrange_exclude, but do not limit transmission output.
    name_temp_par_calctrans = "/temp_"+str(i)+"_calctrans.par"
    shutil.copy(path_to_program+name_temp_par, path_to_program+name_temp_par_calctrans)
    replace(path_to_program+name_temp_par_calctrans,path_to_program+name_temp_par_calctrans,
            "wrange_include: "+include_file_path, "wrange_include: none")
    replace(path_to_program+name_temp_par_calctrans,path_to_program+name_temp_par_calctrans,
            "wrange_exclude: "+exclude_file_path, "wrange_exclude: none")
    replace(path_to_program+name_temp_par_calctrans,path_to_program+name_temp_par_calctrans,
            "prange_exclude: none", "prange_exclude: none")
    os.system(path_to_molecfit+"calctrans " + path_to_program +name_temp_par_calctrans)

    ###
    ### Save the results in a seperate directory
    ###
    
    save_result(path,output_dir_i, False, False, output_index=i)

    # Persist fit-range files in output folder for notebook diagnostics.
    if os.path.exists(include_file_path):
        shutil.copy(include_file_path, output_dir_i+"/include_"+str(i)+".dat")
    if os.path.exists(exclude_file_path):
        shutil.copy(exclude_file_path, output_dir_i+"/exclude_"+str(i)+".dat")
    
    
    #save the results for FWHM in a list
    
    
    ##get the signal to noise
    if use_ascii_input:
        path_final_file = output_dir_i+"/"+path.split("/")[-1].replace(".fits", "_TAC.dat")
        waves_targ = np.genfromtxt(path_final_file)[:,0]
        flux_targ = np.genfromtxt(path_final_file)[:,1]
        if np.shape(np.genfromtxt(path_final_file))[1]>5:
            flux_targ_tac = np.genfromtxt(path_final_file)[:,4]
        else:
            flux_targ_tac = np.genfromtxt(path_final_file)[:,3]
        
        s_n_raw = get_signal_to_noise(waves_targ,flux_targ)
        s_n_tac = get_signal_to_noise(waves_targ,flux_targ_tac)
    else:
        path_final_file = path_to_final_results+"/"+path.split("/")[-1].replace(".fits", "_TAC.fits")
        
        s_n_raw = get_signal_to_noise(get_data(path,"WAVE"),get_data(path,"FLUX"))
        s_n_tac = get_signal_to_noise(get_data(path_final_file,"WAVE"),get_data(path_final_file,"tacflux"))
    
    ##do a rough quality check
    master_trans_path = _get_master_transmission_path(path)
    if master_trans_path is not None and os.path.exists(master_trans_path):
        Master_trans_spec = np.genfromtxt(master_trans_path)
        wave_mas = Master_trans_spec[:,0]
        trans_mas = Master_trans_spec[:,1]

        waves_targ, trans_targ = get_trans_spectrum(path_final_file, use_ascii_input)
        O_2, H2_O = scales(1000*wlgtomicron*waves_targ,wave_mas,trans_targ,trans_mas)

        O_2_accept = False
        H2_O_accept = False
        if (abs(np.mean(O_2)-1)<0.2 and test_region(O_2)) or len(O_2)==0:
            O_2_accept = True
        if test_region(H2_O)or len(H2_O)==0:
            H2_O_accept = True

        if H2_O_accept and O_2_accept:
            quality_label = 0
        elif H2_O_accept and not O_2_accept:
            quality_label = 1
        elif not H2_O_accept and  O_2_accept:
            quality_label = 2
        elif not H2_O_accept and not O_2_accept:
            quality_label = 3
    else:
        # -1 means "not evaluated" (no matching master transmission template).
        quality_label = -1
        
    
    ##get other parameters
    red_chi2, wvlg_sol1, wavlg_sol2 = get_results(output_dir_i,"Spectrum_"+str(i))
    if Own_LSF_input:
        FWHM_result = middle_FWHM

    else:
        FWHM_result = get_FWHM_result(output_dir_i,"Spectrum_"+str(i))
    
    returner = [path.split("/")[-1].replace(".fits"," "), \
                path.split("/")[-1].replace(".fits","_TAC.fits"),\
                FWHM_result,s_n_raw,s_n_tac, red_chi2,quality_label,wvlg_sol1,wavlg_sol2]
        
        
        
    remove(path_to_program+name_temp_par)
    if os.path.exists(path_to_program+name_temp_par_calctrans):
        remove(path_to_program+name_temp_par_calctrans)
    remove(exclude_file_path)
    remove(include_file_path)
                
    if Own_LSF_input:
        remove(path_kernel_file)

    Completion_successfull = True
    return Completion_successfull, returner,[path, wlgtomicron, use_ascii_input,output_dir_i,Including[0], i]


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
