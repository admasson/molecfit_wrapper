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

try:
    from PyAstronomy import pyasl
except Exception:
    pyasl = None


#string that contains the current date
#useful to keep an order in the directory sorting
date_string = time.strftime("_%d_%m_%Y")
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
    "UNKNOWN": 74000.0,
}


#<path to final results>
#this directory will be created
path_to_final_results = "/home/amasson/data/molecfit_wrapper/Final_Results_Molecfit/Final_Results"+date_string

#path to the LSF fiel aht will be creayed if OWN_LSF True
path_kernel_file = path_to_program + "/own_input_kernel.dat"


if ".csv" in path_to_list:
    list_of_spectra= pd.read_csv(path_to_list, header=0, names=["path","z"])
else:
    list_of_spectra = np.genfromtxt(path_to_list,dtype=["U150","<f8"],names=["path","z"])



shutil.copy(path_to_gen_par,path_to_program)
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

def _first_header_value(path_to_file, keys, default=None):
    for key in keys:
        try:
            value = get_header_info(path_to_file, key)
            if value is not None:
                return float(value)
        except Exception:
            pass
    return default

def _first_header_raw(path_to_file, keys, default=None):
    for key in keys:
        try:
            value = get_header_info(path_to_file, key)
            if value is not None:
                return value
        except Exception:
            pass
    return default

def _detect_instrument_family(path_to_file):
    instr = str(_first_header_raw(path_to_file, ["INSTRUME"], default="")).upper()
    subsys = str(_first_header_raw(path_to_file, ["SUBSYS"], default="")).upper()
    if "HARPS" in instr:
        return "HARPS"
    if "NIRPS" in instr:
        return "NIRPS"
    if "CARMENES" in instr:
        if "VIS" in subsys:
            return "CARMENES_VIS"
        if "NIR" in subsys:
            return "CARMENES_NIR"
        return "CARMENES_VIS"
    return "UNKNOWN"

def _get_master_transmission_path(path_to_file):
    family = _detect_instrument_family(path_to_file)
    if family in path_master_trans_by_instrument:
        return path_master_trans_by_instrument[family]
    return path_master_trans

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
    crval1 = header.get("CRVAL1")
    cdelt1 = header.get("CDELT1", header.get("CD1_1"))
    crpix1 = header.get("CRPIX1", 1.0)
    naxis1 = header.get("NAXIS1")

    if crval1 is None or cdelt1 is None or naxis1 is None:
        raise KeyError("Missing linear wavelength WCS keywords (CRVAL1/CDELT1(or CD1_1)/NAXIS1)")

    pixels = np.arange(int(naxis1), dtype=float) + 1.0
    return crval1 + (pixels - float(crpix1)) * float(cdelt1)

def _wavelength_from_harps_drs_header(header, data_len=None):
    """
    Build HARPS wavelengths from DRS polynomial coefficients (same approach
    as reduction.py/build_HARPS_wavelengths).
    """
    degree = int(header["HIERARCH ESO DRS CAL TH DEG LL"])
    n_orders = int(header.get("HIERARCH ESO DRS CAL LOC NBO", 72))
    n_pix = 4096

    x = np.empty((degree + 1, n_pix), dtype=np.int64)
    x[0].fill(1)
    x[1] = np.arange(n_pix)
    for i in range(1, degree):
        x[i + 1] = x[i] * x[1]

    coeff = np.reshape(
        [header["HIERARCH ESO DRS CAL TH COEFF LL" + str(i)] for i in range(n_orders * (degree + 1))],
        (n_orders, degree + 1),
    )

    waves_air = np.dot(coeff, x)
    if pyasl is not None:
        waves = pyasl.airtovac2(waves_air)
    else:
        # Keep air wavelengths when PyAstronomy is unavailable.
        waves = waves_air

    waves_flat = np.asarray(waves, dtype=float).reshape(-1)
    if data_len is None:
        return waves_flat

    if waves_flat.size == int(data_len):
        return waves_flat

    raise ValueError(
        "HARPS DRS wavelength size mismatch: built {} values, expected {}".format(
            waves_flat.size,
            int(data_len),
        )
    )

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
    Falls back to per-instrument defaults and wavelength-derived binning.
    """
    instrument_family = _detect_instrument_family(path_to_file)

    resolution = _first_header_value(path_to_file, [
        "SPEC_RES",
        "SPECRES",
        "HIERARCH CARACAL RESOLUTION",
        "HIERARCH CAHA INS SPEC RES",
        "HIERARCH ESO DRS SPEC RES",
        "HIERARCH ESO INS SPEC RES",
        "HIERARCH ESO INS RESOL",
    ], default=None)
    if resolution is None:
        resolution = default_resolution_by_instrument.get(
            instrument_family,
            default_resolution_by_instrument["UNKNOWN"],
        )

    bin_width = _first_header_value(path_to_file, [
        "SPEC_BIN",
        "CDELT1",
        "CD1_1",
        "HIERARCH CARACAL SPEC BIN",
    ], default=None)

    if bin_width is None:
        if waves is None:
            waves = get_data(path_to_file, "WAVE")
        bin_width = _estimate_bin_width_from_wave(waves)

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
        try:
            len(result_data)
        except:
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

        # HARPS-like files: try DRS polynomial wavelength solution first.
        if expr_upper == "WAVE":
            data_len = None
            if hdul[0].data is not None:
                data_len = np.asarray(hdul[0].data).size
            try:
                return _wavelength_from_harps_drs_header(hdul[0].header, data_len=data_len)
            except Exception:
                # HARPS/NIRPS S1D-like files: fallback to linear WCS from header.
                return _wavelength_from_wcs_header(hdul[0].header)

        # Fallback flux for 1D primary-image spectra.
        if expr_upper == "FLUX" and hdul[0].data is not None:
            return np.array(hdul[0].data)

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
        try:
            resol = 7400/float(get_header_info(filepath,"SKYFWHM"))
        except:
            #check, if correct for SOAR
            resol = 5880
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


def get_FWHM(path_to_fits,min_wlg, max_wlg,palomar,wvlng_to_mic):
    """
    Function that determines the FWHM for gaussian kernels (or the kernel in general)
    :param path_to_fits: path to the fits file
    :param min_wlg: the lower wavelength bound in microns
    :param max_wlg: the upper wavelength bound in microns
    :param palomar: boolean, if the spectrum dealing with is palomar or not
    :param wvlng_to_mic: needed for Xshooter, the conversion factor to microns
    
    :return FWHM: the FWHM in pixels of the central wavelength
    :return var: if long spectrum, do variable with wavelength
    """
    #determine 
    if palomar:
        #in unit Angstrom
        try:
            try:
                FWHM_wlg_unit = float(get_header_info(path_to_fits,"SKYFWHM"))
                
            except Exception as e:
                print("[DEBUG] get_FWHM SKYFWHM fallback:", e)
                nam = get_header_info(path_to_fits,"FPA")
                if "DBSP" in nam:
                    FWHM_wlg_unit = 6
            try:
                binsize = get_header_info(path_to_fits,"CDELT1")
            except Exception as e:
                print("[DEBUG] get_FWHM CDELT1 fallback:", e)
                binsize = get_header_info(path_to_fits,"CD1_1")
                if binsize == 1.:
                    try:
                        wat_2 = get_header_info(path_to_fits,"WAT2_001")
                        binsize = float(wat_2.split("\"")[-1].split(" ")[4])
                    except KeyError:
                        pass
                       
            FWHM = (min_wlg + max_wlg)/2/(7400/FWHM_wlg_unit)/(binsize*wvlng_to_mic)
        except Exception as e:
            print("[DEBUG] get_FWHM Palomar-like branch failed, trying DuPont/Keck/SOAR:", e)
             # Is DuPont
            if "DuPont" in get_header_info(path_to_fits,"TELESCOP"):
                FWHM_wlg_unit = 8
                binsize = get_header_info(path_to_fits,"CDELT1")
                FWHM = (min_wlg + max_wlg)/2/(7400/FWHM_wlg_unit)/(binsize*wvlng_to_mic)
            # is Keck
            elif "Keck" in get_header_info(path_to_fits,"TELESCOP"):
                resol = get_header_info(path_to_fits,"SPECRES")
                try:
                    binsize = get_header_info(path_to_fits,"CDELT1")
                except Exception as e:
                    print("[DEBUG] get_FWHM Keck CDELT1 fallback:", e)
                    binsize = get_header_info(path_to_fits,"CD1_1")
                FWHM = (min_wlg + max_wlg)/2 /resol /(binsize*wvlng_to_mic)
            #is SOAR
            else:
                binsize = get_header_info(path_to_fits, "CDELT1")
                resol = 5880
                FWHM = (min_wlg + max_wlg) / 2 / resol / (binsize * wvlng_to_mic)
    else: 
        waves_for_bin = get_data(path_to_fits, "WAVE")
        resol, binsize = _get_resolution_and_binwidth(path_to_fits, waves=waves_for_bin)
        FWHM = (min_wlg + max_wlg)/2 /resol /(binsize*wvlng_to_mic)
    
    #check, if dealing with a longer spectrum (in order to activate variability with wavelength)
    if max_wlg-min_wlg>0.2:
        var = True
    else: 
        var = False
    
    return FWHM, var



#gives array containing good upper and lower bounds for includion regions
telluric_regions = np.loadtxt(path_to_tellurics)

#gives list with important emission lines
emission_lines  = np.loadtxt(path_to_ems_lines)

tell_for_plot = np.loadtxt(path_to_telluric_plotting_reg)

#speed of light in km/s
c=3e5
def Delta_lambda(wavelenght,Delta_v):
    """
    :param wavelength: wavelenth of emission or absorption feature
    :param Delta_v: Doppler Velocity shift, in km/s
    """
    dlambda = wavelenght*Delta_v/c
    
    return dlambda

v = 500 #velocity due to doppler shift
v_broad = 5000 #velocity due to doppler shift (use in case of h_alpha)
def get_inclusion_region(z,wave_min=0,wave_max=4, is_Dupont = False):
    """
    :param z: redshift of galaxy
    :param wave_min: lower bound of spectrum, in Microns
    :param wave_max: upper bound of spectrum, in Microns
    """
    if is_Dupont:
        wave_max = 0.78
    #redden the emission lines to the corresponding redshift to check wether they lie in exclusion zones
    emission_lines_reddened = np.zeros(len(emission_lines))
    emission_lines_reddened = (1+z)*emission_lines
    #print(emission_lines_reddened)
    #save the inclusion zones still possible
    inclusion_region = []
    
    # only take inclusion region within bounds [l_min,l_max] of the spectrum
    # the 0.1 is arbitrary in order to forcome, that the boundary region is not in a inclusion region
    allowed_tellurics = np.where(np.logical_and(wave_min-0.1<telluric_regions[0,:],\
                                                wave_max+0.1>telluric_regions[0,:]))[0]
    
    
    #check, that allowed tellurics is even (i.e. we have always an upper and a lower bound):
    if len(allowed_tellurics)//2 !=0:
            if len(allowed_tellurics) == 1:
                allowed_tellurics = np.array([])
            else:
                lower_bound = abs(telluric_regions[0,:][allowed_tellurics][0]-telluric_regions[0,:][allowed_tellurics][1])
                upper_bound = abs(telluric_regions[0,:][allowed_tellurics][-1]-telluric_regions[0,:][allowed_tellurics][-2])
                if upper_bound<lower_bound:
                    allowed_tellurics = np.delete(allowed_tellurics,0)
                else:
                    allowed_tellurics = np.delete(allowed_tellurics,-1)
    
    #if allowed_tellurics no empyt
    try:
        lower_bound_index = allowed_tellurics[0]
        upper_bound_index = allowed_tellurics[-1]
    except:
        lower_bound_index=0
        upper_bound_index=0
    #go trhough the various broader inclusion regions
    for i in range(int(len(telluric_regions[0,allowed_tellurics])/2)):
        #go through the various possibilities of a inclusion region
        for j in range(len(telluric_regions[:,lower_bound_index+2*i])):
            add = True
            
            """
            for each emission line, check wether the current inclusion region is acceptable. 
            If not, move to the next inclusion region.
            """
            for wavelength in emission_lines_reddened:
                if wavelength == 0.656493*(1+z):
                     D_Lambda = Delta_lambda(wavelength,v_broad)
                else:
                    D_Lambda = Delta_lambda(wavelength,v)
                if telluric_regions[j,lower_bound_index+2*i] - D_Lambda < wavelength and telluric_regions[j,lower_bound_index+2*i+1] + D_Lambda > wavelength:
                    add = False
            
            if telluric_regions[j,lower_bound_index+2*i]<wave_min or telluric_regions[j,lower_bound_index+2*i+1]>wave_max:
                add = False
            if add:
                break
        if add:
            inclusion_region.append([telluric_regions[j,lower_bound_index+2*i],telluric_regions[j,lower_bound_index+2*i+1]])
        
       
        
        #check, if enough fitting regions (at least two) have been selected
    
    acceptable = len(inclusion_region)>=1
    """
    if z == 0.048:
        inclusion_region = [[0.757,0.7695]]
    if z == 0.152363384:
        inclusion_region = [[0.686257, 0.692463]]
    
    """
    #inclusion_region  = [[0.6825,0.695],[0.7585,0.764],[0.81,0.815],[0.91,0.92],[0.935,0.94]]
    #inclusion_region[-1]=[1.26,1.268]
    if len(inclusion_region)==0:
        # bounds-aware fallback windows for optical data
        for lo, hi in ([0.6846, 0.6972], [0.7570, 0.7710], [0.9050, 0.9210]):
            if lo >= wave_min and hi <= wave_max:
                inclusion_region.append([lo, hi])

    # Keep a narrow O2 A fallback only when it lies inside the actual spectrum range.
    if 0.7575 >= wave_min and 0.765 <= wave_max:
        inclusion_region.append([0.7575, 0.765])

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
        #wavelength region could not be determined
        return False

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
        
    #import the telluric fit for inspection
    hdul_fit = fits.open(path_to_results+str(index)+"_"+ date_string+"/Spectrum_"+str(index)+"_fit.fits")
    waves_fit = hdul_fit[1].data["lambda"]
    flux_fit = hdul_fit[1].data["flux"]
    flux_fit_model = hdul_fit[1].data["mflux"]
    hdul_fit.close()
    
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
            if len(np.shape(hdul_file[0].data))>2:
                hdul_file[0].data = np.vstack([hdul_file[0].data[0][0],hdul_file[0].data[1][0], tac_flux])
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
    
    with open(output_dir+"/"+output_name+"_fit.res") as f:
        for line in f:
            if "FWHM of Gaussian in pixels: " in line:
                FWHM = line.split(" ")[-3]
    return FWHM
def get_results(output_dir, output_name):
    """
    A function that reads the reduced chi2 and wavelength solutions result in the .res file
    
    :param output_dir: directory, where the output is saved
    :param output_name: name of the outputted files
    """
    with open(output_dir+"/"+output_name+"_fit.res") as f:
        red_chi2 = 0
        wvlg_sol_1 = 0
        wvlg_sol_2 = 0
        
        for line in f:
            if "Reduced chi2:" in line:
                red_chi2 = line.split(" ")[-1]
            elif "Chip 1, coef 0:" in line:
                wvlg_sol_1 = line.split(" ")[-3]+line.split(" ")[-2]+line.split(" ")[-1]
            elif "Chip 1, coef 1:" in line:
                wvlg_sol_2 = line.split(" ")[-3]+line.split(" ")[-2]+line.split(" ")[-1]
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

def get_weather_data(path_to_file, telescope_SOAR=False, telescope_DuPont = False, telescope_Palomar=False):
    """
    This program extrapolates the weatherdata
    :return temp, pres, hum: 
    """
    if telescope_SOAR:
        temp = get_header_info(path_to_file,"ENVTEM")
    
    elif telescope_DuPont:
        if path_DuPont_weather is None:
            raise ValueError("path_DuPont_weather is not set for DuPont mode")
        weather_data = pd.read_table(path_DuPont_weather,header = 0,names = [1,2,3,4,5,6,7,8,9,10,11,12,13,14])
        weather_time_stps = Time(list(weather_data[3]),scale='utc').unix
        # time of observations
        t_obs  = Time(get_header_info(path_to_file,"DATE"),format='isot', scale='utc')
        idx = abs(weather_time_stps-t_obs.unix).argmin()
        
        #convert to Fahrenheid
        temp = (weather_data[11][idx]-32)*5./9.
        mirror_temp = temp
        
        rhum = weather_data[12][idx]
        
        pres = weather_data[10][idx] * 33.86389
        
    elif telescope_Palomar:
        if path_p60 is None or path_p200 is None:
            raise ValueError("path_p60/path_p200 are not set for Palomar mode")
        height_palomar = 1712
        try:
            t_obs  = Time(get_header_info(path_to_file,"DATE-OBS"),format='isot', scale='utc')
        except:
            t_obs  = Time(get_header_info(path_to_file,"UTSHUT"),format='isot', scale='utc')
        
        weather_p60=np.genfromtxt(path_p60, skip_footer=True, delimiter=',', dtype = None, names=True)
        weather_p200=np.genfromtxt(path_p200, skip_footer=True, delimiter=',', dtype = None, names=True)
    
        
        #get index of t_obs in list
        idx_p60 = abs(weather_p60["unix_epoch"]-t_obs.unix).argmin()
        idx_p200 = abs(weather_p200["unix_epoch"]-t_obs.unix).argmin()
        
        #outside_temperature
        temp = weather_p200["outside_airtemp"][idx_p200]
        #mirror temperature
        mirror_temp = weather_p200["primary_mirrortemp"][idx_p200]
        
        
        
        #get the humidity
        
        rhum = weather_p60["rh_out"][idx_p60]
        
        #get the humidity
        """
        In the code it looks like there's a simple conversion factor of 186.2 mbar to get it to sea level. 
        So if you take the reported numbers and subtract 186.2, you should get the actual pressure at the 60in telescope.
        """
        pres = weather_p60["baro_pressure"][idx_p60]-186.2

    else:
        # Generic ESO/HARPS/NIRPS/CARMENES fallback (no external weather file)
        temp = _first_header_value(path_to_file, [
            "HIERARCH ESO TEL AMBI TEMP", "ESO TEL AMBI TEMP", "TEL AMBI TEMP",
            "HIERARCH CAHA GEN AMBI TEMPERATURE"
        ], default=10.0)

        pres_start = _first_header_value(path_to_file, [
            "HIERARCH ESO TEL AMBI PRES START", "ESO TEL AMBI PRES START", "TEL AMBI PRES START"
        ], default=None)
        pres_end = _first_header_value(path_to_file, [
            "HIERARCH ESO TEL AMBI PRES END", "ESO TEL AMBI PRES END", "TEL AMBI PRES END"
        ], default=None)
        if pres_start is not None and pres_end is not None:
            pres = 0.5 * (pres_start + pres_end)
        elif pres_start is not None:
            pres = pres_start
        elif pres_end is not None:
            pres = pres_end
        else:
            pres = _first_header_value(path_to_file, [
                "HIERARCH ESO TEL AMBI PRES", "ESO TEL AMBI PRES", "TEL AMBI PRES",
                "HIERARCH CAHA GEN AMBI PRESSURE"
            ], default=743.0)

        rhum = _first_header_value(path_to_file, [
            "HIERARCH ESO TEL AMBI RHUM", "ESO TEL AMBI RHUM", "TEL AMBI RHUM",
            "HIERARCH CAHA GEN AMBI RHUM"
        ], default=20.0)

        m1 = _first_header_value(path_to_file, [
            "HIERARCH ESO TEL TH M1 TEMP", "ESO TEL TH M1 TEMP", "TEL TH M1 TEMP"
        ], default=temp)
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
        
        #check, if region actually included in target spectrum
        try:
            if not index_range[0]:
                continue
        except:
            pass
        
        O_2_scales.append(minimize(residual_scaling,x0=1, args=(waves_target[index_range], trans_target[index_range],\
                                      waves_master[index_range_master],trans_master[index_range_master])).x)
    
    #the H2_O region
    for i in range(len(H_2O_region)):
        wave_min = H_2O_region[i][0]
        wave_max = H_2O_region[i][1]
        
        index_range = np.where(np.logical_and(waves_target>wave_min, waves_target<wave_max))
        index_range_master = np.where(np.logical_and(waves_master>wave_min, waves_master<wave_max))
        
        try:
            if not index_range[0]:
                continue
        except:
            pass
        H2O_scales.append(minimize(residual_scaling,x0=1, args=(waves_target[index_range], trans_target[index_range],\
                                      waves_master[index_range_master],trans_master[index_range_master])).x)
    
    return np.array(O_2_scales), np.array(H2O_scales)


def get_trans_spectrum(path_to, telescope_Palomar = False, telescope_SOAR=False, is_DuPont = False, is_Keck = False):
    """
    :param path: path to the fits file
    :return waves: wavelength array
    :return trans_spec: transmission array
    """
    #import of the data
    
    #different procedure for Palomar or 
    if telescope_Palomar or telescope_SOAR or is_DuPont or is_Keck:
        data_output = np.genfromtxt(path_to)
        flux = data_output[:,1]
        if telescope_Palomar:
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


def save_array_as_ASCII(path_to_fits_file, wave_values, flux_values, flux_err):
    """
    :param path_to_fits_file: path to the fits file 
    :param wave_values: list/array containing the wavelength values
    :param flux_values: list/array containing the flux values
    """
    #save data into a single array
    table = np.zeros((len(wave_values),3))
    table[:,0] = wave_values
    table[:,1] = flux_values
    table[:,2] = flux_err
    
    #save the array
    np.savetxt(path_to_fits_file.replace(".fits",".dat"),table)

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
    try:
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
            dcflag = header.get('dc-flag', 0)
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
    except KeyError:
        pass

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
list_of_fitting_results = [["Name", "Name TAC","Redshift","FWHM", "S/N raw","S/N tac","reduced chi2", "Quality","Wvlg solution 1","Wvlg solution 2"]]

def invoke_molecfit(i):
    
        
    path = list_of_spectra["path"][i]
    """
    if i < 2:
        Completion_successfull = False
        #skip the rest of the for loop
        print(path,", exit 1")
        return Completion_successfull , path+" #Problem: Fileformat not known"
    """    
    print("[INFO]\t Preparing "+path.split("/")[-1])
    ###
    ### get spectrum information
    ### Goal: identify telescope and get wavelength and flux data
    ###
    try:
        lambda_min = get_header_info(path,"WAVELMIN")
        lambda_max = get_header_info(path,"WAVELMAX")
        is_Palomar = False
        is_SOAR = False
        is_DuPont = False
        is_Keck = False
        is_Palomar_no_err = False
    except Exception as e:
        print("[DEBUG] invoke_molecfit WAVELMIN/WAVELMAX branch failed:", e)
        try:
            # check, if dealing with SOAR or Palomar
            is_Palomar = False
            is_DuPont = False
            is_SOAR = False
            is_Keck = False
            is_Palomar_no_err = False
            try:
                if "SOAR" in get_header_info(path, "TELESCOP"):
                    is_SOAR = True
                elif "DuPont" in get_header_info(path, "TELESCOP"):
                     is_DuPont = True
                elif "Keck" in get_header_info(path, "TELESCOP"):
                     is_Keck = True
            except Exception as e:
                print("[DEBUG] invoke_molecfit TELESCOP classifier failed (branch 1):", e)
                pass

            # this is, how to deal with Palomar Fits File
            size = get_header_info(path, "NAXIS1")
            coord_ref_pixel = get_header_info(path, "CRVAL1")
            try:
                wave_pixel = get_header_info(path, 'CDELT1')  # CDELT1
            except KeyError:
                wave_pixel = get_header_info(path, 'CD1_1')  # CDELT1
            ref_pixel = get_header_info(path, 'CRPIX1')

            # calculate the wavelength array and import the flux values
            wstart = get_wstart(ref_pixel, coord_ref_pixel, wave_pixel)
            wavelengths_array = get_wavelength(wstart, wave_pixel, size)    
            flux_array = get_data(path, 0, 0)
            if get_header_info(path,"BUNIT") == "erg/cm2/s/Hz":
                flux_array = 3e18*flux_array/wavelengths_array**2
            if not is_SOAR and not is_DuPont and not is_Keck:
                flux_error_array = get_data(path, 1, 0)
                if get_header_info(path,"BUNIT") == "erg/cm2/s/Hz":
                    flux_error_array = 3e18*flux_error_array/wavelengths_array**2
            lambda_min = wavelengths_array[0]
            lambda_max = wavelengths_array[-1]

            telescope_name = str(_first_header_raw(path, ["TELESCOP"], default=""))
            instrument_name = str(_first_header_raw(path, ["INSTRUME"], default=""))
            if (not is_SOAR and not is_DuPont and not is_Keck and
                ("Palomar" in telescope_name or "P200" in telescope_name or "P60" in telescope_name or "DBSP" in instrument_name)):
                is_Palomar = True

        except Exception as e:
            print("[DEBUG] invoke_molecfit linear-header parsing failed, trying multispec:", e)
            try:
                is_Palomar = False
                is_DuPont = False
                is_SOAR = False
                is_Keck = False
                is_Palomar_no_err = False
                try:
                    if "SOAR" in get_header_info(path, "TELESCOP"):
                        is_SOAR = True
                    elif "DuPont" in get_header_info(path, "TELESCOP"):
                        is_DuPont = True
                    elif "Keck" in get_header_info(path, "TELESCOP"):
                        is_Keck = True
                except Exception as e:
                    print("[DEBUG] invoke_molecfit TELESCOP classifier failed (branch 2):", e)
                    pass
                load_spec = readmultispec(path)
                wavelengths_array = load_spec["wavelen"]
                flux_array = load_spec["flux"]
                if len(flux_array)<10:
                    flux_array = flux_array[0]
                if get_header_info(path,"BUNIT") == "erg/cm2/s/Hz":
                    flux_array = 3e18*flux_array/wavelengths_array**2
                if not is_SOAR and not is_DuPont and not is_Keck:
                    flux_error_array = get_data(path, 1, 0)
                    if get_header_info(path,"BUNIT") == "erg/cm2/s/Hz":
                        flux_error_array = 3e18*flux_error_array/wavelengths_array**2
                lambda_min = wavelengths_array[0]
                lambda_max = wavelengths_array[-1]
                telescope_name = str(_first_header_raw(path, ["TELESCOP"], default=""))
                instrument_name = str(_first_header_raw(path, ["INSTRUME"], default=""))
                if (not is_SOAR and not is_DuPont and not is_Keck and
                    ("Palomar" in telescope_name or "P200" in telescope_name or "P60" in telescope_name or "DBSP" in instrument_name)):
                    is_Palomar = True
            except Exception as e:
                print("[DEBUG] invoke_molecfit multispec parsing failed:", e)
                Completion_successfull = False
                #skip the rest of the for loop
                print(path, " exit 2")
                return Completion_successfull , path+" #Problem: Fileformat not known"
    if is_Palomar:  
        data = fits.getdata(path)
        if len(data)<2 or len(data)>10:
            is_Palomar_no_err = True

    is_generic_ascii = False
    if not is_Palomar and not is_SOAR and not is_DuPont and not is_Keck:
        # For HARPS/NIRPS/CARMENES-like S1D files, writing a temporary ASCII
        # avoids molecfit crashes when FITS WAVE extension is absent.
        try:
            wavelengths_array = get_data(path, "WAVE")
            flux_array = get_data(path, "FLUX")
            try:
                flux_error_array = get_data(path, "ERR")
            except Exception:
                flux_error_array = np.maximum(np.abs(np.asarray(flux_array, dtype=float)) * 0.01, 1e-20)
            save_array_as_ASCII(path, wavelengths_array, flux_array, flux_error_array)
            is_generic_ascii = True
        except Exception as e:
            print("[DEBUG] invoke_molecfit generic ASCII conversion failed:", e)

    if is_Palomar and not is_Palomar_no_err:
        save_array_as_ASCII(path,wavelengths_array,flux_array, flux_error_array)
    if is_SOAR or is_DuPont or is_Keck or is_Palomar_no_err:
        save_array_as_ASCII_no_err(path,wavelengths_array,flux_array)

    use_ascii_input = is_Palomar or is_SOAR or is_DuPont or is_Keck or is_generic_ascii

    #get the wavelength units
    wlgtomicron = get_wlgtomicron(path, use_ascii_input)
                            
    #in this case, the units could not be determined
    if wlgtomicron == False:
        Completion_successfull = False
        return Completion_successfull , path+" #Problem: Wavelenght Unit could not be determined"

    #convert wavelengths to microns
    lambda_min *= wlgtomicron
    lambda_max *= wlgtomicron
    
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
        
        #replace the molecule info
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "list_molec: H2O O2", \
                "#list_molec: H2O O2")
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "#list_molec: H2O CO2 CO CH4 O2", \
                "list_molec: H2O CO2 CO CH4 O2")
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "fit_molec: 1 1", \
                "#fit_molec: 1 1")
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "# fit_molec: 1 0 0 0 1", \
                "fit_molec: 1 0 0 0 1")
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "relcol: 1.0 1.0", \
                "relcol: 1.0 1.06 1.0 1.0 1.0")

    if is_generic_ascii:
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "columns: WAVE FLUX ERR QUAL", \
                "columns: WAVE FLUX ERR NULL")
    if is_DuPont:
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "list_molec: H2O CO2 CO CH4 O2", \
                "list_molec: O2")
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "fit_molec: 1 0 0 0 1", \
                "fit_molec: 1")
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "relcol: 1.0 1.06 1.0 1.0 1.0", \
                "relcol: 1.0")
        
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
            path_to_results+str(i)+"_"+ date_string)
            
    replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "#output_name", \
            "Spectrum_"+str(i))

    # For generic non-Palomar formats (e.g., HARPS/NIRPS/CARMENES), inject
    # weather and site parameters explicitly to avoid keyword mismatches.
    if not is_Palomar and not is_SOAR and not is_DuPont and not is_Keck:
        mjd_obs = _first_header_value(path, ["MJD-OBS", "HIERARCH CARACAL MJD-OBS"], default=None)
        if mjd_obs is None:
            date_obs_generic = _first_header_raw(path, ["DATE-OBS"], default=None)
            if date_obs_generic is not None:
                mjd_obs = Time(str(date_obs_generic), format="isot").mjd
        if mjd_obs is not None:
            replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "obsdate#:", "obsdate: "+str(mjd_obs))
            replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "obsdate_key: MJD-OBS", "obsdate_key: NONE")

        utc_sec = _first_header_value(path, ["UTC"], default=None)
        if utc_sec is None and mjd_obs is not None:
            utc_sec = int((mjd_obs - int(mjd_obs)) * 24 * 3600)
        if utc_sec is not None:
            replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "utc#:", "utc: "+str(utc_sec))
            replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "utc_key: UTC", "utc_key: NONE")

        telalt = _first_header_value(path, ["HIERARCH ESO TEL ALT", "HIERARCH CAHA TEL POS EL_START"], default=None)
        if telalt is None:
            airmass_gen = _first_header_value(path, ["AIRMASS", "HIERARCH ESO TEL AIRM START"], default=None)
            if airmass_gen is not None and airmass_gen > 0:
                telalt = 90 - 360/(2*np.pi)*float(mpm.asec(float(airmass_gen)))
        if telalt is not None:
            replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "telalt#:", "telalt: "+str(telalt))
            replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "telalt_key: ESO TEL ALT", "telalt_key: NONE")

        geoelev = _first_header_value(path, ["HIERARCH ESO TEL GEOELEV", "HIERARCH CAHA TEL GEOELEV"], default=None)
        if geoelev is not None:
            replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "geoelev#:", "geoelev: "+str(geoelev))
            replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "geoelev_key: ESO TEL GEOELEV", "geoelev_key: NONE")

        longitude = _first_header_value(path, ["HIERARCH ESO TEL GEOLON", "HIERARCH CAHA TEL GEOLON"], default=None)
        if longitude is not None:
            replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "longitude#:", "longitude: "+str(longitude))
            replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "longitude_key: ESO TEL GEOLON", "longitude_key: NONE")

        latitude = _first_header_value(path, ["HIERARCH ESO TEL GEOLAT", "HIERARCH CAHA TEL GEOLAT"], default=None)
        if latitude is not None:
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
    ### Further Changes Necessary for Palomar Spectra
    ###

    if is_Palomar:
        if is_Palomar_no_err:
             replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "columns: WAVE FLUX ERR QUAL", \
                "columns: WAVE FLUX NULL NULL")
        else:
            replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "columns: WAVE FLUX ERR QUAL", \
                "columns: WAVE FLUX ERR NULL")
        
        
        
        #calculate the UTC in sec for palomar
        try:
            MJD = get_header_info(path,"MJD")
            sec = int((MJD-int(MJD))*24*3600)
        except:
            try:
                date_of_obs = get_header_info(path,"DATE-OBS")
            except:
                date_of_obs = get_header_info(path,"UTSHUT")
            MJD = Time(date_of_obs, format = "isot").mjd
            sec = int((MJD - int(MJD)) * 24 * 3600)
        
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "obsdate#:", \
                "obsdate: "+str(MJD))
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "obsdate_key: MJD-OBS", \
                "obsdate_key: NONE")
                
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "utc#:", \
                "utc: "+str(sec))
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "utc_key: UTC", "utc_key: NONE")
                
        #get the telescope altitude angle
        airmass = float(get_header_info(path, "AIRMASS"))
        telalt = 90 - 360/(2*np.pi)*float(mpm.asec(airmass))
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "telalt#:", \
                "telalt: "+str(telalt))
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "telalt_key: ESO TEL ALT", \
                "telalt_key: NONE")
                
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "geoelev#:", \
                "geoelev: 1712.0")
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "geoelev_key: ESO TEL GEOELEV", \
                "geoelev_key: NONE")
                
        #location of telescope
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "longitude#:", "longitude: -116.9")
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "longitude_key: ESO TEL GEOLON", "longitude_key: NONE")
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "latitude#:", "latitude: 33.4")
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "latitude_key: ESO TEL GEOLAT", "latitude_key: NONE")
                
        #get the temperature, pressure and humidity of the location
        temp,mir_temp ,pressure, humid =  get_weather_data(path)
        
        
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "temp#:", "temp: "+str(temp))
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "temp_key: ESO TEL AMBI TEMP", "temp_key: NONE")
                
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "pres#:", "pres: "+str(pressure))
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "pres_key: ESO TEL AMBI PRES START", "pres_key: NONE")
                
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "rhum#:", "rhum: "+str(humid))
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "rhum_key: ESO TEL AMBI RHUM", "rhum_key: NONE")
                
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "m1temp#:", "m1temp: "+str(mir_temp))
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "m1temp_key: ESO TEL TH M1 TEMP", "m1temp_key: NONE")
        
        # Determine the continuum offset
        #cont_term = int(np.log10(np.mean(flux_array)))+2
        # replace(path_to_program + name_temp_par, path_to_program + name_temp_par,"cont_const: 1e-13","cont_const: 1e{}".format(cont_term))
        #replace(path_to_program + name_temp_par, path_to_program + name_temp_par,"cont_const: 1e-13","cont_const: 1e-16")
    
    if is_SOAR:
        # Do seperately, as some of the header key names differ by quite a lot

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "columns: WAVE FLUX ERR QUAL", \
                "columns: WAVE FLUX NULL NULL")
        # calculate the UTC in sec for palomar
        date_of_obs = get_header_info(path,"DATE")
        MJD = Time(date_of_obs, format = "isot").mjd
        sec = int((MJD - int(MJD)) * 24 * 3600)

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "obsdate#:", \
                "obsdate: " + str(MJD))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "obsdate_key: MJD-OBS", \
                "obsdate_key: NONE")

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "utc#:", \
                "utc: " + str(sec))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "utc_key: UTC", "utc_key: NONE")

        telalt = get_header_info(path, "MOUNT_EL")
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "telalt#:", \
                "telalt: " + str(telalt))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "telalt_key: ESO TEL ALT", \
                "telalt_key: NONE")

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "geoelev#:", \
                "geoelev: 2738.0")
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "geoelev_key: ESO TEL GEOELEV", \
                "geoelev_key: NONE")

        # location of telescope
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "longitude#:", "longitude: -70.7")
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "longitude_key: ESO TEL GEOLON",
                "longitude_key: NONE")
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "latitude#:", "latitude: -30.2")
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "latitude_key: ESO TEL GEOLAT",
                "latitude_key: NONE")
        
        # get the temperature, pressure and humidity of the location
        temp = get_header_info(path, "ENVTEM")
        mir_temp = temp
        pressure = get_header_info(path, "ENVPRE")
        humid = get_header_info(path, "ENVHUM")

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "temp#:", "temp: " + str(temp))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "temp_key: ESO TEL AMBI TEMP",
                "temp_key: NONE")

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "pres#:", "pres: " + str(pressure))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "pres_key: ESO TEL AMBI PRES START",
            "pres_key: NONE")

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "rhum#:", "rhum: " + str(humid))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "rhum_key: ESO TEL AMBI RHUM",
            "rhum_key: NONE")

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "m1temp#:",
                "m1temp: " + str(mir_temp))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "m1temp_key: ESO TEL TH M1 TEMP",
                "m1temp_key: NONE")
    
    if is_DuPont:
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "columns: WAVE FLUX ERR QUAL", \
                "columns: WAVE FLUX NULL NULL")
       
        # calculate the UTC in sec for palomar
        date_of_obs = get_header_info(path,"DATE")
        MJD = Time(date_of_obs, format = "isot").mjd
        sec = int((MJD - int(MJD)) * 24 * 3600)

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "obsdate#:", \
                "obsdate: " + str(MJD))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "obsdate_key: MJD-OBS", \
                "obsdate_key: NONE")

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "utc#:", \
                "utc: " + str(sec))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "utc_key: UTC", "utc_key: NONE")

        airmass = float(get_header_info(path, "AIRMASS"))
        telalt = 90 - 360/(2*np.pi)*float(mpm.asec(airmass))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "telalt#:", \
                "telalt: " + str(telalt))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "telalt_key: ESO TEL ALT", \
                "telalt_key: NONE")

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "geoelev#:", \
                "geoelev: 2380.0")
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "geoelev_key: ESO TEL GEOELEV", \
                "geoelev_key: NONE")
        
        # location of telescope
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "longitude#:", "longitude: -70.7")
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "longitude_key: ESO TEL GEOLON",
                "longitude_key: NONE")
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "latitude#:", "latitude: -29.0")
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "latitude_key: ESO TEL GEOLAT",
                "latitude_key: NONE")
        
        # get the temperature, pressure and humidity of the location
        temp,mir_temp ,pressure, humid =  get_weather_data(path, telescope_DuPont = is_DuPont)

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "temp#:", "temp: " + str(temp))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "temp_key: ESO TEL AMBI TEMP",
                "temp_key: NONE")

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "pres#:", "pres: " + str(pressure))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "pres_key: ESO TEL AMBI PRES START",
            "pres_key: NONE")

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "rhum#:", "rhum: " + str(humid))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "rhum_key: ESO TEL AMBI RHUM",
            "rhum_key: NONE")

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "m1temp#:",
                "m1temp: " + str(mir_temp))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "m1temp_key: ESO TEL TH M1 TEMP",
                "m1temp_key: NONE")
    
    if is_Keck:
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "columns: WAVE FLUX ERR QUAL", \
                "columns: WAVE FLUX NULL NULL")
       
        # calculate the UTC in sec for palomar
        date_of_obs = get_header_info(path,"DATE_BEG")
        MJD = Time(date_of_obs, format = "isot").mjd
        sec = int((MJD - int(MJD)) * 24 * 3600)
        
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "obsdate#:", \
                "obsdate: " + str(MJD))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "obsdate_key: MJD-OBS", \
                "obsdate_key: NONE")

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "utc#:", \
                "utc: " + str(sec))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "utc_key: UTC", "utc_key: NONE")
    
        airmass = float(get_header_info(path, "AIRMASS"))
        telalt = 90 - 360/(2*np.pi)*float(mpm.asec(airmass))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "telalt#:", \
                "telalt: " + str(telalt))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "telalt_key: ESO TEL ALT", \
                "telalt_key: NONE")

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "geoelev#:", \
                "geoelev: 4145.0")
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "geoelev_key: ESO TEL GEOELEV", \
                "geoelev_key: NONE")

        # location of telescope
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "longitude#:", "longitude: -155.5 ")
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "longitude_key: ESO TEL GEOLON",
                "longitude_key: NONE")
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "latitude#:", "latitude: 19.8")
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "latitude_key: ESO TEL GEOLAT",
                "latitude_key: NONE")
        
        # get the temperature, pressure and humidity of the location
        temp = get_header_info(path, "WXOUTTMP")
        mir_temp = get_header_info(path,"TUBETEMP")
        pressure = get_header_info(path, "WXPRESS")
        humid = get_header_info(path, "WXOUTHUM")

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "temp#:", "temp: " + str(temp))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "temp_key: ESO TEL AMBI TEMP",
                "temp_key: NONE")

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "pres#:", "pres: " + str(pressure))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "pres_key: ESO TEL AMBI PRES START",
            "pres_key: NONE")

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "rhum#:", "rhum: " + str(humid))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "rhum_key: ESO TEL AMBI RHUM",
            "rhum_key: NONE")

        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "m1temp#:",
                "m1temp: " + str(mir_temp))
        replace(path_to_program + name_temp_par, path_to_program + name_temp_par, "m1temp_key: ESO TEL TH M1 TEMP",
                "m1temp_key: NONE")
        
        
    ###
    ###Alter the exclusion & inclusion ASCII files
    ###
    Exluding = []
    """
    file containing the exclusion region
    (regions of intrinsic emission from science
    target)
    """
    exclude_file_path = path_to_program+"/exclude_"+str(i)+".dat"
    include_file_path = path_to_program+"/include_"+str(i)+".dat"
    np.savetxt(exclude_file_path,Exluding)
    
    Including = get_inclusion_region(list_of_spectra["z"][i], lambda_min,lambda_max, is_DuPont)
    
    
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
        middle_FWHM = create_LSF_file(path, is_Palomar or is_SOAR)
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "kernel_file: none", \
                "kernel_file: "+path_kernel_file)
    else:
        #replace the resolution
        try:
            res_FWHM, var_wlg = get_FWHM(path,lambda_min,lambda_max,is_Palomar or is_SOAR or is_DuPont or is_Keck,wlgtomicron)
        except Exception as e:
            Completion_successfull = False
            #skip the rest of the for loop
            remove(path_to_program+name_temp_par)
            remove(exclude_file_path)
            remove(include_file_path)
            return Completion_successfull , path+f" #Problem: {e}"
        
        replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "res_gauss: #res_gauss", \
                "res_gauss: "+str(res_FWHM))
            
        if var_wlg:
            replace(path_to_program+name_temp_par,path_to_program+name_temp_par, "varkern: 0", "varkern: 1")
    ###
    ### Invoke Molecfit
    ###
    
    print("[INFO]\t Running Molecfit")
    os.system(path_to_molecfit+"molecfit " + path_to_program +name_temp_par)
    os.system(path_to_molecfit+"calctrans " + path_to_program +name_temp_par)

    ###
    ### Save the results in a seperate directory
    ###
    
    save_result(path,path_to_results+str(i)+"_"+date_string, is_SOAR or is_DuPont or is_Keck,is_Palomar_no_err, output_index=i)

    # Persist fit-range files in output folder for notebook diagnostics.
    output_dir_i = path_to_results+str(i)+"_"+date_string
    if os.path.exists(include_file_path):
        shutil.copy(include_file_path, output_dir_i+"/include_"+str(i)+".dat")
    if os.path.exists(exclude_file_path):
        shutil.copy(exclude_file_path, output_dir_i+"/exclude_"+str(i)+".dat")
    
    
    #save the results for FWHM in a list
    
    
    ##get the signal to noise
    if use_ascii_input:
        path_final_file = path_to_results+str(i)+"_"+date_string+"/"+path.split("/")[-1].replace(".fits", "_TAC.dat")
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

        waves_targ, trans_targ = get_trans_spectrum(path_final_file,use_ascii_input, is_SOAR, is_DuPont, is_Keck)
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
    red_chi2, wvlg_sol1, wavlg_sol2 = get_results(path_to_results+str(i)+"_"+ date_string,"Spectrum_"+str(i))
    if Own_LSF_input:
        FWHM_result = middle_FWHM

    else:
        FWHM_result = get_FWHM_result(path_to_results+str(i)+"_"+ date_string,"Spectrum_"+str(i))
    
    returner = [path.split("/")[-1].replace(".fits"," "), \
                path.split("/")[-1].replace(".fits","_TAC.fits"),\
                list_of_spectra["z"][i],FWHM_result,s_n_raw,s_n_tac, red_chi2,quality_label,wvlg_sol1,wavlg_sol2]
        
        
        
    remove(path_to_program+name_temp_par)
    remove(exclude_file_path)
    remove(include_file_path)
                
    if Own_LSF_input:
        remove(path_kernel_file)

    Completion_successfull = True
    return Completion_successfull, returner,[path, wlgtomicron, use_ascii_input,path_to_results+str(i)+"_"+date_string,Including[0], i]


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
