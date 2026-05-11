import os
import re
from astropy.io import fits
import reflex
import numpy as np
import importlib
import copy
try :
    am = importlib.import_module('auto_molecule')
    iu = importlib.import_module('inst_utils')
except ImportError:
    am = importlib.import_module('instruments.auto_molecule')
    iu = importlib.import_module('instruments.inst_utils')
except Exception as e:
    raise Exception(e)
# ------------------------------------------------------------------------------------------
def dataset_chooser_keywords() :
    return 'INSTRUME,OBJECT,INS.PATH,INS.MODE,PRO.CATG,PRODCATG'
'''
# ------------------------------------------------------------------------------------------
# This is a pass-through version of the check_format method
# It is commented out as an example of an "optional" method.
def check_format( files, output_dir ):
    output_files=list()
    for file in files:
        output_files.append(reflex.FitsFile(file.name, file.category, None, file.purposes ))
    return output_files
'''
# ------------------------------------------------------------------------------------------
def set_inst_setup(header=None) :
    """
    Return the instrument setup.
    Generally a string built from various (primary) header keywords
    that identifies the instrument setup of the current science file.
    Different values of inst_setup will indicate when check_idp must set more than just
    file-format variables, usually indicates a change in WL-setup, e.g. VIS --> NIR for XSHOOTER
    """
    if header :
      return header.get('HIERARCH ESO INS MODE', 'UNKNOWN')
    return 'None'
# ------------------------------------------------------------------------------------------
def check_format(files, output_dir):
    """
    Create a molecfit compatible input file from an incompatible file:
    e.g. extract a 1-d spectrum from a 2-d spectrum (for example by extracting the fibre with
    the highest S/N (GIRAFFE))
    """
    '''
    The original HARPS pipeline:
    Identifiable by the existance of the keyword 'HIERARCH ESO DRS VERSION' in the header.

    HARPS IDPs are in AIR_RV WAVELENGTH_FRAME, but this is not (as at v-4.2) supported by
    molecfit/telluriccorr,
    So we'll do it here...
    i.e undo the BERV correction and then transform from air to vacuum
TTYPE1  = 'WAVE    '
TFORM1  = '313061D '
TUTYP1  = 'Spectrum.Data.SpectralAxis.Value'
TUCD1   = 'em.wl;obs.atmos'    / Air wavelength
TUNIT1  = 'angstrom'
TCOMM1  = 'Computed from original WCS information'

TTYPE2  = 'FLUX    '
TFORM2  = '313061E '
TUTYP2  = 'Spectrum.Data.FluxAxis.Value'
TUCD2   = 'phot.flux.density;em.wl;stat.uncalib'
TUNIT2  = 'adu     '
TCOMM2  = 'Converted from 1-d pipeline spectrum (s1d_A)'

TTYPE3  = 'ERR     '
TFORM3  = '313061E '
TUTYP3  = 'Spectrum.Data.FluxAxis.Accuracy.StatError'
TUCD3   = 'stat.error;phot.flux.density;em.wl;stat.uncalib'
TUNIT3  = 'adu     '
TCOMM3  = 'Error spectrum not available, filled with NaN'

    
    The 'new' (2024) HARPS variant of the ESPRESSO [ESPDR] pipeline
    Identifiable by the existance of the keyword 'HIERARCH ESO PRO REC1 PIPE ID' in the header.
    HARPS S1D_FINAL_A files should be in VAC_RV == BARYCENTRIC-VACUUM
    And all are IDP compliant, so we just need to add a WAVE_TV column to the existing
    Binary Table
    i.e we need to convert BARY-->TOPO but NOT AIR-->VAC
-------------------------------------------------------------------------------
PRO.CATG=S1D_FINAL
TFIELDS =                   12 / number of fields in each row
TTYPE1  = 'WAVE    '           / label for field   1
TFORM1  = '220811D '           / data format of field: 8-byte DOUBLE
TUNIT1  = 'angstrom'           / physical unit of field
TUTYP1  = 'spec:Data.SpectralAxis.Value' / IVOA data model element for field 1
TUCD1   = 'em.wl;meta.main'    / UCD for field 1

TTYPE2  = 'FLUX    '           / label for field   2
TFORM2  = '220811D '           / data format of field: 8-byte DOUBLE
TUNIT2  = 'erg.cm**(-2).s**(-1).angstrom**(-1)' / physical unit of field
TUTYP2  = 'spec:Data.FluxAxis.Value' / IVOA data model element for field 2
TUCD2   = 'phot.flux.density;em.wl;meta.main' / UCD for field 2

TTYPE3  = 'ERR     '           / label for field   3
TFORM3  = '220811D '           / data format of field: 8-byte DOUBLE
TUNIT3  = 'erg.cm**(-2).s**(-1).angstrom**(-1)' / physical unit of field
TUTYP3  = 'spec:Data.FluxAxis.Accuracy.StatError' / IVOA data model element for
TUCD3   = 'stat.error;phot.flux.density;meta.main' / UCD for field 3

TTYPE4  = 'QUAL    '           / label for field   4
TFORM4  = '220811J '           / data format of field: 4-byte INTEGER
TUNIT4  = '        '           / physical unit of field
TUTYP4  = 'spec:Data.FluxAxis.Accuracy.QualityStatus' / IVOA data model element
TUCD4   = 'meta.code.qual;meta.main' / UCD for field 4

TTYPE5  = 'SNR     '           / label for field   5
TFORM5  = '220811D '           / data format of field: 8-byte DOUBLE
TUNIT5  = '        '           / physical unit of field
TUTYP5  = 'eso:Data.FluxAxis.Accuracy.SNR' / IVOA data model element for field 5
TUCD5   = 'stat.snr'           / UCD for field 5

TTYPE6  = 'WAVE_AIR'           / label for field   6
TFORM6  = '220811D '           / data format of field: 8-byte DOUBLE
TUNIT6  = 'angstrom'           / physical unit of field
TUTYP6  = 'eso:Data.SpectralAxis.Value' / IVOA data model element for field 6
TUCD6   = 'em.wl;obs.atmos'    / UCD for field 6

TTYPE7  = 'FLUX_EL '           / label for field   7
TFORM7  = '220811D '           / data format of field: 8-byte DOUBLE
TUNIT7  = '        '           / physical unit of field
TUTYP7  = 'eso:Data.FluxAxis.Value' / IVOA data model element for field 7
TUCD7   = '        '           / UCD for field 7

TTYPE8  = 'ERR_EL  '           / label for field   8
TFORM8  = '220811D '           / data format of field: 8-byte DOUBLE
TUNIT8  = '        '           / physical unit of field
TUTYP8  = 'eso:Data.FluxAxis.Accuracy.StatError' / IVOA data model element for f
TUCD8   = '        '           / UCD for field 8

TTYPE9  = 'QUAL_EL '           / label for field   9
TFORM9  = '220811J '           / data format of field: 4-byte INTEGER
TUNIT9  = '        '           / physical unit of field
TUTYP9  = 'eso:Data.FluxAxis.Accuracy.QualityStatus' / IVOA data model element f
TUCD9   = 'meta.code.qual'     / UCD for field 9

TTYPE10 = 'FLUX_CAL'           / label for field  10
TFORM10 = '220811D '           / data format of field: 8-byte DOUBLE
TUNIT10 = 'erg.s**(-1).cm**(-2).angstrom**(-1)' / physical unit of field
TUTYP10 = 'eso:Data.FluxAxis.Value' / IVOA data model element for field 10
TUCD10  = 'phot.flux.density;em.wl' / UCD for field 10

TTYPE11 = 'ERR_CAL '           / label for field  11
TFORM11 = '220811D '           / data format of field: 8-byte DOUBLE
TUNIT11 = 'erg.s**(-1).cm**(-2).angstrom**(-1)' / physical unit of field
TUTYP11 = 'eso:Data.FluxAxis.Accuracy.StatError' / IVOA data model element for f
TUCD11  = 'stat.error;phot.flux.density' / UCD for field 11

TTYPE12 = 'QUAL_CAL'           / label for field  12
TFORM12 = '220811J '           / data format of field: 4-byte INTEGER
TUNIT12 = '        '           / physical unit of field
TUTYP12 = 'eso:Data.FluxAxis.Accuracy.QualityStatus' / IVOA data model element f
TUCD12  = 'meta.code.qual'     / UCD for field 12

-------------------------------------------------------------------------------
PRO.CATG=S1D
TFIELDS =                    5 / number of fields in each row
TTYPE1  = 'wavelength'         / label for field   1
TFORM1  = '1D      '           / data format of field: 8-byte DOUBLE
TUNIT1  = 'angstrom'           / physical unit of field

TTYPE2  = 'wavelength_air'     / label for field   2
TFORM2  = '1D      '           / data format of field: 8-byte DOUBLE
TUNIT2  = 'angstrom'           / physical unit of field

TTYPE3  = 'flux    '           / label for field   3
TFORM3  = '1D      '           / data format of field: 8-byte DOUBLE
TUNIT3  = 'e-      '           / physical unit of field

TTYPE4  = 'error   '           / label for field   4
TFORM4  = '1D      '           / data format of field: 8-byte DOUBLE
TUNIT4  = 'e-      '           / physical unit of field

TTYPE5  = 'quality '           / label for field   5
TFORM5  = '1J      '           / data format of field: 4-byte INTEGER

-------------------------------------------------------------------------------
PRO.CATG=S1D_FLUXCAL
TFIELDS =                    3 / number of fields in each row
TTYPE1  = 'wavelength'         / label for field   1
TFORM1  = '1D      '           / data format of field: 8-byte DOUBLE
TUNIT1  = 'angstrom'           / physical unit of field
TTYPE2  = 'flux_cal'           / label for field   2
TFORM2  = '1D      '           / data format of field: 8-byte DOUBLE
TUNIT2  = 'erg.s**(-1).cm**(-2).angstrom**(-1)' / physical unit of field
TTYPE3  = 'error_cal'          / label for field   3
TFORM3  = '1D      '           / data format of field: 8-byte DOUBLE
TUNIT3  = 'erg.s**(-1).cm**(-2).angstrom**(-1)' / physical unit of field
-------------------------------------------------------------------------------
    '''
    output_files=list()
    files_dict={}
    for file in files:
        files_dict[file.category]=file
    has_sci_and_std=(
        ('SCIENCE' in files_dict.keys() or 'SCIENCE_CALCTRANS' in files_dict.keys())
        and
        'STD_MODEL' in files_dict.keys()
    )
    for k in files_dict.keys() :
        file=files_dict[k]
        if k in ['STD_MODEL','SCIENCE','SCIENCE_CALCTRANS',] :
            hl=hdulist=fits.open(file.name)
            procatg=hdulist[0].header.get('PRODCATG','NULL')
            out_file_name="%s/%s" %(output_dir,os.path.basename(file.name))
            if procatg in [
                    "SCIENCE.SPECTRUM",
                ] :
                # Input WAVE is AIR+BARYCENT, so
                # Copy WAVE and FLUX columns to
                # 1) WAVE_VT for WAVE VAC+TOPOCENT
                coldefs=[c for c in hdulist[1].columns]
                wave_vt=fits.Column(
                    name="WAVE_VT",
                    format=hdulist[1].columns['WAVE'].format,
                    unit=hdulist[1].columns['WAVE'].unit,
                    array=copy.deepcopy(hdulist[1].data['WAVE']),
                )
                coldefs+=[wave_vt,]
                hdulist[1]=fits.BinTableHDU.from_columns(coldefs, header=hdulist[1].header)

                # Now transform WAVE (AIR+BARYCENT) --> WAVE_VT (VAC+TOPOCENT)
                # Convert to micron
                hdulist[1].data['WAVE_VT']*=0.0001

                # Undo BERV correction...
                obs_erf_rv=hdulist[0].header.get(
                    'HIERARCH ESO DRS BERV',
                    hdulist[0].header.get(
                        'HIERARCH ESO QC BERV',
                    ),
                )
                if obs_erf_rv is not None :
                    hdulist[1].data['WAVE_VT']=iu.bary_to_topo(hdulist[1].data['WAVE_VT'],obs_erf_rv)
                    # Don't set SPECSYS since we are adding a column with the _VT in the name...
                    #hdulist[0].header['SPECSYS']='TOPOCENT'

                if 'HIERARCH ESO DRS VERSION' in hdulist[0].header :
                    # AIR --> VAC
                    hdulist[1].data['WAVE_VT']=iu.air_to_vac(hdulist[1].data['WAVE_VT'])

                hdulist[1].header[
                    'TUCD%d' %(hdulist[1].columns.names.index('WAVE_VT')+1)
                ]=('em.wl','Vacuum wavelength')

                # Convert back to Angstrom
                hdulist[1].data['WAVE_VT']/=0.0001

                print('writing %s' %(out_file_name))
                hdulist.writeto(out_file_name, overwrite=True, checksum=True, output_verify='silentfix')

                # broadcast the file to use for molecfit_model...
                if (
                    ( has_sci_and_std and k == "STD_MODEL" )
                    or
                    ( not has_sci_and_std )
                ) :
                    output_files.append(
                        reflex.FitsFile(
                            out_file_name, "STD_MODEL", hl[0].header['CHECKSUM'], file.purposes
                        )
                    )

                # broadcast all the files to use for molecfit_calctrans...
                if (
                    ( has_sci_and_std and k != "STD_MODEL" )
                    or
                    ( not has_sci_and_std )
                ) :
                    output_files.append(
                        reflex.FitsFile(
                            out_file_name, "SCIENCE_CALCTRANS", hl[0].header['CHECKSUM'], file.purposes
                        )
                    )

                # broadcast the original frame (for sticking it all back together at the end...)
                output_files.append(reflex.FitsFile(
                    file.name, "ORIG_%s" %(file.category), None, file.purposes
                ))
            else :
                # broadcast it...
                output_files=list()
                for file in files:
                    output_files.append(reflex.FitsFile(file.name, file.category, None, file.purposes ))
                out_file_name="%s/%s" %(output_dir,os.path.basename(file.name))
                '''
                fits.open(file.name).writeto(
                    out_file_name, overwrite=True, checksum=True, output_verify='silentfix',
                )
                output_files.append(reflex.FitsFile(out_file_name, file.category, None, file.purposes ))
                '''
        else :
            output_files.append(reflex.FitsFile(file.name, file.category, None, file.purposes ))

    return output_files
# ------------------------------------------------------------------------------------------
def recombine_idps( files=None, orig_files=None ) :
    return files
    """
    Transform back to input wavelength frame, i.e. BARY+AIR
    """
    #Get the name of the output directory
    pattern = '^--products-dir='
    import sys
    for arg in sys.argv:
        m=re.match(pattern, arg)
        if m is not None :
            output_dir = re.sub(pattern, '', arg)

    for i,file in enumerate(files) :
        # transform back to AIR and BARYCOOR...
        if file.category in ['SCIENCE_TELLURIC_CORR',] :
            hl=hdulist=fits.open(file.name)
            procatg=hdulist[0].header.get('PRODCATG','NULL')
            out_file_name="%s/%s" %(output_dir,os.path.basename(file.name))
            if procatg in [
                    "SCIENCE.SPECTRUM",
                ] :

                # Convert to micron
                hdulist[1].data['WAVE_VT']*=0.0001

                # VAC --> AIR
                hdulist[1].data['WAVE_VT']=iu.vac_to_air(hdulist[1].data['WAVE'])
                hdulist[1].header['TUCD1']='em.wl'


                # re-apply BERV correction...
                if hdulist[0].header['SPECSYS'] == 'TOPOCENT':
                    obs_erf_rv=hdulist[0].header.get(
                        'HIERARCH ESO DRS BERV',
                    )
                    if obs_erf_rv is not None :
                        hdulist[1].data['WAVE_VT']=iu.topo_to_bary(hdulist[1].data['WAVE_VT'],obs_erf_rv)
                        hdulist[0].header['SPECSYS']='BARYCENT'

                # Convert back to Angstrom
                hdulist[1].data['WAVE']/=0.0001

                print('writing %s' %(out_file_name))
                hdulist.writeto(out_file_name, overwrite=True, checksum=True, output_verify='silentfix')

                files[i]=reflex.FitsFile(
                    out_file_name,
                    procatg,
                    hdulist[0].header['CHECKSUM'],
                    ['UNIVERSAL',]
                )

    return files
# ------------------------------------------------------------------------------------------
wlg = 1.e-04  # AA --> micron :: WLG_TO_MICRON
clam='WAVE_VT'
cflux='FLUX'
cdflux='ERR'
def set_parameters(header=None, hdu=None, previous_inst_setup='None', files=None) :
    """
    Set any parameters needed for all inst_settings for this instrument
    """

    # WARNING: no spaces allowed in comma-separated lists!

    #Coarse include windows for H2O and O2
    wave_include='0.5855014996491902,0.6005821970046927,0.6265218805068855,0.6342172934891126,0.6455444803948276,0.652623580158602'
    list_mol = 'H2O,O2'
    rel_col = '1.0,1.0'
    fit_mol ='1,1'
    
    """
    list_mol='H2O,O2,CO2,O3,N2O,CO,CH4'
    rel_col='1.0,1.0,1.06,1.0,1.0,1.0,1.0'

    wave_include_H2O_1 = '0.5887311,0.5887840,0.5901212,0.5901975,0.5920397,0.5921697,0.5933263,0.5934751,0.5958927,0.5960632,0.5969137,0.5970267'
    wave_include_H2O_2 = '0.6476489,0.6477884,0.6481531,0.6482246,0.6488250,0.6488919,0.6492159,0.6492813,0.6516290,0.6516879,0.6517989,0.6519157,0.6521001,0.6521655,0.6525347,0.6526031,0.6535557,0.6536112,0.6545276,0.6546029,0.6550021,0.6550942,0.6554022,0.6554786,0.6555326,0.6556003'    
    wave_include_H2O =  wave_include_H2O_1 + ',' + wave_include_H2O_2
    #wave_include_H2O = ''
    #
    wave_include_O2 = '0.6282545,0.6283314,0.6285215,0.6285957,0.6288845,0.6289905,0.6290663,0.6291471,0.6294415,0.6294984,0.6296612,0.6297214,0.6299813,0.6301262,0.6303381,0.6304708,0.6307295,0.6308550,0.6311371,0.6312609,0.6315548,0.6316981'
    #
    if len(wave_include_H2O) > 0:
        sep_fit_mol = ','
        fit_H2O = '1,'
    else:
        sep_fit_mol = ''
        fit_H2O = '0,'
    if len(wave_include_O2) > 0:
        fit_O2 = '1,'
    else:
        fit_O2 = '0,'
    wave_include = wave_include_H2O + sep_fit_mol + wave_include_O2
    fit_mol = fit_H2O + fit_O2 + '0,0,0,0,0'
    """

    fit_tel_back="False"

    cont_fit = '1'
    cont_poly = int(2)

    varkern="True"
    
    try:
        gauss_fwhm = str((header['WAVELMIN'] + header['WAVELMAX']) / (
            2. * header['SPEC_RES'] * header['SPEC_BIN']))
    except:
        gauss_fwhm = '5'
    kernfac = 5

    return {
        'WLG_TO_MICRON': wlg,
        'LIST_MOLEC': list_mol,
        'FIT_MOLEC': fit_mol,
        'REL_COL': rel_col,
        'COLUMN_LAMBDA': clam,
        'COLUMN_FLUX': cflux,
        'COLUMN_DFLUX': cdflux,
        'FIT_TELESCOPE_BACKGROUND': fit_tel_back,
        'FIT_CONTINUUM': cont_fit,
        'CONTINUUM_N': cont_poly,
        'VARKERN': varkern,
        'KERNFAC': kernfac,
        'WAVE_INCLUDE':wave_include,
        'RES_GAUSS': gauss_fwhm,
    }

# ------------------------------------------------------------------------------------------
def check_idp(hdu, previous_inst_setup='None', files=None, ParameterInitialization=False, recipe=None) :
    """
    Set any parameters needed for the specific input file format (generally either FITS-Image
    or ESO-IDP stlye binary table).
    Also set wavelength ranges and molecules if instrument setup changes.
    """
    if hdu[0].header['NAXIS'] == 0:
        # IDP format
        primary='FALSE'
        errorext=0
        maskext=0
        map_convolve="1,1"
        map_atmosphere="0,1"
        map_correct="0,1"
        dd=hdu[1].data[cflux]
        # Some/all(?) 'old' IDPs have only NaNs in the ERR column...
        cdflux='ERR' # need this here, even though it is defined above, don't understand why...
        if np.isnan(np.nanmin(hdu[1].data[cdflux])) :
            cdflux="NULL"
    elif hdu[0].header['NAXIS'] == 1:
        #these are the values to use for spectra in generic PL format (non IDP)
        primary='TRUE'
        errorext=0
        maskext=0
        map_convolve="1"
        map_atmosphere="1"
        map_correct="1"
        dd=hdu[0].data

    ###median_cont = str(np.median(dd[np.where(~np.isnan(dd))]))
    median_cont = '1'

    # Barycentric radial velocity...
    # WARNING: very ad-hoc to how I massage the data before feeding them to the worflow!
    if 'HIERARCH ESO DRS BERV' in hdu[0].header :
        # 'old' HARPS PL
        QC_BERV_KEY="ESO DRS BERV"
    elif 'HIERARCH ESO QC BERV' in hdu[0].header :
        # 'new' HARPS/ESPDR PL
        QC_BERV_KEY="ESO QC BERV"

    # HARPS IDPs are by default in AIR+BARYCENT WL-frame
    # check_format() undoes BERV and transforms to VAC+TOPOCENT
    wave_frame='VAC'
    
    return dict(
        list(({
            # model settings...
            'COLUMN_LAMBDA': clam,
            'COLUMN_FLUX': cflux,
            'COLUMN_DFLUX': cdflux,
            'USE_ONLY_INPUT_PRIMARY_DATA': primary,
            'USE_DATA_EXTENSION_AS_DFLUX': errorext,
            'USE_DATA_EXTENSION_AS_MASK': maskext,
            'CONTINUUM_CONST': median_cont,
            # calctrans settings...
            'MAPPING_CONVOLVE': map_convolve,
            'MAPPING_ATMOSPHERIC': map_atmosphere,
            # correct settings...
            'MAPPING_CORRECT': map_correct,
            #
            'WAVELENGTH_FRAME': wave_frame,
            'OBS_ERF_RV_KEY': QC_BERV_KEY,
        }).items())
    )
# ------------------------------------------------------------------------------------------
