"""
Last Updated: 10 March, 2020

author:
Dave M. Hyman, PhD
Cooperative Institute for Meteorological Satellite Studies (CIMSS)
Space Science and Engineering Center (SSEC)
University of Wisconsin - Madison
Madison, WI, 53706

dave.hyman(at)ssec.wisc.edu  --or--  dhyman2(at)wisc.edu

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
This scripy contains auxiliary functions to help construct
spectral background statistics from hyperspectral IR data.

Hyman, D. M., and Pavolonis, M. J.: Probabilistic retrieval of volcanic
SO2 layer height and cumulative mass loading using the Cross-track Infrared
Sounder (CrIS), Atmospheric  Measurement  Techniques, (in review), 2020.
"""
import numpy as np
import os
from netCDF4 import Dataset
import time
import sys
sys.path.append('trace_gas_background_spectra/')
################################################################################
#
#
#
################################################################################
def urls_from_link_list(path_to_link_list):
    # --------------------------------------------------------------------------
    # Get a list containing urls from a txt file where each line is a url
    ##
    # --------------------------------------------------------------------------
    list = open(path_to_link_list)
    urls = [line.rstrip('\n') for line in list]
    pdfs = [s for s in urls if ".pdf" in s]
    for pdf in pdfs:
        urls.remove(pdf)
    return urls
################################################################################
#
#
#
################################################################################
def PrintException():
    # --------------------------------------------------------------------------
    # Query sys info to capture exception info such as
    ## file name, line number, exception detail
    #
    # --------------------------------------------------------------------------
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    str_out = 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)
    print str_out
    return str_out
################################################################################
#
#
#
################################################################################
def send_error_email(filename, error_message, email_FROM, email_TO):
    # --------------------------------------------------------------------------
    # Error message via auto sent email
    #
    # --------------------------------------------------------------------------
    for n_email in range(1):
        import smtplib
        subject = "Data processing interrupted"
        body = "The last file was: '" + filename + "'. \n\n" \
            + "Error Message: \n" + error_message
        theEmail = "Subject:  %s\n\n%s" % (subject,body)
        smtpObj = smtplib.SMTP('localhost')
        smtpObj.sendmail(email_FROM,email_TO,theEmail)
################################################################################
#
#
#
################################################################################
def get_season(mm_dd):
    # --------------------------------------------------------------------------
    # Get the 4-letter season string code and season number as defined here:
    ## season codes = 'wint', 'spri', 'summ', 'fall'
    ## season numbers = 0,      1,      2,      3
    ## ** NOTE: Seasons are defined with respect to Northern Hemisphere names **
    #
    # --------------------------------------------------------------------------
    seas = ['wint', 'spri', 'summ', 'fall', 'wint']
    # --------------------------------------------------------------------------
    # Array of season start dates. First row is Jan 1.
    #
    # --------------------------------------------------------------------------
    seas_vec = np.array([[1 , 1], [3 , 20] , [6 , 21] , [ 9 , 22 ] , [12 , 21]])
    # --------------------------------------------------------------------------
    # Arithmetic approach to determine season number
    ## Alternative: switch to datetime arrays and logic
    # --------------------------------------------------------------------------
    mm_dif = mm_dd[0] - seas_vec[:,0]
    dd_dif = mm_dd[1] - seas_vec[:,1]
    mm_dif[mm_dif<0]=100
    midx = np.argmin(mm_dif)
    if (mm_dif[midx] == 0) & (dd_dif[midx] < 0):
        midx = (midx - 1)%4
    # --------------------------------------------------------------------------
    # Return 2-element list with season code and number
    #
    # --------------------------------------------------------------------------
    return seas[midx], midx%4
################################################################################
#
#
#
################################################################################
def planck_rad2bt_wnum(wavenumber,radiance):
    # --------------------------------------------------------------------------
    # Planck Function:
    ## Radiance(wavenumber) -> Brightness Temperature(wavenumber)
    ## Radiance: mW m^-2 sr^-1 cm
    ## BT: K
    ## wavenumber: cm^-1
    #
    # --------------------------------------------------------------------------
    C1 = 0.00001191044
    C2 = 1.43877
    return C2 * wavenumber / np.log(1. + C1 * (wavenumber**3) / radiance)
################################################################################
#
#
#
################################################################################
def sign_string(a):
    # --------------------------------------------------------------------------
    # return sign of "a" as string "+" or "-"
    #
    # --------------------------------------------------------------------------
    if a>=0:
        return '+'
    else:
        return '-'
################################################################################
#
#
#
################################################################################
def sgn(a):
    # --------------------------------------------------------------------------
    # sign (signum) function over integers
    #
    # --------------------------------------------------------------------------
    if a>=0:
        return 1
    elif a<0:
        return -1
    else:
        return 0
################################################################################
#
#
#
################################################################################
def lonbin(LON):
    # --------------------------------------------------------------------------
    # Determin 5-degree longitude bin left edge value
    #
    # --------------------------------------------------------------------------
    return ((((5. * np.floor(LON/5.)).astype(int) + 180)) % 360) - 180
################################################################################
#
#
#
################################################################################
def latbin(LAT):
    # --------------------------------------------------------------------------
    # Determin 5-degree latitude bin bottom edge value
    #
    # --------------------------------------------------------------------------
    return min((5. * np.floor(LAT/5.)).astype(int), 85)
################################################################################
#
#
#
################################################################################
def get_CrIS_data(url,varnames):
    # --------------------------------------------------------------------------
    # Download netCDF CrIS data from url without keeping file locally
    ## get only excat variable name matches for strings in varnames
    ## includes special handling to ensure download (based on past problems)
    #
    # --------------------------------------------------------------------------
    fidx=url.find('SNDR.')
    f = url[fidx:]
    print 'EXTRACTING DATA FROM GRANULE'
    print(f)
    print('-----------------------')
    print(time.ctime(time.time()))
    os.system('curl -n -c ~/.urs_cookies -b ~/.urs_cookies -LJO --url ' + url)
    # --------------------------------------------------------------------------
    # If file still doesnt exist -> repeat download attempt
    #
    # --------------------------------------------------------------------------
    while not os.path.isfile(f):
        print('-----Download failed--trying again-----')
        os.system('curl -n -c ~/.urs_cookies -b ~/.urs_cookies -LJO --url ' + url)
    # --------------------------------------------------------------------------
    # Open netCDF file, extract variables in varnames list
    ## Populate list of extracted data arrays
    #
    # --------------------------------------------------------------------------
    openFile = Dataset(f,'r')
    print('------GRANULE--OPENED-------')
    N = len(varnames)
    vars_out = N * [None]
    for v in range(N):
        vars_out[v] = openFile[varnames[v]][:]
    # --------------------------------------------------------------------------
    # Close and delete file from local directory
    #
    # --------------------------------------------------------------------------
    openFile.close()
    print('------GRANULE--CLOSED-------')
    os.remove(f)
    while os.path.isfile(f):
        print('-----File delete failed--trying again-----')
        os.remove(f)
    print('------GRANULE--REMOVED------')
    # --------------------------------------------------------------------------
    # Return list of data arrays
    #
    # --------------------------------------------------------------------------
    return vars_out
################################################################################
#
#
#
################################################################################
def cris_data_quality(array1, qual, lw_qual, mw_qual, sw_qual):
    # --------------------------------------------------------------------------
    # Generate logical array of CrIS data array size determining non-degraded
    ## quality status and non-masked array using logical array arithmetic
    #
    # --------------------------------------------------------------------------
    q_bad = (qual>0) + (lw_qual>0) + (mw_qual>0) + (sw_qual>0)
    non_degraded = ~(q_bad + np.ma.is_masked(array1))
    return non_degraded
################################################################################
#
#
#
################################################################################
