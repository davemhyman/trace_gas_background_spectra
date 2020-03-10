"""
Last Updated: 6 March, 2020

author:
Dave M. Hyman, PhD
Cooperative Institute for Meteorological Satellite Studies (CIMSS)
Space Science and Engineering Center (SSEC)
University of Wisconsin - Madison
Madison, WI, 53706

dave.hyman(at)ssec.wisc.edu  --or--  dhyman2(at)wisc.edu

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
This scripy contains functions to help construct
spectral background covariance matrices from CrIS data.

Hyman, D. M., and Pavolonis, M. J.: Probabilistic retrieval of volcanic
SO2 layer height and cumulative mass loading using the Cross-track Infrared
Sounder (CrIS), Atmospheric  Measurement  Techniques, (in review), 2020.
"""
import numpy as np
import os
from netCDF4 import Dataset
import time
################################################################################
#
#
#
################################################################################
def urls_from_link_list(path_to_link_list):
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
    ####################################################################
    # SEND ERROR MESSAGE AS EMAIL --------------------------------------
    ####################################################################
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
def get_CrIS_data(url,varnames):
    fidx=url.find('SNDR.')
    f = url[fidx:]
    print 'EXTRACTING DATA FROM GRANULE'
    print(f)
    print('-----------------------')
    print(time.ctime(time.time()))
    os.system('curl -n -c ~/.urs_cookies -b ~/.urs_cookies -LJO --url ' + url)
    ############################################################################
    # IF FILE DOESNT EXIST -> REPEAT DOWNLOAD ATTEMPT AND OPEN FILE
    while not os.path.isfile(f):
        print('-----Download failed--trying again-----')
        os.system('curl -n -c ~/.urs_cookies -b ~/.urs_cookies -LJO --url ' + url)
    ########################################
    # OPEN netCDF FILE AND EXTRACT VARIABLES
    openFile = Dataset(f,'r')
    print('------GRANULE--OPENED-------')
    ############################################################################
    # EXTRACT GRANULE VARIABLE -------------------------------------------------
    ############################################################################
    N = len(varnames)
    vars_out = N * [None]
    for v in range(N):
        vars_out[v] = openFile[varnames[v]][:]
    #CLOSE AND DELETE FILE
    openFile.close()
    print('------GRANULE--CLOSED-------')
    os.remove(f)
    while os.path.isfile(f):
        print('-----File delete failed--trying again-----')
        os.remove(f)
    print('------GRANULE--REMOVED------')
    #
    return vars_out
################################################################################
#
#
#
################################################################################
def get_season(mm_dd):
    seas = ['wint', 'spri', 'summ', 'fall', 'wint']
    seas_vec = np.array([[1 , 1], [3 , 20] , [6 , 21] , [ 9 , 22 ] , [12 , 21]])
    mm_dif = mm_dd[0] - seas_vec[:,0]
    dd_dif = mm_dd[1] - seas_vec[:,1]
    mm_dif[mm_dif<0]=100
    midx = np.argmin(mm_dif)
    if (mm_dif[midx] == 0) & (dd_dif[midx] < 0):
        midx = (midx - 1)%4
    return seas[midx], midx%4
################################################################################
#
#
#
################################################################################
def planck_rad2bt_wnum(wavenumber,radiance):
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
    # return sign of "a" as string
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
    return ((((5. * np.floor(LON/5.)).astype(int) + 180)) % 360) - 180
################################################################################
#
#
#
################################################################################
def latbin(LAT):
    return min((5. * np.floor(LAT/5.)).astype(int), 85)
################################################################################
#
#
#
################################################################################
def cris_data_quality(array1, qual, lw_qual, mw_qual, sw_qual):
    q_bad = (qual>0) + (lw_qual>0) + (mw_qual>0) + (sw_qual>0)
    non_degraded = ~(q_bad + np.ma.is_masked(array1))
    return non_degraded
################################################################################
#
#
#
################################################################################
def save_one_cov_nc(cov_mw, cov_inv_mw, mean_mw, wnum_mw, N_in_bin, bin_season_str, bin_latitude, bin_longitude, path_to_file):
    print('-------------------------------------------------')
    print('GENERATING NETCDF FILE:')
    dataset = Dataset(path_to_file, 'w')
    # DIMENSIONS
    m = len(wnum_mw)
    n_wnum = dataset.createDimension('wnum_mw', m)
    scalar = dataset.createDimension('scalar', 1)
    nchar = dataset.createDimension('str_dim', 1)
    bin_ends = dataset.createDimension('bin_ends', 2)
    # VARIABLES
    cov = dataset.createVariable('cov_mw',np.float32, ('wnum_mw','wnum_mw'))
    cov_inv = dataset.createVariable('cov_inv_mw',np.float32, ('wnum_mw','wnum_mw'))
    mean = dataset.createVariable('mean_mw',np.float32, ('wnum_mw',))
    wnum = dataset.createVariable('wnum_mw',np.float32, ('wnum_mw',))
    bin_count = dataset.createVariable('bin_count',np.int32, ('scalar',))
    bin_seas = dataset.createVariable('bin_season',str, ('str_dim',))
    bin_lat = dataset.createVariable('bin_latitude',np.int16, ('bin_ends',))
    bin_lon = dataset.createVariable('bin_longitude',np.int16, ('bin_ends',))
    # GLOBAL ATTRIBUTES
    dataset.description = 'SO2 - free background brightness temperature spectrum ' \
        'covariance matrix (and inverse) for one latitude, longitude, season bin.'
    dataset.history = 'Created ' + time.ctime(time.time())
    # VARIABLE ATTRIBUTES
    cov.units = 'K^2'
    cov_inv.units = 'K^-2'
    mean.units = 'K'
    wnum.units = 'cm^-1'
    bin_lat.units = 'degree_north'
    bin_lon.units = 'degree_east'
    #
    bin_count.description = 'number of spectra used to compile bin statistics'
    bin_seas.description = 'right half-open time (seasonal) interval'
    bin_lat.description = 'right half-open latitude interval'
    bin_lon.description = 'right half-open longitude interval'
    # ADD VALUES TO VARIABLES
    #
    cov[:] = np.float32(cov_mw)
    mean[:] = np.float32(mean_mw)
    cov_inv[:] = np.float32(cov_inv_mw)
    wnum[:] = np.float32(wnum_mw)
    bin_count[:] = np.int32(N_in_bin)
    bin_seas[:] = bin_season_str
    bin_lat[:] = bin_latitude + np.array([0, 5])
    bin_lon[:] = bin_longitude + np.array([0, 5])
    # WRITE FILE
    dataset.close()
################################################################################
#
#
#
################################################################################
def update_cov_variables_one_spectrum(sum_yyT, sum_y, N_in_bin, bt, seas_num, lat, lon):
    #
    # --------------------------------------------------------------------------
    # Identify background bin index for spectrum
    #
    # --------------------------------------------------------------------------
    ln = lonbin(lon[n])
    lt = latbin(lat[n])
    lon_idx = ((ln+180)/5).astype(int)
    lat_idx = ((lt+90)/5).astype(int)
    bin_idx = seas_num, lat_idx, lon_idx
    # --------------------------------------------------------------------------
    # Augment running covariance variables in [bin_idx]
    #
    # --------------------------------------------------------------------------
    yyT = bt[:,None] * bt[None,:]
    #
    sum_yyT[bin_idx] += yyT
    sum_y[bin_idx] += bt
    N_in_bin[bin_idx] += 1
    #
    return sum_yyT, sum_y, N_in_bin
################################################################################
#
#
#
################################################################################
def update_cov_variables_one_granule(N_in_bin, sum_yyT, sum_y, url, varnames, band, atrack, xtrack, fov, NEdN):
    # --------------------------------------------------------------------------
    # Get relelvant data from CrIS granule
    #
    # --------------------------------------------------------------------------
    DATA = get_CrIS_data(url,varnames)
    t_utc = DATA[0] # (atrack, xtrack, fov, utc_tuple)
    lat = DATA[1] # (atrack, xtrack, fov)
    lon = DATA[2] # (atrack, xtrack, fov)
    rad_mw = DATA[3][:,:,:,band] # (atrack, xtrack, fov, n_mw)
    wnum_mw = DATA[4][band] # (n_mw,)
    cal_qualflag = DATA[5] # (atrack, xtrack, fov)
    cal_lw_qualflag = DATA[6] # (atrack, xtrack, fov)
    cal_mw_qualflag = DATA[7] # (atrack, xtrack, fov)
    cal_sw_qualflag = DATA[8] # (atrack, xtrack, fov)
    #
    # --------------------------------------------------------------------------
    # Bound midwave radiance from below by half-noise value
    ## Convert Radiances to Brightness Temperatures
    #
    # --------------------------------------------------------------------------
    radiance = np.clip(rad_mw , 0.5 * NEdN , None) # (atrack, xtrack, fov, n_mw)
    bt = planck_rad2bt_wnum(wnum_mw[None,None,None,:], radiance)
    # --------------------------------------------------------------------------
    # Determine Data Quality
    #
    # --------------------------------------------------------------------------
    non_degraded = cris_data_quality(t_utc, cal_qualflag, cal_lw_qualflag, cal_mw_qualflag, cal_sw_qualflag)
    # --------------------------------------------------------------------------
    # Augment running covariance variables for each spectrum in the granule
    #
    # --------------------------------------------------------------------------
    print('---------- BEGIN PROCESSING GRANULE FOVs ----------')
    for i in range(atrack):
        print 'ALONG TRACK INDEX: {atrack_idx}'.format(atrack_idx=i)
        for j in range(xtrack):
            date = t_utc[i,j,1:3]
            seas_num = get_season(date)[1]
            for k in range(fov):
                n = i,j,k
                if non_degraded[n]:
                    #
                    sum_yyT, sum_y, N_in_bin = update_cov_variables_one_spectrum(sum_yyT, sum_y, N_in_bin, bt[n], seas_num, lat[n], lon[n])
                    #
                # fov[n] complete
            # all fov complete
        # all xtrack complete
    # all atrack complete
    return N_in_bin, sum_yyT, sum_y, wnum_mw
################################################################################
#
#
#
################################################################################
def generate_CrIS_bg_cov(url_list, season_bins, lat_bins, lon_bins):
    # --------------------------------------------------------------------------
    # Global Variables
    #
    # --------------------------------------------------------------------------
    NEdN = 0.05 # NEdN CrIS mid-wave avg: for Planck func cutoff (set cutoff as 0.5 * NEdN)
    atrack, xtrack, fov = 45, 30, 9 # CrIS granule dimensions
    band = np.arange(146,322+1) # relevant CrIS channels
    n_mw = len(band) # number of relevant CrIS channels
    # required CrIS variable names
    varnames = ['obs_time_utc', 'lat', 'lon', 'rad_mw', 'wnum_mw', 'cal_qualflag', 'cal_lw_qualflag', 'cal_mw_qualflag', 'cal_sw_qualflag']
    # --------------------------------------------------------------------------
    # Initialize Background Dimensions and Intermediate Variables
    #
    # --------------------------------------------------------------------------
    n_lon = len(lon_bins)
    n_lat = len(lat_bins)
    n_seas = len(season_bins)
    N_in_bin = np.zeros((n_seas, n_lat, n_lon), dtype = np.int32)
    sum_yyT = np.zeros((n_seas, n_lat, n_lon, n_mw, n_mw), dtype = np.float64)
    sum_y = np.zeros((n_seas, n_lat, n_lon, n_mw), dtype = np.float64)
    # --------------------------------------------------------------------------
    # Run through every background granule
    ## and update running covariance variables.
    ## Ensure all granules will be processed by keeping them in the list
    ## until they are processed regardless of try/except
    #
    # --------------------------------------------------------------------------
    while len(url_list) > 0:
        try:
            url = url_list[0]
            N_in_bin, sum_yyT, sum_y, wnum_mw = update_cov_variables_one_granule(N_in_bin, sum_yyT, sum_y, url, varnames, band, atrack, xtrack, fov, NEdN)
            url_list.remove(url)
            #
        except Exception as ex:
            print(ex)
            send_error_email(url, PrintException(), "dave.hyman@ssec.wisc.edu", "dave.hyman@ssec.wisc.edu")
            #
        #
    # all granules complete
    # --------------------------------------------------------------------------
    # Compute All Covariance Matrices, All Mean Spectra
    #
    # --------------------------------------------------------------------------
    EYYT = sum_yyT / N_in_bin[:,:,:,None,None].astype(np.float64) # (n_seas, n_lat, n_lon, n_mw, n_mw)
    all_mean = sum_y / N_in_bin[:,:,:,None].astype(np.float64) # (n_seas, n_lat, n_lon, n_mw)
    EYEYT = all_mean[:,:,:,:,None] * all_mean[:,:,:,None,:] # (n_seas, n_lat, n_lon, n_mw, n_mw)
    all_cov = EYYT - EYEYT # (n_seas, n_lat, n_lon, n_mw, n_mw)
    return all_mean, all_cov, wnum_mw, N_in_bin
################################################################################
#
#
#
################################################################################
def CrIS_bg_cov_main(url_list):
    # --------------------------------------------------------------------------
    # Main routine to compute background covariance variables
    ## and save them into individual background bin files.
    ## Backgrround spacetime bins: 5 degrees x 5 degrees x seasons
    ## Define season names as northern hemisphere season names
    #
    # --------------------------------------------------------------------------
    season_bins = ['winter = [21 Dec. 2017 - 03 Mar. 2018)', \
        'spring = [03 Mar. 2018 - 21 Jun. 2018)', \
        'summer = [21 Jun. 2018 - 22 Sep. 2018)', \
        'fall = [01 Nov. 2017 - 21 Dec. 2017) & [22 Sep. 2018 - 01 Nov. 2018)']
    lat_bins = np.arange(-90,90,5) # bin bottom latitudes
    lon_bins = np.arange(-180,180,5) # bin left longitudes
    # --------------------------------------------------------------------------
    # Generate all mean spectra and covariance matrices
    #
    # --------------------------------------------------------------------------
    all_mean, all_cov, wnum_mw, N_in_bin = generate_CrIS_bg_cov(url_list, season_bins, lat_bins, lon_bins)
    # --------------------------------------------------------------------------
    # Dimensions of Data Arrays
    #
    # --------------------------------------------------------------------------
    n_seas, n_lat, n_lon, n_mw = all_mean.shape
    # --------------------------------------------------------------------------
    # Set Covariancve Database Directory and
    ## Covariance file name prefix
    #
    # --------------------------------------------------------------------------
    prestr = '/data/dhyman/CrIS_bg_cov_f32/CrIS.bg.mw_cov.'
    for seas_num in range(n_seas):
        for lat_idx in range(n_lat):
            for lon_idx in range(n_lon):
                # --------------------------------------------------------------
                # Get background bin index
                #
                # --------------------------------------------------------------
                bin_idx = seas_num, lat_idx, lon_idx
                # --------------------------------------------------------------
                # Bin metadata
                #
                # --------------------------------------------------------------
                bin_latitude = lat_bins[lat_idx]
                bin_longitude = lon_bins[lon_idx]
                bin_season_str = season_bins[seas_num]
                # --------------------------------------------------------------
                # Bin identifier string
                #
                # --------------------------------------------------------------
                seas_key = bin_season_str[0:4]
                lonstr = 'lon=' + sign_string(bin_longitude) + '{n:03d}'.format(n = abs(bin_longitude))
                latstr = 'lat=' + sign_string(bin_latitude) + '{n:02d}'.format(n = abs(bin_latitude))
                bin  = seas_key + '.' + lonstr + '.' + latstr + '.nc'
                # --------------------------------------------------------------
                # Bin Covariance Matrix, Inverse Covariance Matrix, Mean Spectrum
                #
                # --------------------------------------------------------------
                cov_mw = all_cov[bin_idx]
                cov_inv_mw = np.linalg.inv(cov_mw)
                mean_mw = all_mean[bin_idx]
                # --------------------------------------------------------------
                # Save background bin covariance file
                #
                # --------------------------------------------------------------
                path_to_file = prestr + bin
                save_one_cov_nc(cov_mw, cov_inv_mw, mean_mw, wnum_mw, N_in_bin, bin_season_str, bin_latitude, bin_longitude, path_to_file)
################################################################################
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
