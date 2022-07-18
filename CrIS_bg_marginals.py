"""
Last Updated: 18 July, 2022

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
import sys
sys.path.append('trace_gas_background_spectra/')
from bg_aux import *
################################################################################
#
#
#
################################################################################
def get_all_bg_mean_and_std(path_to_cov_database, season_bins, lat_bins, lon_bins, n_mw):
    # --------------------------------------------------------------------------
    # Initialize Background Dimensions
    #
    # --------------------------------------------------------------------------
    n_seas = len(season_bins)
    n_lat = len(lat_bins)
    n_lon = len(lon_bins)
    # --------------------------------------------------------------------------
    # Get Background mean and std spectra
    #
    # --------------------------------------------------------------------------
    all_BT_mean = np.zeros((n_seas, n_lat, n_lon, n_mw), dtype = np.float64)
    all_BT_std = np.zeros((n_seas, n_lat, n_lon, n_mw), dtype = np.float64)
    #
    if path_to_cov_database[-1] != '/':
        path_to_cov_database = path_to_cov_database + '/'
    #
    prestr = path_to_cov_database + 'SNPP.CrIS.bg.mw_cov.'
    #
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
                # Bin Coviariance matrix and mean spectrum
                #
                # --------------------------------------------------------------
                path_to_file = prestr + bin
                fcov = Dataset(path_to_file,'r')
                cov = fcov['cov_mw'][:]
                mean = fcov['mean_mw'][:]
                #
                all_BT_mean[bin_idx] = mean
                all_BT_std[bin_idx] = np.diag(cov)**0.5
                #
            # all longitude bins
        # all latitude bins
    # all season bins
    #
    return all_BT_mean, all_BT_std
################################################################################
#
#
#
################################################################################
def save_one_marginal_nc(pdf_mw, cdf_mw, BT_bins_mw, wnum_mw, bin_season_str, bin_latitude, bin_longitude, path_to_file):
    # --------------------------------------------------------------------------
    # Save down netCDF file with variables relevant to
    ## background spectral marginal distributions for one spacetime bin
    ## data to save:
    ##     Brightness Temperature channel-wise PDFs
    ##     Brightness Temperature channel-wise marginal CDFs
    ##     Brightness Temperature channel-wise values (domains of PDFs, CDFs)
    ##     wavenumbers
    ##     Spactime bin identifiers (season, latitude range, longitude range)
    #
    # --------------------------------------------------------------------------
    print('-------------------------------------------------')
    print('GENERATING NETCDF FILE:')
    # --------------------------------------------------------------------------
    # Open Writable File
    #
    # --------------------------------------------------------------------------
    dataset = Dataset(path_to_file, 'w')
    # --------------------------------------------------------------------------
    # Set Dimension Names
    #
    # --------------------------------------------------------------------------
    sh = pdf_mw.shape
    m = len(wnum_mw)
    n_vals = sh[sh != m]
    n_wnum = dataset.createDimension('wnum_mw', m)
    n_hist_bins = dataset.createDimension('n_pdf_vals', n_vals)
    nchar = dataset.createDimension('str_dim', 1)
    bin_ends = dataset.createDimension('bin_ends', 2)
    # --------------------------------------------------------------------------
    # Set Variable Names
    #
    # --------------------------------------------------------------------------
    pdf = dataset.createVariable('pdf_BT_mw',np.float32, ('wnum_mw','n_pdf_vals'))
    cdf = dataset.createVariable('cdf_BT_mw',np.float32, ('wnum_mw','n_pdf_vals'))
    BT_bins = dataset.createVariable('BT_vals_mw',np.float32, ('wnum_mw','n_pdf_vals'))
    wnum = dataset.createVariable('wnum_mw',np.float32, ('wnum_mw',))
    bin_seas = dataset.createVariable('bin_season',str, ('str_dim',))
    bin_lat = dataset.createVariable('bin_latitude',np.int16, ('bin_ends',))
    bin_lon = dataset.createVariable('bin_longitude',np.int16, ('bin_ends',))
    # --------------------------------------------------------------------------
    # Set Attributes: descriptions, file history, units
    #
    # --------------------------------------------------------------------------
    dataset.description = 'SO2 - free background brightness temperature spectrum ' \
        'channel-wise marginal probability distributions for one latitude, longitude, season bin.'
    dataset.history = 'Created ' + time.ctime(time.time())
    #
    pdf.units = 'K^-1'
    cdf.units = '-'
    BT_bins.units = 'K'
    wnum.units = 'cm^-1'
    bin_lat.units = 'degree_north'
    bin_lon.units = 'degree_east'
    #
    bin_seas.description = 'right half-open time (seasonal) interval'
    bin_lat.description = 'right half-open latitude interval'
    bin_lon.description = 'right half-open longitude interval'
    # --------------------------------------------------------------------------
    # Assign Values to Variables
    #
    # --------------------------------------------------------------------------
    pdf[:] = np.float32(pdf_mw)
    cdf[:] = np.float32(cdf_mw)
    BT_bins[:] = np.float32(BT_bins_mw)
    wnum[:] = np.float32(wnum_mw)
    bin_seas[:] = bin_season_str
    bin_lat[:] = bin_latitude + np.array([0, 5])
    bin_lon[:] = bin_longitude + np.array([0, 5])
    # --------------------------------------------------------------------------
    # Write File
    #
    # --------------------------------------------------------------------------
    dataset.close()
################################################################################
#
#
#
################################################################################
def incremental_hist(spec_bin_array, spectrum):
    # --------------------------------------------------------------------------
    # Generates a channel-wise histogram for a single spectrum
    ## with bin left edges in an array. Returns an array of 1s and 0s
    ## Ignores spectral values outside of defined bin edges, so overall
    ## bin ranges must be constructed carefully.
    ## Arguments::
    ##     spec_bin_array (wavenumbers, bins) : bin left edges
    ##     spectrum (wavenumbers,)
    ## Returns::
    ##     hist_array (wavenumbers, bins): array of 1s and 0s
    #
    # --------------------------------------------------------------------------
    bin_width = np.mean(np.diff(spec_bin_array, axis =-1), axis =-1)
    # --------------------------------------------------------------------------
    # On each wavenumber, logical array where data >= left bin edge
    #
    # --------------------------------------------------------------------------
    greater_or_equal = (spectrum[:,None] >= spec_bin_array)
    # --------------------------------------------------------------------------
    # on each wavenumber, logical array where data < right bin edge
    ## right bin edge = left bin edge + bin width
    #
    # --------------------------------------------------------------------------
    less_than = (spectrum[:,None] < spec_bin_array + bin_width[:,None])
    #
    hist_array = np.logical_and(greater_or_equal, less_than).astype(float)
    return hist_array
################################################################################
#
#
#
################################################################################
def update_histograms_one_spectrum(all_hist, BT_bins, bt, seas_num, lat, lon):
    # --------------------------------------------------------------------------
    # Identify background bin index for given Brightness Temperature spectrum
    #
    # --------------------------------------------------------------------------
    ln = lonbin(lon)
    lt = latbin(lat)
    lon_idx = ((ln+180)/5).astype(int)
    lat_idx = ((lt+90)/5).astype(int)
    bin_idx = seas_num, lat_idx, lon_idx
    # --------------------------------------------------------------------------
    # Augment running histogram in [bin_idx]
    #
    # --------------------------------------------------------------------------
    all_hist[bin_idx] += incremental_hist(BT_bins, bt)
    #
    return all_hist
################################################################################
#
#
#
################################################################################
def update_histograms_one_granule(all_hist, BT_bins, url, varnames, band, atrack, xtrack, fov, NEdN):
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
    # Augment running histogram for each spectrum in the granule
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
                    all_hist = update_histograms_one_spectrum(all_hist, BT_bins, bt[n], seas_num, lat[n], lon[n])
                    #
                # fov[n] complete
            # all fov complete
        # all xtrack complete
    # all atrack complete
    return all_hist
################################################################################
#
#
#
################################################################################
def generate_CrIS_bg_marginals(url_list, season_bins, lat_bins, lon_bins, path_to_cov_database):
    # --------------------------------------------------------------------------
    # Global Variables
    #
    # --------------------------------------------------------------------------
    varnames, band, dims, NEdN = sensor_parameters('CrIS')
    atrack, xtrack, fov = dims
    # --------------------------------------------------------------------------
    # Initialize Background Dimensions
    #
    # --------------------------------------------------------------------------
    n_seas = len(season_bins) # number of seasonal bins (4)
    n_lat = len(lat_bins) # number of latitude bins
    n_lon = len(lon_bins) # number of longitude bins
    n_mw = len(band) # number of relevant midwave IR channels
    # --------------------------------------------------------------------------
    # Get Background mean and std spectra
    #
    # --------------------------------------------------------------------------
    all_BT_mean, all_BT_std = get_all_bg_mean_and_std(path_to_cov_database, season_bins, lat_bins, lon_bins, n_mw)
    # --------------------------------------------------------------------------
    # Generate Histogram Domains
    #
    # --------------------------------------------------------------------------
    n_hist_bins = 61
    z_bins = np.linspace(-6., 6., n_hist_bins)
    all_BT_bins = all_BT_mean[:,:,:,:, None] + all_BT_std[:,:,:,:, None] * z_bins[None,None,None,None,:] # (n_seas, n_lat, n_lon, n_mw, n_hist_bins)
    # --------------------------------------------------------------------------
    # Initialize Output Histogram Array
    #
    # --------------------------------------------------------------------------
    all_hist = np.zeros((n_seas, n_lat, n_lon, n_mw, n_hist_bins), dtype = np.float64) # (n_seas, n_lat, n_lon, n_mw, n_hist_bins)
    # --------------------------------------------------------------------------
    # Run through every background granule and update running histograms.
    ## Ensure all granules will be processed by keeping them in the list
    ## until they are processed regardless of try/except
    #
    # --------------------------------------------------------------------------
    while len(url_list) > 0:
        try:
            url = url_list[0]
            all_hist, wnum_mw = update_histograms_one_granule(all_hist, all_BT_bins, url, varnames, band, atrack, xtrack, fov, NEdN)
            url_list.remove(url)
            #
        except Exception as ex:
            print(ex)
            send_error_email(url, PrintException(), "dave.hyman@ssec.wisc.edu", "dave.hyman@ssec.wisc.edu")
            #
        #
    # all granules complete
    # --------------------------------------------------------------------------
    # Compute all pdfs, cdfs
    #
    # --------------------------------------------------------------------------
    d_all_BT_bins = np.mean(np.diff(all_BT_bins, axis =-1), axis =-1) # (n_seas, n_lat, n_lon, n_mw)
    hist_integral_total = np.trapz(all_hist, all_BT_bins, axis =-1) # (n_seas, n_lat, n_lon, n_mw)
    all_pdf = all_hist / hist_integral_total[:,:,:,:,None] # (n_seas, n_lat, n_lon, n_mw, n_hist_bins)
    all_cdf = np.zeros(all_pdf.shape) # (n_seas, n_lat, n_lon, n_mw, n_hist_bins)
    all_cdf[:,:,:,:,1:] = np.cumsum(all_pdf[:,:,:,:,0:-1] * d_all_BT_bins[:,:,:,:,None], axis =-1) # (n_seas, n_lat, n_lon, n_mw, n_hist_bins)
    return all_pdf, all_cdf, all_BT_bins, wnum_mw
################################################################################
#
#
#
################################################################################
def CrIS_bg_marginals_main(url_list, path_to_marginals_database, path_to_cov_database):
    # --------------------------------------------------------------------------
    # Main routine to compute background marginal distributions
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
    # Generate all channel-wise marginal distributions
    #
    # --------------------------------------------------------------------------
    all_pdf, all_cdf, all_BT_bins, wnum_mw = generate_CrIS_bg_marginals(url_list, season_bins, lat_bins, lon_bins, path_to_cov_database)
    # --------------------------------------------------------------------------
    # Dimensions of Data Arrays
    #
    # --------------------------------------------------------------------------
    n_seas, n_lat, n_lon, n_mw, n_hist_bins = all_pdf.shape
    # --------------------------------------------------------------------------
    # Set Marginals Database Directory and
    ## marginals file name prefix
    #
    # --------------------------------------------------------------------------
    if path_to_marginals_database[-1] != '/':
        path_to_marginals_database = path_to_marginals_database + '/'
    #
    prestr = path_to_marginals_database + 'SNPP.CrIS.bg.mw_marginals.'
    #
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
                # Bin Channelwise PDFs, Marginal CDFs, BT value bins
                #
                # --------------------------------------------------------------
                pdf_mw = all_pdf[bin_idx]
                cdf_mw = all_cdf[bin_idx]
                BT_bins_mw = all_BT_bins[bin_idx]
                # --------------------------------------------------------------
                # Save background bin marginals file
                #
                # --------------------------------------------------------------
                path_to_file = prestr + bin
                save_one_marginal_nc(pdf_mw, cdf_mw, BT_bins_mw, wnum_mw, bin_season_str, bin_latitude, bin_longitude, path_to_file)
################################################################################
#
#
#
################################################################################
