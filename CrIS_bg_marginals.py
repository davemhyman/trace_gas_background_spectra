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
def get_all_bg_mean_and_std(path_to_cov_data, season_bins, lat_bins, lon_bins, n_mw):
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
    if path_to_cov_data[-1] != '/':
        path_to_cov_data = path_to_cov_data + '/'
    #
    prestr = path_to_cov_data + 'CrIS.bg.mw_cov.'
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
    hist_array = 1.0 * (greater_or_equal * less_than)
    return hist_array
################################################################################
#
#
#
################################################################################
def histogram_update(all_hist, BT_bins, bt, seas_num, lat, lon):
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
def augment_histogram_one_gran(all_hist, BT_bins, url, varnames, band, atrack, xtrack, fov, NEdN):
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
                    all_hist = histogram_update(all_hist, BT_bins, bt[n], seas_num, lat[n], lon[n])
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
def generate_CrIS_bg_marginals(url_list, season_bins, lat_bins, lon_bins):
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
    path_to_cov_data = '/data/dhyman/CrIS_bg_cov_f32/'
    all_BT_mean, all_BT_std = get_all_bg_mean_and_std(path_to_cov_data, season_bins, lat_bins, lon_bins, n_mw)
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
            all_hist, wnum_mw = augment_histogram_one_gran(all_hist, all_BT_bins, url, varnames, band, atrack, xtrack, fov, NEdN)
            #
        except Exception as ex:
            urls_not_processed.append(url)
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
    all_cdf = 0.0 * all_pdf # (n_seas, n_lat, n_lon, n_mw, n_hist_bins)
    all_cdf[:,:,:,:,1:] = np.cumsum(all_pdf[:,:,:,:,0:-1] * d_all_BT_bins[:,:,:,:,None], axis =-1) # (n_seas, n_lat, n_lon, n_mw, n_hist_bins)
    return all_pdf, all_cdf, all_BT_bins, wnum_mw
################################################################################
#
#
#
################################################################################
def CrIS_bg_cov_main(url_list, path_to_marginals_data):
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
    all_pdf, all_cdf, all_BT_bins, wnum_mw = generate_CrIS_bg_marginals(url_list, season_bins, lat_bins, lon_bins)
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
    if path_to_marginals_data[-1] != '/':
        path_to_marginals_data = path_to_marginals_data + '/'
    #
    prestr = path_to_marginals_data + 'CrIS.bg.mw_marginals.'
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
