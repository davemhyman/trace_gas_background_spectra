"""
bg_spectrum_sampling.py

Last Updated: 11 February, 2020

author:
Dave M. Hyman, PhD
Cooperative Institute for Meteorological Satellite Studies (CIMSS)
Space Science and Engineering Center (SSEC)
University of Wisconsin - Madison
Madison, WI, 53706

dave.hyman(at)ssec.wisc.edu  --or--  dhyman2(at)wisc.edu

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
This script contains functions for generating CrIS background spectrum
samples by the NORTA process for generating
correlated random vectors with arbitrary marginals.

NORTA Process::
Generate samples of random vectors of dimension N:
X = (X1, X2, ..., XN)
subject to the following data:

G1. N marginal distributions (CDFs) of X (one on each component)
G2. A covariance matrix S (N x N) giving a correlation structure to X

The general NORTA proces is as follows:
--------
1.)
----
Given G1 and G2, define a correlation function:
rho_X(i,j) = Corr(Xi,Xj) = Cov(Xi,Xj)/sqrt(Var(Xi)*Var(Xj))
where:
Cov(Xi,Xj) is a (integral) function of the bivariate normal PDF:
phi(z_i,z_j|rho_Z(i,j)) with correlation rho_Z(i,j) between two standard
normal random vairables Zi and Zj.

Note that rho_Z(i,j) = Corr(Zi,Zj) = Cov(Zi,Zj) for standard normal rv's.

This function involves the expectation
of the inverse transforms of the two standard noramls by the given marginals.

Consequently, we define the a function between the pairwise correlation of the
two desired rendom vairables and the two normal random vairables:

rho_X(i,j) = C_ij( rho_Z(i,j) )         (Eqn.1)

where C_ij(.) is the integral function involving the covariances.
--------
2.)
----
Solve for a value of rho_Z(i,j) which gives the required rho_X(i,j).
Because covariance matrices are symmetric and have easily transformed
diagonals (just the variances - depends only on the marginals) this reduces to
N(N-1)/2 independent problems.

Generally, this must be done by iteration since G1, G2 are not explicit.

--------
3.)
----
With each rho_Z(i,j) calculated, calculate a covariance matrix S_Z for the
standard normal vector Z = (Z1, Z2, ..., ZN) and generate sample standard
normal vectors with specified mean (0 vector) and covariance matrix S_Z.

With a sample standard normal vector Z, transform it to the desired random
vector X such that X_i = inv(F_Xi)[Phi(Zi)]
where Phi is the univariate standard normal CDF and the inv(F_Xi) is the
inverse of the marginal CDF for the i-th component of X.



--------
Notes:
----
These function are tuned to work well for generating random samples
of a radiance (or brightness temperature) spectrum for a hyperspectral IR
instrument.

Specifically, this was motivated by the problem of generating
random sample background spectrum for a trace gas (SO2) retrieval tuned to
the Cross Track Infrared Sounder (CrIS) onboard
the JPSS satellites Suomi-NPP and NOAA-20.

Many coding and approximation choices are made in the name of performance
for a large dataset tuned to the particulars of CrIS spectra, though
it is likely that this is somewhat generalizable.

Details can be found in Hyman and Pavolonis, 2020.


--------
References:
----

Cario, M. C. and Nelson, B. L.: Modeling and Generating Random Vectors with
Arbitrary Marginal Distributions and Correlation Matrix, Tech. rep.,
Department of Industrial Engineering and Management Science,
Northwestern University, Evanston, IL,
http://citeseerx.ist.655psu.edu/viewdoc/summary?doi=10.1.1.48.281, 1997.

Hyman, D. M., and Pavolonis, M. J.: Probabilistic retrieval of volcanic
SO2 layer height and cumulative mass loading using the Cross-track Infrared
Sounder (CrIS), Atmospheric  Measurement  Techniques, (in review), 2020.

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""
import os
import numpy as np
import math as m
from netCDF4 import Dataset
import time
from scipy.stats import norm
from scipy.integrate import simps
################################################################################

################################################################################
# FUNCTIONS --------------------------------------------------------------------
################################################################################
#
#
#
################################################################################
def irbnd_pdf(r):
    #
    # --------------------------------------------------------------------------
    # Make the bivariate normal PDF for a pair of
    ## independent standard normal random variables
    ## in polar coordinates
    # --------------------------------------------------------------------------
    return (
        1. / (2 * np.pi) \
        * np.exp( (-1./2) * (r**2) ) \
        )
################################################################################
#
#
#
################################################################################
def Finv(U,X,F):
    #
    # --------------------------------------------------------------------------
    # Generate the inverse marginal distribution X = F^-1(U)
    ## derived from the piecewise linear CDF: F(X).
    ## Ensure that the domian of the inverse is [0,1].
    # --------------------------------------------------------------------------
    #
    F[0] = 0.0
    F[-1] = 1.0
    return np.interp(U,F,X)
################################################################################
#
#
#
################################################################################
def C_n(rho,sq_means,sq_vars,values,marginals,rad_inf,N_uv):
    #
    # --------------------------------------------------------------------------
    # NORTA Forward Model: Correlation function rho_X = C_ij(rho_Z)
    ## Evaluate the correlation function as a vector
    ## where each element is a pairwise correlation.
    ## See Cario & Nelson, 1997
    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    # Generate list of lower triangle indices
    # --------------------------------------------------------------------------
    idzip = zip(*np.tril_indices(values.shape[0],-1))
    #
    # --------------------------------------------------------------------------
    # Initialize output
    # --------------------------------------------------------------------------
    Cn = 0.0 + rho
    #
    # --------------------------------------------------------------------------
    # Set N polar coordinate samples:
    ## N should be odd (even number of intervals), N_rad ~ 2 * N_theta
    # --------------------------------------------------------------------------
    N_rad = N_uv + (1 -  N_uv%2)
    N_theta = N_uv/2 + (1 -  (N_uv/2)%2)
    #
    # --------------------------------------------------------------------------
    # Set up integral sample space:
    ## rad_inf ~ inf up to error
    # --------------------------------------------------------------------------
    rad = np.linspace(0,rad_inf,N_rad)
    theta = np.linspace(0,2*np.pi,N_theta)
    #
    # --------------------------------------------------------------------------
    # zi in polar coordinates
    # --------------------------------------------------------------------------
    zi = rad[:,None]*np.cos(theta)
    #
    # --------------------------------------------------------------------------
    # Generate Simpson's method error noise
    ## most of the error comes from the angular integration
    ## step size: h = 2*pi / (N/2) = 4*pi/N
    ## error ~ O(h**4)
    # --------------------------------------------------------------------------
    tol = (4*np.pi / N_uv)**4
    for n in range(len(idzip)):
        i,j = idzip[n]
        #
        # ----------------------------------------------------------------------
        # zj in polar coordinates
        # ----------------------------------------------------------------------
        zj = rad[:,None]*( rho[n]*np.cos(theta) + m.sqrt(max(1.0-rho[n]**2, 0.0))*np.sin(theta) )
        #
        # ----------------------------------------------------------------------
        # Inverse transform zi, zj, form product Xi * Xj
        # ----------------------------------------------------------------------
        XiXj = Finv(norm.cdf(zi),values[i],marginals[i]) * Finv(norm.cdf(zj),values[j],marginals[j])
        #
        # ----------------------------------------------------------------------
        # Compute Expected Value(Xi * Xj)
        ## Double integral by Simpsons rule
        ## + random noise ~ O(Simpson's Rule Error)
        ## This prevents limit cycles in the minimization
        # ----------------------------------------------------------------------
        E_XiXj = simps( rad * irbnd_pdf(rad) * simps(XiXj , theta, axis =-1), rad) + (2.0*np.random.rand()-1.0)*tol
        #
        # ----------------------------------------------------------------------
        # Compute and output correlation for each pair
        # ----------------------------------------------------------------------
        Cn[n] = (E_XiXj - sq_means[n])/m.sqrt(sq_vars[n])
    return Cn
################################################################################
#
#
#
################################################################################
def K_n(x,y_min,y_max):
    #
    # --------------------------------------------------------------------------
    # Correlation function "Jacobian" ie: d C_ij / d rho_Z
    ## Given: d C_ij / d rho_Z >= 0
    ## Given: C_ij(+1) = rho_X_max
    ## Given: C_ij(-1) = rho_X_min
    ## Given: C_ij(0) = 0
    ## Given: C_ij continuous
    ## Assume: C_ij ~ quadratic fit with data on endpoints and at zero
    ## d C_ij / d rho_Z = d/d rho_Z { quadratic fit to C_ij }
    ## This will speed things up significantly
    ## and does not significantly hurt the minimization
    ## becasue of the "nice" properties of C_ij (Cario & Nelson, 1997)
    # --------------------------------------------------------------------------
    return (y_max - y_min) / 2. + (y_max + y_min) * x
################################################################################
#
#
#
################################################################################
def large_corr_step_size_factor(rho):
    #
    # --------------------------------------------------------------------------
    # the correlation (rho) is a vector (Mx1 array)
    ## M = number of lower triangular elements in rho_X
    ## When elements of rho are very close to +-1.0, the gradient descent step
    ## size is too large and occasionally pushes the updated value
    ## outside of [-1,1].
    ## This scales down the step size in these cases by
    ## a factor between 0.1 - 1.
    ## This is performed on each element and the minimum step
    ## size scaling is used.
    # --------------------------------------------------------------------------
    return np.maximum( 1. - (1.-0.1)*np.exp(500.*(rho**2 - 1.0)) , 0.1 ).min()
################################################################################
#
#
#
################################################################################
def NORTA_sample(values,marginals,marginal_pdf,covariance,size,tol_dist,tol_corr,N_uv,max_iters):
    #
    # --------------------------------------------------------------------------
    # Core NORTA inversion:
    ## Covariance matrix is an N-by-N array
    ## values and marginals are N-by-k (possibly k-by-N) arrays with
    ## N = dimension of random vector
    ## k = number of left bin edges on the marginal distributions
    ## This is intended for histogram (data)-derived marginals (piecewise linear)
    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    # Test if argument array sizes are consistent
    # --------------------------------------------------------------------------
    if len(marginals.shape) != 2 or len(covariance.shape) != 2:
        raise TypeError('Array of marginals and covariance matrix array must be 2-dimensional')
    elif covariance.shape[0] != covariance.shape[1]:
        raise TypeError('covariance matrix array must be square')
    elif covariance.shape[0] not in marginals.shape:
        raise TypeError('covariance matrix and marginals arrays must share one dimension')
    elif values.shape != marginals.shape:
        raise TypeError('values array and marginals array must have like dimensions')
    else:
        N,N = covariance.shape
        k= marginals.shape[1-marginals.shape.index(N)]
        if marginals.shape.index(N) != 0:
            marginals = marginals.T
            values = values.T
        else:
            marginals = marginals
            values = values
        ########################################################################
        print('-------------------------------------------------')
        print('-------------------------------------------------')
        print('-------------------------------------------------')
        print('---------------- BEGINNING NORTA ----------------')
        print('-------------------------------------------------')
        print('-------------------------------------------------')
        print('-------------------------------------------------')
        dvalues = np.diff(values, axis=-1)[:,0]
        #
        # ----------------------------------------------------------------------
        # Compute marginal mean, var
        ## for a piecewise constant pdf (histogram)
        # ----------------------------------------------------------------------
        marginal_mean = np.sum(marginal_pdf * (values + (1./2.)*dvalues[:,None]) * dvalues[:,None],axis = -1)
        marginal_var = np.sum(marginal_pdf * (values**2 + values*dvalues[:,None] + (1./3.)*dvalues[:,None]**2) * dvalues[:,None],axis = -1) - marginal_mean**2
        corr = covariance/(np.sqrt(marginal_var[:,None]*marginal_var[None,:]))
        #
        # ----------------------------------------------------------------------
        # Generate output lower triangular matrices
        ## and get indices of lower triangles
        ## as a list of index pairs
        # ----------------------------------------------------------------------
        lower_triangle_X = np.tril(corr,-1)
        lower_triangle_Z = np.tril(corr,-1)
        idx_ltri = np.tril_indices(N,-1)
        idzip = zip(*idx_ltri)
        M = len(idx_ltri[0])
        rho_X = lower_triangle_X[idx_ltri]
        #
        # ----------------------------------------------------------------------
        # Get min, max, mean, and variance from marginals
        ## Marginals should have compact support
        # ----------------------------------------------------------------------
        max_sqvals = (values[:,-1][:,None]*values[:,-1][:,None].T)[idx_ltri]
        min_sqvals = (values[:,0][:,None]*values[:,0][:,None].T)[idx_ltri]
        sq_means = (marginal_mean[:,None]*marginal_mean[:,None].T)[idx_ltri]
        sq_vars = (marginal_var[:,None]*marginal_var[:,None].T)[idx_ltri]
        #
        # ----------------------------------------------------------------------
        # Standard normal vector is zero-mean:
        # ----------------------------------------------------------------------
        mu_Z = np.zeros(N)
        #
        # ----------------------------------------------------------------------
        # Set tolerance for the expectation integral and
        ## to calculate rad_inf
        ## (minimum approximation of infinite radius)
        # ----------------------------------------------------------------------
        eps = 1e-6
        rad_inf = np.sqrt( 2*np.log(max_sqvals.max()) - 2*np.log(eps* min_sqvals.min()) )
        print('------------- CALCULATING MIN , MAX -------------')
        print('-------------------------------------------------')
        #
        # ----------------------------------------------------------------------
        # Calculate Min, Max of Correlation Function
        # ----------------------------------------------------------------------
        C_min  = C_n(-1.0 + 0.0*rho_X ,sq_means,sq_vars,values,marginals,rad_inf,N_uv)
        C_max  = C_n( 1.0 + 0.0*rho_X ,sq_means,sq_vars,values,marginals,rad_inf,N_uv)
        ########################################################################
        ########################################################################
        ########################################################################
        print('-------------- BEGINNING ITERATION --------------')
        print('-------------------------------------------------')
        #
        # ----------------------------------------------------------------------
        # Clock minimization start time
        # ----------------------------------------------------------------------
        start = time.time()
        #
        # ----------------------------------------------------------------------
        # Begin Iteration:
        ## Initialize variables
        ## Mean error per channel marginal distribution
        ## Mean error per channel-pair correlation
        # ----------------------------------------------------------------------
        err_dist_per = 1.
        err_corr_per = 1.
        #
        # ----------------------------------------------------------------------
        # Initialize rho_Z guess (= rho_X),
        ## interation start time,
        ## and iteration count
        # ----------------------------------------------------------------------
        rho_Z0 = 0.0 + rho_X
        t0 = time.time()
        iter = 1
        #
        # ----------------------------------------------------------------------
        # Run Forward Model and Approximate Jacobian
        # ----------------------------------------------------------------------
        C0 = C_n(rho_Z0 ,sq_means,sq_vars,values,marginals,rad_inf,N_uv)
        J0 = K_n(rho_Z0 , C_min, C_max)
        #
        # ----------------------------------------------------------------------
        # Initial Step Size = 1. unless rho_Z0 very close to 1.0
        ## minimum initial step = 0.1
        # ----------------------------------------------------------------------
        ds = large_corr_step_size_factor(rho_Z0)
        #
        # ----------------------------------------------------------------------
        # Update rho_Z estimate and
        ## maintain rho_Z within 1e-6 of +- 1.0
        ## Compute model - data error
        ## record iteration end time
        # ----------------------------------------------------------------------
        rho_Z1 = np.maximum( np.minimum(rho_Z0 - ds*(C0 - rho_X)*J0 , 1.0 - 1e-6) , -1.0 + 1e-6)
        rho_Z2 = 0. + rho_Z1
        err = C0 - rho_X
        t1 = time.time()
        print('-------------------------------------------------')
        print 'Iteration Number: {n1}'.format(n1 = iter)
        print 'Total C(rho_Z)-rho_X Square Error = {n1}'.format(n1 = err.dot(err))
        print 'Iteration Time = {n1}'.format(n1 = t1 - t0)
        print('-------------------------------------------------')
        print('#')
        print('#')
        print('#')
        #
        # ----------------------------------------------------------------------
        # Begin modified Barzilai - Borwein (BB) iteration:
        # ----------------------------------------------------------------------
        while (err_dist_per > tol_dist or err_corr_per > tol_corr) and iter <= max_iters:
            #
            # ------------------------------------------------------------------
            # clock iteration start time
            ## augment iteration count
            # ------------------------------------------------------------------
            t0 = time.time()
            iter += 1
            #
            # ------------------------------------------------------------------
            # Run Forward Model and Approximate Jacobian
            # ------------------------------------------------------------------
            C1  = C_n(rho_Z1 ,sq_means,sq_vars,values,marginals,rad_inf,N_uv)
            J1 = K_n(rho_Z1 , C_min, C_max)
            #
            # ------------------------------------------------------------------
            # Compute current and last step
            ## gradients and BB step size (gamma)
            # ------------------------------------------------------------------
            grad0 = (C0 - rho_X)*J0
            grad1 = (C1 - rho_X)*J1
            drho_Z = rho_Z1 - rho_Z0
            dgrad = grad1 - grad0
            gamma = drho_Z.dot(dgrad)/(dgrad.dot(dgrad))
            #
            # ------------------------------------------------------------------
            # Scale BB step size to prevent
            ## too agressive a step if |rho_Z1| ~ 1
            # ------------------------------------------------------------------
            ds = large_corr_step_size_factor(rho_Z1)
            #
            # ------------------------------------------------------------------
            # Update rho_Z estimate and
            ## maintain rho_Z within 1e-6 of +- 1.0
            ## Update last step data
            ## Compute model - data error
            ## record iteration end time
            # ------------------------------------------------------------------
            rho_Z2 = np.maximum( np.minimum(rho_Z1 - ds*gamma*grad1  , 1.0 - 1e-6) , -1.0 + 1e-6)
            C0 = 0. + C1
            J0 = 0. + J1
            rho_Z0 = 0. + rho_Z1
            rho_Z1 = 0. + rho_Z2
            err = C1 - rho_X
            t1 = time.time()
            print('-------------------------------------------------')
            print 'Iteration Number: {n1}'.format(n1 = iter)
            print 'Total C(rho_Z)-rho_X Square Error = {n1}'.format(n1 = err.dot(err))
            print 'Iteration Time = {n1}'.format(n1 = t1 - t0)
            print('#')
            #
            # ------------------------------------------------------------------
            # Generate candidate standard normal vector (Z)
            ## index rho_Z values to corr/cov matrix.
            ## Generate samples of Z
            ## initialize samples and pdf of X
            # ------------------------------------------------------------------
            lower_triangle_Z[idx_ltri] = rho_Z2
            cov_Z = lower_triangle_Z + lower_triangle_Z.T + np.eye(N)
            Z_samples = np.random.multivariate_normal(mu_Z,cov_Z,size)
            X_samples = 0. * Z_samples
            pdf = 0.0*marginals[:,0:-1]
            ####################################################################
            #
            # ------------------------------------------------------------------
            # For each element:
            ## Inverse Marginal Transform of standard normal vector element
            ## Generate element marginal distribution where
            ## PDF = histogram of samples (piecwise constant PDF).
            ## Generate marginal from PDF (CDF is piecwise linear)
            ## with same array shape
            # ------------------------------------------------------------------
            for n in range(N):
                X_samples[:,n] = Finv(norm.cdf(Z_samples[:,n]),values[n],marginals[n])
                pdf[n,:],x = np.histogram(X_samples[:,n], bins = values[n], density = True)
            SUM = np.cumsum(dvalues[:,None]*pdf,axis = -1)
            sample_marginals = np.concatenate((0.*dvalues[:,None]  , SUM), axis =-1)
            ####################################################################
            #
            # ------------------------------------------------------------------
            # Generate covariance and correlation matrices
            ## from sampled target random vector (X)
            # ------------------------------------------------------------------
            sample_cov= np.cov(X_samples.T)
            sample_var = np.diag(sample_cov)
            sample_corr = sample_cov/(np.sqrt(sample_var[:,None]*sample_var[None,:]))
            ####################################################################
            #
            # ------------------------------------------------------------------
            # Generate error measurements:
            ## Distribution error: mean L2-Norm per element (Simpson's Integration)
            ## Correlation error: mean square error per pair of distributions
            # ------------------------------------------------------------------
            err_dist = np.sqrt( simps((sample_marginals - marginals)**2 , values, axis = -1)/(values[:,-1]-values[:,0]) )
            tot_err_dist = err_dist.dot(err_dist)
            err_dist_per = tot_err_dist/float(N)
            err_corr_X = (sample_corr - corr)[idx_ltri]
            tot_err_corr = err_corr_X.dot(err_corr_X)
            err_corr_per = tot_err_corr/float(M)
            #
            # ------------------------------------------------------------------
            # Repeat until errors are below tolerances
            ## or iteration exceeds max_iters
            # ------------------------------------------------------------------
            print 'Distribution Error per Channel = {n1}'.format(n1 = err_dist_per)
            print 'Corr Error per Channel Pair = {n1}'.format(n1 = err_corr_per)
            print('-------------------------------------------------')
            print('#')
            print('#')
            print('#')
        ########################################################################
        ########################################################################
        #
        # ----------------------------------------------------------------------
        # Output = size-by-N array of sample random vectors
        # ----------------------------------------------------------------------
        stop = time.time()
        print 'TOTAL TIME = {n1}'.format(n1 = stop - start)
        errors = np.array([err_corr_per, err_dist_per])
        return X_samples,cov_Z,errors,iter
        #
    #
################################################################################
#
#
#
################################################################################
def bg_spectrum_sampler(path_to_cov_file, path_to_marginals_file, path_to_output_file, tol_dist = 1e-5, tol_corr = 5e-5, N_uv = 200, N_s = 10000, max_iters = 15):
    # --------------------------------------------------------------------------
    #
    # Generate a new NetCDF file (path_to_output_file) which contains
    ## samples of the background spectrum characterized by the
    ## covariance matrix and marginal distributions
    ## contained in path_to_cov_file, path_to_marginals_file.
    ##
    ## use the NORTA parameters:
    ## tol_dist = channel marginal distribution error tolerance
    ## tol_corr = channel pair correlation error tolerance
    ## N_uv = number of 1D integration samples
    ## N_s = number of output random spectrum samples
    ## max_iters = max number of NORTA inversion gradient descent iterations
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    #
    # Clock Start Time
    # --------------------------------------------------------------------------
    print('START TIME:')
    print time.ctime(time.time())
    # --------------------------------------------------------------------------
    #
    # Only proceed if file does not already exist
    # --------------------------------------------------------------------------
    if not os.path.isfile(path_to_output_file):
        print('#')
        print('#')
        print('-------------------------------------------------')
        print('-------------------------------------------------')
        print('-------------------------------------------------')
        print('-------------------------------------------------')
        print('-------------------------------------------------')
        print('SAMPLING BACKGROUND SPECTRUM:')
        print('Covariance from: ' + path_to_cov_file)
        print('Marginals from: ' + path_to_marginals_file)
        print('---------------------')
        print('#')
        print('#')
        print('#')
        print('#')
        # ----------------------------------------------------------------------
        #
        # From Covariance, Marginals files (NetCDF), Extract:
        ## wavenumbers for each channel (wnum_mw)
        ## covariance matrix for each channel (cov)
        ## brightness temperatures values for each channel (BT_mw)
        ## PDF, CDF for each channel
        # ----------------------------------------------------------------------
        fileC = Dataset(path_to_cov_file,'r')
        covariance = fileC['cov_mw'][:]
        wavenumbers = fileC['wnum_mw'][:]
        m = len(wavenumbers)
        fileC.close()
        fileM = Dataset(path_to_marginals_file,'r')
        values = fileM['BT_mw'][:]
        marginal_pdf = fileM['PDF'][:]
        marginals = fileM['CDF'][:]
        season = fileM['bin_season'][:]
        lat = fileM['bin_latitude'][:]
        lon = fileM['bin_longitude'][:]
        fileM.close()
        # ----------------------------------------------------------------------
        #
        # Generate mean, variance from piecewise-constant PDFs
        ## (vectorized, elementwise computation)
        ## replace main diagonal of covariance with
        ## variances computed from PDFs
        # ----------------------------------------------------------------------
        dvalues = np.diff(values, axis=-1)[:,0]
        marginal_mean = np.sum(marginal_pdf * (values + (1./2.)*dvalues[:,None]) * dvalues[:,None],axis = -1)
        marginal_var = np.sum(marginal_pdf * (values**2 + values*dvalues[:,None] + (1./3.)*dvalues[:,None]**2) * dvalues[:,None],axis = -1) - marginal_mean**2
        covariance = covariance - np.diag(np.diag(covariance)) + np.diag(marginal_var)
        # ----------------------------------------------------------------------
        #
        # Call to NORTA sampling function
        # ----------------------------------------------------------------------
        bg_spectra, cov_Z, errors, iter = NORTA_sample(
            values,
            marginals,
            marginal_pdf,
            covariance,
            N_s,
            tol_dist,
            tol_corr,
            N_uv,
            max_iters
            )
        ########################################################################
        ########################################################################
        # ----------------------------------------------------------------------
        #
        # Generate NetCDF file of NORTA output
        # ----------------------------------------------------------------------
        print('-------------------------------------------------')
        print('GENERATING NETCDF FILE:')
        dataset = Dataset(path_to_output_file, 'w')
        # DIMENSIONS
        n_wnum = dataset.createDimension('wnum', m)
        n_samples = dataset.createDimension('samples', N_s)
        nchar = dataset.createDimension('str_dim', 1)
        bin_ends = dataset.createDimension('bin_ends', 2)
        # VARIABLES
        samples = dataset.createVariable('bg_spectral_samples',np.float32, ('samples','wnum'))
        norta_cov = dataset.createVariable('norta_cov',np.float32, ('wnum','wnum'))
        wnum = dataset.createVariable('wnum',np.float32, ('wnum',))
        err_corr = dataset.createVariable('CPCE',np.float32) # Channel Pair Correlation Error
        err_dist = dataset.createVariable('CMDE',np.float32) # Channel Marginal Distribution Error
        bin_season = dataset.createVariable('bin_season',str, ('str_dim',))
        bin_latitude = dataset.createVariable('bin_latitude',np.int16, ('bin_ends',))
        bin_longitude = dataset.createVariable('bin_longitude',np.int16, ('bin_ends',))
        # GLOBAL ATTRIBUTES
        dataset.description = 'Correlated background spectral samples conforming to ' \
            + 'the measured channel marginal distribution and covariance matrix ' \
            + 'generated by the NORTA (NORmal To Anything) procedure for non-normal marginals.' \
            + 'These sample cover SO2 - free background brightness temperature spectra representing ' \
            +'a 5 deg latitude x 5 deg longitude x season binning.'
        dataset.history = 'Created ' + time.ctime(time.time())
        # VARIABLE ATTRIBUTES
        samples.units = 'K'
        norta_cov.units = 'none'
        err_corr.units = 'Error per channel'
        err_dist.units = 'Error per channel pair'
        wnum.units = 'cm^-1'
        bin_latitude.units = 'degree_north'
        bin_longitude.units = 'degree_east'
        samples.description = 'NORTA - derived correlated random spectrum samples'
        norta_cov.description = 'NORTA - derived multivariate standard normal covariance'
        wnum.description = 'CrIS FSR midwave wavenumbers used in sample'
        err_corr.description = 'Channel Pair Correlation coefficient Error per channel pair'
        err_dist.description = 'Channel Marginal Distribution Error - L2 mean value norm on each channel'
        bin_season.description = 'right half-open time (seasonal) interval'
        bin_latitude.description = 'right half-open latitude interval'
        bin_longitude.description = 'right half-open longitude interval'
        # ADD VALUES TO VARIABLES
        #
        samples[:] = np.float32(bg_spectra)
        norta_cov[:] = np.float32(cov_Z)
        wnum[:] = np.float32(wavenumbers)
        err_corr[:] = np.float32(errors[0])
        err_dist[:] = np.float32(errors[1])
        bin_season[:] = season
        bin_latitude[:] = lat
        bin_longitude[:] = lon
        # WRITE FILE
        dataset.close()
    ############################################################################
    ############################################################################
    ############################################################################
    ############################################################################
    print('END TIME:')
    print time.ctime(time.time())
    print('-------------------------------------------------')
    print('-------------------------------------------------')
################################################################################
