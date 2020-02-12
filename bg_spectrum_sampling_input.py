"""
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
This script contains the inputs to generate background spectrum
samples from cchannel covariance and marginal distribution
characterization files for use in a probabilistic trace gas retrieval
(Hyman and Pavolonis, 2020).

Hyman, D. M., and Pavolonis, M. J.: Probabilistic retrieval of volcanic
SO2 layer height and cumulative mass loading using the Cross-track Infrared
Sounder (CrIS), Atmospheric  Measurement  Techniques, (in review), 2020.
"""
import sys
sys.path.append('trace_gas_background_spectra/')
from bg_spectrum_sampling import *
# ------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Sampling Inputs:
# ------------------------------------------------------------------------------
path_to_cov_file = 'CrIS.bg.example.cov.nc'
#
path_to_marginals_file = 'CrIS.bg.example.marginals.nc'
#
path_to_output_file = 'CrIS.bg.example.spectra.nc'
#
tol_dist = 1e-5
#
tol_corr = 5e-5
#
N_uv = 200
#
N_s = 10000
#
max_iters = 15
#
################################################################################
################################################################################
# ------------------------------------------------------------------------------
#
# Call to sampler function
# ------------------------------------------------------------------------------
bg_spectrum_sampler(
    path_to_cov_file = path_to_cov_file,
    path_to_marginals_file = path_to_cov_file,
    path_to_output_file = path_to_output_file,
    tol_dist = tol_dist,
    tol_corr = tol_corr,
    N_uv = N_uv,
    N_s = N_s,
    max_iters = max_iters
    )
#
################################################################################
################################################################################
################################################################################
