#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 10:13:52 2018

@author: dave.hyman(at)ssec.wisc.edu  --or--  dhyman2(at)wisc.edu

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
This function generates samples of random vectors of dimension N:
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
vector Y such that Y_i = inv(F_Yi)[Phi(Zi)]
where Phi is the univariate standard normal CDF and the inv(F_Yi) is the
inverse of the marginal CDF for the i-th component of Y.

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""
################################################################################
# PRELIMINARY DEFINITIONS ------------------------------------------------------
################################################################################
import numpy as np
import math as m
import time
from scipy.stats import norm
from scipy.integrate import simps
################################################################################
# INDEPENDENT BIVARIATE NORMAL PDF ---------------------------------------------
################################################################################
def irbnd_pdf(r):
    # irbnd = Independent Radial Bivariate Normal Distribution
    return (
        1. / (2 * np.pi) \
        * np.exp( (-1./2) * (r**2) ) \
        )
################################################################################
# INVERSE MARGINAL DISTRIBUTION: X = F^-1(U) -----------------------------------
################################################################################
def Finv(U,X,F):
    F[0] = 0.0
    F[-1] = 1.0
    return np.interp(U,F,X)
################################################################################
# CORRELATION FUNCTION: rho_X = C_ij(rho_Z) ------------------------------------
################################################################################
def C_n(rho,sq_means,sq_vars,values,marginals,rad_inf,N_uv):
    idzip = zip(*np.tril_indices(values.shape[0],-1))
    Cn = 0.0 + rho
    N_rad = N_uv + (1 -  N_uv%2)
    N_theta = N_uv/2 + (1 -  (N_uv/2)%2)
    rad = np.linspace(0,rad_inf,N_rad)
    theta = np.linspace(0,2*np.pi,N_theta)
    zi = rad[:,None]*np.cos(theta)
    tol = (4*np.pi / N_uv)**4
    for n in range(len(idzip)):
        i,j = idzip[n]
        zj = rad[:,None]*( rho[n]*np.cos(theta) + m.sqrt(max(1.0-rho[n]**2, 0.0))*np.sin(theta) )
        YiYj = Finv(norm.cdf(zi),values[i],marginals[i]) * Finv(norm.cdf(zj),values[j],marginals[j])
        E = simps( rad * irbnd_pdf(rad) * simps(YiYj , theta, axis =-1), rad) + (2.0*np.random.rand()-1.0)*tol
        Cn[n] = (E - sq_means[n])/m.sqrt(sq_vars[n])
    return Cn
################################################################################
# CORRELATION FUNCTION JACOBIAN: d C_ij / d rho_Z ------------------------------
################################################################################
# Given: d C_ij / d rho_Z >= 0
# Given: C_ij(+1) = rho_X_max
# Given: C_ij(-1) = rho_X_min
# Given: C_ij(0) = 0
# Given: C_ij continuous
# TRY:  d C_ij / d rho_Z = d/d rho_Z { quadratic fit to C_ij }
def K_n(x,y_min,y_max):
    return (y_max - y_min) / 2. + (y_max + y_min) * x
################################################################################
# BEGIN NORTA FUNCTION----------------------------------------------------------
################################################################################
def NORTA_sample(values,marginals,marginal_pdf,covariance,size,tol_dist,tol_corr,N_uv,max_iters):
    #############################
    #############################
    #############################
    #############################
    # covariance matrix is an N-by-N array
    # values and marginals are N-by-k (possibly k-by-N) arrays with
    # N = dimension of random vector
    # k = number of left bin edges on the marginal distributions
    # This is intended for histogram (data)-derived marginals (piecewise linear)
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
        marginal_mean = np.sum(marginal_pdf * (values + (1./2.)*dvalues[:,None]) * dvalues[:,None],axis = -1)
        marginal_var = np.sum(marginal_pdf * (values**2 + values*dvalues[:,None] + (1./3.)*dvalues[:,None]**2) * dvalues[:,None],axis = -1) - marginal_mean**2
        corr = covariance/(np.sqrt(marginal_var[:,None]*marginal_var[None,:]))
        lower_triangle_X = np.tril(corr,-1)
        lower_triangle_Z = np.tril(corr,-1)
        idx_ltri = np.tril_indices(N,-1)
        mu_Z = np.zeros(N)
        idzip = zip(*idx_ltri)
        M = len(idx_ltri[0])
        rho_X = lower_triangle_X[idx_ltri]
        max_sqvals = (values[:,-1][:,None]*values[:,-1][:,None].T)[idx_ltri]
        min_sqvals = (values[:,0][:,None]*values[:,0][:,None].T)[idx_ltri]
        sq_means = (marginal_mean[:,None]*marginal_mean[:,None].T)[idx_ltri]
        sq_vars = (marginal_var[:,None]*marginal_var[:,None].T)[idx_ltri]
        eps = 1e-6
        rad_inf = np.sqrt( 2*np.log(max_sqvals.max()) - 2*np.log(eps* min_sqvals.min()) )
        print('------------- CALCULATING MIN , MAX -------------')
        print('-------------------------------------------------')
        C_min  = C_n(-1.0 + 0.0*rho_X ,sq_means,sq_vars,values,marginals,rad_inf,N_uv)
        C_max  = C_n( 1.0 + 0.0*rho_X ,sq_means,sq_vars,values,marginals,rad_inf,N_uv)
        ########################################################################
        ########################################################################
        ########################################################################
        print('-------------- BEGINNING ITERATION --------------')
        print('-------------------------------------------------')
        # BEGIN ITERATION
        err_dist_per = 1.
        err_corr_per = 1.
        #
        start = time.time()
        rho_Z0 = 0.0 + rho_X
        t0 = time.time()
        iter = 1
        C0 = C_n(rho_Z0 ,sq_means,sq_vars,values,marginals,rad_inf,N_uv)
        J0 = K_n(rho_Z0 , C_min, C_max)
        ds = np.maximum( 1. - (1.-0.1)*np.exp(500.*(rho_Z0**2 - 1.0)) , 0.1 ).min()
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
        while (err_dist_per > tol_dist or err_corr_per > tol_corr) and iter <= max_iters:
            t0 = time.time()
            iter += 1
            C1  = C_n(rho_Z1 ,sq_means,sq_vars,values,marginals,rad_inf,N_uv)
            J1 = K_n(rho_Z1 , C_min, C_max)
            grad0 = (C0 - rho_X)*J0
            grad1 = (C1 - rho_X)*J1
            drho_Z = rho_Z1 - rho_Z0
            dgrad = grad1 - grad0
            gamma = drho_Z.dot(dgrad)/(dgrad.dot(dgrad))
            ds = np.maximum( 1. - (1.-0.1)*np.exp(500.*(rho_Z1**2 - 1.0)) , 0.1 ).min()
            rho_Z2 = np.maximum( np.minimum(rho_Z1 - ds*gamma*grad1  , 1.0 - 1e-6) , -1.0 + 1e-6)
            # SET VALUES FOR NEXT STEP
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
            lower_triangle_Z[idx_ltri] = rho_Z2
            cov_Z = lower_triangle_Z + lower_triangle_Z.T + np.eye(N)
            Z_samples = np.random.multivariate_normal(mu_Z,cov_Z,size)
            X_samples = 0. * Z_samples
            F = 0.0*marginals
            f = 0.0*F[:,0:-1]
            ####################################################################
            # TRANSFORM STANDARD NORMAL RANDOM VECTOR
            # BY INVERSE MARGINAL DISTRIBUTION TRANSFORM
            # AND GENERATE DISTRIBUTION FOR SAMPLE RANDOM VECTORS
            for n in range(N):
                X_samples[:,n] = Finv(norm.cdf(Z_samples[:,n]),values[n],marginals[n])
                f[n,:],x = np.histogram(X_samples[:,n], bins = values[n], density = True)
            SUM = np.cumsum(dvalues[:,None]*f,axis = -1)
            sample_marginals = np.concatenate((0.*dvalues[:,None]  , SUM), axis =-1)
            ####################################################################
            # GENERATE COVARIANCE AND CORRELATION FOR SAMPLED RANDOM VECTOR
            #
            sample_cov= np.cov(X_samples.T)
            sample_var = np.diag(sample_cov)
            sample_corr = sample_cov/(np.sqrt(sample_var[:,None]*sample_var[None,:]))
            ####################################################################
            # GENERATE ERROR MEASUREMENTS: DISTRIBUTION & CORRELATION STRUCTURE
            # DISTRIBUTION ERROR = L2 mean-value NORM per channel
            # CORR ERROR = square error per pair of distributions
            err_dist = np.sqrt( simps((sample_marginals - marginals)**2 , values, axis = -1)/(values[:,-1]-values[:,0]) )
            tot_err_dist = err_dist.dot(err_dist)
            err_dist_per = tot_err_dist/float(N)
            err_corr_X = (sample_corr - corr)[idx_ltri]
            tot_err_corr = err_corr_X.dot(err_corr_X)
            err_corr_per = tot_err_corr/float(M)
            print 'Distribution Error per Channel = {n1}'.format(n1 = err_dist_per)
            print 'Corr Error per Channel Pair = {n1}'.format(n1 = err_corr_per)
            print('-------------------------------------------------')
            print('#')
            print('#')
            print('#')
        ########################################################################
        ########################################################################
        # OUTPUT = size-by-N ARRAY OF SAMPLE RANDOM VECTORS
        stop = time.time()
        print 'TOTAL TIME = {n1}'.format(n1 = stop - start)
        errors = np.array([err_corr_per, err_dist_per])
        return X_samples,cov_Z,errors,iter
        ########################################################################
    ############################################################################
################################################################################
