# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
functions

"""

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
# Import Python packages and C functions.

"""

#!python
#cython: language_level=3

import cython
cimport cython
import numpy as np
import copy
from scipy import interpolate
from scipy import integrate

from libc.math cimport int, round, ceil, floor, len
from libc.math cimport log, exp, sqrt, pow, erf, sin, cos, pi
from libc.stdlib cimport srand, rand, RAND_MAX

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
Functions for model parameters.

"""

# total number of model parameters
DEF N_phi = 9

# function for the drift rate
@cython.cdivision(True)
cdef double drift(double phi[N_phi], double x, double t):
    cdef double mu = phi[0]
    cdef double l = phi[1]
    cdef double k = phi[2]
    cdef double t_flip = phi[3]
    cdef double v = 0.0
    
    if (t < t_flip):
        v = mu*(1.0 + k*t) - (l - k/(1.0 + k*t) )*x
    else:
        v = -mu*(1.0 + k*t) - (l - k/(1.0 + k*t) )*x

    return v

# function for the diffusion rate
@cython.cdivision(True)
cdef double diffusion(double phi[N_phi], double x, double t):
    cdef double k = phi[2]
    cdef double sigma = phi[4]
    cdef double D = sigma*(1.0 + k*t)
    return D

# function for the decision threshold
@cython.cdivision(True)
cdef double threshold(double phi[N_phi], double t):
    return phi[5]

# function for the decision threshold derivative
@cython.cdivision(True)
cdef double threshold_derivative(double phi[N_phi], double t):
    return 0.0

# function for the contamination strength
@cython.cdivision(True)
cdef double contamination_strength(double phi[N_phi]):
    return phi[6]

# function for the contamination probability distribution
@cython.cdivision(True)
cdef double contamination_probability(double phi[N_phi], double t):
    cdef double gl = phi[7]
    cdef double gu = phi[8]
    cdef double pg = 0.0
    if (t >= gl) and (t <= gu):
        pg = 1.0/(gu - gl)
    return pg

# function to modify the time step
@cython.cdivision(True)
cdef double modify_dt(double phi[N_phi], double t):
    return 1.0

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
Parameters and functions for across-trial variability.

"""

# set max and min value for the relative start point (for stability at boundaries)
DEF w_min = 0.05
DEF w_max = 0.95

# total number start point and non-deicision related parameters
DEF N_psi = 5

# function for a normally distributed relative start
@cython.cdivision(True)
cdef double relative_start_normal(double psi[N_psi], double eps):
    cdef double mean_w = psi[0]
    cdef double sd_w = psi[1]
    cdef double w = 0.0
    if (eps >= w_min) and (eps <= w_max):
        w = (1.0/sd_w) * f1((eps - mean_w)/sd_w) / ( f2((w_max - mean_w)/sd_w) - f2((w_min - mean_w)/sd_w) )
    return w

# function for a normally distributed non-decision time
@cython.cdivision(True)
cdef double tnd_normal(double psi[N_psi], double tnd):
    cdef double mean_tnd = psi[2]
    cdef double sd_tnd = psi[3]
    cdef double p_tnd = 0.0 
    if (tnd >= 0.0):
        p_tnd = (1.0/sd_tnd) * f1((tnd - mean_tnd)/sd_tnd) / ( 1.0 - f2(-mean_tnd/sd_tnd) )
    return p_tnd

# function for the pdf of the standard normal distribution
@cython.cdivision(True)
cdef double f1(double xi):
    return exp(-0.5*pow(xi, 2.0))/sqrt(2.0*pi)

# function for the cumulative distribution of the standard normal distribution
@cython.cdivision(True)
cdef double f2(double xi):
    return 0.5*(1.0 + erf(xi/sqrt(2.0)))

# function for the distribution of the standard normal distribution for integrating mu
@cython.cdivision(True)
cdef double mu_normal(double mu, double mean_mu, double sd_mu):
    return 1.0/(sd_mu*sqrt(2.0*pi)) * exp( -0.5 * pow( (mu - mean_mu)/sd_mu, 2.0 ) )

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
Defined constants for following functions.

"""

# constants for function likelihood
DEF N_dt = 2500 # sets the maximum number of time steps
DEF N_deps_max = 501 # sets the maximum number of spatial steps
DEF N_dt_ini = 25 # sets the number of initial, small time steps
DEF N_dt_scaleup = 100 # sets the number of scale up time steps
DEF dt_max = 0.1 # sets the largest possible time step
DEF dt_ini_scale = 0.01 # sets the scale of the initial time steps
DEF ds_ratio_cutoff = 0.02 # sets the max threshold collapse ratio
DEF dt_mod_scale = 0.5 # sets the time step change if exceeds ds_ratio_cutoff
DEF threshold_cutoff = 1.0e-4 # sets the minimum threshold value
DEF p_fpt_min = 1.0e-5 # sets the minimum likelihood probability
DEF int_prob_min = 0.25 # used to check if enough probability has accumulated to cutoff solver
DEF gg_max = 0.9999 # sets the maximum contamination value

# constants for integrating likelihood function
DEF N_tnd_max = 250 # max number of non-decision time integrations
DEF N_mu_max = 100 # max number of drift rate integrations
DEF int_range = 3.5 # sets range over which to integrate the drift rate (4 stdevs)
DEF sd_tnd_min = 0.001 # sets the minimum standard deviation values

# constants for function approx_avg_fpt
DEF t_max = 100.0 # simulates only up to t_max seconds
DEF N_afpt = 10 # number of simulated accumulators
DEF dt_afpt_scale = 0.01 # used to set simulated time step
DEF dt_afpt_max = 0.025 # simulated time step cannot excded this value

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
Function which generates the model likelihood.

"""

@cython.cdivision(True)
@cython.boundscheck(False) 
cpdef likelihood(phi, psi, w_dist, tnd_dist, N_tnd, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rt_max):
    
    # -------------------- #
    
    cdef int ii = 0, jj = 0, kk = 0, ll = 0
    
    # -------------------- #
    
    cdef double phic[N_phi]
    cdef double psic[N_psi]

    for ii in range(N_phi):
        phic[ii] = phi[ii]
    
    for ii in range(N_psi):
        psic[ii] = psi[ii]

    cdef double dt_interp_scalec = dt_interp_scale
    cdef double dt_scalec = dt_scale
    cdef double dt_interp_overidec = dt_interp_overide
    cdef double rt_maxc = rt_max
    cdef int N_tndc = N_tnd
    
    cdef int N_depsc = 0

    if (N_deps == 'auto'):
        N_depsc = int(ceil(threshold(phic, 0.0)/diffusion(phic, 0.0, 0.0) * 100.0))
        if (N_depsc < 101):
            N_depsc = 101
        elif (N_depsc > 151):
            N_depsc = 151
    else:
        N_depsc = N_deps
        
    # -------------------- #

    cdef double p_fptc[4][N_dt]
    cdef double p_n[N_deps_max]
    cdef double p_ncn[N_deps_max]
    cdef double p_np1[N_deps_max]
    cdef double eps[N_deps_max]
    cdef double v_n[N_deps_max]
    cdef double v_np1[N_deps_max]
    cdef double sigma_n[N_deps_max]
    cdef double sigma_np1[N_deps_max]
    cdef double sigma2_n[N_deps_max]
    cdef double sigma2_np1[N_deps_max]
    cdef double AA[N_deps_max]
    cdef double BB[N_deps_max]
    cdef double CC[N_deps_max]
    cdef double DD[N_deps_max]
    cdef double EE[N_deps_max]
    cdef double FF[N_deps_max]
    cdef double w_s[2]
    cdef int w_i[2]
    
    for ii in range(4):
        for jj in range(N_dt):
            p_fptc[ii][jj] = 0.0
    
    for ii in range(N_deps_max):
        p_n[ii] = 0.0
        p_ncn[ii] = 0.0
        p_np1[ii] = 0.0
        eps[ii] = 0.0
        v_n[ii] = 0.0
        v_np1[ii] = 0.0
        sigma_n[ii] = 0.0
        sigma_np1[ii] = 0.0
        sigma2_n[ii] = 0.0
        sigma2_np1[ii] = 0.0
        AA[ii] = 0.0
        BB[ii] = 0.0
        CC[ii] = 0.0
        DD[ii] = 0.0
        EE[ii] = 0.0
        FF[ii] = 0.0   
        
    w_s[0] = 0.0
    w_s[1] = 0.0
    w_i[0] = 0
    w_i[1] = 0
    
    # -------------------- #
    
    cdef double s_ini = 2.0*threshold(phic, 0.0)
    cdef double deps = 1.0/(N_depsc - 1.0)
    cdef double dt_base = 0.0, dt_ini = 0.0, dt = 0.0
    cdef double mean_w = psic[0]
    cdef double sd_w = psic[1]
    cdef double a_w = mean_w - 0.5*sd_w
    cdef double b_w = mean_w + 0.5*sd_w
    cdef double p_n_sum = 0.0
    cdef double bu_n = 0.0, dbudt_n = 0.0
    cdef double bl_n = 0.0, dbldt_n = 0.0
    cdef double s_n = 0.0, dsdt_n = 0.0
    cdef double t_in = 0.0
    cdef double bu_np1 = 0.0, dbudt_np1 = 0.0
    cdef double bl_np1 = 0.0, dbldt_np1 = 0.0
    cdef double s_np1 = 0.0, dsdt_np1 = 0.0
    cdef double int_prob = 0.0
    cdef double bu_nm1 = 0.0, bl_nm1 = 0.0
    cdef double s_nm1 = 0.0
    cdef double ds_ratio = 0.0, ds_ratio_np1 = 0.0, ds_ratio_nm1 = 0.0
    cdef double x_n = 0.0
    cdef double x_np1 = 0.0
    cdef double alpha_n = 0.0, alpha_np1 = 0.0
    cdef double beta_n = 0.0, beta_np1 = 0.0
    cdef double gamma_n = 0.0, gamma_np1 = 0.0
    cdef double WW = 0.0
    cdef double tt = 0.0
    cdef double gg = contamination_strength(phic)
    cdef int N_cut = 0
    
    # -------------------- #
    
    # create array of spatial locations
    for ii in range(N_depsc):
        eps[ii] = ii*deps
        
    # calculate time step sizes
    dt_base = dt_scalec*approx_avg_fpt(phic, psic)
    if (dt_base > dt_max):
        dt_base = dt_max
    dt_ini = dt_ini_scale*dt_base
    dt = dt_ini
    
    # determine initial threshold locations
    t_in = dt_ini/10.0
    bu_np1 = threshold(phic, t_in)
    dbudt_np1 = threshold_derivative(phic, t_in)
    bl_np1 = -bu_np1
    dbldt_np1 = -dbudt_np1
    s_np1 = bu_np1 - bl_np1
    dsdt_np1 = dbudt_np1 - dbldt_np1   
    
    # determine initial probability distribution
    
    if (w_dist == 'constant'):
        
        # determine nearest array elements to delta distribution
        w_i[0] = int( ceil(mean_w/deps) )
        w_i[1] = int( floor(mean_w/deps) )
        
        # determine distance between delta distribution and nearest array elements
        w_s[0] = abs( mean_w - deps*w_i[0] )
        w_s[1] = abs( mean_w - deps*w_i[1] )
        
        # numerically approximate delta function, distribute between nearest array elements
        if (w_i[0] == w_i[1]):
            p_n[ w_i[0] ] = 1.0/deps
        else:
            p_n[ w_i[0] ] = (1.0 - w_s[0]/deps)/deps
            p_n[ w_i[1] ] = (1.0 - w_s[1]/deps)/deps
            
    elif (w_dist == 'uniform'):
                
        for ii in range(1, N_depsc-1):
            
            if (a_w < w_min):
                a_w = w_min
            if (b_w > w_max):
                b_w = w_max
            
            if (eps[ii-1] < a_w) and (eps[ii] > a_w):
                p_n[ii-1] = 1.0 - (a_w - eps[ii-1])/deps
                            
            if (eps[ii-1] < b_w) and (eps[ii] > b_w):
                p_n[ii] = 1.0 - (eps[ii] - b_w)/deps
                
            if (eps[ii] >= a_w) and (eps[ii] <= b_w):
                p_n[ii] = 1.0
            
        for ii in range(N_depsc):
            p_n_sum += p_n[ii]
            
        for ii in range(N_depsc):
            p_n[ii] = p_n[ii]/(p_n_sum*deps)
        
    elif (w_dist == 'normal'):
                
        for ii in range(1, N_depsc-1):
            
            if (eps[ii-1] < w_min) and (eps[ii] > w_min):
                p_n[ii-1] = (1.0 - (w_min - eps[ii-1])/deps)*relative_start_normal(psic, w_min)
                
            if (eps[ii-1] < w_max) and (eps[ii] > w_max):
                p_n[ii] = (1.0 - (eps[ii] - w_max)/deps)*relative_start_normal(psic, w_max)
                
            if (eps[ii] >= w_min) and (eps[ii] <= w_max):
                p_n[ii] = relative_start_normal(psic, eps[ii])
            
        for ii in range(N_depsc):
            p_n_sum += p_n[ii]
            
        for ii in range(N_depsc):
            p_n[ii] = p_n[ii]/(p_n_sum*deps)
            
    # -------------------- #
    
    # calculate likelihood function
    for ii in range(N_dt):

        # set decison threshold location at previous time step
        bu_n = bu_np1
        dbudt_n = dbudt_np1
        bl_n = bl_np1
        dbldt_n = dbldt_np1
        s_n = bu_n - bl_n
        dsdt_n = dbudt_n - dbldt_n
                
        # set time step
        if (ii < N_dt_ini):
            dt = dt_ini
        elif (ii >= N_dt_ini) and (ii < N_dt_ini + N_dt_scaleup):
            dt = dt_ini + (ii + 1.0 - N_dt_ini)*(dt_base - dt_ini)/(N_dt_scaleup)
        else:
            dt = dt_base
            
        # decrease time step if decision thresholds decrease too rapidly
        jj = 0
        kk = 0
        while (jj < 1) and (kk < 9):
            t_in = tt + dt
            bu_np1 = threshold(phic, t_in)
            bl_np1 = -bu_np1

            t_in = tt - dt
            if (t_in <= 0.0):
                t_in = dt_ini/10.0
                bu_nm1 = threshold(phic, t_in)
                bl_nm1 = -bu_nm1
            else:
                bu_nm1 = threshold(phic, t_in)
                bl_nm1 = -bu_nm1

            s_np1 = bu_np1 - bl_np1
            s_nm1 = bu_nm1 - bl_nm1

            ds_ratio = 0.0;
            ds_ratio_np1 = abs(s_n - s_np1)/s_n
            ds_ratio_nm1 = abs(s_n - s_nm1)/s_n

            if (ds_ratio_np1 < ds_ratio_nm1):
                ds_ratio = ds_ratio_nm1
            else:
                ds_ratio = ds_ratio_np1

            if (ds_ratio > ds_ratio_cutoff):
                dt = dt*dt_mod_scale
            else:
                jj = 1

            kk += 1

        # calculate new decision thresholds
        dt = modify_dt(phic, tt)*dt
        tt += dt
        
        bu_np1 = threshold(phic, tt)
        dbudt_np1 = threshold_derivative(phic, tt)
    
        if (bu_np1 <= threshold_cutoff):
            bu_np1 = threshold_cutoff
            dbudt_np1 = 0.0
        
        bl_np1 = -bu_np1
        dbldt_np1 = -dbudt_np1

        if (bl_np1 >= -threshold_cutoff):
            bl_np1 = -threshold_cutoff
            dbldt_np1 = 0.0
        
        s_np1 = bu_np1 - bl_np1
        dsdt_np1 = dbudt_np1 - dbldt_np1
        
        # invert matrix using tridiagonal matrix algorithm
        for jj in range(2):
            
            p_np1[jj] = 0.0

            if (ii == 0):
                x_n = s_n*eps[jj] + bl_n
                v_n[jj] = drift(phic, x_n, tt-dt)
                sigma_n[jj] = diffusion(phic, x_n, tt-dt)
                sigma2_n[jj] = sigma_n[jj]*sigma_n[jj]
            else:
                x_n = x_np1
                v_n[jj] = v_np1[jj]
                sigma_n[jj] = sigma_np1[jj]
                sigma2_n[jj] = sigma2_np1[jj]

            x_np1 = s_np1*eps[jj] + bl_np1
            v_np1[jj] = drift(phic, x_np1, tt)
            sigma_np1[jj] = diffusion(phic, x_np1, tt)
            sigma2_np1[jj] = sigma_np1[jj]*sigma_np1[jj]
        
        for jj in range(1, N_depsc-1):

            p_np1[jj+1] = 0.0

            if (ii == 0):
                x_n = s_n*eps[jj+1] + bl_n
                v_n[jj+1] = drift(phic, x_n, tt-dt)
                sigma_n[jj+1] = diffusion(phic, x_n, tt-dt)
                sigma2_n[jj+1] = sigma_n[jj+1]*sigma_n[jj+1]
            else:
                x_n = x_np1
                v_n[jj+1] = v_np1[jj+1]
                sigma_n[jj+1] = sigma_np1[jj+1]
                sigma2_n[jj+1] = sigma2_np1[jj+1]

            x_np1 = s_np1*eps[jj+1] + bl_np1
            v_np1[jj+1] = drift(phic, x_np1, tt)
            sigma_np1[jj+1] = diffusion(phic, x_np1, tt)
            sigma2_np1[jj+1] = sigma_np1[jj+1]*sigma_np1[jj+1]
            
            alpha_n = dt/(4.0*s_n*deps)
            alpha_np1 = dt/(4.0*s_np1*deps)
            
            beta_n = eps[jj]*dsdt_n + dbldt_n
            beta_np1 = eps[jj]*dsdt_np1 + dbldt_np1
            
            gamma_n = alpha_n/(s_n*deps)
            gamma_np1 = alpha_np1/(s_np1*deps)
            
            AA[jj] = alpha_np1*beta_np1 - alpha_np1*v_np1[jj-1] - gamma_np1*sigma2_np1[jj-1]
            BB[jj] = 1.0 + 2.0*gamma_np1*sigma2_np1[jj]
            CC[jj] = -alpha_np1*beta_np1 + alpha_np1*v_np1[jj+1] - gamma_np1*sigma2_np1[jj+1]
            
            DD[jj] = -alpha_n*beta_n + alpha_n*v_n[jj-1] + gamma_n*sigma2_n[jj-1]
            EE[jj] = 1.0 - 2.0*gamma_n*sigma2_n[jj]
            FF[jj] = alpha_n*beta_n - alpha_n*v_n[jj+1] + gamma_n*sigma2_n[jj+1]
            
            p_ncn[jj] = DD[jj]*p_n[jj-1] + EE[jj]*p_n[jj] + FF[jj]*p_n[jj+1]
            
        for jj in range(2, N_depsc - 1):
            WW = AA[jj]/BB[jj-1]
            BB[jj] = BB[jj] - WW*CC[jj-1]
            p_ncn[jj] = p_ncn[jj] - WW*p_ncn[jj-1]
            
        p_np1[N_depsc-2] = p_ncn[N_depsc-2]/BB[N_depsc-2]
        p_n[N_depsc-2] = p_np1[N_depsc-2]
        
        for jj in range(N_depsc - 3, 0, -1):
            p_np1[jj] = (p_ncn[jj] - CC[jj]*p_np1[jj+1])/BB[jj]
            p_n[jj] = p_np1[jj]
            
        # calculate probability of first passage time
        p_fptc[0][ii] = tt
        p_fptc[1][ii] = abs( 0.5 * sigma2_np1[N_depsc-1] * (4.0*p_np1[N_depsc-2] - p_np1[N_depsc-3])/(2.0*deps*s_np1*s_ini) )
        p_fptc[2][ii] = abs( 0.5 * sigma2_np1[N_depsc-1] * (4.0*p_np1[1] - p_np1[2])/(2.0*deps*s_np1*s_ini) )
        p_fptc[3][ii] = 0.5*gg*contamination_probability(phic, tt)
        
        # determine integrated probability
        int_prob += p_fptc[1][ii]*dt + p_fptc[2][ii]*dt
        
        # if p_fpt less than p_fpt_min, set to p_fpt_min
        if (p_fptc[1][ii] <= p_fpt_min):
            p_fptc[1][ii] = p_fpt_min
        if (p_fptc[2][ii] <= p_fpt_min):
            p_fptc[2][ii] = p_fpt_min
                    
        if (p_fptc[0][ii] > rt_maxc):
            N_cut = ii + 1
            break
        elif (int_prob > int_prob_min) and (p_fptc[1][ii] <= p_fpt_min) and (p_fptc[2][ii] <= p_fpt_min):
            N_cut = ii + 1
            break
            
    # -------------------- #
    
    cdef double dt_tnd = 0.0
    cdef double t_tnd = 0.0
    cdef double tnd[N_tnd_max]
    cdef double t_pts[N_tnd_max]
    cdef double p_tu[N_tnd_max]
    cdef double p_tl[N_tnd_max]
    cdef double tempu[N_tnd_max]
    cdef double templ[N_tnd_max]
    cdef double norm_prob[N_tnd_max]
    
    for ii in range(N_tnd_max):
        tnd[ii] = 0.0
        tempu[ii] = 0.0
        templ[ii] = 0.0
        t_pts[ii] = 0.0
        p_tu[ii] = 0.0
        p_tl[ii] = 0.0
        norm_prob[ii] = 0.0
        
    if (psic[3] < sd_tnd_min):
        psic[3] = sd_tnd_min
    
    if (tnd_dist == 'constant'):
        
        p_fpt = np.zeros([4,N_cut+1])
        p_fpt[0,0] = 0.0
        p_fpt[1,0] = p_fpt_min
        p_fpt[2,0] = p_fpt_min
        p_fpt[3,0] = 0.0

        for ii in range(1,N_cut+1):
            p_fpt[0,ii] = p_fptc[0][ii-1]
            p_fpt[1,ii] = p_fptc[1][ii-1]
            p_fpt[2,ii] = p_fptc[2][ii-1]
            p_fpt[3,ii] = p_fptc[3][ii-1]
    
        p_fpt[0,:] += psic[2]
            
        return p_fpt
        
    elif (tnd_dist == 'uniform'):

        p_fpt_integ = np.zeros([4, N_cut])
        
        # generate array of non-decision times
        dt_tnd = (psic[3])/(N_tndc - 1.0)
        for ii in range(N_tndc):
            tnd[ii] = psic[2] - psic[3]/2.0 + ii*dt_tnd
            
        for ii in range(N_cut):
            
            kk = N_cut-1
            
            for jj in range(N_tndc):

                t_pts[jj] = p_fptc[0][ii] - tnd[jj]
                
                if (t_pts[jj] <= 0.0):
                    p_tu[jj] = p_fpt_min/psic[3]
                    p_tl[jj] = p_fpt_min/psic[3]

                else:
                    for ll in range(kk, 0, -1):
                        if (p_fptc[0][ll] >= t_pts[jj]) and (p_fptc[0][ll-1] < t_pts[jj]):
                            kk = int(ll)
                            break
                    p_tu[jj] = 1.0/psic[3] * (p_fptc[1][kk] + (t_pts[jj] - p_fptc[0][kk]) * ( p_fptc[1][kk+1] - p_fptc[1][kk] ) / ( p_fptc[0][kk+1] - p_fptc[0][kk] ))
                    p_tl[jj] = 1.0/psic[3] * (p_fptc[2][kk] + (t_pts[jj] - p_fptc[0][kk]) * ( p_fptc[2][kk+1] - p_fptc[2][kk] ) / ( p_fptc[0][kk+1] - p_fptc[0][kk] ))
                    
                tempu[jj] = p_tu[jj]
                templ[jj] = p_tl[jj]
                
            p_fpt_integ[0,ii] = p_fptc[0][ii]
            p_fpt_integ[1,ii] = tnd_trap(psic, tempu, tnd, dt_tnd, N_tnd)
            p_fpt_integ[2,ii] = tnd_trap(psic, templ, tnd, dt_tnd, N_tnd)
            p_fpt_integ[3,ii] = p_fptc[3][ii]
            
        return p_fpt_integ
    
    elif (tnd_dist == 'normal'):
        
        p_fpt_integ = np.zeros([4, N_cut])
        
        # generate array of non-decision times
        if (psic[2] - int_range*psic[3] <= 0.0):
            dt_tnd = (psic[2] + int_range*psic[3])/(N_tndc - 1.0)
            for ii in range(N_tndc):
                tnd[ii] = ii*dt_tnd      
        else:  
            dt_tnd = (2.0*int_range*psic[3])/(N_tndc - 1.0)
            for ii in range(N_tndc):
                tnd[ii] = psic[2] - int_range*psic[3] + ii*dt_tnd
                
        for ii in range(N_tndc):
            norm_prob[ii] = tnd_normal(psic, tnd[ii])
            
        for ii in range(N_cut):
            
            kk = N_cut-1
            
            for jj in range(N_tndc):
                
                t_pts[jj] = p_fptc[0][ii] - tnd[jj]
                
                if (t_pts[jj] <= 0.0):
                    p_tu[jj] = p_fpt_min/psic[3]
                    p_tl[jj] = p_fpt_min/psic[3]

                else:
                    for ll in range(kk, 0, -1):
                        if (p_fptc[0][ll] >= t_pts[jj]) and (p_fptc[0][ll-1] < t_pts[jj]):
                            kk = int(ll)
                            break
                    p_tu[jj] = norm_prob[jj] * (p_fptc[1][kk] + (t_pts[jj] - p_fptc[0][kk]) * ( p_fptc[1][kk+1] - p_fptc[1][kk] ) / ( p_fptc[0][kk+1] - p_fptc[0][kk] ))
                    p_tl[jj] = norm_prob[jj] * (p_fptc[2][kk] + (t_pts[jj] - p_fptc[0][kk]) * ( p_fptc[2][kk+1] - p_fptc[2][kk] ) / ( p_fptc[0][kk+1] - p_fptc[0][kk] ))
                    
                tempu[jj] = p_tu[jj]
                templ[jj] = p_tl[jj]
                
            p_fpt_integ[0,ii] = p_fptc[0][ii]
            p_fpt_integ[1,ii] = tnd_trap(psic, tempu, tnd, dt_tnd, N_tnd)
            p_fpt_integ[2,ii] = tnd_trap(psic, templ, tnd, dt_tnd, N_tnd)
            p_fpt_integ[3,ii] = p_fptc[3][ii]
            
        return p_fpt_integ
    
    else:
        return 0
        
    # -------------------- #

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
approx_avg_fpt

"""

@cython.cdivision(True)
@cython.boundscheck(False)
cdef double approx_avg_fpt(double phi[N_phi], double psi[N_psi]):

    cdef double ww = psi[0]
    cdef double vv = 0.0
    cdef double DD = diffusion(phi, 0.0, 0.0)
    cdef double cu = threshold(phi, 0.0)
    cdef double cl = -cu
    cdef double zz = cl + ww*(cu - cl)
    cdef double tt = 0.0
    cdef double uu = 0.0
    cdef double xx = 0.0
    cdef double fpt = 0.0
    cdef double dt = 0.0
    cdef double sqrtdt = 0.0
    cdef int ii = 0, jj = 0
    
    # seed rng
    srand(90210)
    
    # set time step
    dt = dt_afpt_scale*(cu - cl)*DD
    if (dt > dt_afpt_max):
        dt = dt_afpt_max
    dt = dt_afpt_max
    sqrtdt = sqrt(dt)
    
    # simulate model
    for ii in range(N_afpt):
        
        tt = 0.0
        xx = zz
        jj = 0
        
        while (jj < 1) and (tt <= t_max):

            # update time
            tt += dt

            # update drift rate, diffusion rate, and decision threhsolds
            vv = drift(phi, xx, tt)
            DD = diffusion(phi, xx, tt)
            cu = threshold(phi, tt)
            cl = -cu
                        
            # draw random number for diffusion process
            uu = -1 +  2 * ( rand() % 2 )
            
            # calculate accumulated evidence
            xx += dt*vv + sqrtdt*DD*uu

            # check if accumulated evidence has crossed a decision threshold
            if (xx >= cu) or (xx <= cl):
                fpt += tt
                jj = 1

    # if not threshold crossings, set fpt to max
    if (fpt == 0.0):
        fpt = t_max*N_afpt

    # output average first passage time
    return fpt/N_afpt

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
tnd_trap

"""

cdef double tnd_trap(double psic[N_psi], double lh[N_tnd_max], double tnd[N_tnd_max], double dt_tnd, int N_tnd):
    cdef int ii = 0
    cdef double out = 0.0
        
    for ii in range(1, N_tnd):
        out += dt_tnd * (lh[ii] + lh[ii-1])
        
    out = 0.5*out
    return out

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
simulate

"""

@cython.cdivision(True)
cpdef simulate(N_sims, phi, psi, w_dist, tnd_dist, mu_dist, dt, seed):
    
    # initialize variables
    cdef int ii = 0, jj = 0
    cdef int N_simsc = N_sims
    cdef double phic[N_phi]
    cdef double phic_temp[N_phi]
    cdef double psic[N_psi]

    for ii in range(N_phi):
        phic[ii] = phi[ii]
        phic_temp[ii] = phi[ii]
    for ii in range(N_psi):
        psic[ii] = psi[ii]

    cdef double dtc = dt
    cdef double sqrtdt = sqrt(dtc)
    cdef int seedc = seed
    cdef double tnd = 0.0
    cdef double ww = 0.0
    cdef double vv = 0.0
    cdef double DD = 0.0
    cdef double cu = 0.0
    cdef double cl = 0.0
    cdef double zz = 0.0
    cdef double tt = 0.0
    cdef double u1 = 0.0
    cdef double u2 = 0.0
    cdef double uu = 0.0
    cdef double xx = 0.0
    cdef double gg = contamination_strength(phic)
    cdef double guess = 0.0
    cdef double side = 0.0
    cdef double t_guess = 0.0
    
    # seed rng
    srand(seedc)
    np.random.seed(seed)
    
    # initilize first passage time array
    fpt = np.zeros(N_sims)
    
    # simulate model N_sims times
    for ii in range(N_simsc):
        
        # set the start location
        if (w_dist == 'constant'):
            ww = psic[0]
            cu = threshold(phic, 0.0)
            cl = -cu
            zz = cl + ww*(cu - cl)
        elif (w_dist == 'uniform'):
            ww = np.random.uniform(psic[0] - psic[1]/2.0, psic[0] + psic[1]/2.0)
            if (ww < w_min) or (ww > w_max):
                ww = np.random.uniform(psic[0] - psic[1]/2.0, psic[0] + psic[1]/2.0)                
            cu = threshold(phic, 0.0)
            cl = -cu
            zz = cl + ww*(cu - cl)
        elif (w_dist == 'normal'):
            ww = np.random.normal(psic[0], psic[1])
            if (ww < w_min) or (ww > w_max):
                ww = np.random.normal(psic[0], psic[1])
            cu = threshold(phic, 0.0)
            cl = -cu
            zz = cl + ww*(cu - cl)
            
        # set the non-decision time
        if (tnd_dist == 'constant'):
            tnd = psic[2]
        elif (tnd_dist == 'uniform'):
            tnd = np.random.uniform(psic[2] - psic[3]/2.0, psic[2] + psic[3]/2.0)
            while (tnd < 0.0):
                tnd = np.random.uniform(psic[2] - psic[3]/2.0, psic[2] + psic[3]/2.0)
        elif (tnd_dist == 'normal'):
            tnd = np.random.normal(psic[2], psic[3])
            while (tnd < 0.0):
                tnd = np.random.normal(psic[2], psic[3])    
        if (tnd < 0.0):
            tnd = 0.0
            
        # set the drift rate if not constant
        if (mu_dist == 'uniform'):
            phic_temp[0] = np.random.uniform(phic[0] - 0.5*psic[4], phic[0] + 0.5*psic[4])
            vv = drift(phic_temp, xx, tt)
        elif (mu_dist == 'normal'):
            phic_temp[0] = np.random.normal(phic[0], psic[4])
            vv = drift(phic_temp, xx, tt)
        
        # initialize values
        tt = 0.0
        xx = zz
        jj = 0
        
        guess = rand() / (RAND_MAX*1.0)
        side = 0.0

        if (guess <= gg):

            t_guess = np.random.uniform(phi[4], phi[5])
            side = rand() / (RAND_MAX*1.0)
            if (side <= 0.5):
                fpt[ii] = t_guess
            else:
                fpt[ii] = -t_guess

        else:
        
            # simulate the model
            while (jj < 1) and (tt <= t_max):

                # update time
                tt += dtc

                # update drift rate, diffusion rate, and decision threhsolds
                if (mu_dist == 'constant'):
                    vv = drift(phic, xx, tt)
                DD = diffusion(phic, xx, tt)
                cu = threshold(phic, tt)
                cl = -cu
                            
                # draw random number for diffusion process
                u1 = rand() / (RAND_MAX*1.0)
                u2 = rand() / (RAND_MAX*1.0)
                uu = sqrt(-2.0*log(u1))*cos(2.0*pi*u2)
                
                # calculate accumulated evidence
                xx += dtc*vv + sqrtdt*DD*uu

                # check if accumulated evidence has crossed a decision threshold
                if (xx >= cu):
                    fpt[ii] = tnd + tt
                    jj = 1
                elif (xx <= cl):
                    fpt[ii] = -(tnd + tt)
                    jj = 1

    # output average first passage time
    return fpt

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
likelihood plot

"""

@cython.cdivision(True)
@cython.boundscheck(False)
cpdef likelihood_plot(phi, psi, w_dist, tnd_dist, N_tnd, mu_dist, N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rt_max):
    
    cdef double phic[N_phi]
    for ii in range(N_phi):
        phic[ii] = phi[ii]
    cdef double gg = contamination_strength(phic)
    if (gg > gg_max):
        gg = gg_max
    
    if (mu_dist == 'constant'):
        lh = likelihood(phi, psi, w_dist, tnd_dist, N_tnd, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rt_max)
    else:
        lh = integrate_drift(phi, psi, w_dist, tnd_dist, N_tnd, mu_dist, N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rt_max)

    lh_out = np.zeros([3,len(lh[0])])
    lh_out[0] = lh[0]
    lh_out[1] = (1.0 - gg)*lh[1] + lh[3]
    lh_out[2] = (1.0 - gg)*lh[2] + lh[3]
    
    return lh_out

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
loglikelihood

"""

@cython.cdivision(True)
@cython.boundscheck(False)
cpdef loglikelihood(phi, psi, w_dist, tnd_dist, N_tnd, mu_dist, N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rtu, rtl, rt_max):

    cdef double phic[N_phi]
    for ii in range(N_phi):
        phic[ii] = phi[ii]
    cdef double gg = contamination_strength(phic)
    
    if (mu_dist == 'constant'):
        lh = likelihood(phi, psi, w_dist, tnd_dist, N_tnd, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rt_max)
    else:
        lh = integrate_drift(phi, psi, w_dist, tnd_dist, N_tnd, mu_dist, N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rt_max)
    
    if (gg > 0.0):
        if (gg >= gg_max):
            gg = gg_max
        lhu = interpolate.interp1d(lh[0], (1.0 - gg)*lh[1] + lh[3], bounds_error = False, fill_value = p_fpt_min)
        lhl = interpolate.interp1d(lh[0], (1.0 - gg)*lh[2] + lh[3], bounds_error = False, fill_value = p_fpt_min)
    else:
        lhu = interpolate.interp1d(lh[0], lh[1], bounds_error = False, fill_value = p_fpt_min)
        lhl = interpolate.interp1d(lh[0], lh[2], bounds_error = False, fill_value = p_fpt_min)

    llh = np.sum(np.log(lhu(rtu))) + np.sum(np.log(lhl(rtl)))
            
    return llh
    
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

@cython.cdivision(True)
@cython.boundscheck(False)
cpdef integrate_drift(phi, psi, w_dist, tnd_dist, N_tnd, mu_dist, N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rt_max):
    
    cdef int ii = 0
    cdef int jj = 0
    cdef double phic[N_phi]
    cdef double psic[N_psi]

    for ii in range(N_phi):
        phic[ii] = phi[ii]
    for ii in range(N_psi):
        psic[ii] = psi[ii]
            
    cdef int N_muc = N_mu
    cdef double dmu = 0.0
    cdef double mu[N_mu_max]
    cdef int dt_int = 0
    cdef int N_max = 0
    cdef double tempu[N_mu_max]
    cdef double templ[N_mu_max]
    cdef double munorm[N_mu_max]
    lh = [0]*N_mu
    lhi = [0]*N_mu
    lh_interp = [0]*N_mu
    
    for ii in range(N_mu_max):
        mu[ii] = 0.0
    
    if (mu_dist == 'uniform'):
        dmu = psic[4]/(N_muc - 1.0)
        for ii in range(N_muc):
            mu[ii] = phic[0] - 0.5*psic[4]+ ii*dmu
    elif (mu_dist == 'normal'):
        dmu = 2.0*int_range*psic[4]/(N_muc - 1)
        for ii in range(N_muc):
            mu[ii] = phic[0] - int_range*psic[4] + ii*dmu
            
    dt_int = int(round(N_muc/2.0))
    phi_temp = phi.copy()
    phi_temp[0] = mu[dt_int]
    lh[dt_int] = likelihood(phi_temp, psi, w_dist, tnd_dist, N_tnd, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rt_max)
            
    for ii in range(N_muc):
        if (ii != dt_int):
            phi_temp = phi.copy()
            phi_temp[0] = mu[ii]
            lh[ii] = likelihood(phi_temp, psi, w_dist, tnd_dist, N_tnd, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rt_max)
            lhi[ii] = [interpolate.interp1d(lh[ii][0], lh[ii][1], bounds_error = False, fill_value = p_fpt_min), interpolate.interp1d(lh[ii][0], lh[ii][2], bounds_error = False, fill_value = p_fpt_min)]
            lh_interp[ii] = [lhi[ii][0](lh[dt_int][0]), lhi[ii][1](lh[dt_int][0])]
            
    N_max = len(lh[dt_int][0])
    lhu_integ = np.zeros(N_max)
    lhl_integ = np.zeros(N_max)
    lh_integ = np.zeros([4, N_max])
    
    if (mu_dist == 'uniform'):
    
        for ii in range(N_max):
            for jj in range(N_muc):
                if (jj == dt_int):
                    tempu[jj] = lh[jj][1][ii]/psic[4]
                    templ[jj] = lh[jj][2][ii]/psic[4]
                else:
                    tempu[jj] = lh_interp[jj][0][ii]/psic[4]
                    templ[jj] = lh_interp[jj][1][ii]/psic[4]
                
            lh_integ[0,ii] = lh[dt_int][0][ii]
            lh_integ[1,ii] = drift_trap(N_muc, dmu, tempu)
            lh_integ[2,ii] = drift_trap(N_muc, dmu, templ)
            lh_integ[3,ii] = lh[dt_int][3][ii]
            
    elif (mu_dist == 'normal'):
        
        for ii in range(N_muc):
            munorm[ii] = mu_normal(mu[ii], phic[0], psic[4])
            
        for ii in range(N_max):
            for jj in range(N_muc):
                if (jj == dt_int):
                    tempu[jj] = lh[jj][1][ii]*munorm[jj]
                    templ[jj] = lh[jj][2][ii]*munorm[jj]
                else:
                    tempu[jj] = lh_interp[jj][0][ii]*munorm[jj]
                    templ[jj] = lh_interp[jj][1][ii]*munorm[jj]
                        
            lh_integ[0,ii] = lh[dt_int][0][ii]
            lh_integ[1,ii] = drift_trap(N_muc, dmu, tempu)
            lh_integ[2,ii] = drift_trap(N_muc, dmu, templ)
            lh_integ[3,ii] = lh[dt_int][3][ii]        
    
    return lh_integ
    
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

cdef double drift_trap(int N_mu, double dmu, double lh[N_mu_max]):
    cdef int ii = 0
    cdef double out = 0.0
        
    for ii in range(1, N_mu):
        out += dmu * (lh[ii] + lh[ii-1])
        
    out = 0.5*out
    return out
    