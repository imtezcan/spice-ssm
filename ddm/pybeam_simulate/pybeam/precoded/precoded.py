"""
File: prebuilt.py
Description: Functions for the pybeam.precoded sub-module.
Creator: Matthew Murrow
Email: matthew.a.murrow@vanderbilt.edu

"""

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
Import Python modules and other functions required by pybeam.

"""

# import python modules
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as at
import arviz as az
import copy
import warnings
from scipy import interpolate

# import pybeam functions and class objects
from . import loglike_class
from .model_classes import *
from . import functions_constant
from . import functions_leakage
from . import functions_changing_thresholds
from . import functions_UGM
from . import functions_UGM_flip
from . import functions_DMC
    
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
        
def simulate(model, N_sims, phi, dt = 0.0001, seed = False):
    
    """
    Check if function input(s) are of the correct form.

    """

    # check if model is input properly
    model.test_inputs()
    
    # check if N_sims is input properly
    if (isinstance(N_sims, int) == False):
        raise RuntimeError("N_sims must be a positive integer.")
    if (N_sims <= 0):
        raise RuntimeError("N_sims must be more than zero.")

    # check if phi is input properly
    if (isinstance(phi, dict) == False):
        raise RuntimeError("phi must be a dictionary.")
        
    # check if phi contains the correct keys
    phi_keys = model.parameters()
    if (set(phi) != set(phi_keys)):
        raise RuntimeError("phi has incorrect keys.")

    # check if seed is input properly
    if (seed != False):
        if (isinstance(seed, int) == False):
            raise RuntimeError("seed must be of type int.")
    elif (seed < 0):
        raise RuntimeError("seed must be a positive integer.")

    # check if dt is input properly
    if (isinstance(dt, float) == False):
        raise RuntimeError("dt must be of type float.")
    elif (dt <= 0.0):
        raise RuntimeError("dt must be more than zero.")
        
    """
    Parse phi input.

    """

    # create array of parameters for phi based on inputs
    phi_bulk = model.phi_bulk()
    for ii in range(len(phi_bulk)):
        if ( isinstance(phi_bulk[ii], str) ):
            phi_bulk[ii] = phi[phi_bulk[ii]]
           
    # create array of parameters for psi based on inputs
    psi_bulk = model.psi_bulk()
    for ii in range(len(psi_bulk)):
        if ( isinstance(psi_bulk[ii], str) ):
            psi_bulk[ii] = phi[psi_bulk[ii]]
            
    """
    Simulate model, output dictonary of reaction times.

    """
        
    # generate random seed if no seed set
    if (seed == False):
        seed = np.random.randint(0,100000)
        
    # call simulate function
    if (type(model) == simpleDDM):
        rt = np.asarray( functions_constant.simulate(N_sims, phi_bulk, psi_bulk, model.start, model.non_decision, model.drift, dt, seed) )
    elif (type(model) == leakage):
        rt = np.asarray( functions_leakage.simulate(N_sims, phi_bulk, psi_bulk, model.start, model.non_decision, 'constant', dt, seed) )
    elif (type(model) == fullDDM):
        rt = np.asarray( functions_constant.simulate(N_sims, phi_bulk, psi_bulk, model.start, model.non_decision, model.drift, dt, seed) )
    elif (type(model) == changing_thresholds):
        rt = np.asarray( functions_changing_thresholds.simulate(N_sims, phi_bulk, psi_bulk, model.start, model.non_decision, model.drift, dt, seed) )
    elif (type(model) == UGM):
        rt = np.asarray( functions_UGM.simulate(N_sims, phi_bulk, psi_bulk, model.start, model.non_decision, 'constant', dt, seed) )
    elif (type(model) == UGM_flip):
        rt = np.asarray( functions_UGM_flip.simulate(N_sims, phi_bulk, psi_bulk, model.start, model.non_decision, 'constant', dt, seed) )
    elif (type(model) == DMC):
        rt = np.asarray( functions_DMC.simulate(N_sims, phi_bulk, psi_bulk, model.start, model.non_decision, 'constant', dt, seed) )
        
    # load results into dictionary to be output to user
    rt_upper = rt[rt >= 0.0]
    rt_lower = -rt[rt < 0.0]
    rt_dict = { 'rt_upper' : rt_upper , 'rt_lower' : rt_lower }
            
    return rt_dict

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

def likelihood(model, phi, rt_max, N_tnd = 100, N_mu = 10, x_res = 'default', t_res = 'default', dt_interp_scale = 1.0, dt_interp_overide = 0.0):
    
    """
    Check if function input(s) are of the correct form.

    """

    # check if model is input properly
    model.test_inputs()
    
    # check if phi is input properly
    if (isinstance(phi, dict) == False):
        raise RuntimeError("phi must be a dictionary.")
        
    # check if phi contains the correct keys
    phi_keys = model.parameters()
    if (set(phi) != set(phi_keys)):
        raise RuntimeError("phi has incorrect keys.")
        
    # check if N_tnd is input properly
    if (isinstance(N_tnd, int) == False):
        raise RuntimeError("N_tnd must be an integer greater than zero.")
    elif (N_tnd <= 0):
        raise RuntimeError("N_tnd must be an integer greater than zero.")
    elif (N_tnd < 10):
        warnings.warn("It is recommended that N_tnd > 10. Strongly suggest defualt setting of N_tnd = 100.")
    elif (N_tnd < 100):
        warnings.warn("Use caution when changing N_tnd. Strongly suggest defualt setting of N_tnd = 100.")
        
    # check if N_mu is input properly
    if (isinstance(N_mu, int) == False):
        raise RuntimeError("N_mu must be an integer greater than zero.")
    elif (N_mu <= 0):
        raise RuntimeError("N_mu must be an integer greater than zero.")
    elif (N_mu < 10):
        warnings.warn("It is recommended that N_mu >= 10. Strongly suggest defualt setting of N_mu = 10.")
        
    # check if spatial resolution is input properly   
    xres_list = ['default', 'high', 'very_high', 'max']
    if (x_res not in xres_list) and (isinstance(x_res, int) == False):
        raise RuntimeError("x_res must equal 'default', 'high', 'very_high', or 'max', or be an integer greater than 25.")
    if (isinstance(x_res, int) == True):
        if (x_res < 25):
            raise RuntimeError("x_res must equal 'default', 'high', 'very_high', or 'max', or be an integer greater than 25.")
        if (x_res < 101):
            warnings.warn("It is recommended that x_res >= 101. Strongly suggest leaving at x_res = 'default'.")
        if (x_res > 501):
            raise RuntimeError("x_res must be less than or equal to 501.")
               
    # check if temporal resolution is input properly   
    tres_list = ['default', 'high', 'very_high', 'max']
    if (t_res not in tres_list):
        raise RuntimeError("t_res must equal 'default', 'high', 'very_high', or 'max'.")
        
    # check if dt_interp_scale is input properly
    if (isinstance(dt_interp_scale, float) == False) or (dt_interp_scale <= 0.0):
        raise RuntimeError("dt_interp_scale must be a float above zero. Strongly suggest using defualt of 1.0.")
    elif (dt_interp_scale != 1.0):
        warnings.warn("Use caution when modifying dt_interp_scale. Strongly suggest using default of 1.0.")
        
    # check if dt_interp_overide is input properly
    if (isinstance(dt_interp_overide, float) == False) or (dt_interp_overide < 0.0):
        raise RuntimeError("dt_interp_overide must be a float above or equal to zero. Strongly suggest using defualt of 0.0.")
    elif (dt_interp_overide != 0.0):
        warnings.warn("Use caution when modifying dt_interp_overide. Strongly suggest using default of 0.0.")
        
    # check if rt_max is input properly
    if (isinstance(rt_max, float) == False) or (rt_max <= 0.0):
        raise RuntimeError("rt_max must be a float more than or equal to zero.")
        
    """
    Set solver resolution.

    """
    
    # set solver spatial resolution
    if (x_res == 'fast'):
        N_deps = int(101)
    elif (x_res == 'default'):
        N_deps = int(151)
    elif (x_res == 'high'):
        N_deps = int(251)
    elif (x_res == 'max'):
        N_deps = int(501)
    elif (isinstance(x_res, int) == True):
        N_deps = x_res
        
    # set solver time resolution
    if (t_res == 'default'):
        dt_scale = 0.025
    elif (t_res == 'high'):
        dt_scale = 0.015
    elif (t_res == 'very_high'):
        dt_scale = 0.01
    elif (t_res == 'max'):
        dt_scale = 0.005
        
    """
    Parse phi input.

    """

    # create array of parameters for phi based on inputs
    phi_bulk = model.phi_bulk()
    for ii in range(len(phi_bulk)):
        if ( isinstance(phi_bulk[ii], str) ):
            phi_bulk[ii] = phi[phi_bulk[ii]]
           
    # create array of parameters for psi based on inputs
    psi_bulk = model.psi_bulk()
    for ii in range(len(psi_bulk)):
        if ( isinstance(psi_bulk[ii], str) ):
            psi_bulk[ii] = phi[psi_bulk[ii]]
        
    """
    Generate likelihood function.

    """
    
    # call likelihood function
    if (type(model) == simpleDDM):
        lh = functions_constant.likelihood_plot(phi_bulk, psi_bulk, model.start, model.non_decision, N_tnd, model.drift, N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rt_max)
    elif (type(model) == leakage):
        lh = functions_leakage.likelihood_plot(phi_bulk, psi_bulk, model.start, model.non_decision, N_tnd, 'constant', N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rt_max)
    elif (type(model) == fullDDM):
        lh = functions_constant.likelihood_plot(phi_bulk, psi_bulk, model.start, model.non_decision, N_tnd, model.drift, N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rt_max)
    elif (type(model) == changing_thresholds):
        lh = functions_changing_thresholds.likelihood_plot(phi_bulk, psi_bulk, model.start, model.non_decision, N_tnd, model.drift, N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rt_max)
    elif (type(model) == UGM):
        lh = functions_UGM.likelihood_plot(phi_bulk, psi_bulk, model.start, model.non_decision, N_tnd, 'constant', N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rt_max)
    elif (type(model) == UGM_flip):
        lh = functions_UGM_flip.likelihood_plot(phi_bulk, psi_bulk, model.start, model.non_decision, N_tnd, 'constant', N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rt_max)
    elif (type(model) == DMC):
        lh = functions_DMC.likelihood_plot(phi_bulk, psi_bulk, model.start, model.non_decision, N_tnd, 'constant', N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rt_max)
        
    # load likelihood into dictionary to be output by function
    lh_out = {'time' : lh[0], 'lh_upper' : lh[1], 'lh_lower' : lh[2]}
    return lh_out

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

def loglikelihood(model, phi, rt, pointwise = False, N_tnd = 100, N_mu = 10, x_res = 'default', t_res = 'default', dt_interp_scale = 1.0, dt_interp_overide = 0.0):
    
    """
    Check if function input(s) are of the correct form.

    """

    # check if model is input properly
    model.test_inputs()
    
    # check if phi is input properly
    if (isinstance(phi, dict) == False):
        raise RuntimeError("phi must be a dictionary.")
        
    # check if phi contains the correct keys
    phi_keys = model.parameters()
    if (set(phi) != set(phi_keys)):
        raise RuntimeError("phi has incorrect keys.")
        
    # check if rt is input properly
    if (isinstance(rt, dict) == False):
        raise RuntimeError("rt must be a dictionary containing two keys: 'rt_upper' and 'rt_lower'.")
        
    # check if pointwise is input properly
    if (pointwise != False) and (pointwise != True):
        raise RuntimeError("pointwise must equal True or False.")
        
    # check if N_tnd is input properly
    if (isinstance(N_tnd, int) == False):
        raise RuntimeError("N_tnd must be an integer greater than zero.")
    elif (N_tnd <= 0):
        raise RuntimeError("N_tnd must be an integer greater than zero.")
    elif (N_tnd < 10):
        warnings.warn("It is recommended that N_tnd > 10. Strongly suggest defualt setting of N_tnd = 100.")
    elif (N_tnd < 100):
        warnings.warn("Use caution when changing N_tnd. Strongly suggest defualt setting of N_tnd = 100.")
        
    # check if N_mu is input properly
    if (isinstance(N_mu, int) == False):
        raise RuntimeError("N_mu must be an integer greater than zero.")
    elif (N_mu <= 0):
        raise RuntimeError("N_mu must be an integer greater than zero.")
    elif (N_mu < 10):
        warnings.warn("It is recommended that N_mu >= 10. Strongly suggest defualt setting of N_mu = 10.")
        
    # check if spatial resolution is input properly   
    xres_list = ['fast', 'default', 'high', 'max']
    if (x_res not in xres_list) and (isinstance(x_res, int) == False):
        raise RuntimeError("x_res must equal 'fast', 'default', 'high', or 'max', or be an integer greater than 25.")
    if (isinstance(x_res, int) == True):
        if (x_res < 25):
            raise RuntimeError("x_res must equal 'fast', 'default', 'high', or 'max', or be an integer greater than 25.")
        if (x_res < 101):
            warnings.warn("It is recommended that x_res >= 101. Strongly suggest leaving at x_res = 'default'.")
        if (x_res > 501):
            raise RuntimeError("x_res must be less than or equal to 501.")
               
    # check if temporal resolution is input properly   
    tres_list = ['default', 'high', 'very_high', 'max']
    if (t_res not in tres_list):
        raise RuntimeError("t_res must equal 'default', 'high', 'very_high', or 'max'.")
        
    # check if dt_interp_scale is input properly
    if (isinstance(dt_interp_scale, float) == False) or (dt_interp_scale <= 0.0):
        raise RuntimeError("dt_interp_scale must be a float above zero. Strongly suggest using defualt of 1.0.")
    elif (dt_interp_scale != 1.0):
        warnings.warn("Use caution when modifying dt_interp_scale. Strongly suggest using default of 1.0.")
        
    # check if dt_interp_overide is input properly
    if (isinstance(dt_interp_overide, float) == False) or (dt_interp_overide < 0.0):
        raise RuntimeError("dt_interp_overide must be a float above or equal to zero. Strongly suggest using defualt of 0.0.")
    elif (dt_interp_overide != 0.0):
        warnings.warn("Use caution when modifying dt_interp_overide. Strongly suggest using default of 0.0.")
        
    """
    Set solver resolution.

    """
    
    # set solver spatial resolution
    if (x_res == 'fast'):
        N_deps = int(101)
    elif (x_res == 'default'):
        N_deps = int(151)
    elif (x_res == 'high'):
        N_deps = int(251)
    elif (x_res == 'max'):
        N_deps = int(501)
    elif (isinstance(x_res, int) == True):
        N_deps = x_res
        
    # set solver time resolution
    if (t_res == 'default'):
        dt_scale = 0.025
    elif (t_res == 'high'):
        dt_scale = 0.015
    elif (t_res == 'very_high'):
        dt_scale = 0.01
    elif (t_res == 'max'):
        dt_scale = 0.005
        
    """
    Parse phi input.

    """

    # create array of parameters for phi based on inputs
    phi_bulk = model.phi_bulk()
    for ii in range(len(phi_bulk)):
        if ( isinstance(phi_bulk[ii], str) ):
            phi_bulk[ii] = phi[phi_bulk[ii]]
           
    # create array of parameters for psi based on inputs
    psi_bulk = model.psi_bulk()
    for ii in range(len(psi_bulk)):
        if ( isinstance(psi_bulk[ii], str) ):
            psi_bulk[ii] = phi[psi_bulk[ii]]
        
    """
    Call the likelihood function.

    """
    
    rtu = rt['rt_upper']
    rtl = rt['rt_lower']
    rt_max = np.max( np.concatenate( (rtu, rtl) ) )
    
    if (pointwise == False):
        
        if (type(model) == simpleDDM):
            llh = functions_constant.loglikelihood(phi_bulk, psi_bulk, model.start, model.non_decision, N_tnd, model.drift, N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rtu, rtl, rt_max)
        if (type(model) == fullDDM):
            llh = functions_constant.loglikelihood(phi_bulk, psi_bulk, model.start, model.non_decision, N_tnd, model.drift, N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rtu, rtl, rt_max)
        if (type(model) == leakage):
            llh = functions_leakage.loglikelihood(phi_bulk, psi_bulk, model.start, model.non_decision, N_tnd, 'constant', N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rtu, rtl, rt_max)
        if (type(model) == changing_thresholds):
            llh = functions_changing_thresholds.loglikelihood(phi_bulk, psi_bulk, model.start, model.non_decision, N_tnd, model.drift, N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rtu, rtl, rt_max)
        if (type(model) == UGM):
            llh = functions_UGM.loglikelihood(phi_bulk, psi_bulk, model.start, model.non_decision, N_tnd, 'constant', N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rtu, rtl, rt_max)
        if (type(model) == UGM_flip):
            llh = functions_UGM_flip.loglikelihood(phi_bulk, psi_bulk, model.start, model.non_decision, N_tnd, 'constant', N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rtu, rtl, rt_max)
        if (type(model) == DMC):
            llh = functions_DMC.loglikelihood(phi_bulk, psi_bulk, model.start, model.non_decision, N_tnd, 'constant', N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rtu, rtl, rt_max)
            
    elif (pointwise == True):

        llh = {}
        lh = likelihood(model = model, phi = phi, N_tnd = N_tnd, N_mu = N_mu, x_res = x_res, t_res = t_res, dt_interp_scale = dt_interp_scale, dt_interp_overide = dt_interp_overide, rt_max = rt_max)
        lhu = interpolate.interp1d(lh['time'], lh['lh_upper'], bounds_error = False, fill_value = np.min(lh['lh_upper']))
        lhl = interpolate.interp1d(lh['time'], lh['lh_lower'], bounds_error = False, fill_value = np.min(lh['lh_lower']))
        llh['llh_upper'] = np.log(lhu(rt['rt_upper']))
        llh['llh_lower'] = np.log(lhl(rt['rt_lower']))

    return llh

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

def plot_rt(model, phi, rt_max, rt = False, bins = 25, figsize = (6.4, 4), dpi = 100, N_tnd = 100, N_mu = 10, x_res = 'default', t_res = 'default', dt_interp_scale = 1.0, dt_interp_overide = 0.0):
    
    """
    Check if function input(s) are of the correct form.

    """

    # check if model is input properly
    model.test_inputs()
    
    # check if rt is input properly
    if (isinstance(rt, dict) == False) and (rt != False):
        raise RuntimeError("rt must be False or a dictionary containing two keys: 'rt_upper' and 'rt_lower'.")
        
    """
    Plot likelihood function and (optionally) rt data.

    """

    # call likelihood function
    lh = likelihood(model = model, phi = phi, N_tnd = N_tnd, N_mu = N_mu, x_res = x_res, t_res = t_res, dt_interp_scale = dt_interp_scale, dt_interp_overide = dt_interp_overide, rt_max = rt_max)
    
    # plot likelihood and rt data
    fig, axs = plt.subplots(1, 1, figsize = figsize, dpi = dpi)
    
    if (rt != False):
        rt_all = np.concatenate((rt['rt_upper'], -rt['rt_lower']))
        axs.hist(rt_all, bins = bins, density=True, ec='k', fc='lightgrey')
    
    axs.plot(lh['time'], lh['lh_upper'], c = 'r', lw = 2);
    axs.plot(-lh['time'], lh['lh_lower'], c = 'r', lw = 2);
    
    return axs
    
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

def inference(model, priors, conditions, samples, chains, cores, file_name, dt_interp_scale = 1.0, dt_interp_overide = 0.0, N_tnd = 100, N_mu = 10, sampler = 'DEMetropolisZ', x_res = 'default', t_res = 'default', tune = 0, save_loglikelihood = False, progressbar = True): 
    
    """
    Check if function input(s) are of the correct form.

    """

    # check if model is input properly
    model.test_inputs()
    
    # check if priors input properly
    if (isinstance(priors, dict) == False):
        raise RuntimeError("priors must be a dictionary. See documentation for proper use.")
        
    # check if conditions input properly
    if (isinstance(conditions, dict) == False):
        raise RuntimeError("conditions must be a dictionary. See documentation for proper use.")
        
    # check if conditions as the correct key definitions
    N_conditions = len(conditions)
    for ii in range(N_conditions):
        if (ii not in conditions):
            raise RuntimeError("Keys for condition must be positive integers going from 0, 1, ... , N_conditions-1.")
            
    # check conditions sub-dictionaries to see if they have correct keys
    phi_keys = model.parameters()
    phi_keys.append('rt')
    for ii in range(N_conditions):
        if ( set(conditions[ii]) != set(phi_keys) ):
            raise RuntimeError("condition sub-dictionary " + str(ii) + " has incorrect keys.")
            
    # check condition sub-dictionaries rt input
    rt_keys = ['rt_upper', 'rt_lower']
    for ii in range(N_conditions):
        if (isinstance(conditions[ii]['rt'], dict) == False):
            raise RuntimeError("rt for condition " + str(ii) + " must be a dictionary with two entries: 'rt_upper' and 'rt_lower'.")
        if (set(list(conditions[ii]['rt'].keys())) != set(rt_keys)):
            raise RuntimeError("rt for condition " + str(ii) + " must be a dictionary with two entries: 'rt_upper' and 'rt_lower'.")
            
    # check if samples input properly
    if (isinstance(samples, int) == False):
        raise RuntimeError("samples must be of type int.")
        
    # check if chains input properly
    if (isinstance(chains, int) == False):
        raise RuntimeError("chains must be of type int.")
        
    # check if cores input properly
    if (isinstance(cores, int) == False):
        raise RuntimeError("cores must be of type int.")
        
    # check if file_name input properly
    if (isinstance(file_name, str) == False):
        raise RuntimeError("file_name must be a string.")
    
    # check if sampler input properly   
    sampler_list = ['DEMetropolisZ', 'DEMetropolis']
    test = sampler not in sampler_list
    if (test == True):
        raise RuntimeError("sampler must equal 'DEMetropolisZ' or 'DEMetropolis'.")
        
    # check if N_tnd is input properly
    if (isinstance(N_tnd, int) == False):
        raise RuntimeError("N_tnd must be an integer greater than zero.")
    elif (N_tnd <= 0):
        raise RuntimeError("N_tnd must be an integer greater than zero.")
    elif (N_tnd < 10):
        warnings.warn("It is recommended that N_tnd > 10. Strongly suggest defualt setting of N_tnd = 100.")
    elif (N_tnd < 100):
        warnings.warn("Use caution when changing N_tnd. Strongly suggest defualt setting of N_tnd = 100.")
        
    # check if N_mu is input properly
    if (isinstance(N_mu, int) == False):
        raise RuntimeError("N_mu must be an integer greater than zero.")
    elif (N_mu <= 0):
        raise RuntimeError("N_mu must be an integer greater than zero.")
    elif (N_mu < 10):
        warnings.warn("It is recommended that N_mu >= 10. Strongly suggest defualt setting of N_mu = 10.")
        
    # check if spatial resolution is input properly   
    xres_list = ['fast', 'default', 'high', 'max']
    if (x_res not in xres_list) and (isinstance(x_res, int) == False):
        raise RuntimeError("x_res must equal 'fast', 'default', 'high', or 'max', or be an integer greater than 25.")
    if (isinstance(x_res, int) == True):
        if (x_res < 25):
            raise RuntimeError("x_res must equal 'fast', 'default', 'high', or 'max', or be an integer greater than 25.")
        if (x_res < 101):
            warnings.warn("It is recommended that x_res >= 101. Strongly suggest leaving at x_res = 'default'.")
        if (x_res > 501):
            raise RuntimeError("x_res must be less than or equal to 501.")
               
    # check if temporal resolution is input properly   
    tres_list = ['default', 'high', 'very_high', 'max']
    if (t_res not in tres_list):
        raise RuntimeError("t_res must equal 'default', 'high', 'very_high', or 'max'.")
        
    # check if dt_interp_scale is input properly
    if (isinstance(dt_interp_scale, float) == False) or (dt_interp_scale <= 0.0):
        raise RuntimeError("dt_interp_scale must be a float above zero. Strongly suggest using defualt of 1.0.")
    elif (dt_interp_scale != 1.0):
        warnings.warn("Use caution when modifying dt_interp_scale. Strongly suggest using default of 1.0.")
        
    # check if dt_interp_overide is input properly
    if (isinstance(dt_interp_overide, float) == False) or (dt_interp_overide < 0.0):
        raise RuntimeError("dt_interp_overide must be a float above or equal to zero. Strongly suggest using defualt of 0.0.")
    elif (dt_interp_overide != 0.0):
        warnings.warn("Use caution when modifying dt_interp_overide. Strongly suggest using default of 0.0.")
        
    """
    Set solver resolution.

    """
    
    # set solver spatial resolution
    if (x_res == 'fast'):
        N_deps = int(101)
    elif (x_res == 'default'):
        N_deps = int(151)
    elif (x_res == 'high'):
        N_deps = int(251)
    elif (x_res == 'max'):
        N_deps = int(501)
    elif (isinstance(x_res, int) == True):
        N_deps = x_res
        
    # set solver time resolution
    if (t_res == 'default'):
        dt_scale = 0.025
    elif (t_res == 'high'):
        dt_scale = 0.015
    elif (t_res == 'very_high'):
        dt_scale = 0.01
    elif (t_res == 'max'):
        dt_scale = 0.005
        
    """
    Run parameter inference.
    
    """
    
    # copy prior dictionary
    priors = copy.deepcopy(priors)
    priors_keys = list( priors.keys() )
    N_priors = len(priors_keys)
    
    # determine arrays for phi and psi
    phi_bulk = model.phi_bulk()
    psi_bulk = model.psi_bulk()
    N_phi = len(phi_bulk)
    N_psi = len(psi_bulk)
    phi = [[0 for ii in range(N_phi)] for jj in range(N_conditions)]
    psi = [[0 for ii in range(N_psi)] for jj in range(N_conditions)]
    
    # associate phi parameters with each condition
    for ii in range(N_conditions):
        for jj in range(N_phi):
            if ( isinstance(phi_bulk[jj], str) ):
                for kk in range(N_priors):
                    if (conditions[ii][ phi_bulk[jj] ] in priors):
                        phi[ii][jj] = 'priors[' + str( priors_keys.index(conditions[ii][phi_bulk[jj]]) ) + ']'
            else:
                phi[ii][jj] = phi_bulk[jj]   
     
    # associate psi parameters with each condition
    for ii in range(N_conditions):
        for jj in range(N_psi):
            if ( isinstance(psi_bulk[jj], str) ):
                for kk in range(N_priors):
                    if (conditions[ii][ psi_bulk[jj] ] in priors):
                        psi[ii][jj] = 'priors[' + str( priors_keys.index(conditions[ii][psi_bulk[jj]]) ) + ']'
            else:
                psi[ii][jj] = psi_bulk[jj]
                
    # set ops for each condition
    rtu = [0]*N_conditions
    rtl = [0]*N_conditions
    rt_max = [0]*N_conditions
    logl_op = [0]*N_conditions
    
    for ii in range(N_conditions):
        rtu[ii] = conditions[ii]['rt']['rt_upper']
        rtl[ii] = conditions[ii]['rt']['rt_lower']
        rt_max[ii] = np.max(np.concatenate((rtu[ii], rtl[ii])))
    
        if ( type(model) == simpleDDM ):
            logl_op[ii] = loglike_class.LogLike(functions_constant.loglikelihood,
                                                model.start,
                                                model.non_decision,
                                                N_tnd,
                                                model.drift,
                                                N_mu,
                                                N_deps,
                                                dt_scale,
                                                dt_interp_scale,
                                                dt_interp_overide,
                                                rtu[ii],
                                                rtl[ii],
                                                rt_max[ii])
    
        elif ( type(model) == fullDDM ):
            logl_op[ii] = loglike_class.LogLike(functions_constant.loglikelihood,
                                                model.start,
                                                model.non_decision,
                                                N_tnd,
                                                model.drift,
                                                N_mu,
                                                N_deps,
                                                dt_scale,
                                                dt_interp_scale,
                                                dt_interp_overide,
                                                rtu[ii],
                                                rtl[ii],
                                                rt_max[ii])

        elif ( type(model) == leakage ):
            logl_op[ii] = loglike_class.LogLike(functions_leakage.loglikelihood,
                                                model.start,
                                                model.non_decision,
                                                N_tnd,
                                                'constant',
                                                N_mu,
                                                N_deps,
                                                dt_scale,
                                                dt_interp_scale,
                                                dt_interp_overide,
                                                rtu[ii],
                                                rtl[ii],
                                                rt_max[ii])
            
        elif ( type(model) == changing_thresholds ):
            logl_op[ii] = loglike_class.LogLike(functions_changing_thresholds.loglikelihood,
                                                model.start,
                                                model.non_decision,
                                                N_tnd,
                                                model.drift,
                                                N_mu,
                                                N_deps,
                                                dt_scale,
                                                dt_interp_scale,
                                                dt_interp_overide,
                                                rtu[ii],
                                                rtl[ii],
                                                rt_max[ii])

        elif ( type(model) == UGM ):
            logl_op[ii] = loglike_class.LogLike(functions_UGM.loglikelihood,
                                                model.start,
                                                model.non_decision,
                                                N_tnd,
                                                'constant',
                                                N_mu,
                                                N_deps,
                                                dt_scale,
                                                dt_interp_scale,
                                                dt_interp_overide,
                                                rtu[ii],
                                                rtl[ii],
                                                rt_max[ii])

        elif ( type(model) == UGM_flip ):
            logl_op[ii] = loglike_class.LogLike(functions_UGM_flip.loglikelihood,
                                                model.start,
                                                model.non_decision,
                                                N_tnd,
                                                'constant',
                                                N_mu,
                                                N_deps,
                                                dt_scale,
                                                dt_interp_scale,
                                                dt_interp_overide,
                                                rtu[ii],
                                                rtl[ii],
                                                rt_max[ii])
            
        elif ( type(model) == DMC ):
            logl_op[ii] = loglike_class.LogLike(functions_DMC.loglikelihood,
                                                model.start,
                                                model.non_decision,
                                                N_tnd,
                                                'constant',
                                                N_mu,
                                                N_deps,
                                                dt_scale,
                                                dt_interp_scale,
                                                dt_interp_overide,
                                                rtu[ii],
                                                rtl[ii],
                                                rt_max[ii])
            
            
    # build model and run parameter inference
    if (save_loglikelihood == False):
        with pm.Model() as hm:
                
            for ii in range(N_priors):
                if ( isinstance(priors[ priors_keys[ii] ], str) ):
                    priors[ priors_keys[ii] ] = eval( 'pm.' + priors[ priors_keys[ii] ] )
            priors = list(priors.values())
                    
            for ii in range(N_conditions):
                for jj in range(N_phi):
                    if (isinstance(phi[ii][jj], str)):
                        phi[ii][jj] = eval(phi[ii][jj])
                phi[ii] = at.as_tensor_variable(phi[ii])
                
            for ii in range(N_conditions):
                for jj in range(N_psi):
                    if (isinstance(psi[ii][jj], str)):
                        psi[ii][jj] = eval(psi[ii][jj])
                psi[ii] = at.as_tensor_variable(psi[ii])

            for ii in range(N_conditions):
                pm.Potential('ll' + str(ii), logl_op[ii](phi[ii], psi[ii]))
                
            if (sampler == 'DEMetropolisZ'):
                step = pm.DEMetropolisZ(tune = 'lambda')
                idata = pm.sample(draws = samples, tune = tune, step = step, compute_convergence_checks = False, chains = chains, cores = cores, return_inferencedata = True, progressbar = progressbar);
            elif (sampler == 'DEMetropolis'):            
                step = pm.DEMetropolis(tune = 'lambda')
                idata = pm.sample(draws = samples, tune = tune, step = step, compute_convergence_checks = False, chains = chains, cores = cores, return_inferencedata = True, progressbar = progressbar);

    elif (save_loglikelihood == True):
        with pm.Model() as hm:
                
            for ii in range(N_priors):
                if ( isinstance(priors[ priors_keys[ii] ], str) ):
                    priors[ priors_keys[ii] ] = eval( 'pm.' + priors[ priors_keys[ii] ] )
            priors = list(priors.values())
                    
            for ii in range(N_conditions):
                for jj in range(N_phi):
                    if (isinstance(phi[ii][jj], str)):
                        phi[ii][jj] = eval(phi[ii][jj])
                phi[ii] = at.as_tensor_variable(phi[ii])
                
            for ii in range(N_conditions):
                for jj in range(N_psi):
                    if (isinstance(psi[ii][jj], str)):
                        psi[ii][jj] = eval(psi[ii][jj])
                psi[ii] = at.as_tensor_variable(psi[ii])

            llh_det = [0]*N_conditions
            for ii in range(N_conditions):
                llh_det[ii] = pm.Deterministic('llh' + str(ii), logl_op[ii](phi[ii], psi[ii]))
            pm.Potential('ll', at.sum(llh_det))
                
            if (sampler == 'DEMetropolisZ'):
                step = pm.DEMetropolisZ(tune = 'lambda')
                idata = pm.sample(draws = samples, tune = tune, step = step, compute_convergence_checks = False, chains = chains, cores = cores, return_inferencedata = True, progressbar = progressbar);
            elif (sampler == 'DEMetropolis'):            
                step = pm.DEMetropolis(tune = 'lambda')
                idata = pm.sample(draws = samples, tune = tune, step = step, compute_convergence_checks = False, chains = chains, cores = cores, return_inferencedata = True, progressbar = progressbar);
            
    # save .nc file containing idata information
    idata.to_netcdf(file_name + '.nc')
    return idata
    
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

def plot_idata(file_name, burnin, combined = False, compact = False):

    # check if samples input properly
    if (isinstance(burnin, int) == False):
        raise RuntimeError("burnin must be of type int.")
    if (burnin < 0):
        raise RuntimeError("burnin must be an integer more than or equal to zero.")

    subject = file_name + '.nc'
    idata = az.from_netcdf(subject)
    idata_plot = az.plot_trace(idata.sel(draw=slice(burnin,None)), combined = combined, compact = compact);
    return idata_plot;

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

def summary(file_name, burnin):

    # check if samples input properly
    if (isinstance(burnin, int) == False):
        raise RuntimeError("burnin must be of type int.")
    if (burnin < 0):
        raise RuntimeError("burnin must be an integer more than or equal to zero.")

    subject = file_name + '.nc'
    trace = az.from_netcdf(subject)
    summary = az.summary(trace.sel(draw=slice(burnin,None)));
    return summary
                    