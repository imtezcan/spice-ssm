"""
File: custom.py
Description: Functions for the pybeam.custom sub-module.
Creator: Matt Murrow
Email: matthew.a.murrow@vanderbilt.edu

"""

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
Import Python modules and other functions required by pybeam.

"""

# standard python module imports
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as tt
import arviz as az
import warnings
import sys

# import pybeam functions
from . import loglike_class

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""

"""

def functions_test(model_dir, phi, x, t):

    # -------------------------------------------------- #
    # -------------------------------------------------- #

    """
    Check if function inputs are correct.

    """

    # check if model_dir is input properly
    if (isinstance(model_dir, str) == False):
        raise RuntimeError("model_dir must be a string containing the full path of your model file.")

    # import functions file
    sys.path.insert(1, model_dir)
    import functions

    # chceck if function is input properly
    functions_list = ['non_decision_time',
                      'relative_start',
                      'drift',
                      'diffusion',
                      'upper_decision_threshold',
                      'lower_decision_threshold',
                      'contamination_strength',
                      'contamination_probability',
                      'modify_dt']

    # check if phi is input properly
    if (isinstance(phi, dict) == False):
        raise RuntimeError("phi must be a dictionary.")

    # check if input array phi has same amount of parameters as N_phi
    check = functions.model_check(len(phi));
    if (check == False):
        raise RuntimeError("Length of dictionary phi does not equal N_phi.")

    # check if dictionary phi has the correct keys (0, 1, ... , N_phi)
    for ii in range(len(phi)):
        if ( 'phi[' + str(ii) + ']' in phi):
            continue
        else:
            raise RuntimeError("Keys for phi must be 'phi[0]', 'phi[1]', ... , 'phi[N_phi-1]'.")

    # check if x is input properly
    if (isinstance(x, (float,int)) == False):
        raise RuntimeError("x must be of typer float or int.")

    # check if t is input properly
    if (isinstance(t, (float,int)) == False):
        raise RuntimeError("x must be of typer float or int.")

    # -------------------------------------------------- #
    # -------------------------------------------------- #

    """
    Load input dictionary phi into list.

    """
        
    phi_list = [0]*len(phi)
    for ii in range(len(phi)):
        phi_list[ii] = phi['phi[' + str(ii) + ']']

    # -------------------------------------------------- #
    # -------------------------------------------------- #

    """
    Run relevant function, return their outputs.

    """

    functions_out = [[0 for i in range(2)] for j in range(len(functions_list))]
    for ii in range(len(functions_list)):
        functions_out[ii][0] = functions_list[ii]
        functions_out[ii][1] = functions.model_functions_check(functions_list[ii], phi_list, x, t)

    return functions_out

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
function: simulate_model
Simulates model defined in model.pyx.

Inputs:

    N_sims (int) -
        Number of accumulators to simulate.

    model_dir (str) -
        Sets directory where model .so/.pyd file is located at.

    phi (dict) -
        Dictionary containing model parameter values. Keys are the model parameters, while values are the 
        value associated with that parameter. If keys are unknown for your model, run function parse_model.
        The list output by that function provides the keys for this dictionary.

    seed (int, False); OPTIONAL, defaults to False -
        Sets the random number seed. Defaults to False. If False, program randomly chooses a seed.

    dt (float) OPTIONAL - 
        Sets the simulation time step. By default, dt = 1.0e-4.
        
Outputs:

    (dict) containing two keys: 'rt_upper' and 'rt_lower'. These contain the reaction times for the upper
    and lower threshold crossings, respectively.

"""

def simulate(N_sims, model_dir, phi, seed = False, dt = 0.0001, get_traces=False):

    # -------------------------------------------------- #
    # -------------------------------------------------- #

    """
    Check if function inputs are of correct form.

    """

    # check if model_dir is input properly
    if (isinstance(model_dir, str) == False):
        raise RuntimeError("model_dir must be a string containing the full path of your model file.")

    # import functions file
    sys.path.insert(1, model_dir)
    import functions

    # check if N_sims is input properly
    if (isinstance(N_sims, int) == False):
        raise RuntimeError("N_sims must be an integer.")
        
    # check if phi is input properly
    if (isinstance(phi, dict) == False):
        raise RuntimeError("phi must be a dictionary.")
        
    # check if input array phi has same amount of parameters as N_phi
    check = functions.model_check(len(phi));
    if (check == False):
        raise RuntimeError("Length of dictionary phi does not equal N_phi.")
        
    # check if dictionary phi has the correct keys ('phi[0]', 'phi[1]', ... , 'phi[N_phi-1]')
    for ii in range(len(phi)):
        if ( 'phi[' + str(ii) + ']' in phi):
            continue
        else:
            raise RuntimeError("Keys for phi must be 'phi[0]', 'phi[1]', ... , 'phi[N_phi-1]'.")
        
    # check if seed is input properly
    if (seed != False):
        if (isinstance(seed, int) == False):
            raise RuntimeError("seed must be of type int.")
    elif (seed < 0):
        raise RuntimeError("seed must be positive.")
            
    # check if dt is input properly
    if (isinstance(dt, float) == False):
        raise RuntimeError("dt must be of type float.")
    elif (dt <= 0.0):
        raise RuntimeError("dt must be more than zero.")
    
    # -------------------------------------------------- #
    # -------------------------------------------------- #

    """
    Parse model and phi inputs.

    """
        
    phi_list = [0]*len(phi)
    for ii in range(len(phi)):
        phi_list[ii] = phi['phi[' + str(ii) + ']']
        
    # -------------------------------------------------- #
    # -------------------------------------------------- #
    
    """
    Simulate model, output dictonary of reaction times.

    """
                
    if (seed == False):
        seed = np.random.randint(0,100000)
    # return [N_sims, dt, phi_list, seed]
    if not get_traces:
        rt = np.asarray(functions.simulate_model_c(N_sims, dt, phi_list, seed))
    else:
        rt, traces = functions.simulate_model_c(N_sims, dt, phi_list, seed)
    rt_upper = rt[rt >= 0.0]
    rt_lower = -rt[rt < 0.0]
    rt_dict = { 'rt_upper' : rt_upper ,
                'rt_lower' : rt_lower }
                
    if not get_traces:
        return rt_dict
    else:
        return rt_dict, traces

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
function: model_rt
Outputs the response time distribution predicted by the model. Also
allows for calculation of the loglikelihood of input data.

Inputs:

    model_dir (str) -
        Sets directory where model .so/.pyd file is located at.

    phi (list, numpy array) -
        List or array of parameters as defined in "model.pyx".

    x_res (str), OPTIONAL - 
        Sets spatial resolution of Fokker-Planck solver (Crank-Nicolson algorithm). 'default', the default 
        setting, places 101  spatial steps between the two decision thresholds. 'high' places 151 spatial 
        steps, 'very_high' places 251 spatial steps, while 'max' places 501 spatial steps. 'default' is
        recommened in nearly all cases.
        
    t_res (str), OPTIONAL - 
        Sets temporal resolution of Fokker-Planck solver (Crank-Nicolson algorithm). The time step used
        by the solver is set by simulating 10 accumulators from the model and finding their average rt.
        This is then mulitiplied by a constant determined by t_res. 'default', the default setting,
        multiplies this by 0.025. 'high' multiplies this by 0.0175, 'very_high' by 0.01, and 'max' by
        0.005. 'default' is recommended in nearly all cases.

Outputs:

    (dict) containing three keys: 'time',  rt_upper' and 'rt_lower'. These contain the reaction time 
    distributions (likelihood functions) for the upper and lower threshold crossings, respectively.

"""
    
def likelihood(model_dir, phi, rt_max, x_res = 'default', t_res = 'default'):

    # -------------------------------------------------- #
    # -------------------------------------------------- #

    """
    Check if function inputs are correct.

    """

    # check if model_dir is input properly
    if (isinstance(model_dir, str) == False):
        raise RuntimeError("model_dir must be a string containing the path of your model file.")

    # import functions file
    # sys.path.append(model_dir)
    sys.path.insert(1, model_dir)
    import functions

    # check if parameters is input properly
    if (isinstance(phi, dict) == False):
        raise RuntimeError("phi must be a dictionary.")
    
    # check if input array phi has same amount of parameters as N_phi
    check = functions.model_check(len(phi));
    if (check == False):
        raise RuntimeError("Length of dictionary phi does not equal N_phi.")
        
    # check if dictionary phi has the correct keys ('phi[0]', 'phi[1]', ... , 'phi[N_phi-1]')
    for ii in range(len(phi)):
        if ( 'phi[' + str(ii) + ']' in phi):
            continue
        else:
            raise RuntimeError("Keys for phi must be 'phi[0]', 'phi[1]', ... , 'phi[N_phi-1]'.")
        
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
    
    # check if rt_max is input properly
    if (isinstance(rt_max, float) == False) or (rt_max <= 0.0):
        raise RuntimeError("rt_max must be a float more than or equal to zero.")
    
    # -------------------------------------------------- #
    # -------------------------------------------------- #

    """
    Parse phi input.

    """ 
    
    phi_list = [0]*len(phi)
    for ii in range(len(phi)):
        phi_list[ii] = phi['phi[' + str(ii) + ']']

    # -------------------------------------------------- #
    # -------------------------------------------------- #

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
        
    # -------------------------------------------------- #
    # -------------------------------------------------- #
    
    """
    Calculate model rt distribution.

    """
       
    p_fpt = np.asarray( functions.solveFP(phi_list, N_deps, dt_scale, [rt_max], [rt_max], False) )
    
    jj = len(p_fpt[0])
    for ii in range(len(p_fpt[0])):
        if (p_fpt[0,ii] == 0.0):
            jj = ii
            break

    mrt = { 'time' : p_fpt[0,:jj] , 'lh_upper' : p_fpt[1,:jj] , 'lh_lower' : p_fpt[2,:jj] }

    return mrt

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
function: model_loglike
Calculate the loglikelihood of a data set from the model's likelihood function.

Inputs:

    model_dir (str) -
        Sets directory where model .so/.pyd file is located at.

    phi (dict) -
        Dictionary containing model parameter values. Keys are the model parameters, while values are the 
        value associated with that parameter. If keys are unknown for your model, run function parse_model.
        The list output by that function provides the keys for this dictionary.
        
    rt (dict) -
        Dictionary contining two keys 'rt_upper' and 'rt_lower'. The former contians the reaction time data
        for the upper threshold crossing, while the latter contains the data for the lower decison threshold
        crossing.

    x_res (str), OPTIONAL - 
        Sets spatial resolution of Fokker-Planck solver (Crank-Nicolson algorithm). 'default', the default 
        setting, places 101  spatial steps between the two decision thresholds. 'high' places 151 spatial 
        steps, 'very_high' places 251 spatial steps, while 'max' places 501 spatial steps. 'default' is
        recommened in nearly all cases.
        
    t_res (str), OPTIONAL - 
        Sets temporal resolution of Fokker-Planck solver (Crank-Nicolson algorithm). The time step used
        by the solver is set by simulating 10 accumulators from the model and finding their average rt.
        This is then mulitiplied by a constant determined by t_res. 'default', the default setting,
        multiplies this by 0.025. 'high' multiplies this by 0.0175, 'very_high' by 0.01, and 'max' by
        0.005. 'default' is recommended in nearly all cases.
        
Outputs:

    (float) The loglikelihood of the input data set calculated from the model rt (likelihood function).

"""
    
def loglikelihood(model_dir, phi, rt, x_res = 'default', t_res = 'default'):

    # -------------------------------------------------- #
    # -------------------------------------------------- #

    """
    Check if function inputs are correct.

    """

    # check if model_dir is input properly
    if (isinstance(model_dir, str) == False):
        raise RuntimeError("model_dir must be a string containing the path of your model file.")

    # import functions file
    sys.path.insert(1, model_dir)
    import functions

    # check if parameters is input properly
    if (isinstance(phi, dict) == False):
        raise RuntimeError("phi must be a dictionary.")
    
    # check if input array phi has same amount of parameters as N_phi
    check = functions.model_check(len(phi));
    if (check == False):
        raise RuntimeError("Length of dictionary phi does not equal N_phi.")
        
    # check if dictionary phi has the correct keys ('phi[0]', 'phi[1]', ... , 'phi[N_phi-1]')
    for ii in range(len(phi)):
        if ( 'phi[' + str(ii) + ']' in phi):
            continue
        else:
            raise RuntimeError("Keys for phi must be 'phi[0]', 'phi[1]', ... , 'phi[N_phi-1]'.")

    # check if rt is input properly
    if (isinstance(rt, dict) == False):
        raise RuntimeError("rt must be a dictionary with two entries: 'rt_upper' and 'rt_lower'. Otherwise set to False")
        
    if (('rt_upper' in rt) == False) & (('rt_lower' in rt) == False):
        raise RuntimeError("rt must be a dictionary with two entries: 'rt_upper' and 'rt_lower'.")
        
    if (isinstance(rt['rt_upper'], (list, np.ndarray)) == False) & (isinstance(rt['rt_lower'], (list, np.ndarray)) == False):
        raise RuntimeError("rt dictionary entries must be lists or numpy arrays.")
        
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
    
    # -------------------------------------------------- #
    # -------------------------------------------------- #

    """
    Parse phi input.

    """
    
    phi_list = [0]*len(phi)
    for ii in range(len(phi)):
        phi_list[ii] = phi['phi[' + str(ii) + ']']
        
    # -------------------------------------------------- #
    # -------------------------------------------------- #

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
        
    # -------------------------------------------------- #
    # -------------------------------------------------- #
    
    """
    Calculate loglikelihood of rt data from model rt distribution.

    """

    # check if rt is input properly
    if (isinstance(rt, dict) == False):
        raise RuntimeError("rt must be a dictionary with two entries: 'rt_upper' and 'rt_lower'. Otherwise set to False")
        
    if (('rt_upper' in rt) == False) & (('rt_lower' in rt) == False):
        raise RuntimeError("rt must be a dictionary with two entries: 'rt_upper' and 'rt_lower'.")
        
    if (isinstance(rt['rt_upper'], (list, np.ndarray)) == False) & (isinstance(rt['rt_lower'], (list, np.ndarray)) == False):
        raise RuntimeError("rt dictionary entries must be lists or numpy arrays.")

    loglike = functions.solveFP(phi_list, N_deps, dt_scale, rt['rt_upper'], rt['rt_lower'], True)
    return loglike

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
function: plot_rt
    Generates a figure containing the model rt distributions (red lines) and, optionally is rt data is given,
    the data rt distribution (grey histogram).

Inputs:

    model_dir (str) -
        Sets directory where model .so/.pyd file is located at.

    phi (dict) -
        Dictionary containing model parameter values. Keys are the model parameters, while values are the 
        value associated with that parameter. If keys are unknown for your model, run function parse_model.
        The list output by that function provides the keys for this dictionary.
        
    rt (False, dict) OPTIONAL -
        Dictionary contining two keys 'rt_upper' and 'rt_lower'. The former contians the reaction time data
        for the upper threshold crossing, while the latter contains the data for the lower decison threshold
        crossing.

    x_res (str), OPTIONAL - 
        Sets spatial resolution of Fokker-Planck solver (Crank-Nicolson algorithm). 'default', the default 
        setting, places 101  spatial steps between the two decision thresholds. 'high' places 151 spatial 
        steps, 'very_high' places 251 spatial steps, while 'max' places 501 spatial steps. 'default' is
        recommened in nearly all cases.
        
    t_res (str), OPTIONAL - 
        Sets temporal resolution of Fokker-Planck solver (Crank-Nicolson algorithm). The time step used
        by the solver is set by simulating 10 accumulators from the model and finding their average rt.
        This is then mulitiplied by a constant determined by t_res. 'default', the default setting,
        multiplies this by 0.025. 'high' multiplies this by 0.0175, 'very_high' by 0.01, and 'max' by
        0.005. 'default' is recommended in nearly all cases.
        
    bins (False, int), OPTIONAL -
        Sets how many histogram bins to be used to plot the data. If False, automatically sets the amount
        of bins using the Freedman Diaconis Estimator.
        
Outputs:

    (fig) Figure containing the model rt distribution (red) and the rt data (optional, grey histogram).

"""

def plot_rt(model_dir, phi, rt_max, x_res = 'default', t_res = 'default', rt = False, bins = False):

    # -------------------------------------------------- #
    # -------------------------------------------------- #

    """
    Check if function inputs are correct.

    """

    # check if model_dir is input properly
    if (isinstance(model_dir, str) == False):
        raise RuntimeError("model_dir must be a string containing the path of your model file.")

    # import functions file
    sys.path.insert(1, model_dir)
    import functions

    # check if parameters is input properly
    if (isinstance(phi, dict) == False):
        raise RuntimeError("phi must be a dictionary.")
    
    # check if input array phi has same amount of parameters as N_phi
    check = functions.model_check(len(phi));
    if (check == False):
        raise RuntimeError("Length of dictionary phi does not equal N_phi.")
        
    # check if dictionary phi has the correct keys ('phi[0]', 'phi[1]', ... , 'phi[N_phi-1]')
    for ii in range(len(phi)):
        if ( 'phi[' + str(ii) + ']' in phi):
            continue
        else:
            raise RuntimeError("Keys for phi must be 'phi[0]', 'phi[1]', ... , 'phi[N_phi-1]'.")
        
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
    
    # check if rt_max is input properly
    if (isinstance(rt_max, float) == False) or (rt_max <= 0.0):
        raise RuntimeError("rt_max must be a float more than or equal to zero.")
    
    # -------------------------------------------------- #
    # -------------------------------------------------- #

    """
    Parse phi input.

    """  
    
    phi_list = [0]*len(phi)
    for ii in range(len(phi)):
        phi_list[ii] = phi['phi[' + str(ii) + ']']
        
    # -------------------------------------------------- #
    # -------------------------------------------------- #

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

    # -------------------------------------------------- #
    # -------------------------------------------------- #

    """
    Generate figures with data and likelihood function.

    """
        
    p_fpt = np.asarray( functions.solveFP(phi_list, N_deps, dt_scale, [rt_max], [rt_max], False) )
    
    jj = len(p_fpt[0])
    for ii in range(len(p_fpt[0])):
        if (p_fpt[0,ii] == 0.0):
            jj = ii
            break
        
    fig, axs = plt.subplots(1, 1, figsize=(6,4))
        
    if (rt != False):
    
        # check if rt is input properly
        if (isinstance(rt, dict) == False):
            raise RuntimeError("rt must be a dictionary with two entries: 'rt_upper' and 'rt_lower'.")
            
        if (('rt_upper' in rt) == True) & (('rt_lower' in rt) == True):
            rt_array = np.concatenate(( np.asarray(rt['rt_upper']), -np.asarray(rt['rt_lower']) ))
        else:
            raise RuntimeError("rt must be a dictionary with two entries: 'rt_upper' and 'rt_lower'.")
            
        if (isinstance(rt['rt_upper'], (list, np.ndarray)) == False) & (isinstance(rt['rt_lower'], (list, np.ndarray)) == False):
            raise RuntimeError("rt dictionary entries must be lists or numpy arrays.")
        
        # if bins not input by user, approximate appropriate amount
        if (bins == False):
            bins = int(2*len(np.histogram_bin_edges(rt_array, bins = 'fd')))
            
        # plot histogram of user data
        axs.hist(rt_array, bins=bins, density = True, edgecolor='k', color = 'grey', alpha = 0.75)
        
    axs.plot(p_fpt[0,:jj], p_fpt[1,:jj], linewidth = 3, color = 'r')
    axs.plot(-p_fpt[0,:jj], p_fpt[2,:jj], linewidth = 3, color = 'r')
    
    return axs

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
function: inference
    Runs MCMC routine using PyMC3.

Inputs:

    model_dir (str) -
        Sets directory where model .so/.pyd file is located at.

    priors (dict) -
        Dictionary containing parameter priors. Key names are arbitrary. The values for each key are 
        strings written in PyMC3's syntax for priors. See below for example.
        
    conditions (dict) -
        Dictionary containing dictionaries for each model condition. See below and example files for use.
        
    samples (int) -
        Sets the number of MCMC samples to run. Recommend at least 25000 samples.
        
    chains (int) -
        Sets the number of MCMC chains to run. Recomend at least 3 chains.
        
    cores (int) - 
        Sets the number of cpu cores to run the chains on.
        
    file_name (str) -
        Sets the name of the .nc file output by the solver containing the posterios. Automatically adds 
        the .nc extension to the string.
        
    solver (str), OPTIONAL -
        Sets the MCMC algorithm used by PyMC3. Defaults to 'DEMetropolisZ' (recommended), but can be changed
        to 'DEMetropolis' and 'Slice'.

    x_res (str), OPTIONAL - 
        Sets spatial resolution of Fokker-Planck solver (Crank-Nicolson algorithm). 'default', the default 
        setting, places 101  spatial steps between the two decision thresholds. 'high' places 151 spatial 
        steps, 'very_high' places 251 spatial steps, while 'max' places 501 spatial steps. 'default' is
        recommened in nearly all cases.
        
    t_res (str), OPTIONAL - 
        Sets temporal resolution of Fokker-Planck solver (Crank-Nicolson algorithm). The time step used
        by the solver is set by simulating 10 accumulators from the model and finding their average rt.
        This is then mulitiplied by a constant determined by t_res. 'default', the default setting,
        multiplies this by 0.025. 'high' multiplies this by 0.0175, 'very_high' by 0.01, and 'max' by
        0.005. 'default' is recommended in nearly all cases.
        
    tune (int), OPTIONAL -
        Sets the amount of MCMC tuning steps. Defaults to zero (recommended for DEMetropolisZ and
        DEMetropolis, but can sometimes be useful. Test on data if need be). If using Slice, 
        tuning steps are necessary (recommend at least 500).
        
Outputs:

    (trace object) Posterior trace information.

    (file) Outputs .nc file containing all trace information. See pycm3/arviz documentation for how
    to work with this file type.

"""

def inference(model_dir, priors, conditions, samples, chains, cores, file_name, solver = 'DEMetropolisZ', x_res = 'default', t_res = 'default', tune = 0, save_loglikelihood = False):
    
    # -------------------------------------------------- #
    # -------------------------------------------------- #

    """
    Check if function inputs are correct.

    """

    # check if model_dir is input properly
    if (isinstance(model_dir, str) == False):
        raise RuntimeError("model_dir must be a string containing the path of your model file.")

    # import functions file
    sys.path.insert(1, model_dir)
    import functions
    
    # check if priors input properly
    if (isinstance(priors, dict) == False):
        raise RuntimeError("priors must be a dictionary.")
        
    # check if conditions input properly
    if (isinstance(conditions, dict) == False):
        raise RuntimeError("conditions must be a dictionary. See documentation for proper use.")
        
    # check if conditions has the right key definitions
    for ii in range(len(conditions)):
        if ii in conditions:
            continue
        else:
            raise RuntimeError("Keys for condition must be positive integers going from 0, 1, ... , N_conditions-1.")
            
    # check if conditions sub-dictionaries are defined properly
    for ii in range(len(conditions)):        
    
        # check if conditions has the proper amount of inputs
        check = functions.model_check(len(conditions[ii])-1)
        if (check == False):
            raise RuntimeError("Condition dictonary has an incorrect number of inputs.")
            
        # check if reaction time data present in condition dictionary
        if (('rt' in conditions[ii]) == False):
            raise RuntimeError("'rt' is a required input for each condition.")
        
        if (isinstance(conditions[ii]['rt'], dict) == False):
            raise RuntimeError("rt must be a dictionary with two entries: 'rt_upper' and 'rt_lower'.")
            
        # check if dictionary phi has the correct keys (0, 1, ... , N_phi)
        for jj in range(len(conditions[ii])-1):
            if ( 'phi[' + str(jj) + ']' in conditions[ii]):
                continue
            else:
                raise RuntimeError("Keys for each condition must be positive integers going from 0, 1, ... , N_phi-1.")
        
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
    
    # check if solver input properly   
    solver_list = ['DEMetropolisZ', 'Slice', 'DEMetropolis']
    test = solver not in solver_list
    if (test == True):
        raise RuntimeError("solver must equal 'DEMetropolisZ' or 'Slice'.")
        
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
        
    # check if tune input properly
    if (isinstance(tune, int) == False):
        raise RuntimeError("tune must be an integer.")
        
    # -------------------------------------------------- #
    # -------------------------------------------------- #

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
    
    # -------------------------------------------------- #
    # -------------------------------------------------- #

    """
    Run parameter inference.

    """
    
    # define pymc3 model
    if (save_loglikelihood == False):
        with pm.Model() as model:

            # load priors into tensor variable
            N_conditions = len(conditions)
            phi = [0]*N_conditions
            priors_pybeam = priors.copy()
            
            priors_list = list(priors.keys())
            for ii in range(len(priors_list)):
                if ( isinstance(priors[priors_list[ii]], float) == False ) & ( isinstance(priors[priors_list[ii]], int) == False ):
                    priors_pybeam[priors_list[ii]] = eval( 'pm.' + priors[ priors_list[ii] ] )
                    
            for ii in range(N_conditions):
                phi_in = conditions[ii]
                N_phi = len(phi_in) - 1

                phi_temp = [0]*N_phi
                for jj in range(N_phi):
                    phi_temp[jj] = priors_pybeam[conditions[ii]['phi[' + str(jj) + ']']]
                phi[ii] = tt.as_tensor_variable( phi_temp )
                
            # Bayesian parameter estimation
            for ii in range(N_conditions):
                
                # call theano op to define function for loglikelihood    
                logl_op = loglike_class.LogLike(functions.solveFP, N_deps, dt_scale, conditions[ii]['rt']['rt_upper'], conditions[ii]['rt']['rt_lower'], True)
                def logl_fun(parameters):
                    return logl_op(parameters)
                pm.Potential( str(ii) , logl_fun(phi[ii]) )
                
            if (solver == 'DEMetropolisZ'):
                step = pm.DEMetropolisZ()
                trace = pm.sample(draws = samples, tune = tune, step = step, compute_convergence_checks = False, chains = chains, cores = cores, return_inferencedata = True);
            elif (solver == 'DEMetropolis'):            
                step = pm.DEMetropolis(tune = 'lambda')
                trace = pm.sample(draws = samples, tune = tune, step = step, compute_convergence_checks = False, chains = chains, cores = cores, return_inferencedata = True);
            elif (solver == 'Slice'):            
                if (tune == 0):
                    warnings.warn("The Slice sampler needs to be tuned (recommend tune >= 500).")
                step = pm.Slice()
                trace = pm.sample(draws = samples, tune = tune, step = step, compute_convergence_checks = False, chains = chains, cores = cores, return_inferencedata = True);
    
    elif (save_loglikelihood == True):
        with pm.Model() as model:

            # load priors into tensor variable
            N_conditions = len(conditions)
            phi = [0]*N_conditions
            priors_pybeam = priors.copy()
            
            priors_list = list(priors.keys())
            for ii in range(len(priors_list)):
                if ( isinstance(priors[priors_list[ii]], float) == False ) & ( isinstance(priors[priors_list[ii]], int) == False ):
                    priors_pybeam[priors_list[ii]] = eval( 'pm.' + priors[ priors_list[ii] ] )
                    
            for ii in range(N_conditions):
                phi_in = conditions[ii]
                N_phi = len(phi_in) - 1

                phi_temp = [0]*N_phi
                for jj in range(N_phi):
                    phi_temp[jj] = priors_pybeam[conditions[ii]['phi[' + str(jj) + ']']]
                phi[ii] = tt.as_tensor_variable( phi_temp )
                
            # Bayesian parameter estimation
            llh_det = [0]*N_conditions
            for ii in range(N_conditions):
                
                # call theano op to define function for loglikelihood    
                logl_op = loglike_class.LogLike(functions.solveFP, N_deps, dt_scale, conditions[ii]['rt']['rt_upper'], conditions[ii]['rt']['rt_lower'], True)
                def logl_fun(parameters):
                    return logl_op(parameters)
                pm.Potential( str(ii) , logl_fun(phi[ii]) )

                llh_det[ii] = pm.Deterministic('llh' + str(ii), logl_fun(phi[ii]))
                
            if (solver == 'DEMetropolisZ'):
                step = pm.DEMetropolisZ()
                trace = pm.sample(draws = samples, tune = tune, step = step, compute_convergence_checks = False, chains = chains, cores = cores, return_inferencedata = True);
            elif (solver == 'DEMetropolis'):            
                step = pm.DEMetropolis(tune = 'lambda')
                trace = pm.sample(draws = samples, tune = tune, step = step, compute_convergence_checks = False, chains = chains, cores = cores, return_inferencedata = True);
            elif (solver == 'Slice'):            
                if (tune == 0):
                    warnings.warn("The Slice sampler needs to be tuned (recommend tune >= 500).")
                step = pm.Slice()
                trace = pm.sample(draws = samples, tune = tune, step = step, compute_convergence_checks = False, chains = chains, cores = cores, return_inferencedata = True);

    # save .nc file containing trace information
    trace.to_netcdf(file_name + '.nc')
       
    # output trace information
    return trace;

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
function: plot_trace
    Plots figure containing all posteriors.

Inputs:

    file_name (str) -
        File name as a string. Must be a .nc file. Do not include the .nc extension in the string,
        it is automatically added by the program.
        
    burnin (int) -
        Number of samples to throw out form the posterior when plotting.
        
Outputs:

    (fig) Figure containing all posteriors.

"""
    
def plot_idata(file_name, burnin, combine_chains = False):
    trace_in = az.from_netcdf(file_name + '.nc')
    trace = trace_in.sel(draw=slice(burnin,None))
    if (combine_chains == True):
        trace_comb = trace.posterior.stack(draws=("chain", "draw"))
        trace_plot = az.plot_trace(trace_comb);    
    else:
        trace_plot = az.plot_trace(trace);
    return trace_plot;

# ---------------------------------------------------------------------- #    
# ---------------------------------------------------------------------- #    
# ---------------------------------------------------------------------- #

"""
function: summary
    Generates data frame containing useful posetior information. Pulled from the arviz function, see their
    documentation for understaning table information.

Inputs:

    file_name (str) -
        File name as a string. Must be a .nc file. Do not include the .nc extension in the string,
        it is automatically added by the program.
        
    burnin (int) -
        Number of samples to throw out form the posterior when plotting.
        
Outputs:

    (df) Data frame containing useful trace information.

"""
    
def summary(file_name, burnin):
    subject = file_name + '.nc'
    trace = az.from_netcdf(subject)
    summary = az.summary(trace.sel(draw=slice(burnin,None)));
    return summary
