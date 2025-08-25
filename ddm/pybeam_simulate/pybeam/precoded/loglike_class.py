"""
File: loglike_class.py
Description: Defines class for loglikelihood function used by PyMC.
Creator: Matthew Murrow
Email: matthew.a.murrow@vanderbilt.edu

"""

# module imports
import numpy as np
import pytensor.tensor as at

# define a aesara Op for our likelihood function
class LogLike(at.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. We pass a two vectors of values: phi (the parameters that define our model),
    and psi (the parameters that define inter-trial varaibility in our model). It returns 
    a single "scalar" value (the log-likelihood).

    """

    itypes = [at.dvector, at.dvector]  # expects two vectors of parameter values when called
    otypes = [at.dscalar]  # outputs a single scalar value (the loglikelihood)

    def __init__(self, loglike, w_dist, tnd_dist, N_tnd, mu_dist, N_mu, N_deps, dt_scale, dt_interp_scale, dt_interp_overide, rtu, rtl, rt_max):
    
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this program.

        Parameters
        ----------
        loglike:
            The log-likelihood function we've defined.
        w_dist (str):
            The relative start distribution ('constant', 'uniform', or 'normal').
        tnd_dist (str):
            The non-decision time distribution ('constant', 'uniform', or 'normal').
        N_tnd (int):
            Number of integration points for non-decision time distribution.
        mu_dist (str):
            Drift rate distribution ('constant', 'uniform', or 'normal').
        N_mu (int):
            Number of integration points for drift rate distribution.
        N_deps (int):
            Number of spatial mesh points.
        dt_scale (positive float):
            Used to set the time step. Recommended to be 0.025.
        dt_interp_scale (positive float):
            Sets interpolated time step size. Calculated by dt_interp = dt_interp_scale*dt. Should set to 1.0.
        dt_interp_overide (positive float):
            Sets exactly what the interpolated time step is. Ignored if equals 0 (recommended).
        rtu (list or numpy array):
            List/array of upper decision threhsold crossing times.
        rtl (list or numpy array):
            List/array of lower decision threhsold crossing times.
        rt_max (positive float):
            Calculate first passage time distribution until this time.

        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.w_dist = w_dist
        self.tnd_dist = tnd_dist
        self.N_tnd = N_tnd
        self.mu_dist = mu_dist
        self.N_mu = N_mu
        self.N_deps = N_deps
        self.dt_scale = dt_scale
        self.dt_interp_scale = dt_interp_scale
        self.dt_interp_overide = dt_interp_overide
        self.rtu = rtu
        self.rtl = rtl
        self.rt_max = rt_max

    def perform(self, node, inputs, outputs):
    
        # the method that is used when calling the Op
        (phi, psi) = inputs  # contains model parameters

        # call the log-likelihood function
        logl = self.likelihood(phi, psi, self.w_dist, self.tnd_dist, self.N_tnd, self.mu_dist, self.N_mu, self.N_deps, self.dt_scale, self.dt_interp_scale, self.dt_interp_overide, self.rtu, self.rtl, self.rt_max)

        # output the log-likelihood
        outputs[0][0] = np.array(logl)