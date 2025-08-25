# module imports
import pytensor.tensor as tt
import numpy as np

# define a theano Op for our likelihood function
class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. We pass a vector of values phi (the parameters that define our model)
    and return a single "scalar" value (the log-likelihood).

    """

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, N_deps, dt_scale, fpt_bu, fpt_bl, output_loglike):
    
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this program.

        Parameters
        ----------
        loglike:
            The log-likelihood function we've defined.
        model:
            Which model being used (flat, linear, exponential, weibull, or ugm).
        bounds:
            Set to symmetric or asymmetric.
        g_range:
            Range of uniform contamination model.
        N_deps:
            Number of spatial mesh points.
        dt_scale:
            Sets the factor by by which the estimated time step is multiplied.  High res is 0.01, default is 0.02.
        fpt_bu:
            Upper boundary crossing times.
        fpt_bl:
            Lower boundary crossing times.
        x_res:
            Solver spatial resolution, either 'high', 'default', or 'fast'.
        t_res:
            Solver time resolution, either 'high' or 'default'.
        output_loglike:
            If True, output loglike.  If False, output first passage time.
            
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.N_deps = N_deps
        self.dt_scale = dt_scale
        self.fpt_bu = fpt_bu
        self.fpt_bl = fpt_bl
        self.output_loglike = output_loglike

    def perform(self, node, inputs, outputs):
    
        # the method that is used when calling the Op
        (phi,) = inputs  # contains variables

        # call the log-likelihood function
        logl = self.likelihood(phi, self.N_deps, self.dt_scale, self.fpt_bu, self.fpt_bl, self.output_loglike)

        outputs[0][0] = np.array(logl)  # output the log-likelihood
