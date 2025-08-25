"""
# Import Python packages and C libraries.

"""

import cython
cimport cython

from libc.math cimport pow, exp, log, sin, cos

"""
# Model parameter functions.

"""

# total number of input parameters
DEF N_phi = 5

# function for the non-decision time
cdef double non_decision_time(double phi[N_phi]):
    return phi[0]

# function for the relative start point
cdef double relative_start(double phi[N_phi]):
    return phi[1]

# function for the drift rate
cdef double drift(double phi[N_phi], double x, double t):
    cdef double mu = phi[2]
    cdef double v = 0.0

    v = mu + t * 2
    # print(f't: {t}, mu: {mu}, v: {v}')

    return v

# function for the diffusion rate
cdef double diffusion(double phi[N_phi], double x, double t):
    return phi[3]

# function for the upper decision threshold
cdef double upper_decision_threshold(double phi[N_phi], double t):
    return phi[4]

# function for the lower decision threshold
cdef double lower_decision_threshold(double phi[N_phi], double t):
    return -phi[4]

# function for the contamination strength
cdef double contamination_strength(double phi[N_phi]):
    return 0.0

# function for the contamination probability
cdef double contamination_probability(double phi[N_phi], double t):
    return 0.0
        
"""
# Function to modify time step with unusual likelihood functions.

"""

# function to modify the time step
cdef double modify_dt(double phi[N_phi], double t):
    return 1.0
