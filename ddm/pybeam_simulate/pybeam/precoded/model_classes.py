"""
File: model_classes.py
Description: Define classes for each PyBEAM model type.
Creator: Matthew Murrow
Email: matthew.a.murrow@vanderbilt.edu

"""

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
Class object for the simpleDDM model.

"""

class simpleDDM:
    
    # initialize class values
    def __init__(self, sigma = 1.0, non_decision = 'constant', start = 'constant', drift = 'constant', contamination = 'none'):
        self.sigma = sigma
        self.non_decision = non_decision
        self.start = start
        self.drift = drift
        self.contamination = contamination
        
    # determine what parameters are associated with model
    def parameters(self):
        pl = ['tnd', 'w', 'mu', 'b']
        if (self.non_decision == 'uniform') or (self.non_decision == 'normal'):
            pl.append('sd_tnd')
        if (self.start == 'uniform') or (self.start == 'normal'):
            pl.append('sd_w')
        if (self.drift == 'uniform') or (self.drift == 'normal'):
            pl.append('sd_mu')
        if (self.contamination == 'uniform'):
            pl.extend(['g', 'gl', 'gu'])
        return pl
    
    def phi_bulk(self):
        pl = self.parameters()
        phi_bulk = ['mu', 'sigma', 'b', 'g', 'gl', 'gu']
        for ii in range( len(phi_bulk) ):
            if (phi_bulk[ii] == 'sigma'):
                phi_bulk[ii] = self.sigma
            elif (phi_bulk[ii] == 'gu') and (phi_bulk[ii] not in pl):
                phi_bulk[ii] = 1.0
            elif (phi_bulk[ii] not in pl):
                phi_bulk[ii] = 0.0
        return phi_bulk
    
    def psi_bulk(self):
        pl = self.parameters()
        psi_bulk = ['w', 'sd_w', 'tnd', 'sd_tnd', 'sd_mu']
        for ii in range( len(psi_bulk) ):
            if (psi_bulk[ii] not in pl):
                psi_bulk[ii] = 0.0
        return psi_bulk

    def test_inputs(self):

        # check if sigma was input properly
        if (isinstance(self.sigma, (float,int)) == False):
            raise RuntimeError("model input sigma must equal a number more than 0.")
        if (self.sigma <= 0.0):
            raise RuntimeError("model input sigma must equal a number more than 0.")

        # check if non_decision was input properly
        non_decision_list = ['constant', 'uniform', 'normal']
        if (self.non_decision not in non_decision_list):
            raise RuntimeError("non_decision must equal 'constant', 'uniform', or 'normal'.")

        # check if start was input properly
        start_list = ['constant', 'uniform', 'normal']
        if (self.start not in start_list):
            raise RuntimeError("start must equal 'constant', 'uniform', or 'normal'.")

        # check if contamination was input properly
        if (self.contamination != 'none') and (self.contamination != 'uniform'):
            raise RuntimeError("model input contamination must equal 'none' or 'uniform'.")

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
Class object for the fullDDM model.

"""

class fullDDM:
    
    # initialize class values
    def __init__(self, sigma = 1.0, non_decision = 'normal', start = 'uniform', drift = 'normal', contamination = 'none'):
        self.sigma = sigma
        self.non_decision = non_decision
        self.start = start
        self.drift = drift
        self.contamination = contamination
        
    # determine what parameters are associated with model
    def parameters(self):
        pl = ['tnd', 'w', 'mu', 'b']
        if (self.non_decision == 'uniform') or (self.non_decision == 'normal'):
            pl.append('sd_tnd')
        if (self.start == 'uniform') or (self.start == 'normal'):
            pl.append('sd_w')
        if (self.drift == 'uniform') or (self.drift == 'normal'):
            pl.append('sd_mu')
        if (self.contamination == 'uniform'):
            pl.extend(['g', 'gl', 'gu'])
        return pl
    
    def phi_bulk(self):
        pl = self.parameters()
        phi_bulk = ['mu', 'sigma', 'b', 'g', 'gl', 'gu']
        for ii in range( len(phi_bulk) ):
            if (phi_bulk[ii] == 'sigma'):
                phi_bulk[ii] = self.sigma
            elif (phi_bulk[ii] == 'gu') and (phi_bulk[ii] not in pl):
                phi_bulk[ii] = 1.0
            elif (phi_bulk[ii] not in pl):
                phi_bulk[ii] = 0.0
        return phi_bulk
    
    def psi_bulk(self):
        pl = self.parameters()
        psi_bulk = ['w', 'sd_w', 'tnd', 'sd_tnd', 'sd_mu']
        for ii in range( len(psi_bulk) ):
            if (psi_bulk[ii] not in pl):
                psi_bulk[ii] = 0.0
        return psi_bulk

    def test_inputs(self):

        # check if sigma was input properly
        if (isinstance(self.sigma, (float,int)) == False):
            raise RuntimeError("model input sigma must equal a number more than 0.")
        if (self.sigma <= 0.0):
            raise RuntimeError("model input sigma must equal a number more than 0.")

        # check if non_decision was input properly
        non_decision_list = ['constant', 'uniform', 'normal']
        if (self.non_decision not in non_decision_list):
            raise RuntimeError("non_decision must equal 'constant', 'uniform', or 'normal'.")

        # check if start was input properly
        start_list = ['constant', 'uniform', 'normal']
        if (self.start not in start_list):
            raise RuntimeError("start must equal 'constant', 'uniform', or 'normal'.")

        # check if drift was input properly
        drift_list = ['constant', 'uniform', 'normal']
        if (self.drift not in drift_list):
            raise RuntimeError("drift must equal 'constant', 'uniform', or 'normal'.")

        # check if contamination was input properly
        if (self.contamination != 'none') and (self.contamination != 'uniform'):
            raise RuntimeError("model input contamination must equal 'none' or 'uniform'.")

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
Class object for the leakage model set.

"""

class leakage:
    
    # initialize class values
    def __init__(self, sigma = 1.0, non_decision = 'constant', start = 'constant', contamination = 'none'):
        self.sigma = sigma
        self.non_decision = non_decision
        self.start = start
        self.contamination = contamination
        
    # determine what parameters are associated with model
    def parameters(self):
        pl = ['tnd', 'w', 'mu', 'l', 'b']
        if (self.non_decision == 'uniform') or (self.non_decision == 'normal'):
            pl.append('sd_tnd')
        if (self.start == 'uniform') or (self.start == 'normal'):
            pl.append('sd_w')
        if (self.contamination == 'uniform'):
            pl.extend(['g', 'gl', 'gu'])
        return pl
    
    def phi_bulk(self):
        pl = self.parameters()
        phi_bulk = ['mu', 'l', 'sigma', 'b', 'g', 'gl', 'gu']
        for ii in range( len(phi_bulk) ):
            if (phi_bulk[ii] == 'sigma'):
                phi_bulk[ii] = self.sigma
            elif (phi_bulk[ii] == 'gu') and (phi_bulk[ii] not in pl):
                phi_bulk[ii] = 1.0
            elif (phi_bulk[ii] not in pl):
                phi_bulk[ii] = 0.0
        return phi_bulk
    
    def psi_bulk(self):
        pl = self.parameters()
        psi_bulk = ['w', 'sd_w', 'tnd', 'sd_tnd', 'sd_mu']
        for ii in range( len(psi_bulk) ):
            if (psi_bulk[ii] not in pl):
                psi_bulk[ii] = 0.0
        return psi_bulk

    def test_inputs(self):
        # check if sigma was input properly
        if (isinstance(self.sigma, (float,int)) == False):
            raise RuntimeError("model input sigma must equal a number more than 0.")
        if (self.sigma <= 0.0):
            raise RuntimeError("model input sigma must equal a number more than 0.")

        # check if non_decision was input properly
        non_decision_list = ['constant', 'uniform', 'normal']
        if (self.non_decision not in non_decision_list):
            raise RuntimeError("non_decision must equal 'constant', 'uniform', or 'normal'.")

        # check if start was input properly
        start_list = ['constant', 'uniform', 'normal']
        if (self.start not in start_list):
            raise RuntimeError("start must equal 'constant', 'uniform', or 'normal'.")

        # check if contamination was input properly
        if (self.contamination != 'none') and (self.contamination != 'uniform'):
            raise RuntimeError("model input contamination must equal 'none' or 'uniform'.")
    
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
Class object for the changing_thresholds model set.

"""

class changing_thresholds:
    
    # initialize class values
    def __init__(self, thresholds = 'weibull', sigma = 1.0, non_decision = 'constant', start = 'constant', drift = 'constant', contamination = 'none'):
        self.sigma = sigma
        self.non_decision = non_decision
        self.start = start
        self.drift = drift
        self.thresholds = thresholds
        self.contamination = contamination
        
    # determine what parameters are associated with model
    def parameters(self):
        pl = ['tnd', 'w', 'mu', 'b']
        if (self.thresholds == 'linear'):
            pl.append('m')
        if (self.thresholds == 'exponential'):
            pl.append('tau')
        if (self.thresholds == 'weibull'):
            pl.append('lamb')
            pl.append('kappa')
            pl.append('c')
        if (self.non_decision == 'uniform') or (self.non_decision == 'normal'):
            pl.append('sd_tnd')
        if (self.start == 'uniform') or (self.start == 'normal'):
            pl.append('sd_w')
        if (self.drift == 'uniform') or (self.drift == 'normal'):
            pl.append('sd_mu')
        if (self.contamination == 'uniform'):
            pl.extend(['g', 'gl', 'gu'])
        return pl
    
    def phi_bulk(self):
        pl = self.parameters()
        phi_bulk = ['mu', 'sigma', 'b', 'm', 'lamb', 'tau', 'kappa', 'c', 'g', 'gl', 'gu']
        for ii in range( len(phi_bulk) ):
            if (phi_bulk[ii] == 'sigma'):
                phi_bulk[ii] = self.sigma
            elif (phi_bulk[ii] == 'gu') and (phi_bulk[ii] not in pl):
                phi_bulk[ii] = 1.0
            elif (phi_bulk[ii] not in pl):
                phi_bulk[ii] = 0.0
        if (self.thresholds == 'exponential'):
            phi_bulk[7] = -1.0
        elif (self.thresholds == 'linear'):
            phi_bulk[7] = 1.0
        return phi_bulk
    
    def psi_bulk(self):
        pl = self.parameters()
        psi_bulk = ['w', 'sd_w', 'tnd', 'sd_tnd', 'sd_mu']
        for ii in range( len(psi_bulk) ):
            if (psi_bulk[ii] not in pl):
                psi_bulk[ii] = 0.0
        return psi_bulk

    def test_inputs(self):
        # check if sigma was input properly
        if (isinstance(self.sigma, (float,int)) == False):
            raise RuntimeError("model input sigma must equal a number more than 0.")
        if (self.sigma <= 0.0):
            raise RuntimeError("model input sigma must equal a number more than 0.")

        # check if thresholds was input properly
        thresholds_list = ['linear', 'exponential', 'weibull']
        if (self.thresholds not in thresholds_list):
            raise RuntimeError("thresholds must equal 'linear', 'exponential', or 'weibull'.")

        # check if non_decision was input properly
        non_decision_list = ['constant', 'uniform', 'normal']
        if (self.non_decision not in non_decision_list):
            raise RuntimeError("non_decision must equal 'constant', 'uniform', or 'normal'.")

        # check if start was input properly
        start_list = ['constant', 'uniform', 'normal']
        if (self.start not in start_list):
            raise RuntimeError("start must equal 'constant', 'uniform', or 'normal'.")

        # check if contamination was input properly
        if (self.contamination != 'none') and (self.contamination != 'uniform'):
            raise RuntimeError("model input contamination must equal 'none' or 'uniform'.")

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
Class object for the UGM model.

"""
class UGM:
    
    # initialize class values
    def __init__(self, sigma = 1.0, non_decision = 'constant', start = 'constant', contamination = 'none'):
        self.sigma = sigma
        self.non_decision = non_decision
        self.start = start
        self.contamination = contamination
        
    # determine what parameters are associated with model
    def parameters(self):
        pl = ['tnd', 'w', 'mu', 'l', 'k', 'b']
        if (self.non_decision == 'uniform') or (self.non_decision == 'normal'):
            pl.append('sd_tnd')
        if (self.start == 'uniform') or (self.start == 'normal'):
            pl.append('sd_w')
        if (self.contamination == 'uniform'):
            pl.extend(['g', 'gl', 'gu'])
        return pl
    
    def phi_bulk(self):
        pl = self.parameters()
        phi_bulk = ['mu', 'l', 'k', 'sigma', 'b', 'g', 'gl', 'gu']
        for ii in range( len(phi_bulk) ):
            if (phi_bulk[ii] == 'sigma'):
                phi_bulk[ii] = self.sigma
            elif (phi_bulk[ii] == 'gu') and (phi_bulk[ii] not in pl):
                phi_bulk[ii] = 1.0
            elif (phi_bulk[ii] not in pl):
                phi_bulk[ii] = 0.0
        return phi_bulk
    
    def psi_bulk(self):
        pl = self.parameters()
        psi_bulk = ['w', 'sd_w', 'tnd', 'sd_tnd', 'sd_mu']
        for ii in range( len(psi_bulk) ):
            if (psi_bulk[ii] not in pl):
                psi_bulk[ii] = 0.0
        return psi_bulk

    def test_inputs(self):
        # check if sigma was input properly
        if (isinstance(self.sigma, (float,int)) == False):
            raise RuntimeError("model input sigma must equal a number more than 0.")
        if (self.sigma <= 0.0):
            raise RuntimeError("model input sigma must equal a number more than 0.")

        # check if non_decision was input properly
        non_decision_list = ['constant', 'uniform', 'normal']
        if (self.non_decision not in non_decision_list):
            raise RuntimeError("non_decision must equal 'constant', 'uniform', or 'normal'.")

        # check if start was input properly
        start_list = ['constant', 'uniform', 'normal']
        if (self.start not in start_list):
            raise RuntimeError("start must equal 'constant', 'uniform', or 'normal'.")

        # check if contamination was input properly
        if (self.contamination != 'none') and (self.contamination != 'uniform'):
            raise RuntimeError("model input contamination must equal 'none' or 'uniform'.")

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
Class object for the UGM_flip model (i.e. UGM where drift rate changes 
from positive to negative partway through the trial).

"""

class UGM_flip:
    
    # initialize class values
    def __init__(self, sigma = 1.0, non_decision = 'constant', start = 'constant', contamination = 'none'):
        self.sigma = sigma
        self.non_decision = non_decision
        self.start = start
        self.contamination = contamination
        
    # determine what parameters are associated with model
    def parameters(self):
        pl = ['tnd', 'w', 'mu', 'l', 'k', 't_flip', 'b']
        if (self.non_decision == 'uniform') or (self.non_decision == 'normal'):
            pl.append('sd_tnd')
        if (self.start == 'uniform') or (self.start == 'normal'):
            pl.append('sd_w')
        if (self.contamination == 'uniform'):
            pl.extend(['g', 'gl', 'gu'])
        return pl
    
    def phi_bulk(self):
        pl = self.parameters()
        phi_bulk = ['mu', 'l', 'k', 't_flip', 'sigma', 'b', 'g', 'gl', 'gu']
        for ii in range( len(phi_bulk) ):
            if (phi_bulk[ii] == 'sigma'):
                phi_bulk[ii] = self.sigma
            elif (phi_bulk[ii] == 'gu') and (phi_bulk[ii] not in pl):
                phi_bulk[ii] = 1.0
            elif (phi_bulk[ii] not in pl):
                phi_bulk[ii] = 0.0
        return phi_bulk
    
    def psi_bulk(self):
        pl = self.parameters()
        psi_bulk = ['w', 'sd_w', 'tnd', 'sd_tnd', 'sd_mu']
        for ii in range( len(psi_bulk) ):
            if (psi_bulk[ii] not in pl):
                psi_bulk[ii] = 0.0
        return psi_bulk

    def test_inputs(self):
        # check if sigma was input properly
        if (isinstance(self.sigma, (float,int)) == False):
            raise RuntimeError("model input sigma must equal a number more than 0.")
        if (self.sigma <= 0.0):
            raise RuntimeError("model input sigma must equal a number more than 0.")

        # check if non_decision was input properly
        non_decision_list = ['constant', 'uniform', 'normal']
        if (self.non_decision not in non_decision_list):
            raise RuntimeError("non_decision must equal 'constant', 'uniform', or 'normal'.")

        # check if start was input properly
        start_list = ['constant', 'uniform', 'normal']
        if (self.start not in start_list):
            raise RuntimeError("start must equal 'constant', 'uniform', or 'normal'.")

        # check if contamination was input properly
        if (self.contamination != 'none') and (self.contamination != 'uniform'):
            raise RuntimeError("model input contamination must equal 'none' or 'uniform'.")

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

"""
Class object for the DMC model.

"""
class DMC:
    
    # initialize class values
    def __init__(self, sigma = 1.0, non_decision = 'constant', start = 'constant', contamination = 'none'):
        self.sigma = sigma
        self.non_decision = non_decision
        self.start = start
        self.contamination = contamination
        
    # determine what parameters are associated with model
    def parameters(self):
        pl = ['tnd', 'w', 'A', 'tau', 'a', 'mu_c', 'b']
        if (self.non_decision == 'uniform') or (self.non_decision == 'normal'):
            pl.append('sd_tnd')
        if (self.start == 'uniform') or (self.start == 'normal'):
            pl.append('sd_w')
        if (self.contamination == 'uniform'):
            pl.extend(['g', 'gl', 'gu'])
        return pl
    
    def phi_bulk(self):
        pl = self.parameters()
        phi_bulk = ['A', 'tau', 'a', 'mu_c', 'sigma', 'b', 'g', 'gl', 'gu']
        for ii in range( len(phi_bulk) ):
            if (phi_bulk[ii] == 'sigma'):
                phi_bulk[ii] = self.sigma
            elif (phi_bulk[ii] == 'gu') and (phi_bulk[ii] not in pl):
                phi_bulk[ii] = 1.0
            elif (phi_bulk[ii] not in pl):
                phi_bulk[ii] = 0.0
        return phi_bulk
    
    def psi_bulk(self):
        pl = self.parameters()
        psi_bulk = ['w', 'sd_w', 'tnd', 'sd_tnd', 'sd_mu']
        for ii in range( len(psi_bulk) ):
            if (psi_bulk[ii] not in pl):
                psi_bulk[ii] = 0.0
        return psi_bulk

    def test_inputs(self):
        # check if sigma was input properly
        if (isinstance(self.sigma, (float,int)) == False):
            raise RuntimeError("model input sigma must equal a number more than 0.")
        if (self.sigma <= 0.0):
            raise RuntimeError("model input sigma must equal a number more than 0.")
        
        # check if non_decision was input properly
        non_decision_list = ['constant', 'uniform', 'normal']
        if (self.non_decision not in non_decision_list):
            raise RuntimeError("non_decision must equal 'constant', 'uniform', or 'normal'.")

        # check if start was input properly
        start_list = ['constant', 'uniform', 'normal']
        if (self.start not in start_list):
            raise RuntimeError("start must equal 'constant', 'uniform', or 'normal'.")

        # check if contamination was input properly
        if (self.contamination != 'none') and (self.contamination != 'uniform'):
            raise RuntimeError("model input contamination must equal 'none' or 'uniform'.")
