# Functions for vectorizing a Jenkins model AFT calculation

import numpy as np
from scipy.interpolate import interp1d
from nonlinear_force import HystereticForce

# Harmonic Functions for AFT
import sys
sys.path.append('../')
import harmonic_utils as hutils

from jenkins_element import JenkinsForce 


class VectorJenkins(JenkinsForce):
    """
    Jenkins Slider Element Nonlinearity, Vectorized Force Calculations
    
    See documentation for JenkinsForce for details. This class reorganizes 
    computation to increase parallelization (vectorization) by making use of 
    the fact that some the inclusion of some intermediate history points does
    not change the result at the current instant.
    """
    
        
    def local_force_history(self, unlt, unltdot, h, cst, unlth0, max_repeats=2, \
                            atol=1e-10, rtol=1e-10):
        """
        For evaluating local force history, used by AFT. Always does at least 
        two loops to verify convergence.
        
        Convergence criteria is atol or rtol passes. To require a choice, pass 
        in -1 for the other
        
        WARNING: Derivatives with respect to harmonic velocities are not implemented.
        
        Parameters
        ----------
        unlt : Local displacements for Force
        unltdot : Local velocities for Force
        h : list of harmonics
        cst : evaluation of cosine and sine of the harmonics at the times for aft
        unlth0 : 0th harmonic of nonlinear forces for initializing history to start.
        max_repeats: Number of times to repeat the time series to converge the 
             initial state. Two is sufficient for slider models. 
             The default is 2.
        atol: Absolute tolerance on AFT convergence (final state of cycle)
             The default is 1e-10.
        rtol: Relative tolerance on AFT convergence (final state of cycle)
             The default is 1e-10.
        
        Returns
        -------
        ft : Local nonlinear forces
        dfduh : Derivative of forces w.r.t. displacement harmonic coefficients
        dfdudh : Derivative of forces w.r.t. velocities harmonic coefficients

        """
        
        # Initialize output memory - Assumption on shape is reasonable for mechanical 
        # systems, but may not be perfect.
        Nt,Ndnl = unlt.shape
        Nhc = hutils.Nhc(h)
        
        ft = np.zeros_like(unlt)
        dfduh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
        dfdudh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
        
        # Identify reversal points 
        dup = unlt - np.roll(unlt, 1, axis=0) # du to current
        dun = np.roll(unlt, -1, axis=0) - unlt # du to next
        
        vector_set = np.equal(np.sign(dup), np.sign(dun))
        vector_set[0] = False # This makes it much easier to write the loop below and is assumed.
        
        # Critical points that must be evaluated serially
        unlt_crit = unlt[np.logical_not(vector_set).reshape(-1), :]
        unltdot_crit = unltdot[np.logical_not(vector_set).reshape(-1), :]
        cst_crit = cst[np.logical_not(vector_set).reshape(-1), :]
        
        
        ft_crit, dfduh_crit, dfdudh_crit = super().local_force_history(\
                                                   unlt_crit, unltdot_crit, h, \
                                                   cst_crit, unlth0, max_repeats=2, \
                                                   atol=1e-10, rtol=1e-10)
        
        ft[np.logical_not(vector_set).reshape(-1), :] = ft_crit
        dfduh[np.logical_not(vector_set).reshape(-1), :] = dfduh_crit
        
        # No velocity dependence for Jenkins
        # dfdudh[np.logical_not(vector_set).reshape(-1), :] = dfdudh_crit
        
        crit_inds = np.asarray(np.logical_not(vector_set)).nonzero()[0]
        crit_inds = np.append(crit_inds, crit_inds[0]) # Wrap around without logic in for loop below
        
        # EVERYTHING BELOW HERE IS TRIVIALLY PARALLELIZABLE
        
        # Loop over the set of all crit points and evaluate their subsequent history points.
        # Alternative try doing something fancy with creating an index array, 
        # but that's just as likely to either add a bunch of memory or mess up vectorization.
        for i in range( len(crit_inds)-1 ):
            start = crit_inds[i]+1
            stop  = crit_inds[i+1] # want to end on the previous index (e.g., this minus 1)
            
            stop = stop + Nt*(stop == 0) # Wrap at end
            
            if(stop > start):

                ft[start:stop, :] = self.kt*(unlt[start:stop, :]-unlt[start-1, :]) + ft[start-1, :]
                
                mask = np.abs(ft[start:stop, :]) > self.Fs
                
                ft[start:stop, :] = np.where(mask, \
                                             self.Fs*np.sign(ft[start:stop, :]),  # Slipped \
                                             ft[start:stop, :]) # Stuck
                
                # 4D Mask for derivatives
                mask.shape += (1,) * (4 - mask.ndim)
                
                # Probably some bad things memorywise here to get the vectorization working
                delta_cst = cst[start:stop, :] - cst[start-1, :]
                dfduh_prev = dfduh[start-1:start, :, :, :]
                delta_cst = np.reshape(delta_cst, (delta_cst.shape[0], 1, 1, -1), order='F')
                
                dfduh[start:stop, :, :, :] = np.where(mask,\
                                                0.0, # Slipped\
                                                self.kt*(delta_cst) + dfduh_prev, # Stuck\
                                                )
            
        
        # Apply standard Jenkins from the critical point to the current point 
        # for the full vector_set at once.
        
        return ft, dfduh, dfdudh
        
        
        
        
        
        
    
    