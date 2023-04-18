# Functions for vectorizing a Iwan (4-par) model AFT calculation

import numpy as np
# from .nonlinear_force import HystereticForce

# Harmonic Functions for AFT
from .. import harmonic_utils as hutils

from .iwan4_element import Iwan4Force 


class VectorIwan4(Iwan4Force):
    """
    4-Par Iwan Element Nonlinearity, Vectorized Force Calculations
    
    See documentation for Iwan4Force for details. This class reorganizes 
    computation to increase parallelization (vectorization) by making use of 
    the fact that some the inclusion of some intermediate history points does
    not change the result at the current instant.
    """

    def local_force_history_crit(self, unlt, unltdot, h, cst, unlth0, \
                                 max_repeats=2, atol=1e-10, rtol=1e-10):
        """
        For evaluating local force history, used by AFT. Always does at least 
        two loops to verify convergence.
        
        Convergence criteria is atol or rtol passes. To require a choice, pass 
        in -1 for the other
        
        This function is slightly modified from the normal local_force_history
        function from HystereticForce in that it passes out slider states in
        addition to the forces and derivatives at the time points of interest
        
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
        its = 0
        
        rcheck = 0
        acheck = 0
        
        # Initialize Memory - Assumption on shape is reasonable for mechanical 
        # systems, but may not be perfect.
        Nt,Ndnl = unlt.shape
        Nhc = hutils.Nhc(h)
        
        ft = np.zeros_like(unlt)
        dfduh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
        dfdudh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
        
        # Slider state storage
        fsliders = np.zeros((Nt, self.Nsliders+1))
        dfslidersduh = np.zeros((Nt,self.Nsliders+1,hutils.Nhc(h)))
        
        # Only initialize before the loop. History is propogated through 
        # repeated loops over the period
        self.init_history(unlth0, h)
        fp = self.fp
        
        while( (its == 0) or (acheck > atol and rcheck > rtol and its < max_repeats) ):
            
            # Time Loop                
            for ti in range(Nt):
                # Update this to immediately save into array without tmps
                fttmp,dfdutmp,dfdudtmp = \
                    self.instant_force_harmonic(unlt[ti, :], unltdot[ti, :], \
                                                h, cst[ti, :], update_prev=True)
                
                ft[ti,:] = fttmp
                dfduh[ti,:,:,:] = dfdutmp
                dfdudh[ti,:,:,:] = dfdudtmp
                
                fsliders[ti, :] = self.fpsliders
                dfslidersduh[ti, :, :] = self.dfpslidersduh

                
            its = its + 1
            
            acheck = np.abs(ft[ti, :] - fp)
            rcheck = np.abs(acheck / (ft[ti, :]+np.finfo(float).eps) )
            
            fp = ft[ti, :]
        
        return ft, dfduh, dfdudh, fsliders, dfslidersduh
    
        
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
        # systems, but may not be perfect. Use Q and T to get more exact shapes.
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
        
        
        ft_crit, dfduh_crit, dfdudh_crit, fsliders_crit, dfslidersduh_crit\
                        = self.local_force_history_crit(unlt_crit, unltdot_crit, h, \
                                                   cst_crit, unlth0, max_repeats=2, \
                                                   atol=1e-10, rtol=1e-10)
        
        # If one rewrote the Iwan4Force class, it may be faster to recalculate
        # this here instead of doing the weight integration in the loop. However,
        # it is not worth that rewrite now.
        ft[np.logical_not(vector_set).reshape(-1), :] = ft_crit
        
        dfduh[np.logical_not(vector_set).reshape(-1), :] = dfduh_crit
        
        # No velocity dependence for Iwan
        # dfdudh[np.logical_not(vector_set).reshape(-1), :] = dfdudh_crit
        
        crit_inds = np.asarray(np.logical_not(vector_set)).nonzero()[0]
        crit_inds = np.append(crit_inds, crit_inds[0]) # Wrap around without logic in for loop below
        
        # EVERYTHING BELOW HERE IS TRIVIALLY PARALLELIZABLE
        
        # Loop over the set of all crit points and evaluate their subsequent history points.
        # Alternative try doing something fancy with creating an index array, 
        # but that's just as likely to either add a bunch of memory or mess up vectorization.
        for i in range( len(crit_inds)-1 ):
            start = crit_inds[i]+1
            stop  = crit_inds[i+1] # want to end on the previous index (i.e., this minus 1)
            
            stop = stop + Nt*(stop == 0) # Wrap at end
            
            if(stop > start): # Skip case of stop == start
                
                # Apply standard Jenkins from the critical point to the current point 
                # for the full vector_set at once.
        
                # Previous States for readability
                up = unlt[start-1, :]
                fpsliders = fsliders_crit[i, :]
                
                dupduh = cst[start-1, :]
                dfpslidersduh = dfslidersduh_crit[i, :, :]
                
                # Stuck Force
                # Ntimes x Nsliders where Ntimes = stop-start-1
                fnlsliders = (unlt[start:stop, :] - up) + fpsliders.reshape(1,-1)
                
                # Mask of stuck sliders == places with unit derivative
                dfnlsliders_dunl = np.less_equal(np.abs(fnlsliders), \
                                                 self.phisliders.reshape(1,-1))

                # Slipped Force
                # This line does some unnecessary multiplication for the False 
                # case
                fnlsliders = np.where(dfnlsliders_dunl, fnlsliders, \
                                      self.phisliders.reshape(1,-1)*np.sign(fnlsliders))
                
                ft[start:stop, :] = (fnlsliders @ self.sliderweights).reshape(-1,1)
                
                # Derivative of Force Calculation
                delta_cst = cst[start:stop, :] - dupduh

                # Second line is dfnlsliders_dfslidersp*...
                dfnlsliders_duh = np.einsum('ti,tj->tij', dfnlsliders_dunl, \
                                            delta_cst) \
                        + np.einsum('ti,ij->tij', dfnlsliders_dunl, dfpslidersduh)
                
                dfduh[start:stop, 0, 0, :] = np.einsum('tij,i->tj', \
                                                       dfnlsliders_duh, \
                                                       self.sliderweights) 
                
        
        return ft, dfduh, dfdudh
        
        
        
        
        
        
    
    