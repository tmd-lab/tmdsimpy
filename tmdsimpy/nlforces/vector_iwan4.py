# Functions for vectorizing a Iwan (4-par) model AFT calculation

import numpy as np
# from .nonlinear_force import HystereticForce

# Harmonic Functions for AFT
from ..utils import harmonic as hutils

from .iwan4_element import Iwan4Force 


class VectorIwan4(Iwan4Force):
    """
    4-Parameter Iwan Element Nonlinearity with vectorized force calculations.

    Parameters
    ----------
    Q : (Nnl, N) numpy.ndarray
        Matrix tranform from the `N` degrees of freedom (DOFs) of the system
        to the `Nnl` local nonlinear DOFs.
    T : (N, Nnl) numpy.ndarray
        Matrix tranform from the local `Nnl` forces to the `N` global DOFs.
    kt : float
        Tangential stiffness coefficient.
    Fs : float
        Slip force.
    chi : float
        Controls microslip damping slope. Recommended to have `chi > -1`.
        Smaller values of `chi` may not work.
    beta : float, positive
        Controls discontinuity at beginning of macroslip (zero is smooth).
    Nsliders : int, optional
        Number of discrete sliders for the Iwan element.
        Note that this does not include 1 additional slider for the
        delta function at phimax.
        Default is 100 (commonly used in literature).
    alphasliders : float, optional
        Determines the non-uniform discretization (see [1]_).
        For midpoint rule, using anything other than 1.0 has
        significantly higher error.
        The default is 1.0.

    See Also
    --------
    Iwan4Force :
        Standard implementation of the Iwan element, generally a slower
        implementation than the present class.

    Notes
    -----
    This class exploits the fact that only reversal points need to be
    calculated to reach steady-state.
    After that, all intermediate times can be calculated in parallel
    (vectorized here), to be faster.
    This does not change the results.

    This implementation is only tested for `Nnl == 1`.

    `local_force_history` implementation is the only difference relative to
    `Iwan4Force` for standard functions. This also adds a new function
    to help in calculations of `local_force_history_crit`, but that function
    should not be needed for must public calls.

    Iwan nonlinearity is based on [1]_.

    References
    ----------
    .. [1]
       Segalman, D.J., 2005. A Four-Parameter Iwan Model for Lap-Type
       Joints. J. Appl. Mech 72, 752–760.

    """

    def local_force_history_crit(self, unlt, unltdot, h, cst, unlth0, \
                                 max_repeats=2, atol=1e-10, rtol=1e-10):
        """
        Modified `local_force_history` to pass out slider states as well
        as other returns.

        Parameters
        ----------
        unlt : (Nt,Nnl) numpy.ndarray
            Local displacements, rows are different time instants and
            columns are different displacement DOFs.
        unltdot : (Nt,Nnl) numpy.ndarray
            Local velocities, rows are different time instants and
            columns are different displacement DOFs.
        h : 1D numpy.ndarray, sorted
            List of harmonics used in subsequent analysis. Corresponds
            to `Nhc` harmonic components.
        cst : (Nt,Nhc) numpy.ndarray
            Evaluation of each harmonic component (columns) at a given instant
            in time (row = instant in time). These are without any harmonic
            coefficients, so are just cosine and sine evaluations.
        unlth0 : (Nnl,) numpy.ndarray
            Zeroth harmonic contributions to a time series of displacements.
            This is passed to `init_history_harmonic` to initialize model.
        max_repeats : int, optional
            Number of times to repeat the time series to converge the 
            initial state with `local_force_history`. 
            Two is sufficient for slider models. 
            The default is 2.
        atol : float, optional
            Absolute tolerance on force time series convergence to steady-state
            (final state of cycle).
            The default is 1e-10.
        rtol : float, optional
            Relative tolerance on force time series convergence to steady-state
            (final state of cycle).
            The default is 1e-10.

        Returns
        -------
        ft : (Nt,Nnl) numpy.ndarray
            Local nonlinear forces. First index is time instants, second index
            is which local nonlinear force DOF.
        dfduh : (Nt,Nnl,Nnl,Nhc) numpy.ndarray
            Derivative of forces with respect to displacement harmonic
            coefficients.
            First two indices correspond to `ft`. Third index corresponds to
            which local nonlinear displacement. 
            Fourth index corresponds to which of the `Nhc` harmonic 
            components.
        dfdudh : (Nt,Nnl,Nnl,Nhc) numpy.ndarray
            Derivative of forces with respect to velocities harmonic
            coefficients.
            First two indices correspond to `ft`. Third index corresponds to
            which local nonlinear displacement. 
            Fourth index corresponds to which of the `Nhc` harmonic 
            components.
        fsliders : (Nt, Nsliders+1) numpy.ndarray
            For each instant in time (row), the columns are the force of each
            slider in integrating the Iwan nonlinearity force.
        dfslidersduh : (Nt, Nsliders+1, Nhc) numpy.ndarray
            The derivative of `fsliders` with respect to the harmonic
            coefficients of the displacement `unlt`.

        Notes
        -----

        Function is intended to be called for only a subset of the full times
        of a cycle. These times should just be the velocity reversal points.
        This allows to the calculation of those points more directly
        to improve the efficiency of `local_force_history`.

        Shapes of outputs rely on having `Nnl == 1`.

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
        self.init_history_harmonic(unlth0, h)
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
        Evaluate the local forces for steady-state harmonic motion used in AFT.
        
        Parameters
        ----------
        unlt : (Nt,Nnl) numpy.ndarray
            Local displacements, rows are different time instants and
            columns are different displacement DOFs.
        unldot : (Nt,Nnl) numpy.ndarray
            Local velocities, rows are different time instants and
            columns are different displacement DOFs.
        h : 1D numpy.ndarray, sorted
            List of harmonics used in subsequent analysis. Corresponds
            to `Nhc` harmonic components.
        cst : (Nt,Nhc) numpy.ndarray
            Evaluation of each harmonic component (columns) at a given instant
            in time (row = instant in time). These are without any harmonic
            coefficients, so are just cosine and sine evaluations.
        unlth0 : (Nnl,) numpy.ndarray
            Zeroth harmonic contributions to a time series of displacements.
            This is passed to `init_history_harmonic` to initialize model.
        max_repeats : int, optional
            This is included for compatibility, but is ignored.
            Two repeats of the hysteresis loop are used by default
            to ensure convergence since this is a slider model that
            converges with two repeats.
            The default is 2.
        atol : float, optional
            Absolute tolerance on force time series convergence to steady-state
            (final state of cycle).
            The default is 1e-10.
        rtol : float, optional
            Relative tolerance on force time series convergence to steady-state
            (final state of cycle).
            The default is 1e-10.
            
        Returns
        -------
        ft : (Nt,Nnl) numpy.ndarray
            Local nonlinear forces. First index is time instants, second index
            is which local nonlinear force DOF.
        dfduh : (Nt,Nnl,Nnl,Nhc) numpy.ndarray
            Derivative of forces with respect to displacement harmonic
            coefficients.
            First two indices correspond to `ft`. Third index corresponds to
            which local nonlinear displacement. 
            Fourth index corresponds to which of the `Nhc` harmonic 
            components.
        dfdudh : (Nt,Nnl,Nnl,Nhc) numpy.ndarray
            Derivative of forces with respect to velocities harmonic coefficients.
            First two indices correspond to `ft`. Third index corresponds to
            which local nonlinear displacement. 
            Fourth index corresponds to which of the `Nhc` harmonic 
            components.
        
        Notes
        -----
        
        Convergence criteria is atol or rtol passes. To require a choice, pass 
        in -1 for the other. Convergence should be exact within two cycles
        since this is a slider based model.
        
        This function is reimplemented from `Iwan4Force` with the more
        efficient vectorized algorithm.

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
