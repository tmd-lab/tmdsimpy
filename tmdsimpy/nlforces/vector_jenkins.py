# Functions for vectorizing a Jenkins model AFT calculation

import numpy as np

# Harmonic Functions for AFT
from ..utils import harmonic as hutils

from .jenkins_element import JenkinsForce 


class VectorJenkins(JenkinsForce):
    """
    Jenkins slider element nonlinearity with vectorized force calculations.

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

    See Also
    --------
    JenkinsForce :
        Standard implementation of the Jenkins element, generally a slower
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
    `JenkinsForce`.

    """
    
        
    def local_force_history(self, unlt, unltdot, h, cst, unlth0, max_repeats=2, \
                            atol=1e-10, rtol=1e-10):
        """
        Evaluate the local forces for steady-state harmonic motion used in AFT.

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
            Derivative of forces with respect to velocities harmonic
            coefficients.
            First two indices correspond to `ft`. Third index corresponds to
            which local nonlinear displacement.
            Fourth index corresponds to which of the `Nhc` harmonic
            components.

        Notes
        -----

        Convergence criteria is atol or rtol passes. To require a choice, pass 
        in -1 for the other. Convergence should be exact within two cycles
        since this is a slider based model.

        This function is reimplemented from `JenkinsForce` with the more
        efficient vectorized algorithm.

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
        
        
        
        
        
        
    
    