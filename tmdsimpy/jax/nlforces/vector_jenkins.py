# Standard imports
import numpy as np

# JAX imports
import jax
import jax.numpy as jnp

# Decoractions for Partial compilation
from functools import partial

# Imports of Custom Functions and Classes
from ... import harmonic_utils as hutils
from ...jax import harmonic_utils as jhutils # Jax version of harmonic utils
from ...nlforces.nonlinear_force import NonlinearForce


class VectorJenkins(NonlinearForce):
    """
    Demonstration only class of JAX with vector Jenkins algorithm.
    
    Class not intended for use in simulations, use an alternative
    implementation.
    
    Parameters
    ----------
    Q : (1, N) numpy.ndarray
        Matrix tranform from the `N` degrees of freedom (DOFs) of the system 
        to the `Nnl` local nonlinear DOFs.
    T : (N, 1) numpy.ndarray
        Matrix tranform from the local `Nnl` forces to the `N` global DOFs.
    kt : float
        Tangential stiffness.
    Fs : float
        Slip force.
    u0 : float or None, optional
        Initialization value for the slider for AFT. 
        If `u0 = None`, then the zeroth harmonic is used to initialize 
        the slider position.
        Highly recommended not to use `u0 = None` because it may result in
        non-unique solutions.
        The default is 0.

    See Also
    --------
    tmdsimpy.nlforces.VectorJenkins :
        non-JAX implementation of a Jenkins element that has full functionality
        and can be faster.
    tmdsimpy.jax.nlforces.JenkinsForce : 
        JAX implementation of Jenkins element without vectorization.

    Notes
    -----

    This implementation tries to use a more efficient Jenkins implementation
    for AFT that requires fewer operations. However, given JAX constraints
    for just-in-time compliation (JIT), this approach is not good here.

    The Jenkins element consists of a linear spring of stiffness `kt` that
    stretches until a slip force `Fs` is reached.
    Once the slip force is reached, an anchor point for the other side of the
    spring moves to maintain the slip force until reversal.
    
    The `force` method is not implemented here since this is just an example
    of automatic differentiation for AFT.
    
    """

    def __init__(self, Q, T, kt, Fs, u0=0):
        
        self.Q = Q
        self.T = T
        self.kt = kt
        self.Fs = Fs
        
        self.u0 = u0
        
    def aft(self, U, w, h, Nt=128, tol=1e-7):
        """
        Implementation of the alternating frequency-time (AFT) method to
        extract harmonic nonlinear force coefficients.
        
        Parameters
        ----------
        U : (N*Nhc,) numpy.ndarray
            Displacement harmonic DOFs (global)
        w : float
            Frequency in rad/s. Needed in case there is velocity dependency.
        h : numpy.ndarray, sorted
            List of harmonics. The list corresponds to `Nhc` harmonic 
            components.
        Nt : int power of 2, optional
            Number of time steps used in evaluation. 
            The default is 128.
        tol : float, optional
            This argument is ignored, and is included for compatability of 
            interface. 
            The default is 1e-7.
        
        Returns
        -------
        Fnl : (N*Nhc,) numpy.ndarray
            Nonlinear hamonic force coefficients
        dFnldU : (N*Nhc,N*Nhc) numpy.ndarray
            Jacobian of `Fnl` with respect to `U`
        dFnldw : (N*Nhc,) numpy.ndarray
            Jacobian of `Fnl` with respect to `w`
        
        Notes
        -----
        The tolerance `tol` is ignored because the Jenkins element converges
        to steady-state with two cycles of the hysteresis loop. Two cycles of
        the nonlinear forces are calculated automatically without the option to
        change this setting.

        """
        
        #########################
        # Memory Initialization 
        
        Fnl = np.zeros_like(U)
        dFnldU = np.zeros((U.shape[0], U.shape[0]))
        dFnldw = np.zeros_like(U)
        
        
        #########################
        # Transform to Local Coordinates
        
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components        
        Ulocal = (self.Q @ np.reshape(U, (self.Q.shape[1], Nhc), 'F')).T
        
        # Number of Nonlinear DOFs
        Ndnl = self.Q.shape[0]
        
        
        #########################
        # Determine Slider Starting Position
        
        if self.u0 is None:
            # Initialize based on the zeroth harmonic.
            u0 = Ulocal[0, :]
            u0h0 = True
        else:
            u0 = self.u0
            u0h0 = False
        
        
        #########################
        # Conduct AFT in Local Coordinates with JAX
        Uwlocal = np.hstack((np.reshape(Ulocal.T, (Ndnl*Nhc,), 'F'), w))
        
        pars = np.array([self.kt, self.Fs])
        
        # # If no Grad is needed use:
        # Flocal = _local_aft_jenkins(Uwlocal, pars, u0, tuple(h), Nt, u0h0)[0]
        
        # Case with gradient and local force
        dFdUwlocal, Flocal = _local_aft_jenkins_grad(Uwlocal, pars, u0, \
                                                     tuple(h), Nt, u0h0)
        
        
        #########################
        # Convert AFT to Global Coordinates
        
        # Reshape Flocal
        Flocal = jnp.reshape(Flocal, (Ndnl, Nhc), 'F')
                
        # Global coordinates        
        Fnl = np.reshape(self.T @ Flocal, (U.shape[0],), 'F')
        dFnldU = np.kron(np.eye(Nhc), self.T) @ dFdUwlocal[:, :-1] \
                                                @ np.kron(np.eye(Nhc), self.Q)
        
        dFnldw = np.reshape(self.T @ \
                            np.reshape(dFdUwlocal[:, -1], (Ndnl, Nhc)), \
                            (U.shape[0],), 'F')
        
        return Fnl, dFnldU, dFnldw
        

def _local_jenkins_loop_body(ind, ft, unlt, kt, Fs):
    """
    Function for calculating a single force update for Jenkins. This is
    constructed as a loop body function for JAX and thus evaluates for a 
    specific index given the full arrays for the force and displacement 
    time series.

    Parameters
    ----------
    ind : Index that is being updated for this loop step.
    ft : Array of force values for all time (Nt,)
    unlt : Displacements for Jenkins for all times (Nt,)
    kt : Tangential stiffness parameter
    Fs : Slip Force parameter

    Returns
    -------
    ft : Force array with the entry at ind updated for Jenkins nonlinear force

    """
    
    fcurr = jnp.minimum(kt*(unlt[ind, :]-unlt[ind-1, :]) + ft[ind-1, :], Fs)
    
    ft = ft.at[ind, :].set(jnp.maximum(fcurr, -Fs))
    
    return ft
 
def _local_jenkins_loop_body_crit(ind, ft, unlt, kt, Fs, crit):
    """
    Function for calculating a single force update for Jenkins. This is
    constructed as a loop body function for JAX and thus evaluates for a 
    specific index given the full arrays for the force and displacement 
    time series.

    Parameters
    ----------
    ind : Index that is being updated for this loop step.
    ft : Array of force values for all time (Nt,)
    unlt : Displacements for Jenkins for all times (Nt,)
    kt : Tangential stiffness parameter
    Fs : Slip Force parameter
    crit : index of the critical point to always use as history variable

    Returns
    -------
    ft : Force array with the entry at ind updated for Jenkins nonlinear force

    """
    
    fcurr = jnp.minimum(kt*(unlt[ind, :]-unlt[crit, :]) + ft[crit, :], Fs)
    
    ft = ft.at[ind, :].set(jnp.maximum(fcurr, -Fs))
    
    return ft

@partial(jax.jit, static_argnums=(3,4,5)) 
def _local_aft_jenkins(Uwlocal, pars, u0, htuple, Nt, u0h0):
    """
    Conducts AFT in a functional form that can be used with JAX and JIT

    NOTES:
        1. Jenkins converges to steady-state in at most 2 repeats of the 
        hysteresis loop. This code always does exactly two loops. 
        Different logic for while loops could be implemented in the future.
        Other models may want to use better logic or allow for additional 
        repeated loops. 
        2. Vectorized Version assumes and requires Ndnl = 1 NL DOF

    Parameters
    ----------
    Uwlocal : jax.numpy array with displacements at local nonlinear DOFs 
                followed by frequency. Each harmonic is listed in full
                then the next harmonic ect. Size (Nhc*Ndnl + 1,)
    pars : jax.numpy array with parameters [kt, Fs]. Bundled this way in case
            future work is interested in applying autodiff w.r.t. parameters
    u0 : scalar value for the displacement to initialize the slider to
    htuple : tuple containing the list of harmonics. Tuple is used so the 
            argument can be made static. 
    Nt : Number of AFT time steps to be used. 
    u0h0 : set to True if u0 should be taken from harmonic zero instead of from
            the input u0. Cannot set u0 in that case outside function because
            miss gradient pieces
    

    Returns
    -------
    Flocal : Nhc*Ndl array of the harmonic force coefficients locally. 
             Same format as U part of Uwlocal 
    Flocal : Flocal is returned again as aux data so it can be accessed when
                gradient is calculated with JAX

    """
    
    ########################################
    #### Initialization
    
    # Size Calculation
    Nhc = hutils.Nhc(np.array(htuple))
    Ndnl = int(Uwlocal.shape[0] / Nhc)
    
    # Recover pars for convenience
    kt = pars[0]
    Fs = pars[1]
    
    # Uwlocal is arranged as all of harmonic 0, then all of 1c, etc. 
    # For each harmonic it has the DOFs in order. Finally there is frequency.
    # This is a 1d array. 
    #
    # Ulocal is (Nhc x Ndnl) - each column is the harmonic components for a 
    # single nonlinear DOF.
    Ulocal = jnp.reshape(Uwlocal[:-1], (Ndnl, Nhc), 'F').T

    
    ########################################
    #### Displacements
    
    # Nonlinear displacements, velocities in time
    # Nt x Ndnl
    unlt = jhutils.time_series_deriv(Nt, htuple, Ulocal, 0) # Nt x Ndnl
    
    # Do not need velocity for Jenkins, this is how it would be calculated:
    # # Nt x Ndnl
    # unltdot = Uwlocal[-1]*jhutils.time_series_deriv(Nt, htuple, Ulocal, 1) 
    
    # Initialize force time memory
    ft = jnp.zeros_like(unlt)
    
    
    
    ########################################
    #### Critical Points of Velocity Reversals
    
    # Identify reversal points 
    dup = unlt - jnp.roll(unlt, 1, axis=0) # du to current
    dun = jnp.roll(unlt, -1, axis=0) - unlt # du to next
    
    vector_set = jnp.not_equal(jnp.sign(dup), jnp.sign(dun))
    vector_set = vector_set.at[0].set(True) # This makes it much easier to write the loop below and is assumed.
    
    # Maximum number of critical points is 2*(Hmax)
    # Add 1 for ease of implementation by including the first time instant
    Ncrit = htuple[-1]*2+1
    
    _,inds = jax.lax.top_k(vector_set.T, Ncrit)
    
    inds = jnp.sort(inds)[0, :] #reset the transpose back to a row
    
    ########################################
    #### Traditional Loop over Critical Point Subset
    
    
    #### Start Slider in correct position for subset
    
    u0 = jnp.where(u0h0, Ulocal[0, 0:1], u0)
    
    ft = ft.at[inds[-1], :].set(kt*(unlt[inds[-1], :] - u0))
    
    #### Do loop
    # If there are a lot of harmonics, one may want to use a lax loop
    # expect that it should be fine to unroll the loop for the compiler since 
    # it is relatively small
    for outer in range(2):
        for cind in range(Ncrit):
            
            fcurr = jnp.minimum(kt*(unlt[inds[cind], :]-unlt[inds[cind-1], :]) \
                                + ft[inds[cind-1], :], Fs)
            
            ft = ft.at[inds[cind], :].set(jnp.maximum(fcurr, -Fs))
            
    inds = jnp.hstack((inds, np.array([Nt])))
            
    ########################################
    #### Special Loop w/ vectorized for filling in between critical points
    
    # Options here
    
    # 1.
    # Just evaluating the blocks between critical points does not work with JAX
    # because the indices are dynamic and it is a JIT function. 
    
    # 2.
    # Alternatively full matrices of ftprev and a rotated unlt could be 
    # constructed based on the critical indices to do everything in one vector 
    # operation. Such an approach adds computations and memory requirements, 
    # so is not considered.
    
    # 3.
    # Run the same loop as the normal jax jenkins, but only once. 
    # This approach is adopted since it is the simplest and lowest memory.
    
    # Do a loop function for each update at index i
    loop_fun = lambda i,f : _local_jenkins_loop_body(i, f, unlt, kt, Fs)
    
    # save a single computation since know that index 0 was chosen as a critical
    # point for convenience in indexing
    ft = jax.lax.fori_loop(1, Nt, loop_fun, ft)
    
    # 4. 
    # Run a set of individual loops between critical points to fill in the 
    # center. Hopefully LAX will realize they can be vectorized.
    # This is a rough outline of how 4 would be implemented, but fails the test
    # Also, quick timing indicates this isn't really any faster than option 3. 
    
    # for i in range( len(inds)-1 ):
    #     start = inds[i]+1
    #     stop  = inds[i+1] # want to end on the previous index (e.g., this minus 1)
        
    #     crit_loop_fun = lambda i,f : _local_jenkins_loop_body_crit(i, f, unlt, kt, Fs, inds[i])
        
    #     ft = jax.lax.fori_loop(start, stop, crit_loop_fun, ft)
        

    ########################################
    #### Final Conversions
    
    # Convert back into frequency domain
    Flocal = jhutils.get_fourier_coeff(htuple, ft)
    
    # Flatten back to a 1D array
    Flocal = jnp.reshape(Flocal.T, (-1,), 'F')
    
    return Flocal,Flocal
        

@partial(jax.jit, static_argnums=(3,4,5)) 
def _local_aft_jenkins_grad(Uwlocal, pars, u0, htuple, Nt, u0h0):
    """
    Function that computes the gradient of AFT. Using Aux data allows for 
    returning Flocal also from one function call. 

    Parameters
    ----------
    Uwlocal : Displacements and frequency as defined for _local_aft_jenkins
    pars : Parameters as defined for _local_aft_jenkins
    u0 : scalar value for the displacement to initialize the slider to
    htuple : List of harmonics, tuple, use tuple(h) so can be set to static.
    Nt : Number of time steps used in AFT
    u0h0 : set to True if u0 should be taken from harmonic zero instead of from
            the input u0. Cannot set u0 in that case outside function because
            miss gradient pieces

    Returns
    -------
    J : Jacobian of _local_aft_jenkins w.r.t. Uwlocal
    F : Normal output argument (nonlinear force) of _local_aft_jenkins

    """
    
    J,F = jax.jacfwd(_local_aft_jenkins, has_aux=True)(Uwlocal, pars, u0, 
                                                       htuple, Nt, u0h0)
    
    return J,F


