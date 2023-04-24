"""
Definition of Jenkins Element using JAX for automatic derivatives
"""

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


class JenkinsForce(NonlinearForce):
    """
    Jenkins Slider Element Nonlinearity with JAX for automatic differentiation

    The AFT formulation assumes that the spring starts at 0 force
    at zero displacement.    
    """

    def __init__(self, Q, T, kt, Fs, u0=0):
        """
        Initialize a nonlinear force model

        Parameters
        ----------
        Q : Transformation matrix from system DOFs (n) to nonlinear DOFs (Nnl), 
            Nnl x n
        T : Transformation matrix from local nonlinear forces to global 
            nonlinear forces, n x Nnl
        kt : Tangential stiffness, tested for scalar, may work for vector of size 
                Nnl
        Fs : slip force, tested for scalar, may work for vector of size 
                Nnl
        u0 : initialization value for the slider. If u0 = None, then 
                the zeroth harmonic is used to initialize the slider position.
                For JAX, u0 must be an appropriately sized np array and not a 
                scalar quantity
                Highly recommended not to use u0=None because may result in
                non-unique solutions.

        """
        self.Q = Q
        self.T = T
        self.kt = kt
        self.Fs = Fs
        
        self.u0 = u0
        
    def aft(self, U, w, h, Nt=128, tol=1e-7):
        """
        
        Tolerances are ignored since Jenkins converges to steady-state with 
        two cycles of the hysteresis loop, so that is done by default. 

        Parameters
        ----------
        U : Global DOFs harmonic coefficients, all 0th, then 1st cos, etc, 
            shape: (Nhc*nd,)
        w : Frequency, scalar
        h : List of harmonics that are considered, zeroth must be first
        Nt : Number of time points to evaluate at. 
             The default is 128.
        max_repeats : number of hysteresis loops to calculate to reach steady
                        state. Maximum value allowed. 
                        The default is 2
        atol : Convergence criteria, absolute for steady-state hysteresis loops
              The default is 1e-10.
        rtol : Convergence criteria, relative for steady-state hysteresis loops
               The default is 1e-10.

        Returns
        -------
        None.

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
    
    # Do a loop function for each update at index i
    loop_fun = lambda i,f : _local_jenkins_loop_body(i, f, unlt, kt, Fs)
    
    
    ########################################
    #### Start slider in desired position
    
    # if u0 comes from the zeroth harmonic, pull it from the jax traced 
    # array rather than the separate input value, which is constant as far as 
    # gradients are concerned.
    u0 = jnp.where(u0h0, Ulocal[0, 0:1], u0)
    
    # The first evaluation is based on the last entry of ft and therefore 
    # initialize the last entry of ft based on a linear spring
    # slip limit does not need to be applied since this just needs to get stuck
    # regime correct for the first step to be through zero. 
    ft = ft.at[-1, :].set(kt*(unlt[-1, :] - u0))
    
    # Conduct exactly 2 repeats of the hysteresis loop to be converged to 
    # steady-state
    for out_ind in range(2):
        
        # This construct must be used otherwise compiling tries writing out
        # all Nt steps of the loop updates and is excessively slow
        ft = jax.lax.fori_loop(0, Nt, loop_fun, ft)
    
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


