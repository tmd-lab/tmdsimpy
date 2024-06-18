"""
Definition of Elastic Dry Friction Element using JAX for automatic derivatives
"""

# Standard imports
import numpy as np
import warnings

# JAX imports
import jax
import jax.numpy as jnp

# Decoractions for Partial compilation
from functools import partial

# Imports of Custom Functions and Classes
from ... import harmonic_utils as hutils
from ...jax import harmonic_utils as jhutils # Jax version of harmonic utils
from ...nlforces.nonlinear_force import NonlinearForce


class ElasticDryFriction2D(NonlinearForce):
    """
    Elastic Dry friction Slider Element Nonlinearity with JAX for automatic 
    differentiation

    The AFT formulation assumes that the spring starts at 0 force
    at zero displacement.    
    """

    def __init__(self, Q, T, kt, kn, mu, u0=0, meso_gap=0):
        """
        Initialize a nonlinear force model
        
        Implementation currently assumes that Nnl=2 (two nonlinear DOFs)
        The Nonlinear DOFs must first be tangential displacement then normal
        displacement
        
        Has been updated to allow for multiple elastic dry friction sliders. 
        However, that may be memory inefficient since the Jacobians that are 
        calculated aren't sparse.

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
        u0 : float or None or (Nnl,) numpy.ndarray
            initialization value for the slider. If u0 = None, then 
                the zeroth harmonic is used to initialize the slider position.
                u0 should be size of number of tangential DOFs 
                (e.g., 1 right now)
                Highly recommended not to use u0=None because may result in
                non-unique solutions. Not fully verified for None option.
         meso_gap : float or (Nnl/2,) numpy.ndarray
                 gap between frictional elements (Topology and distribution 
                                                     of asperity parameters )

        """
        self.Q = np.asarray(Q)
        self.T = np.asarray(T)
        
        if not type(self.Q) == type(Q):
            warnings.warn('Matrix Q argument is not a numpy array. Conversion '
                          'to numpy array was attempted, but not '
                          'guaranteed to work.')
            
        if not type(self.T) == type(T):
            warnings.warn('Matrix T argument is not a numpy array. Conversion '
                          'to numpy array was attempted, but not '
                          'guaranteed to work.')
  
        
        self.kt = kt
        self.kn = kn
        self.mu = mu
        self.prestress_mu = 0.0
        self.real_mu = mu
        
        self.u0 = u0
        
        self.init_history()
        # Topology and distribution of asperity parameters 
        self.meso_gap = meso_gap
        
    def nl_force_type(self):
        """
        Marks as a hysteretic force.
        """
        return 1
        
    def set_prestress_mu(self):
        """
        Set friction coefficient to a different value (generally 0.0) for
        prestress analysis
        """
        self.mu = self.prestress_mu
        
    def reset_real_mu(self): 
        """
        Reset friction coefficient to a real value (generally not 0.0) for
        dynamic analysis
        """
        self.mu = self.real_mu
        
       
    def init_history(self):
        """
        Initialize history variables to zero
        Only applies to tangent displacement and force
        """
        self.up = np.zeros(self.Q.shape[0]//2)
        self.fp = np.zeros(self.Q.shape[0]//2)


    def set_aft_initialize(self, X):
        """
        Set a center for frictional sliders to be initialized at zero force
        for AFT routines. 

        Parameters
        ----------
        X : np.array, size (Ndof,)
            Solution to a static solution or other desired set of displacements
            that will be used as the baseline position of frictional sliders.

        Returns
        -------
        None.

        """
        self.u0 = self.Q @ X
        
        return
        
        
    def update_history(self, unl, fnl):
        """
        Updates hysteretic states

        Parameters
        ----------
        unl : nonlinear displacements to update
        fnl : nonlinear forces to save as update

        Returns
        -------
        None.

        """
        self.up = unl
        self.fp = fnl
        
    def force(self, X, update_hist=False):
        """
        Forces from nonlinearity given some global displacement vector

        Parameters
        ----------
        X : Displacements
        update_hist : Option to save the inputs for future calls as history 
                        variables

        Returns
        -------
        F : Global nonlinear force
        dFdX : Global nonlinear force gradient

        """

        
        # Get local displacements
        unl = self.Q @ X
        
        ############################
        # fnl = np.zeros_like(unl)
        # dfnldunl = np.zeros((unl.shape[0], unl.shape[0]))
        
        pars = np.array([self.kt, self.kn, self.mu])
        
        dfnldunl,fnl = _local_eldy_force_grad(unl, self.up, self.fp, pars, 
                                              self.meso_gap)
        #############################
        
        F = self.T @ fnl
        
        dFdX = self.T @ dfnldunl @ self.Q
        
        if update_hist: 
            self.update_history(unl[0::2], fnl[0::2])
        
        return F, dFdX
        
    def aft(self, U, w, h, Nt=128, tol=1e-7):
        """
        
        Tolerances are ignored since Elastic Dry Friction converges to steady
        -state with 
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
        
        pars = np.array([self.kt, self.kn, self.mu])

        # # If no Grad is needed use:
        # Flocal = _local_aft_jenkins(Uwlocal, pars, u0, tuple(h), Nt, u0h0)[0]
        
        # Case with gradient and local force
        dFdUwlocal, Flocal = _local_aft_eldry_grad(Uwlocal, pars, u0, \
                                        tuple(h), Nt, u0h0, self.meso_gap)
        
        
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
    
    
    def local_force_history(self, unlt, unltdot, h, cst, unlth0, max_repeats=2, \
                            atol=1e-10, rtol=1e-10):
         """
         Calculates local force history for a single element of the rough 
         contact model given the local nonlinear displacements over a cycle.
    
         Parameters
         ----------
         unlt : Displacement history for a single cycle.
         unltdot : Velocity history for the cycle (ignored)
         h : list of harmonics used (ignored)
         cst : (ignored)
         unlth0 : Initialization displacements for the sliders 
         max_repeats : Number of times to repeat the cycle of displacements
                       to converge the hysteresis loops to steady-state
         atol : Ignored
         rtol : Ignored
    
         Returns
         -------
         fxyn_t : Force history for the element
    
         """
         #########################
         # Determine Slider Starting Position
         
         if self.u0 is None:
             # Initialize based on the zeroth harmonic.
             u0 = unlth0
         else:
             u0 = self.u0*np.ones_like(unlth0[0])
         
         pars = np.array([self.kt, self.kn, self.mu])
         
         fxyn_t = _local_force_history(unlt, pars, u0[::2], self.meso_gap)

         return (fxyn_t,)
        
    
def _local_eldy_force(unl, up, fp, pars, meso_gap):
    """
    See _local_eldy_force_grad for gradient call.

    Parameters
    ----------
    unl : Local nonlinear force vector of size (2*N) for N elastic dry friction 
            elements
    up : Previous displacements for tangential direction only (N, )
    fp : Previous tangential forces only (N,)
    pars : Parameters

    Returns
    -------
    fnl : Nonlinear forces, output twice for gradient function

    """
    # Recover pars for convenience / readability
    kt = pars[0]
    kn = pars[1]
    mu = pars[2]
    
    fnl = jnp.zeros_like(unl)
    
    # Normal Force
    fnl = fnl.at[1::2].set(jnp.maximum((unl[1::2]-meso_gap)*kn, 0.0))
    
    # Tangent Force - must use where to get tradient consistently correct
    fpred = kt*(unl[0::2]-up)
    
    fcurr = jnp.where(fpred < mu*fnl[1::2], fpred, mu*fnl[1::2])
        
    # Other Directly slip limit tangent force
    fnl = fnl.at[0::2].set(jnp.where(fcurr>-mu*fnl[1::2], fcurr, -mu*fnl[1::2]))

    return fnl,fnl

@partial(jax.jit) 
def _local_eldy_force_grad(unl, up, fp, pars, meso_gap):
    """
    Function that computes the gradient of local force. Using Aux data allows for 
    returning Fnl also from one function call. 

    Parameters
    ----------
    unl : Local nonlinear force vector of size (2*N) for N elastic dry friction 
            elements
    up : Previous displacements for tangential direction only (N, )
    fp : Previous tangential forces only (N,)
    pars : Parameters

    Returns
    -------
    dfnldunl : gradient of nonlinear forces
    fnl : Nonlinear forces


    """
    
    dfnldunl,fnl = jax.jacfwd(_local_eldy_force, has_aux=True)(unl, up, fp, pars,meso_gap)
    
    return dfnldunl,fnl


def _local_eldry_loop_body(ind, ft, unlt, kt, mu):
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
    
    fcurr = jnp.minimum(kt*(unlt[ind, 0::2]-unlt[ind-1, 0::2]) + ft[ind-1, 0::2],
                            mu*ft[ind, 1::2])
    
    ft = ft.at[ind, 0::2].set(jnp.maximum(fcurr, -mu*ft[ind, 1::2]))
    
    return ft
 

@partial(jax.jit, static_argnums=(3,4,5)) 
def _local_aft_eldry(Uwlocal, pars, u0, htuple, Nt, u0h0, meso_gap):
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
    u0 : scalar/vector value for the displacement to initialize the slider to
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
    
    # Do not need velocity, this is how it would be calculated:
    # # Nt x Ndnl
    # unltdot = Uwlocal[-1]*jhutils.time_series_deriv(Nt, htuple, Ulocal, 1) 
    
    ########################################
    #### Start slider in desired position
    u0 = u0*jnp.ones_like(Ulocal[0])
    
    # if u0 comes from the zeroth harmonic, pull it from the jax traced 
    # array rather than the separate input value, which is constant as far as 
    # gradients are concerned.
    u0 = jnp.where(u0h0, Ulocal[0, 0::2], u0[0::2])
    
    ft = _local_force_history(unlt, pars, u0, meso_gap)
    
    # Convert back into frequency domain
    Flocal = jhutils.get_fourier_coeff(htuple, ft)
    
    # Flatten back to a 1D array
    Flocal = jnp.reshape(Flocal.T, (-1,), 'F')
    
    return Flocal,Flocal
        

@partial(jax.jit, static_argnums=(3,4,5)) 
def _local_aft_eldry_grad(Uwlocal, pars, u0, htuple, Nt, u0h0, meso_gap):
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
    
    J,F = jax.jacfwd(_local_aft_eldry, has_aux=True)(Uwlocal, pars, u0, 
                                                       htuple, Nt, u0h0, meso_gap)    
    return J,F

@partial(jax.jit) 
def _local_force_history(unlt, pars, u0, meso_gap):
    """
    Calculates the steady-state displacement history for a set of displacements
    
    NOTES:
        This function throws an error if not JIT compiled because the loop body
        would be dependent on a non-static argument in that case.

    Parameters
    ----------
    unlt : Time history of displacements to evaluate cycles over - size (Nt, 3)
    pars : 


    Returns
    -------
    fxyn_t : History of total forces (Nt, 3)
    
    
    """
    Nt = unlt.shape[0]
    
    # Recover pars for convenience / readability
    kt = pars[0]
    kn = pars[1]
    mu = pars[2]
    
    # Initialize force time memory
    ft = jnp.zeros_like(unlt)
    
    # Normal Force
    ft = ft.at[:, 1::2].set(jnp.maximum((unlt[:, 1::2]-meso_gap)*kn, 0.0))
    
    # Do a loop function for each update at index i
    loop_fun = lambda i,f : _local_eldry_loop_body(i, f, unlt, kt, mu)
    
    
    ########################################
    #### Start slider in desired position
    
    # The first evaluation is based on the last entry of ft and therefore 
    # initialize the last entry of ft based on a linear spring
    # slip limit does not need to be applied since this just needs to get stuck
    # regime correct for the first step to be through zero. 
    ft = ft.at[-1, 0::2].set(kt*(unlt[-1, 0::2] - u0))
        
    # Conduct exactly 2 repeats of the hysteresis loop to be converged to 
    # steady-state
    for out_ind in range(2):
        
        # This construct must be used otherwise compiling tries writing out
        # all Nt steps of the loop updates and is excessively slow
        ft = jax.lax.fori_loop(0, Nt, loop_fun, ft)
    

    return ft
