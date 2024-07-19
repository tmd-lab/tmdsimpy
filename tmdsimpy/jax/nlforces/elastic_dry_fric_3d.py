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


class ElasticDryFriction3D(NonlinearForce):
    """
    2D Elastic Dry Friction (Spring + Coulomb Friction) Model
    
    Parameters
    ----------
    Q : (Nnl, N) numpy.ndarray
        Matrix tranform from the `N` degrees of freedom (DOFs) of the system 
        to the `Nnl` local nonlinear DOFs. 
        `Nnl` should be even.
        Rows `0::2` correspond to local tangential DOFs.
        Rows `1::2` correspond to local normal DOFs.
    T : (Nnl, N) numpy.ndarray
        Matrix tranform from the local `Nnl` forces to the `N` global DOFs.
        Columns `0::2` correspond to local tangential forces.
        Columns `1::2` correspond to local normal forces.
    kt : float
        Tangential stiffness
    kn : float
        Tangential stiffness
    mu : float
        Friction coefficient
    u0 : float, (Nnl,) numpy.ndarray, or None, optional
        If a float, all sliders in AFT are initialized at that displacement and
        zero force.
        If a numpy.ndarray, entries `0::2` correspond to the initial slider
        displacements and zero force for AFT.
        If None, then the zeroth harmonic displacements are used to initialize
        the slider position in AFT.
        Highly recommended not to use `u0=None` because may result in
        non-unique solutions. Not fully verified for None option.
        The default is 0.
    
    See Also
    --------
    set_aft_initialize :
        Overrides the value of `u0`.
    
    Notes
    -----
    
    Positive normal displacements are in contact and yield positive normal
    forces. Negative normal displacements are out of contact and yield zero
    normal force.
    
    Derivatives are calculated with automatic differentiation using JAX. 
    JAX calculates dense derivative matrices, so the calculation may become
    very inefficient if more than 1 slider is included in the object 
    (i.e., `Nnl` > 2).
    
    `kt`, `kn`, and `mu` may work for inputs of `(Nnl//2,) numpy.ndarrays`. 
    However, this is not tested.
    
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
        
        dof = self.Q.shape[0]
        
        # Create a mask with the same length as the array
        ti = np.zeros(dof, dtype=bool)
        # Set True for the 1st and 2nd element of each group of three
        ti[0::3] = True
        ti[1::3] = True
        
        self.ti = ti
        
    def nl_force_type(self):
        """
        Method to identify the force type as hysteretic. 
        
        Returns
        -------
        int
            1, indicating hysteretic force type.
        """
        
        return 1
        
    def set_prestress_mu(self):
        """
        Sets friction coefficient to zero while saving initial value in a 
        different variable. Useful for prestress analysis.
        
        Returns
        -------
        None
        """
        self.mu = self.prestress_mu
        
    def reset_real_mu(self): 
        """
        Resets friction coefficient to initial value. 
        Useful for after prestress analysis with zero friction coefficient.
        
        Returns
        -------
        None
        """
        self.mu = self.real_mu
        
    def init_history(self):
        """
        Method to initialize history variables to zero force and displacement.

        Returns
        -------
        None.
        
        Notes
        -----
        History variables are just initialized for tangential displacements.
        
        """
        self.up = np.zeros(self.Q.shape[0]//3*2)
        self.fp = np.zeros(self.Q.shape[0]//3*2)

    def set_aft_initialize(self, X):
        """
        Set an initial slider position with zero force for AFT calculation.

        Parameters
        ----------
        X : (N,) numpy.ndarray
            Global displacements to be used with `self.Q` to calculate
            local slider positions for initializing AFT.
            Generally, a solution to a static problem.

        Returns
        -------
        None.

        """
        self.u0 = self.Q @ X
        
    def update_history(self, unl, fnl):
        """
        Updates hysteretic states to be the input displacement and force.

        Parameters
        ----------
        unl : (2*Nnl//3,) numpy.ndarray
            Local tangential nonlinear displacements to save
        fnl : (2*Nnl//3,) numpy.ndarray
            Local tangential nonlinear forces to save

        Returns
        -------
        None.

        """
        
        self.up = unl
        self.fp = fnl
        
    def force(self, X, update_hist=False):
        """
        Calculate global nonlinear forces for some global displacement vector.

        Parameters
        ----------
        X : (N,) numpy.ndarray
            Global displacements
        update_hist : bool
            Flag to save displacement and force from the evaluation as history
            variables for subsequent calls to this function.

        Returns
        -------
        F : (N,) numpy.ndarray
            Global nonlinear force
        dFdX : (N,N) numpy.ndarray
            Derivative of `F` with respect to `X`.
            
        Notes
        -----
        If `update_hist` is True, then `update_history` is called on the 
        local results of this calculation to save the history.

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
            self.update_history(unl[self.ti], fnl[self.ti])
        
        return F, dFdX
        
    def aft(self, U, w, h, Nt=128, tol=1e-7, calc_grad=True):
        """
        Implementation of the alternating frequency-time (AFT) method to extract 
        harmonic nonlinear force coefficients.
        
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
        calc_grad : boolean
            Flag where True indicates that the gradients should be calculated 
            and returned. If False, then returns only (Fnl,) as a tuple. 
            The default is True
        
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
        The tolerance `tol` is ignored because elastic dry friction converges
        to steady-state with two cycles of the hysteresis loop. Two cycles of
        the nonlinear forces are calculated automatically without the option to
        change this setting.

        """

        
        #########################
        # Memory Initialization 
        
        Fnl = np.zeros_like(U)
        
        if calc_grad:
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
        
        if calc_grad:
        # Case with gradient and local force
            dFdUwlocal, Flocal = _local_aft_eldry_grad(Uwlocal, pars, u0, \
                                        tuple(h), Nt, u0h0, self.meso_gap)
        else:
            Flocal,_ = _local_aft_eldry(Uwlocal, pars, u0, \
                                        tuple(h), Nt, u0h0, self.meso_gap)
            
        #########################
        # Convert AFT to Global Coordinates
        
        # Reshape Flocal
        Flocal = jnp.reshape(Flocal, (Ndnl, Nhc), 'F')
        
        # Global coordinates        
        Fnl = np.reshape(self.T @ Flocal, (U.shape[0],), 'F')
        
        if calc_grad:
            dFnldU = np.kron(np.eye(Nhc), self.T) @ dFdUwlocal[:, :-1] \
                                                    @ np.kron(np.eye(Nhc), self.Q)
            
            dFnldw = np.reshape(self.T @ \
                                np.reshape(dFdUwlocal[:, -1], (Ndnl, Nhc)), \
                                (U.shape[0],), 'F')
            
            return Fnl, dFnldU, dFnldw
        else:
            return (Fnl,)
    
    
    def local_force_history(self, unlt, unltdot, h, cst, unlth0, max_repeats=2,
                            atol=1e-10, rtol=1e-10):

        """
        Evaluate the local forces for steady-state harmonic motion used in AFT.
        
        Parameters
        ----------
        unl : (Nt,Nnl) numpy.ndarray
            Local displacements, rows are different time instants and
            columns are different displacement DOFs.
        unldot : (Nt,Nnl) numpy.ndarray
            Ignored here, included for compatibility of interface.
            Local velocities, rows are different time instants and
            columns are different displacement DOFs.
        h : 1D numpy.ndarray, sorted
            Ignored here, included for compatibility of interface.
            List of harmonics used in subsequent analysis. Corresponds
            to `Nhc` harmonic components.
        cst: (Nt,Nhc) numpy.ndarray
            Ignored here, included for compatibility of interface.
            Evaluation of each harmonic component (columns) at a given instant
            in time (row = instant in time). These are without any harmonic
            coefficients, so are just cosine and sine evaluations.
        unlth0 : (Nnl,) numpy.ndarray
            Zeroth harmonic contributions to a time series of displacements.
            This is passed to `init_history_harmonic` to initialize model is
            `self.u0` is None.
        max_repeats : int, optional
            Ignored here, included for compatibility of interface.
            The default is 2.
        atol : float, optional
            Ignored here, included for compatibility of interface.
            Absolute tolerance on force time series convergence to steady-state
            (final state of cycle).
            The default is 1e-10.
        rtol : float, optional
            Ignored here, included for compatibility of interface.
            Relative tolerance on force time series convergence to steady-state
            (final state of cycle).
            The default is 1e-10.
            
        Returns
        -------
        ft : (Nt,Nnl) numpy.ndarray
            Local nonlinear forces. First index is time instants, second index
            is which local nonlinear force DOF. This is returned as the first
            entry in a tuple.
        
        Notes
        -----
        This method is for the post-processing of force displacement
        relationships of the model from harmonic solutions.
        
        This method is not directly called by AFT for the elastic dry
        friction model. Rather this just provides a public interface to the 
        same private JAX function that AFT uses. As such, only the forces and
        not the derivatives are returned.


        """
        
         #########################
         # Determine Slider Starting Position
         
        if self.u0 is None:
            # Initialize based on the zeroth harmonic.
            u0 = unlth0
        else:
            u0 = self.u0*np.ones_like(unlth0[0])
        
        pars = np.array([self.kt, self.kn, self.mu])
        
        fxyn_t = _local_force_history(unlt, pars, u0[self.ti], self.meso_gap)
        
        #########################
        # Determine Slider Starting Position

        if self.u0 is None:
            # Initialize based on the zeroth harmonic.
            u0 = unlth0
        else:
            u0 = self.u0*np.ones_like(unlth0[0])

        pars = np.array([self.kt, self.kn, self.mu])

        fxyn_t = _local_force_history(unlt, pars, u0[self.ti], self.meso_gap)

        return (fxyn_t,)
        
    
def _local_eldy_force(unl, up, fp, pars, meso_gap):
    """
    Private function for local force evaluation.

    Parameters
    ----------
    unl : (Nnl,) numpy.ndarray
        Local nonlinear displacements for (Nnl//2) elastic dry friction 
        elements
    up : (2*Nnl//3,) numpy.ndarray
        Previous displacements for tangential direction only
    fp : (2*Nnl//3,) numpy.ndarray
        Previous tangential forces only
    pars : (3, Nnl//3) or (Nnl//3) numpy.ndarray
        Contains `kt = pars[0]` (tangential stiffness), 
        `kn = pars[1]` (normal stiffness), 
        and `mu = pars[2]` (friction coefficient).
    
    Returns
    -------
    fnl : (Nnl,) numpy.ndarray
        Nonlinear forces in local domain
    fnl : (Nnl,) numpy.ndarray
        Nonlinear forces in local domain (repeated to have access in gradient 
        function)
    
    See Also
    --------
    _local_eldy_force_grad :
        Function with gradient evaluation.

    """
    # Recover pars for convenience / readability
    kt = pars[0]
    kn = pars[1]
    mu = pars[2]
    
    fnl = jnp.zeros_like(unl)
    
    # Normal Force
    fnl = fnl.at[2::3].set(jnp.maximum((unl[2::3]-meso_gap)*kn, 0.0))
    
    # Tangent Force - must use where to get tradient consistently correct
    fpredx = kt*(unl[0::3]-up[0::2])
    fpredy = kt*(unl[1::3]-up[1::2])
    
    fcurrx = jnp.where(fpredx < mu*fnl[2::3], fpredx, mu*fnl[2::3])
    fcurry = jnp.where(fpredy < mu*fnl[2::3], fpredy, mu*fnl[2::3])
        

    # Other Directly slip limit tangent force
    fnl = fnl.at[0::3].set(jnp.where(fcurrx>-mu*fnl[2::3], fcurrx, -mu*fnl[2::3]))
    fnl = fnl.at[1::3].set(jnp.where(fcurry>-mu*fnl[2::3], fcurry, -mu*fnl[2::3]))

    return fnl, fnl

@partial(jax.jit) 
def _local_eldy_force_grad(unl, up, fp, pars, meso_gap):
    """
    Function that computes the gradient of local force. Using Aux data allows for 
    returning Fnl also from one function call. 

    Parameters
    ----------
    unl : (Nnl,) numpy.ndarray
        Local nonlinear displacements for (Nnl//3) elastic dry friction 
        elements
    up : (2*Nnl//3,) numpy.ndarray
        Previous displacements for tangential direction only
    fp : (2*Nnl//3,) numpy.ndarray
        Previous tangential forces only
    pars : (3, Nnl//3) or (Nnl//3) numpy.ndarray
        Contains `kt = pars[0]` (tangential stiffness), 
        `kn = pars[1]` (normal stiffness), 
        and `mu = pars[2]` (friction coefficient).

    See Also
    --------
    _local_eldy_force : 
        Function with same input parameters, but does not return gradient.

    Returns
    -------
    dfnldunl : (Nnl,Nnl) numpy.ndarray
        Derivative of nonlinear forces in local domain with respect to `unl`.
    fnl : (Nnl,) numpy.ndarray
        Nonlinear forces in local domain (repeated to have access in gradient 
        function)

    """
    
    dfnldunl,fnl = jax.jacfwd(_local_eldy_force, has_aux=True)(unl, up, fp, pars,meso_gap)
    
    return dfnldunl,fnl


def _local_eldry_loop_body(ind, ft, unlt, kt, mu):
    """
    Calculation of elastic dry friction force at a given instant for 
    steady-state calculation in AFT.

    Parameters
    ----------
    ind : int
        Index that is being updated for this loop step corresponding to a row
        of `unlt` and `ft`
    ft : (Nt,Nnl) numpy.ndarray
        Array of force values for all time
    unlt : (Nt,Nnl) numpy.ndarray
        Displacements for Jenkins for all times
    kt : float or (Nnl//2,) numpy.ndarray
        Tangential stiffness parameter
    Fs : float or (Nnl//2,) numpy.ndarray
        Friction coeffient.

    Returns
    -------
    ft : (Nt,Nnl) numpy.ndarray
        Force array with the entry at row `ind` updated for the nonlinear force
        evaluation.
    
    Notes
    -----
    
    The normal forces corresponding to `ft[:, 1::2]` are precalculated before
    the call to this loop body function.
    
    History appears at index `ind-1`. For `ind=0`, this corresponds to the last
    entry of the arrays.

    """
    
   
    fcurrx = jnp.minimum(kt*(unlt[ind, 0::3]-unlt[ind-1, 0::3]) + ft[ind-1, 0::3],
                            mu*ft[ind, 2::3])
    
    fcurry = jnp.minimum(kt*(unlt[ind, 1::3]-unlt[ind-1, 1::3]) + ft[ind-1, 1::3],
                            mu*ft[ind, 2::3])
    
    ft = ft.at[ind, 0::3].set(jnp.maximum(fcurrx, -mu*ft[ind, 2::3]))
    ft = ft.at[ind, 1::3].set(jnp.maximum(fcurry, -mu*ft[ind, 2::3]))
    
    return ft
 

@partial(jax.jit, static_argnums=(3,4,5)) 
def _local_aft_eldry(Uwlocal, pars, u0, htuple, Nt, u0h0, meso_gap):
    """
    Conducts AFT in a functional form that can be used with JAX and JIT

    Parameters
    ----------
    Uwlocal : (Nnl*Nhc + 1,) numpy.ndarray
        Local nonlinear displacements for each harmonic component in order
        followed by the frequency in rad/s. Each harmonic component is listed
        in full before the next one.
    pars : (3, Nnl//3) or (Nnl//3) numpy.ndarray
        Contains `kt = pars[0]` (tangential stiffness), 
        `kn = pars[1]` (normal stiffness), 
        and `mu = pars[2]` (friction coefficient).
    u0 : float or (Nnl,) numpy.ndarray
        Value for the displacement to initialize the slider to.
    htuple : tuple of int, sorted
        tuple list of harmonics. Tuple is used so that it can be made static
    Nt : int, power of 2
        Number of AFT time steps to be used. 
    u0h0 : bool
        Flag for if the zeroth harmonic displacements should be taken 
        rather than u0. For correct gradient calculation, u0 is ignored if
        u0h0 is True, even if they are identical.
    
    Returns
    -------
    Flocal : (Nhc*Nnl,) numpy.ndarray 
        Harmonic force coefficients locally, same format as U part of Uwlocal 
    Flocal : (Nhc*Nnl,) numpy.ndarray
        Repeated return value to allow for JAX auto diff calculation.
    
    Notes
    -----
    Elastic dry friction converges in at most 2 repeats of the hysteresis loop.
    This always does exactly two loops.
    
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
      
    # Create a mask with the same length as the array
    ti = np.zeros(Ndnl, dtype=bool)
    # Set True for the 1st and 2nd element of each group of three
    ti[0::3] = True
    ti[1::3] = True

    u0 = jnp.where(u0h0, Ulocal[0, ti], u0[ti])
    
    ft = _local_force_history(unlt, pars, u0, meso_gap)
    
    # Convert back into frequency domain
    Flocal = jhutils.get_fourier_coeff(htuple, ft)
    
    # Flatten back to a 1D array
    Flocal = jnp.reshape(Flocal.T, (-1,), 'F')
    
    return Flocal,Flocal
        

@partial(jax.jit, static_argnums=(3,4,5)) 
def _local_aft_eldry_grad(Uwlocal, pars, u0, htuple, Nt, u0h0, meso_gap):
    """
    Private function that computes the gradient of AFT for elastic dry 
    friction.

    Parameters
    ----------
    Uwlocal : (Nnl*Nhc + 1,) numpy.ndarray
        Local nonlinear displacements for each harmonic component in order
        followed by the frequency in rad/s. Each harmonic component is listed
        in full before the next one.
    pars : (3, Nnl//3) or (Nnl//3) numpy.ndarray
        Contains `kt = pars[0]` (tangential stiffness), 
        `kn = pars[1]` (normal stiffness), 
        and `mu = pars[2]` (friction coefficient).
    u0 : float or (Nnl,) numpy.ndarray
        Value for the displacement to initialize the slider to.
    htuple : tuple of int, sorted
        tuple list of harmonics. Tuple is used so that it can be made static
    Nt : int, power of 2
        Number of AFT time steps to be used. 
    u0h0 : bool
        Flag for if the zeroth harmonic displacements should be taken 
        rather than u0. For correct gradient calculation, u0 is ignored if
        u0h0 is True, even if they are identical.
    
    Returns
    -------
    J : (Nhc*Nnl,Nhc*Nnl+1) numpy.ndarray 
        Derivative of `F` with respect to `Uwlocal`
    F : (Nhc*Nnl,) numpy.ndarray
        Repeated return value to allow for JAX auto diff calculation.
    
    Notes
    -----
    Elastic dry friction converges in at most 2 repeats of the hysteresis loop.
    This always does exactly two loops.
    
    """
    
    J,F = jax.jacfwd(_local_aft_eldry, has_aux=True)(Uwlocal, pars, u0, 
                                                       htuple, Nt, u0h0, meso_gap)    
    return J,F

@partial(jax.jit) 
def _local_force_history(unlt, pars, u0, meso_gap):
    """
    Calculates the steady-state force history for elastic dry friction.

    Parameters
    ----------
    unlt : (Nt,Nnl)
        Time history of displacements to evaluate cycles over. Rows are time
        instants, columns are local displacements for nonlinear evaluation.
    pars : (3, Nnl//3) or (Nnl//3) numpy.ndarray
        Contains `kt = pars[0]` (tangential stiffness), 
        `kn = pars[1]` (normal stiffness), 
        and `mu = pars[2]` (friction coefficient).
    u0 : float or (Nnl,) numpy.ndarray
        Value for the displacement to initialize the slider to.

    Returns
    -------
    fxyn_t : (Nt,Nnl) jax.numpy.ndarray
        History of total forces

    Notes
    -----
    Function is JIT compiled because it may throw an error if not compiled.
    Previous comment suggested that it is because of non-static arguments,
    but that issue is not immediately clear looking at it here.

    """
    Nt = unlt.shape[0]
    
    # Recover pars for convenience / readability
    kt = pars[0]
    kn = pars[1]
    mu = pars[2]
    
    # Initialize force time memory
    ft = jnp.zeros_like(unlt)                                                                                                                                                                                                                                                                   
     
    # Normal Force
    ft = ft.at[:, 2::3].set(jnp.maximum((unlt[:, 2::3]-meso_gap)*kn, 0.0))
    
    # Do a loop function for each update at index i
    loop_fun = lambda i,f : _local_eldry_loop_body(i, f, unlt, kt, mu)
    
    
    ########################################
    #### Start slider in desired position
    # The first evaluation is based on the last entry of ft and therefore 
    # initialize the last entry of ft based on a linear spring
    # slip limit does not need to be applied since this just needs to get stuck
    # regime correct for the first step to be through zero. 
    ft = ft.at[-1, 0::3].set(kt*(unlt[-1, 0::3] - u0[::2]))
    ft = ft.at[-1, 1::3].set(kt*(unlt[-1, 1::3] - u0[1::2]))
        
    # Conduct exactly 2 repeats of the hysteresis loop to be converged to 
    # steady-state
    for out_ind in range(2):
        
        # This construct must be used otherwise compiling tries writing out
        # all Nt steps of the loop updates and is excessively slow
        ft = jax.lax.fori_loop(0, Nt, loop_fun, ft)
    

    return ft
