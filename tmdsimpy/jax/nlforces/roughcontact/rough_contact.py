"""
Rough Contact Friction Model utilizing JAX for Derivatives

This model is a python implementation of the work in: 
    J. H. Porter and M. R. W. Brake, 2023, Towards a predictive, physics-based 
    friction model for the dynamics of jointed structures, Mechanical Systems 
    and Signal Processing
    
This model only considers the simplest tangent model and plasticity in the 
normal direction

Code for the MATLAB version is available on GitHub: 
    https://github.com/tmd-lab/microslip-rough-contact
"""

# Standard imports
import numpy as np

# JAX imports
import jax
import jax.numpy as jnp

# Decoractions for Partial compilation
from functools import partial

# Imports of Custom Functions and Classes
from .... import harmonic_utils as hutils
from ....jax import harmonic_utils as jhutils # Jax version of harmonic utils
from ....nlforces.nonlinear_force import NonlinearForce

# Import of functions stored in a different file
from . import _asperity_functions as asp_funs

###############################################################################
########## Class for Rough Contact Models                            ##########
###############################################################################

class RoughContactFriction(NonlinearForce):
    """
    Elastic Dry friction Slider Element Nonlinearity with JAX for automatic 
    differentiation

    The AFT formulation assumes that the spring starts at 0 force
    at zero displacement.    
    """

    def __init__(self, Q, T, ElasticMod, PoissonRatio, Radius, TangentMod, 
                 YieldStress, mu, u0=0, meso_gap=0, gaps=None, gap_weights=None):
        """
        Initialize a nonlinear force model
        
        Implementation currently assumes that Nnl=3 (three nonlinear DOFs)
        The Nonlinear DOFs must first be both tangential displacement then 
        normal displacement
        

        Parameters
        ----------
        Q : Transformation matrix from system DOFs (n) to nonlinear DOFs (Nnl), 
            Nnl x n
        T : Transformation matrix from local nonlinear forces to global 
            nonlinear forces, n x Nnl
        ElasticMod : Elastic Modulus
        PoissonRatio : Poisson's Ratio
        Radius : Initial Undeformed Effective Asperity Radius 
                (Half of real radius of one asperity in general - see citation at top)
        TangentMod : Plasticity Hardening Modulus
        YieldStress : Yield Strength / Yield Stress
        mu : Friction Coefficient
        
        
        [Additional Inputs will be added for asperity distribution statistics] 
        [SlipType : Flag for different types of friction coefficients - add later]   
        
        u0 : initialization value for the slider. If u0 = 'Harm0', then 
                the zeroth harmonic is used to initialize the slider position.
                u0 should be broadcastable to the tangential DOFs
                Highly recommended not to use u0='Harm0' because may result in
                non-unique solutions. This option is solely included for 
                testing against previous versions

        """
        self.Q = Q
        self.T = T
        self.elastic_mod = ElasticMod
        self.poisson = PoissonRatio
        self.Re = Radius
        self.tangent_mod = TangentMod
        self.sys = YieldStress
        self.mu = mu # This friction coefficient sometimes switches between real and prestress
        self.real_mu = mu # Save real friction coefficient
        self.prestress_mu = 0.0 # Prestress should use zero friction coefficient
        
        self.u0 = u0
        
        # Calculated Initial Parameters
        self.Estar = ElasticMod / 2.0 / (1.0 - PoissonRatio**2)
        
        self.shear_mod = ElasticMod / 2.0 / (1.0 + PoissonRatio)

        self.Gstar = self.shear_mod / 2.0 / (2.0 - PoissonRatio)
        
        # Plasticity Parameters
        C = 1.295*np.exp(0.736*self.poisson);
        
        # displacement of one sphere against a rigid flat to cause yielding.
        delta_y1s = (np.pi*C*self.sys/(2*(2*self.Estar)))**2*(2*self.Re); 
        
        self.delta_y = delta_y1s*2
        
        # Topology and distribution of asperity parameters 
        self.meso_gap = meso_gap
        
        if gaps is None:
            # self.gaps = 
            # self.gap_weights = 
            assert False, "Need to implement case without specifying gap weights directly."
        else:
            self.gaps = gaps
            self.gap_weights = gap_weights
            
            assert gap_weights is not None, "Need to specify gap weights if specifying gap values."
            
        # Just consider default of starting sliders at origin for AFT
        self.uxyn_initialize = np.array([0.0, 0.0, 0.0])
            
        # Initialize History Variables
        self.init_history()

    def nl_force_type(self):
        """
        Marks as a hysteretic force.
        """
        return 1 
    
    def init_history(self):
        
        self.unmax = 0
        self.Fm_prev = np.zeros_like(self.gap_weights)
        self.fxy0 = np.zeros((self.gap_weights.shape[0], 2))
        self.uxyn0 = np.zeros(3)
    
    def update_history(self, uxyn, Fm_curr, fxy_curr):
        
        self.unmax = np.maximum(uxyn[-1], self.unmax)
        self.Fm_prev = Fm_curr
        self.fxy0 = fxy_curr
        self.uxyn0 = uxyn
        
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
        
        self.uxyn_initialize = self.Q @ X
        
        return
    
    def force(self, X, update_hist=False, return_aux=False):
        """
        Static Force Evaluation
        
        NOTE: Static Force evaluation does not support non-zero tangential 
        displacements. In short, the contact radius is discontinuous upon 
        normal load reversal and thus tangential stiffness is discontinuous
        This can potentially cause a large jump in tangential force based on 
        slight normal displacement variations and break solvers. 
        
        [Future Update - Allow for tangential displacements so can get 
         stiffness for linearized eigen solve on unloading curve.]

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        update_hist : TYPE, optional
            DESCRIPTION. The default is False.
        return_aux : flag to return extra results about the simulation (aux)

        Returns
        -------
        F : TYPE
            DESCRIPTION.
        dFdX : TYPE
            DESCRIPTION.
        aux : Tuple of extra results includes (Fm_prev, deltabar, Rebar, a)
                Fm_prev : previous maximum normal force per asperity
                deltabar : permanent deformation displacement of each asperity
                Rebar : flattened (new) radius of each asperity
                a : radius of contact area of each asperity.

        """
        uxyn = self.Q @ X
        
        # Local Force evaluation based on unl
        dfnldunl, fnl, aux = _static_force_grad(uxyn, self.uxyn0, self.fxy0, 
                                            self.unmax, self.Fm_prev, 
                                            self.mu, self.meso_gap, self.gaps, 
                                            self.gap_weights, self.Re, 
                                            self.poisson, self.Estar, 
                                            self.elastic_mod, 
                                            self.tangent_mod, self.delta_y, 
                                            self.sys, self.Gstar)
        
        Fm_curr = aux[1]
        fxy_curr = aux[2]
        
        # Convert Back to Physical
        F = self.T @ fnl
        
        dFdX = self.T @ dfnldunl @ self.Q
        
        if update_hist:
            self.update_history(uxyn, Fm_curr, fxy_curr)
            
        if return_aux:
            return F, dFdX, aux[1:]
            
        else:
            return F, dFdX
    
    
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
        
        fxyn_t = _local_force_history(unlt, unlth0, 
                                    self.mu, self.meso_gap, self.gaps, 
                                    self.gap_weights, self.Re, self.poisson, 
                                    self.Estar, self.elastic_mod, self.tangent_mod, 
                                    self.delta_y, self.sys, self.Gstar, 
                                    repeats=max_repeats)
        
        # typical return statement also requires derivatives, but this is just
        # for external processing and AFT will use the private function
        # return ft, dfduh, dfdudh
        
        return fxyn_t
        

    def aft(self, U, w, h, Nt=128, tol=1e-7, max_repeats=2):
        """
        
        Tolerances are ignored since slider representation should converge
        with two repeated cycles (done by default). 
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

        Returns
        -------
        Fnl : Vector of nonlinear forces in frequency domain
        dFnldU : Gradient of nonlinear forces w.r.t. U
        dFnldw : Gradient w.r.t. w

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
        # Conduct AFT in Local Coordinates with JAX
        Uwlocal = np.hstack((np.reshape(Ulocal.T, (Ndnl*Nhc,), 'F'), w))
        
        
        # # If no Grad is needed use:
        # Flocal = _local_aft_jenkins(Uwlocal, pars, u0, tuple(h), Nt, u0h0)[0]
        
        # Case with gradient and local force
        dFdUwlocal, Flocal = _local_aft_grad(Uwlocal, self.uxyn_initialize, 
                                    self.mu, self.meso_gap, self.gaps, 
                                    self.gap_weights, self.Re, self.poisson, 
                                    self.Estar, self.elastic_mod, self.tangent_mod, 
                                    self.delta_y, self.sys, self.Gstar, 
                                    tuple(h), Nt, repeats=max_repeats)
        
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

    
###############################################################################
########## Static Force Functions                                    ##########
###############################################################################
    
# @partial(jax.jit, static_argnums=(7, 8, 9, 10, 11, 12, 13)) 
def _static_force(uxyn, uxyn0, fxy0, unmax, Fm_prev, mu, meso_gap, gaps, gap_weights,
                  Re, Possion, Estar, Emod, Etan, delta_y, Sys, Gstar):
    """
    Calculates the static Rough Contact Force between two surfaces at a single
    quadrature point of the FEM model.
    
    Recommended: mu = 0 for prestress analysis 
        and mu > 0 for linearized eigen analysis after prestress
        
    The rough contact model uses Nasp asperities that can be in contact
    
    Returns traction on element if gap_weights includes the asperity density. 

    Parameters
    ----------
    uxyn : Displacements in x,y,n directions at point. normal is positive into 
            surface
    uxyn : Displacements in x,y,n directions at previous instant
    fxy0 : Previous tangential forces for each asperity (Nasp, 2)
    unmax : Maximum previous displacement in normal direction at point (not 
                                                        including this step)
    Fm_prev : Previous maximum asperity contact forces for each asperity 
              size: (Nasp,)
    mu : Scalar friction coefficient
    meso_gap : Mesoscale topology gap of the element used to adjust 
                uxyn and unmax, scalar
    gaps : List of initial asperity gaps (w/o mesoscale topology), size (Nasp,)
    gap_weights : weights for integration over asperity gaps, size (Nasp,) equal to:
                 (quadrature weights)*(probability distribution)*(asperity density)
    Re : Effective radius of asperities in contact (half of real radius, see paper)
    Possion : Poisson's ratio of asperities
    Estar : Effective elastic modulus of contact
    Emod : Linear elastic modulus
    Etan : Hardening modulus for plasticity regime
    delta_y : Displacement of asperities that causes yielding to occur
    Sys : Yield strength of asperities
    Gstar : Combined shear modulus used for Mindlin tangential contact stiffness

    Returns
    -------
    fxyn : Contact forces or tractions in x,y,n directions. Exact definition
            depends on choice of gap_weights for the integration
    aux : containts fields of: (fxyn, Fm_prev, deltabar, Rebar)
            fxyn - same as above, duplicated for autodiff output
            Fm_prev - updated maximum normal forces for each asperity
            deltabar - permanent deformation of each asperity
            Rebar - updated radius due to permanent deformation of each asperity

    """
    
    # Recover deltam for each asperity based on previous maximum displacement
    deltam = unmax - meso_gap - gaps
    
    # Calculate normal displacement of each aspertiy
    un = uxyn[-1] - meso_gap - gaps

    fn_curr, a, deltabar, Rebar = asp_funs._normal_asperity_general(un, deltam, Fm_prev, 
                                 Re, Possion, Estar, Emod, Etan, delta_y, Sys)
    
    fxy_curr = asp_funs._tangential_asperity(uxyn[:2], uxyn0[:2], fxy0, 
                                             fn_curr, a, Gstar, mu)
    
    # Update normal history variables
    Fm_prev = jnp.maximum(fn_curr, Fm_prev)
    
    fxyn = jnp.zeros(3)
    fxyn = fxyn.at[-1].set(fn_curr @ gap_weights)
    fxyn = fxyn.at[:2].set(gap_weights @ fxy_curr)
    
    # Extra outputs
    #   includes force so have the undifferentiated force when calling jax.jacfwd
    aux = (fxyn, Fm_prev, fxy_curr, deltabar, Rebar, a)
    
    return fxyn, aux

@partial(jax.jit, static_argnums=tuple(range(9, 17))) 
def _static_force_grad(uxyn, uxyn0, fxy0, unmax, Fm_prev, mu, meso_gap, gaps, gap_weights,
                       Re, Possion, Estar, Emod, Etan, delta_y, Sys, Gstar):
    """
    Returns Jacobian, Force, and Aux Data from "_static_force"
    
    See "_static_force" for documentation of inputs/outputs

    """
    
    jax_diff_fun = jax.jacfwd(_static_force, has_aux=True) 
    
    J, aux = jax_diff_fun(uxyn, uxyn0, fxy0, unmax, Fm_prev, mu, meso_gap, gaps, gap_weights,
                           Re, Possion, Estar, Emod, Etan, delta_y, Sys, Gstar)
    
    F = aux[0]
    
    return J, F, aux


    
###############################################################################
########## Harmonic Force History Functions                          ##########
###############################################################################


def _local_loop_body(ind, history, unlt, mu, meso_gap, gaps, gap_weights,
                       Re, Estar, Gstar):
    """
    Calculation of total rough contact forces for a given instant in a time 
    series. Formatted to allow for calling jax.lax.fori_loop
    
    In general, considering the loop for Nt time points. 
    Considers Nasp=gaps.shape[0] asperities in contact for the element.

    Parameters
    ----------
    ind : Time index to be calculated for this iteration
    history : Tuple of variables to use for history includes in order: 
                fxyn_t : contact forces/tractions totalled over all asperities 
                        for all time instants (Nt, 3)
                uxyn0 : previous set of displacements (in general unlt[ind-1, :])
                            required so that initialization can be user defined
                fxy0 : tangential force for each asperity at the previous 
                        instant shape (Nasp, 2)
                deltam : Maximum normal displacement of each asperity for any 
                        time (Nasp,)
                Fm : Forces in the normal direction for instant of maximum 
                        displacement
    unlt : Time series of displacements for nonlinear force evaluations
            Size (Nt, 3)
    mu, meso_gap, gaps, gap_weights, Re, Estar, Gstar : 
            See RoughContactFriction Class Documentation

    Returns
    -------
    history : Same as input, but updated for the instant ind

    """
    
    fxyn_t, uxyn0, fxy0, deltam, Fm = history
    
    # Asperity force calculation
    
    # Normal Direction forces, exclusively on the plastic unloading curve
    # Elastic Unloading after Plasticity
    un = unlt[ind, 2] - meso_gap - gaps
    
    fn_curr, a, deltabar, Rebar = asp_funs._normal_asperity_unloading(un, 
                                                        deltam, Fm, Re, Estar)
    
    # Tangential Forces
    fxy_curr = asp_funs._tangential_asperity(unlt[ind, :2], uxyn0[:2], fxy0, 
                                             fn_curr, a, Gstar, mu)
    
    # Integrate asperity forces into total element in contact forces
    fxyn_t = fxyn_t.at[ind, :2].set(gap_weights @ fxy_curr)
    fxyn_t = fxyn_t.at[ind, -1].set(fn_curr @ gap_weights)
    
    history = (fxyn_t, unlt[ind, :], fxy_curr, deltam, Fm)
    
    return history
    

@partial(jax.jit, static_argnums=tuple(range(6, 15))) 
def _local_force_history(unlt, unlth0, mu, meso_gap, gaps, gap_weights,
                         Re, Possion, Estar, Emod, Etan, delta_y, Sys, Gstar, 
                         repeats=2):
    """
    Calculates the steady-state displacement history for a set of displacements
    
    NOTES:
        This function throws an error if not JIT compiled because the loop body
        would be dependent on a non-static argument in that case.

    Parameters
    ----------
    unlt : Time history of displacements to evaluate cycles over - size (Nt, 3)
    unlth0 : Displacements to initialize the first loop iteration to. 
             For asperities that do not slip, this influences the static force.
    mu, meso_gap, gaps, gap_weights, Re, Possion, Estar, Emod, Etan, delta_y, 
    Sys, Gstar : 
        See RoughContactFriction Class Documentation
    repeats : The number of repeated cycles the nonlinear forces should be 
                evaluated for to obtain steady-state forces. Default is 2

    Returns
    -------
    fxyn_t : History of total forces (Nt, 3)

    """
    
    Nt = unlt.shape[0]
    
    ###########
    # Initialize the Normal Direction Force Calculation
    unmax = unlt[:, -1].max()
    
    Fm_prev = jnp.zeros_like(gap_weights)
    
    # Recover deltam for each asperity based on previous maximum displacement
    deltam = 0 - meso_gap - gaps
    
    # Calculate normal displacement of each aspertiy
    unmax_asp = unmax - meso_gap - gaps
    
    fn, a, deltabar, Rebar = asp_funs._normal_asperity_general(unmax_asp, deltam, Fm_prev, 
                                 Re, Possion, Estar, Emod, Etan, delta_y, Sys)
    
    ###########
    # Generate a history tuple for use in the function
    fxy0 = np.zeros((gap_weights.shape[0], 2)) # previous instant of asperity forces
    uxyn0 = unlth0*1.0 # Previous instant of displacements (force to be double)
    fxyn_t = jnp.zeros((Nt, 3)) # History of total contact forces (summed over asperities)
    # deltam = unmax_asp # Maximum normal displacement at each element
    # fm = fn # Normal asperity forces for maximum normal displacement
    history = (fxyn_t, uxyn0, fxy0, unmax_asp, fn)
    
    ###########
    # Loop body function
    loop_fun = lambda i,hist : _local_loop_body(i, hist, unlt, mu, meso_gap, 
                                            gaps, gap_weights, Re, Estar, Gstar)
    
    # import pdb; pdb.set_trace()
    
    ###########
    # Do a loop over the set of Nt samples, repeating to get convergence 
    # to steady-state forces
    for i in range(repeats):
        history = jax.lax.fori_loop(0, Nt, loop_fun, history)
    
    ###########
    # Extract the steady-state forces
    fxyn_t = history[0]

    return fxyn_t


###############################################################################
########## AFT Functions                                             ##########
###############################################################################


# @partial(jax.jit, static_argnums=tuple(range(6, 15))) 
def _local_aft(Uwlocal, unlth0, mu, meso_gap, gaps, gap_weights,
                         Re, Possion, Estar, Emod, Etan, delta_y, Sys, Gstar, 
                         htuple, Nt, repeats=2):
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
    
    
    ########################################
    #### Evaluate nonlinear forces
    
    ft = _local_force_history(unlt, unlth0, mu, meso_gap, gaps, gap_weights,
                             Re, Possion, Estar, Emod, Etan, delta_y, Sys, Gstar, 
                             repeats)
    
    # Convert back into frequency domain
    Flocal = jhutils.get_fourier_coeff(htuple, ft)
    
    # Flatten back to a 1D array
    Flocal = jnp.reshape(Flocal.T, (-1,), 'F')
    
    return Flocal,Flocal


@partial(jax.jit, static_argnums=tuple(range(6, 17))) 
def _local_aft_grad(Uwlocal, unlth0, mu, meso_gap, gaps, gap_weights,
                         Re, Possion, Estar, Emod, Etan, delta_y, Sys, Gstar, 
                         htuple, Nt, repeats=2):
    """
    Gradient of _local_aft - see _local_aft for documentation. 

    """
    
    J,F = jax.jacfwd(_local_aft, has_aux=True)(Uwlocal, unlth0, mu, meso_gap, 
                                               gaps, gap_weights, Re, Possion, 
                                               Estar, Emod, Etan, delta_y, Sys, 
                                               Gstar, htuple, Nt, repeats)    
    return J,F















