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
from ... import harmonic_utils as hutils
from ....jax import harmonic_utils as jhutils # Jax version of harmonic utils
from ....nlforces.nonlinear_force import NonlinearForce

# Import of functions stored in a different file
from . import _asperity_functions as asp_funs


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
        self.mu = mu
        
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
            
        # Initialize History Variables
        self.init_history()

    def init_history(self):
        
        self.unmax = 0
        self.Fm_prev = np.zeros_like(self.gap_weights)
    
    def update_history(self, uxyn, Fm_curr):
        
        self.unmax = np.maximum(uxyn[-1], self.unmax)
        self.Fm_prev = Fm_curr
        
    
    def force(self, X, update_hist=False):
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

        Returns
        -------
        F : TYPE
            DESCRIPTION.
        dFdX : TYPE
            DESCRIPTION.

        """
        uxyn = self.Q @ X
        
        # Local Force evaluation based on unl
        dfnldunl, fnl, aux = _static_force_grad(uxyn, self.unmax, self.Fm_prev, 
                                            self.mu, self.meso_gap, self.gaps, 
                                            self.gap_weights, self.Re, 
                                            self.poisson, self.Estar, 
                                            self.elastic_mod, 
                                            self.tangent_mod, self.delta_y, 
                                            self.sys)
        
        Fm_curr = aux[1]
        
        # Convert Back to Physical
        F = self.T @ fnl
        
        dFdX = self.T @ dfnldunl @ self.Q
        
        if update_hist:
            self.update_history(uxyn, Fm_curr)
            
        
        return F, dFdX
    
    
    
@partial(jax.jit, static_argnums=(7, 8, 9, 10, 11, 12, 13)) 
def _static_force(uxyn, unmax, Fm_prev, mu, meso_gap, gaps, gap_weights,
                  Re, Possion, Estar, Emod, Etan, delta_y, Sys):
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
    
    # Update normal history variables
    Fm_prev = jnp.maximum(fn_curr, Fm_prev)
    
    
    fxyn = jnp.zeros(3)
    fxyn = fxyn.at[-1].set(fn_curr @ gap_weights)
    
    # Extra outputs
    #   includes force so have the undifferentiated force when calling jax.jacfwd
    aux = (fxyn, Fm_prev, deltabar, Rebar)
    
    return fxyn, aux

@partial(jax.jit, static_argnums=(7, 8, 9, 10, 11, 12, 13)) 
def _static_force_grad(uxyn, unmax, Fm_prev, mu, meso_gap, gaps, gap_weights,
                       Re, Possion, Estar, Emod, Etan, delta_y, Sys):
    """
    Returns Jacobian, Force, and Aux Data from "_static_force"
    
    See "_static_force" for documentation of inputs/outputs

    """
    
    jax_diff_fun = jax.jacfwd(_static_force, has_aux=True) 
    
    J, aux = jax_diff_fun(uxyn, unmax, Fm_prev, mu, meso_gap, gaps, gap_weights,
                           Re, Possion, Estar, Emod, Etan, delta_y, Sys)
    
    F = aux[0]
    
    return J, F, aux