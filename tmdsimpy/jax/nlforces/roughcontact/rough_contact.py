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
from ...jax import harmonic_utils as jhutils # Jax version of harmonic utils
from ...nlforces.nonlinear_force import NonlinearForce


class RoughContactFriction(NonlinearForce):
    """
    Elastic Dry friction Slider Element Nonlinearity with JAX for automatic 
    differentiation

    The AFT formulation assumes that the spring starts at 0 force
    at zero displacement.    
    """

    def __init__(self, Q, T, ElasticMod, PoissonRatio, Radius, TangentMod, 
                 YieldStress, mu, u0=0):
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
        delta_y1s = (np.pi*C*self.sys/(2*(2*self.Estar)))^2*(2*self.Re); 
        
        self.delta_y = delta_y1s*2
        
        # Call "init_history" here?

    def init_history(self):
        pass
    
    def update_history(self):
        pass
    
    def force(self, X, update_hist=False):
        
        unl = self.Q @ X
        
        # Local Force evaluation based on unl
        fnl = np.zeros(self.T.shape[1])
        dfnldunl = np.zeros((self.T.shape[1], self.Q.shape[0]))
        
        # Convert Back to Physical
        F = self.T @ fnl
        
        dFdX = self.T @ dfnldunl @ self.Q
        
        if update_hist:
            # Call self.update_history()
            pass
        
        return F, dFdX