import numpy as np
from .nonlinear_force import HystereticForce

# Harmonic Functions for AFT
from .. import harmonic_utils as hutils


class JenkinsForce(HystereticForce):
    """
    Jenkins Slider Element Nonlinearity
    """
    
    def __init__(self, Q, T, kt, Fs):
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

        """
        
        self.Q = Q
        self.T = T
        self.kt = kt
        self.Fs = Fs
    
    
    def init_history(self, unlth0, h=np.array([0])):
        """
        Initialize History

        Parameters
        ----------
        unlt0 : 0th Harmonic Displacement as a reference configuration. 
                Not required to use as slider reference, but makes a good 
                invariant choice. Could lead to non-unique solutions though.
        h : List of harmonics. Only use default if not interested in harmonic 
            information and derivatives.
            The default is [0].

        Returns
        -------
        None.

        """
        self.up = unlth0
        self.fp = 0
        self.dupduh = np.zeros((hutils.Nhc(h)))
        
        self.dupduh[0] = 1 # Base slider position taken as zeroth harmonic 
        
        self.dfpduh = np.zeros((1,1,hutils.Nhc(h)))
        
        return
    
    def force(self, X):
        
        unl = self.Q @ X
        
        fnl = self.kt*(unl - self.up) + self.fp
        
        dfnldunl = self.kt
        
        if np.abs(fnl) > self.Fs:
            fnl = np.sign(fnl)*self.Fs
            dfnldunl = 0.0
                    
        F = self.T @ fnl
        
        dFdX = self.T @ dfnldunl @ self.Q
        
        return F, dFdX
    
    def instant_force(self, unl, unldot, update_prev=False):
        """
        Gives the force for a given loading instant using the previous history
        Then updates the stored history. 

        Parameters
        ----------
        unl : Displacements
        unldot : Velocities

        Returns
        -------
        fnl : Nonlinear Forces
        dfnldunl : derivative of forces w.r.t. current displacements
        dfnldup : derivative of forces w.r.t. previous displacements
        dfnldfp : derivative of forces w.r.t. previous forces

        """
                
        # Stuck Force
        fnl = self.kt*(unl - self.up) + self.fp
        dfnldunl = self.kt
        dfnldup = -self.kt
        dfnldfp = 1.0
        
        # Slipping Force
        if np.abs(fnl) > self.Fs:
            fnl = np.sign(fnl)*self.Fs
            dfnldunl = 0.0
            dfnldup = 0.0
            dfnldfp = 0.0
            
        if update_prev:
            # Update History
            self.up = unl
            self.fp = fnl
        
        return fnl, dfnldunl, dfnldup, dfnldfp
    
    def instant_force_harmonic(self, unl, unldot, h, cst, update_prev=False):
        """
        For evaluating a force state, uses history initialized in init_history.
        Updates history for the next call based on the current results. 
        
        WARNING: Derivatives including unldot are not calculated.
        
        Parameters
        ----------
        unl : Local displacements for Force
        unltdot : Local velocities for Force
        
        Returns
        -------
        fnl : Local nonlinear forces (Ndnl, Ndnl)
        dfduh : Derivative of forces w.r.t. displacement harmonic coefficients (Ndnl, Ndnl, Nhc)
        dfdudh : Derivative of forces w.r.t. velocities harmonic coefficients (Ndnl, Ndnl, Nhc)

        """
                
        # Number of nonlinear DOFs
        Ndnl = unl.shape[0]
        Nhc = hutils.Nhc(h)
        
        dfduh = np.zeros((Ndnl, Ndnl, Nhc))
        dfdudh = np.zeros((Ndnl, Ndnl, Nhc))
        
        fnl, dfnldunl, dfnldup, dfnldfp = self.instant_force(unl, unldot, update_prev=update_prev)
        
        fnl = np.atleast_1d(fnl)
        
        dfduh = np.einsum('ij,k->ijk', np.atleast_2d(dfnldunl), cst) \
                + np.einsum('ij,k->ijk', np.atleast_2d(dfnldup), self.dupduh)\
                + np.einsum('ij,jkl->ikl', np.atleast_2d(dfnldfp), self.dfpduh)
        
        # Save derivatives into history for next call. 
        self.dupduh = cst
        self.dfpduh = dfduh
                
        return fnl, dfduh, dfdudh
    
    
    
    
    