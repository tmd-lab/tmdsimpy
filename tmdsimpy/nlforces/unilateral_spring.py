import numpy as np
from ..nlforces.nonlinear_force import InstantaneousForce
from .. import harmonic_utils as hutils

class UnilateralSpring(InstantaneousForce):
    """
    Unilateral Spring for contact and impact type nonlinear forces 
    (with potential preload)
    
    Force displacement Graph
    
               |            /
    F = 0 _____|___________/____
               |          /
          _____|_________/  (-Npreload)
               |---d ---|
               |
    """
    
    def __init__(self, Q, T, k, Npreload=0, delta=0):
        """
        Initialize a nonlinear force model

        Parameters
        ----------
        Q : Transformation matrix from system DOFs (n) to nonlinear DOFs (Nnl), 
            Nnl x n
        T : Transformation matrix from local nonlinear forces to global 
            nonlinear forces, n x Nnl
        k : stiffness coefficient
        Npreload : the minimum force is -Npreload
        delta : offset of the elbow in the force from being at zero displacement

        """
        
        self.Q = Q
        self.T = T
        self.k = k
        self.Npreload = Npreload
        self.delta = delta
    
    def force(self, X):
        """
        This function is not fully tested and should not be used 

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.

        Returns
        -------
        F : TYPE
            DESCRIPTION.
        dFdX : TYPE
            DESCRIPTION.

        """
        # raise Exception('Test this function before using it')
        unl = self.Q @ X 
        
        fnl = np.maximum(self.k*(unl - self.delta) - self.Npreload, -self.Npreload)
        
        F = self.T @ fnl

        mask = np.greater(fnl, -self.Npreload)
        temp = (mask*self.k)
       
        dFdX = self.T @ np.diag(temp) @ self.Q
        
        return F, dFdX
    
    def local_force_history(self, unlt, unltdot):
                
        ft = np.maximum(self.k*(unlt - self.delta) - self.Npreload, \
                        -self.Npreload)
        
        mask = np.greater(ft, -self.Npreload)
        dfdu = self.k*mask
            
        dfdud = np.zeros_like(dfdu)

        return ft, dfdu, dfdud
    
    