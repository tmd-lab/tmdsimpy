import numpy as np
from ..nlforces.nonlinear_force import InstantaneousForce

class CubicDamping(InstantaneousForce):
    """
    Cubic Damping Nonlinearity
    """
    
    def __init__(self, Q, T, calpha):
        """
        Initialize a nonlinear force model

        Parameters
        ----------
        Q : Transformation matrix from system DOFs (n) to nonlinear DOFs (Nnl), 
            Nnl x n
        T : Transformation matrix from local nonlinear forces to global 
            nonlinear forces, n x Nnl
        calpha : Coefficient for cubic damping for each nonlinear DOF, 1D size Ndnl

        """
        self.Q = Q
        self.T = T
        self.calpha = calpha
    
    def force(self, V):
        
        unldot = self.Q @ V
        
        fnl = self.calpha * (unldot**3)
        
        F = self.T @ fnl
        
        dFdV = self.T @ np.diag(3 * self.calpha * (unldot**2)) @ self.Q
        
        return F, dFdV
    
    def local_force_history(self, unlt, unltdot):
                
        ft = self.calpha * (unltdot**3)
        dfdu = np.zeros_like(unlt)
        dfdud = (3 * self.calpha) * (unltdot**2)
        
        return ft, dfdu, dfdud