import numpy as np
from nonlinear_force import InstantaneousForce

class CubicForce(InstantaneousForce):
    """
    Cubic Nonlinearity
    """
    
    def __init__(self, Q, T, kalpha):
        """
        Initialize a nonlinear force model

        Parameters
        ----------
        Q : Transformation matrix from system DOFs (n) to nonlinear DOFs (Nnl), 
            Nnl x n
        T : Transformation matrix from local nonlinear forces to global 
            nonlinear forces, n x Nnl
        kalpha : Coefficient for cubic stiffness for each nonlinear DOF, 1D size Ndnl

        """
        self.Q = Q
        self.T = T
        self.kalpha = kalpha
    
    def force(self, X):
        
        unl = self.Q @ X
        
        fnl = self.kalpha * (unl**3)
        
        F = self.T @ fnl
        
        dFdX = self.T @ np.diag(3 * self.kalpha * (unl**2)) @ self.Q
        
        return F, dFdX
    
    def local_force_history(self, unlt, unltdot):
                
        ft = self.kalpha * (unlt**3)
        dfdu = (3 * self.kalpha) * (unlt**2)
        dfdud = np.zeros_like(unlt)
        
        return ft, dfdu, dfdud