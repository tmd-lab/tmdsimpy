import numpy as np
from ..nlforces.nonlinear_force import InstantaneousForce


class UnilateralSpring(InstantaneousForce):
    """
    Unilateral spring for contact and impact type nonlinear forces 
    (with potential preload)
    
    Parameters
    ----------
    Q : (Nnl, N) numpy.ndarray
        Matrix tranform from the `N` degrees of freedom (DOFs) of the system 
        to the `Nnl` local nonlinear DOFs.
    T : (Nnl, N) numpy.ndarray
        Matrix tranform from the local `Nnl` forces to the `N` global DOFs.
    k : float or (Nnl,) numpy.ndarray
        Stiffness coefficient
    Npreload : float or (Nnl,) numpy.ndarray
        The minimum force is -Npreload for displacements less than `delta`
    delta : float or (Nnl,) numpy.ndarray
        Offset of the elbow in the force from being at zero displacement.
    
    Notes
    -----
    
    Force displacement relationship to calculate the force 
    given a displacement u:
    
    >>> if u > delta: # in contact
    ...     force = k * (u - delta) - Npreload
    ... else: # out of contact
    ...     force = -Npreload
    
    """
    
    """
    F(u) = \begin{cases}
               k * (u - \delta) - N_{preload} & u > \delta (in contact) \\
               - N_{preload} & u \leq \delta (out of contact) \\
           \end{cases}
    """
    
    def __init__(self, Q, T, k, Npreload=0, delta=0):
        
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

        dFdX = self.T @ np.diag(mask*self.k) @ self.Q
        
        return F, dFdX
    
    def local_force_history(self, unlt, unltdot):
                
        ft = np.maximum(self.k*(unlt - self.delta) - self.Npreload, \
                        -self.Npreload)
        
        mask = np.greater(ft, -self.Npreload)
        dfdu = self.k*mask
            
        dfdud = np.zeros_like(dfdu)

        return ft, dfdu, dfdud
    
    