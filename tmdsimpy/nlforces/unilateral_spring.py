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
    T : (N, Nnl) numpy.ndarray
        Matrix tranform from the local `Nnl` forces to the `N` global DOFs.
    k : float or (Nnl,) numpy.ndarray
        Stiffness coefficient
    Npreload : float or (Nnl,) numpy.ndarray, optional
        The minimum force is `-1*Npreload` for displacements less than `delta`.
        The default is 0.
    delta : float or (Nnl,) numpy.ndarray, optional
        Offset of the elbow in the force from being at zero displacement.
        The default is 0.
    
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
        Calculate global nonlinear forces for some global displacement vector.

        Parameters
        ----------
        X : (N,) numpy.ndarray
            Global displacements.

        Returns
        -------
        F : (N,) numpy.ndarray
            Global nonlinear force.
        dFdX : (N,N) numpy.ndarray
            Derivative of `F` with respect to `X`.

        """
        
        unl = self.Q @ X 
        
        fnl = np.maximum(self.k*(unl - self.delta) - self.Npreload, -self.Npreload)
        
        F = self.T @ fnl

        mask = np.greater(fnl, -self.Npreload)

        dFdX = self.T @ np.diag(mask*self.k) @ self.Q
        
        return F, dFdX
    
    def local_force_history(self, unlt, unltdot):
        """
        Evaluates the local nonlinear forces based on local nonlinear 
        displacements for a time series.
        
        Parameters
        ----------
        unl : (Nt,Nnl) numpy.ndarray
            Local displacements, rows are different time instants and
            columns are different displacement DOFs.
        unldot : (Nt,Nnl) numpy.ndarray
            Local velocities, rows are different time instants and
            columns are different displacement DOFs.
        
        Returns
        -------
        ft : (Nt,Nnl) numpy.ndarray
            Local nonlinear forces, rows are different time instants and
            columns are different local force DOFs.
        dfdu : (Nt,Nnl) numpy.ndarray
            Derivative of forces of `ft` with resepct to displacements `unl`.
            Each index `i, j` is the derivative `ft[i, j]` with respect
            to `unl[i, j]`.
        dfdud : (Nt,Nnl) numpy.ndarray
            Derivative of forces of `ft` with resepct to velocities `unltdot`.
            Each index `i, j` is the derivative `ft[i, j]` with respect
            to `unltdot[i, j]`.
        
        Notes
        -----
        
        Since the nonlinear forces are dependent on only one of the local DOFs, 
        the derivative matrix need not be three dimensional to contain all
        necessary information.

        """
                
        ft = np.maximum(self.k*(unlt - self.delta) - self.Npreload, \
                        -self.Npreload)
        
        mask = np.greater(ft, -self.Npreload)
        dfdu = self.k*mask
            
        dfdud = np.zeros_like(dfdu)

        return ft, dfdu, dfdud
    
    