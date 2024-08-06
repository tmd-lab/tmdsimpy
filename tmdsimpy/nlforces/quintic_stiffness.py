import numpy as np
from ..nlforces.nonlinear_force import InstantaneousForce

class QuinticForce(InstantaneousForce):
    """
    Quintic nonlinear force (force proportional to displacement to power 5).

    Parameters
    ----------
    Q : (Nnl, N) numpy.ndarray
        Matrix tranform from the `N` degrees of freedom (DOFs) of the system
        to the `Nnl` local nonlinear DOFs.
    T : (N, Nnl) numpy.ndarray
        Matrix tranform from the local `Nnl` forces to the `N` global DOFs.
    kalpha : (Nnl,) numpy.ndarray
        Coefficient for cubic stiffness for each nonlinear DOF.

    """
    
    def __init__(self, Q, T, kalpha):

        self.Q = Q
        self.T = T
        self.kalpha = kalpha

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
        
        fnl = self.kalpha * (unl**5)
        
        F = self.T @ fnl
        
        dFdX = self.T @ np.diag(5 * self.kalpha * (unl**4)) @ self.Q
        
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
                
        ft = self.kalpha * (unlt**5)
        dfdu = (5 * self.kalpha) * (unlt**4)
        dfdud = np.zeros_like(unlt)
        
        return ft, dfdu, dfdud