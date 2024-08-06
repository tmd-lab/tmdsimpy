import numpy as np
from ..nlforces.nonlinear_force import InstantaneousForce

class CubicDamping(InstantaneousForce):
    """
    Cubic damping nonlinear force (proportional to velocity cubed).

    Parameters
    ----------
    Q : (Nnl, N) numpy.ndarray
        Matrix tranform from the `N` degrees of freedom (DOFs) of the system
        to the `Nnl` local nonlinear DOFs.
    T : (N, Nnl) numpy.ndarray
        Matrix tranform from the local `Nnl` forces to the `N` global DOFs.
    calpha : (Nnl,) numpy.ndarray
        Coefficient for cubic damping for each nonlinear DOF.

    Notes
    -----
    
    The `force` method does not match the template of other classes, which take
    input of only the global displacements. Here, the input is only the global
    velocities.
    
    """
    
    def __init__(self, Q, T, calpha):
        
        self.Q = Q
        self.T = T
        self.calpha = calpha
    
    def force(self, V):
        """
        Calculate global nonlinear forces for some global velocity vector.

        Parameters
        ----------
        V : (N,) numpy.ndarray
            Global velocities.

        Returns
        -------
        F : (N,) numpy.ndarray
            Global nonlinear force.
        dFdV : (N,N) numpy.ndarray
            Derivative of `F` with respect to `V`.
            
        Notes
        -----
        This method does not match the exact template of other nonlinear force
        classes that take input here of the global displacements.

        """
        
        unldot = self.Q @ V
        
        fnl = self.calpha * (unldot**3)
        
        F = self.T @ fnl
        
        dFdV = self.T @ np.diag(3 * self.calpha * (unldot**2)) @ self.Q
        
        return F, dFdV
    
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
                
        ft = self.calpha * (unltdot**3)
        dfdu = np.zeros_like(unlt)
        dfdud = (3 * self.calpha) * (unltdot**2)
        
        return ft, dfdu, dfdud