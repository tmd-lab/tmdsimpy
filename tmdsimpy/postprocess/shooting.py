import numpy as np

# from .. import VibrationSystem

def time_stability(vib_sys, XlamP_shoot, Fl, Nt=128):
    """
    Calculate time series and stability of a shooting solution

    Parameters
    ----------
    vib_sys : tmdsimp.VibrationSystem
        Object that was used for calculating the shooting solution that has
        `N` DOFs.
    XlamP_shoot : (M, 2*N+1) numpy.ndarray
        A set of solutions to the shooting equations for `vib_sys`.
        Has `N` displacements, then `N` velocities, then frequency in rad/s.
        Each row is an independent solution point.
    Fl : (2*N) numpy.ndarray
        First `N` entries are cosine forcing at frequency `XlamP_shoot[-1]`.
        The second `N` are the sine forcing terms.
    Nt : int, optional
        Number of time steps to use in shooting calculations.
        The default is 128.

    Returns
    -------
    y_t : (N, Nt+1, M) numpy.ndarray
        Time series of displacements for each solution point.
        Includes the first and last time point, which are numerically identical
        when the solution is converged.
    ydot_t : (N, Nt+1, M) numpy.ndarray
        Time seres of velocities for each solution point.
    stable : (M,) numpy.ndarray of bool
        Is true where the solution is stable
        (maximum eigenvalue of Monodromy matrix is less than 1.0).
    max_eig : (M,) numpy.ndarray
        Maximum eigenvalue of the Monodromy matrix.
    
    See Also
    --------
    tmdsimpy.VibrationSystem.shooting_res :
        Residual function for shooting. This postprocesses results that satisfy
        these equations.
        
    Notes
    -----
    
    For theory about shooting and stability analysis, see Section 3 of [1]_.
    
    There are no formal tests for this function, but there is an example
    for an SDOF Duffing oscillator.
    
    References
    ----------
    
    .. [1] 
        Peeters, M., R. Viguie, G. Sérandour, G. Kerschen, 
        and J. -C. Golinval. 2009. "Nonlinear Normal Modes, Part II: Toward a
        Practical Computation Using Numerical Continuation Techniques."
        Mechanical Systems and Signal Processing, 
        Special Issue: Non-linear Structural Dynamics, 23 (1): 195–216.
        https://doi.org/10.1016/j.ymssp.2008.04.003.
        
    """
    
    max_eig = np.zeros(XlamP_shoot.shape[0])
    
    Ndof = vib_sys.M.shape[0]
    
    y_t = np.zeros((Ndof, Nt+1, XlamP_shoot.shape[0]))
    ydot_t = np.zeros((Ndof, Nt+1, XlamP_shoot.shape[0]))
    
    for i in range(XlamP_shoot.shape[0]):
        
        Uw = XlamP_shoot[i, :]
        
        R,dRdX,dRdw,aux_res = vib_sys.shooting_res(Uw, Fl, Nt=Nt, return_aux=True)
        
        y_t[:, :, i] = aux_res[1]
        ydot_t[:, :, i] = aux_res[2]
        
        monodromy = aux_res[0]
        
        eigvals = np.linalg.eigvals(monodromy)
        
        max_eig[i] = np.max(np.abs(eigvals))
        

    stable = max_eig <= 1.0
    
    return y_t, ydot_t, stable, max_eig