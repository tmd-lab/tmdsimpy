import numpy as np
from ..nlforces.nonlinear_force import InstantaneousForce

class ConservativeIwanBB(InstantaneousForce):
    """
    4-parameter Iwan model backbone curve as a conservative (softening) 
    nonlinearity
    
    Parameters
    ----------
    Q : (Nnl, N) numpy.ndarray
        Matrix tranform from the `N` degrees of freedom (DOFs) of the system 
        to the `Nnl` local nonlinear DOFs.
    T : (N, Nnl) numpy.ndarray
        Matrix tranform from the local `Nnl` forces to the `N` global DOFs.
    kt : float
        Tangential Stiffness.
    Fs : float
        Slip force.
    chi : float
        Controls microslip damping slope. Recommended to have `chi > -1`.
        Smaller values of `chi` may not work.
    beta : float, positive
        Controls discontinuity at beginning of macroslip (zero is smooth).
    
    Notes
    -----
    
    Parameterization of the backbone (initial loading curve) of the 4-parameter
    Iwan model.
    This is a standard equation in joints community and the exact equations
    are taken from [1]_. Original work for the 4-paramater Iwan model is
    [2]_.
    
    This is not tested for multiple simultaneous elements.
    
    
    References
    ----------
    .. [1]
        Porter, J.H., N.N. Balaji, C.R. Little, M.R.W. Brake, 2022. A
        quantitative assessment of the model form error of friction models
        across different interface representations for jointed structures.
        Mechanical Systems and Signal Processing.
    .. [2]
       Segalman, D.J., 2005. A Four-Parameter Iwan Model for Lap-Type
       Joints. J. Appl. Mech 72, 752–760.
    
    """
    
    def __init__(self, Q, T, kt, Fs, chi, beta):
        
        self.Q = Q
        self.T = T
        self.kt = kt*1.0
        self.Fs = Fs*1.0
        self.chi = chi*1.0
        self.beta = beta*1.0
        
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
        
        # Parameters to change parameterization
        # max slider
        phi_max = self.Fs*(1+self.beta)/(self.kt*(self.beta+(self.chi+1)/(self.chi+2)))
        
        # coefficient to pull out that simplifies later expressions
        coef_E = (self.kt*(self.kt*(self.beta+(self.chi+1)/(self.chi+2))\
                           /(self.Fs*(1+self.beta)))**(1+self.chi))\
                           /((1+self.beta)*(self.chi+2))
        
        # Evaluate force then add sign back
        sunl = np.sign(unl)
        unl = sunl * unl
        
        # Evaluate Local Nonlinear Force
        fnl = ((self.kt*unl - coef_E*unl**(2+self.chi))*(unl<phi_max) \
               + (self.Fs)*(unl>=phi_max))*sunl
        
        F = self.T @ fnl
        
        # local Gradient
        dfnldu = (self.kt-coef_E*(2+self.chi)*unl**(1+self.chi))*(unl<phi_max)
        
        dFdX = self.T @ np.diag(dfnldu) @ self.Q
        
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
        
        dfdud = np.zeros_like(unlt)
        
        # Parameters to change parameterization
        # max slider
        phi_max = self.Fs*(1+self.beta)/(self.kt*(self.beta+(self.chi+1)/(self.chi+2)))
        
        # coefficient to pull out that simplifies later expressions
        coef_E = (self.kt*(self.kt*(self.beta+(self.chi+1)/(self.chi+2))\
                           /(self.Fs*(1+self.beta)))**(1+self.chi))\
                           /((1+self.beta)*(self.chi+2))
        
        # Evaluate force then add sign back
        sunl = np.sign(unlt)
        unlt = sunl * unlt
        
        # Evaluate Local Nonlinear Force
        ft = ((self.kt*unlt - coef_E*unlt**(2+self.chi))*(unlt<phi_max) \
               + (self.Fs)*(unlt>=phi_max))*sunl
            
        dfdu = (self.kt-coef_E*(2+self.chi)*unlt**(1+self.chi))*(unlt<phi_max)

        return ft, dfdu, dfdud