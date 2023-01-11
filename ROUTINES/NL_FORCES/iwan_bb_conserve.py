import numpy as np
from nonlinear_force import InstantaneousForce

class ConservativeIwanBB(InstantaneousForce):
    """
    Using the Iwan (4-parameter) backbone curve as a conservative (softening) 
    stiffness
    
    Not Verified for more than one input displacement.
    """
    
    def __init__(self, Q, T, kt, Fs, chi, beta):
        """
        Initialize a nonlinear force model
        
        Parameterization is one of the standard forms used in the joints 
        community. Specifically, copying code from: 
            Porter, J.H., Balaji, N.N., Little, C.R., Brake, M.R.W., 2022. A 
            quantitative assessment of the model form error of friction models 
            across different interface representations for jointed structures. 
            Mechanical Systems and Signal Processing.


        Parameters
        ----------
        Q : Transformation matrix from system DOFs (n) to nonlinear DOFs (Nnl), 
            Nnl x n
        T : Transformation matrix from local nonlinear forces to global 
            nonlinear forces, n x Nnl
        kt : Tangential Stiffness
        Fs : slip force
        chi : controls microslip damping slope
        beta : controls discontinuity at beginning of macroslip (zero is smooth)

        """
        self.Q = Q
        self.T = T
        self.kt = kt*1.0
        self.Fs = Fs*1.0
        self.chi = chi*1.0
        self.beta = beta*1.0
    
        assert self.Q[0] == 1, 'Not tested for simultaneous Iwan elements.'
        
    def force(self, X):
        
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