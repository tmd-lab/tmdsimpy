import numpy as np
from ..nlforces.nonlinear_force import InstantaneousForce
from .. import harmonic_utils as hutils

class GenPolyForce(InstantaneousForce):
    """
    Cubic and Quadratic Polynomial Nonlinearity (Used for geometric nonlinearity)

    Parameters
    ----------
    Q : (Nd,n) numpy.ndarray
        Transformation matrix from system DOFs (n) to nonlinear DOFs (Nd), Nd x n
    T : (n, Nd) numpy.ndarray
        Transformation matrix from local nonlinear DOFs (Nd) to stsyem dofs
        nonlinear forces, n x Nd
    Emat: (Nd,cnl) numpy.ndarray
        Matrix to transform modal coordinates into modal nonlinear force. 
        cnl is swum of all cubic and quadratic nonlinear forces associated with Nd
    qq : (cnl,Nd) numpy.ndarray
        Matrix corresponding to exponnent of modal coordinates. 
        

    """
    
    def __init__(self, Q, T, Emat, qq):
        self.Q = Q
        self.T = T
        self.Emat = Emat
        self.qq = qq
        
        self.qd = np.zeros((self.qq.shape[1],self.qq.shape[0],self.qq.shape[1]))
        
        
        for i in range(self.qq.shape[1]):
            self.qd[i,:,:] = self.qq 
            self.qd[i,:,i] -= self.qq[:,i] != 0
        
    def force(self, X):
        """
        Function evaluating nonlinear force and force gradient wrt displacement)
        
        Inputs
        ----------
        X : (Nd,) numpy.ndarray
            displacement vector
        
        Outputs
        -------------
        F : (Nd,) numpy.ndarray
            Nonlinear force
        dFdX: (Nd,Nd) numpy.ndarray  
            Force gradient with respect to displacement

        """
        
        unl = self.Q @ X
        
        fnl = self.Emat @ np.prod(unl ** self.qq, axis=1)
        
        F = self.T @ fnl
        
        dFdX =  self.T @ self.Emat @ (self.qq*np.prod(unl ** self.qd, axis=2).T) @ self.Q
        
        return F, dFdX
    
    def local_force_history(self, unlt, unltdot):
        """
        Function evaluating nonlinear force and force gradient wrt displacement 
        from time history)
        
        Inputs
        ----------
        unlt : (Nt,Nd) numpy.ndarray
            displacement time history
        
        Outputs
        -------------
        F : (Nt, Nd, Nd,) numpy.ndarray
            Nonlinear force history
        dFdX: (Nd,Nd) numpy.ndarray  
            Time history of force gradient with respect to displacement

        """
        
        ## This function is modified to take multiple unlt rows from aft
                

        dfdu = np.zeros((unlt.shape[0],unlt.shape[1],unlt.shape[1]))
        
        ft= np.prod(unlt.reshape(unlt.shape[0],1,unlt.shape[1]) ** self.qq, axis=2) @ (self.Emat).T 
        # Size of ft (Nt,Nd)
   
        for k_row in range(unlt.shape[0]):
             u1=unlt[k_row,:]
             
             dfdu[k_row,:,:]= self.Emat @ (self.qq*np.prod(u1 ** self.qd, axis=2).T)
             # Size of dffu(Nt,Nd*Nd) where Jacobian is stacked [row1 row2 row2 ...rowNd]  
               
        dfdud = np.zeros_like(dfdu)

        return ft, dfdu, dfdud
    


    def aft(self, U, w, h, Nt=128, tol=1e-7, calc_grad=True):
        """
        Alternating Frequency Time Domain Method for calculating the Fourier
        Coefficients of instantaneous forces. 
        Notation: The variable names should be cleaned up. Currently U and Fnl
        correspond to global DOFs while Unl and F correspond to reduced DOFs 
        that the nonlinear force is evaluated at. 
         
        WARNING: Needs further verification for cases using multiple nonlinear 
        displacements and or nonlinear output forces.
         
        Parameters
        ----------
        U : Global DOFs harmonic coefficients, all 0th, then 1st cos, etc, 
        Nhc*nd x 1
        w : Frequency, scalar
        h : List of harmonics that are considered, zeroth must be first
        Nt : Number of time points to evaluate at. 
         The default is 128.
        tol : Convergence criteria, irrelevant for instantaneous forcing.
        The default is 1e-7.
        calc_grad : boolean
        This argument is ignored for now. It is included for 
        compatability of interface. Future work could use it to
        avoid calculating dFnldU.
        The default is True
        
        Returns
        -------
        Fnl : (Nhc,) numpy.ndarray
            Fourier coefficients of nonlinear force
        dFnldU : (Nhc,Nhc) numpy.ndarray
            Fourier coefficients of gradient
        """       
        
        Fnl = np.zeros_like(U)
        dFnldU = np.zeros((U.shape[0], U.shape[0]))
        dFnldw = np.zeros_like(U)
         
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components        
        Unl = (self.Q @ np.reshape(U, (self.Q.shape[1], Nhc), 'F')).T
         
        # Number of Nonlinear DOFs
        Ndnl = self.Q.shape[0]
         
        # Nonlinear displacements, velocities in time
        unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl
        unltdot = w*hutils.time_series_deriv(Nt, h, Unl, 1) # Nt x Ndnl
         
        # Forces
        ft, dfdu, dfdud = self.local_force_history(unlt, unltdot)
         
        # assert dfdud.sum() == 0, 'Gradient for instantaneous velocity -> force is not implemented'
        F = hutils.get_fourier_coeff(h, ft)
         
        # Gradient Calculation
        cst = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 0)
        
        # Derivative of the time series of forces w.r.t harmonic coefficients
        dfduh = ( np.reshape(dfdu,(Nt,Ndnl,Ndnl,1)))*np.reshape(cst, (Nt,1,1,Nhc), 'F')
        
        # # Velocity Dependency
        # sct = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 1)
        
        # dfduh = dfduh + np.reshape(( np.reshape(dfdud,(Nt,Ndnl,Ndnl,1)))*np.reshape(sct, (Nt,1,1,Nhc), 'F'),\
        #  (Nt,Ndnl*Ndnl,Nhc), 'F') #dfdud is zero (No chnage in dfduh) 
        # #for nonzero dfduh terms we might have to take another derivatives also
        
        
        # Derivative of the time series of forces w.r.t harmonic coefficients
        dfduh = np.reshape( dfduh, (Nt,Ndnl*Ndnl*Nhc), 'F')
        
        # Derivative of Harmonic Coefs w.r.t. Harmonic Coefs
        dFdUnl = np.reshape(hutils.get_fourier_coeff(h, dfduh), (Nhc,Ndnl,Ndnl,Nhc), 'F')
        
        # Flatten dFdUnl to a 2D representation
        J = np.zeros((Nhc*Ndnl, Nhc*Ndnl))
        for di in range(Ndnl):
            for dj in range(Ndnl):
                J[di::Ndnl, dj::Ndnl] = dFdUnl[:, di, dj, :]
        
        Fnl = np.reshape(self.T @ F.T, (U.shape[0],), 'F')
        dFnldU = np.kron(np.eye(Nhc), self.T) @ J @ np.kron(np.eye(Nhc), self.Q)
        dFnldw = np.reshape(dFnldw, (U.shape[0],), 'F')
      
        return Fnl, dFnldU, dFnldw