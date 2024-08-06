import numpy as np
from ..nlforces.nonlinear_force import InstantaneousForce
from ..utils import harmonic as hutils

class GenPolyForce(InstantaneousForce):
    """
    Nonlinear force based on general polynomial combinations of nonlinear DOFs.

    Parameters
    ----------
    Q : (Nnl, N) numpy.ndarray
        Matrix tranform from the `N` degrees of freedom (DOFs) of the system
        to the `Nnl` local nonlinear DOFs.
    T : (N, Nnl) numpy.ndarray
        Matrix tranform from the local `Nnl` forces to the `N` global DOFs.
    Emat : (Nnl,cnl) numpy.ndarray
        Stiffness coefficients that multiply polynomial combinations of
        `Nnl` DOFs as defined by `qq`.
        `cnl` is the number of polynomial combinations considered.
    qq : (cnl,Nnl) numpy.ndarray
        Matrix defining the exponents of the nonlinear DOFs for evaluating
        the force.
        Each of the `cnl` rows defines a different polynomial term.
        Within a row, the column `i` is the exponent for nonlinear DOF `i`
        of the `Nnl` nonlinear DOFs.

    Notes
    -----

    This class can commonly be used with cubic and quadratic polynomial
    nonlinearities to simulate geometric nonlinearity.
    Applications can be more general than geometric nonlinearity.

    This class calculates an instantaneous force, but does not exactly match
    the template of `tmdsimpy.nlforces.InstantaneousForce` because here
    each of the force outputs can depend in a nonlinear fashion on other
    nonlinear DOFs.

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
        
        fnl = self.Emat @ np.prod(unl ** self.qq, axis=1)
        
        F = self.T @ fnl
        
        dFdX =  self.T @ self.Emat @ (self.qq*np.prod(unl ** self.qd,
                                                      axis=2).T) @ self.Q
        
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
        dfdu : (Nt,Nnl,Nnl) numpy.ndarray
            Derivative of forces of `ft` with resepct to displacements `unl`.
            Each index `i, j, k` is the derivative `ft[i, j]` with respect
            to `unl[i, k]`.
        dfdud : (Nt,Nnl,Nnl) numpy.ndarray
            Derivative of forces of `ft` with resepct to velocities `unltdot`.
            Each index `i, j, k` is the derivative `ft[i, j]` with respect
            to `unltdot[i, k]`.

        """
        
        ## This function is modified to take multiple unlt rows from aft
        dfdu = np.zeros((unlt.shape[0],unlt.shape[1],unlt.shape[1]))
        
        ft = np.prod(unlt.reshape(unlt.shape[0],1,unlt.shape[1])**self.qq,
                     axis=2) @ self.Emat.T
        # Size of ft (Nt,Nd)
   
        for k_row in range(unlt.shape[0]):
             u1=unlt[k_row,:]
             
             dfdu[k_row,:,:]= self.Emat @ (self.qq*np.prod(u1 ** self.qd,
                                                           axis=2).T)
             # Size of dffu(Nt,Nd*Nd) where Jacobian is stacked 
             # [row1 row2 row2 ...rowNd]  
               
        dfdud = np.zeros_like(dfdu)

        return ft, dfdu, dfdud
    
    def aft(self, U, w, h, Nt=128, tol=1e-7, calc_grad=True):
        """
        Implementation of the alternating frequency-time method to extract 
        harmonic nonlinear force coefficients.
        
        Parameters
        ----------
        U : (N*Nhc,) numpy.ndarray
            displacement harmonic DOFs
        w : float
            Frequency in rad/s. Needed in case there is velocity dependency.
        h : numpy.ndarray, sorted
            List of harmonics. The list corresponds to `Nhc` harmonic 
            components.
        Nt : int power of 2, optional
            Number of time steps used in evaluation. 
            The default is 128.
        tol : float, optional
            This argument is ignored for instantaneous forces. 
            It is included for compatability of interface. 
            The default is 1e-7.
        calc_grad : boolean, optional
            This argument is ignored for instantaneous forces. 
            It is included for compatability of interface. 
            The default is True.

        Returns
        -------
        Fnl : (N*Nhc,) numpy.ndarray
            Nonlinear hamonic force coefficients
        dFnldU : (N*Nhc,N*Nhc) numpy.ndarray
            Jacobian of `Fnl` with respect to `U`
        dFnldw : (N*Nhc,) numpy.ndarray
            Jacobian of `Fnl` with respect to `w`
        
        """
        
        # Notes omitted from docstring about understanding implementation
        """
        Implementation notation could use cleaning up: 
        
        Currently U and Fnl
        correspond to global DOFs while Unl and F correspond to reduced DOFs 
        that the nonlinear force is evaluated at. 
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