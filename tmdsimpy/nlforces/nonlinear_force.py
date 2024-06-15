import numpy as np
from .. import harmonic_utils as hutils

class NonlinearForce:
    """
    Template class for nonlinear forces. 
    
    This class does not actually implement any nonlinear force.
    
    Parameters
    ----------
    Q : (Nnl, N) numpy.ndarray
        Matrix tranform from the `N` degrees of freedom (DOFs) of the system 
        to the `Nnl` local nonlinear DOFs.
    T : (Nnl, N) numpy.ndarray
        Matrix tranform from the local `Nnl` forces to the `N` global DOFs.
    
    """
    
    def __init__(self, Q, T):
        
        self.Q = Q
        self.T = T
        
    def nl_force_type(self):
        """
        Method to return a flag for the force type.
        
        Returns
        -------
        int
            Value indicates the force type.
            0 == Instantaneous Force Type.
            1 == Hysteretic Force Type.
        """
        
        return 0
    
    def force(self, X):
        """
        Evaluate the nonlinear force for a set of global displacements

        Parameters
        ----------
        X : (N,) numpy.ndarray
            Global displacements

        Returns
        -------
        F: (N,) numpy.ndarray
            Global forces
        dFdX: (N,N) numpy.ndarray
            Derivative of global forces with respect to global 
            displacements `X`

        """
        
        F = np.zeros_like(X)
        dFdX = np.zeros((X.shape[0], X.shape[0]))
        
        return F, dFdX
    
    def aft(self, U, w, h, Nt=128, tol=1e-7, calc_grad=True):
        """
        Implementation of the alternating frequency-time method to extract 
        harmonic nonlinear force coefficients

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
            Tolerance on convergence of force at start and end of AFT. 
            Most cases ignore this tolerance, but it is included to ensure
            a compatible interface. 
            The default is 1e-7.
        calc_grad : boolean, optional
            Flag for if to calculate the gradients and return them.
            `Fnl` should always be returned as the first entry of a tuple 
            regardless of if other returned values are calculated.
            Flag is ignored in many cases, but in others can significantly 
            decrease computation time.
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
        Fnl = np.zeros_like(U)
        dFnldU = np.zeros((U.shape[0], U.shape[0]))
        dFnldw = np.zeros_like(U)
        
        return Fnl, dFnldU, dFnldw
    
class InstantaneousForce(NonlinearForce):
    """ 
    Template class for instantaneous nonlinear forces. 
    
    This type of forces can be evaluated at the current state without knowledge
    of history.
    This class does not actually implement any nonlinear forces.
    
    Parameters
    ----------
    Q : (Nnl, N) numpy.ndarray
        Matrix tranform from the `N` degrees of freedom (DOFs) of the system 
        to the `Nnl` local nonlinear DOFs.
    T : (Nnl, N) numpy.ndarray
        Matrix tranform from the local `Nnl` forces to the `N` global DOFs.
    
    Notes
    -----
    
    Each of the local `Nnl` nonlinear forces is a function of just a single
    local displacement with the same index. 
    This does not imply that the nonlinear forces must be a function of a 
    single global DOF. 
    Rather the local displacements can be a linear combination of any of the 
    global DOFs through the mappings `Q` and `T`.
    See `tmdsimpy.nlforces.GenPolyForce` for a instantaneous force type class 
    without this restriction.
    
    """
    
    def nl_force_type(self):
        """
        Method to identify the force type as instantaneous. 
        
        Returns
        -------
        int
            0, indicating instanteous force type.
        """
        
        return 0
    
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
            Each column index `i` is the derivative `ft[:, i]` with respect
            to `unl[:, i]`.
        dfdud : (Nt,Nnl) numpy.ndarray
            Derivative of forces of `ft` with resepct to velocities `unltdot`. 
            Each column index `i` is the derivative `ft[:, i]` with respect
            to `unltdot[:, i]`.
        
        Notes
        -----
        
        This function is used for AFT evaluations of nonlinear forces
        and should support vectorization for necessary forces.
        
        Since the nonlinear forces are dependent on only one of the local DOFs, 
        the derivative matrix need not be three dimensional to contain all
        necessary information.

        """
        ft = np.zeros_like(unlt)
        dfdu = np.zeros_like(unlt)
        dfdud = np.zeros_like(unlt)
        
        return ft, dfdu, dfdud
        
    
    def aft(self, U, w, h, Nt=128, tol=1e-7, calc_grad=True):
        """
        Implementation of the alternating frequency-time method to extract 
        harmonic nonlinear force coefficients (instantaneous forces).
        
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
        dfduh = np.reshape( np.atleast_3d(dfdu)*np.reshape(cst, (Nt,1,Nhc), 'F'),\
                           (Nt,Ndnl*Nhc), 'F')
            
        # Velocity Dependency
        sct = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 1)
        
        dfduh = dfduh + np.reshape( np.atleast_3d(dfdud)*np.reshape(w*sct, (Nt,1,Nhc), 'F'),\
                           (Nt,Ndnl*Nhc), 'F')
        
        # Derivative of local harmonic force coefs w.r.t. local harmonic displacements
        dFdUnl = np.reshape(hutils.get_fourier_coeff(h, dfduh), (Nhc,Ndnl,Nhc), 'F')
        
        # Flatten dFdUnl to a 2D representation
        J = np.zeros((Nhc*Ndnl, Nhc*Ndnl))
        for di in range(Ndnl):
            J[di::Ndnl, di::Ndnl] = dFdUnl[:, di, :]
        
        
        # Convert to Global Coordinates
        Fnl = np.reshape(self.T @ F.T, (U.shape[0],), 'F')
        dFnldU = np.kron(np.eye(Nhc), self.T) @ J @ np.kron(np.eye(Nhc), self.Q)
        
        if not (w == 0):
            # Derivative of force w.r.t. frequency
            dftdw = dfdud * (unltdot/w)
            dFdw = hutils.get_fourier_coeff(h, dftdw)
            dFnldw = np.reshape(self.T @ dFdw.T, (U.shape[0],), 'F')
        
        return Fnl, dFnldU, dFnldw
    
    
    
########################################
# Place Specific Force Nonlinearities in Different Files

class HystereticForce(NonlinearForce):
    """ 
    Class of forces that cannot be evaluated at current state without 
    knowledge of history
    """
    
    """
    __init__ may need to be updated for some hysteretic models. 
    """
    
    
    def nl_force_type(self):
        return 1
    
    def init_history(self):
        pass
    
    def init_history_harmonic(self, unlth0, h):
        pass
    
    def instant_force_harmonic(self, unl, unldot, h, cst):
        """
        For evaluating a force state, uses history initialized in init_history_harmonic.
        Updates history for the next call based on the current results. 
        
        Parameters
        ----------
        unl : Local displacements for Force
        unltdot : Local velocities for Force
        
        Returns
        -------
        ft : Local nonlinear forces
        dfduh : Derivative of forces w.r.t. displacement harmonic coefficients
        dfdudh : Derivative of forces w.r.t. velocities harmonic coefficients

        """
        
        f = np.zeros_like(unl)
        dfduh = np.zeros_like(unl)
        dfdudh = np.zeros_like(unl)
        
        return f, dfduh, dfdudh
    
    def local_force_history(self, unlt, unltdot, h, cst, unlth0, max_repeats=2, atol=1e-10, rtol=1e-10):
        """
        For evaluating local force history, used by AFT. Always does at least 
        two loops to verify convergence.
        
        Convergence criteria is atol or rtol passes. To require a choice, pass 
        in -1 for the other
        
        WARNING: Derivatives with respect to harmonic velocities are not implemented.
        
        Parameters
        ----------
        unlt : Local displacements for Force
        unltdot : Local velocities for Force
        h : list of harmonics
        cst : evaluation of cosine and sine of the harmonics at the times for aft
        unlth0 : 0th harmonic of nonlinear forces for initializing history to start.
        max_repeats: Number of times to repeat the time series to converge the 
             initial state. Two is sufficient for slider models. 
             The default is 2.
        atol: Absolute tolerance on AFT convergence (final state of cycle)
             The default is 1e-10.
        rtol: Relative tolerance on AFT convergence (final state of cycle)
             The default is 1e-10.
        
        Returns
        -------
        ft : Local nonlinear forces
        dfduh : Derivative of forces w.r.t. displacement harmonic coefficients
        dfdudh : Derivative of forces w.r.t. velocities harmonic coefficients

        """
        its = 0
        
        rcheck = 0
        acheck = 0
        
        # Initialize Memory - Assumption on shape is reasonable for mechanical 
        # systems, but may not be perfect.
        Nt,Ndnl = unlt.shape
        Nhc = hutils.Nhc(h)
        
        ft = np.zeros_like(unlt)
        dfduh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
        dfdudh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
        
        # Only initialize before the loop. History is propogated through 
        # repeated loops over the period
        self.init_history_harmonic(unlth0, h)
        fp = self.fp
        
        while( (its == 0) or (acheck > atol and rcheck > rtol and its < max_repeats) ):
            
            # Time Loop                
            for ti in range(Nt):
                # Update this to immediately save into array without tmps
                fttmp,dfdutmp,dfdudtmp = \
                    self.instant_force_harmonic(unlt[ti, :], unltdot[ti, :], \
                                                h, cst[ti, :], update_prev=True)
                
                ft[ti,:] = fttmp
                dfduh[ti,:,:,:] = dfdutmp
                dfdudh[ti,:,:,:] = dfdudtmp
                
            its = its + 1
            
            acheck = np.abs(ft[ti, :] - fp)
            rcheck = np.abs(acheck / (ft[ti, :]+np.finfo(float).eps) )
            
            fp = ft[ti, :]
        
        return ft, dfduh, dfdudh
        
    
    # Write a slightly different AFT than instant since have derivatives output
    # w.r.t. harmonic coefs rather than u.
    def aft(self, U, w, h, Nt=128, tol=1e-7, max_repeats=2, atol=1e-10, rtol=1e-10):
        """
        Alternating Frequency Time Domain Method for calculating the Fourier
        Coefficients of instantaneous forces. 
        Notation: The variable names should be cleaned up. Currently U and Fnl
        correspond to global DOFs while Unl and F correspond to reduced DOFs 
        that the nonlinear force is evaluated at. 
        
        WARNING: Needs further verification for cases using multiple nonlinear 
        displacements and or nonlinear output forces.
        
        WARNING: Does not support velocity dependencies

        Parameters
        ----------
        U : Global DOFs harmonic coefficients, all 0th, then 1st cos, etc, 
            shape: (Nhc*nd,)
        w : Frequency, scalar
        h : List of harmonics that are considered, zeroth must be first
        Nt : Number of time points to evaluate at. 
             The default is 128.
        tol : Convergence criteria, irrelevant for instantaneous forcing.
              The default is 1e-7.

        Returns
        -------
        Fnl : TYPE
            DESCRIPTION.
        dFnldU : TYPE
            DESCRIPTION.
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
        
        
        cst = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 0)
        unlth0 = Unl[0, :]
        
        # Forces
        ft, dfduh, dfdudh = self.local_force_history(unlt, unltdot, h, cst, unlth0, \
                                                   max_repeats=max_repeats, \
                                                   atol=atol, rtol=rtol)
        
        # assert dfdudh.sum() == 0, 'Gradient for instantaneous velocity -> force is not implemented'
        
        F = hutils.get_fourier_coeff(h, ft)
        
        # Gradient Calculation
        
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
        
        