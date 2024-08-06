import numpy as np
from ..utils import harmonic as hutils

class NonlinearForce:
    """
    Template class for nonlinear forces. 
    
    This class does not actually implement any nonlinear force.
    
    Parameters
    ----------
    Q : (Nnl, N) numpy.ndarray
        Matrix tranform from the `N` degrees of freedom (DOFs) of the system
        to the `Nnl` local nonlinear DOFs.
    T : (N, Nnl) numpy.ndarray
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
        Template force function for evaluating the nonlinear force for a set
        of global displacements.

        Parameters
        ----------
        X : (N,) numpy.ndarray
            Global displacements

        Returns
        -------
        F : (N,) numpy.ndarray
            Global forces
        dFdX : (N,N) numpy.ndarray
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
    T : (N, Nnl) numpy.ndarray
        Matrix tranform from the local `Nnl` forces to the `N` global DOFs.
    
    See Also
    --------
    tmdsimpy.nlforces.GenPolyForce : 
        An alternative instantaneous force type class where nonlinear forces 
        can depend on all nonlinear DOFs.
    
    Notes
    -----
    
    Each of the local `Nnl` nonlinear forces is a function of just a single
    local displacement with the same index. 
    This does not imply that the nonlinear forces must be a function of a 
    single global DOF. 
    Rather the local displacements can be a linear combination of any of the 
    global DOFs through the mappings `Q` and `T`.
    
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
        unlt : (Nt,Nnl) numpy.ndarray
            Local displacements, rows are different time instants and
            columns are different displacement DOFs.
        unltdot : (Nt,Nnl) numpy.ndarray
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
        
        F = hutils.get_fourier_coeff(h, ft)
        
        # Gradient Calculation
        cst = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 0)
        
        # Derivative of the time series of forces w.r.t harmonic coefficients
        dfduh = np.reshape( 
                   np.atleast_3d(dfdu)*np.reshape(cst, (Nt,1,Nhc), 'F'),
                           (Nt,Ndnl*Nhc), 'F')
            
        # Velocity Dependency
        sct = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 1)
        
        dfduh = dfduh + np.reshape( 
                    np.atleast_3d(dfdud)*np.reshape(w*sct, (Nt,1,Nhc), 'F'),
                    (Nt,Ndnl*Nhc), 'F')
        
        # Derivative of local harmonic force coefs w.r.t. local harmonic 
        # displacements
        dFdUnl = np.reshape(hutils.get_fourier_coeff(h, dfduh), 
                            (Nhc,Ndnl,Nhc), 'F')
        
        # Flatten dFdUnl to a 2D representation
        J = np.zeros((Nhc*Ndnl, Nhc*Ndnl))
        for di in range(Ndnl):
            J[di::Ndnl, di::Ndnl] = dFdUnl[:, di, :]
        
        
        # Convert to Global Coordinates
        Fnl = np.reshape(self.T @ F.T, (U.shape[0],), 'F')
        dFnldU = np.kron(np.eye(Nhc), self.T) @ J \
                 @ np.kron(np.eye(Nhc), self.Q)
        
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
    Template class for hysteretic nonlinear forces. 
    
    This type of forces requires knowledge of the displacement history to 
    calculate forces.
    This class does not actually implement any nonlinear forces.
    
    Parameters
    ----------
    Q : (Nnl, N) numpy.ndarray
        Matrix tranform from the `N` degrees of freedom (DOFs) of the system 
        to the `Nnl` local nonlinear DOFs.
    T : (N, Nnl) numpy.ndarray
        Matrix tranform from the local `Nnl` forces to the `N` global DOFs.
    
    Notes
    -----
    
    Each of the local `Nnl` nonlinear forces can depend on all nonlinear DOFs.
    
    """
    
    
    def nl_force_type(self):
        """
        Method to identify the force type as hysteretic. 
        
        Returns
        -------
        int
            1, indicating hysteretic force type.
        """
        
        return 1
    
    def init_history(self):
        """
        Method to initialize history variables for the hysteretic model.
        
        This generally consists of setting previous displacements and forces
        to be zero.

        Returns
        -------
        None.

        """
        
        pass
    
    def init_history_harmonic(self, unlth0, h):
        """
        Initialize history variables for harmonic (AFT) analysis.

        Parameters
        ----------
        unlth0 : (Nnl,) numpy.ndarray
            Zeroth harmonic contributions to a time series of displacements.
            History displacements can be initialized at this value if desired.
            Some models may choose to initialize harmonic displacements at 
            other values. This is included for compatibility in those cases.
        h : numpy.ndarray, sorted
            List of harmonics used in subsequent analysis.

        Returns
        -------
        None.

        """
        
        pass
    
    def instant_force_harmonic(self, unl, unldot, h, cst):
        """
        Evaulating local forces at an instant in time with harmonic gradients.
                
        Parameters
        ----------
        unl : (Nnl,) numpy.ndarray
            Local displacements for force calculation
        unltdot : (Nnl,) numpy.ndarray
            Local velocities for force calculation
        h : 1D numpy.ndarray, sorted
            List of harmonics used in subsequent analysis. Corresponds
            to `Nhc` harmonic components.
        cst : (Nhc,) numpy.ndarray
            Evaluation of harmonics without coefficients at the given instant 
            in time. 
            If zeroth harmonic is included, the first entry is 1.0. 
            Beyond that, it is cosine and then sine at the appropriate harmonic
            for the given instant in time then the next harmonic etc.
        
        Returns
        -------
        ft : (Nnl,) numpy.ndarray
            Local nonlinear forces
        dfduh : (Nnl,Nnl,Nhc) numpy.ndarray
            Derivative of forces w.r.t. displacement harmonic coefficients.
            First index corresponds to `ft`. Second index corresponds to
            `unl`. Third index corresponds to which of the `Nhc` harmonic 
            components.
        dfdudh : (Nnl,Nnl,Nhc) numpy.ndarray
            Derivative of forces w.r.t. velocities harmonic coefficients.
            First index corresponds to `ft`. Second index corresponds to
            `unl`. Third index corresponds to which of the `Nhc` harmonic 
            components.
            
        
        Notes
        -----
        Uses history initialized in `init_history_harmonic`.
        Updates history for the next call based on the current results.

        """
        
        Ndnl = unl.shape[0]
        Nhc = hutils.Nhc(h)
        
        f = np.zeros((Ndnl,))
        dfduh = np.zeros((Ndnl, Ndnl, Nhc))
        dfdudh = np.zeros((Ndnl, Ndnl, Nhc))
        
        return f, dfduh, dfdudh
    
    def local_force_history(self, unlt, unltdot, h, cst, unlth0, 
                            max_repeats=2, atol=1e-10, rtol=1e-10):
        """
        Evaluate the local forces for steady-state harmonic motion used in AFT.
        (General hysteretic model implementation)
        
        Parameters
        ----------
        unlt : (Nt,Nnl) numpy.ndarray
            Local displacements, rows are different time instants and
            columns are different displacement DOFs.
        unltdot : (Nt,Nnl) numpy.ndarray
            Local velocities, rows are different time instants and
            columns are different displacement DOFs.
        h : 1D numpy.ndarray, sorted
            List of harmonics used in subsequent analysis. Corresponds
            to `Nhc` harmonic components.
        cst : (Nt,Nhc) numpy.ndarray
            Evaluation of each harmonic component (columns) at a given instant
            in time (row = instant in time). These are without any harmonic
            coefficients, so are just cosine and sine evaluations.
        unlth0 : (Nnl,) numpy.ndarray
            Zeroth harmonic contributions to a time series of displacements.
            This is passed to `init_history_harmonic` to initialize model.
        max_repeats : int, optional
            Number of times to repeat the time series to converge the 
            initial state. Two is sufficient for slider models. 
            The default is 2.
        atol : float, optional
            Absolute tolerance on force time series convergence to steady-state
            (final state of cycle).
            The default is 1e-10.
        rtol : float, optional
            Relative tolerance on force time series convergence to steady-state
            (final state of cycle).
            The default is 1e-10.
            
        Returns
        -------
        ft : (Nt,Nnl) numpy.ndarray
            Local nonlinear forces. First index is time instants, second index
            is which local nonlinear force DOF.
        dfduh : (Nt,Nnl,Nnl,Nhc) numpy.ndarray
            Derivative of forces w.r.t. displacement harmonic coefficients.
            First two indices correspond to `ft`. Third index corresponds to
            which local nonlinear displacement. 
            Fourth index corresponds to which of the `Nhc` harmonic 
            components.
        dfdudh : (Nt,Nnl,Nnl,Nhc) numpy.ndarray
            Derivative of forces w.r.t. velocities harmonic coefficients.
            First two indices correspond to `ft`. Third index corresponds to
            which local nonlinear displacement. 
            Fourth index corresponds to which of the `Nhc` harmonic 
            components.
        
        Notes
        -----
        
        WARNING: Derivatives with respect to harmonic velocities are not 
        fully tested or considered so may be incorrect.
        
        Convergence criteria is atol or rtol passes. To require a choice, pass 
        in -1 for the other.

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
        
        while( (its == 0) 
              or (acheck > atol and rcheck > rtol and its < max_repeats) ):
            
            # Time Loop                
            for ti in range(Nt):
                # Update this to immediately save into array without tmps
                fttmp,dfdutmp,dfdudtmp = \
                    self.instant_force_harmonic(unlt[ti, :], unltdot[ti, :], h,
                                                cst[ti, :], update_prev=True)
                
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
    def aft(self, U, w, h, Nt=128, tol=1e-7, max_repeats=2, atol=1e-10, 
            rtol=1e-10):
        """
        Implementation of the alternating frequency-time method to extract 
        harmonic nonlinear force coefficients (hysteretic forces).
        
        Parameters
        ----------
        U : (N*Nhc,) numpy.ndarray
            Displacement harmonic DOFs (global)
        w : float
            Frequency in rad/s. Needed in case there is velocity dependency.
        h : numpy.ndarray, sorted
            List of harmonics. The list corresponds to `Nhc` harmonic 
            components.
        Nt : int power of 2, optional
            Number of time steps used in evaluation. 
            The default is 128.
        tol : float, optional
            This argument is ignored for hysteretic forces. 
            It is included for compatability of interface. 
            The default is 1e-7.
        max_repeats : int, optional
            Number of times to repeat the time series to converge the 
            initial state with `local_force_history`. 
            Two is sufficient for slider models. 
            The default is 2.
        atol : float, optional
            Absolute tolerance on `local_force_history` force convergence 
            to steady-state (final state of cycle).
            The default is 1e-10.
        rtol : float, optional
            Relative tolerance on `local_force_history` force convergence 
            to steady-state (final state of cycle).
            The default is 1e-10.

        Returns
        -------
        Fnl : (N*Nhc,) numpy.ndarray
            Nonlinear hamonic force coefficients
        dFnldU : (N*Nhc,N*Nhc) numpy.ndarray
            Jacobian of `Fnl` with respect to `U`
        dFnldw : (N*Nhc,) numpy.ndarray
            Jacobian of `Fnl` with respect to `w`
        
        Notes
        -----
        
        A `calc_grad` optional argument should be added in the future to allow
        for compatibility with other functions/methods/classes.
        
        WARNING: Needs further verification for cases using multiple nonlinear 
        displacements and or nonlinear output forces.
        
        WARNING: Does not support velocity dependencies in gradient calculation

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
        
        
        cst = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 0)
        unlth0 = Unl[0, :]
        
        # Forces
        ft, dfduh, dfdudh = self.local_force_history(unlt, unltdot, h, cst,
                                                     unlth0,
                                                     max_repeats=max_repeats,
                                                     atol=atol, rtol=rtol)
        
        # assert dfdudh.sum() == 0, 'Gradient for instantaneous velocity '\
        #                               + '-> force is not implemented'
        
        F = hutils.get_fourier_coeff(h, ft)
        
        # Gradient Calculation
        
        # Derivative of the time series of forces w.r.t harmonic coefficients
        dfduh = np.reshape( dfduh, (Nt,Ndnl*Ndnl*Nhc), 'F')
        
        # Derivative of Harmonic Coefs w.r.t. Harmonic Coefs
        dFdUnl = np.reshape(hutils.get_fourier_coeff(h, dfduh), 
                            (Nhc,Ndnl,Ndnl,Nhc), 'F')
        
        # Flatten dFdUnl to a 2D representation
        J = np.zeros((Nhc*Ndnl, Nhc*Ndnl))
        for di in range(Ndnl):
            for dj in range(Ndnl):
                J[di::Ndnl, dj::Ndnl] = dFdUnl[:, di, dj, :]
        
        Fnl = np.reshape(self.T @ F.T, (U.shape[0],), 'F')
        
        dFnldU = np.kron(np.eye(Nhc), self.T) @ J \
                    @ np.kron(np.eye(Nhc), self.Q)
        
        dFnldw = np.reshape(dFnldw, (U.shape[0],), 'F')
        
        return Fnl, dFnldU, dFnldw
