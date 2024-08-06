import numpy as np
from .nonlinear_force import HystereticForce

# Harmonic Functions for AFT
from ..utils import harmonic as hutils


class JenkinsForce(HystereticForce):
    """
    Implementation of the a Jenkins element model for hysteresis in joints.

    Parameters
    ----------
    Q : (Nnl, N) numpy.ndarray
        Matrix tranform from the `N` degrees of freedom (DOFs) of the system 
        to the `Nnl` local nonlinear DOFs.
    T : (N, Nnl) numpy.ndarray
        Matrix tranform from the local `Nnl` forces to the `N` global DOFs.
    kt : float
        Tangential stiffness coefficient.
    Fs : float
        Slip force.

    See Also
    --------
    VectorJenkins :
        Implementation that uses a more efficient vectorization of local forces
        for AFT calculation.

    Notes
    -----
    Implementation is tested for float `kt` and `Fs`, but may work for vectors
    of size `Nnl`.

    Tests also focus on `Nnl == 1`, class may not work for simultaneous
    elements.

    """
    
    def __init__(self, Q, T, kt, Fs):
        
        self.Q = Q
        self.T = T
        self.kt = kt
        self.Fs = Fs
        self.prestress_Fs = 0.0
        self.real_Fs = Fs
    
    
    def set_prestress_mu(self):
        """
        Sets friction slip force to zero while saving initial value in a 
        different variable. Useful for prestress analysis.
        
        Returns
        -------
        None
        
        """
        self.Fs = self.prestress_Fs
        
    def reset_real_mu(self):
        """
        Resets friction slip force to initial value. 
        Useful for after prestress analysis with zero friction coefficient.
        
        Returns
        -------
        None
        """
        
        self.Fs = self.real_Fs
        
    def init_history(self):
        """
        Method to initialize history variables for the hysteretic model.
        
        This consists of setting previous displacements and forces
        to be zero.

        Returns
        -------
        None.

        """
        
        self.up = 0
        self.fp = 0
        
    def init_history_harmonic(self, unlth0, h=np.array([0])):
        """
        Initialize history variables for harmonic (AFT) analysis.

        Parameters
        ----------
        unlth0 : (Nnl,) numpy.ndarray
            Zeroth harmonic contributions to a time series of displacements.
            History displacements are initialized at this value.
        h : numpy.ndarray, sorted
            List of harmonics used in subsequent analysis.
            The default is `numpy.array([0])`.

        Returns
        -------
        None.

        """
        
        self.up = unlth0
        self.fp = 0
        self.dupduh = np.zeros((hutils.Nhc(h)))
        
        self.dupduh[0] = 1 # Base slider position taken as zeroth harmonic 
        
        self.dfpduh = np.zeros((1,1,hutils.Nhc(h)))
        
        return
    
    def update_history(self, unl, fnl):
        """
        Updates hysteretic states by storing displacement and force values.

        Parameters
        ----------
        unl : (Nnl,) numpy.ndarray
            Local nonlinear element displacements to be saved.
        fnl : (Nnl,) numpy.ndarray
            Local nonlinear element forces to be saved.

        Returns
        -------
        None.

        """
        
        self.up = unl
        self.fp = fnl
        
    def force(self, X, update_hist=False):
        """
        Calculate global nonlinear forces for some global displacement vector.

        Parameters
        ----------
        X : (N,) numpy.ndarray
            Global displacements.
        update_hist : bool, optional
            Flag to save displacement and force from the evaluation as history
            variables for subsequent calls to this function.
            The default is False.

        Returns
        -------
        F : (N,) numpy.ndarray
            Global nonlinear force.
        dFdX : (N,N) numpy.ndarray
            Derivative of `F` with respect to `X`.
        
        """
        
        unl = self.Q @ X
        
        fnl = self.kt*(unl - self.up) + self.fp
        
        dfnldunl = self.kt
        
        if np.abs(fnl) > self.Fs:
            fnl = np.sign(fnl)*self.Fs
            dfnldunl = 0.0
                    
        fnl = np.atleast_1d(fnl)
        dfnldunl = np.atleast_2d(dfnldunl)
            
        F = self.T @ fnl
        
        dFdX = self.T @ dfnldunl @ self.Q
        
        if update_hist: 
            self.update_history(unl, fnl)
        
        return F, dFdX
    
    def instant_force(self, unl, unldot, update_prev=False):
        """
        Calculates local force based on local nonlinear displacements.

        Parameters
        ----------
        unl : (Nnl,) numpy.ndarray
            Local nonlinear displacements to evaluate force at.
        unldot : (Nnl,) numpy.ndarray
            Local nonlinear velocities to evaluate force at.
        update_prev : bool, optional
            Flag to store the results of the evaluation for the start of the
            subsequent step. 
            The default is False.

        Returns
        -------
        fnl : float
            Evaluated local nonlinear forces.
        dfnldunl : float
            Derivative of `fnl` with respect to current displacements `unl`.
        dfnldup : float
            Derivative of `fnl` with respect to previous displacements.
        dfnldfp : float
            Derivative of `fnl` with respect to previous forces.

        Notes
        -----
        
        Implementation only allows for a single nonlinear element, thus
        output shapes are reduced to scalar.
        
        """
                
        # Stuck Force
        fnl = self.kt*(unl - self.up) + self.fp
        dfnldunl = self.kt
        dfnldup = -self.kt
        dfnldfp = 1.0
        
        # Slipping Force
        if np.abs(fnl) > self.Fs:
            fnl = np.sign(fnl)*self.Fs
            dfnldunl = 0.0
            dfnldup = 0.0
            dfnldfp = 0.0
            
        if update_prev:
            # Update History
            self.up = unl
            self.fp = fnl
        
        return fnl, dfnldunl, dfnldup, dfnldfp
    
    def instant_force_harmonic(self, unl, unldot, h, cst, update_prev=False):
        """
        Evaluates the force at a instantaneous set of displacement and velocity
        along with harmonic derivatives.
        
        Parameters
        ----------
        unl : (Nnl,) numpy.ndarray
            Local nonlinear displacements to evaluate force at.
        unldot : (Nnl,) numpy.ndarray
            Local nonlinear velocities to evaluate force at.
        h : 1D numpy.ndarray, sorted
            List of harmonics used in subsequent analysis. Corresponds
            to `Nhc` harmonic components.
        cst : (Nhc,) numpy.ndarray
            Evaluation of harmonics without coefficients at the given instant 
            in time. 
            If zeroth harmonic is included, the first entry is 1.0. 
            Beyond that, it is cosine and then sine at the appropriate harmonic
            for the given instant in time then the next harmonic etc.
        update_prev : bool, optional
            Flag to store the results of the evaluation for the start of the
            subsequent step. 
            The default is False.
        
        Returns
        -------
        fnl : (1,) numpy.ndarray
            Local nonlinear forces
        dfduh : (1, 1, Nhc) numpy.ndarray
            Derivative of `fnl` with respect to displacement harmonic
            coefficients.
        dfdudh : (1, 1, Nhc) numpy.ndarray
            Derivative of `fnl` with respect to velocities harmonic
            coefficients.

        Notes
        -----
        
        Starts calculation based on `init_history_harmonic`.
        
        Implementation of this function may work for `Nnl > 1`, but this
        function depends on `instant_force`, which does not support `Nnl > 1`.
        
        """
                
        # Number of nonlinear DOFs
        Ndnl = unl.shape[0]
        Nhc = hutils.Nhc(h)
        
        dfduh = np.zeros((Ndnl, Ndnl, Nhc))
        dfdudh = np.zeros((Ndnl, Ndnl, Nhc))
        
        fnl, dfnldunl, dfnldup, dfnldfp = self.instant_force(unl, unldot, 
                                                     update_prev=update_prev)
        
        fnl = np.atleast_1d(fnl)
        
        dfduh = np.einsum('ij,k->ijk', np.atleast_2d(dfnldunl), cst) \
                + np.einsum('ij,k->ijk', np.atleast_2d(dfnldup), self.dupduh)\
                + np.einsum('ij,jkl->ikl', np.atleast_2d(dfnldfp), self.dfpduh)
        
        # Save derivatives into history for next call. 
        self.dupduh = cst
        self.dfpduh = dfduh
                
        return fnl, dfduh, dfdudh
    
    
    
    
    