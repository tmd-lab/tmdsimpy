import numpy as np
from . import harmonic_utils as hutils
from scipy.integrate import solve_ivp

import warnings

class VibrationSystem:
    """
    Create a vibration system model with several useful residual functions. 
    The system has N degrees of Freedom. 
    
    Parameters
    ----------
    M : (N,N) numpy.ndarray
        Mass Matrix
    K : (N,N) numpy.ndarray
        Stiffness Matrix, n x n
    C : (N,N) numpy.ndarray or None, optional
        Damping Matrix. If ab is provided, that will be used instead to 
        construct a damping matrix. If both are None, then a zero damping 
        matrix will be used.
        Default is None.
    ab : list of length 2 or None, optional
        Mass and Stiffness Proportional Damping Coefficients. If provided, 
        used to recalculate stiffness matrix as
        C = ab[0]*self.M + ab[1]*self.K.
        The default is None.

    See Also
    ----------
    VibrationSystem.set_new_C : 
        sets the damping matrix to a new value for an existing object
    """
    
    def __init__(self, M, K, C=None, ab=None):
        """
        Initialize the linear part of a system
        """
        self.M = M
        self.K = K
        self.ab = None
        
        if not (ab is None):
            if not (C is None):
                print('Ignoring C to overwrite with proportional damping.')
            
            C = ab[0]*self.M + ab[1]*self.K
            self.ab = ab
        
        if C is None:
            self.C = np.zeros_like(M)
        else:
            self.C = C
        
        self.nonlinear_forces = []

    def add_nl_force(self, nlforce):
        """
        Add a nonlinear force to the model
        
        Parameters
        ----------
        nlforce : Nonlinear force to be added to model of type NonlinearForce
        """
        
        self.nonlinear_forces = self.nonlinear_forces + [nlforce]
    
    def init_force_history(self):
        """
        Initialize all hysteretic forces to have initial zero force states. 

        Returns
        -------
        None.

        """
        
        for nlforce in self.nonlinear_forces:
            if nlforce.nl_force_type() == 1: #Hysteretic Force
                nlforce.init_history()
    
    def update_force_history(self, U):
        """
        Update internal nonlinear force history variables so that the current
        state is used for history. Generally called after a static analysis
        such as for prestress.

        Parameters
        ----------
        U : Displacements (generally of prestressed state)

        Returns
        -------
        None.

        """
        
        # Call a force calculation function with update flag set to True
        for nlforce in self.nonlinear_forces:
            if nlforce.nl_force_type() == 1: #Hysteretic Force
                Fnl_curr, dFnldU_curr = nlforce.force(U, update_hist=True)
    
    def set_prestress_mu(self):
        """
        Set friction coefficients to 0.0 for prestress analysis
        """
        
        # Check if nonlinear force has a set prestress mu function and call if 
        # it exists
        for nlforce in self.nonlinear_forces:
            pre_mu = getattr(nlforce, "set_prestress_mu", None)
            if callable(pre_mu):
                nlforce.set_prestress_mu()
                
        return
    
    
    def reset_real_mu(self):
        """
        Reset friction coefficients for relevant nonlinear forces to real 
        (non-zero) values after prestress analysis
        """
        
        # Check if nonlinear force has a reset mu function and call if 
        # it exists
        for nlforce in self.nonlinear_forces:
            pre_mu = getattr(nlforce, "reset_real_mu", None)
            if callable(pre_mu):
                nlforce.reset_real_mu()
                
        return
    
    def set_aft_initialize(self, X):
        """
        Reset friction coefficients for relevant nonlinear forces to real 
        (non-zero) values after prestress analysis
        """
        
        # Check if nonlinear force has a reset mu function and call if 
        # it exists
        for nlforce in self.nonlinear_forces:
            set_aft = getattr(nlforce, "set_aft_initialize", None)
            if callable(set_aft):
                nlforce.set_aft_initialize(X)
            else:
                warnings.warn('Nonlinear force: {} does not have an AFT initialize'.format(nlforce))
                
        return
    
    def set_new_C(self, C=None, ab=None):
        """
        Set the damping matrix to a new value after the VibrationSystem has
        already been created. 

        Parameters
        ----------
        C : (N,N) numpy.ndarray, optional
            New damping matrix. If ab is not None, then ab is used instead. If 
            neither C nor ab is provided, the damping matrix is set to zeros.
            The default is None.
        ab : list of length 2, optional
            If provided, the damping matrix is set to 
            C = ab[0]*self.M + ab[1]*self.K. 
            The default is None.

        Returns
        -------
        None.

        """
        
        if not (ab is None):
            if not (C is None):
                print('Ignoring C to overwrite with proportional damping.')
            
            C = ab[0]*self.M + ab[1]*self.K
            self.ab = ab
        
        if C is None:
            self.C = np.zeros_like(self.M)
        else:
            self.C = C
        
        return
    
    def static_res(self, U, Fstatic):
        """
        Static solution Residual

        Parameters
        ----------
        U : Input displacements for DOFs (Ndof,)
        Fstatic : Externally applied static forces (Ndof,)

        Returns
        -------
        R : Residual for static analysis (Ndof,)
        dRdU : Derivative of residual w.r.t. displacements (Ndof,Ndof)

        """
        
        Fnl = np.zeros_like(U)
        dFnldU = np.zeros((U.shape[0], U.shape[0]))
        
        for nlforce in self.nonlinear_forces:
            Fnl_curr, dFnldU_curr = nlforce.force(U)
            
            Fnl += Fnl_curr
            dFnldU += dFnldU_curr
        
        R = self.K @ U + Fnl - Fstatic
        dRdU = self.K + dFnldU
        
        return R, dRdU
    
    def total_aft(self, U, w, h, Nt=128, aft_tol=1e-7, calc_grad=True):
        """
        Apply Alternating Time Frequency Method to calculate nonlinear force
        coefficients for all nonlinear forces in system
        
        Nhc is the number of harmonic components in that h represents
        can be calculated by harmonic_utils.Nhc(h)
        
        Parameters
        ----------
        U : np.array (n * Nhc,) 
            Harmonic DOFs, displacements, np.hstack((U0, U1c, U1s...)) with 
            harmonics h
        w : double
            Frequency
        h : 1D np.array
            Sorted list of harmonics
        Nt : integer, power of 2
            Number of Time Steps for AFT. The default is 128.
        aft_tol : double
            Tolerance for AFT. The default is 1e-7.
        calc_grad : boolean
            Flag where True indicates that the gradients should be calculated 
            and returned. If False, then returns only (Fnl,) as a tuple. 
            The default is True

        Returns
        -------
        Fnl : np.array (n*Nhc,)
            Nonlinear Force Harmonic Coefficients
        dFnldU : np.array (n*Nhc, n*Nhc)
            Jacobian of Fnl w.r.t. Harmonic DOFs
        dFnldw : np.array (n*Nhc,)
            Derivative of Fnl w.r.t. frequency
        
        """
        
        # Counting:
        Nhc = hutils.Nhc(h) # Number of Harmonic Components
        Ndof = self.M.shape[0]
        
        # Initialize Memory
        Fnl = np.zeros((Nhc*Ndof,), np.double)
        
        if calc_grad:
            dFnldU = np.zeros((Nhc*Ndof,Nhc*Ndof), np.double)
            dFnldw = np.zeros((Nhc*Ndof,), np.double)
            
            # AFT for every set of nonlinear forces
            for nlforce in self.nonlinear_forces:
                Fnl_curr, dFnldU_curr, dFnldw_curr = nlforce.aft(U, w, h, Nt, aft_tol)
                
                Fnl += Fnl_curr
                dFnldU += dFnldU_curr
                dFnldw += dFnldw_curr
            
            return Fnl, dFnldU, dFnldw
                
        else: 
            # AFT for every set of nonlinear forces
            for nlforce in self.nonlinear_forces:
                Fnl_curr = nlforce.aft(U, w, h, Nt, aft_tol, calc_grad=False)[0]
                
                Fnl += Fnl_curr
            
            
            return (Fnl,)

    def hbm_res(self, Uw, Fl, h, Nt=128, aft_tol=1e-7):
        """
        Residual for Harmonic Balance Method (HBM). 
        The system as N=self.M.shape[0] degrees of freedom.

        Parameters
        ----------
        Uw : (N*Nhc+1,) numpy.ndarray
            Harmonic DOFs followed by frequency in rad/s. Has all of 0th
            harmonic (if included), then all of 1st cosine, then all of 1st 
            sine etc. There are Nhc harmonic components in h.
        Fl : (N*Nhc,) numpy.ndarray
            Applied forcing harmonic coefficients
        h : numpy.ndarray of integers, sorted
            List of Harmonics. The total number of harmonic components is
            Nhc = harmonic_utils.Nhc(h)
        Nt : integer power of 2, optional
            Number of Time Steps for AFT, use powers of 2. The default is 128.
        aft_tol : float, optional
            Tolerance for AFT. The default is 1e-7.

        Returns
        -------
        R : (N*Nhc,) numpy.ndarray
            Residual
        dRdU : (N*Nhc,N*Nhc) numpy.ndarray
            Jacobian of residual w.r.t. Harmonic DOFs
        dRdw : (N*Nhc,) numpy.ndarray
            Derivative of residual w.r.t. frequency
        
        See Also
        --------
        hbm_res_dFl : 
            Harmonic balance residual with a different input/third output
            that allows for continuation with respect to scaling of external 
            force.
        
        """
        
        # Frequency (rad/s)
        w = Uw[-1]
        
        E,dEdw = hutils.harmonic_stiffness(self.M, self.C, self.K, w, h)
        
        #### OLD AFT:
        # # Counting:
        # Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        # Ndof = self.M.shape[0]
        
        # Fnl = np.zeros((Nhc*Ndof,), np.double)
        # dFnldU = np.zeros((Nhc*Ndof,Nhc*Ndof), np.double)
        
        # # AFT for every set of nonlinear forces
        # for nlforce in self.nonlinear_forces:
        #     Fnl_curr, dFnldU_curr = nlforce.aft(Uw[:-1], w, h, Nt, aft_tol)
            
        #     Fnl += Fnl_curr
        #     dFnldU += dFnldU_curr
        
        # Alternating Frequency Time Call
        Fnl, dFnldU, dFnldw = self.total_aft(Uw[:-1], w, h, Nt=Nt, aft_tol=aft_tol)
        
        # Conditioning applied here in previous MATLAB version
        
        R = E @ Uw[:-1] + Fnl - Fl
        dRdU = E + dFnldU
        dRdw = dEdw @ Uw[:-1] + dFnldw
        
        return R, dRdU, dRdw
        
    def hbm_res_dFl(self, UF, w, Fl, h, Nt=128, aft_tol=1e-7):
        """
        Residual for Harmonic Balance Method (HBM). 
        The system as N=self.M.shape[0] degrees of freedom.

        Parameters
        ----------
        UF : (N*Nhc+1,) numpy.ndarray
            Harmonic DOFs followed by scaling of force vector. Has all of 0th
            harmonic (if included), then all of 1st cosine, then all of 1st 
            sine etc. There are Nhc harmonic components in h.
        w : float
            Frequency in rad/s of the 1st harmonic.
        Fl : (N*Nhc,) numpy.ndarray
            Applied forcing harmonic coefficients that will be scaled by UF[-1]
        h : numpy.ndarray of integers, sorted
            List of Harmonics. The total number of harmonic components is
            Nhc = harmonic_utils.Nhc(h)
        Nt : integer power of 2, optional
            Number of Time Steps for AFT, use powers of 2. The default is 128.
        aft_tol : float, optional
            Tolerance for AFT. The default is 1e-7.

        Returns
        -------
        R : (N*Nhc,) numpy.ndarray
            Residual
        dRdU : (N*Nhc,N*Nhc) numpy.ndarray
            Jacobian of residual w.r.t. Harmonic DOFs
        dRdF : (N*Nhc,) numpy.ndarray
            Derivative of residual w.r.t. scaling of Fl
        
        See Also
        --------
        hbm_res : 
            Harmonic balance residual with a different input/output
            that allows for continuation with respect to frequency
        
        """
        
        R, dRdU, _ = self.hbm_res(np.hstack((UF[:-1], w)), UF[-1]*Fl, 
                                     h, Nt=Nt, aft_tol=aft_tol)
        
        dRdF = -Fl
        
        return R, dRdU, dRdF
        
        
    def linear_frf(self, w, Fl, solver, neigs=3, Flsin=None):
        """
        Returns response to single harmonic (cosine) forcing in shape of Fl at 
        frequencies w

        Parameters
        ----------
        w : TYPE
            DESCRIPTION.
        Fl : Cosine only terms
        solver : TYPE
            DESCRIPTION.
        neigs : Number of modes to calculate for use in construction of the FRF
            DESCRIPTION. The default is 3.

        Returns
        -------
        Xw : Rows represent response amplitudes and then frequency at a single 
            frequency. 
            Xw[i] = [X1cos, X2cos,..., Xncos, X1sin, X2sin,..., Xnsin, w[i]]

        """
        
        assert not (self.ab is None), 'Modal FRF requires proportional damping.'
        
        if neigs > self.M.shape[0]:
            neigs = self.M.shape[0]
            print('linear_frf: Reducing Number of Modes to Number of DOFs = %d' % (neigs))
            
        if Flsin is None:
            Flsin = np.zeros_like(Fl)
            
        # Calculate Mode Shapes:
        wnsq,V = solver.eigs(self.K, self.M, subset_by_index=[0, neigs-1] )
        wn = np.sqrt(wnsq)

        # Modal Forcing:
        modal_f = V.T @ Fl
        modal_fsin = V.T @ Flsin
        modal_z = self.ab[0]/wn/2 +  self.ab[1]*wn/2
        
        # Modes in second direction
        modal_f = np.atleast_2d(modal_f).T
        modal_fsin = np.atleast_2d(modal_fsin).T
        modal_z = np.atleast_2d(modal_z).T
        wn2d = np.atleast_2d(wn).T
        
        # Rederived by Hand:
        qcos = ((wn2d**2 - w**2)*modal_f - (2*wn2d*w*modal_z)*modal_fsin)\
            / ( (wn2d**2 - w**2)**2 + (2*wn2d*w*modal_z)**2 )
        
        qsin = ((2*wn2d*w*modal_z)*modal_f + (wn2d**2 - w**2)*modal_fsin) \
            / ( (wn2d**2 - w**2)**2 + (2*wn2d*w*modal_z)**2 )
        
        Xw = np.vstack((V @ qcos, V @ qsin, w)).T
        
        return Xw


    def epmc_res(self, Uwxa, Fl, h, Nt=128, aft_tol=1e-7, calc_grad=True):
        """
        Residual for Extended Periodic Motion Concept
        
        System has n=self.M.shape[0] degrees of freedom and this call has Nhc 
        harmonic components (Nhc = hutils.Nhc(h))

        Parameters
        ----------
        Uwxa : np.array, size (n*Nhc + 3,)
            Harmonic DOFs followed by frequency, damping coefficient, 
            and log10(amplitude). Harmonic DOFs are the mass normalized 
            mode shape. (n*Nhc + 3,)
        Fl : np.array, size (n*Nhc,)
            Applied forcing harmonic coefficients.
            First n entries are an applied static force if the zeroeth 
            harmonic is included
        h : np.array
            List of Harmonics, assumed to be sorted start [0, 1, ...] or 
            [1, ...].
        Nt : Integer power of 2
            Number of Time Steps for AFT, use powers of 2. The default is 128.
        aft_tol : scalar
            Tolerance for AFT. The default is 1e-7.
        calc_grad : boolean
            Flag where True indicates that the gradients should be calculated 
            and returned. If False, then returns only (R,) as a tuple. 
            The default is True

        Returns
        -------
        R : np.array, size (n*Nhc+3)
            Residual
        dRdUwx : np.array, size (n*Nhc+3,n*Nhc+3)
            Jacobian of residual w.r.t. Harmonic DOFs, frequency, damping
        dRda : np.array, size (n*Nhc+3,)
            Derivative of residual w.r.t. log amplitude
        
        Notes
        -------
        1. Mass normalization constraint for amplitude is only applied to 
        harmonic 1 here. If you need subharmonic components, then some 
        restructuring is likely needed. 
        """
        
        # Shapes and Sizes
        Nhc = hutils.Nhc(h) # Number of Harmonic Components
        Ndof = self.M.shape[0]
        
        # Initialize Outputs
        R = np.zeros(Nhc*Ndof + 2)
        
        if calc_grad:
            dRdUwx = np.zeros((Nhc*Ndof + 2, Nhc*Ndof + 2))
            dRda = np.zeros(Nhc*Ndof+2)
        
        # Convert Log Amplitude
        la = Uwxa[-1]
        Amp = 10**la
        
        # Separate out Zeroth Harmonic (no amplitude scaling and applied force)
        h0 = 1*(h[0] == 0)
        
        Ascale = np.kron(np.hstack((np.ones(h0), Amp*np.ones(Nhc-h0))), \
                         np.ones(Ndof))
        
        if calc_grad:
            dAmpdla = Amp*np.log(10.0)
            dAscaledla = np.kron(np.hstack((np.zeros(h0), dAmpdla*np.ones(Nhc-h0))), \
                         np.ones(Ndof))
        
        # Static forces applied to zeroth harmonic
        Fstat = np.kron(np.hstack((np.ones(h0), np.zeros(Nhc-h0))), \
                         np.ones(Ndof)) * Fl
        
        # Phase constraint of the harmonics 1+
        Fdyn  = np.kron(np.hstack((np.zeros(h0), np.ones(Nhc-h0))), \
                         np.ones(Ndof)) * Fl
        
        # Frequency (rad/s)
        w = Uwxa[-3]
        
        # Negative mass prop damping coefficient
        xi = Uwxa[-2]
        
        # Harmonic Stiffness Matrices
        E_dEdw = hutils.harmonic_stiffness(self.M, self.C - xi*self.M, self.K, w, h,
                                           calc_grad=calc_grad)
        
        E = E_dEdw[0]
        
        if calc_grad:
            dEdw = E_dEdw[1] # only exists if calc_grad=True
            
            dEdxi = hutils.harmonic_stiffness(0, -self.M, 0, 
                                                w, h, calc_grad=False,
                                                only_C=True)[0]
        
        
        ########### # OLD AFT:
        # Fnl = np.zeros((Nhc*Ndof,), np.double)
        # dFnldU = np.zeros((Nhc*Ndof,Nhc*Ndof), np.double)
        
        # # AFT for every set of nonlinear forces
        # for nlforce in self.nonlinear_forces:
        #     Fnl_curr, dFnldU_curr = nlforce.aft(Ascale*Uwxa[:-3], w, h, Nt, aft_tol)
            
        #     Fnl += Fnl_curr
        #     dFnldU += dFnldU_curr
        
        
        # Alternating Frequency Time Call
        
        AFT_res = self.total_aft(Ascale*Uwxa[:-3], w, h, Nt=Nt, 
                                             aft_tol=aft_tol,
                                             calc_grad=calc_grad)
        if calc_grad:
            Fnl, dFnldU, dFnldw = AFT_res
        else:
            Fnl = AFT_res[0]
            
        # Output Residual and Derivatives
        # Force Balance
        R[:-2] = E @ (Ascale*Uwxa[:-3]) + Fnl - Fstat
        
        # Amplitude Constraint
        R[-2]  = Uwxa[h0*Ndof:((h0+1)*Ndof)] @ (self.M @ Uwxa[h0*Ndof:((h0+1)*Ndof)]) \
                  + Uwxa[(h0+1)*Ndof:((h0+2)*Ndof)] @ (self.M @ Uwxa[(h0+1)*Ndof:((h0+2)*Ndof)]) \
                  - 1.0
        
        # Phase Constraint
        R[-1]  = Fdyn @ Uwxa[:-3]
        
        if calc_grad:
            # d Force Balance / d Displacements
            dRdUwx[:-2, :-2] = (E + dFnldU) * Ascale #.reshape(-1,1)
            
            # d Force Balance / d w
            dRdUwx[:-2, -2] = dEdw @ (Ascale * Uwxa[:-3]) + dFnldw
            
            # d Force Balance / d xi
            dRdUwx[:-2, -1] = dEdxi @ (Ascale * Uwxa[:-3])
            
            # d Amplitude Constraint / d Displacements (only 1st harmonic)
            dRdUwx[-2, h0*Ndof:(h0+1)*Ndof] = 2*Uwxa[h0*Ndof:((h0+1)*Ndof)] @ self.M
                
            dRdUwx[-2, (h0+1)*Ndof:(h0+2)*Ndof] = 2*Uwxa[(h0+1)*Ndof:((h0+2)*Ndof)] @ self.M
            
            # d Phase Constraint / d Displacements
            dRdUwx[-1, :-2] = Fdyn
            
            # d Force Balance / d Total Amplitude Scaling
            dRda[:-2] = (E + dFnldU) @ (dAscaledla * Uwxa[:-3])
        
            return R, dRdUwx, dRda
        else:
            # Still return as a tuple, so can always index first result to get
            # residual
            return (R,) 
    
    def hbm_base_res(self, Uw, Ub, base_flag, h, Nt=128, aft_tol=1e-7):
        """
        Residual for Harmonic Balance Method with applied base excitation

        system has n free DOFs and nbase DOFs with prescribed displacements
        
        Assumes no harmonic forcing other than the base excitation

        Parameters
        ----------
        Uw : Harmonic DOFs followed by frequency, (n * Nhc + 1) x 1
        Ub : Applied Harmonic Displacements to base DOFS, (nbase * Nhc) x 1
        base_flag : vector of length (n+nbase) with True for base DOFs
        h : List of Harmonics
        Nt : Number of Time Steps for AFT, use powers of 2. The default is 128.
        aft_tol : Tolerance for AFT. The default is 1e-7.

        Returns
        -------
        R : Residual (n * Nhc)
        dRdU : Jacobian of residual w.r.t. Harmonic DOFs (n * Nhc x n * Nhc)
        dRdw : Derivative of residual w.r.t. frequency (n * Nhc)
        """
        
        # Mask of which DOFs are base excitation
        Nhc = hutils.Nhc(h)
        base_mask = np.hstack(( np.kron(np.full((Nhc,), True), base_flag), \
                               np.array([False]))) # Frequency Component
        
        # Convert Uw and Ub into a full vector
        Uw_full = np.zeros(base_mask.shape)
        
        Uw_full[base_mask] = Ub
        Uw_full[np.logical_not(base_mask)] = Uw
        
        Fl = np.zeros_like(Uw_full[:-1])
        
        # Call hbm_res
        R_full, dRdU_full, dRdw_full = self.hbm_res(Uw_full, Fl, h, Nt=Nt, aft_tol=aft_tol)
        
        # Remove rows/columns of hbm_res results
        R = R_full[np.logical_not(base_mask[:-1])]
        
        # This double indexing keeps the 2D shape for the masking of indices
        dRdU = dRdU_full[np.logical_not(base_mask[:-1]), :]\
                        [:, np.logical_not(base_mask[:-1])]
        
        dRdw = dRdw_full[np.logical_not(base_mask[:-1])]
        
        
        return R, dRdU, dRdw
    
    def linear_frf_base(self, w, Ub, base_flag, solver, neigs=3):
        """
        Returns response to single harmonic base excitation at frequencies w
        
        While this is structured to allow multiple base DOFs, the analytical 
        calculation implicitly assumes that all base DOFs move together

        Parameters
        ----------
        w : Frequencies of interest
        Ub : Vector of cosine base displacements followed by sine (length 2)
        solver : TYPE
            DESCRIPTION.
        neigs : Number of modes to calculate for use in construction of the FRF
            DESCRIPTION. The default is 3.

        Returns
        -------
        Xw : Rows represent response amplitudes and then frequency at a single 
            frequency. 
            Xw[i] = [X1cos, X2cos,..., Xncos, X1sin, X2sin,..., Xnsin, w[i]]

        """
        
        
        if neigs > self.M.shape[0]-base_flag.sum():
            neigs = self.M.shape[0]-base_flag.sum()
            print('linear_frf_base: Reducing Number of Modes to Number of DOFs = %d' % (neigs))
        
        Mrel = self.M[np.logical_not(base_flag), :]\
                        [:, np.logical_not(base_flag)]
                        
        Krel = self.K[np.logical_not(base_flag), :]\
                        [:, np.logical_not(base_flag)]
        
        
        # Calculate Mode Shapes:
        wnsq,V = solver.eigs(Krel, Mrel, subset_by_index=[0, neigs-1] )
        wn = np.sqrt(wnsq)

        # # Base Excitation as real force (based on relative coords)
        # f_base_cos = -(Mrel.sum(axis=0) * Ub[0]) * w.reshape(-1,1)**2
        # f_base_sin = -(Mrel.sum(axis=0) * Ub[1]) * w.reshape(-1,1)**2
        
        # # Modal Forcing:
        # modal_f = V.T @ f_base_cos.T
        # modal_fsin = V.T @ f_base_sin.T
        
        # Just using the stiffness/damping matrix column to generate base forces
        f_base_cos = -self.K[np.logical_not(base_flag),:][:, base_flag]*Ub[0]\
                      - self.C[np.logical_not(base_flag),:][:, base_flag]*Ub[1]*w
                     
         
        f_base_sin = -self.K[np.logical_not(base_flag),:][:, base_flag]*Ub[1]\
                      + self.C[np.logical_not(base_flag),:][:, base_flag]*Ub[0]*w
        

        # Now essentially copied from linear_frf
        # Modal Forcing:
        modal_f = V.T @ f_base_cos
        modal_fsin = V.T @ f_base_sin
        modal_z = self.ab[0]/wn/2 +  self.ab[1]*wn/2
        
        # Modes in second direction
        modal_z = np.atleast_2d(modal_z).T
        wn2d = np.atleast_2d(wn).T
        
        #######################
        # # From Inman (SDOF):
        # phase = np.arctan2(2*modal_z*wn2d*w, wn2d**2 - w**2)
        
        # cos_amp = 2*modal_z*wn2d*w*Ub[0]\
        #             / np.sqrt( (wn2d**2 - w**2)**2 + (2*wn2d*w*modal_z)**2 )
            
        # qcos = cos_amp*np.cos(phase)
        
        # qsin = cos_amp*np.sin(phase)
        
        #######################
        # Rederived by Hand:
        qcos = ((wn2d**2 - w**2)*modal_f - (2*wn2d*w*modal_z)*modal_fsin)\
            / ( (wn2d**2 - w**2)**2 + (2*wn2d*w*modal_z)**2 )
        
        qsin = ((2*wn2d*w*modal_z)*modal_f + (wn2d**2 - w**2)*modal_fsin) \
            / ( (wn2d**2 - w**2)**2 + (2*wn2d*w*modal_z)**2 )
        
        # Stack output plus transform back from relative to absolute coordinates
        
        # Xw = np.vstack(( (V @ qcos) + Ub[0], (V @ qsin) + Ub[1], w)).T
        Xw = np.vstack(( (V @ qcos), (V @ qsin), w)).T
        
        
        
        return Xw
        
    def shooting_res(self, Uw, Fl, Nt=128, return_aux=False):
        """
        Residual for shooting calculations

        Parameters
        ----------
        Uw : Vector of statespace initial states - all displacements, 
                                                    then all velocities
                                                    then frequency (rad/s)
                                                    size: (2*Ndof+1,)
        Fl : Forcing vector of size 2*Ndofs - first Ndofs are cos forcing. 
             Second Ndofs are sin forcing
             size: (2*Ndof,)
        Nt : Number of time steps to use.
            DESCRIPTION. The default is 128.
        return_aux : flag to return extra outputs (e.g., timeseries and Jacobian)

        Returns
        -------
        None.
        
        Notes: 
            1. Only supports instantaneous nonlinear forces.
            2. Implementation also does not support cubic damping
            3. WARNING: Most instantaneous nonlinear force calculations are 
                untested since unit tests focused on HBM/AFT.

        """
        
        Ndof = self.M.shape[0]
        
        # R = np.zeros(2*Ndof)
        # dRdU = np.zeros(2*Ndof, 2*Ndof)
        # dRdw = np.zeros(2*Ndof)
        
        # Check that only instantaneous force used. 
        for nlforce in self.nonlinear_forces:
            force_type = nlforce.nl_force_type()
            assert force_type == 0, 'Only instantaneous nonlinear forces are' \
                    + ' supported by this shooting implementation'
        
        assert Uw[:-1].shape[0] == 2*Ndof, 'Wrong size input for shooting.'
                    
        # Run Time Integration
        UV_dUVdUV0 = np.hstack((Uw[:-1], np.eye(2*Ndof).reshape(-1), np.zeros(2*Ndof) ))
        Period = 2*np.pi/Uw[-1]
        
        
        ydotfun = lambda t,y : _shooting_state_space(t, y, self, Fl, Uw[-1])
    
        
        ivp_res = solve_ivp(ydotfun, (0.0, Period), UV_dUVdUV0, 
                        max_step=Period/Nt,
                        t_eval=np.array(range(Nt+1))*Period/Nt) 
        
        Yfinal = ivp_res['y'][:, -1]
        Yfinal_dot = ydotfun(Period, Yfinal)
        
        # Derivative of the final state w.r.t. the frequency influence on the
        # external forcing period.
        dYfinaldF_dFdw = Yfinal[-2*Ndof:]
        
        
        # Prepare Outputs
        R = Yfinal[:2*Ndof] - Uw[:-1]
        dRdU = Yfinal[2*Ndof:-2*Ndof].reshape(2*Ndof, 2*Ndof) - np.eye(2*Ndof)
        dRdw = Yfinal_dot[:2*Ndof] * (2 * np.pi) * (-1.0 / Uw[-1]**2) + dYfinaldF_dFdw
        
        
        if return_aux:
            
            monodromy = Yfinal[2*Ndof:-2*Ndof].reshape(2*Ndof, 2*Ndof)
            
            y_t = ivp_res['y'][:Ndof, :]
            ydot_t = ivp_res['y'][Ndof:2*Ndof, :]
            
            aux = (monodromy, y_t, ydot_t, ivp_res)
            
            return R, dRdU, dRdw, aux
            
        else:
            return R, dRdU, dRdw
    
    
    def vprnm_res(self, UwF, h, rhi, Fl, Nt=128, aft_tol=1e-7):
        """
        Function implements a residual option for Variable Phase Resonance 
        Nonlinear Modes for Multiple Degree of Freedom (MDOF) systems.
        
        Parameters
        ----------
        UwF : Vector of Displacements, Frequency, Force Magnitude scaling of Fl
        h : Vector of harmonics to include
        rhi : index in h of higher harmonic in the superharmonic resonance 
        Fl : external force direction experienced by system 
                (to be scaled by UwF[-1])
        Nt : Number of AFT Time steps
            DESCRIPTION. The default is 128.
        aft_tol : AFT Tolerance
            DESCRIPTION. The default is 1e-7.

        Returns
        -------
        R : Residual
        dRdUw : Derivative w.r.t. Uw
        dRdF : Derivative vector w.r.t. F

        """
        
        #############
        # Initialization
        
        # Initialize output shapes
        R = np.zeros_like(UwF[:-1])
        dRdUw = np.zeros((R.shape[0], R.shape[0]))
        dRdF = np.zeros_like(R)
        
        # Harmonic indices
        Nhc = hutils.Nhc(h)
        Ndof = self.M.shape[0]
        rhi_index = hutils.Nhc(h[:rhi]) # the index of the first rhi harmonic component
        
        
        #############
        # Baseline HBM for first set of Equations
        
        # Evaluate the N Harmonic Balance Equations
        Rhbm, dRhbmdU, dRhbmdw = self.hbm_res(UwF[:-1], UwF[-1]*Fl, h, Nt=Nt, aft_tol=aft_tol)
        
        #############
        # VPRNM Equation
        
        # Eliminate higher harmonics to determine Fbroad
        Uw_fundamental = np.copy(UwF[:-1])
        Uw_fundamental[Ndof*rhi_index:-1] = 0.0 # remove higher harmonics
        
        # Evaluate the special case of the nonlinear forces here
        Fint, dFintdU, dFintdw = self.total_aft(Uw_fundamental[:-1], 
                                                Uw_fundamental[-1], 
                                                h, Nt=Nt, aft_tol=aft_tol)
            
            
        # Excitation of rhi acting as an external force
        Frhi = -Fint[Ndof*rhi_index:Ndof*(rhi_index+2)]
    
        # Preserve derivative info w.r.t. harmonics up to rhi - 1
        dFrhidU01 = -dFintdU[Ndof*rhi_index:Ndof*(rhi_index+2), 0:Ndof*rhi_index] 
        
        dFrhidw = -dFintdw[Ndof*rhi_index:Ndof*(rhi_index+2)]
        
        # Normalize the force vector to be of unit length
        Fnorm = np.sqrt(np.sum(Frhi**2))
        
        #############
        # Assemble Full Gradient and Residual
        
        # Add Orthogonality constrait using Frhi 
        # (harmonic rhi orthogonal to forcing for resonance)
        R = np.hstack((Rhbm, (Frhi @ UwF[Ndof*rhi_index:Ndof*(rhi_index+2)])/Fnorm))
        dRdUw[:Ndof*Nhc, :Ndof*Nhc] = dRhbmdU
        dRdUw[:Ndof*Nhc, -1]   = dRhbmdw
                
        Xrhi = UwF[Ndof*rhi_index:Ndof*(rhi_index+2)]
        
        dRdUw[-1, :Ndof*rhi_index] = (dFrhidU01.T @ Xrhi) / Fnorm \
                    - (Frhi @ Xrhi) / Fnorm**3 * (dFrhidU01.T @ Frhi)

        dRdUw[-1, Ndof*rhi_index:Ndof*(rhi_index+2)] = Frhi / Fnorm
        
        dRdUw[-1, -1] = (dFrhidw @ Xrhi) / Fnorm \
                    - (Frhi @ Xrhi) / Fnorm**3 * (dFrhidw @ Frhi)
        
        # negative since HBM is internal minus external force
        dRdF[:Fl.shape[0]] = -Fl
        
        
        # Return outputs include those needed for arclength equation        
        return R, dRdUw, dRdF
    
    def hbm_amp_control_res(self, UFw, Fl, h, Recov, amp, order, 
                            Nt=128, aft_tol=1e-7):
        """
        Amplitude Control with Harmonic Balance (rather than fixing force 
                                                 level)
        
        Control is applied exclusively to the 1st harmonic
        
        For documentation Nhc is the number of harmonics
        Ndof is the number of Degree of Freedoms

        Parameters
        ----------
        UFw : Harmonic Displacements, Force Scaling, Frequency (Ndof*Nhc+2,)
                Harmonic Displacements are all of zeroth, 1c, 1s, 2c, 2s etc.
        Fl : Forcing Vector without scaling for all harmonics (Nhc*Ndof,)
        h : List of harmonics used, must be sorted and include 1st harmonic.
        Recov : Recovery matrix for the DOF that has amplitude control (Ndof,)
        amp : Amplitude that the recovered DOF is controlled to 
        order : order of the derivative that is controlled. order=0 means 
                displacement control, order=2 means acceleration control
        Nt : Number of time steps for AFT. The default is 128.
        aft_tol : Tolerance for AFT evaluations. The default is 1e-7.

        Returns
        -------
        R : residual vector
        dRdUF : derivative of the residual w.r.t. UF
        dRdw : derivative w.r.t. frequency

        """
        
        Uw = np.hstack((UFw[:-2], UFw[-1]))
        
        Rhbm, dRhbmdU, dRhbmdw = self.hbm_res(Uw, UFw[-2]*Fl, h, 
                                               Nt=Nt, aft_tol=aft_tol)
        
        Ndof = self.M.shape[0]
        Nhc = hutils.Nhc(h)
        
        h0 = h[0] == 0
        
        # 1st harmonic displacements
        u1c = UFw[h0*Ndof:(1+h0)*Ndof]
        u1s = UFw[(1+h0)*Ndof:(2+h0)*Ndof]
        
        udofc = Recov @ u1c
        udofs = Recov @ u1s
        
        # Augmented Equation for amplitude constraint
        # Power is twice the order of the derivative being controlled because
        # residual is on the amplitude squared.
        Raug =  (UFw[-1]**(2*order))*(udofc**2 + udofs**2) - amp**2
        
        # dRhbmdF = -Fl # don't create extra memory at this point
        
        dRaugdUF = np.zeros((1, Nhc*Ndof+1))
        dRaugdUF[0,     h0*Ndof:(1+h0)*Ndof] = (UFw[-1]**(2*order))*(2*udofc*Recov)
        dRaugdUF[0, (1+h0)*Ndof:(2+h0)*Ndof] = (UFw[-1]**(2*order))*(2*udofs*Recov)
        
        # dRaugdUF[0, -1] = 0 # augmented equation is independent of force scale
        
        dRaugdw = (2*order)*(UFw[-1]**((2*order)-1))*(udofc**2 + udofs**2)
        
        R = np.hstack((Rhbm, Raug))
        dRdUF = np.vstack((np.hstack((dRhbmdU, -Fl.reshape(-1,1))),
                           dRaugdUF))
    
        dRdw = np.hstack((dRhbmdw, dRaugdw))
    
        return R, dRdUF, dRdw
    
    
    
def _shooting_state_space(t, UV_dUVdUV0, vib_sys, Fl, omega):
    """
    Returns the state space representation time derivative for shooting

    Parameters
    ----------
    vib_sys : TYPE
        DESCRIPTION.
    U : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    Ndof = vib_sys.M.shape[0]
    
    U = UV_dUVdUV0[:Ndof] # Displacement
    V = UV_dUVdUV0[Ndof:2*Ndof] # Velocity
    dUVdUV0 = UV_dUVdUV0[2*Ndof:-2*Ndof].reshape(2*Ndof, 2*Ndof)
    dUVdw = UV_dUVdUV0[-2*Ndof:]

    
    Fnltot = np.zeros_like(U)
    dFnldUtot = np.zeros((Ndof, Ndof))
    
    # Internal Nonlinear Forces
    for nlforce in vib_sys.nonlinear_forces:
        Fnl, dFdU = nlforce.force(U)
        
        Fnltot += Fnl
        dFnldUtot += dFdU
        
    # External Forcing
    Fext = Fl[:Ndof]*np.cos(omega*t) + Fl[Ndof:2*Ndof]*np.sin(omega*t)
    
    # State Derivatives
    Udot = V
    
    rhs = -(vib_sys.C @ V + vib_sys.K @ U + Fnl) + Fext
    Minv = np.linalg.inv(vib_sys.M)
    Vdot = Minv @ rhs
    
    # Time Derivatives of States w.r.t. Initial Conditions
    # Equation (14) of Second NNM Paper - by Peeters, 2009
    # g(z) = time derivative vector of states (Udot, Vdot)
    
    dg_dz = np.vstack((np.hstack(( np.zeros((Ndof, Ndof)), np.eye(Ndof) )), \
                      np.hstack(( Minv @ (-vib_sys.K -dFnldUtot), \
                                  Minv @ (-vib_sys.C))) ))
        
    dUVdUV0_dot = dg_dz @ dUVdUV0
    
    # The NNM paper does not include external force and the derivative of that 
    # force w.r.t. frequency of the shooting period. 
    # These lines add those components
    dFextdw = -t*Fl[:Ndof]*np.sin(omega*t) + t*Fl[Ndof:2*Ndof]*np.cos(omega*t)
    
    dgdF_dFdw = np.hstack((np.zeros(Ndof), Minv @ dFextdw))
    
    dgdX_dXdw = dg_dz @ dUVdw
    
    # Combined Derivative Vector
    UV_dUVdUV0_dot = np.hstack((Udot, Vdot, dUVdUV0_dot.reshape(-1), 
                                dgdF_dFdw + dgdX_dXdw))
    
    return UV_dUVdUV0_dot
    
    
    
    