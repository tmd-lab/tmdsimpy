import numpy as np
from . import harmonic_utils as hutils
from scipy.integrate import solve_ivp

import warnings

class VibrationSystem:
    
    def __init__(self, M, K, C=None, ab=None):
        """
        Initialize the linear part of a system

        Parameters
        ----------
        M : Mass Matrix, n x n
        K : Stiffness Matrix, n x n
        C : Damping Matrix, n x n, Default is Zero
        ab : Mass and Stiffness Proportional Damping Coefficients. If provided, 
             used to recalculate stiffness matrix

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
        Residual for Harmonic Balance Method

        Parameters
        ----------
        Uw : Harmonic DOFs followed by frequency, (n * Nhc + 1)
        Fl : Applied forcing harmonic coefficients, (n * Nhc)
        h : List of Harmonics
        Nt : Number of Time Steps for AFT, use powers of 2. The default is 128.
        aft_tol : Tolerance for AFT. The default is 1e-7.

        Returns
        -------
        R : Residual
        dRdU : Jacobian of residual w.r.t. Harmonic DOFs
        dRdw : Derivative of residual w.r.t. frequency
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
    
    
    
    