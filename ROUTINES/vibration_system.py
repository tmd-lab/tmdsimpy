import numpy as np
import harmonic_utils as hutils


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

    def hbm_res(self, Uw, Fl, h, Nt=128, aft_tol=1e-7):
        """
        Residual for Harmonic Balance Method

        Parameters
        ----------
        Uw : Harmonic DOFs followed by frequency, (n * Nhc + 1) x 1
        Fl : Applied forcing harmonic coefficients, (n * Nhc) x 1
        h : List of Harmonics
        Nt : Number of Time Steps for AFT, use powers of 2. The default is 128.
        aft_tol : Tolerance for AFT. The default is 1e-7.

        Returns
        -------
        R : Residual
        dRdU : Jacobian of residual w.r.t. Harmonic DOFs
        dRdw : Derivative of residual w.r.t. frequency
        """
        
        # Counting:
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        Ndof = self.M.shape[0]
        
        # Frequency (rad/s)
        w = Uw[-1]
        
        E,dEdw = hutils.harmonic_stiffness(self.M, self.C, self.K, w, h)
        
        Fnl = np.zeros((Nhc*Ndof,), np.double)
        dFnldU = np.zeros((Nhc*Ndof,Nhc*Ndof), np.double)
        
        # AFT for every set of nonlinear forces
        for nlforce in self.nonlinear_forces:
            Fnl_curr, dFnldU_curr = nlforce.aft(Uw[:-1], w, h, Nt, aft_tol)
            
            Fnl += Fnl_curr
            dFnldU += dFnldU_curr
        
        # Conditioning applied here in previous MATLAB version
        
        R = E @ Uw[:-1] + Fnl - Fl
        dRdU = E + dFnldU
        dRdw = dEdw @ Uw[:-1]
        
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


    def epmc_res(self, Uwxa, Fl, h, Nt=128, aft_tol=1e-7):
        """
        Residual for Extended Periodic Motion Concept

        Parameters
        ----------
        Uwxa : Harmonic DOFs followed by frequency, damping coefficient, 
                and log10(amplitude). Harmonic DOFs are the mass normalized 
                mode shape. (n * Nhc + 3) x 1
        Fl : Applied forcing harmonic coefficients, (n * Nhc) x 1.
                First n entries are an applied static force if the zeroeth 
                harmonic is included
        h : List of Harmonics, assumed to be sorted start [0, 1, ...] or 
                [1, ...]
        Nt : Number of Time Steps for AFT, use powers of 2. The default is 128.
        aft_tol : Tolerance for AFT. The default is 1e-7.

        Returns
        -------
        R : Residual
        dRdUwx : Jacobian of residual w.r.t. Harmonic DOFs, frequency, damping
        dRda : Derivative of residual w.r.t. log amplitude
        
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
        dRdUwx = np.zeros((Nhc*Ndof + 2, Nhc*Ndof + 2))
        dRda = np.zeros(Nhc*Ndof+2)
        
        # Convert Log Amplitude
        la = Uwxa[-1]
        Amp = 10**la
        dAmpdla = Amp*np.log(10.0)
        
        # Separate out Zeroth Harmonic (no amplitude scaling and applied force)
        h0 = 1*(h[0] == 0)
        
        Ascale = np.kron(np.hstack((np.ones(h0), Amp*np.ones(Nhc-h0))), \
                         np.ones(Ndof))
        
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
        E,dEdw = hutils.harmonic_stiffness(self.M, self.C - xi*self.M, self.K, w, h)
        dEdxi,_ = hutils.harmonic_stiffness(self.M*0, -self.M, self.K*0, w, h)
        
        Fnl = np.zeros((Nhc*Ndof,), np.double)
        dFnldU = np.zeros((Nhc*Ndof,Nhc*Ndof), np.double)
        
        # AFT for every set of nonlinear forces
        for nlforce in self.nonlinear_forces:
            Fnl_curr, dFnldU_curr = nlforce.aft(Ascale*Uwxa[:-3], w, h, Nt, aft_tol)
            
            Fnl += Fnl_curr
            dFnldU += dFnldU_curr
        
        # Output Residual and Derivatives
        # Force Balance
        R[:-2] = E @ (Ascale*Uwxa[:-3]) + Fnl - Fstat
        
        # Amplitude Constraint
        R[-2]  = Uwxa[h0*Ndof:((h0+1)*Ndof)] @ (self.M @ Uwxa[h0*Ndof:((h0+1)*Ndof)]) \
                  + Uwxa[(h0+1)*Ndof:((h0+2)*Ndof)] @ (self.M @ Uwxa[(h0+1)*Ndof:((h0+2)*Ndof)]) \
                  - 1.0
        
        # Phase Constraint
        R[-1]  = Fdyn @ Uwxa[:-3]
        
        # d Force Balance / d Displacements
        dRdUwx[:-2, :-2] = (E + dFnldU) * Ascale #.reshape(-1,1)
        
        # d Force Balance / d w
        dRdUwx[:-2, -2] = dEdw @ (Ascale * Uwxa[:-3])
        
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
        