import numpy as np
from .utils import harmonic as hutils
from scipy.integrate import solve_ivp

import warnings

class VibrationSystem:
    """
    A vibration system model with several useful residual functions.
    The system has `N` displacement degrees of Freedom.

    Parameters
    ----------
    M : (N,N) numpy.ndarray
        Mass Matrix
    K : (N,N) numpy.ndarray
        Stiffness Matrix, n x n
    C : (N,N) numpy.ndarray or None, optional
        Damping Matrix. If `ab` is provided, that will be used instead to 
        construct a damping matrix. If both are None, then a zero damping 
        matrix will be used.
        Default is None.
    ab : list of length 2 or None, optional
        Mass and Stiffness Proportional Damping Coefficients. If provided, 
        used to recalculate stiffness matrix as
        `C = ab[0]*self.M + ab[1]*self.K`.
        The default is None.

    See Also
    --------
    VibrationSystem.set_new_C : 
        Sets the damping matrix to a new value for an existing object.

    Notes
    -----
    A number of methods are provided for adding and modifying the state of
    nonlinear forces. Other methods are provided to give residual evaluations
    for a range of nonlinear vibration methods.

    """
    
    def __init__(self, M, K, C=None, ab=None):
        
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
        Add a nonlinear force to the model.

        Parameters
        ----------
        nlforce : tmdsimpy.nlforces.NonlinearForce or similar
            Nonlinear force to be added to model.

        Notes
        -----
        The provided nonlinear force is appendended to the end of
        `self.nonlinear_forces`.

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
        Update internal nonlinear force history variables to reflect the
        current state.

        This is generally called after a static analysis such as for prestress.

        Parameters
        ----------
        U : (N,) numpy.ndarray
            Displacements to set an update of the force history at.
            These are generally of prestressed state.

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
        Set friction coefficients to a prestress value for analysis.

        Returns
        -------
        None.

        See Also
        --------
        reset_real_mu :
            Method for returning to the real value of the friction slip limit.

        Notes
        -----
        This method calls the `set_prestress_mu` method on nonlinear forces
        with that method defined. For hysteretic models, `set_prestress_mu`
        is generally defined to set the slip force or friction coefficient to
        zero. This is done for prestress analysis only to eliminate tangential
        forces.

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
        Reset the friction coefficient to the real value for analysis.

        Returns
        -------
        None.

        See Also
        --------
        set_prestress_mu :
            Method for setting friction limit to prestress value
            (generally zero).

        Notes
        -----
        This method calls the `reset_real_mu` on all nonlinear forces that have
        that method defined. This generally is defined to return the friction
        coefficient or slip limit to the initialized value after prestress
        analysis. For prestress analysis, the slip limit is frequently set to
        zero.

        """
        
        # Check if nonlinear force has a reset mu function and call if 
        # it exists
        for nlforce in self.nonlinear_forces:
            pre_mu = getattr(nlforce, "reset_real_mu", None)
            if callable(pre_mu):
                nlforce.reset_real_mu()
                
        return
    
    def set_aft_initialize(self, U):
        """
        Set an initial position for friction model slides to be used during
        frequency domain analysis.

        Parameters
        ----------
        U : (N,) numpy.ndarray
            Global static displacements.

        Returns
        -------
        None.

        Notes
        -----

        Frequency domain analyses utilize the alternating frequency time (AFT)
        method to evaluate nonlinear forces. For hysteretic models, solutions
        may not be unique depending on the choice of initial slider positions
        used to initialize the AFT force evaluations. This method allows for
        setting all friction model sliders based on a set of global static
        displacements.

        """
        
        # Check if nonlinear force has a reset mu function and call if 
        # it exists
        for nlforce in self.nonlinear_forces:
            set_aft = getattr(nlforce, "set_aft_initialize", None)
            if callable(set_aft):
                nlforce.set_aft_initialize(U)
                
            elif nlforce.nl_force_type() == 0:
                pass
            else:
                warnings.warn('Nonlinear force: {} '.format(nlforce)
                              + 'does not have an AFT initialize')
                
        return
    
    def set_new_C(self, C=None, ab=None):
        """
        Set the damping matrix to a new value after initialization.

        Parameters
        ----------
        C : (N,N) numpy.ndarray, optional
            New damping matrix. If `ab` is not `None`, then `ab` is used
            instead. If
            neither `C` nor `ab` is provided, nothing is changed.
            The default is None.
        ab : list of length 2, optional
            If provided, the damping matrix is set to 
            `C = ab[0]*self.M + ab[1]*self.K`.
            The default is None.

        Returns
        -------
        None.

        Notes
        -----

        This method is useful if one wants to set the damping matrix to be
        mass and stiffness proportional around a prestressed state using the
        linearization of the nonlinear forces. In that case, one manually
        calculates `C` with the proportional coefficients and linearized
        stiffness and passes it here.

        """
        
        if not (ab is None):
            if not (C is None):
                warnings.warn('Ignoring C to overwrite with proportional'
                              + ' damping.')
            
            C = ab[0]*self.M + ab[1]*self.K
            self.ab = ab
        
        if C is None:
            self.C = np.zeros_like(self.M)
        else:
            self.C = C
        
        return
    
    def static_res(self, U, Fstatic):
        """
        Residual function for static analysis.

        Parameters
        ----------
        U : (N,) numpy.ndarray
            Global static displacements to evaluate residual at.
        Fstatic : (N,) numpy.ndarray
            Externally applied static forces on global coordinates.

        Returns
        -------
        R : (N,) numpy.ndarray
            Evaluated residual for static analysis.
        dRdU : (N,N) numpy.ndarray
            Derivative of `R` with respect to `U`.

        See Also
        --------
        update_force_history :
            Method for updating force history after a static analysis.

        Notes
        -----

        This method is generally used to solve for a prestress analysis.
        Additionally, `dRdU` gives the linearized stiffness around a state.
        For friction models, make sure the friction coefficient is set to
        the nonzero value with `reset_real_mu` and the history variables are
        correct prior to evaluating the
        linearized stiffness.

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
        Apply Alternating Frequency Time (AFT) Method to calculate all
        nonlinear harmonic forces.
        
        Parameters
        ----------
        U : (N*Nhc,) numpy.ndarray
            Global harmonic degrees of freedom, all DOFs for each harmonic
            component and then the next harmonic component in `h`.
        w : float
            Frequency (rad/s) of first harmonic.
        h : 1D np.array, sorted
            List of included harmonics, sorted and without repeats.
            Harmonics should be positive integers or zero.
        Nt : int, power of 2, optional
            Number of time steps for AFT. 
            The default is 128.
        aft_tol : float, optional
            Tolerance for AFT.
            The default is 1e-7.
        calc_grad : bool
            Flag where `True` indicates that the gradients should be calculated
            and returned. If `False`, then returns only `(Fnl,)` as a tuple.
            `False` should only be passed if all nonlinear forces have AFT
            methods that accept the `calc_grad` keyword.
            If `True`, the argument is not passed to nonlinear forces.
            The default is `True`.

        Returns
        -------
        Fnl : (N*Nhc,) numpy.ndarray
            Internal nonlinear force harmonic coefficients in global
            coordiantes.
        dFnldU : (N*Nhc,N*Nhc) numpy.ndarray
            Derivative of `Fnl` with respect to `U`.
        dFnldw : (N*Nhc,) numpy.ndarray
            Derivative of `Fnl` with respect to `w`.

        Notes
        -----
        The number of harmonic components is 
        `Nhc = tmdsimpy.utils.harmonic.Nhc(h)`

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

    def hbm_res(self, Uw, Fl, h, Nt=128, aft_tol=1e-7, calc_grad=True):
        """
        Residual for the basic Harmonic Balance Method (HBM).

        Parameters
        ----------
        Uw : (N*Nhc+1,) numpy.ndarray
            Global harmonic degrees of freedom, all DOFs for each harmonic
            component and then the next harmonic component in `h`.
            These are followed by the frequency in rad/s of first harmonic.
        Fl : (N*Nhc,) numpy.ndarray
            Applied external forcing harmonic coefficients in the same ordering
            as displacements in `Uw`.
        h : 1D np.array, sorted
            List of included harmonics, sorted and without repeats.
            Harmonics should be positive integers or zero.
        Nt : int, power of 2, optional
            Number of time steps for AFT.
            The default is 128.
        aft_tol : float, optional
            Tolerance for AFT.
            The default is 1e-7.
        calc_grad : bool
            Flag where `True` indicates that the gradients should be calculated
            and returned. If `False`, then returns only `(R,)` as a tuple.
            `False` should only be passed if all nonlinear forces have AFT
            methods that accept the `calc_grad` keyword.
            If `True`, the argument is not passed to nonlinear forces.
            The default is `True`.

        Returns
        -------
        R : (N*Nhc,) numpy.ndarray
            Evaluated residual for HBM analysis.
        dRdU : (N*Nhc,N*Nhc) numpy.ndarray
            Derivative of `R` with respect to `U = Uw[:-1]`.
        dRdw : (N*Nhc,) numpy.ndarray
            Derivative of `R` with respect to `w = Uw[-1]`.
        
        See Also
        --------
        tmdsimpy.utils.harmonic.predict_harmonic_solution : 
            Function for generating initial guesses to HBM type problems.
        hbm_res_dFl : 
            HBM residual with a different input/third output
            that allows for continuation with respect to scaling of external 
            force.
        hbm_base_res : 
            HBM for base excited systems (prescribed displacement at DOFs)
        hbm_amp_control_res : 
            HBM with one extra unknown / equation that constrains solution
            to constant amplitude, variable response phase, fixed forcing phase
        hbm_amp_phase_control_res : 
            HBM with two extra equations and unknowns that allows for solutions
            along a constant response amplitude and phase.
        hbm_amp_phase_control_dA_res : 
            HBM with amplitude and phase control for continuation with
            respect to amplitude.

        Notes
        -----
        The number of harmonic components is
        `Nhc = tmdsimpy.utils.harmonic.Nhc(h)`

        """
        
        # Frequency (rad/s)
        w = Uw[-1]
        
        E_dEdw = hutils.harmonic_stiffness(self.M, self.C, self.K, w, h, 
                                           calc_grad=calc_grad)
        
        # Alternating Frequency Time Call
        Fnl_dFnldU_dFnldw = self.total_aft(Uw[:-1], w, h, Nt=Nt, 
                                           aft_tol=aft_tol, 
                                           calc_grad=calc_grad)
        
        R = E_dEdw[0] @ Uw[:-1] + Fnl_dFnldU_dFnldw[0] - Fl
        
        if calc_grad:
            dRdU = E_dEdw[0] + Fnl_dFnldU_dFnldw[1]
            dRdw = E_dEdw[1] @ Uw[:-1] + Fnl_dFnldU_dFnldw[2]
        
            return R, dRdU, dRdw
        else:
            return (R,)
        
    def hbm_res_dFl(self, UF, w, Fl, h, Nt=128, aft_tol=1e-7, calc_grad=True):
        """
        Residual for Harmonic Balance Method (HBM) for continuation with
        respect to force scaling.
        
        Parameters
        ----------
        UF : (N*Nhc+1,) numpy.ndarray
            Global harmonic degrees of freedom, all DOFs for each harmonic
            component and then the next harmonic component in `h`.
            These are followed by a scaling factor for the external forces.
            All harmonics except the zeroth harmonic are scaled by this factor.
        w : float
            Frequency in rad/s of the 1st harmonic.
        Fl : (N*Nhc,) numpy.ndarray
            Applied forcing harmonic coefficients that will be scaled by
            `UF[-1]`.
            The zeroth harmonic of Fl is not scaled.
        h : 1D np.array, sorted
            List of included harmonics, sorted and without repeats.
            Harmonics should be positive integers or zero.
        Nt : int, power of 2, optional
            Number of time steps for AFT.
            The default is 128.
        aft_tol : float, optional
            Tolerance for AFT.
            The default is 1e-7.
        calc_grad : bool
            Flag where `True` indicates that the gradients should be calculated
            and returned. If `False`, then returns only `(R,)` as a tuple.
            `False` should only be passed if all nonlinear forces have AFT
            methods that accept the `calc_grad` keyword.
            If `True`, the argument is not passed to nonlinear forces.
            The default is `True`.

        Returns
        -------
        R : (N*Nhc,) numpy.ndarray
            Evaluated residual for HBM analysis.
        dRdU : (N*Nhc,N*Nhc) numpy.ndarray
            Derivative of `R` with respect to `U = UF[:-1]`.
        dRdF : (N*Nhc,) numpy.ndarray
            Derivative of `R` with respect to `F = UF[-1]`.

        See Also
        --------
        hbm_res : 
            Harmonic balance residual with a different input/output
            that allows for continuation with respect to frequency. 
            See documentation of this function for a full list of HBM variants.
        tmdsimpy.utils.harmonic.predict_harmonic_solution : 
            Function for generating initial guesses to HBM type problems.

        Notes
        -----
        The number of harmonic components is
        `Nhc = tmdsimpy.utils.harmonic.Nhc(h)`

        """
        
        Ndof = self.M.shape[0]
        h0 = h[0] == 0
        
        Fstat = np.copy(Fl)
        Fstat[h0*Ndof:] = 0.0
        
        Fdyn = np.copy(Fl)
        Fdyn[:h0*Ndof] = 0.0
        
        R_dRdU_ = self.hbm_res(np.hstack((UF[:-1], w)), UF[-1]*Fdyn+Fstat, 
                                     h, Nt=Nt, aft_tol=aft_tol, 
                                     calc_grad=calc_grad)
        
        if calc_grad:
            dRdF = -Fdyn
            
            return R_dRdU_[0], R_dRdU_[1], dRdF
        
        else:
            return (R_dRdU_[0],)
        
        
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
        Residual for Extended Periodic Motion Concept (EPMC).

        Parameters
        ----------
        Uwxa : (N*Nhc+3,) numpy.ndarray
            Global harmonic degrees of freedom, all DOFs for each harmonic
            component and then the next harmonic component in `h`.
            These are followed by the modal frequency in rad/s
            (first harmonic), the mass proportional self excitation
            coefficient, and the log10(modal amplitude).
            Harmonic DOFs are the mass normalized mode shape.
        Fl : (N*Nhc,) numpy.ndarray
            First `N` entries are the applied static forces if harmonic 0 is
            included in `h`.
            All other entries define a phase constraint to make the solution
            unique.
        h : 1D np.array, sorted
            List of included harmonics, sorted and without repeats.
            Harmonics should be positive integers or zero.
        Nt : int, power of 2, optional
            Number of time steps for AFT.
            The default is 128.
        aft_tol : float, optional
            Tolerance for AFT.
            The default is 1e-7.
        calc_grad : bool
            Flag where `True` indicates that the gradients should be calculated
            and returned. If `False`, then returns only `(R,)` as a tuple.
            `False` should only be passed if all nonlinear forces have AFT
            methods that accept the `calc_grad` keyword.
            If `True`, the argument is not passed to nonlinear forces.
            The default is `True`.

        Returns
        -------
        R : (N*Nhc,) numpy.ndarray
            Evaluated residual for EPMC analysis.
        dRdUwx : (N*Nhc,N*Nhc) numpy.ndarray
            Derivative of `R` with respect to `Uwx = Uwxa[:-1]`.
        dRda : (N*Nhc,) numpy.ndarray
            Derivative of `R` with respect to `a = Uwxa[-1]` (log amplitude).

        Notes
        -----
        The number of harmonic components is
        `Nhc = tmdsimpy.utils.harmonic.Nhc(h)`

        Mass normalization constraint for amplitude is applied to
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
        
        See Also
        --------
        hbm_res : 
            Harmonic balance residual for constant force input to the system.
            See documentation of this function for a full list of HBM variants
        linear_frf_base : 
            Update this docstring to have the correct linear FRF method for 
            base excitation here.
        
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
        Uw : (2*N+1,) numpy.ndarray
            Test solution point to the shooting residual.
            Has `N` displacements, then `N` velocities, then frequency in rad/s.
        Fl : (2*N) numpy.ndarray
            First `N` entries are cosine forcing at frequency `XlamP_shoot[-1]`.
            The second `N` are the sine forcing terms.
        Nt : int, optional
            Number of time steps to use in shooting calculations.
            The default is 128.
        return_aux : bool, optional
            Flag to return extra output variables 
            (time series, Monodromy matrix etc.).
            The default is False.
        
        Returns
        -------
        None.
        
        See Also
        --------
        tmdsimpy.postprocess.shooting.time_stability :
            Function for post processing the time series and stability from
            a solution point to these equations.
        
        Notes
        -----
        
        For theory about shooting and stability analysis, see Section 3
        of [1]_.
        
        Implementation currently only supports instantaneous nonlinear forces,
        but does not support cubic damping.
        
        Most instantaneous forces are tested focusing only on HBM/AFT rather
        than the instant force that is used here.
        
        References
        ----------
        
        .. [1] 
            Peeters, M., R. Viguie, G. Sérandour, G. Kerschen, 
            and J. -C. Golinval. 2009. "Nonlinear Normal Modes, Part II: Toward a
            Practical Computation Using Numerical Continuation Techniques."
            Mechanical Systems and Signal Processing, 
            Special Issue: Non-linear Structural Dynamics, 23 (1): 195–216.
            https://doi.org/10.1016/j.ymssp.2008.04.003.
    
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
    
    def vprnm_single_eqn(self, U, w, h, rhi, Nt=128, aft_tol=1e-7, 
                         calc_grad=True, superharmonic_filter=None):
        """
        Single residual equation evaluation for Variable Phase
        Resonance Nonlinear Modes (VPRNM).
        
        This equation in general needs to be added to a set of HBM equations
        to find HBM solutions along the superharmonic resonance.
        
        Parameters
        ----------
        U : (N*Nhc+m,) numpy.ndarray
            Global harmonic degrees of freedom, all DOFs for each harmonic
            component and then the next harmonic component in `h`.
            Only the first `N*Nhc` entries are directly indexed with positive
            numbers, so it is allowable to have extra values at the end of the 
            array.
        w : float
            Frequency (rad/s) of first harmonic.
        h : 1D np.array, sorted
            List of included harmonics, sorted and without repeats.
            Harmonics should be positive integers or zero.
        rhi : int
            Superharmonic resonance harmonic number of interest,
            must be included in `h`.
        Nt : int, power of 2, optional
            Number of time steps for AFT.
            The default is 128.
        aft_tol : float, optional
            Tolerance for AFT.
            The default is 1e-7.
        calc_grad : bool
            Flag where `True` indicates that the gradients should be calculated
            and returned. If `False`, then returns only `(Fnl,)` as a tuple.
            `False` should only be passed if all nonlinear forces have AFT
            methods that accept the `calc_grad` keyword.
            If `True`, the argument is not passed to nonlinear forces.
            The default is `True`.
        superharmonic_filter : None or (N,) numpy.ndarray, optional
            If None, VPRNM is calculated without a modal filter.
            If a `numpy.ndarray`, then VPRNM is modally filtered with the
            array.
            The modal filter is applied to the superharmonic resonance to 
            extract a specific mode.
            The default is None.

        Returns
        -------
        R : float
            Evaluated residual for VPRNM equation.
        dRdUw : (N*Nhc+1,) numpy.ndarray
            Derivative of `R` with respect to `Uw = numpy.hstack((U, w))`.

        See Also
        --------
        vprnm_res : 
            Full implementation of VPRNM including the HBM residual equations.

        Notes
        -----
        The number of harmonic components is 
        `Nhc = tmdsimpy.utils.harmonic.Nhc(h)`

        Theory for VPRNM is developed in [1]_, [2]_, [3]_.

        References
        ----------

        .. [1]
           Porter, J. H., and M. R. W. Brake. 2024. "Tracking Superharmonic
           Resonances for Nonlinear Vibration of Conservative and Hysteretic
           Single
           Degree of Freedom Systems." Mechanical Systems and Signal Processing
           215:111410. https://doi.org/10.1016/j.ymssp.2024.111410.
           arXiv:2401.08790

        .. [2]
           Porter, J. H., and M. R. W. Brake. Under Review. "Efficient Model
           Reduction and Prediction of Superharmonic Resonances in Frictional
           and Hysteretic Systems." Mechanical Systems and Signal Processing.
           arXiv:2405.15918.

        .. [3]
           Porter, J. H. 2024. Modal Interactions and Jointed Structures.
           PhD Thesis.
           Rice University.

        """
        
        # Harmonic indices
        Nhc = hutils.Nhc(h)
        Ndof = self.M.shape[0]
        
        # the index of the first rhi harmonic component
        rhi_index = hutils.Nhc(h[h < rhi]) 
        
        # Eliminate higher harmonics to determine Fbroad
        U_fundamental = np.copy(U[:Ndof*Nhc])
        U_fundamental[Ndof*rhi_index:] = 0.0 # remove higher harmonics
        
        # Evaluate the special case of the nonlinear forces here
        Fint_dFintdU_dFintdw = self.total_aft(U_fundamental, 
                                                w, 
                                                h, Nt=Nt, aft_tol=aft_tol,
                                                calc_grad=calc_grad)
        
        # Excitation of rhi acting as an external force
        Frhi = -Fint_dFintdU_dFintdw[0][Ndof*rhi_index:Ndof*(rhi_index+2)]

        Xrhi = U[Ndof*rhi_index:Ndof*(rhi_index+2)]

        # Modal filter of the superharmonic resonance
        if superharmonic_filter is not None:
            Frhi = np.hstack((superharmonic_filter @ Frhi[:Ndof],
                              superharmonic_filter @ Frhi[Ndof:]))
            
            modeT_M = superharmonic_filter @ self.M
            
            Xrhi = np.hstack((modeT_M @ Xrhi[:Ndof],
                              modeT_M @ Xrhi[Ndof:]))

        # Normalize the force vector to be of unit length
        Fnorm = np.sqrt(np.sum(Frhi**2))
        
        # Normalize Xrhi as well as F
        Xrhi_norm = np.sqrt(np.sum(Xrhi**2))
        
        R = (Frhi @ Xrhi)/(Xrhi_norm*Fnorm)
        
        if calc_grad and superharmonic_filter is None:
            # Gradients without modal filter
            
            dFintdU = Fint_dFintdU_dFintdw[1]
            dFintdw = Fint_dFintdU_dFintdw[2]
            
            # Preserve derivative info w.r.t. harmonics up to rhi - 1
            dFrhidU01 = -dFintdU[Ndof*rhi_index:Ndof*(rhi_index+2), 0:Ndof*rhi_index] 
            
            dFrhidw = -dFintdw[Ndof*rhi_index:Ndof*(rhi_index+2)]
        
            # Organize into final gradient form
            dRdUw = np.zeros(Ndof*Nhc+1)
            
            dRdUw[:Ndof*rhi_index] = (dFrhidU01.T @ Xrhi) / (Xrhi_norm*Fnorm) \
                   -(Frhi @ Xrhi) / (Xrhi_norm*Fnorm**3) * (dFrhidU01.T @ Frhi)

            dRdUw[Ndof*rhi_index:Ndof*(rhi_index+2)] = Frhi / (Xrhi_norm*Fnorm)\
                                - (Frhi @ Xrhi)/(Xrhi_norm**3*Fnorm) * Xrhi
        
            dRdUw[-1] = (dFrhidw @ Xrhi) / (Xrhi_norm*Fnorm) \
                    - (Frhi @ Xrhi) / (Xrhi_norm*Fnorm**3) * (dFrhidw @ Frhi)
                    
            return R, dRdUw
        
        if calc_grad and superharmonic_filter is not None:
            # Gradients with modal filter
            
            dFintdU = Fint_dFintdU_dFintdw[1]
            dFintdw = Fint_dFintdU_dFintdw[2]
            
            # Preserve derivative info w.r.t. harmonics up to rhi - 1
            dFrhidU01 = -dFintdU[Ndof*rhi_index:Ndof*(rhi_index+2), 0:Ndof*rhi_index] 
            
            dFrhidw = -dFintdw[Ndof*rhi_index:Ndof*(rhi_index+2)]
            
            # Modal filter derivative component parts
            
            # derivative of what is now called Xrhi (modal coordinates)
            # with respect to the original coordinates
            # eye(2) because there are cosine/sine components = 2 components
            dXrhi_dXrhiorig = np.kron(np.eye(2), modeT_M)
            
            # Derivative of modal force w.r.t. lower harmonics
            dFrhidU01 = np.kron(np.eye(2), superharmonic_filter) @ dFrhidU01
            
            dFrhidw = np.hstack((superharmonic_filter @ dFrhidw[:Ndof],
                                 superharmonic_filter @ dFrhidw[Ndof:]))
        
        
            # Organize into final gradient form
            dRdUw = np.zeros(Ndof*Nhc+1)
            
            dRdUw[:Ndof*rhi_index] = (dFrhidU01.T @ Xrhi) / (Xrhi_norm*Fnorm) \
                   -(Frhi @ Xrhi) / (Xrhi_norm*Fnorm**3) * (dFrhidU01.T @ Frhi)
            
            dRdUw[Ndof*rhi_index:Ndof*(rhi_index+2)] \
                = Frhi @ dXrhi_dXrhiorig / (Xrhi_norm*Fnorm) \
                - (Frhi @ Xrhi)/(Xrhi_norm**3*Fnorm) * (Xrhi @ dXrhi_dXrhiorig)
        
            dRdUw[-1] = (dFrhidw @ Xrhi) / (Xrhi_norm*Fnorm) \
                    - (Frhi @ Xrhi) / (Xrhi_norm*Fnorm**3) * (dFrhidw @ Frhi)

            return R, dRdUw
        
        else:
            return (R,)
        
        
    def vprnm_res(self, UwF, h, rhi, Fl, Nt=128, aft_tol=1e-7, 
                  calc_grad=True, superharmonic_filter=None,
                  constraint_scale=1.0):
        """
        Residual for the basic Variable Phase Resonance
        Nonlinear Modes (VPRNM).

        Method adds a constraint to HBM to follow a superharmonic resonance.

        Parameters
        ----------
        UwF : (N*Nhc+2,) numpy.ndarray
            Global harmonic degrees of freedom, all DOFs for each harmonic
            component and then the next harmonic component in `h`.
            These are followed by the frequency in rad/s of first harmonic
            and the force magnitude scalingn of `Fl` (zeroth harmonic is not
            scaled).
        h : 1D np.array, sorted
            List of included harmonics, sorted and without repeats.
            Harmonics should be positive integers or zero.
        rhi : int
            Superharmonic resonance harmonic number of interest, 
            must be included in `h`.
        Fl : (N*Nhc,) numpy.ndarray
            Applied external forcing harmonic coefficients in the same ordering
            as displacements in `UwF`.
            All harmonics except harmonic 0 are multiplied in this method
            by `UwF[-1]`.
        Nt : int, power of 2, optional
            Number of time steps for AFT.
            The default is 128.
        aft_tol : float, optional
            Tolerance for AFT.
            The default is 1e-7.
        calc_grad : bool
            Flag where `True` indicates that the gradients should be calculated
            and returned. If `False`, then returns only `(R,)` as a tuple.
            `False` should only be passed if all nonlinear forces have AFT
            methods that accept the `calc_grad` keyword.
            If `True`, the argument is not passed to nonlinear forces.
            The default is `True`.
        superharmonic_filter : None or (N,) numpy.ndarray, optional
            If None, VPRNM is calculated without a modal filter.
            If a `numpy.ndarray`, then VPRNM is modally filtered with the
            array.
            The modal filter is applied to the superharmonic resonance to
            extract a specific mode.
            The default is None.
        constraint_scale : float
            Number to scale the residual of the constraint equation by.
            This is useful when a solver does not put sufficient weight on
            the constraint equation and just solves the HBM equations ignoring
            the constraint. It may need to be dynamically updated between
            solutions along continuation to avoid problems.
            The default is 1.0.

        Returns
        -------
        R : (N*Nhc+1,) numpy.ndarray
            Evaluated residual for VPRNM analysis.
        dRdUw : (N*Nhc+1,N*Nhc+1) numpy.ndarray
            Derivative of `R` with respect to `Uw = UwF[:-1]`.
        dRdw : (N*Nhc+1,) numpy.ndarray
            Derivative of `R` with respect to `F = UwF[-1]`.

        See Also
        --------
        hbm_res : 
            Harmonic balance residual with a different input/output
            that allows for continuation with respect to frequency.
            See documentation of this function for a full list of HBM variants.

        Notes
        -----
        The number of harmonic components is 
        `Nhc = tmdsimpy.utils.harmonic.Nhc(h)`.

        Theory for VPRNM is developed in [1]_, [2]_, [3]_.

        References
        ----------

        .. [1]
           Porter, J. H., and M. R. W. Brake. 2024. "Tracking Superharmonic
           Resonances for Nonlinear Vibration of Conservative and Hysteretic
           Single
           Degree of Freedom Systems." Mechanical Systems and Signal Processing
           215:111410. https://doi.org/10.1016/j.ymssp.2024.111410.
           arXiv:2401.08790
           
        .. [2]
           Porter, J. H., and M. R. W. Brake. Under Review. "Efficient Model
           Reduction and Prediction of Superharmonic Resonances in Frictional
           and Hysteretic Systems." Mechanical Systems and Signal Processing.
           arXiv:2405.15918.

        .. [3]
           Porter, J. H. 2024. Modal Interactions and Jointed Structures.
           PhD Thesis.
           Rice University.

        """
        
        #############
        # Initialization
        
        # Initialize output shapes
        R = np.zeros_like(UwF[:-1])
        
        # Harmonic indices
        Nhc = hutils.Nhc(h)
        Ndof = self.M.shape[0]
        
        #############
        # Baseline HBM for first set of Equations
        
        # Evaluate the N Harmonic Balance Equations
        Fscale = np.copy(Fl)
        Fscale[(h[0]==0)*Ndof:] *= UwF[-1]
        
        Rhbm_dRhbmdU_dRhbmdw = self.hbm_res(UwF[:-1], Fscale, h, 
                                              Nt=Nt, aft_tol=aft_tol,
                                              calc_grad=calc_grad)
        
        #############
        # VPRNM Equation
        Rvprnm_dRdUw_vprnm = self.vprnm_single_eqn(UwF[:-2], UwF[-2], h, rhi, 
                                    Nt=Nt, 
                                    aft_tol=aft_tol, 
                                    calc_grad=calc_grad,
                                    superharmonic_filter=superharmonic_filter)
        
        #############
        # Assemble Full Gradient and Residual
        
        # Add Orthogonality constrait using Frhi 
        # (harmonic rhi orthogonal to forcing for resonance)
        # breakpoint()
        R = np.hstack((Rhbm_dRhbmdU_dRhbmdw[0], 
                       constraint_scale*Rvprnm_dRdUw_vprnm[0]))
        
        if calc_grad:
            
            # Initialize Memory
            dRdUw = np.zeros((R.shape[0], R.shape[0]))
            dRdF = np.zeros_like(R)
        
            # HBM Derivatives
            dRdUw[:Ndof*Nhc, :Ndof*Nhc] = Rhbm_dRhbmdU_dRhbmdw[1]
            dRdUw[:Ndof*Nhc, -1]   = Rhbm_dRhbmdU_dRhbmdw[2]
            
            # Derivatives of VPRNM Equation
            dRdUw[-1] = constraint_scale*Rvprnm_dRdUw_vprnm[1]
            
            # negative since HBM is internal minus external force 
            # (but not static force)
            dRdF[:Fl.shape[0]] = -Fl
            dRdF[:Ndof*(h[0]==0)] = 0.0
            
            # Return outputs include those needed for continuation equation        
            return R, dRdUw, dRdF
        else:
            return (R,)
    
    def hbm_amp_control_res(self, UFw, Fl, h, recov, amp, order,
                            Nt=128, aft_tol=1e-7, calc_grad=True):
        """
        Residual for harmonic balance method (HBM) with amplitude control.
        
        Control is applied exclusively to the 1st harmonic.
        
        Parameters
        ----------
        UFw : (N*Nhc+2,) numpy.ndarray
            Global harmonic degrees of freedom, all DOFs for each harmonic
            component and then the next harmonic component in `h`.
            These are followed by
            the force scaling for all harmonics except the zeroth harmonic
            and
            the frequency in rad/s of first harmonic.
        Fl : (N*Nhc,) numpy.ndarray
            Applied external forcing harmonic coefficients in the same ordering
            as displacements in `UFw`.
            Static (zeroth harmonic) forces are not scaled. Other harmonics are
            scaled in this method by `UFw[-2]`.
        h : 1D np.array, sorted
            List of included harmonics, sorted and without repeats.
            Harmonics should be positive integers or zero.
        recov : (N,) numpy.ndarray
            Recovery matrix to extract the DOF that has amplitude
            and phase control.
        amp : float
            Amplitude that the recovered DOF is controlled to 
            (defined by `recov` and `order`).
        order : int, positive or zero
            Exponent on frequency to multiply the controlled displacement by.
            This allows for control of 0=displacement, 1=velocity,
            or 2=acceleration.
            Control does not consider sign changes due to derivatives when
            controlling `order != 0`.
        Nt : int, power of 2, optional
            Number of time steps for AFT.
            The default is 128.
        aft_tol : float, optional
            Tolerance for AFT.
            The default is 1e-7.
        calc_grad : bool
            Flag where `True` indicates that the gradients should be calculated
            and returned. If `False`, then returns only `(R,)` as a tuple.
            `False` should only be passed if all nonlinear forces have AFT
            methods that accept the `calc_grad` keyword.
            If `True`, the argument is not passed to nonlinear forces.
            The default is `True`.

        Returns
        -------
        R : (N*Nhc+1,) numpy.ndarray
            Evaluated residual for HBM analysis with amplitude control.
        dRdUF : (N*Nhc+1,N*Nhc+1) numpy.ndarray
            Derivative of `R` with respect to `UF = UFw[:-1]`.
        dRdw : (N*Nhc+1,) numpy.ndarray
            Derivative of `R` with respect to `w = UFw[-1]`.

        See Also
        --------
        hbm_res : 
            Harmonic balance residual for constant force input to the system.
            See documentation of this function for a full list of HBM variants.
        tmdsimpy.utils.harmonic.predict_harmonic_solution :
            Function for generating initial guesses to HBM type problems.

        Notes
        -----
        The number of harmonic components is
        `Nhc = tmdsimpy.utils.harmonic.Nhc(h)`
        """
        
        ###### Basic Initialization
        Ndof = self.M.shape[0]
        Nhc = hutils.Nhc(h)
        
        h0 = h[0] == 0
        Uw = np.hstack((UFw[:-2], UFw[-1]))
        
        ###### Static v. Dynamic Forces
        Fstat = np.copy(Fl)
        Fstat[h0*Ndof:] = 0.0
        
        Fdyn = np.copy(Fl)
        Fdyn[:h0*Ndof] = 0.0
        
        ###### Normal HBM
        Rhbm_dRhbmdU_dRhbmdw = self.hbm_res(Uw, Fstat + Fdyn*UFw[-2], h, 
                                               Nt=Nt, aft_tol=aft_tol, 
                                               calc_grad=calc_grad)
        
        ###### Apply the amplitude control
        # 1st harmonic displacements
        u1c = UFw[h0*Ndof:(1+h0)*Ndof]
        u1s = UFw[(1+h0)*Ndof:(2+h0)*Ndof]
        
        udofc = recov @ u1c
        udofs = recov @ u1s
        
        # Augmented Equation for amplitude constraint
        # Power is twice the order of the derivative being controlled because
        # residual is on the amplitude squared.
        Raug =  (UFw[-1]**(2*order))*(udofc**2 + udofs**2) - amp**2
        
        R = np.hstack((Rhbm_dRhbmdU_dRhbmdw[0], Raug))
        
        if calc_grad:
            # dRhbmdF = -Fl # don't create extra memory at this point
            
            dRaugdUF = np.zeros((1, Nhc*Ndof+1))
            dRaugdUF[0,     h0*Ndof:(1+h0)*Ndof] = (UFw[-1]**(2*order))*(2*udofc*recov)
            dRaugdUF[0, (1+h0)*Ndof:(2+h0)*Ndof] = (UFw[-1]**(2*order))*(2*udofs*recov)
            
            # dRaugdUF[0, -1] = 0 # augmented equation is independent of force scale
            
            dRaugdw = (2*order)*(UFw[-1]**((2*order)-1))*(udofc**2 + udofs**2)
            
            dRdUF = np.vstack((np.hstack((Rhbm_dRhbmdU_dRhbmdw[1], -Fdyn.reshape(-1,1))),
                               dRaugdUF))
        
            dRdw = np.hstack((Rhbm_dRhbmdU_dRhbmdw[2], dRaugdw))
        
            return R, dRdUF, dRdw
        else:
            return (R,)
    
    def hbm_amp_phase_control_res(self, UFcFsw, Fl, h, recov, amp, order, 
                            Nt=128, aft_tol=1e-7, calc_grad=True):
        """
        Amplitude Control with Harmonic Balance Method (HBM) rather than fixing
        force at a constant value and/or phase. 
        The phase at the response point is controlled to be only cosine by 
        varying the phase of the forcing.
        
        Control is applied exclusively to the 1st harmonic
        
        For documentation: Nhc is the number of harmonics
        and
        Ndof is the number of Degree of Freedoms

        Parameters
        ----------
        UFcFsw : (Nhc*Ndof+3,) numpy.ndarray
            Harmonic Displacements, Force Scaling Cosine, Force Scaling
            Sine, Frequency.
            Harmonic Displacements are all of zeroth, 1c, 1s, 2c, 2s etc.
        Fl : (Nhc*Ndof,) numpy.ndarray
            Forcing Vector without scaling for all harmonics 
            Static force is correctly scaled already (if included).
            Forcing of the first harmonic is of the form
            UFcFsw[-3]*Fl_cos1*cos(w*t) + UFcFsw[-2]*Fl_cos1*sin(w*t)
            where Fl_cos1 is the first harmonic cosine terms in Fl.
            Forcing on all other harmonics is ignored. Forcing input for
            sine terms of the first harmonic is currently ignored. 
        h : 1D numpy.ndarray, sorted
            List of harmonics used, must be sorted and include 1st harmonic.
        recov : (Ndof,) numpy.ndarray
            Recovery matrix for the DOF that has amplitude and phase control 
        amp : float
            Amplitude that the recovered DOF is controlled to 
        order : int, positive or zero
            Order of derivative of interest, but this is just used as an 
            exponent on the frequency 
            (e.g., negative signs from taking 2 derivatives of cos/sine are 
            ignored in phase constraint). 
            Control is always applied to have response amplitude at recovery 
            DOF that is purely cosine and the same sign as amp.
        Nt : int, power of 2
            Number of time steps for AFT. 
            The default is 128.
        aft_tol : float
            Tolerance for AFT evaluations. 
            The default is 1e-7.
        calc_grad : boolean
            Flag where True indicates that the gradients should be calculated 
            and returned. If False, then returns only (R,) as a tuple. 
            False should only be passed if all nonlinear forces have aft 
            methods that accept the calc_grad keyword.
            The default is True.

        Returns
        -------
        R : (Nhc*Ndof+2,) numpy.ndarray
            Residual vector, always returned as first entry of a tuple.
            First Nhc*Ndof entries correspond to HBM solution.
            Second to last is cosine amplitude being equal to desired value.
            Last is sine amplitude equal to zero.
        dRdUFcFs : (Nhc*Ndof+2,Nhc*Ndof+2) numpy.ndarray
            Derivative of the residual w.r.t. UFcFs. 
            Only returned if calc_grad=True (default behavior).
        dRdw : (Nhc*Ndof+2,) numpy.ndarray
            Derivative w.r.t. frequency.
            Only returned if calc_grad=True (default behavior).

        See Also
        --------
        hbm_res : 
            Harmonic balance residual for constant force input to the system.
            See documentation of this function for a full list of HBM variants.
        tmdsimpy.utils.harmonic.predict_harmonic_solution :
            Function for generating initial guesses to HBM type problems.
        """
        
        # Size of Problem
        Ndof = self.M.shape[0]
        Nhc = hutils.Nhc(h)
        h0 = h[0] == 0
        
        # Baseline HBM Solution
        Fl_hbm = np.zeros(Nhc*Ndof)
        Fl_hbm[:h0*Ndof] = Fl[:h0*Ndof]
        Fl_hbm[h0*Ndof:(h0+1)*Ndof] = UFcFsw[-3] * Fl[h0*Ndof:(h0+1)*Ndof]
        Fl_hbm[(h0+1)*Ndof:(h0+2)*Ndof] = UFcFsw[-2] * Fl[h0*Ndof:(h0+1)*Ndof]
        
        R_dRdU_dRdw_hbm = self.hbm_res(np.hstack((UFcFsw[:-3], UFcFsw[-1])), 
                     Fl_hbm, h, Nt=Nt, aft_tol=aft_tol, calc_grad=calc_grad)
        
        # Recovery Amplitude Constraint
        amp_cos = (UFcFsw[-1]**order) * (recov @ UFcFsw[h0*Ndof:(h0+1)*Ndof])
        amp_sin = (UFcFsw[-1]**order) * (recov @ UFcFsw[(h0+1)*Ndof:(h0+2)*Ndof])
        
        R = np.hstack((R_dRdU_dRdw_hbm[0], (amp_cos-amp), amp_sin))
        
        if calc_grad:
            dRdUFcFs = np.zeros((Nhc*Ndof+2,Nhc*Ndof+2))
            dRdw = np.zeros((Nhc*Ndof+2,))
            
            # Copy HBM solutions
            dRdUFcFs[:Ndof*Nhc,:Ndof*Nhc] = R_dRdU_dRdw_hbm[1]
            dRdw[:Ndof*Nhc] = R_dRdU_dRdw_hbm[2]
            
            # Two New Columns / Unknowns
            dRdUFcFs[h0*Ndof:(h0+1)*Ndof, -2] = -Fl[h0*Ndof:(h0+1)*Ndof]
            dRdUFcFs[(h0+1)*Ndof:(h0+2)*Ndof, -1] = -Fl[h0*Ndof:(h0+1)*Ndof]
            
            # Last 2 equations
            dRdUFcFs[-2, h0*Ndof:(h0+1)*Ndof] = (UFcFsw[-1]**order) * recov
            dRdUFcFs[-1, (h0+1)*Ndof:(h0+2)*Ndof] = (UFcFsw[-1]**order) * recov
            
            dRdw[-2] = order*amp_cos/UFcFsw[-1]
            dRdw[-1] = order*amp_sin/UFcFsw[-1]
            
            return R, dRdUFcFs, dRdw
        else:
            return (R,)
        
    def hbm_amp_phase_control_dA_res(self, UFcFsA, Fl, h, recov, w, order, 
                            Nt=128, aft_tol=1e-7, calc_grad=True):
        """
        Amplitude Control with Harmonic Balance Method (HBM) rather than fixing
        force at a constant value and/or phase. 
        The phase at the response point is controlled to be only cosine by 
        varying the phase of the forcing.
        
        This version has outputs for continuation w.r.t. amplitude at constant 
        frequency
        
        Control is applied exclusively to the 1st harmonic
        
        For documentation: Nhc is the number of harmonics
        and
        Ndof is the number of Degree of Freedoms

        Parameters
        ----------
        UFcFsA : (Nhc*Ndof+3,) numpy.ndarray
            Harmonic Displacements, Force Scaling Cosine, Force Scaling
            Sine, Amplitude Level.
            Harmonic Displacements are all of zeroth, 1c, 1s, 2c, 2s etc.
        Fl : (Nhc*Ndof,) numpy.ndarray
            Forcing Vector without scaling for all harmonics 
            Static force is correctly scaled already (if included).
            Forcing of the first harmonic is of the form
            UFcFsw[-3]*Fl_cos1*cos(w*t) + UFcFsw[-2]*Fl_cos1*sin(w*t)
            where Fl_cos1 is the first harmonic cosine terms in Fl.
            Forcing on all other harmonics is ignored. Forcing input for
            sine terms of the first harmonic is currently ignored. 
        h : 1D numpy.ndarray, sorted
            List of harmonics used, must be sorted and include 1st harmonic.
        recov : (Ndof,) numpy.ndarray
            Recovery matrix for the DOF that has amplitude and phase control 
        w : float
            Frequency for HBM, rad/s.
        order : int, positive or zero
            Order of derivative of interest, but this is just used as an 
            exponent on the frequency 
            (e.g., negative signs from taking 2 derivatives of cos/sine are 
            ignored in phase constraint). 
            Control is always applied to have response amplitude at recovery 
            DOF that is purely cosine and the same sign as amp.
        Nt : int, power of 2
            Number of time steps for AFT. 
            The default is 128.
        aft_tol : float
            Tolerance for AFT evaluations. 
            The default is 1e-7.
        calc_grad : boolean
            Flag where True indicates that the gradients should be calculated 
            and returned. If False, then returns only (R,) as a tuple. 
            False should only be passed if all nonlinear forces have aft 
            methods that accept the calc_grad keyword.
            The default is True.

        Returns
        -------
        R : (Nhc*Ndof+2,) numpy.ndarray
            Residual vector, always returned as first entry of a tuple.
            First Nhc*Ndof entries correspond to HBM solution.
            Second to last is cosine amplitude being equal to desired value.
            Last is sine amplitude equal to zero.
        dRdUFcFs : (Nhc*Ndof+2,Nhc*Ndof+2) numpy.ndarray
            Derivative of the residual w.r.t. UFcFs. 
            Only returned if calc_grad=True (default behavior).
        dRdA : (Nhc*Ndof+2,) numpy.ndarray
            Derivative w.r.t. amplitude.
            Only returned if calc_grad=True (default behavior).

        See Also
        --------
        hbm_res : 
            Harmonic balance residual for constant force input to the system.
            See documentation of this function for a full list of HBM variants
        tmdsimpy.utils.harmonic.predict_harmonic_solution :
            Function for generating initial guesses to HBM type problems.
        """
        
        R_dRdUFcFs_dRdw = self.hbm_amp_phase_control_res(
                                np.hstack((UFcFsA[:-1], w)), 
                                Fl, h, recov, UFcFsA[-1], order, 
                                Nt=Nt, aft_tol=aft_tol, calc_grad=calc_grad)
        
        if calc_grad:
            
            dRdA = np.zeros_like(R_dRdUFcFs_dRdw[2])
            dRdA[-2] = -1.0
            
            return R_dRdUFcFs_dRdw[0],R_dRdUFcFs_dRdw[1],dRdA
        else:
            return (R_dRdUFcFs_dRdw[0],)
        
        
    def vprnm_amp_phase_res(self, UFcFswA, Fl, h, rhi, recov, order, 
                            Nt=128, aft_tol=1e-7, 
                            calc_grad=True, superharmonic_filter=None,
                            constraint_scale=1.0):
        """
        Residual for the Variable Phase Resonance 
        Nonlinear Modes (VPRNM) with extra constraints.
        
        Method adds a constraint to HBM to follow a superharmonic resonance
        and constraints on amplitude and phase of the response to make it
        easier to solve the set of equations.
        
        Parameters
        ----------
        UFcFswA : (N*Nhc+4,) numpy.ndarray
            Global harmonic degrees of freedom, all DOFs for each harmonic
            component and then the next harmonic component in `h`.
            These are followed by 
            the force scaling for first harmonic cosine external force,
            the force scaling for first harmonic sine external force,
            the frequency in rad/s of first harmonic,
            and the amplitude level.
        Fl : (N*Nhc,) numpy.ndarray
            Applied external forcing harmonic coefficients in the same ordering
            as displacements in `UFcFswA`.
            The zeroth harmonic force is taken directly from this.
            The first harmonic cosine force coefficients are applied as cosine
            and sine forcing with the scaling of `UFcFswA[-4]` and
            `UFcFswA[-3]` respectively.
        h : 1D np.array, sorted
            List of included harmonics, sorted and without repeats.
            Harmonics should be positive integers or zero.
        rhi : int
            Superharmonic resonance harmonic number of interest, 
            must be included in `h`.
        recov : (N,) numpy.ndarray
            Recovery matrix to extract the DOF that has amplitude 
            and phase control.           
        order : int, positive or zero
            Exponent on frequency to multiply the controlled displacement by.
            This allows for control of 0=displacement, 1=velocity, 
            or 2=acceleration.
            Control does not consider sign changes due to derivatives when
            controlling `order != 0`.
        Nt : int, power of 2, optional
            Number of time steps for AFT. 
            The default is 128.
        aft_tol : float, optional
            Tolerance for AFT. 
            The default is 1e-7.
        calc_grad : bool
            Flag where `True` indicates that the gradients should be calculated
            and returned. If `False`, then returns only `(R,)` as a tuple.
            `False` should only be passed if all nonlinear forces have AFT 
            methods that accept the `calc_grad` keyword.
            If `True`, the argument is not passed to nonlinear forces.
            The default is `True`.
        superharmonic_filter : None or (N,) numpy.ndarray, optional
            If None, VPRNM is calculated without a modal filter. 
            If a `numpy.ndarray`, then VPRNM is modally filtered with the
            array.
            The modal filter is applied to the superharmonic resonance to 
            extract a specific mode.
            The default is None.
        constraint_scale : float
            Number to scale the residual of the constraint equation by. 
            This is useful when a solver does not put sufficient weight on
            the constraint equation and just solves the HBM equations ignoring
            the constraint. It may need to be dynamically updated between 
            solutions along continuation to avoid problems.
            The default is 1.0.
            
        Returns
        -------
        R : (N*Nhc+3,) numpy.ndarray
            Evaluated residual for VPRNM analysis.
        dRdUFcFsw : (N*Nhc+3,N*Nhc+3) numpy.ndarray
            Derivative of `R` with respect to `UFcFsw = UFcFswA[:-1]`.
        dRdA : (N*Nhc+3,) numpy.ndarray
            Derivative of `R` with respect to `A = UFcFswA[-1]`.
        
        See Also
        --------
        hbm_res : 
            Harmonic balance residual with a different input/output
            that allows for continuation with respect to frequency.
            See documentation of this function for a full list of HBM variants.
        vprnm_res : 
            VPRNM implementation without additional amplitude and phase 
            constraints (constant force excitation)
        tmdsimpy.utils.harmonic.predict_harmonic_solution : 
            Function for generating initial guesses to HBM type problems.
        
        
        Notes
        -----
        The number of harmonic components is 
        `Nhc = tmdsimpy.utils.harmonic.Nhc(h)`.
        
        Theory for VPRNM is developed in [1]_, [2]_, [3]_.

        References
        ----------

        .. [1]
           Porter, J. H., and M. R. W. Brake. 2024. "Tracking Superharmonic
           Resonances for Nonlinear Vibration of Conservative and Hysteretic
           Single 
           Degree of Freedom Systems." Mechanical Systems and Signal Processing 
           215:111410. https://doi.org/10.1016/j.ymssp.2024.111410.
           arXiv:2401.08790
           
        .. [2]
           Porter, J. H., and M. R. W. Brake. Under Review. "Efficient Model 
           Reduction and Prediction of Superharmonic Resonances in Frictional
           and Hysteretic Systems." Mechanical Systems and Signal Processing.
           arXiv:2405.15918.

        .. [3]
           Porter, J. H. 2024. Modal Interactions and Jointed Structures.
           PhD Thesis.
           Rice University.

        """
                
        hbm_R_dRdUFcFs_dRdw = self.hbm_amp_phase_control_res(UFcFswA[:-1], 
                                                        Fl, h, 
                                                        recov, UFcFswA[-1], 
                                                        order, 
                                                        Nt=Nt, 
                                                        aft_tol=aft_tol, 
                                                        calc_grad=calc_grad)
        
        vprnm_R_dRdUw = self.vprnm_single_eqn(UFcFswA[:-4], UFcFswA[-2], 
                                    h, rhi, 
                                    Nt=Nt, 
                                    aft_tol=aft_tol, 
                                    calc_grad=calc_grad,
                                    superharmonic_filter=superharmonic_filter)
        
        R = np.hstack((hbm_R_dRdUFcFs_dRdw[0], 
                       constraint_scale*vprnm_R_dRdUw[0]))
        
        if calc_grad:
            dRdUFcFsw = np.block([[hbm_R_dRdUFcFs_dRdw[1], 
                                   hbm_R_dRdUFcFs_dRdw[2].reshape(-1,1)],
                                  [constraint_scale*vprnm_R_dRdUw[1][:-1], 0, 
                                   0, constraint_scale*vprnm_R_dRdUw[1][-1]]])
            
            dRdA = np.zeros_like(R)
            dRdA[-3] = -1.0
            
            return R, dRdUFcFsw, dRdA
        else:
            return (R,)
        
        
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
    
    
    
    