import numpy as np


def Nhc(h):
    """
    Quick function to calculate the number of harmonic components

    Parameters
    ----------
    h : 1D np.array 
        Harmonics that should be included. E.g., numpy.array(range(5))

    Returns
    -------
    Nhc : int
        Number of harmonic components (1 for zeroth, 2 for rest)

    """
    
    h_unique = np.unique(h)
    
    assert len(h_unique) == len(h), 'Repeated Harmonics in h are not allowed.'
   
    return 2*(h !=0).sum() + (h==0).sum()

def harmonic_stiffness(M, C, K, w, h, calc_grad=True, only_C=False):
    """
    Returns the harmonic stiffness and its derivative w.r.t. frequency w. 
    This implementation is optimized for small systems.

    Parameters
    ----------
    M : (N,N) numpy.ndarray
        Mass Matrix
    C : (N,N) numpy.ndarray
        Damping Matrix
    K : (N,N) numpy.ndarray
        Stiffness Matrix
    w : float
        Frequency (fundamental)
    h : 1D numpy.ndarray
        List of harmonics, zeroth harmonic must be first if included (best 
        practice for it to be sorted order).
        The number of harmonic components is 
        Nhc = tmdsimpy.harmonic_utils.Nhc(h)
    calc_grad: bool, optional
        If True, both outputs are calculated. If False, only E is calculated 
        and returned. Return is still in a tuple so 
        E=harmonic_stiffness_many_dof(...)[0] always works.
        The default is True.
    only_C : bool, optional
        Flag to indicate that M and K should both be assumed to be zero. This 
        is a call that is used to calculate a gradient in EPMC, so it is worth
        optimizing to cut out half of the operations here. M and K are 
        completely ignored in this case and do not need to be passed in with 
        correct shapes or values. The default is False.

    Returns
    -------
    E : (N*Nhc, N*Nhc) numpy.ndarray
        Square stiffness matrix corresponding to linear properties at every
        harmonic. Ordered as all dofs for each of (if h[0]==0 is included)
        [0, 
        cos(h[1]*w*t), sin(h[1]*w*t), 
        cos(h[2]*w*t), sin(h[2]*w*t)]
        If only_C==True, then only the damping properties are applied.
    dEdw : (N*Nhc, N*Nhc) numpy.ndarray
        Square derivative matrix. Not returned if calc_grad==False.
    
    """
    
    nd = C.shape[0]
    
    Nhc2 = Nhc(h) # Number of Harmonic Components
    
    E = np.zeros((Nhc2*nd, Nhc2*nd))
    
    if calc_grad:
        dEdw = np.zeros((Nhc2*nd, Nhc2*nd))
    
    # Starting index for first harmonic
    zi = 1*(h[0] == 0)
    
    # apply not here so that boolean does not have to be repeatedly applied
    include_KM = not only_C
    
    if zi == 1 and include_KM:
        E[:nd, :nd] = K
    
    for hind in range(zi, h.shape[0]):
        
        TR = (1.0*h[hind]*w)*C
        BL = (-1.0*h[hind]*w)*C
        
        E[nd*(hind*2 - zi):nd*(hind*2 - zi+1), \
          nd*(hind*2 - zi+1):nd*(hind*2 - zi+2)] = TR
            
        E[nd*(hind*2 - zi+1):nd*(hind*2 - zi+2), \
          nd*(hind*2 - zi):nd*(hind*2 - zi+1)] = BL
        
        if include_KM:
            TL = K + (-1.0*(h[hind]*w)**2)*M
            BR = K + (-1.0*(h[hind]*w)**2)*M
        
            E[nd*(hind*2 - zi):nd*(hind*2 - zi+1), \
              nd*(hind*2 - zi):nd*(hind*2 - zi+1)] = TL
            
            E[nd*(hind*2 - zi+1):nd*(hind*2 - zi+2), \
              nd*(hind*2 - zi+1):nd*(hind*2 - zi+2)] = BR
            
        if calc_grad:
            
            TRdw = h[hind]*C
            BLdw = (-1.0*h[hind])*C
            
            dEdw[nd*(hind*2 - zi):nd*(hind*2 - zi+1), \
              nd*(hind*2 - zi+1):nd*(hind*2 - zi+2)] = TRdw
                
            dEdw[nd*(hind*2 - zi+1):nd*(hind*2 - zi+2), \
              nd*(hind*2 - zi):nd*(hind*2 - zi+1)] = BLdw
            
            if include_KM:

                TLdw = (-2.0*w*(h[hind]**2))*M
                BRdw = (-2.0*w*(h[hind]**2))*M
            
                dEdw[nd*(hind*2 - zi):nd*(hind*2 - zi+1), \
                  nd*(hind*2 - zi):nd*(hind*2 - zi+1)] = TLdw
                 
                dEdw[nd*(hind*2 - zi+1):nd*(hind*2 - zi+2), \
                  nd*(hind*2 - zi+1):nd*(hind*2 - zi+2)] = BRdw
                
    if calc_grad:
        return E, dEdw
    else:
        return (E,)

def time_series_deriv(Nt, h, X0, order):
    """
    Returns Derivative of a time series defined by a set of harmonics
    
    Parameters
    ----------
    Nt : int, power of 2
        Number of times considered, must be even
    h : 1D numpy.ndarray
        Harmonics considered, 0th harmonic must be first if included
    X0 : (Nhc, nd) numpy.ndarray
        Harmonic Coefficients for columns corresponding to degrees of freedom
        and rows corresponding to different harmonic components
        Nhc = Nhc(h)
    order : int
        Order of the derivative returned
    
    Returns
    -------
    x_t : (Nt, nd) numpy.ndarray
        Time series of each DOF
    """
    
    #Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
    
    assert ((h == 0).sum() == 0 or h[0] == 0), 'Zeroth harmonic must be first'
    
    nd = X0.shape[1] # Degrees of Freedom
    Nh = np.max(h)
    
    # Create list including all harmonic components
    X0full = np.zeros((2*Nh+1, nd))
    if h[0] == 0:
        X0full[0, :] = X0[0, :]
        X0full[2*h[1:]-1, :] = X0[1::2, :]
        X0full[2*h[1:], :] = X0[2::2, :]
    else:
        X0full[2*h-1, :] = X0[0::2, :]
        X0full[2*h, :] = X0[1::2, :]
        
    # Check that sufficient time is considered
    assert Nt > 2*Nh + 1, 'More times are required to avoid truncating harmonics.'
    
    if order > 0:
        D1 = np.zeros((2*Nh+1, 2*Nh+1))
        
        for k in h[h != 0]:
            # Only rotates the derivatives for the non-zero harmonic components
            cosrows = (k-1)*2 + 1
            sinrows = (k-1)*2 + 2
            
            D1[cosrows, sinrows] = k
            
            # -k can give the wrong number if it is a positive only integer type 
            # (e.g., from the MATLAB import test). In those cases -k != -1*k
            D1[sinrows, cosrows] = -1*k 
            
        # This is not particularly fast, consider optimizing this portion.
        #   D could be constructed just be noting if rows flip for odd/even
        #   and sign changes as appropriate.
        D = np.linalg.matrix_power(D1, order)
        
        X0full = D @ X0full
    
    # Extend X0full to have coefficients corresponding to Nt times for ifft
    #   Previous MATLAB implementation did this before rotating harmonics, but
    #   that seems rather inefficient in increasing the size of the matrix 
    #   multiplication
    Nht = int(Nt/2 -1)
    X0full = np.vstack((X0full,np.zeros((2*(Nht-Nh), nd)) ))
    Nt = 2*Nht+2

    # Fourier Coefficients    
    Xf = np.vstack((2*X0full[0, :], \
         X0full[1::2, :] - 1j*X0full[2::2], \
         np.zeros((1, nd)), \
         X0full[-2:0:-2, :] + 1j*X0full[-1:1:-2]))
        
    Xf = Xf * (Nt/2)
         
    assert Xf.shape[0] == Nt, 'Unexpected length of Fourier Coefficients'
    
    x_t = np.real(np.fft.ifft(Xf, axis=0))
    
    return x_t

def get_fourier_coeff(h, x_t):
    """
    Calculates the Fourier coefficients corresponding to the harmonics in h of
    the input x_t

    Parameters
    ----------
    h : 1D numpy.ndarray
        Harmonics considered, 0th harmonic must be first if included
    x_t : (Nt, nd) numpy.ndarray
        Time series of each DOF

    Returns
    -------
    v : (Nhc, nd) numpy.ndarray
        Containing fourier coefficients of harmonics h. Nhc = Nhc(h)
    """
    
    Nt, nd = x_t.shape
    Nhc = 2*(h != 0).sum() + (h == 0).sum() # Number of Harmonic Components
    n = h.shape[0] - (h[0] == 0)
    
    assert ((h == 0).sum() == 0 or h[0] == 0), 'Zeroth harmonic must be first'
    
    v = np.zeros((Nhc, nd))
    
    xf = np.fft.fft(x_t, axis=0)
        
    if h[0] == 0:
        v[0, :] = np.real(xf[0, :])/Nt
        zi = 1
    else:
        zi = 0
        
    for i in range(n):
        hi = h[i + zi]
        v[2*i+zi] = np.real(xf[hi, :]) / (Nt/2)
        v[2*i+1+zi] = -np.imag(xf[hi, :]) / (Nt/2)
    
    return v

def harmonic_wise_conditioning(X, Ndof, h, delta=1e-4):
    """
    Function returns a conditioning vector for harmonic solutions. 
    Each harmonic is assigned a constant equal to the larger of delta or the
    mean absolute value of all components at that harmonic in X (sine and cosine)

    Parameters
    ----------
    X : (Ndof*Nhc+m,) numpy.ndarray
        Baseline harmonics values, consecutive sets of Ndof correspond to 
        harmonic components as listed in h (sine and cosine for h[i] != 0).
        Here Nhc = harmonic_utils.Nhc(h).
        The m extra components will be individually assigned delta or their 
        absolute value.
    Ndof : int
        Number of degrees of freedom associated with the model.
    h : 1D numpy.ndarray, sorted
        List of harmonics
    delta : scalar or 1D array like
        Small value to prevent divide by zero (minimum value that will be 
        returned in CtoP).
        When delta is array like, the array entries correspond to minimum 
        values for each harmonic in h, and then a single minimum value for
        all terms after the harmonic unknowns.
        
    Returns
    -------
    CtoP : (Ndof*Nhc+m,) numpy.ndarray
        Vector of same size as X to convert Xphysical=CtoP*Xconditioned

    """
    
    m = X.shape[0] - Nhc(h)*Ndof
    extras = m > 0
    
    # Default Conditioning level when some components are small
    if type(delta) == float:
        CtoP = delta*np.ones_like(X) 
    else: 
        h0 = h[0] == 0
        
        CtoP = np.ones((Ndof*2, h.shape[0]+extras))*np.atleast_2d(delta)
        CtoP = CtoP.reshape(-1, order='F')[Ndof*h0:]
        
        if extras:
            # Trim off extra at end
            CtoP = CtoP[:(m-2*Ndof)]
        
    # Loop over Harmonics and Potentially increase each harmonic
    haszero = 0
    for hindex in range(len(h)):
        if h[hindex] == 0:
            # Normalize only Ndof variables
            inds = slice(0, Ndof)
            haszero = 1
            assert hindex == 0, 'Zeroth harmonic must be first.'
        else:
            # Normalize sine and cosine components together
            inds = slice((2*hindex-haszero)*Ndof, (2*hindex+2-haszero)*Ndof)
            
        CtoP[inds] = np.maximum(CtoP[inds], np.mean(np.abs(X[inds])))
        
    if extras:
        CtoP[-m:] = np.maximum(CtoP[-m:], np.abs(X[-m:]))
        
    return CtoP

def predict_harmonic_solution(vib_sys, w, Fl, h, solver,
                         equation, 
                         Xstat=None,
                         fmag=None, 
                         control_amp=None, control_recov=None, 
                         control_order=None,
                         rhi=None, neigs=3, vib_sys_nl=None):
    """
    Generate an initial guess to an harmonic balance method (HBM) type problem 
    based on a linear system

    Parameters
    ----------
    vib_sys : tmdsimpy.VibrationSystem
        Vibration system for the initial prediction. The initial prediction is
        linear, so this should be linear around the state of interest (e.g., 
        for frictional systems create a new VibrationSystem around the 
        prestressed state using the linearized stiffness).
        This must use mass and stiffness proportional damping.
    w : float
        frequency in rad/s.
    h : 1D numpy.ndarray, sorted
        List of harmonics to include in solution.
    Fl : 1D numpy.ndarray
        Forcing corresponding to the harmonics.
        Only the first harmonic terms of Fl are considered.
        For equations that do amplitude and phase control, Fl should
        only include first harmonic cosine forcing terms to get correct results
    solver : tmdsimpy.NonlinearSolver
        Solver to be used in calculating an eigenproblem for vib_sys.
    equation : {'HBM', 'HBM_AMP', 'HBM_AMP_PHASE', 'VPRNM_AMP_PHASE'}
        Which equation to provide an initial guess for. Note that not all 
        options may be implemented yet.
        Options are 
        HBM=standard HBM for frequency continuation,
        HBM_AMP=Amplitude HBM with constant phase forcing ,
        HBM_AMP_PHASE=Amplitude HBM with constant response phase, variable 
        force phase
        (for frequency continuation),
        VPRNM_AMP_PHASE=VPRNM with amplitude and phase controlled HBM.
        If you need HBM_AMP_PHASE for amplitude continuation, just change the 
        last entry of the returned array to be the desired prediction amplitude
        instead of frequency.
    Xstat : numpy.ndarray, optional
        Static displacement vector for the zeroth harmonic solution. 
        If None, zeros will be returned for the zeroth harmonic.
        The default is None.
    fmag : float, optional
        Scaling for the Fl vector. 
        This only matters if the mode is 'HBM'. 
        The default is None.
    control_amp : float, optional
        Amplitude of response is controlled to. 
        This does not matter for HBM
        The default is None.
    control_recov : numpy.ndarray, optional
        recovery vector which extracts the DOF of interest for amplitude 
        control. 
        The default is None.
    control_order : int, optional
        power of frequency to multiply amplitude control by. 
        The default is None.
    rhi : int, optional
        Higher harmonic of interest to include in initial guess. 
        This only matters for VPRNM.
        The default is None.
    neigs : int, optional
        The number of modes that should be used in constructing the linear
        prediction of the response amplitude.
        The default is 3.
    vib_sys_nl : tmdsimpy.VibrationSystem or None
        Vibration system with nonlinear forces (linear portions do not matter)
        Used exclusively for predicting nonlinear forces acting on superharmonic
        for VPRNM.

    Returns
    -------
    Ulam0 : numpy.ndarray
        Initial guess for the given set of equations. 
        This will generally include harmonic displacements as the first entries. 
        The last few entries vary based on the equation mode.

    """
    
    ############################## 
    # Initial Setup / Problem size etc.
    
    Ndof = vib_sys.M.shape[0]
    Nhc2 = Nhc(h)
    h0 = h[0] ==0
    
    equation = equation.upper()
    
    if equation != 'HBM':
        fmag = 1.0
        
    if equation == 'HBM_AMP_PHASE' or equation == 'VPRNM_AMP_PHASE':
        assert np.sum(np.abs(Fl[(h0+1)*Ndof:(h0+2)*Ndof])) == 0.0, \
            'Initial guess for phase control does not support Fl including sine terms.'
    
    
    ############################## 
    # Linear Initial Response Estimate
    
    Xw_lin = vib_sys.linear_frf(w, fmag*Fl[h0*Ndof:(1+h0)*Ndof], solver,
                                neigs=neigs, 
                                Flsin=fmag*Fl[(h0+1)*Ndof:(2+h0)*Ndof])
    
    
    if equation == 'HBM':
        # HBM Harmonic 1 Guess
        Ulam0 = np.zeros(Nhc2*Ndof+1)
        
        # Static is set for all models at the end.
        Ulam0[h0*Ndof:(2+h0)*Ndof] = Xw_lin[0, :2*Ndof]
        Ulam0[-1] = w
        
    
    
    ##############################
    # Scale Amplitude
    
    if equation == 'HBM_AMP' \
        or equation == 'HBM_AMP_PHASE' \
        or equation == 'VPRNM_AMP_PHASE':
    
        # Calculate Amplitude to get a force scaling for the initial guess
        amp_cos = control_recov @ Xw_lin[0, :Ndof]
        amp_sin = control_recov @ Xw_lin[0, Ndof:2*Ndof]
        
        lin_amp = np.sqrt(amp_cos**2 + amp_sin**2)
            
        lin_amp = lin_amp*(w**control_order)
            
        fmag_initial = control_amp / lin_amp
        
        # Rescale the linear amplitude by the forcing magnitude
        Xw_lin = Xw_lin * fmag_initial
        
        
        
    ##############################
    # Construct HBM_AMP Guess
    
    if equation == 'HBM_AMP':
        
        Ulam0 = np.zeros(Nhc2*Ndof+2)
        
        # Static set at end
        Ulam0[h0*Ndof:(2+h0)*Ndof] = Xw_lin[0, :2*Ndof]
        Ulam0[-2] = fmag_initial # Force scaling
        Ulam0[-1] = w # Frequency, rad/s
    
    ##############################
    # Rotate Phase 
    if equation == 'HBM_AMP_PHASE' or equation == 'VPRNM_AMP_PHASE':
        # Scale amplitude and then also rotate
        phase = np.arctan2(amp_sin, amp_cos)
        
        rot_max = np.array([[ np.cos(phase), np.sin(phase)],
                            [-np.sin(phase), np.cos(phase)]])
        
        X_rot = rot_max @ Xw_lin[0, :-1].reshape(2, -1)
        
        FcFs = rot_max[:, 0] * fmag_initial
        
    # Construct Initial Guesses for phase rotated
    if equation == 'HBM_AMP_PHASE':
        Ulam0 = np.zeros(Nhc2*Ndof+3)
        
        # Static set at end for all
        Ulam0[h0*Ndof:(2+h0)*Ndof] = X_rot.reshape(-1)
        Ulam0[-3:-1] = FcFs
        Ulam0[-1] = w
    
    if equation == 'VPRNM_AMP_PHASE':
        Ulam0 = np.zeros(Nhc2*Ndof+4)
            
        # Static set at end for all
        Ulam0[h0*Ndof:(2+h0)*Ndof] = X_rot.reshape(-1)
        Ulam0[-4:-2] = FcFs
        Ulam0[-2] = w
        Ulam0[-1] = control_amp
            
    
    ##############################
    # Set static for all models
    
    if Xstat is not None and h0:
        Ulam0[:Ndof] = Xstat
        
    ##############################
    # Call again to get higher harmonic for VPRNM
    if equation == 'VPRNM_AMP_PHASE':
        # Recall this function as HBM to get the higher harmonic estimate
        # Use 'vib_sys_nl' to get the forces on the higher harmonic.
        
        Fnl_int = vib_sys_nl.total_aft(Ulam0[:Ndof*Nhc2], w, h, 
                                       calc_grad=False)[0]
        
        # Index the harmonic of interest 
        rhi_index = Nhc(h[h < rhi]) 
        
        # Flip the sign to negative to for internal -> external
        Fl_rhi = -Fnl_int[Ndof*rhi_index:Ndof*(rhi_index+2)]
        
        Ulam_rhi = predict_harmonic_solution(vib_sys, rhi*w, Fl_rhi,
                                             np.array([1]), 
                                             solver, 'HBM', 
                                             fmag=1.0, neigs=neigs)
        
        Ulam0[Ndof*rhi_index:Ndof*(rhi_index+2)] = Ulam_rhi[:2*Ndof]
    
    return Ulam0

def zero_crossing(X, zero_tol=np.Inf):
    """
    Finds the locations where the array X crosses values zero. 

    Parameters
    ----------
    X : numpy.ndarray
        Array to find approximate zero crossings in
    zero_tol : scalar, optional
        Require X at crossing to be less than this tolerance.
        The default is np.Inf.

    Returns
    -------
    TF : numpy.ndarray of bool
        Has size the same as X, has True for indices of approximate zero 
        crossings

    """
    
    TF = X[:-1]*X[1:] < 0
    TF = np.concatenate((TF, np.array([False])))
    TF = np.logical_and(np.abs(X) < zero_tol, TF)
    
    return TF

def shift_pm_pi(phase):
    """
    Intended to shift phase to be within (-pi, pi].
    This algorithm here is incomplete, so this function does not fully achieve
    that. Need to clean up or remove / update dependencies.

    Parameters
    ----------
    phase : numpy.ndarray
        Phase vector to be shifted.

    Returns
    -------
    phase : numpy.ndarray
        Shifted phase.

    """
    phase = np.copy(phase)
    
    phase = ((phase + np.pi) % 2*np.pi) - np.pi
    
    return phase
