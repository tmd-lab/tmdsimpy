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

def harmonic_stiffness(M, C, K, w, h):
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

    Returns
    -------
    E : (N*Nhc, N*Nhc) numpy.ndarray
        Square stiffness matrix corresponding to linear properties at every
        harmonic. Ordered as all dofs for each of (if h[0]==0 is included)
        [0, 
        cos(h[1]*w*t), sin(h[1]*w*t), 
        cos(h[2]*w*t), sin(h[2]*w*t)]
    dEdw : (N*Nhc, N*Nhc) numpy.ndarray
        Square derivative matrix
        
    See Also
    --------
    harmonic_stiffness_many_dof : Similar function optimized for large systems.
    """
    
    nd = M.shape[0]
    
    Nhc2 = Nhc(h) # Number of Harmonic Components
    
    E = np.zeros((Nhc2*nd, Nhc2*nd))
    dEdw = np.zeros((Nhc2*nd, Nhc2*nd))
    
    # Starting index for first harmonic
    zi = 1*(h[0] == 0)
    n = h.shape[0] - zi
    
    damp_rot = np.array([[0, 1], [-1, 0]])
    
    if zi == 1:
        E[:nd, :nd] = K
    
    if n > 0:
        E[(nd*zi):, (nd*zi):] = np.kron(np.eye(2*n),K) \
            - np.kron(np.kron(np.diag(h[zi:]*w)**2,np.eye(2)),M) \
            + np.kron(np.diag(h[zi:]*w), np.kron(damp_rot,C))
                
        dEdw[(nd*zi):, (nd*zi):] = \
            - np.kron(np.kron(2*w*np.diag(h[zi:])**2, np.eye(2)),M) \
            + np.kron(np.diag(h[zi:]), np.kron(damp_rot,C))
    
    
    return E, dEdw

def harmonic_stiffness_many_dof(M, C, K, w, h, calc_grad=True, only_C=False):
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
        
    See Also
    --------
    harmonic_stiffness : Similar function optimized for small systems.
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
        
        TR = (h[hind]*w)*C
        BL = (-h[hind]*w)*C
        
        E[nd*(hind*2 - zi):nd*(hind*2 - zi+1), \
          nd*(hind*2 - zi+1):nd*(hind*2 - zi+2)] = TR
            
        E[nd*(hind*2 - zi+1):nd*(hind*2 - zi+2), \
          nd*(hind*2 - zi):nd*(hind*2 - zi+1)] = BL
        
        if include_KM:
            TL = K + (-(h[hind]*w)**2)*M
            BR = K + (-(h[hind]*w)**2)*M
        
            E[nd*(hind*2 - zi):nd*(hind*2 - zi+1), \
              nd*(hind*2 - zi):nd*(hind*2 - zi+1)] = TL
            
            E[nd*(hind*2 - zi+1):nd*(hind*2 - zi+2), \
              nd*(hind*2 - zi+1):nd*(hind*2 - zi+2)] = BR
            
        if calc_grad:
            
            TRdw = h[hind]*C
            BLdw = (-h[hind])*C
            
            dEdw[nd*(hind*2 - zi):nd*(hind*2 - zi+1), \
              nd*(hind*2 - zi+1):nd*(hind*2 - zi+2)] = TRdw
                
            dEdw[nd*(hind*2 - zi+1):nd*(hind*2 - zi+2), \
              nd*(hind*2 - zi):nd*(hind*2 - zi+1)] = BLdw
            
            if include_KM:

                TLdw = (-2*w*(h[hind]**2))*M
                BRdw = (-2*w*(h[hind]**2))*M
            
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
    Nt : Number of times considered, must be even
    h : Harmonics considered, 0th harmonic must be first if included
    X0 : Harmonic Coefficients for Nhc x nd
    order : Order of the derivative returned
    
    Returns
    -------
    x_t : time series of each DOF, Nt x nd
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
    h : Harmonics of interest, 0th harmonic must be first if included
    x_t : Time history of interest, Nt x nd

    Returns
    -------
    v : Vector containing fourier coefficients of harmonics h
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
    X : Baseline harmonics values of size at least (Ndof*Nhc+m,). 
        The m extra components will be individually assigned delta or their absolute value
    Ndof : Number of degrees of freedom associated with the model
    h : List of harmonics
    delta : Small value to prevent divide by zero

    Returns
    -------
    CtoP : Vector of same size as X to convert Xphysical=CtoP*Xconditioned

    """
    
    CtoP = delta*np.ones_like(X) # Default Conditioning level when some components are small

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
        
    m = X.shape[0] - Nhc(h)*Ndof
    
    CtoP[-m:] = np.maximum(CtoP[-m:], np.abs(X[-m:]))
        
    return CtoP


def zero_crossing(X, zero_tol=np.Inf):
    """
    Finds the locations where the array X crosses values zero. 

    Parameters
    ----------
    X : Array to find approximate zero crossings in
    zero_tol : Optional, require X at crossing to be less than this tolerance.
               The default is np.Inf.

    Returns
    -------
    TF : Has size the same as X, has True for indices of approximate zero crossings

    """
    TF = X[:-1]*X[1:] < 0
    TF = np.concatenate((TF, np.array([False])))
    TF = np.logical_and(np.abs(X) < zero_tol, TF)
    return TF

def shift_pm_pi(phase):
    """
    Shifts phase to be within (-pi, pi]

    Parameters
    ----------
    phase : TYPE
        DESCRIPTION.

    Returns
    -------
    phase - shifted phase.

    """
    phase = np.copy(phase)
    phase[phase > np.pi] -= 2*np.pi
    phase[phase <= -np.pi] += 2*np.pi
    
    return phase
