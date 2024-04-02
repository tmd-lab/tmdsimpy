"""
This module contains a number of reduced order models based on 
the Extended Periodic Motion Concept (EPMC) to capture primary resonance
behavior of a nonlinear vibration system.

See Also
--------
vibration_system.VibrationSystem.epmc_res : 
    EPMC residual method that can be used to solve the set of EPMC equations
continuation.Continuation.continuation :
    Method of obtaining solutions to EPMC at multiple points with continuation
    
Notes
-----
EPMC [1]_ solves for a nonlinear mode behavior. 

References
----------
.. [1] Krack, Malte. "Nonlinear Modal Analysis of Nonconservative Systems: 
    Extension of the Periodic Motion Concept." Computers & Structures 154 
    (July 1, 2015): 59–71. https://doi.org/10.1016/j.compstruc.2015.03.008.
"""

import numpy as np

from .. import harmonic_utils as hutils


def constant_force(epmc_bb, Ndof, h, Fl=None, w=None, zeta=None, 
                   phiH_Fl_real=None, phiH_Fl_imag=None):
    """
    This function utilizes a set of solutions to the Extended Periodic Motion
    Concept (EPMC) to approximate an isolated nonlinear resonance under 
    constant forcing
    
    For the purposes of this documentation, Nhc is the number of harmonic 
    components corresponding to 
    `Nhc = harmonic_utils.Nhc(h)`.

    Parameters
    ----------
    epmc_bb : (M, Nhc*Ndof+3) numpy.ndarray
        Each row corresponds to an EPMC solution at a given amplitude level. 
        The first Nhc*Ndof entries of each row are the displacements of the 
        harmonics in h (e.g., [U0, U1c, U1s] where U0 is static, U1c is the 
        Ndof displacements corresponding to h=1 and cosine, U1c is for sine). 
        The last three entries of each row are frequency (rad/s), 
        xi in EPMC formulation (coefficient in front of mass matrix to create
        a damping matrix), and log10(modal amplitude).
        Harmonic displacements must be multiplied by the modal amplitude to 
        get the physical displacements except for displacements corresponding
        to the zeroth harmonic.
    Ndof : int
        Number of degrees of freedom in the EPMC solution.
    h : numpy.ndarray
        Array of the harmonics used to calculate `epmc_bb`, must be sorted.
    Fl : (Nhc*Ndof) numpy.ndarray or None, optional
        External constant excitation to be considered. 
        If None, then `phiH_Fl_real` and `phiH_Fl_imag` must be provided.
        The default is None.
    w : None, float, or (M,) numpy.ndarray, optional
        Alternative natural frequencies to use instead of those provided in 
        `epmc_bb`. If provided, it is also used in calculating the fraction of 
        critical damping (if `zeta` is not provided).
        Units should be rad/s.
        The default is None.
    zeta : None, float, or (M,) numpy.ndarray, optional
        Alternative modal damping (fraction of critical) to use rather
        than calculating it from the `epmc_bb`. 
        The default is None.
    phiH_Fl_real : None, float, or (M,) numpy.ndarray, optional
        Real component of the complex product between the conjugate or
        Hermitian transpose
        of the mode shape and the forcing vector (considering on the 1st 
        harmonic). 
        Should only be provided when Fl=None.
        If a numpy array, the order corresponds to entries in `epmc_bb`.
        The default is None.
    phiH_Fl_imag : None, float, or (M,) numpy.ndarray, optional
        Imaginary component of the complex product between the conjugate or
        Hermitian
        transpose of the mode shape and the forcing vector (considering on the
        1st harmonic). 
        Should only be provided when Fl=None.
        If a numpy array, the order corresponds to entries in `epmc_bb`.
        The default is None.

    Returns
    -------
    FRC_reconstruct : (Mout,Nhc*Ndof+1) numpy.ndarray
        Reconstructed Frequency Response Curve (FRC) based on the EPMC solution.
        Each row corresponds to a different forcing frequency.
        The first Nhc*Ndof columns are harmonic displacements in the same 
        order as `epmc_bb`, but in physical displacement coordinates.
        The last column is the forcing frequency in rad/s.
    modal_amplitude : (Mout,) numpy.ndarray
        Modal amplitude at output FRC points.
    modal_phase : (Mout,) numpy.ndarray
        Modal phase at the output FRC points.
        
    See Also
    --------
    vibration_system.VibrationSystem.epmc_res : 
        EPMC residual method that each line of epmc_bb solves as the Uwxa 
        input
    vibration_system.VibrationSystem.hbm_res : 
        HBM residual method that each row of FRC_reconstruct approximates
        a solution to the input Uw of this function.
    continuation.Continuation.continuation :
        Method of obtaining solutions to EPMC at multiple points to create
        the epmc_bb input to this function
        
    Notes
    -----
    The fraction of critical damping from an EPMC backbone can be calculated
    as `zeta = epmc_bb[:, -2] / (2*epmc_bb[:, -3])`.
    
    The complex mode shape is represented as 
    `psi = U1c + 1j*U1s` where U1c and U1s are the first harmonic cosine 
    and sine respectively
    
    Only 1st harmonic forcing is considered.
    
    EPMC was proposed in [1]_ and a ROM without phase information is given in 
    [2]_. Full derivation is available in [3]_.
    
    It is not possible to request specific forcing frequencies, rather the
    output is the forcing frequencies that are calculated directly to give
    the amplitudes in `epmc_bb`. To get more frequency resolution, consider
    increasing the resolution of `epmc_bb` either by finding more points or
    by linearly interpolating to upsample. 
    
    References
    ----------
    .. [1] Krack, Malte. "Nonlinear Modal Analysis of Nonconservative Systems: 
        Extension of the Periodic Motion Concept." Computers & Structures 154 
        (July 1, 2015): 59–71. https://doi.org/10.1016/j.compstruc.2015.03.008.
    .. [2] S. Schwarz, L. Kohlmann, A. Hartung, J. Gross, M. Scheel, and
       M. Krack. “Validation of a turbine blade component test with frictional 
       contacts by phase-locked-loop and force-controlled measurements”. In:
       Journal of Engineering for Gas Turbines and Power 142.5 (2020). 
       issn: 0742-4795. https://doi.org/10.1115/1.4044772.
    .. [3] Justin H. Porter, PhD Thesis, 2024.
    """
    
    # Sizes of the problem
    Nhc = hutils.Nhc(h)
    
    h0 = h[0] == 0 # Flag for if the zeroth harmonic is included.
    
    M = epmc_bb.shape[0] # number of EPMC points
    
    ################################
    # Process Inputs to Standard Format
    if w is None:
        w = epmc_bb[:, -3]
    else:
        w = w*np.ones(M) 
        
    if zeta is None:
        zeta = epmc_bb[:, -2] / (2*w)
    else:
        zeta = zeta*np.ones(M) 
        
    assert (phiH_Fl_real is None) == (phiH_Fl_imag is None), \
        'Should provide both phiH_Fl_real and phiH_Fl_imag or neither.'

    assert (phiH_Fl_real is None) != (Fl is None), \
        'Should either provide Fl or phiH_Fl_real and phiH_Fl_imag.'
    
    if Fl is not None:
        
        phiH_Fl_real = (Fl[h0*Ndof:(1+h0)*Ndof] @ epmc_bb[:, h0*Ndof:(1+h0)*Ndof].T \
                        + Fl[(1+h0)*Ndof:(2+h0)*Ndof] @ epmc_bb[:, (1+h0)*Ndof:(2+h0)*Ndof].T)
        
        phiH_Fl_imag = (Fl[(1+h0)*Ndof:(2+h0)*Ndof] @ epmc_bb[:, h0*Ndof:(1+h0)*Ndof].T \
                            - Fl[h0*Ndof:(1+h0)*Ndof] @ epmc_bb[:, (1+h0)*Ndof:(2+h0)*Ndof].T)
        
    else:
        phiH_Fl_real = phiH_Fl_real * np.ones(M)
        phiH_Fl_imag = phiH_Fl_imag * np.ones(M)
    
    
    phi_Fl = np.sqrt( phiH_Fl_real**2 + phiH_Fl_imag**2)
    
    q = 10**epmc_bb[:, -1] # modal amplitude
    
    p2 = w**2 - 2 * (zeta * w)**2 # Intermediate step defined in papers
    
    pm = np.array([[1], [-1]])
    
    
    ################################
    # Forcing Frequency + Modal Amplitude 
    
    p2 = w**2 - 2 * (zeta * w)**2 # Intermediate step defined in papers
    
    pm = np.array([[1], [-1]])
    
    Omega_sq = p2 + pm * np.sqrt(p2**2 - w**4 + phi_Fl**2 / q**2)
    
    Omega_sq = np.hstack((Omega_sq[0, :], np.flipud(Omega_sq[1, :])))
    
    mask = np.logical_and(Omega_sq > 0, ~np.isnan(Omega_sq))
    
    Omega = np.sqrt(Omega_sq[mask])
    
    ###############################
    # Phase calculation
    
    phiH_Fl_real_frc = np.hstack((phiH_Fl_real, np.flipud(phiH_Fl_real)))[mask]
    phiH_Fl_imag_frc = np.hstack((phiH_Fl_imag, np.flipud(phiH_Fl_imag)))[mask]
    
    import warnings
    warnings.warn('Incorrectly flipping sines on phi_H_Fl_imag')
    phiH_Fl_imag_frc = -1*phiH_Fl_imag_frc
    
    wn_frc = np.hstack((w, np.flipud(w)))[mask]
    zeta_frc = np.hstack((zeta, np.flipud(zeta)))[mask]
    modal_amplitude = np.hstack((q, np.flipud(q)))[mask]
    
    arg_real = phiH_Fl_real_frc*(wn_frc**2 - Omega**2) \
                + phiH_Fl_imag_frc*(2*Omega*wn_frc*zeta_frc)
                
    arg_imag = phiH_Fl_real_frc*(2*Omega*wn_frc*zeta_frc) \
                - phiH_Fl_imag_frc*(wn_frc**2 - Omega**2)
    
    modal_phase = np.arctan2(arg_imag, arg_real)
    
    ###############################
    # Build Reconstruction
    
    # Get what index in the epmc mode that each point corresponds to for reference
    mode_ind = np.array(range(epmc_bb.shape[0]))
    mode_ind = np.hstack((mode_ind, np.flipud(mode_ind)))[mask]
    
    FRC_reconstruct = np.zeros((Omega.shape[0], Nhc*Ndof+1))
    FRC_reconstruct[:, -1] = Omega
    
    # Copy all displacements without rotations
    FRC_reconstruct[:, :Nhc*Ndof] = epmc_bb[mode_ind, :Nhc*Ndof] 
    
    # apply the modal amplitude to everything except the zeroth harmonic
    FRC_reconstruct[:, h0*Ndof:Nhc*Ndof] *= modal_amplitude.reshape(-1,1)
    
    # Apply phase rotation
    h_rotate = 1
    FRC_reconstruct = hutils.rotate_subtract_phase(FRC_reconstruct, Ndof, h, 
                                                   modal_phase, h_rotate)
    
    ###############################
    # Flip Everything to be increasing frequency
    
    FRC_reconstruct = np.flipud(FRC_reconstruct)
    modal_amplitude = np.flipud(modal_amplitude)
    modal_phase = np.flipud(modal_phase)
    
    return FRC_reconstruct, modal_amplitude, modal_phase