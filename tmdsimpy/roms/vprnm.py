"""
Module for reduced order models (ROMs) based on Variable Phase Resonance
Nonlinear Modes (VPRNM) for capturing superharmonic resonances.

This module utilizes precalculated VPRNM and EPMC solutions.

See Also
--------
tmdsimpy.VibrationSystem.epmc_res : 
    EPMC residual method that can be used to solve the set of EPMC equations.
tmdsimpy.VibrationSystem.vprnm_res : 
    VPRNM residual method that can be used to solve the set of VPRNM equations.
tmdsimpy.VibrationSystem.vprnm_amp_phase_res : 
    VPRNM residual method that can be used to solve the set of VPRNM equations
    with some extra constraints to allow for easier convergence.
tmdsimpy.VibrationSystem.hbm_res :
    Harmonic balance method residual function for calculating truth solutions.
tmdsimpy.Continuation.continuation :
    Method of obtaining solutions to EPMC/VPRNM at multiple points with 
    continuation.
    
Notes
-----
The formulation of VPRNM and VPRNM based ROMs are described in [1]_, [2]_, [3]_.

References
----------

.. [1]
   Porter, J. H., and M. R. W. Brake. 2024. "Tracking Superharmonic
   Resonances for Nonlinear Vibration of Conservative and Hysteretic Single 
   Degree of Freedom Systems." Mechanical Systems and Signal Processing 
   215:111410. https://doi.org/10.1016/j.ymssp.2024.111410.
   arXiv:2401.08790
   
.. [2]
   Porter, J. H., and M. R. W. Brake. Under Review. "Efficient Model 
   Reduction and Prediction of Superharmonic Resonances in Frictional and 
   Hysteretic Systems." Mechanical Systems and Signal Processing.
   arXiv:2405.15918.

.. [3]
   Porter, J. H. 2024. Modal Interactions and Jointed Structures. PhD Thesis.
   Rice University.

"""

import numpy as np

from ..utils import harmonic as hutils
from ..postprocess import continuation as cpost # Interpolation
from . import epmc # EPMC ROMs

def constant_h1_displacement(epmc_fund_bb, h_fund, epmc_rhi_bb, h_rhi, 
                             vprnm_bb, h_vprnm, rhi,
                             control_point_h1, control_amp_h1, 
                             control_point_rhi, Flcos, extra_Omega=None,
                             correct_force=True):
    """
    Reduced order model (ROM) for a superharmonic resonance based VPRNM, EPMC, 
    and on constant first harmonic amplitude. 

    Parameters
    ----------
    epmc_fund_bb : (Mfund, Nhc_fund*Ndof+3) numpy.ndarray
        Each row corresponds to an EPMC solution at a given amplitude level
        for the mode that response at the forcing frequency (fundamental). 
        The first Nhc*Ndof entries of each row are the displacements of the 
        harmonics in h 
        (all of the first harmonic component, then all of next etc.). 
        The last three entries of each row are frequency (rad/s), 
        xi in EPMC formulation (coefficient in front of mass matrix to create
        a damping matrix), and log10(modal amplitude).
        Harmonic displacements must be multiplied by the modal amplitude to 
        get the physical displacements except for displacements corresponding
        to the zeroth harmonic.
    h_fund : numpy.ndarray, sorted
        Array of the harmonics used to calculate `epmc_fund_bb`, must be sorted.
    epmc_rhi_bb : (Mrhi, Nhc_rhi*Ndof+3) numpy.ndarray
        EPMC solution for the mode that corresponds in superharmonic resonance.
        `Nhc_rhi = utils.harmonic.Nhc(h_rhi)`.
    h_rhi : numpy.ndarray, sorted
        Array of the harmonics used to calculate `epmc_rhi_bb`, must be sorted.
    vprnm_bb : (Mvprnm, Nhc_v*Ndof+4) or (Mvprnm, Nhc_v*Ndof+2) numpy.ndarray
        VPRNM solution points. The first `Nhc_vprnm*Ndof` columns correspond
        to harmonic displacements in physical coordinates.
        If the second dimension is `Nhc_v*Ndof+4`, then the last 4 columns are
        force scaling for cosine, force scaling for sine, frequency (rad/s), 
        controlled amplitude
        (amplitude and phase were controlled in VPRNM solution).
        If the second dimension is `Nhc_v*Ndof+2`, then the last 2 columns are
        frequency (rad/s) and force magnitude scaling 
        (amplitude and phase not controlled in VPRNM solution).
        `Nhc_v = utils.harmonic.Nhc(h_vprnm)`.
    h_vprnm : numpy.ndarray
        Array of the harmonics used to calculate `vprnm_bb`, must be sorted.
    rhi : int
        The resonant superharmonic of interest
        (multiple that multiplies forcing frequency).
    control_point_h1 : (Ndof,) numpy.ndarray
        Vector for extracting the degree of freedom that should be controlled
        to constant amplitude as `control_point @ U(t)` where `U(t)` is the 
        Ndof displacement vector.
    control_amp_h1 : float
        Desired amplitude that the control point should be controlled to for 
        the first harmonic motion.
        This is displacement amplitude (0th derivative of motion)
    control_point_rhi : (Ndof,) numpy.ndarray
        Vector for extracting the degree of freedom that is considered to 
        interpolate/estimate the amplitude of the superharmonic resonance.
    Flcos : (Ndof,) numpy.ndarray
        External constant excitation to be considered. Excitation is just
        applied to the first harmonic (at undetermined phase)
    extra_Omega : None or 1D numpy.ndarray, optional
        Extra forcing frequencies (rad/s) to calculate the force scaling 
        magnitude at. Superharmonic responses are linearly interpolated from
        calculated points to these points. 
        Responses at these frequencies are returned as well as directly 
        calculated frequencies for the superharmonic resonance EPMC ROM.
        The default is None.
    correct_force : bool, optional
        Flag to calculate force correction for VPRNM compared to the EPMC ROM
        for the first harmonic force.
        This correction is required to give good normalized responses for some
        cases. 
        However, if extra_Omega includes values outside of what the 
        superharmonic EPMC ROM identifies, settings this to False will allow
        for getting the EPMC ROM for the 1st mode at those points (and nan for 
        the superharmonic)
        The default is True.

    Returns
    -------
    Uw_reconstruct : (Mout,Nhc_out*Ndof+1) numpy.ndarray
        Reconstructed Frequency Response Curve (FRC) based on the VPRNM and 
        EPMC solutions.
        Each row corresponds to a different forcing frequency.
        The first Nhc_out*Ndof columns are harmonic displacements in physical
        coordinates corresponding to the harmonics in `h_reconstruct`.
        The last column is the forcing frequency in rad/s.
        `Nhc_out = utils.harmonic.Nhc(h_reconstruct)`.
    force_reconstruct : (Mout,) numpy.ndarray
        External excitation force magnitude (scaling for `Flcos`) to create
        the response. Phase information of this force is not calculated.
    h_reconstruct : numpy.ndarray
        List of sorted harmonics for the output.
        This list includes harmonics in `h_fund` and `rhi*h_rhi`.
    
    See Also
    --------
    tmdsimpy.roms.vprnm :
        Module includes more details and discussion on VPRNM based ROMs
    tmdsimpy.roms.epmc.constant_force : 
        Constant force EPMC based ROM for a single mode
    tmdsimpy.roms.epmc.constant_displacement :
        Constant amplitude EPMC based ROM for a single mode.
    tmdsimpy.VibrationSystem.epmc_res : 
        EPMC residual method for finding EPMC backbones.
    tmdsimpy.VibrationSystem.vprnm_res : 
        VPRNM residual method for finding VPRNM backbone.
    tmdsimpy.VibrationSystem.hbm_amp_control_res : 
        HBM residual method calculates something similar to a truth solution
        to compare this ROM to.
    tmdsimpy.Continuation.continuation :
        Method of obtaining solutions to EPMC and VPRNM at multiple points
        to create inputs to this function
    
    Notes
    -----
    
    Formulation of this ROM is derived and explained in detail in [1]_, [2]_.
    
    `control_point_h1` and `control_point_rhi` need not be the same, but can 
    be the same. Depending on mode shapes it may make sense to use different
    locations to approximate the amplitudes of the two modes used.
    
    In general, `epmc_fund_bb` should be calculated with `h_fund` not including
    `rhi`.
    
    Static (harmonic 0) displacements in `Uw_reconstruct` are taken solely from
    the VPRNM solution. If vprnm is calculated without a zeroth harmonic and
    a zeroth harmonic is included in one or both of the EPMC solutions, then
    zeros are returned for the static displacements ignoring static components
    of the EPMC solutions.
    
    In general, specific response frequencies cannot be requested because
    the resonance of the superharmonic provides the frequency as a direct
    calculation while requesting a specific frequency may require a nonlinear 
    solution.
    The parameter `extra_Omega` allows for requesting higher resolution for the 
    output force scaling (which can be directly calculated at requested 
    frequencies). For these requested frequencies, the superharmonic response
    is simply linearly interpolated to fill in the points.
    
    For any frequencies in `extra_Omega` lower than the lowest frequency
    that this function calculates without extra_Omega, nan values will be
    returned for parts of the solution.
    
    The superharmonic EPMC backbone is abbreviated to include the highest 
    amplitude equal to the VPRNM amplitude of the superharmonic resonance. 
    However, for a linear system, the peak amplitude is slightly more than the 
    amplitude at phase resonance. This slightly higher amplitude is not 
    currently captured, but could be implemented in the future (at least for
    the linear case). For the nonlinear case, it may be nontrivial to 
    interpolate the exact resonance amplitude, but extra EPMC points could be
    allowed in the superharmonic part of the ROM to try to improve resolution
    rather than the current approach that eliminates any points greater than
    the VPRNM amplitude.
    
    References
    ----------
    .. [1]
       Porter, J. H., and M. R. W. Brake. Under Review. "Efficient Model 
       Reduction and Prediction of Superharmonic Resonances in Frictional and 
       Hysteretic Systems." Mechanical Systems and Signal Processing.
       arXiv:2405.15918.
    
    .. [2]
       Porter, J. H. 2024. Modal Interactions and Jointed Structures. PhD Thesis.
       Rice University.

    """
    
    ###########################################################################
    # 0. Preprocessing
    Ndof = control_point_h1.shape[0]
    Nhc_v = hutils.Nhc(h_vprnm)

    # determine of VPRNM is constant force or constant phase/amplitude 
    # and create flag for later use
    vprnm_amp_phase = (Nhc_v*Ndof+4) == vprnm_bb.shape[1]
    
    assert (vprnm_bb.shape[1] == Nhc_v*Ndof+4) \
        or (vprnm_bb.shape[1] == Nhc_v*Ndof+2), \
        'vprnm_bb.shape[1] does not match either expected option.'
        
    # Create an EPMC backbone formatted with last entry being modal amplitude
    # not on a log scale.
    epmc_rhi_bb_lin_q = np.copy(epmc_rhi_bb)
    epmc_rhi_bb_lin_q[:, -1] = 10**epmc_rhi_bb_lin_q[:, -1]
    
    ###########################################################################
    # 1. Interpolated VPRNM to the correct first harmonic amplitude
    
    amp_h1_vprnm = _extract_harmonic_amplitude(vprnm_bb, h_vprnm, 
                                               1, control_point_h1)
    
    vprnm_point = cpost.linear_interp(vprnm_bb, control_amp_h1, 
                                      reference_values=amp_h1_vprnm)[0]
    
    Omega_VPRNM = vprnm_point[-2]
    
    if vprnm_amp_phase:
        f_mag_VPRNM = np.sqrt(vprnm_point[-4]**2 + vprnm_point[-3]**2)
    else:
        f_mag_VPRNM = vprnm_point[-1]
    
    ###########################################################################
    # 2. Interpolate epmc_rhi_bb to match the VPRNM point
    
    # desired amplitude of the superharmonic resonance
    A_rhi_vprnm = _extract_harmonic_amplitude(vprnm_point, h_vprnm, 
                                              rhi, control_point_rhi)
    
    # EPMC amplitude for interpolation
    A_rhi_epmc = _extract_harmonic_amplitude(epmc_rhi_bb, h_rhi, 
                                              1, control_point_rhi, 
                                              is_epmc=True)
    
    # Interpolate, and interpolate the true modal amplitude
    # so the last entry is linear scale modal amplitude (not log)
    epmc_rhi_point = cpost.linear_interp(epmc_rhi_bb_lin_q, A_rhi_vprnm, 
                                         reference_values=A_rhi_epmc)[0]
    
    ###########################################################################
    # 3. Approximate the modal forcing on the superharmonic resonance
    
    q_S_EPMC = epmc_rhi_point[-1]
    zeta_S_EPMC = epmc_rhi_point[-2] / (2*epmc_rhi_point[-3])
    
    phiSH_FS_mag = 2*q_S_EPMC*(rhi*Omega_VPRNM)**2*zeta_S_EPMC
    
    ###########################################################################
    # 4. Superharmonic Resonance Response
    
    w_modified = epmc_rhi_bb_lin_q[:, -3]*(rhi*Omega_VPRNM)/epmc_rhi_point[-3]
    
    # Create a new EPMC backbone to pass to FRC reconstruction that includes 
    # the interpolated point.
    mask = epmc_rhi_bb_lin_q[:, -1] < q_S_EPMC
    
    epmc_rhi_bb_abbrev = np.vstack((epmc_rhi_bb_lin_q[mask, :], 
                                    epmc_rhi_point))
    
    # want to preserve the modal damping factor, but the frequency has changed
    # therefore, need to update the epmc xi parameter
    z_abbrev = epmc_rhi_bb_abbrev[:, -2] / (2*epmc_rhi_bb_abbrev[:, -3])
    
    # convert modal amplitude back to log scale to call epmc rom.
    epmc_rhi_bb_abbrev[:, -1] = np.log10(epmc_rhi_bb_abbrev[:, -1])
    
    w_abbrev = np.hstack((w_modified[mask], rhi*Omega_VPRNM))
    
    Uw_S,q_S,_ = epmc.constant_force(epmc_rhi_bb_abbrev, Ndof, 
                                          h_rhi, w=w_abbrev, zeta=z_abbrev,
                                          phiH_Fl_real=phiSH_FS_mag, 
                                          phiH_Fl_imag=0.0)
    
    ###########################################################################
    # 5. Rotate Superharmonic Response to Match Phase
    
    super_peak_ind = np.argmin(np.abs(Uw_S[:, -1] - rhi*Omega_VPRNM))
    
    # this is a check everytime since it is very difficult to write a test
    # for it. This check failed due to bugs with frequency scaling to match
    # the VPRNM frequency.
    assert np.abs(q_S[super_peak_ind] - q_S_EPMC) <= 1e-12, \
        'Superharmonic reconstruction did not recover VPRNM peak.'
    
    h0_rhi = h_rhi[0] == 0
    
    # superharmonic only from the FRC reconstruction of the superharmonic
    Uw_S_point = Uw_S[super_peak_ind, h0_rhi*Ndof:(2+h0_rhi)*Ndof]
    
    # superharmonic only from the VPRNM point
    rhi_index_v = hutils.Nhc(h_vprnm[h_vprnm < rhi]) 
    vprnm_point_super = vprnm_point[rhi_index_v*Ndof:(rhi_index_v+2)*Ndof]
    
    # Phase that the superharmonic needs to be rotated through
    phi_rot_S = np.arctan2(control_point_rhi @ vprnm_point_super[Ndof:2*Ndof],
                           control_point_rhi  @ vprnm_point_super[:Ndof]) \
                - np.arctan2(control_point_rhi @ Uw_S_point[Ndof:2*Ndof],
                             control_point_rhi  @ Uw_S_point[:Ndof])
    
    # the superharmonic is still the 1st harmonic of this reconstruction
    Uw_S = hutils.rotate_subtract_phase(Uw_S, Ndof, h_rhi, phi_rot_S, 1)
    
    ###########################################################################
    # Extra Step: Add some extra frequency points to get higher resolution of 
    # the force scaling around the primary resonance
    
    if extra_Omega is not None:
        new_w = np.hstack((rhi*extra_Omega, Uw_S[:, -1]))
        new_w = np.sort(new_w)
        
        # super_peak_ind and q_S are used in the correction of the force.
        super_peak_ind += np.sum(rhi*extra_Omega < Uw_S[super_peak_ind, -1])
        q_S = np.interp(new_w, Uw_S[:, -1], q_S)
        
        Uw_S = cpost.linear_interp(Uw_S, new_w)
        Uw_S[:, -1] = new_w
    
    ###########################################################################
    # 6 and 9. Interpolate EPMC for the Fundamental to the correct amplitude
    
    Omega = Uw_S[:, -1] / rhi
    
    force_magnitude, epmc_point_fund = epmc.constant_displacement(epmc_fund_bb, 
                                                h_fund, Flcos, Omega, 
                                                control_point_h1, 
                                                control_amp_h1)
    
    Nhc_fund = hutils.Nhc(h_fund)
    
    h0_fund = h_fund[0] == 0
    epmc_point_fund[h0_fund*Ndof:Ndof*Nhc_fund] *= (10**epmc_point_fund[-1])
    
    ###########################################################################
    # 7. Rotate Primary Response to Match Phase
    
    h0_v = h_vprnm[0] == 0
    
    vprnm_h1 = vprnm_point[h0_v*Ndof:(h0_v+2)*Ndof]
    epmc_fund_h1 = epmc_point_fund[h0_fund*Ndof:(h0_fund+2)*Ndof]
    
    phi_rot_F = np.arctan2(control_point_h1 @ vprnm_h1[Ndof:2*Ndof],
                           control_point_h1  @ vprnm_h1[:Ndof]) \
                - np.arctan2(control_point_h1 @ epmc_fund_h1[Ndof:2*Ndof],
                             control_point_h1  @ epmc_fund_h1[:Ndof])
    
    # the superharmonic is still the 1st harmonic of this reconstruction
    epmc_point_fund = hutils.rotate_subtract_phase(epmc_point_fund, 
                                               Ndof, h_fund, phi_rot_F, 1)[0]
    
    ###########################################################################
    # 8. Combine FRC components from the different parts of the solution
    
    # Include the first harmonic from the VPRNM list to ensure that the zeroth
    # harmonic is included if needed.
    h_out = np.hstack((h_vprnm[0], h_fund, rhi*h_rhi))
    h_out = np.unique(h_out)
    
    Nhc_out = hutils.Nhc(h_out)
    
    Uw_out = np.zeros((Omega.shape[0], Ndof*Nhc_out+1))
    
    if h0_v:
        Uw_out[:, :Ndof] = vprnm_point[:Ndof]
        
    # Add motion from fundamental EPMC
    for hind in range(h0_fund*1, h_fund.shape[0]):
        h_curr = h_fund[hind]
        
        Nhc_ind_fund = hutils.Nhc(h_fund[h_fund < h_curr]) 
        Nhc_ind_out = hutils.Nhc(h_out[h_out < h_curr]) 

        Uw_out[:, Nhc_ind_out*Ndof:(Nhc_ind_out+2)*Ndof] \
            = epmc_point_fund[Nhc_ind_fund*Ndof:(Nhc_ind_fund+2)*Ndof]
        
    # Add motion from superharmonic EPMC
    for hind in range(h0_rhi*1, h_rhi.shape[0]):
        h_curr = h_rhi[hind]
        h_curr_out = h_curr*rhi
        
        Nhc_ind_rhi = hutils.Nhc(h_rhi[h_rhi < h_curr]) 
        Nhc_ind_out = hutils.Nhc(h_out[h_out < h_curr_out]) 

        Uw_out[:, Nhc_ind_out*Ndof:(Nhc_ind_out+2)*Ndof] \
            += Uw_S[:, Nhc_ind_rhi*Ndof:(Nhc_ind_rhi+2)*Ndof]
    
    Uw_out[:, -1] = Omega
    
    ###########################################################################
    # 10. Correction to the Force Magnitude (9. was done with 6.)
    
    if correct_force:
        # Error between VPRNM and EPMC reconstruction
        Delta_force = f_mag_VPRNM - force_magnitude[super_peak_ind]
        
        force_out = force_magnitude + Delta_force * q_S / q_S.max()
    else:
        force_out = force_magnitude
    
    ###########################################################################
    # Rename outputs
    
    Uw_reconstruct = Uw_out
    force_reconstruct = force_out
    h_reconstruct = h_out
    
    return Uw_reconstruct, force_reconstruct, h_reconstruct


def _extract_harmonic_amplitude(U, h, hi, recov, is_epmc=False):
    """
    Extracts the harmonic amplitude at a DOF for a given harmonic

    Parameters
    ----------
    U : (M, Ndof*Nhc+a) or (Ndof*Nhc+a,) numpy.ndarray
        Rows are different solution points, columns are harmonic displacements.
        In general last a columns are ignored.
        If `is_epmc == True`, the last column must be the 
        log10(modal amplitude).
    h : numpy.ndarray
        Sorted list of integers for the harmonics.
    hi : int
        harmonic of interest, must be in h.
    recov : (Ndof,) numpy.ndarray
        Vector for extracting the displacement of interest.
    is_epmc : bool
        Flag to include modal amplitude scaling for EPMC if True.
        The default is False.

    Returns
    -------
    amplitude : (M,) or (1,) numpy.ndarray
        Extracted amplitude for the input points.

    Notes
    -----
    
    Function is not directly tested and is intended to be private.
    Further code development may move this function to harmonic utilities,
    add tests, and make the function public.
    
    """
    
    Ndof = recov.shape[0]
    
    # Ensure that U is at least 2D in case only 1 row was provided
    U = np.atleast_2d(U)
    
    hi_index = hutils.Nhc(h[h < hi]) 
    
    Xcos = U[:, Ndof*hi_index:Ndof*(hi_index+1)] @ recov
    Xsin = U[:, Ndof*(hi_index+1):Ndof*(hi_index+2)] @ recov
    
    amplitude = np.sqrt(Xcos**2 + Xsin**2)
    
    if is_epmc:
        amplitude = amplitude * (10**U[:, -1])

    return amplitude
    
