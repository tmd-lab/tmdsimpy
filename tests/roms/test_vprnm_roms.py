"""
Tests for verifying the accuracy of the ROMs based on VPRNM.
""" 

import sys
import numpy as np
import unittest

sys.path.append('../..')

from tmdsimpy.roms import vprnm as rom_vprnm

import tmdsimpy.utils.harmonic as hutils
from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.postprocess import continuation as cpost

class TestVprnmRoms(unittest.TestCase):
    
    def test_constant_displacement_rom(self):
        """
        Test VPRNM ROM against an analytical solution based on a linear system.
        
        Notes
        -----
        
        The linear system has two modes that are near an internal resonance. 
        The linear system will be excited at the first harmonic. 
        The exctation of the higher harmonic will be `Flcos_h3 = 0.1*Flcos` 
        and `Flsin_h3 = 0.07*Flcos`.
        EPMC solutions include some higher harmonics to check those 
        contributions.
        
        Analytical solutions for the linear system are constructed for only a 
        single mode so that comparison should be exact. For the higher mode, 
        this is the FRF with 2 modes minus the FRF with 1 mode since only the 
        number of modes can be specified starting from the smallest frequency.
        """
        
        #############################
        # Construct a linear system with prescribed mode shapes and frequencies
        Phi = np.array([[1, 2], [2, 1]])
        Phi_inv = np.linalg.inv(Phi)
        
        wn = np.array([1.5, 3.0])
        
        M = Phi_inv.T @ Phi_inv
        K = Phi_inv.T @ np.diag(wn)**2 @ Phi_inv
        Ndof = M.shape[0]
        
        ab_damp = [0.2, 0.0]
        
        vib_sys = VibrationSystem(M, K, ab=ab_damp)
        
        rhi = 3 # look at 3:1 superharmonic resonance of the second mode
        
        # contains eigen analysis method for linear FRF
        solver = NonlinearSolver()
        
        Flcos = np.array([1.0, 0.5])
        
        mag_S_sin = 0.07
        mag_S_cos = 0.1
        
        #############################
        # Construct nonlinear solutions
        
        # fundamental mode solution
        h_fund = np.array([0, 1, 2])
        
        # arbitrary static displacement, some higher harmonic info to check 
        epmc_fund_point = np.hstack(([0.1, 0.9],
                                     np.sqrt(2)/2*Phi[:, 0], 
                                     np.sqrt(2)/2*Phi[:, 0],
                                     0.1*np.sqrt(2)/2*Phi[:, 0], 
                                     0.1*np.sqrt(2)/2*Phi[:, 0],
                                     wn[0],
                                     ab_damp[0],
                                     0))
        
        epmc_fund_bb = np.ones((50, 1)) @ np.atleast_2d(epmc_fund_point)
        epmc_fund_bb[:, -1] = np.linspace(-3, 3, epmc_fund_bb.shape[0])
        
        
        # superharmonic mode solution
        h_rhi = np.array([1, 3])
        
        epmc_rhi_point = np.hstack((np.sqrt(3)/2*Phi[:, 1], 
                                    -0.5*Phi[:, 1],
                                    0.25*np.sqrt(2)*Phi[:, 1], 
                                    0.1*np.sqrt(2)*Phi[:, 1],
                                    wn[1],
                                    ab_damp[0],
                                    0))
        
        epmc_rhi_bb = np.ones((100, 1)) @ np.atleast_2d(epmc_rhi_point)
        epmc_rhi_bb[:, -1] = np.linspace(-5, 1, epmc_rhi_bb.shape[0])
        
        # vprnm solution
        h_vprnm = np.array([0, 1, 3])
        Xstat = np.array([5.9, 6.3])
        
        vprnm_force = np.array([0.3, 1.0, 3.0])
        
        Nhc_v = hutils.Nhc(h_vprnm)
        
        vprnm_bb = np.zeros((vprnm_force.shape[0], Ndof*Nhc_v+2))
        
        u_h1 = vib_sys.linear_frf(wn[1]/rhi, Flcos, solver, neigs=1)
        
        vprnm_bb[:, :Ndof] = Xstat
        vprnm_bb[:, Ndof:3*Ndof] = vprnm_force.reshape(-1, 1) @ u_h1[:, :2*Ndof]
        
        # Construct the linear solution only with the second mode
        u_h3_neig2 = vib_sys.linear_frf(wn[1], mag_S_cos*Flcos, solver, 
                                        neigs=2, 
                                        Flsin=mag_S_sin*Flcos)
        
        u_h3_neig1 = vib_sys.linear_frf(wn[1], mag_S_cos*Flcos, solver, 
                                        neigs=1, 
                                        Flsin=mag_S_sin*Flcos)
        
        u_h3 = u_h3_neig2 - u_h3_neig1
        
        vprnm_bb[:, 3*Ndof:5*Ndof] = vprnm_force.reshape(-1, 1) @ u_h3[:, :2*Ndof]
        
        vprnm_bb[:, -2] = wn[1]/rhi
        
        vprnm_bb[:, -1] = vprnm_force
        
        #############################
        # Checks on the Nonlinear solutions
        
        # Fundamental mode EPMC
        Nhc_f = hutils.Nhc(h_fund)
        
        Fl_epmc_fund = np.zeros(Nhc_f*Ndof)
        
        Fl_epmc_fund[:Ndof] = K @ epmc_fund_bb[0, :Ndof]
        
        R = vib_sys.epmc_res(epmc_fund_bb[0], Fl_epmc_fund, h_fund)[0]
        
        self.assertLess(np.linalg.norm(R[:3*Ndof]), 1e-12,
                    'Constructed solution does not pass EPMC residual check.')
        
        self.assertLess(np.linalg.norm(R[-2:]), 1e-12,
                    'Constructed solution does not pass EPMC residual check.')
        
        # Superharmonic EPMC
        Nhc_rhi = hutils.Nhc(h_rhi)
        
        Fl_epmc_rhi = np.zeros(Nhc_rhi*Ndof)
        
        R = vib_sys.epmc_res(epmc_rhi_bb[0], Fl_epmc_rhi, h_rhi)[0]
        
        self.assertLess(np.linalg.norm(R[:2*Ndof]), 1e-12,
                    'Constructed solution does not pass EPMC residual check.')
        
        self.assertLess(np.linalg.norm(R[-2:]), 1e-12,
                    'Constructed solution does not pass EPMC residual check.')
        
        # VPRNM solution
        # It does not make sense to check the VPRNM solution because the 
        # constructed solution only uses a single mode for each component, so
        # does not satisfy HBM exactly.
        
        # Fl_vprnm = np.zeros(Ndof*Nhc_v)
        # Fl_vprnm[:Ndof] = K @ Xstat
        # Fl_vprnm[Ndof:2*Ndof] = Flcos
        # R = vib_sys.vprnm_res(vprnm_bb[0], h_vprnm, rhi, Fl_vprnm)[0]
        
        #############################
        # Create ROM output
        
        control_point_h1 = np.array([1.0,1.0])
        control_amp_h1 = 5.0
        control_point_rhi = np.array([0.75, 0.25])
        
        Uw_rom, force_rom, h_rom = rom_vprnm.constant_h1_displacement(
                                      epmc_fund_bb, h_fund,
                                      epmc_rhi_bb, h_rhi,
                                      vprnm_bb, h_vprnm, rhi, 
                                      control_point_h1, control_amp_h1, 
                                      control_point_rhi, Flcos)
        
        self.assertEqual(tuple(h_rom), (0, 1, 2, 3, 9), 
                         'Output harmonic list is incorrect')
        
        # Static solution
        self.assertLess(np.abs(Uw_rom[:, :Ndof] - Xstat).max(), 1e-12, 
                        'Static solution in ROM is wrong.')
        
        # 1st harmonic solution
        # - should be constant scaling of the linear solution used in vprnm_bb
        # - should have requested amplitude
        
        tmp = Uw_rom[:, Ndof:3*Ndof] / vprnm_bb[0, Ndof:3*Ndof]
        
        self.assertLess(np.abs(tmp - tmp[0, 0]).max(), 1e-12, 
            'Harmonic 1 solution has the wrong phase or is not repeated correctly.')
        
        amp_cos = Uw_rom[0, Ndof:2*Ndof] @ control_point_h1
        amp_sin = Uw_rom[0, 2*Ndof:3*Ndof] @ control_point_h1
        amp_rom = np.sqrt(amp_cos**2 + amp_sin**2)
        
        self.assertLess(np.abs(amp_rom - control_amp_h1), 1e-12,
                        'Returned first harmonic amplitude is wrong.')
        
        # 3rd harmonic - superharmonic resonance solution
        # - Determine excitation force level
        # - Calculate linear FRF (only using 2nd mode)
        # - Check errors
        vprnm_point_F = Uw_rom[0, Ndof] / vprnm_bb[0, Ndof] *  vprnm_bb[0, -1]
        
        u_h3_neig2 = vib_sys.linear_frf(rhi*Uw_rom[:, -1], 
                                        mag_S_cos*vprnm_point_F*Flcos, 
                                        solver, neigs=2, 
                                        Flsin=mag_S_sin*vprnm_point_F*Flcos)
        
        u_h3_neig1 = vib_sys.linear_frf(rhi*Uw_rom[:, -1], 
                                        mag_S_cos*vprnm_point_F*Flcos, 
                                        solver, neigs=1, 
                                        Flsin=mag_S_sin*vprnm_point_F*Flcos)
        
        Uh3_ref = u_h3_neig2 - u_h3_neig1
        
        error = np.abs(Uw_rom[:, 5*Ndof:7*Ndof] - Uh3_ref[:, :-1]).max()
        
        self.assertLess(error, 1e-10,
                        'Superharmonic resonance is wrong in reconstruction.')
        
        # force magnitude
        
        Uw_lin_h1 = vib_sys.linear_frf(Uw_rom[:, -1], Flcos, solver, neigs=1)
        
        h1_cos = Uw_lin_h1[:, :Ndof] @ control_point_h1
        h1_sin = Uw_lin_h1[:, Ndof:2*Ndof] @ control_point_h1
        h1_amp = np.sqrt(h1_cos**2 + h1_sin**2)
        
        lin_force = control_amp_h1 / h1_amp
        
        self.assertLess(np.abs(lin_force - force_rom).max(), 1e-10,
                        'External force scaling for VPRNM ROM is wrong.')
        
        # Extra EPMC harmonics
        # 2nd Harmonic Outputs
        # - Constant amplitude
        # - Amplitude of second harmonic is 0.1x 1st harmonic amplitude
        # - Correctly phase rotated from EPMC solution at 2x phase of H1
        
        error = np.abs(Uw_rom[:, 3*Ndof:5*Ndof] / Uw_rom[0, 3*Ndof:5*Ndof] - 1)
        
        self.assertLess(error.max(), 1e-12, 
                        'Intermediate harmonic 2 is not repeated as constant.')
        
        h2_cos = Uw_rom[0, 3*Ndof:4*Ndof] @ control_point_h1
        h2_sin = Uw_rom[0, 4*Ndof:5*Ndof] @ control_point_h1
        
        h2_amp = np.sqrt(h2_cos**2 + h2_sin**2)
        
        self.assertLess(np.abs(0.1*control_amp_h1 - h2_amp), 1e-12,
                'Harmonic 2 should be 0.1*harmonic 1 by construction of EPMC.')
        
        # phase of harmonic 2
        phase_h1 = np.arctan2(Uw_rom[0, 2*Ndof], Uw_rom[0, 1*Ndof]) \
                    - np.arctan2(epmc_fund_bb[0, 2*Ndof], epmc_fund_bb[0, 1*Ndof])
        
        phase_h2 = np.arctan2(Uw_rom[0, 4*Ndof], Uw_rom[0, 3*Ndof]) \
                    - np.arctan2(epmc_fund_bb[0, 4*Ndof], epmc_fund_bb[0, 3*Ndof])
        
        self.assertLess(np.abs(phase_h2 - 2*phase_h1), 1e-12, 
                        'Phase shift of second harmonic should be twice that '\
                        + 'of the first harmonic.')
        
        # Extra EPMC harmonics
        # 9th Harmonic Outputs
        # - Relative amplitude magnitude to the 3rd harmonic amplitude
        # - Compare phase shift from EPMC (should be 3x that of 3rd harmonic)
        
        epmc_h1c = epmc_rhi_bb[0, 0*Ndof:1*Ndof] @ control_point_rhi
        epmc_h1s = epmc_rhi_bb[0, 1*Ndof:2*Ndof] @ control_point_rhi
        epmc_h3c = epmc_rhi_bb[0, 2*Ndof:3*Ndof] @ control_point_rhi
        epmc_h3s = epmc_rhi_bb[0, 3*Ndof:4*Ndof] @ control_point_rhi
        
        amp_ratio = np.sqrt(epmc_h3c**2 + epmc_h3s**2) \
                        / np.sqrt(epmc_h1c**2 + epmc_h1s**2)
        
        
        rom_h1c = Uw_rom[:, 5*Ndof:6*Ndof] @ control_point_rhi
        rom_h1s = Uw_rom[:, 6*Ndof:7*Ndof] @ control_point_rhi
        rom_h3c = Uw_rom[:, 7*Ndof:8*Ndof] @ control_point_rhi
        rom_h3s = Uw_rom[:, 8*Ndof:9*Ndof] @ control_point_rhi
        
        
        amp_ratio_rom = np.sqrt(rom_h3c**2 + rom_h3s**2) \
                        / np.sqrt(rom_h1c**2 + rom_h1s**2)
        
        self.assertLess(np.abs(amp_ratio_rom - amp_ratio).max(), 1e-12,
                'Amplitude ratio of output higher harmonics from '\
                +'superharmonic EPMC backbone is wrong.')
        
        # Phase shift of the 9th harmonic
        phase_h3 = np.arctan2(Uw_rom[:, 6*Ndof], Uw_rom[:, 5*Ndof]) \
                    - np.arctan2(epmc_rhi_bb[0, 1*Ndof], epmc_rhi_bb[0, 0*Ndof])
        
        phase_h9 = np.arctan2(Uw_rom[:, 8*Ndof], Uw_rom[:, 7*Ndof]) \
                    - np.arctan2(epmc_rhi_bb[0, 3*Ndof], epmc_rhi_bb[0, 2*Ndof])
        
        phase_error = np.mod((phase_h9  - h_rhi[-1] * phase_h3), 2*np.pi)
        phase_error[phase_error > np.pi] -= 2*np.pi
        
        self.assertLess(np.abs(phase_error).max(), 1e-12,
            'Phase rotation of higher harmonics of epmc_rhi_bb is incorrect.')
        
        #############################
        # Try again with VPRNM in the other format (amp + phase control)
        
        vprnm_bb2 = np.zeros((vprnm_force.shape[0], Ndof*Nhc_v+4))
        vprnm_bb2[:, :-4] = vprnm_bb[:, :-2]
        vprnm_bb2[:, -4] = vprnm_bb[:, -1]*np.sqrt(3)/2 # cosine forcing
        vprnm_bb2[:, -3] = vprnm_bb[:, -1]*(-1/2) # sine forcing
        vprnm_bb2[:, -2] = vprnm_bb[:, -2] # Frequency
        vprnm_bb2[:, -1] = -1 # Amplitude, should be ignored in ROM
        
        Uw_rom2, force_rom2, h_rom2 = rom_vprnm.constant_h1_displacement(
                                      epmc_fund_bb, h_fund,
                                      epmc_rhi_bb, h_rhi,
                                      vprnm_bb2, h_vprnm, rhi, 
                                      control_point_h1, control_amp_h1, 
                                      control_point_rhi, Flcos)
        
        self.assertLess(np.abs(Uw_rom - Uw_rom2).max(), 1e-12,
                        'Alternative VPRNM input should not change output.')
        
        self.assertLess(np.abs(force_rom - force_rom2).max(), 1e-12,
                        'Alternative VPRNM input should not change output.')
        
        self.assertLess(np.abs(h_rom - h_rom2).max(), 1e-12,
                        'Alternative VPRNM input should not change output.')
        
        #############################
        # Check that higher harmonics from EPMC backbones are correctly added
        # when they overlap
        
        h_fund_alt = np.array([0, 1, 6])
        h_rhi_alt = np.array([1, 2])
        
        Uw_rom_alt1, force_rom_alt1, h_rom_alt1 \
            = rom_vprnm.constant_h1_displacement(
                                      epmc_fund_bb, h_fund_alt,
                                      epmc_rhi_bb, h_rhi,
                                      vprnm_bb2, h_vprnm, rhi, 
                                      control_point_h1, control_amp_h1, 
                                      control_point_rhi, Flcos)

        Uw_rom_alt2, force_rom_alt2, h_rom_alt2 \
            = rom_vprnm.constant_h1_displacement(
                                      epmc_fund_bb, h_fund,
                                      epmc_rhi_bb, h_rhi_alt,
                                      vprnm_bb2, h_vprnm, rhi, 
                                      control_point_h1, control_amp_h1, 
                                      control_point_rhi, Flcos)

        Uw_rom_alt3, force_rom_alt3, h_rom_alt3 \
            = rom_vprnm.constant_h1_displacement(
                                      epmc_fund_bb, h_fund_alt,
                                      epmc_rhi_bb, h_rhi_alt,
                                      vprnm_bb2, h_vprnm, rhi, 
                                      control_point_h1, control_amp_h1, 
                                      control_point_rhi, Flcos)

        self.assertEqual(tuple(h_rom_alt3), (0,1,3,6),
                         'Not handling harmonics list for overlapped EPMC '\
                         + 'harmonics correctly.')

        # Check that the sum of the 6th harmonic from the two alternative cases
        # is equal to the 6th harmonic from the final case
        
        rom_h6 = Uw_rom_alt1[:, 5*Ndof:7*Ndof] + Uw_rom_alt2[:, 7*Ndof:9*Ndof]

        self.assertLess(np.abs(rom_h6 - Uw_rom_alt3[:, 5*Ndof:7*Ndof]).max(), 
                        1e-12, 
                        'Harmonic above superharmonic should be sum '
                        + 'of epmc_fund and epmc_rhi contributions.')
        
        #############################
        # Try again with VPRNM up sampling the frequency near the 
        # primary resonance
        
        extra_Omega = np.linspace(0.25, 5, 100)
        
        Uw_rom_ex, force_rom_ex, h_rom_ex = rom_vprnm.constant_h1_displacement(
                                      epmc_fund_bb, h_fund,
                                      epmc_rhi_bb, h_rhi,
                                      vprnm_bb, h_vprnm, rhi, 
                                      control_point_h1, control_amp_h1, 
                                      control_point_rhi, Flcos,
                                      extra_Omega=extra_Omega)
        
        self.assertEqual(Uw_rom_ex.shape[0], 
                 Uw_rom.shape[0] + extra_Omega.shape[0],
                 'Added wrong number of points with extra_Omega parameter.')
        
        Uw_interp = cpost.linear_interp(Uw_rom, Uw_rom_ex[:, -1])
        
        self.assertLess(np.abs(Uw_interp - Uw_rom_ex).max(), 1e-12,
                        'extra_Omega option is inconsistent with '\
                        'interpolating Uw_rom')
        
        force_interp = np.interp(Uw_rom_ex[:, -1], Uw_rom[:, -1], force_rom)
        
        self.assertGreater(np.abs(force_interp - force_rom_ex).max(), 0.01,
                        'extra_Omega option should provide more information '\
                        'than linearly interpolating force outputs.')
            
        # Check if the extra force outputs are accurate to linear model.
        Uw_lin_h1 = vib_sys.linear_frf(Uw_rom_ex[:, -1], Flcos, solver, neigs=1)
        
        h1_cos = Uw_lin_h1[:, :Ndof] @ control_point_h1
        h1_sin = Uw_lin_h1[:, Ndof:2*Ndof] @ control_point_h1
        h1_amp = np.sqrt(h1_cos**2 + h1_sin**2)
        
        lin_force = control_amp_h1 / h1_amp
        
        self.assertLess(np.abs(lin_force - force_rom_ex).max(), 1e-10,
                'extra_Omega option does not match linear force calculation.')
        
        
if __name__ == '__main__':
    unittest.main()
