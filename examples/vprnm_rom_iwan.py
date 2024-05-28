"""
3 Degree of Freedom example with a superharmonic resonance from [1]_.
Results are plotted for the Harmonic Balance Method and the Variable Phase
Resonance Nonlinear Modes Reduced Order Model (VPRNM ROM).

References
----------
.. [1] Porter, J. H. and M. R. W. Brake. "Efficient Model Reduction and 
       Prediction of Superharmonic Resonances in Frictional and Hysteretic 
       Systems." Mechanical Systems and Signal Processing. Under Review.
       Preprint: arXiv:2405.15918, https://arxiv.org/abs/2405.15918
"""


import sys
import os
import numpy as np

sys.path.append('..')
from tmdsimpy.nlforces.vector_iwan4 import VectorIwan4
from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.continuation import Continuation

from tmdsimpy.solvers import NonlinearSolver

import tmdsimpy.harmonic_utils as hutils

from tmdsimpy import roms


import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
import matplotlib.pyplot as plt

###############################################################################
####### Settings (Vary to Get Different Paper Curves)                   #######
###############################################################################

amp_level = 20 # Paper uses: 10, 20, 30, 40, 50, 70

###############################################################################
####### System parameters                                               #######
###############################################################################

phi1 = np.array([1, 2, 3])
phia = np.array([2, 1, -1])
phib = np.array([-2, 1, 1])

omega = np.array([1, 3.0, 7.5]) 

Lambda = np.diag(omega**2)
Phi = np.vstack((phi1, phia, phib)).T
Phi_inv = np.linalg.inv(Phi)

M    = Phi_inv.T @ Phi_inv
K_ir = Phi_inv.T @ Lambda @ Phi_inv

Fext = np.array([1.0, 0.0, 0.0])

ab_damp = [0.01, 0] # C = 0.01*M

# Force Mappings
Q = np.array([[0.0, 1.0, -1.0]])
T = np.array([[0.0], [1.0], [-1.0]])

kt = 0.6
Fs = 10.0
chi = -0.5
beta = 0.0
Nsliders = 100

# Modifying the Stiffness Matrix
K = K_ir - T @ np.array([[0.5*kt]]) @ Q

h_max = 3 # 3 or 8 # maximum harmonic number to use
Recov = np.array([1.0, 0.0, 0.0]) # Amplitude control at the forcing DOF
control_order = 0 # Displacement control (0th derivative)

# Starting and ending frequency for continuation with harmonic balance method
omega0 = 0.7
omega1 = 1.3

rhi = 3 # superharmonic number, 3rd harmonic

###############################################################################
####### Model Construction                                              #######
###############################################################################

# Nonlinear Force
iwan_force = VectorIwan4(Q, T, kt, Fs, chi, beta, Nsliders=Nsliders, 
                         alphasliders=1.0)

# Setup Vibration System
vib_sys = VibrationSystem(M, K, ab=ab_damp)

vib_sys.add_nl_force(iwan_force)

h = np.arange(h_max+1)

Nhc = hutils.Nhc(h)
Ndof = M.shape[0]
h0 = h[0] == 0

Fl = np.zeros(Nhc*Ndof)

Fl[h0*Ndof:(h0+1)*Ndof] = Fext

solver = NonlinearSolver()

###############################################################################
####### Harmonic Balance Initial Guess                                  #######
###############################################################################

Klin = vib_sys.static_res(0*Fext, 0*Fext)[1]

vib_sys_lin = VibrationSystem(M, Klin, ab=ab_damp)

UFw0 = hutils.predict_harmonic_solution(vib_sys_lin, omega0, Fl, h, 
                                          solver, 'HBM_AMP', 
                                          control_amp=amp_level, 
                                          control_recov=Recov, 
                                          control_order=control_order)

###############################################################################
####### Harmonic Balance Solution                                       #######
###############################################################################

cont_config = {'dsmin'      : 0.0005,
               'verbose'    : False,
               'corrector'  : 'Ortho', # Ortho, Pseudo
               'xtol'       : 1e-9*UFw0.shape[0], 
               'corrector'  : 'Ortho', # Ortho, Pseudo
               'FracLamList' : [0.5, 1.0, 0.75, 0.25, 0.9, 0.1, 0.0],
               }

cont_solver = Continuation(solver, config=cont_config)

# Harmonic balance method function to solve
fun = lambda UFw : vib_sys.hbm_amp_control_res(UFw, Fl, h, Recov, 
                                               amp_level, control_order)


UFw_hbm = cont_solver.continuation(fun, UFw0, omega0, omega1)


###############################################################################
####### EPMC Solutions for Mode 1 and 2                                 #######
###############################################################################

#### EPMC Settings
epmc_dsmin = 0.01
epmc_dsmax = 0.004

Astart_fund = -3
Aend_fund = 2

Astart_rhi = -3
Aend_rhi = 1.5

cont_config = {'DynamicCtoP' : True,
               'dsmin'      : epmc_dsmin,
               'dsmax'      : epmc_dsmax,
               'verbose'    : False,
               'corrector'  : 'Ortho', # Ortho, Pseudo
               'xtol'       : 1e-9*UFw0.shape[0], 
               'corrector'  : 'Ortho', # Ortho, Pseudo
               'FracLamList' : [0.5, 1.0, 0.75, 0.25, 0.9, 0.1, 0.0],
               'MaxSteps'     : 1000
               }

##### EPMC Initial Guesses


h_epmc_fund = h[h != rhi]
Fl_fund = Fl[:-2*Ndof]

##### EPMC Simulations
# May need to remake continuation objects each time since the CtoP vector sizes 
# change.

##### EPMC initial guesses

eigvals, eigvecs = solver.eigs(Klin, M, subset_by_index=[0,2])

Nhc_fund = hutils.Nhc(h_epmc_fund)

Uwxa0_fund = np.zeros(Ndof*Nhc_fund+3)
Uwxa0_rhi = np.zeros(Ndof*Nhc+3)


Uwxa0_fund[(h0+1)*Ndof:(h0+2)*Ndof] = eigvecs[:, 0]
Uwxa0_fund[-3] = np.sqrt(eigvals[0])
Uwxa0_fund[-2] = ab_damp[0]
Uwxa0_fund[-1] = Aend_fund

Uwxa0_rhi[(h0+1)*Ndof:(h0+2)*Ndof] = eigvecs[:, 1]
Uwxa0_rhi[-3] = np.sqrt(eigvals[1])
Uwxa0_rhi[-2] = ab_damp[0]
Uwxa0_rhi[-1] = Astart_rhi


##### EPMC of Mode 1
cont_solver = Continuation(solver, config=cont_config)

epmc_fun = lambda Uwxa : vib_sys.epmc_res(Uwxa, Fl_fund, h_epmc_fund)

epmc_fund_bb = cont_solver.continuation(epmc_fun, Uwxa0_fund, Astart_fund, Aend_fund)

##### EPMC of Mode 2

epmc_fun = lambda Uwxa : vib_sys.epmc_res(Uwxa, Fl, h)

epmc_rhi_bb = cont_solver.continuation(epmc_fun, Uwxa0_rhi, Astart_rhi, Aend_rhi)

###############################################################################
####### VPRNM Solutions                                                 #######
###############################################################################

vprnm_amp_min = 5
vprnm_amp_max = 100

cont_config = {'dsmin'      : 0.005,
               'dsmax'      : 0.1,
               'verbose'    : False,
               'corrector'  : 'Ortho', # Ortho, Pseudo
               'xtol'       : 1e-9*UFw0.shape[0], 
               'corrector'  : 'Ortho', # Ortho, Pseudo
               'FracLamList' : [0.5, 1.0, 0.75, 0.25, 0.9, 0.1, 0.0],
               'MaxSteps'     : 1000
               }

cont_solver = Continuation(solver, config=cont_config)

# VPRNM initial guess
w_vprnm0 = np.sqrt(eigvals[1]) / rhi # fraction of modal frequency

UFcFswA0 = hutils.predict_harmonic_solution(vib_sys_lin, w_vprnm0, Fl, h, 
                                          solver, 'VPRNM_AMP_PHASE', 
                                          control_amp=vprnm_amp_min, 
                                          control_recov=Recov, 
                                          control_order=control_order,
                                          rhi=rhi,
                                          vib_sys_nl=vib_sys)

# VPRNM function to solve
fun = lambda UFcFswA : vib_sys.vprnm_amp_phase_res(UFcFswA, Fl, h, rhi, 
                                                   Recov, control_order)

vprnm_bb = cont_solver.continuation(fun, UFcFswA0, 
                                    vprnm_amp_min, 
                                    vprnm_amp_max)

###############################################################################
####### EPMC and VPRNM ROMs                                             #######
###############################################################################

force_epmc, epmc_point = roms.epmc.constant_displacement(epmc_fund_bb, 
                                                         h_epmc_fund, Fext, 
                                                         UFw_hbm[:, -1], 
                                                         Recov, amp_level)

U_epmc_rom = np.copy(epmc_point[:-3])
U_epmc_rom[h0*Ndof:] *= 10**epmc_point[-1]

Uw_rom_vprnm,F_rom_vprnm,h_rom_vprnm = roms.vprnm.constant_h1_displacement(
                                    epmc_fund_bb, h_epmc_fund, epmc_rhi_bb, h, 
                                    vprnm_bb, h, rhi, Recov, amp_level, 
                                    Recov, Fext)

###############################################################################
####### Plotting Solutions                                              #######
###############################################################################

for dof in range(M.shape[0]):
    
    # Maximum HBM Displacement at any point on Cycle
    Xt = hutils.time_series_deriv(1<<10, h, UFw_hbm[:,  dof:-2:Ndof].T, 0)
    Xmax_hbm = np.max(np.abs(Xt), axis=0) / UFw_hbm[:, -2]
    
    # EPMC ROM Amplitude
    Xt = hutils.time_series_deriv(1<<10, h_epmc_fund, U_epmc_rom[dof::Ndof].reshape(-1,1), 0)
    Xmax_epmc = np.max(np.abs(Xt), axis=0) / force_epmc
    
    # VPRNM ROM Amplitude
    Xt = hutils.time_series_deriv(1<<10, h_rom_vprnm, Uw_rom_vprnm[:,  dof:-1:Ndof].T, 0)
    Xmax_vprnm = np.max(np.abs(Xt), axis=0) / F_rom_vprnm
    
    
    plt.plot(UFw_hbm[:, -1], Xmax_hbm, 'k', label='HBM')
    
    plt.plot(UFw_hbm[:, -1], Xmax_epmc, '--', label='EPMC ROM', color='#009E73')
    
    plt.plot(Uw_rom_vprnm[:, -1], Xmax_vprnm, '-.', 
             label='VPRNM ROM', color='#0072B2')
    
    plt.ylabel('Maximum Displacement/Force [m/N]')
    plt.xlabel('Frequency [rad/s]')
    plt.xlim((omega0, omega1))
    plt.legend()
    plt.title('DOF {}'.format(dof))
    plt.show()
