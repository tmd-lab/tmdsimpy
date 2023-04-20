# Simulation of a Two Degree of Freedom Duffing Oscillator
#
# References for NFRC's with identical parameters:
#   M. Volvert and G. Kerschen, 2021, Phase resonance nonlinear modes of 
#   mechanical systems. 


import sys
import numpy as np

# Path to Harmonic balance / vibration system 
sys.path.append('../DEPENDENCIES/tmd-sim-py/ROUTINES')
sys.path.append('../DEPENDENCIES/tmd-sim-py/ROUTINES/NL_FORCES')
sys.path.append('../DEPENDENCIES/tmd-sim-py/TESTS/')


from cubic_stiffness import CubicForce
from vibration_system import VibrationSystem

from solvers import NonlinearSolver
from continuation import Continuation

import harmonic_utils as hutils


import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
import matplotlib.pyplot as plt

###############################################################################
####### System parameters                                               #######
###############################################################################

Ndof = 2

M = np.array([[1, 0], [0, 1]])

K = np.array([[2, -1],[-1, 2]])

C = np.array([[0.02, -0.01],[-0.01, 0.11]])

# Nonlinear Force
knl = 1.0
kalpha = np.array([knl])

Q = np.array([[1.0, 0.0]])
T = np.array([[1.0], [0.0]])


###############################################################################
####### Model Construction                                              #######
###############################################################################

# Nonlinear Force
duff_force = CubicForce(Q, T, kalpha)

vib_sys = VibrationSystem(M, K, C=C)

vib_sys.add_nl_force(duff_force)


###############################################################################
####### Frequency Response                                              #######
###############################################################################

# Curve Parameters
h_max = 8 # 8 is consistent with the PRNM Paper SDOF
h = np.array(range(h_max+1))
fmag = 1.5 #0.161 # 1.5 is used for highest in Fig 1 of PRNM paper.
lam0 = 0.01
lam1 = 1.4

# Setup
Nhc = hutils.Nhc(h)

Fl = np.zeros(Nhc*Ndof)
Fl[2*Ndof] = 1 # Sine Forcing at Fundamental Harmonic on first DOF

# Solution at initial point
solver = NonlinearSolver

fun = lambda U : vib_sys.hbm_res(np.hstack((U, lam0)), fmag*Fl, h)[0:2]

U0 = np.zeros_like(Fl)

# Initial Nonlinear
X, R, dRdX, sol = solver.nsolve(fun, fmag*U0)

Uw0 = np.hstack((X, lam0))

continue_config = {'DynamicCtoP': True, 
                   'TargetNfev' : 200,
                   'MaxSteps'   : 8000,
                   'dsmin'      : 0.001,
                   'dsmax'      : 0.015, # 0.015 for plotting
                   'verbose'    : 100,
                   'xtol'       : 1e-9*Uw0.shape[0], 
                   'corrector'  : 'Ortho'} # Ortho, Pseudo

CtoP = hutils.harmonic_wise_conditioning(Uw0, Ndof, h, delta=1e-3*fmag)
CtoP[-1] = lam1 # so steps are approximately ds/sqrt(2) of lam1

dRdXC = dRdX*CtoP[:-1]
RPtoC = 1/np.max(np.abs(np.diag(dRdXC)))

cont_solver = Continuation(solver, ds0=0.01, CtoP=CtoP, RPtoC=RPtoC, config=continue_config)

fun = lambda Uw : vib_sys.hbm_res(Uw, fmag*Fl, h)

XlamP_full = cont_solver.continuation(fun, Uw0, lam0, lam1)


###############################################################################
####### Plot Frequency Response                                         #######
###############################################################################

dof = 0 

x1h1mag = np.sqrt(XlamP_full[:, Ndof+dof]**2 + XlamP_full[:, 2*Ndof+dof]**2)
if h_max >= 3:
    x1h3mag = np.sqrt(XlamP_full[:, 5*Ndof+dof]**2 + XlamP_full[:, 5*Ndof+Ndof+dof]**2)
if h_max >= 5:
    x1h5mag = np.sqrt(XlamP_full[:, 9*Ndof+dof]**2 + XlamP_full[:, 9*Ndof+Ndof+dof]**2)

#plt.ion() # Interactive
plt.plot(XlamP_full[:, -1], (x1h1mag), '-', label='Harmonic 1')
if h_max >= 3:
    plt.plot(XlamP_full[:, -1], (x1h3mag), '-', label='Harmonic 3')
if h_max >= 5:
    plt.plot(XlamP_full[:, -1], (x1h5mag), '-', label='Harmonic 5')
plt.ylabel('Harmonic Amplitude Coefficient [m]')
plt.xlabel('Frequency [rad/s]')
plt.xlim((0.5, 1.4))
plt.legend()
plt.show()


# Maximum Displacement Form of Response
Xt = hutils.time_series_deriv(1<<10, h, XlamP_full[:,  0:-1:2].T, 0)
Xmax = np.max(np.abs(Xt), axis=0)


plt.plot(XlamP_full[:, -1], Xmax, label='Total Max')
plt.ylabel('Maximum Displacement [m]')
plt.xlabel('Frequency [rad/s]')
plt.xlim((0.5, 1.4))
plt.ylim((0.0, 2.0))
plt.legend()
plt.show()


