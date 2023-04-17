# Simulation of a Single Degree of Freedom Duffing Oscillator
#
# References for FRC's with identical parameters:
#   M. Volvert and G. Kerschen, 2021, Phase resonance nonlinear modes of 
#   mechanical systems. 
# Comments note the values that match PRNM paper for parameters that are 
# frequently changed.

import sys
import numpy as np

sys.path.append('..')
# import tmdsimpy

from tmdsimpy.nlforces.cubic_stiffness import CubicForce
from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.continuation import Continuation
import tmdsimpy.harmonic_utils as hutils

# # Path to Harmonic balance / vibration system 
# sys.path.append('../ROUTINES')
# sys.path.append('../ROUTINES/NL_FORCES')

# from cubic_stiffness import CubicForce
# from vibration_system import VibrationSystem

# from solvers import NonlinearSolver
# from continuation import Continuation

# import harmonic_utils as hutils


###############################################################################
####### System parameters                                               #######
###############################################################################

m = 1 # kg
c = 0.01 # kg/s
k = 1 # N/m
knl = 1 # N/m^3 # Nonlinear Stiffness

ab_damp = [c/m, 0]

###############################################################################
####### Model Construction                                              #######
###############################################################################

# Nonlinear Force
Q = np.array([[1.0]])
T = np.array([[1.0]])

kalpha = np.array([knl])

duff_force = CubicForce(Q, T, kalpha)


# Setup Vibration System
M = np.array([[m]])

K = np.array([[k]])

ab_damp = [c/m, 0]

vib_sys = VibrationSystem(M, K, ab=ab_damp)

vib_sys.add_nl_force(duff_force)

###############################################################################
####### Frequency Response Initialization                               #######
###############################################################################

# Curve Parameters
h_max = 8 # 8 is consistent with the PRNM Paper, maximum number of super harmonics
h = np.array(range(h_max+1))
fmag = 1 #N
lam0 = 0.01 # Starting Frequency
lam1 = 10 #lam1 = 10 to recreate PRNM Paper. (Ending Frequency)

# Setup
Ndof = 1
Nhc = hutils.Nhc(h) # Number of Harmonic Components (2*h_max + 1)

# External Forcing Vector
Fl = np.zeros(Nhc*Ndof)
Fl[1] = 1 # Cosine Forcing at Fundamental Harmonic

# Solution at initial point
solver = NonlinearSolver

fun = lambda U : vib_sys.hbm_res(np.hstack((U, lam0)), fmag*Fl, h)[0:2]

# Initial Linear Guess
Ulin_lam0 = vib_sys.linear_frf(np.array([lam0]), Fl[Ndof:2*Ndof], solver, neigs=1)

U0 = np.zeros_like(Fl)
U0[Ndof:3*Ndof] = Ulin_lam0[0][0:2*Ndof]

# Initial Nonlinear Solution Point
X, R, dRdX, sol = solver.nsolve(fun, fmag*U0)

Uw0 = np.hstack((X, lam0))

###############################################################################
####### Frequency Response Continuation                                 #######
###############################################################################

continue_config = {'DynamicCtoP': True, 
                   'TargetNfev' : 200,
                   'MaxSteps'   : 8000,
                   'dsmin'      : 0.001,
                   'dsmax'      : 0.015 , # 0.015 for plotting
                   'verbose'    : 100,
                   'xtol'       : 1e-9*Uw0.shape[0], 
                   'corrector'  : 'Ortho'} # Ortho, Pseudo

# Include conditioning so the solution works better
CtoP = hutils.harmonic_wise_conditioning(Uw0, Ndof, h, delta=1e-3*fmag)
CtoP[-1] = lam1 # so steps are approximately ds/sqrt(2) of lam1

# More conditioning vector options
dRdXC = dRdX*CtoP[:-1]
RPtoC = 1/np.max(np.abs(np.diag(dRdXC)))


# Set up an object to do the continuation
cont_solver = Continuation(solver, ds0=0.01, CtoP=CtoP, RPtoC=RPtoC, config=continue_config)

# Set up a function to pass to the continuation
fun = lambda Uw : vib_sys.hbm_res(Uw, fmag*Fl, h)

# Actually solve the continuation problem. 
XlamP_full = cont_solver.continuation(fun, Uw0, lam0, lam1)



###############################################################################
####### Plot Frequency Response (Various)                               #######
###############################################################################

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2

import matplotlib.pyplot as plt

dof = 0 # SDOF, can only do dof=0 here.

Xwlinear = vib_sys.linear_frf(XlamP_full[:, -1], fmag*Fl[Ndof:2*Ndof], solver, 3)
x1h1mag_lin = np.sqrt(Xwlinear[:, dof]**2 + Xwlinear[:, Ndof+dof]**2)

x1h1mag = np.sqrt(XlamP_full[:, Ndof+dof]**2 + XlamP_full[:, 2*Ndof+dof]**2)
if h_max >= 3:
    x1h3mag = np.sqrt(XlamP_full[:, 5*Ndof+dof]**2 + XlamP_full[:, 5*Ndof+Ndof+dof]**2)
if h_max >= 5:
    x1h5mag = np.sqrt(XlamP_full[:, 9*Ndof+dof]**2 + XlamP_full[:, 9*Ndof+Ndof+dof]**2)

#plt.ion() # Interactive
plt.plot(XlamP_full[:, -1], np.log10(x1h1mag), '-', label='Harmonic 1')
# plt.plot(Xwlinear[:, -1], np.log10(x1h1mag_lin), label='Linear 1', linestyle='--')
if h_max >= 3:
    plt.plot(XlamP_full[:, -1], np.log10(x1h3mag), '-', label='Harmonic 3')
if h_max >= 5:
    plt.plot(XlamP_full[:, -1], np.log10(x1h5mag), '-', label='Harmonic 5')
plt.ylabel('Log Harmonic Amplitude Coefficient [m]')
plt.xlabel('Frequency [rad/s]')
plt.xlim((lam0, lam1))
plt.legend()
plt.show()

# Maximum Displacement Form of Response
Xt = hutils.time_series_deriv(128, h, XlamP_full[:,  :-1].T, 0)
Xmax = np.max(np.abs(Xt), axis=0)


plt.plot(XlamP_full[:, -1], Xmax, label='Total Max')
plt.ylabel('Maximum Displacement [m]')
plt.xlabel('Frequency [rad/s]')
plt.xlim((lam0, lam1))
plt.legend()
plt.show()



# Third harmonic phase for phase resonance marker
phih3 = np.arctan2(XlamP_full[:, 2*3*Ndof+dof], XlamP_full[:, (2*3-1)*Ndof+dof])
phih1 = np.arctan2(XlamP_full[:, 2*Ndof+dof], XlamP_full[:, Ndof+dof])

plt.plot(XlamP_full[:, -1], Xmax, label='Total Max')
plt.ylabel('Maximum Displacement [m]')
plt.xlabel('Frequency [rad/s]')
plt.xlim((lam0, 1))
plt.ylim((0, 2))
plt.legend()
plt.show()



# First Harmonic Plot - Phase:
x1h1mag = np.sqrt(XlamP_full[:, Ndof+dof]**2 + XlamP_full[:, 2*Ndof+dof]**2)
phih1 = np.arctan2(XlamP_full[:, 2*Ndof+dof], XlamP_full[:, Ndof+dof])

plt.plot(XlamP_full[:, -1], phih1, label="First Harmonic Phase")

plt.ylabel('Harmonic Phase [rad]')
plt.xlabel('Frequency [rad/s]')
plt.xlim((lam0, lam1))
plt.legend()
plt.show()



###############################################################################
####### Calculate the Displacement and Velocity Histories               #######
###############################################################################

# In general, each row represents a solution point. 
# Columns are displacements followed by control variable (lambda = frequency)
# Displacement ordering is 
# [U0, U1c, U1s, U2c, U2s . . . ] with c and s being cosine and sine respectively.

Nt = 2**7 # Number of points within a cycle to use (must be power of 2)

Uw = XlamP_full

# List of Frequencies of the solution points
w = Uw[:, -1]

# Position time series for each solution point
# First dimension is time, second dimension is solution point
Xt = hutils.time_series_deriv(Nt, h, Uw[:,  :-1].T, 0)

# Normalized velocity divided by frequency time histories
# First dimension is time, second dimension is solution point
Xtdot = hutils.time_series_deriv(Nt, h, Uw[:,  :-1].T, 1)

# Velocities in real units
Xtdot = Xtdot*w

# Normalized time within a cycle [0,1] = t/T
# Note that time t=T is not included since it just repeates time t=0.
tau = np.linspace(0, 1, Nt+1)[:-1]

# Use Xt and Xtdot to plot phase portraits