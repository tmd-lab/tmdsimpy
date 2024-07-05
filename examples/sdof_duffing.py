"""
Simulations of a single degree of freedom (SDOF) Duffing oscillator

For verification, you can compare plots from this script to those published
in the paper:
    M. Volvert and F. Kerschen, 2021. Phase resonance nonlinear modes of
    mechanical systems. Journal of Sound and Vibration 511, 116355.
    https://doi.org/10.1016/j.jsv.2021.116355

Comments note some parameters that may require changes to exactly match
the cases shown in that paper.

The variable 'run_shooting' can be set to False to run faster without running
shooting. Shooting is used for stability assessment.

"""

import sys
import numpy as np

sys.path.append('..')
# import tmdsimpy

from tmdsimpy.nlforces.cubic_stiffness import CubicForce
from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.continuation import Continuation
import tmdsimpy.harmonic_utils as hutils
from tmdsimpy import postprocess


###############################################################################
####### System parameters                                               #######
###############################################################################

m = 1 # kg
c = 0.01 # kg/s
k = 1 # N/m
knl = 1 # N/m^3 # Nonlinear Stiffness

ab_damp = [c/m, 0]

# Flag to produce shooting results and compare to HBM (or False=don't do it)
run_shooting = True

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
h_max = 12 # 8 is consistent with the PRNM Paper, maximum number of super harmonics
h = np.array(range(h_max+1))
fmag = 1 # N
lam0 = 0.01 # Starting Frequency
lam1 = 10 # lam1 = 10 to recreate PRNM Paper. (Ending Frequency)
Nt = 128 # get reasonable results with 128

# Setup
Ndof = 1
Nhc = hutils.Nhc(h) # Number of Harmonic Components (2*h_max + 1)

# External Forcing Vector
Fl = np.zeros(Nhc*Ndof)
Fl[1] = 1 # Cosine Forcing at Fundamental Harmonic

# Solution at initial point
solver = NonlinearSolver()

fun = lambda U : vib_sys.hbm_res(np.hstack((U, lam0)), fmag*Fl, h, Nt=Nt)[0:2]

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
fun = lambda Uw : vib_sys.hbm_res(Uw, fmag*Fl, h, Nt=Nt)

# Actually solve the continuation problem.
XlamP_full = cont_solver.continuation(fun, Uw0, lam0, lam1)


###############################################################################
####### Shooting Calculations                                           #######
###############################################################################

if run_shooting:
    # lam0_shoot = 0.4 # Mostly follows the 3:1 IR
    lam0_shoot = 0.8 # Does a good job of the primary resonance in 100 steps

    Uw0_shoot = np.zeros(2*Ndof+1)

    Uw0_shoot[:2*Ndof] = Ulin_lam0[0][0:2*Ndof]
    Uw0_shoot[Ndof:2*Ndof] = lam0_shoot*Uw0_shoot[Ndof:2*Ndof]
    Uw0_shoot[-1] = lam0_shoot

    Fl_shoot = Fl[Ndof:3*Ndof] # Cosine and Sine of HBM vector

    # Initial Solve
    fun = lambda U : vib_sys.shooting_res(np.hstack((U, lam0_shoot)), Fl_shoot)[0:2]

    # Initial Nonlinear Solution Point
    X, R, dRdX, sol = solver.nsolve(fun, Uw0_shoot[:-1])

    Uw0_shoot2 = np.hstack((X, lam0_shoot))

    continue_config_shoot = {'DynamicCtoP': True,
                            'TargetNfev' : 10,
                            'MaxSteps'   : 100,
                            'dsmin'      : 0.001,
                            'dsmax'      : 0.1 , # 0.015 for plotting
                            'verbose'    : 10,
                            'xtol'       : 1e-9*Uw0_shoot.shape[0],
                            'corrector'  : 'Ortho'} # Ortho, Pseudo

    CtoP = np.array([1, 1/lam0_shoot, 1/lam0_shoot])

    # Set up an object to do the continuation
    cont_solver = Continuation(solver, ds0=0.01, CtoP=CtoP, RPtoC=None, 
                               config=continue_config_shoot)

    # Set up a function to pass to the continuation
    fun = lambda Uw : vib_sys.shooting_res(Uw, Fl_shoot)

    # Actually solve the continuation problem.
    XlamP_shoot = cont_solver.continuation(fun, Uw0_shoot2, lam0_shoot, lam1)

    print('Post processing shooting results.')
    y_shoot, ydot_shoot, stable, max_eig \
        = postprocess.shooting.time_stability(vib_sys, 
                                                XlamP_shoot, Fl_shoot, Nt=128)

    print('Finished post processing shooting.')

    print('Warning: Shooting and continuation are not well tuned and show some poor behavior.')

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

if run_shooting:
    y_max_stable = y_shoot[0, :, :].max(axis=0)
    y_max_stable[np.logical_not(stable)] = np.nan

    y_max_unstable = y_shoot[0, :, :].max(axis=0)
    y_max_unstable[stable] = np.nan

    plt.plot(XlamP_shoot[:, -1], y_max_stable, '--',
             label='Stable Shooting')

    plt.plot(XlamP_shoot[:, -1], y_max_unstable, ':',
             label='Unstable Shooting')

plt.ylabel('Maximum Displacement [m]')
plt.xlabel('Frequency [rad/s]')
plt.xlim((lam0, lam1))
plt.legend()
plt.show()

# Third harmonic phase for phase resonance marker
phih3 = np.arctan2(XlamP_full[:, 2*3*Ndof+dof], XlamP_full[:, (2*3-1)*Ndof+dof])
phih1 = np.arctan2(XlamP_full[:, 2*Ndof+dof], XlamP_full[:, Ndof+dof])

plt.plot(XlamP_full[:, -1], Xmax, label='Total Max')

if run_shooting:
    plt.plot(XlamP_shoot[:, -1], y_max_stable, '--',
             label='Stable Shooting')

    plt.plot(XlamP_shoot[:, -1], y_max_unstable, ':',
             label='Unstable Shooting')

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
####### Plot Frequency Response 3:1 Decomposition                       #######
###############################################################################

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

plt.rcParams['font.size'] = 16 # Default 10

mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Nice Labeled Settings
lw = 2 # Line Width
show_labels = True
h3_line = '-.'

# # Settings to Show Schematic without labels
# lw = 4 # Line Width
# show_labels = False
# h3_line = '-'

dof = 0 # SDOF, can only do dof=0 here.

# Maximum Displacement Form of Response
Xt = hutils.time_series_deriv(128, h, XlamP_full[:,  :-1].T, 0)
Xmax = np.max(np.abs(Xt), axis=0)

# Harmonic Amplitude Responses
x1h1mag = np.sqrt(XlamP_full[:, Ndof+dof]**2 + XlamP_full[:, 2*Ndof+dof]**2)
if h_max >= 3:
    x1h3mag = np.sqrt(XlamP_full[:, 5*Ndof+dof]**2 + XlamP_full[:, 5*Ndof+Ndof+dof]**2)


plt.plot(XlamP_full[:, -1], Xmax, '-', color='0.0',
         label='Total Max', linewidth=lw)

plt.plot(XlamP_full[:, -1], x1h1mag, '--', color='#0072B2',
         label='Harmonic 1', linewidth=lw)
# color = '0.4' or '#0072B2'

if h_max >= 3:
    plt.plot(XlamP_full[:, -1], x1h3mag, h3_line, color='#D55E00',
             label='Harmonic 3', linewidth=lw)
    # color = '0.6' or '#D55E00'

ax = plt.gca()
ax.tick_params(bottom=True, top=True, left=True, right=True,direction="in")

plt.xlim((0.345, 0.63))
plt.ylim((0.0, 1.6))

if show_labels:
    plt.ylabel('Amplitude [m]')
    plt.xlabel('Forcing Frequency [rad/s]')
    plt.legend(framealpha=1.0, frameon=False)
else:
    ax = plt.gca()
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

plt.savefig('duffing_superharmonic.eps', bbox_inches='tight')
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
