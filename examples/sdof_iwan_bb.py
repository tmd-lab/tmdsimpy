"""
Example figures for 3:1 superharmonic resonance for conservative Iwan backbone
model
"""

import sys
import numpy as np

sys.path.append('..')
# import tmdsimpy

from tmdsimpy.nlforces.iwan_bb_conserve import ConservativeIwanBB
from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.continuation import Continuation
import tmdsimpy.harmonic_utils as hutils


###############################################################################
####### System parameters                                               #######
###############################################################################

m = 1 # kg
c = 0.01 # kg/s
k = 0.75 # N/m

# Nonlinear Parameters
kt = 0.25 # N/m
Fs = 0.2 # N
chi = 0.0
beta = 0.0

ab_damp = [c/m, 0]


###############################################################################
####### Model Construction                                              #######
###############################################################################

# Nonlinear Force
Q = np.array([[1.0]])
T = np.array([[1.0]])


softening_force = ConservativeIwanBB(Q, T, kt, Fs, chi, beta)

# Setup Vibration System
M = np.array([[m]])

K = np.array([[k]])

ab_damp = [c/m, 0]

vib_sys = VibrationSystem(M, K, ab=ab_damp)

vib_sys.add_nl_force(softening_force)

###############################################################################
####### Frequency Response Initialization                               #######
###############################################################################

# Curve Parameters
h_max = 3 # Maximum number of super harmonics
h = np.array(range(h_max+1))
fmag = 2 #N
lam0 = 0.01 # Starting Frequency
lam1 = 0.5 #lam1 = 10 to recreate PRNM Paper. (Ending Frequency)

# Setup
Ndof = 1
Nhc = hutils.Nhc(h) # Number of Harmonic Components (2*h_max + 1)

# External Forcing Vector
Fl = np.zeros(Nhc*Ndof)
Fl[1] = 1 # Cosine Forcing at Fundamental Harmonic

# Solution at initial point
solver = NonlinearSolver()

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
####### Frequency Response Continuation with only 1st Harmonic          #######
###############################################################################

# Initial Solution
h2 = np.array([0, 1])
Nhc2 = hutils.Nhc(h2)

# External Forcing Vector
Fl2 = np.zeros(Nhc2*Ndof)
Fl2[1] = 1 # Cosine Forcing at Fundamental Harmonic

U0 = np.zeros_like(Fl2)
U0[Ndof:3*Ndof] = Ulin_lam0[0][0:2*Ndof]

# Initial Nonlinear Solution Point
fun = lambda U : vib_sys.hbm_res(np.hstack((U, lam0)), fmag*Fl2, h2)[0:2]

X, R, dRdX, sol = solver.nsolve(fun, fmag*U0)

Uw0_2 = np.hstack((X, lam0))

# Actual Continuation

continue_config = {'DynamicCtoP': True, 
                   'TargetNfev' : 200,
                   'MaxSteps'   : 8000,
                   'dsmin'      : 0.001,
                   'dsmax'      : 0.015 , # 0.015 for plotting
                   'verbose'    : 100,
                   'xtol'       : 1e-9*Uw0_2.shape[0], 
                   'corrector'  : 'Ortho'} # Ortho, Pseudo

# Include conditioning so the solution works better
CtoP = hutils.harmonic_wise_conditioning(Uw0_2, Ndof, h2, delta=1e-3*fmag)
CtoP[-1] = lam1 # so steps are approximately ds/sqrt(2) of lam1

# More conditioning vector options
dRdXC = dRdX*CtoP[:-1]
RPtoC = 1/np.max(np.abs(np.diag(dRdXC)))


# Set up an object to do the continuation
cont_solver = Continuation(solver, ds0=0.01, CtoP=CtoP, RPtoC=RPtoC, config=continue_config)

# Set up a function to pass to the continuation
fun = lambda Uw : vib_sys.hbm_res(Uw, fmag*Fl2, h2)

# Actually solve the continuation problem. 
XlamP_full2 = cont_solver.continuation(fun, Uw0_2, lam0, lam1)


###############################################################################
####### Plot Frequency Response (Various)                               #######
###############################################################################

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 3


# plt.rcParams['text.usetex'] = True
mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amsmath}')

mpl.style.use('seaborn-v0_8-colorblind')

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14 # Default 10


# Maximum Displacement Form of Response
Xt = hutils.time_series_deriv(128, h, XlamP_full[:,  :-1].T, 0)
Xmax = np.max(np.abs(Xt), axis=0)


# Maximum Displacement Form of Response
Xt2 = hutils.time_series_deriv(128, h2, XlamP_full2[:,  :-1].T, 0)
Xmax2 = np.max(np.abs(Xt2), axis=0)


plt.plot(XlamP_full[:, -1], Xmax/fmag, label='Superharmonic Resonance')
plt.plot(XlamP_full2[:, -1], Xmax2/fmag, '--', label='Single Harmonic Solution')
plt.ylabel('Maximum Displacement [m/N]')
plt.xlabel('Frequency [rad/s]')
plt.xlim((0.26, 0.36))
plt.ylim((0, 2.75))
plt.legend()

# plt.savefig('iwanbb_h3_superharmonic.png', bbox_inches='tight', dpi=300)

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
Uw2 = XlamP_full2


# List of Frequencies of the solution points
w = Uw[:, -1]
w2 = Uw2[:, -1]


# Position time series for each solution point
# First dimension is time, second dimension is solution point
Xt = hutils.time_series_deriv(Nt, h, Uw[:,  :-1].T, 0)
Xt2 = hutils.time_series_deriv(Nt, h2, Uw2[:,  :-1].T, 0)


h1_mask = np.zeros((Nhc, 1))
h1_mask[1:3] = 1

h3_mask = np.zeros_like(h1_mask)
h3_mask[5:7] = 1

Xt_h1 = hutils.time_series_deriv(Nt, h, Uw[:,  :-1].T*h1_mask, 0)
Xt_h3 = hutils.time_series_deriv(Nt, h, Uw[:,  :-1].T*h3_mask, 0)

Xt = np.vstack((Xt, Xt[0, :]))
Xt2 = np.vstack((Xt2, Xt2[0, :]))
Xt_h1 = np.vstack((Xt_h1, Xt_h1[0, :]))
Xt_h3 = np.vstack((Xt_h3, Xt_h3[0, :]))


# Normalized velocity divided by frequency time histories
# First dimension is time, second dimension is solution point
Xtdot = hutils.time_series_deriv(Nt, h, Uw[:,  :-1].T, 1)

# Velocities in real units
Xtdot = Xtdot*w

# Normalized time within a cycle [0,1] = t/T
# Note that time t=T is not included since it just repeates time t=0.
tau = np.linspace(0, 1, Nt+1)



time_ind = np.argmax(Xmax)
time_ind2 = np.argmin(np.abs(w2 - w[time_ind]))

ylimits = (-2.75, 2.75)

fig, axs = plt.subplots(2, gridspec_kw={'hspace': 0.05, 'wspace': 0.2})

axs[0].plot(tau, Xt2[:, time_ind2]/fmag, label='Total Motion')
axs[0].plot(tau, Xt2[:, time_ind2]/fmag, '--', label='Harmonic 1')
axs[0].plot(tau, Xt_h3[:, time_ind]/fmag*np.nan, '-.', label='Harmonic 3')
axs[0].set_xlim((0, 1))
axs[0].set_ylim(ylimits)
axs[0].set_ylabel('$\\boldsymbol{X}$ [m/N]')
axs[0].legend(framealpha=1.0, frameon=False)
axs[0].set_title('Single Harmonic Solution', pad=-100)

axs[1].plot(tau, Xt[:, time_ind]/fmag)
axs[1].plot(tau, Xt_h1[:, time_ind]/fmag, '--')
axs[1].plot(tau, Xt_h3[:, time_ind]/fmag, '-.')
axs[1].set_xlim((0, 1))
axs[1].set_ylim(ylimits)
axs[1].set_ylabel('$\\boldsymbol{X}$ [m/N]')
axs[1].set_xlabel('$\\boldsymbol{t/T}$')
axs[1].set_title('Superharmonic Resonance', pad=-20)



axs[0].tick_params(bottom=True, top=True, left=True, right=True,direction="in")
axs[1].tick_params(bottom=True, top=True, left=True, right=True,direction="in")
axs[0].xaxis.set_tick_params(labelbottom=False)

# fig.savefig('iwanbb_time_series.eps', bbox_inches='tight')

plt.show()
