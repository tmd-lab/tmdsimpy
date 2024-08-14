"""
This example demonstrates a case with a negative duffing oscillator (softening)

The case uses Harmonic Balance Method (HBM) continuation with respect to 
frequency. Then HBM continuation with respect to force is conducted to try to 
identify what appears to be an isola at lower force levels.

Steps:
    1. Full FRC curve with HBM continuation w.r.t. frequency at low force level
    2. Repeat full FRC with HBM continuation w.r.t. frequency at high force 
    level. This time, it is run in positive and negative frequency directions 
    to get two branches. 
    3. Run continuation w.r.t. force scaling from a point on the high force 
    magnitude branch down to a similar point on the low force magnitude branch
    This bridges the gap to the isola
    4. Run contination w.r.t. frequency along the low force magnitude curve 
    using the point identified with the bridge.

The system here is the same as described in:
    J. H. Porter & M. R. W. Brake, Tracking Superharmonic Resonances for 
    Nonlinear Vibration. In MSSP (Under Review) Preprint: 
    arxiv.org/abs/2401.08790
"""


import sys
import numpy as np

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
import matplotlib.pyplot as plt

sys.path.append('..')

from tmdsimpy.nlforces.cubic_stiffness import CubicForce
from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.continuation import Continuation
import tmdsimpy.utils.harmonic as hutils


###############################################################################
####### System parameters                                               #######
###############################################################################

m = 1 # kg
c = 0.01 # kg/s
k = 1 # N/m
knl = -2.5e-4 # N/m^3 # Nonlinear Stiffness

ab_damp = [c/m, 0]

fmag0 = 0.33 # N # 0.33 N does not go negative
fmag1 = 8.0 # N - 8.0 is the high value in the paper, 0.38 also goes to zero frequency.

# Curve Parameters
h_max = 8 # Highest harmonic considered
lam0 = 0.01 # Starting Frequency
lam1 = 2 # Ending Frequency
Nt = 128 # get reasonable results with 128

bridge_w = 0.25 # Frequency to try to do continuation to bridge between force magnitudes

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

h = np.array(range(h_max+1))

# Setup
Ndof = 1
Nhc = hutils.Nhc(h) # Number of Harmonic Components (2*h_max + 1)

# External Forcing Vector
Fl = np.zeros(Nhc*Ndof)
Fl[1] = 1 # Cosine Forcing at Fundamental Harmonic

# Solution at initial point
solver = NonlinearSolver()


# Initial Linear Guess
Ulin_lam0 = vib_sys.linear_frf(np.array([lam0]), Fl[Ndof:2*Ndof], solver, neigs=1)

U0 = np.zeros_like(Fl)
U0[Ndof:3*Ndof] = Ulin_lam0[0][0:2*Ndof]

fun = lambda U : vib_sys.hbm_res(np.hstack((U, lam0)), fmag0*Fl, h, Nt=Nt)[0:2]
# Initial Nonlinear Solution Point
X, R, dRdX, sol = solver.nsolve(fun, fmag0*U0)

Uw0 = np.hstack((X, lam0))

###############################################################################
####### Frequency Response Continuation at fmag0                        #######
###############################################################################

continue_config = {'DynamicCtoP': True, 
                   'TargetNfev' : 200,
                   'MaxSteps'   : 4000,
                   'dsmin'      : 0.001,
                   'dsmax'      : 0.05 , # 0.015 for plotting
                   'verbose'    : 100,
                   'xtol'       : 1e-9*Uw0.shape[0], 
                   'corrector'  : 'Ortho',# Ortho, Pseudo
                   'backtrackStop' : lam0 # Just stop if start going to negative frequencies
                   } 

# Include conditioning so the solution works better
CtoP = hutils.harmonic_wise_conditioning(Uw0, Ndof, h, delta=1e-3*fmag0)
CtoP[-1] = lam1 # so steps are approximately ds/sqrt(2) of lam1

# More conditioning vector options
dRdXC = dRdX*CtoP[:-1]
RPtoC = 1/np.max(np.abs(np.diag(dRdXC)))


# Set up an object to do the continuation
cont_solver = Continuation(solver, ds0=0.01, CtoP=CtoP, RPtoC=RPtoC, config=continue_config)

# Set up a function to pass to the continuation
fun = lambda Uw : vib_sys.hbm_res(Uw, fmag0*Fl, h, Nt=Nt)

# Actually solve the continuation problem. 
XlamP_full0 = cont_solver.continuation(fun, Uw0, lam0, lam1)

###############################################################################
####### Frequency Response Continuation at fmag1                        #######
###############################################################################

fun = lambda U : vib_sys.hbm_res(np.hstack((U, lam0)), fmag1*Fl, h, Nt=Nt)[0:2]
# Initial Nonlinear Solution Point
X, R, dRdX, sol = solver.nsolve(fun, fmag1*U0)

Uw0 = np.hstack((X, lam0))

continue_config = {'DynamicCtoP': True, 
                   'TargetNfev' : 200,
                   'MaxSteps'   : 4000,
                   'dsmin'      : 0.001,
                   'dsmax'      : 0.02 , # 0.015 for plotting
                   'verbose'    : 100,
                   'xtol'       : 1e-9*Uw0.shape[0], 
                   'corrector'  : 'Ortho',# Ortho, Pseudo
                   'backtrackStop' : lam0 # Just stop if start going to negative frequencies
                   } 

# Include conditioning so the solution works better
CtoP = hutils.harmonic_wise_conditioning(Uw0, Ndof, h, delta=1e-3*fmag1)
CtoP[-1] = lam1 # so steps are approximately ds/sqrt(2) of lam1

# More conditioning vector options
dRdXC = dRdX*CtoP[:-1]
RPtoC = 1/np.max(np.abs(np.diag(dRdXC)))


# Set up an object to do the continuation
cont_solver = Continuation(solver, ds0=0.01, CtoP=CtoP, RPtoC=RPtoC, config=continue_config)

# Set up a function to pass to the continuation
fun = lambda Uw : vib_sys.hbm_res(Uw, fmag1*Fl, h, Nt=Nt)

# Actually solve the continuation problem. 
XlamP_full_forward = cont_solver.continuation(fun, Uw0, lam0, lam1)

# Solve continuation problem backwards to get other branch
XlamP_full_back = cont_solver.continuation(fun, Uw0, lam1, lam0)

XlamP_full1 = np.vstack((XlamP_full_forward, 
                         np.ones_like(XlamP_full_forward[0])*np.nan, 
                         XlamP_full_back))


###############################################################################
####### Continuation w.r.t. Force between to force magnitudes           #######
###############################################################################

# Find closest solution on top branch of high force magnitude to pull an exact
# starting point

bridge_ind = np.argmin(np.abs(XlamP_full_back[:, -1] - bridge_w))

Uw_fmag1 = XlamP_full_back[bridge_ind]

UF0 = np.hstack((Uw_fmag1[:-1], fmag1))


fun = lambda UF : vib_sys.hbm_res_dFl(UF, Uw_fmag1[-1], Fl, h, Nt=Nt)

R, dRdU, dRdF = fun(UF0)


continue_config = {'DynamicCtoP': True, 
                   'TargetNfev' : 200,
                   'MaxSteps'   : 4000,
                   'dsmin'      : 0.001,
                   'dsmax'      : 0.01 , # 0.015 for plotting
                   'verbose'    : 100,
                   'xtol'       : 1e-9*Uw0.shape[0], 
                   'corrector'  : 'Ortho',# Ortho, Pseudo
                   'backtrackStop' : lam0 # Just stop if start going to negative frequencies
                   } 


# Include conditioning so the solution works better
CtoP = hutils.harmonic_wise_conditioning(UF0, Ndof, h, delta=1e-3*fmag1)
CtoP[-1] = fmag1 # so steps are approximately ds/sqrt(2) of fmag1

# More conditioning vector options
dRdXC = dRdU*CtoP[:-1]
RPtoC = 1/np.max(np.abs(np.diag(dRdXC)))

# Set up an object to do the continuation
cont_solver = Continuation(solver, ds0=0.01, CtoP=CtoP, RPtoC=RPtoC, config=continue_config)


XFP_full_bridge = cont_solver.continuation(fun, UF0, fmag1, fmag0)


###############################################################################
####### Continuation w.r.t. Frequency from Bridge Point                 #######
###############################################################################

bridge_ind = np.argmin(np.abs(XFP_full_bridge[:, -1] - fmag0))

UF_bridge = XFP_full_bridge[bridge_ind]

Uw_low_bridge = np.hstack((UF_bridge[:-1], Uw_fmag1[-1]))

# Start a normal frequency continuation from this point now

# Set up a function to pass to the continuation
fun = lambda Uw : vib_sys.hbm_res(Uw, UF_bridge[-1]*Fl, h, Nt=Nt)

R, dRdX, dRdw = fun(Uw_low_bridge)

continue_config = {'DynamicCtoP': True, 
                   'TargetNfev' : 200,
                   'MaxSteps'   : 2000,
                   'dsmin'      : 0.001,
                   'dsmax'      : 0.05 , # 0.015 for plotting
                   'verbose'    : 100,
                   'xtol'       : 1e-9*Uw0.shape[0], 
                   'corrector'  : 'Ortho',# Ortho, Pseudo
                   'backtrackStop' : Uw_low_bridge[-1] 
                   } 

# Include conditioning so the solution works better
CtoP = hutils.harmonic_wise_conditioning(Uw0, Ndof, h, delta=1e-3*fmag0)
CtoP[-1] = lam1 # so steps are approximately ds/sqrt(2) of lam1

# More conditioning vector options
dRdXC = dRdX*CtoP[:-1]
RPtoC = 1/np.max(np.abs(np.diag(dRdXC)))


# Set up an object to do the continuation
cont_solver = Continuation(solver, ds0=0.01, CtoP=CtoP, RPtoC=RPtoC, config=continue_config)


# Actually solve the continuation problem. 
XlamP_full_isola_1 = cont_solver.continuation(fun, Uw_low_bridge, Uw_low_bridge[-1], lam1)
XlamP_full_isola_2 = cont_solver.continuation(fun, Uw_low_bridge, Uw_low_bridge[-1], 0)

XlamP_full_isola = np.vstack((np.flipud(XlamP_full_isola_2), XlamP_full_isola_1))


###############################################################################
####### Plot Frequency Response (Various)                               #######
###############################################################################

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

Nt_output = 1<<10

# Maximum Displacement Form of Response
Xt = hutils.time_series_deriv(Nt_output, h, XlamP_full0[:,  :-1].T, 0)
Xmax0 = np.max(np.abs(Xt), axis=0)

Xt = hutils.time_series_deriv(Nt_output, h, XlamP_full1[:,  :-1].T, 0)
Xmax1 = np.max(np.abs(Xt), axis=0)

Xt = hutils.time_series_deriv(Nt_output, h, XlamP_full_isola[:,  :-1].T, 0)
Xmax_isola = np.max(np.abs(Xt), axis=0)

Xt = hutils.time_series_deriv(Nt_output, h, XFP_full_bridge[:,  :-1].T, 0)
Xmax_bridge = np.max(np.abs(Xt), axis=0)

plt.plot(XlamP_full0[:, -1], Xmax0, label='F={:.3f} [N]'.format(fmag0))
plt.plot(XlamP_full1[:, -1], Xmax1, '--', label='F={:.3f} [N]'.format(fmag1))
plt.plot(XlamP_full_isola[:, -1], Xmax_isola, label='F={:.3f} [N]'.format(UF_bridge[-1]))

plt.ylabel('Maximum Displacement [m]')
plt.xlabel('Frequency [rad/s]')
plt.xlim((lam0, lam1))
plt.ylim((0.0, Xmax1[np.isfinite(Xmax1)].max()*1.05))
plt.legend()
plt.savefig('soften_duffing_large.eps', bbox_inches='tight')
plt.show()




plt.plot(XFP_full_bridge[:, -1], Xmax_bridge)
plt.xlabel('External Force Scaling [N]')
plt.ylabel('Maximum Amplitude [m]')
plt.show()


plt.plot(XlamP_full_isola[:, -1], Xmax_isola, label='F={:.3f} [N]'.format(UF_bridge[-1]))
plt.ylabel('Maximum Displacement [m]')
plt.xlabel('Frequency [rad/s]')
plt.xlim((0, XlamP_full_isola[:, -1].max()*1.05))
plt.legend()
plt.savefig('soften_duffing_isola.eps', bbox_inches='tight')
plt.show()


