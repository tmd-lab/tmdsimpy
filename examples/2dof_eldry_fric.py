"""
Simulation of 2 Degree of Freedom System with elastic dry friction

Diagram 
(There are also springs in the normal and tangential directions, not shown)
               ____________
               |    m     | 
               |__________|
                   |
                   <
                   >  kn
       ___/\/\/\__| 
      |     kt 
_____\/__mu*N____________________

"""

import sys
import numpy as np

sys.path.append('..')
# import tmdsimpy

from tmdsimpy.jax.nlforces.elastic_dry_fric_2d import ElasticDryFriction2D
from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.continuation import Continuation
import tmdsimpy.harmonic_utils as hutils


import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
import matplotlib.pyplot as plt


###############################################################################
####### System Parameters                                               #######
###############################################################################

# Parameters based octave-jim repo example

m = 1.0
kx = 12.0 
cx = 2*0.05*np.sqrt(kx/m)
kn = 100.0 # This gets overwritten shortly but this matches the octave-jim example
cn = 2*0.1*np.sqrt(kn/m)

kt = 5.0 
kn = 21.0
mu = 0.85

Fstat = np.array([0, 100.0])
Fdyn  = np.array([1.0, 0.1]) # First harmonic

###########################
# Setup given Parameters - build model

M_mat = m * np.eye(2)
K_mat = np.array([[kx, 0], [0, kn]])
C_mat = np.array([[cx, 0], [0, cn]])

Q = np.eye(2)
T = np.eye(2)


# Friction Model
eldry = ElasticDryFriction2D(Q, T, kt, kn, mu, u0=0.0)

# Vibration system
vib_sys = VibrationSystem(M_mat, K_mat, C=C_mat)

vib_sys.add_nl_force(eldry)


###############################################################################
####### Frequency Response                                              #######
###############################################################################

# Curve Parameters
h_max = 8 # 8 is consistent with the PRNM Paper SDOF
h = np.array(range(h_max+1))
fmag = [0.5, 1, 10, 25, 50, 80, 100, 300] 
lam0 = 0.01
lam1 = 6.0

# Setup
Nhc = hutils.Nhc(h)
Ndof = M_mat.shape[0]

Fl = np.zeros(Nhc*Ndof)

# Solution at initial point
solver = NonlinearSolver()

continue_config = {'DynamicCtoP': True, 
                   'TargetNfev' : 200,
                   'MaxSteps'   : 8000,
                   'dsmin'      : 0.001,
                   'dsmax'      : 0.1, # 0.015 for plotting
                   'verbose'    : 100,
                   'xtol'       : 1e-9*Fl.shape[0], 
                   'corrector'  : 'Ortho'} # Ortho, Pseudo

Xmax = len(fmag) * [None] # Tangent
Xnmax = len(fmag) * [None] # Normal
XlamPList = len(fmag) * [None]


for ind in range(len(fmag)):
    
    Fl = np.zeros(Nhc*Ndof)
    Fl[:Ndof] = Fstat
    Fl[Ndof:2*Ndof] = fmag[ind]*Fdyn # Cosine Forcing at Fundamental Harmonic

    fun = lambda U : vib_sys.hbm_res(np.hstack((U, lam0)), Fl, h)[0:2]
    
    U0 = np.zeros_like(Fl)
    
    # Initial Nonlinear
    X, R, dRdX, sol = solver.nsolve(fun, fmag[ind]*U0, verbose=False)
    
    Uw0 = np.hstack((X, lam0))
    
    
    CtoP = hutils.harmonic_wise_conditioning(Uw0, Ndof, h, delta=1e-3*fmag[ind])
    CtoP[-1] = lam1 # so steps are approximately ds/sqrt(2) of lam1
    
    dRdXC = dRdX*CtoP[:-1]
    RPtoC = 1/np.max(np.abs(np.diag(dRdXC)))
    
    cont_solver = Continuation(solver, ds0=0.05, CtoP=CtoP, RPtoC=RPtoC, 
                               config=continue_config)
    
    fun = lambda Uw : vib_sys.hbm_res(Uw, Fl, h)
    
    XlamP_full = cont_solver.continuation(fun, Uw0, lam0, lam1)
    
    
    # Maximum Displacement Form of Response
    Xt = hutils.time_series_deriv(1<<10, h, XlamP_full[:,  0:-1:2].T, 0)
    Xmax[ind] = np.max(np.abs(Xt), axis=0)
    
    # Normal direction
    Xt = hutils.time_series_deriv(1<<10, h, XlamP_full[:,  1:-1:2].T, 0)
    Xnmax[ind] = np.max(np.abs(Xt), axis=0)
    
    XlamPList[ind] = XlamP_full

###############################################################################
####### Extended Periodic Motion Concept                                #######
###############################################################################

a0 = -2 # Log scale amplitude
a1 = 4

Uwx0 = np.zeros(Nhc*Ndof+2)
Uwx0[1] = Fstat[1] / (kn + kn) # Static normal displacement
Uwx0[4] = 1.0 # 1 in sine direction since Fl is in cose direction
Uwx0[-2] = np.sqrt( (kt+kx) / m )

Fl = np.zeros(Nhc*Ndof)
Fl[:Ndof] = Fstat
Fl[Ndof:2*Ndof] = Fdyn # Cosine Forcing at Fundamental Harmonic for orthogonality


fun = lambda Uwx : vib_sys.epmc_res(np.hstack((Uwx, a0)), \
                                 Fl, h, Nt=1<<10)[0:2]
    
X, R, dRdX, sol = solver.nsolve(fun, Uwx0, verbose=False)

Uwxa0 = np.hstack((X, a0))


CtoP = hutils.harmonic_wise_conditioning(Uwxa0, Ndof, h, delta=1e-3)
CtoP[-1] = np.abs(a0) # so steps are approximately ds/sqrt(2) of lam1
    
dRdXC = dRdX*CtoP[:-1]
RPtoC = 1/np.max(np.abs(np.diag(dRdXC)))

cont_solver = Continuation(solver, ds0=0.05, CtoP=CtoP, RPtoC=RPtoC, 
                           config=continue_config)
    
fun = lambda Uwxa : vib_sys.epmc_res(Uwxa, Fl, h, Nt=1<<10)

Uwxa_bb = cont_solver.continuation(fun, Uwxa0, a0, a1)
    

# Maximum Displacement Form of Response - Tangent
Xharm =  Uwxa_bb[:,  0:-3:2].T
Xharm[1:, :] = Xharm[1:, :]*(10**Uwxa_bb[:, -1:].T) # Amplitude scaling of all except zeroth
Xt_epmc = hutils.time_series_deriv(1<<10, h, Xharm, 0)
Xt_epmc = np.max(np.abs(Xt_epmc), axis=0)

# Normal direction
Xharm =  Uwxa_bb[:,  1:-3:2].T
Xharm[1:, :] = Xharm[1:, :]*(10**Uwxa_bb[:, -1:].T) # Amplitude scaling of all except zeroth
Xn_epmc = hutils.time_series_deriv(1<<10, h, Xharm, 0)
Xn_epmc = np.max(np.abs(Xn_epmc), axis=0)

###############################################################################
####### Plotting                                                        #######
###############################################################################



for ind in range(len(fmag)):
    plt.plot(XlamPList[ind][:, -1], Xmax[ind], 
             label='Fscale={}'.format(fmag[ind]))
    
plt.plot(Uwxa_bb[:, -3], Xt_epmc, 'k--', label='EPMC')
    
plt.ylabel('Maximum Displacement [m]')
plt.xlabel('Frequency [rad/s]')
plt.xlim((lam0, lam1))
# plt.ylim((0.0, 2.0))
plt.yscale('log')
plt.legend()
plt.show()



for ind in range(len(fmag)):
    plt.plot(XlamPList[ind][:, -1], Xnmax[ind], 
             label='Fscale={}'.format(fmag[ind]))
    
plt.plot(Uwxa_bb[:, -3], Xn_epmc, 'k--', label='EPMC')
    
plt.ylabel('Maximum Displacement [m]')
plt.xlabel('Frequency [rad/s]')
plt.xlim((lam0, lam1))
# plt.ylim((0.0, 2.0))
plt.yscale('log')
plt.legend()
plt.show()
