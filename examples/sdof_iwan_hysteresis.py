"""
Example of Frequency Response Curves (FRCs) for the Iwan hysteretic 
nonlinearity and a single degree of freedom (SDOF) system. 

Parameters are taken from: 
    J. H. Porter and M. R. W. Brake, in preparation, tracking superharmonic 
    resonances for nonlinear vibration.
    [See updated and final citation on README.md]

"""


import sys
import numpy as np

sys.path.append('..')
# import tmdsimpy

from tmdsimpy.nlforces.vector_iwan4 import VectorIwan4
from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.continuation import Continuation
import tmdsimpy.utils.harmonic as hutils

###############################################################################
####### System parameters                                               #######
###############################################################################

m = 1 # kg
c = 0.01 # kg/s
k = 0.75 # N/m


kt = 0.25 # N/m, Match Jenkins
Fs = 0.2 # N, Match Jenkins
chi = -0.5 # Have a more full hysteresis loop than chi=0.0
beta = 0.0 # Smooth Transition
Nsliders = 100

ab_damp = [c/m, 0]


h_max = 3 # Run 3 for paper or 8 for verification.
Nt = 1<<10 # number of steps for AFT evaluations

force_levels = [3.3, 38.0]

lam0 = 0.2
lam1 = 0.4


###############################################################################
####### Model Construction                                              #######
###############################################################################

# Nonlinear Force
Q = np.array([[1.0]])
T = np.array([[1.0]])

iwan_force = VectorIwan4(Q, T, kt, Fs, chi, beta, Nsliders=Nsliders, alphasliders=1.0)

# Setup Vibration System
M = np.array([[m]])

K = np.array([[k]])

ab_damp = [c/m, 0]

vib_sys = VibrationSystem(M, K, ab=ab_damp)

vib_sys.add_nl_force(iwan_force)



###############################################################################
####### Building Initial Guess                                          #######
###############################################################################

# Harmonic List to use in HBM
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


###############################################################################
####### Frequency Response Continuation                                 #######
###############################################################################

FRCs = [None] * len(force_levels)

for i in range(len(force_levels)):

    fmag = force_levels[i]
    
    fun = lambda U : vib_sys.hbm_res(np.hstack((U, lam0)), fmag*Fl, h, Nt=Nt)[0:2]
    # Initial Nonlinear Solution Point
    X, R, dRdX, sol = solver.nsolve(fun, fmag*U0)
    
    Uw0 = np.hstack((X, lam0))
    
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
    cont_solver = Continuation(solver, ds0=0.01, CtoP=CtoP, RPtoC=RPtoC, 
                               config=continue_config)
    
    # Set up a function to pass to the continuation
    fun = lambda Uw : vib_sys.hbm_res(Uw, fmag*Fl, h, Nt=Nt)
    
    # Actually solve the continuation problem. 
    XlamP_full = cont_solver.continuation(fun, Uw0, lam0, lam1)
    
    FRCs[i] = XlamP_full
    


###############################################################################
####### Plotting                                                        #######
###############################################################################

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

plt.rcParams['font.size'] = 16 # Default 10

mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amsmath}')

dof = 0 # SDOF, can only do dof=0 here.


for i in range(len(force_levels)):
    
    fmag = force_levels[i]
    
    XlamP_full = FRCs[i]
    
    # Maximum Displacement Form of Response
    Xt = hutils.time_series_deriv(128, h, XlamP_full[:,  :-1].T, 0)
    Xmax = np.max(np.abs(Xt), axis=0)
    
    # Harmonic Amplitude Responses
    x1h1mag = np.sqrt(XlamP_full[:, Ndof+dof]**2 + XlamP_full[:, 2*Ndof+dof]**2)
    if h_max >= 3:
        x1h3mag = np.sqrt(XlamP_full[:, 5*Ndof+dof]**2 + XlamP_full[:, 5*Ndof+Ndof+dof]**2)
    
    
    plt.plot(XlamP_full[:, -1], Xmax/fmag, '-', color='0.0', label='Total Max')
    
    plt.plot(XlamP_full[:, -1], x1h1mag/fmag, '--', color='#0072B2', label='Harmonic 1')
    # color = '0.4' or '#0072B2' 
    
    if h_max >= 3:
        plt.plot(XlamP_full[:, -1], x1h3mag/fmag, '-.', color='#D55E00', label='Harmonic 3')
        # color = '0.6' or '#D55E00'
        
    ax = plt.gca()
    ax.tick_params(bottom=True, top=True, left=True, right=True,direction="in")
    
    plt.ylabel('Amplitude Response/Force [m/N]')
    plt.xlabel('Forcing Frequency [rad/s]')
    plt.xlim((0.25, 0.35))
    plt.ylim((0.0, 1.85))
    plt.legend(framealpha=1.0, frameon=False)
    plt.savefig('iwan_superharmonic{}.eps'.format(i+1), bbox_inches='tight')
    plt.show()


