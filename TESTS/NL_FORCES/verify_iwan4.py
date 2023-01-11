# Verification of the hysteretic 4-parameter Iwan model

import sys
import numpy as np
import time
import matplotlib.pyplot as plt

sys.path.append('../.')
import verification_utils as vutils

sys.path.append('../../')
import harmonic_utils as hutils

# Python Utilities
sys.path.append('../../NL_FORCES')

from iwan4_element import Iwan4Force
from iwan_bb_conserve import ConservativeIwanBB

# Items to verify
#   1. Correct loading force displacement relationship 
#       (against conservative implementation)
#   2. The correct hysteretic forces (compare with Masing assumptions)
#   2a. Consistent dissipation across discretization / range of N to choose?
#   3. Correct derivatives of force at a given time instant with respect to 
#       those displacements
#   4. Correct harmonic derivaitves
#   5. Test for a range of values of parameters

###############################################################################
###### Parameters / Initialization                                       ######
###############################################################################

# Simple Mapping to spring displacements
Q = np.array([[1.0]])
T = np.array([[1.0]])

kt = 2.0
Fs = 3.0
chi = -0.1
beta = 0.1

Nsliders = 500

conservative_force = ConservativeIwanBB(Q, T, kt, Fs, chi, beta)
hysteretic_force   = Iwan4Force(Q, T, kt, Fs, chi, beta, Nsliders=Nsliders, alphasliders=1.0)

###############################################################################
###### Backbone Load-Displacement Relationship                           ######
###############################################################################

phimax = hysteretic_force.phimax

amp = 1.01*hysteretic_force.phimax

u_test = np.linspace(-amp, amp, 301)

f_hyst = np.zeros_like(u_test)
f_conv = np.zeros_like(u_test)

hysteretic_force.init_history(0)

for i in range(len(u_test)):
    
    f_hyst[i] = hysteretic_force.instant_force(u_test[i], 0, update_prev=False)[0]
    
    f_conv[i] = conservative_force.local_force_history(u_test[i], 0)[0]


print('Maximum difference on backbone: {:.3e}'.format(np.max(np.abs(f_hyst - f_conv))))

###############################################################################
###### Masing Load-Displacement Relationship                             ######
###############################################################################

# Generate Time Series
amp = 1.2*hysteretic_force.phimax

Unl = amp*np.array([0, 1, 0])
h = np.array([0, 1])
Nt = 1 << 8

w = 1.0
Nhc = hutils.Nhc(h)

# Nonlinear displacements, velocities in time
unlt = hutils.time_series_deriv(Nt, h, Unl.reshape(-1, 1), 0) # Nt x Ndnl
unltdot = w*hutils.time_series_deriv(Nt, h, Unl.reshape(-1, 1), 1) # Nt x Ndnl
cst = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 0)
unlth0 = unlt[0, :]

f_hyst = hysteretic_force.local_force_history(unlt, unltdot, h, cst, unlth0)[0]

# Construct the hysteretic response from the backbone and the Masing conditions
f_masing = np.zeros_like(f_hyst)

u_unload = (amp - unlt[0:129])/2
f_masing_unload = conservative_force.local_force_history(u_unload, 0*u_unload)[0]
f_masing[0:128] = f_masing_unload[128] - f_masing_unload[0:128]*2

u_reload = (amp + unlt[128:])/2
f_masing_reload = conservative_force.local_force_history(u_reload, 0*u_reload)[0]
f_masing[128:] = -f_masing_unload[128] + f_masing_reload*2


print('Maximum difference of hysteresis loops: {:.3e}'.format(np.max(np.abs(f_hyst - f_masing))))



# Quick Manual Check That the Force Makes Sense, can be commented out etc for 
# automated testing. 
plt.plot(unlt/hysteretic_force.phimax, f_hyst/hysteretic_force.Fs, label='Iwan Force')
plt.ylabel('Displacement/phi max')
plt.xlabel('Iwan Force/Fs')
plt.xlim((-1.1*amp/hysteretic_force.phimax, 1.1*amp/hysteretic_force.phimax))
plt.ylim((-1.1, 1.1))
# plt.legend()
plt.show()


###############################################################################
#### Verification of Force Time Serives Derivatives w.r.t. Harmonics       ####
###############################################################################

print('\nStarting checks of time series derivatives.')

def time_series_forces(Unl, h, Nt, w, iwan_force):
    
    # Unl = np.reshape(Unl, ((-1,1)))

    # Nonlinear displacements, velocities in time
    unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl
    unltdot = w*hutils.time_series_deriv(Nt, h, Unl, 1) # Nt x Ndnl
    
    Nhc = hutils.Nhc(h)
    cst = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 0)
    
    unlth0 = Unl[0]
    
    fnl, dfduh, dfdudh = iwan_force.local_force_history(unlt, unltdot, h, cst, unlth0)
    
    fnl = np.einsum('ij -> i', fnl)
    dfduh = np.einsum('ijkl -> il', dfduh)
    
    return fnl, dfduh

Nt = 1 << 7

h = np.array([0, 1, 2, 3])
Unl = phimax*np.array([[0.75, 0.2, 1.3, 2, 3, 4, 5]]).T

w = 1.0

fnl, dfduh = time_series_forces(Unl, h, Nt, w, hysteretic_force)


# Basic with some slipping: 
h = np.array([0, 1])
Unl = phimax*np.array([[0.75, 0.2, 1.3]]).T
fun = lambda Unl : time_series_forces(Unl, h, Nt, w, hysteretic_force)
vutils.check_grad(fun, Unl, verbose=False, atol=1e-9)

# Lots of harmonics and slipping check
h = np.array([0, 1, 2, 3])
Unl = phimax*np.array([[0.75, 0.2, 1.3, 2, 3, 4, 5]]).T
fun = lambda Unl : time_series_forces(Unl, h, Nt, w, hysteretic_force)
vutils.check_grad(fun, Unl, verbose=False, atol=1e-9)

# Stuck Check
h = np.array([0, 1, 2, 3])
Unl = phimax*np.array([[0.1, -0.1, 0.3, 0.1, 0.05, -0.1, 0.1]]).T
fun = lambda Unl : time_series_forces(Unl, h, Nt, w, hysteretic_force)
vutils.check_grad(fun, Unl, verbose=False, atol=1e-9)

print('Finished Checking Derivatives of time series w.r.t. harmonic coefficients.')



###############################################################################
#### Verification of Full AFT                                              ####
###############################################################################

print('\nStarting to check AFT Derivatives.')

w = 2.7

# Basic with some slipping: 
h = np.array([0, 1])
U = np.array([[0.75, 0.2, 1.3]]).T
fun = lambda U : hysteretic_force.aft(U, w, h)
vutils.check_grad(fun, U, verbose=False)
Fnl, dFnldU = fun(U)


# Lots of harmonics and slipping check
h = np.array([0, 1, 2, 3])
U = np.array([[0.75, 0.2, 1.3, 2, 3, 4, 5]]).T
fun = lambda U : hysteretic_force.aft(U, w, h)
vutils.check_grad(fun, U, verbose=False)
Fnl, dFnldU = fun(U)


# Stuck Check
h = np.array([0, 1, 2, 3])
U = np.array([[0.1, -0.1, 0.3, 0.1, 0.05, -0.1, 0.1]]).T
fun = lambda U : hysteretic_force.aft(U, w, h)
vutils.check_grad(fun, U, verbose=False)
Fnl, dFnldU = fun(U)

# Lots of harmonics and slipping check
h = np.array([0, 1, 2, 3])
U = np.array([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
fun = lambda U : hysteretic_force.aft(U, w, h)
vutils.check_grad(fun, U, verbose=False)
Fnl, dFnldU = fun(U)


# Limit of Full Slip Analytical Check
h = np.array([0, 1, 2, 3])
U = np.array([[0.0, 1e30, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
fun = lambda U : hysteretic_force.aft(U, w, h)
vutils.check_grad(fun, U, verbose=False, atol=1e-9)

print('Finished Checking AFT Derivatives.')

print('\nChecking against analytical fully slipped forces:')

# Need lots of AFT Points to accurately converge slipping state:
fun = lambda U : hysteretic_force.aft(U, w, h, Nt=1<<17)
Fnl, dFnldU = fun(U)

force_error = np.abs(Fnl - np.array([0, 0.0, -4*Fs/np.pi, 0.0, 0.0, 0.0, -4*Fs/np.pi/3])).max()

print('Fully Slipping Regime to Analytical Force Error: {:.4e} (expected: 9e-5 with Nt=1<<17)'.format(force_error))

print('Finished Checking AFT.')





