"""
Verification of the AFT implementation(s).
Currently:
    -Hysteretic SDOF Jenkins Element
"""

import sys
import numpy as np
import verification_utils as vutils
import matplotlib.pyplot as plt

sys.path.append('../../')
import harmonic_utils as hutils

# Python Utilities
sys.path.append('../../NL_FORCES')

from jenkins_element import JenkinsForce


###############################################################################
#### Function To Wrap History                                              ####
###############################################################################

def modify_history_fun(jenkins_force, u, udot, up, fp):
    
    jenkins_force.up = up
    jenkins_force.fp = fp
    
    fnl,dfnldunl,dfnldup,dfnldfp = jenkins_force.instant_force(u, udot, update_prev=False)
    
    return fnl,dfnldunl,dfnldup,dfnldfp

###############################################################################
#### Verification of SDOF Jenkins Implementation (Just the Force)          ####
###############################################################################

# Simple Mapping to displacements
Q = np.array([[1.0]])
T = np.array([[1.0]])

kt = 2.0
Fs = 3.0
umax = 5.0
freq = 1.4 # rad/s
Nt = 1<<8

jenkins_force = JenkinsForce(Q, T, kt, Fs)

jenkins_force.init_history(unlth0=0)

t = np.linspace(0, 2*np.pi, Nt+1)
t = t[:-1]

u = umax*np.sin(t)
udot = umax*freq*np.cos(t)

fhist = np.zeros_like(u)

for indext in range(t.shape[0]):
    fhist[indext],_,_,_ = jenkins_force.instant_force(u[indext], udot[indext], update_prev=True)
    

plt.plot(u, fhist, label='Jenkins Force')
plt.ylabel('Jenkins Displacement [m]')
plt.xlabel('Jenkins Force [N]')
plt.xlim((-1.1*umax, 1.1*umax))
plt.ylim((-1.1*Fs, 1.1*Fs))
# plt.legend()
plt.show()

###############################################################################
#### Verification of Derivatives of Jenkins Force                          ####
###############################################################################

# jenkins_force.init_history(unlth0=0)
Nt = 1<<5
t = np.linspace(0, 2*np.pi, Nt+1)
t = t[:-1]

u = umax*np.sin(t)
udot = umax*freq*np.cos(t)

fhist = np.zeros_like(u)

for indext in range(t.shape[0]):
    fnl,dfnldunl,dfnldup,dfnldfp = jenkins_force.instant_force(u[indext], udot[indext], update_prev=False)
    
    up = jenkins_force.up
    fp = jenkins_force.fp
    
    # Check U derivatives
    fun = lambda U: jenkins_force.instant_force(U, udot[indext], update_prev=False)[0:2]
    vutils.check_grad(fun, np.array([u[indext]]), verbose=False)
    
    # Check U previous Derivative
    jenkins_force.up = up
    jenkins_force.fp = fp
    fun = lambda Up: modify_history_fun(jenkins_force, u[indext], udot[indext], Up, fp)[0:3:2]
    vutils.check_grad(fun, np.array([up]), verbose=False)
    
    # Check F previous derivative
    jenkins_force.up = up
    jenkins_force.fp = fp
    fun = lambda Fp: modify_history_fun(jenkins_force, u[indext], udot[indext], up, Fp)[0:4:3]
    vutils.check_grad(fun, np.array([fp]), verbose=False)
    
    # Update History so derivatives can be checked at the next state.
    jenkins_force.up = up
    jenkins_force.fp = fp
    fhist[indext],_,_,_ = jenkins_force.instant_force(u[indext], udot[indext], update_prev=True)

print('Finished Checking Derivatives at points in time series.')



###############################################################################
#### Verification of Force Time Serives Derivatives w.r.t. Harmonics       ####
###############################################################################

def time_series_forces(Unl, h, Nt, w, jenkins_force):
    
    # Unl = np.reshape(Unl, ((-1,1)))

    # Nonlinear displacements, velocities in time
    unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl
    unltdot = w*hutils.time_series_deriv(Nt, h, Unl, 1) # Nt x Ndnl
    
    Nhc = hutils.Nhc(h)
    cst = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 0)
    
    unlth0 = Unl[0]
    
    fnl, dfduh, dfdudh = jenkins_force.local_force_history(unlt, unltdot, h, cst, unlth0)
    
    fnl = np.einsum('ij -> i', fnl)
    dfduh = np.einsum('ijkl -> il', dfduh)
    
    return fnl, dfduh

Nt = 1 << 7

h = np.array([0, 1, 2, 3])
Unl = np.array([[0.75, 0.2, 1.3, 2, 3, 4, 5]]).T

w = freq

fnl, dfduh = time_series_forces(Unl, h, Nt, w, jenkins_force)


# Basic with some slipping: 
h = np.array([0, 1])
Unl = np.array([[0.75, 0.2, 1.3]]).T
fun = lambda Unl : time_series_forces(Unl, h, Nt, w, jenkins_force)
vutils.check_grad(fun, Unl, verbose=False)

# Lots of harmonics and slipping check
h = np.array([0, 1, 2, 3])
Unl = np.array([[0.75, 0.2, 1.3, 2, 3, 4, 5]]).T
fun = lambda Unl : time_series_forces(Unl, h, Nt, w, jenkins_force)
vutils.check_grad(fun, Unl, verbose=False, atol=1e-9)

# Stuck Check
h = np.array([0, 1, 2, 3])
Unl = np.array([[0.1, -0.1, 0.3, 0.1, 0.05, -0.1, 0.1]]).T
fun = lambda Unl : time_series_forces(Unl, h, Nt, w, jenkins_force)
vutils.check_grad(fun, Unl, verbose=False)

print('Finished Checking Derivatives of time series w.r.t. harmonic coefficients.')


###############################################################################
#### Verification of Full AFT                                              ####
###############################################################################

w = 2.7

# Basic with some slipping: 
h = np.array([0, 1])
U = np.array([[0.75, 0.2, 1.3]]).T
fun = lambda U : jenkins_force.aft(U, w, h)
vutils.check_grad(fun, U, verbose=False)
Fnl, dFnldU = fun(U)


# Lots of harmonics and slipping check
h = np.array([0, 1, 2, 3])
U = np.array([[0.75, 0.2, 1.3, 2, 3, 4, 5]]).T
fun = lambda U : jenkins_force.aft(U, w, h)
vutils.check_grad(fun, U, verbose=False)
Fnl, dFnldU = fun(U)


# Stuck Check
h = np.array([0, 1, 2, 3])
U = np.array([[0.1, -0.1, 0.3, 0.1, 0.05, -0.1, 0.1]]).T
fun = lambda U : jenkins_force.aft(U, w, h)
vutils.check_grad(fun, U, verbose=False)
Fnl, dFnldU = fun(U)

stiffness_error = np.abs(dFnldU - np.diag(np.array([0., 1., 1., 1., 1., 1., 1.])*kt)).max()
force_error = np.abs(Fnl / U.T / kt - np.array([0, 1, 1, 1, 1, 1, 1])).max()

print('Linear Regime to Analytical Force Error: {:.4e}, Stiffness Error: {:.4e}'.format(force_error, stiffness_error))

# Lots of harmonics and slipping check
h = np.array([0, 1, 2, 3])
U = np.array([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
fun = lambda U : jenkins_force.aft(U, w, h)
vutils.check_grad(fun, U, verbose=False)
Fnl, dFnldU = fun(U)


# Limit of Full Slip Analytical Check
h = np.array([0, 1, 2, 3])
U = np.array([[0.0, 1e30, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
fun = lambda U : jenkins_force.aft(U, w, h)
vutils.check_grad(fun, U, verbose=False, atol=1e-9)

# Need lots of AFT Points to accurately converge Jenkins:
fun = lambda U : jenkins_force.aft(U, w, h, Nt=1<<17)
Fnl, dFnldU = fun(U)

force_error = np.abs(Fnl - np.array([0, 0.0, -4*Fs/np.pi, 0.0, 0.0, 0.0, -4*Fs/np.pi/3])).max()


print('Fully Slipping Regime to Analytical Force Error: {:.4e} (expected: 9e-5 with Nt=1<<17)'.format(force_error))


print('Finished Checking Derivatives of Harmonic Force w.r.t. Harmonic Displacement.')


