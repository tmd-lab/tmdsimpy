"""
Verification of the AFT implementation(s).
Currently:
    -Hysteretic SDOF Jenkins Element
    

failed_flag = False, changes to true if a test fails at any point 
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
import verification_utils as vutils

# Python Utilities
sys.path.append('../../ROUTINES/')
sys.path.append('../../ROUTINES/NL_FORCES')
import harmonic_utils as hutils

from jenkins_element import JenkinsForce


###############################################################################
#### Test Details                                                          ####
###############################################################################
# Test Details

failed_flag = False

analytical_sol_tol_stick = 1e-14 # Tolerance comparing to analytical solution
analytical_sol_tol_slip  = 1e-4 # Tolerance comparing to analytical solution

atol_grad = 1e-10 # Absolute gradient tolerance, a few places use 10* this


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
    grad_failed = vutils.check_grad(fun, np.array([u[indext]]), verbose=False, atol=atol_grad)
    failed_flag = failed_flag or grad_failed
    
    # Check U previous Derivative
    jenkins_force.up = up
    jenkins_force.fp = fp
    fun = lambda Up: modify_history_fun(jenkins_force, u[indext], udot[indext], Up, fp)[0:3:2]
    grad_failed = vutils.check_grad(fun, np.array([up]), verbose=False, atol=atol_grad)
    failed_flag = failed_flag or grad_failed
    
    # Check F previous derivative
    jenkins_force.up = up
    jenkins_force.fp = fp
    fun = lambda Fp: modify_history_fun(jenkins_force, u[indext], udot[indext], up, Fp)[0:4:3]
    grad_failed = vutils.check_grad(fun, np.array([fp]), verbose=False, atol=atol_grad)
    failed_flag = failed_flag or grad_failed
    
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
grad_failed = vutils.check_grad(fun, Unl, verbose=False, atol=atol_grad)
failed_flag = failed_flag or grad_failed

# Lots of harmonics and slipping check
h = np.array([0, 1, 2, 3])
Unl = np.array([[0.75, 0.2, 1.3, 2, 3, 4, 5]]).T
fun = lambda Unl : time_series_forces(Unl, h, Nt, w, jenkins_force)
grad_failed = vutils.check_grad(fun, Unl, verbose=False, atol=atol_grad*10)
failed_flag = failed_flag or grad_failed

# Stuck Check
h = np.array([0, 1, 2, 3])
Unl = np.array([[0.1, -0.1, 0.3, 0.1, 0.05, -0.1, 0.1]]).T
fun = lambda Unl : time_series_forces(Unl, h, Nt, w, jenkins_force)
grad_failed = vutils.check_grad(fun, Unl, verbose=False, atol=atol_grad)
failed_flag = failed_flag or grad_failed

print('Finished Checking Derivatives of time series w.r.t. harmonic coefficients.')


###############################################################################
#### Verification of Full AFT                                              ####
###############################################################################

w = 2.7

#############
# Basic with some slipping: 
h = np.array([0, 1])
U = np.array([[0.75, 0.2, 1.3]]).T
fun = lambda U : jenkins_force.aft(U, w, h)[0:2]
grad_failed = vutils.check_grad(fun, U, verbose=False, atol=atol_grad)
failed_flag = failed_flag or grad_failed
Fnl, dFnldU = fun(U)


# Numerically Verify Frequency Gradient
fun = lambda w: jenkins_force.aft(U, w[0], h)[0::2]
grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, atol=atol_grad)
failed_flag = failed_flag or grad_failed

#############
# Lots of harmonics and slipping check
h = np.array([0, 1, 2, 3])
U = np.array([[0.75, 0.2, 1.3, 2, 3, 4, 5]]).T
fun = lambda U : jenkins_force.aft(U, w, h)[0:2]
grad_failed = vutils.check_grad(fun, U, verbose=False, atol=atol_grad)
failed_flag = failed_flag or grad_failed
Fnl, dFnldU = fun(U)

# Numerically Verify Frequency Gradient
fun = lambda w: jenkins_force.aft(U, w[0], h)[0::2]
grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, atol=atol_grad)
failed_flag = failed_flag or grad_failed

#############
# Stuck Check
h = np.array([0, 1, 2, 3])
U = np.array([[0.1, -0.1, 0.3, 0.1, 0.05, -0.1, 0.1]]).T
fun = lambda U : jenkins_force.aft(U, w, h)[0:2]
grad_failed = vutils.check_grad(fun, U, verbose=False, atol=atol_grad)
failed_flag = failed_flag or grad_failed


# Numerically Verify Frequency Gradient
fun = lambda w: jenkins_force.aft(U, w[0], h)[0::2]
grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, atol=atol_grad)
failed_flag = failed_flag or grad_failed


fun = lambda U : jenkins_force.aft(U, w, h)[0:2]
Fnl, dFnldU = fun(U)

stiffness_error = np.abs(dFnldU - np.diag(np.array([0., 1., 1., 1., 1., 1., 1.])*kt)).max()
force_error = np.abs(Fnl / U.T / kt - np.array([0, 1, 1, 1, 1, 1, 1])).max()

failed_flag = failed_flag or stiffness_error > analytical_sol_tol_stick \
                          or force_error > analytical_sol_tol_stick

print('Linear Regime to Analytical Force Error: {:.4e}, Stiffness Error: {:.4e}'.format(force_error, stiffness_error))

# Lots of harmonics and slipping check
h = np.array([0, 1, 2, 3])
U = np.array([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
fun = lambda U : jenkins_force.aft(U, w, h)[0:2]
grad_failed = vutils.check_grad(fun, U, verbose=False, atol=atol_grad)
failed_flag = failed_flag or grad_failed
Fnl, dFnldU = fun(U)

# Numerically Verify Frequency Gradient
fun = lambda w: jenkins_force.aft(U, w[0], h)[0::2]
grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, atol=atol_grad)
failed_flag = failed_flag or grad_failed


# Limit of Full Slip Analytical Check
h = np.array([0, 1, 2, 3])
U = np.array([[0.0, 1e30, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
fun = lambda U : jenkins_force.aft(U, w, h)[0:2]
grad_failed = vutils.check_grad(fun, U, verbose=False, atol=atol_grad*10)
failed_flag = failed_flag or grad_failed

# Numerically Verify Frequency Gradient
fun = lambda w: jenkins_force.aft(U, w[0], h)[0::2]
grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, atol=atol_grad)
failed_flag = failed_flag or grad_failed

# Need lots of AFT Points to accurately converge Jenkins:
fun = lambda U : jenkins_force.aft(U, w, h, Nt=1<<17)[0:2]
Fnl, dFnldU = fun(U)

force_error = np.abs(Fnl - np.array([0, 0.0, -4*Fs/np.pi, 0.0, 0.0, 0.0, -4*Fs/np.pi/3])).max()

failed_flag = failed_flag or force_error > analytical_sol_tol_slip



print('Fully Slipping Regime to Analytical Force Error: {:.4e} (expected: 9e-5 with Nt=1<<17)'.format(force_error))


print('Finished Checking Derivatives of Harmonic Force w.r.t. Harmonic Displacement.')


###############################################################################
#### Test Results                                                          ####
###############################################################################

if failed_flag:
    print('\n\nTest FAILED, investigate results further!\n')
else:
    print('\n\nTest passed.\n')
