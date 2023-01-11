# Verification for Coupling Utils Functions
# Steps to Verify
#   1. Duffing Oscillator
#   2. Jenkins Element
#
# Parts to Verify
#   a. Phase condition being correct
#   b. Derivatives making sense


import sys
import numpy as np

import verification_utils as vutils

sys.path.append('../')
from solvers import NonlinearSolver
import coupling_utils as cutils

# Python Utilities
sys.path.append('../NL_FORCES')

from vector_jenkins import VectorJenkins
from cubic_stiffness import CubicForce

###############################################################################
########## Duffing Oscillator                                        ##########
###############################################################################

print('\n\nChecks with Duffing Oscillator:\n')

# Properties from SDOF Duffing
c = 0.01 # kg/s

# Nonlinear Force
Q = np.array([[1.0]])
T = np.array([[1.0]])
kalpha = np.array([1]) # N/m^3
duff_force = CubicForce(Q, T, kalpha)

# Approximate solution from graphs of internal resonant point
q31 = np.array([0.0, -1.1, 0.375]) 
omega_tilde = 0.5 # rad/s

### First Round of Checks

R, dRdq3, dRdq1 = cutils.ir_amp_ratio_res(duff_force, q31, c, omega_tilde)

print('Error in Phase Residual:', np.abs(R[0] - 0.0))

# Numerically Verify Gradient
fun = lambda q3: cutils.ir_amp_ratio_res(duff_force, np.hstack((q3, q31[-1])), c, omega_tilde)[0:2]
vutils.check_grad(fun, q31[:2])

fun = lambda q1: cutils.ir_amp_ratio_res(duff_force, np.hstack((q31[:-1], q1)), c, omega_tilde)[0::2]
vutils.check_grad(fun, q31[-1:])

### Amplitude Set 2
print('\nInput Values 2:')

q31 = np.array([1.1, 0.0, 0.375]) 

R, dRdq3, dRdq1 = cutils.ir_amp_ratio_res(duff_force, q31, c, omega_tilde)

print('Error in Phase Residual:', np.abs(R[0] - np.pi/2))

# Numerically Verify Gradient
fun = lambda q3: cutils.ir_amp_ratio_res(duff_force, np.hstack((q3, q31[-1])), c, omega_tilde)[0:2]
vutils.check_grad(fun, q31[:2])

fun = lambda q1: cutils.ir_amp_ratio_res(duff_force, np.hstack((q31[:-1], q1)), c, omega_tilde)[0::2]
vutils.check_grad(fun, q31[-1:])

print('Last set of derivatives of dRdq3 should analytically all be zero, thus check is ill-conditioned.')


###############################################################################
########## Check the Results Against Points from Continuation        ##########
###############################################################################

print('\nDuffing Oscillator Solution Verification:\n')

# Note: Only solutions with 3 harmonics from continuation can be compared here. 
# Adding more harmonics in continuation changes the result compared to 
# assumptions here and thus is not good for verification. 

# Exact matches are not expected since this is an approximation, this is just 
# to verify that it is reasonable. 

### Test 1

q31_0 = np.zeros(3)
q31_0[1] = -5
q31_0[-1] = 0.39802
omega_tilde = .48026

q3_expected = 1.05951 # From Continuation

fun = lambda q3 : cutils.ir_amp_ratio_res(duff_force, np.hstack((q3, q31_0[-1])), c, omega_tilde)[0:2]

# Solution at initial point
solver = NonlinearSolver
q3, R, dRdX, sol = solver.nsolve(fun, q31_0[:-1])

q3_mag = np.sqrt((q3**2).sum())

print('Test 1 Error: ', np.abs(q3_mag - q3_expected), \
      ' Pass: ', np.abs(q3_mag - q3_expected) < 0.04)


### Test 2

q31_0 = np.zeros(3)
q31_0[1] = -5
q31_0[-1] = 0.75
omega_tilde = 1.05946

q3_expected = 3.3183 # From Continuation

fun = lambda q3 : cutils.ir_amp_ratio_res(duff_force, np.hstack((q3, q31_0[-1])), c, omega_tilde)[0:2]

# Solution at initial point
solver = NonlinearSolver
q3, R, dRdX, sol = solver.nsolve(fun, q31_0[:-1])

q3_mag = np.sqrt((q3**2).sum())

print('Test 1 Error: ', np.abs(q3_mag - q3_expected), \
      ' Pass: ', np.abs(q3_mag - q3_expected) < 1e-3)


###############################################################################
########## Jenkins Element                                           ##########
###############################################################################

print('\n\nChecks with Jenkins Element:\n')

c = 0.01 # kg/s

# Nonlinear Force
Q = np.array([[1.0]])
T = np.array([[1.0]])

kt = 0.25*1e20 # N/m
Fs = 0.2 # N
Nt = 1<<12

vector_jenkins_force = VectorJenkins(Q, T, kt, Fs)

q31 = np.array([-1.0, 0.0, 1e20]) 
omega_tilde = 0.5 # rad/s

### First Round of Checks (Phase Check)

R, dRdq3, dRdq1 = cutils.ir_amp_ratio_res(vector_jenkins_force, q31, c, omega_tilde, Nt=Nt)

print('Error in Phase Residual (expected: 0.00230097 with Nt=1<<12):', np.abs(R[0] - 0.0))

# Numerically Verify Gradient
fun = lambda q3: cutils.ir_amp_ratio_res(vector_jenkins_force, np.hstack((q3, q31[-1])), c, omega_tilde, Nt=Nt)[0:2]
vutils.check_grad(fun, q31[:2])

fun = lambda q1: cutils.ir_amp_ratio_res(vector_jenkins_force, np.hstack((q31[:-1], q1)), c, omega_tilde, Nt=Nt)[0::2]
vutils.check_grad(fun, q31[-1:])

### Amplitude Set 2 / Reasonable values for parameters

print('\nInput Values 2:')

c = 0.01 # kg/s

# Nonlinear Force
Q = np.array([[1.0]])
T = np.array([[1.0]])

kt = 0.25 # N/m
Fs = 0.2 # N
uslip = Fs/kt
Nt = 1<<12

vector_jenkins_force = VectorJenkins(Q, T, kt, Fs)

omega_tilde = 0.5 # rad/s
q31 = np.array([1.1, 0.0, 1.01*uslip]) # q1 should be > uslip

R, dRdq3, dRdq1 = cutils.ir_amp_ratio_res(vector_jenkins_force, q31, c, omega_tilde, Nt=Nt)

# Numerically Verify Gradient
fun = lambda q3: cutils.ir_amp_ratio_res(vector_jenkins_force, np.hstack((q3, q31[-1])), c, omega_tilde)[0:2]
vutils.check_grad(fun, q31[:2])

fun = lambda q1: cutils.ir_amp_ratio_res(vector_jenkins_force, np.hstack((q31[:-1], q1)), c, omega_tilde)[0::2]
vutils.check_grad(fun, q31[-1:])


### Amplitude Set 2 / Reasonable values for parameters

print('\nInput Values 3:')

c = 0.01 # kg/s

# Nonlinear Force
Q = np.array([[1.0]])
T = np.array([[1.0]])

kt = 0.25 # N/m
Fs = 0.2 # N
Nt = 1<<12

vector_jenkins_force = VectorJenkins(Q, T, kt, Fs)

omega_tilde = 0.5 # rad/s
q31 = np.array([0.0, 0.1, 1.0]) # q1 should be > uslip

R, dRdq3, dRdq1 = cutils.ir_amp_ratio_res(vector_jenkins_force, q31, c, omega_tilde, Nt=Nt)

# Numerically Verify Gradient
fun = lambda q3: cutils.ir_amp_ratio_res(vector_jenkins_force, np.hstack((q3, q31[-1])), c, omega_tilde)[0:2]
vutils.check_grad(fun, q31[:2])

fun = lambda q1: cutils.ir_amp_ratio_res(vector_jenkins_force, np.hstack((q31[:-1], q1)), c, omega_tilde)[0::2]
vutils.check_grad(fun, q31[-1:])