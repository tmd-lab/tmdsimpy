"""
Script for verifying the correctness of Harmonic Balance Method
    
failed_flag = False, changes to true if a test fails at any point 

check_matlab can be set to False if you do not have MATLAB/python integration

Notes:
    1. It would be better to have all the tolerances defined somewhere together
    rather than the current check of having them wherever they are used.
""" 

check_matlab = True # Set to False if you do not have MATLAB/python integration


import sys
import numpy as np

# Path to Harmonic balance / vibration system 
sys.path.append('../ROUTINES/')
sys.path.append('../ROUTINES/NL_FORCES')

from cubic_stiffness import CubicForce
from vibration_system import VibrationSystem
from solvers import NonlinearSolver
import harmonic_utils as hutils
import verification_utils as vutils

if check_matlab:
    # Location of mat file to compare
    import os
    wdir = os.getcwd()
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.cd(wdir + '/MATLAB_VERSIONS/')


###########################
# Tolerances and Failure Flag

# Flag for checking all tests at end.
failed_flag = False

matlab_tol = 1e-12

grad_rtol = 5e-10

nearlin_tol = 1e-12 # Tolerance for linear analytical v. HBM check

###########################
# Setup Nonlinear Force

# Simple Mapping to spring displacements
Q = np.array([[-1.0, 1.0, 0.0]])

# Weighted / integrated mapping back for testing purposes
# MATLAB implementation only supported T = Q.T for instantaneous forcing.
T = np.array([[-1.0], \
              [1.0], \
              [0.0] ])

kalpha = np.array([3.2])

duff_force = CubicForce(Q, T, kalpha)

###########################
# Setup Vibration System

M = np.array([[6.12, 3.33, 4.14],
              [3.33, 4.69, 3.42],
              [4.14, 3.42, 3.7 ]])

K = np.array([[2.14, 0.77, 1.8 ],
              [0.77, 2.15, 1.71],
              [1.8 , 1.71, 2.12]])

C = 0.01 * M + 0.02*K

vib_sys = VibrationSystem(M, K, C)

# Verify Mass and Stiffness Matrices are Appropriate
solver = NonlinearSolver

# lam,V = solver.eigs(M) # M must be positive definite.
# lam,V = solver.eigs(K) # K should be at least positive semi-definite.
lam,V = solver.eigs(K, M)

vib_sys.add_nl_force(duff_force)

###########################
# Evaluate Harmonic Balance Residual

h = np.array([0, 1, 2, 3, 4]) 

Nhc = hutils.Nhc(h)

Uw = np.array([2.79, 2.14, 4.06, 2.61, 1.02, 0.95, 1.25, 3.28, 2.09, 0.97, 4.98,
       1.48, 1.13, 2.49, 3.34, 4.35, 0.69, 4.84, 3.27, 2.03, 3.82, 2.86,
       0.99, 3.52, 3.8 , 3.4 , 1.89, 0.75])

#Uw = np.atleast_2d(Uw).T

Fl = np.zeros((27,))
Fl[1*3] = 1.0 # First Harmonic Cosine, DOF 1
Fl[3*3] = 0.8 # Second Harmonic Cosine, DOF 1

R, dRdU, dRdw = vib_sys.hbm_res(Uw, Fl, h, Nt=128, aft_tol=1e-7)

###########################
# Compare to the MATLAB Solution

if check_matlab:
    print('\nMATLAB Comparison:')
    mat_sol = eng.load('duffing_3DOF', 'R', 'dRdU', 'dRdw')
    
    print('Residual: ')
    error = vutils.compare_mats(R, mat_sol['R'])
    failed_flag = failed_flag or error > matlab_tol
    
    print('Gradient: ')
    error = vutils.compare_mats(dRdU, mat_sol['dRdU'])
    failed_flag = failed_flag or error > matlab_tol
    
    print('Gradient w.r.t. w: ')
    error = vutils.compare_mats(dRdw, mat_sol['dRdw'])
    failed_flag = failed_flag or error > matlab_tol


###########################
# Verify Gradients

print('\nDisplacement Gradient:')
fun = lambda U : vib_sys.hbm_res(np.hstack((U, Uw[-1])), Fl, h, Nt=128, aft_tol=1e-7)[0:2]
grad_failed = vutils.check_grad(fun, Uw[:-1], rtol=grad_rtol)
failed_flag = failed_flag or grad_failed


print('Frequency Gradient:')
fun = lambda w : vib_sys.hbm_res(np.hstack((Uw[:-1], w)), Fl, h, Nt=128, aft_tol=1e-7)[0:3:2]
grad_failed = vutils.check_grad(fun, np.atleast_1d(Uw[-1]), rtol=grad_rtol)
failed_flag = failed_flag or grad_failed


###########################
# Solve at a point

# Remove the rigid body mode.
vib_sys.K = np.array([[3.0, 0.77, 1.8 ],
                       [0.77, 2.48, 1.71],
                       [1.8 , 1.71, 2.51]])

ab_damp = [0.0001, 0.0003]
vib_sys.C = ab_damp[0]*vib_sys.M + ab_damp[1]*vib_sys.K

lam,V = solver.eigs(vib_sys.K, vib_sys.M)


h = np.array([0, 1, 2, 3, 4, 5]) 

Nhc = hutils.Nhc(h)
Ndof = M.shape[0]

fmag = 0.0000001

mi = 0 # mode of interest
wn = np.sqrt(lam[mi])
w = wn # Force at near resonance.
vi = V[:, mi]

Fl = np.zeros((Nhc*Ndof,))
Fl[1*Ndof:2*Ndof] = (M @ vi) # First Harmonic Cosine, DOF 1


Uw = np.zeros((Nhc*Ndof+1,))
Uw[-1] = w 

# Mode 2 proportional damping
zeta = ab_damp[0]/w/2 +  ab_damp[1]*w/2

qlinear = fmag*(vi @ Fl[1*Ndof:2*Ndof]) / np.sqrt( (wn**2 - w**2)**2 + (2*zeta*w*wn)**2)

# 90 deg. phase lag near resonance.
Uw[2*Ndof:3*Ndof] = qlinear * vi

fun = lambda U : vib_sys.hbm_res(np.hstack((U, Uw[-1])), \
                                 fmag*Fl, h, Nt=128, aft_tol=1e-7)[0:2]

X, R, dRdX, sol = solver.nsolve(fun, Uw[:-1])

R_fun, dRdX_fun = fun(X)

print('Verifying solver outputs final state R and dRdX:')

solver_error_R = np.abs(R-R_fun).max()
solver_error_dRdX = np.abs(dRdX-dRdX_fun).max()

failed_flag = failed_flag or solver_error_R > 1e-16
failed_flag = failed_flag or solver_error_dRdX > 1e-16

print('max(R-R_fun):', solver_error_R, \
      ', max(dRdX-dRdX_fun):', solver_error_dRdX)

'''
print('Comparison [U0_sin, Ufinal_cos, Ufinal_sin, mag]')
print( np.hstack((Uw[2*Ndof:3*Ndof], X[1*Ndof:2*Ndof].reshape((-1,1)), \
                  X[2*Ndof:3*Ndof].reshape((-1,1)), \
                  np.sqrt(X[1*Ndof:2*Ndof].reshape((-1,1))**2 + X[2*Ndof:3*Ndof].reshape((-1,1))**2) )))
'''

linear_solve_error = np.abs(X - Uw[:-1].reshape((-1))).max()

failed_flag = failed_flag or linear_solve_error > nearlin_tol

print('Essentially Linear System, resonance: Max(abs(U - U0))=', linear_solve_error)

###################
# Nonlinear Point Solution
print('\nNonlinear Solution:')

fmag = 1.0

qlinear = fmag*(vi @ Fl[1*Ndof:2*Ndof]) / np.sqrt( (wn**2 - w**2)**2 + (2*zeta*w*wn)**2)

# 90 deg. phase lag near resonance.
Uw[2*Ndof:3*Ndof] = (qlinear * vi)


fun = lambda U : vib_sys.hbm_res(np.hstack((U, Uw[-1])), \
                                 fmag*Fl, h, Nt=128, aft_tol=1e-7)[0:2]

X, R, dRdX, sol = solver.nsolve(fun, Uw[:-1])

R_fun, dRdX_fun = fun(X)

print('Verifying solver outputs final state R and dRdX:')

solver_error_R = np.abs(R-R_fun).max()
solver_error_dRdX = np.abs(dRdX-dRdX_fun).max()

failed_flag = failed_flag or solver_error_R > 1e-16
failed_flag = failed_flag or solver_error_dRdX > 1e-16

print('max(R-R_fun):', solver_error_R, \
      ', max(dRdX-dRdX_fun):', solver_error_dRdX)

print('norm(R)=', np.linalg.norm(R))


'''
# As expected, shows stiffening behavior and low amplitude since it is not 
# actually near resonance.

print('Comparison [U0_sin, Ufinal_cos, Ufinal_sin, mag]')
print( np.hstack((Uw[2*Ndof:3*Ndof], X[1*Ndof:2*Ndof].reshape((-1,1)), \
                  X[2*Ndof:3*Ndof].reshape((-1,1)), \
                  np.sqrt(X[1*Ndof:2*Ndof].reshape((-1,1))**2 + X[2*Ndof:3*Ndof].reshape((-1,1))**2) )))
'''


###########################
# Verify Gradients

print('\nDisplacement Gradient:')
fun = lambda U : vib_sys.hbm_res(np.hstack((U, Uw[-1])), fmag*Fl, h, Nt=128, aft_tol=1e-7)[0:2]
grad_failed = vutils.check_grad(fun, X, rtol=grad_rtol)
failed_flag = failed_flag or grad_failed


print('Frequency Gradient:')
fun = lambda w : vib_sys.hbm_res(np.hstack((X, w)), fmag*Fl, h, Nt=128, aft_tol=1e-7)[0:3:2]
grad_failed = vutils.check_grad(fun, np.atleast_1d(Uw[-1]), rtol=grad_rtol)
failed_flag = failed_flag or grad_failed


###########################
# Final Result
if failed_flag:
    print('\n\nTest FAILED, investigate results further!\n')
else:
    print('\n\nTest passed.\n')
    