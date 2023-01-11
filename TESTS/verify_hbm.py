import sys
import numpy as np

# Path to Harmonic balance / vibration system 
sys.path.append('../')
sys.path.append('../NL_FORCES')

from cubic_stiffness import CubicForce
from vibration_system import VibrationSystem
from solvers import NonlinearSolver
import harmonic_utils as hutils
import verification_utils as vutils

# Location of mat file to compare
import os
wdir = os.getcwd()
import matlab.engine
eng = matlab.engine.start_matlab()
eng.cd(wdir + '/MATLAB_VERSIONS/')

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

print('\nMATLAB Comparison:')
mat_sol = eng.load('duffing_3DOF', 'R', 'dRdU', 'dRdw')

print('Residual: ')
vutils.compare_mats(R, mat_sol['R'])

print('Gradient: ')
vutils.compare_mats(dRdU, mat_sol['dRdU'])

print('Gradient w.r.t. w: ')
vutils.compare_mats(dRdw, mat_sol['dRdw'])


###########################
# Verify Gradients

print('\nDisplacement Gradient:')
fun = lambda U : vib_sys.hbm_res(np.hstack((U, Uw[-1])), Fl, h, Nt=128, aft_tol=1e-7)[0:2]
vutils.check_grad(fun, Uw[:-1])


print('Frequency Gradient:')
fun = lambda w : vib_sys.hbm_res(np.hstack((Uw[:-1], w)), Fl, h, Nt=128, aft_tol=1e-7)[0:3:2]
vutils.check_grad(fun, np.atleast_1d(Uw[-1]))


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

print('max(R-R_fun):', np.abs(R-R_fun).max(), \
      ', max(dRdX-dRdX_fun):', np.abs(dRdX-dRdX_fun).max())

'''
print('Comparison [U0_sin, Ufinal_cos, Ufinal_sin, mag]')
print( np.hstack((Uw[2*Ndof:3*Ndof], X[1*Ndof:2*Ndof].reshape((-1,1)), \
                  X[2*Ndof:3*Ndof].reshape((-1,1)), \
                  np.sqrt(X[1*Ndof:2*Ndof].reshape((-1,1))**2 + X[2*Ndof:3*Ndof].reshape((-1,1))**2) )))
'''

print('Essentially Linear System, resonance: Max(abs(U - U0))=', np.abs(X - Uw[:-1].reshape((-1))).max())

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

print('max(R-R_fun):', np.abs(R-R_fun).max(), \
      ', max(dRdX-dRdX_fun):', np.abs(dRdX-dRdX_fun).max())
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
vutils.check_grad(fun, X)


print('Frequency Gradient:')
fun = lambda w : vib_sys.hbm_res(np.hstack((X, w)), fmag*Fl, h, Nt=128, aft_tol=1e-7)[0:3:2]
vutils.check_grad(fun, np.atleast_1d(Uw[-1]))

