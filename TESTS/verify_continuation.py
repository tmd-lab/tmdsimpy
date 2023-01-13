"""
Script for verifying the accuracy of the continuation methods

Outline:
    1. Setup a Nonlinear Model with Harmonic Balance (HBM)
    2. Solve HBM at a point
    3. Verify that the residual is appropriate at that point (e.g., 0 except arc length)
    4. Verify Gradients at the solution point
    5. Verify that the forward stepping solves the arc length residual exactly
    6. Try a full continuation (linear against FRF)
    
failed_flag = False, changes to true if a test fails at any point 

Notes:
    1. It would be better to have all the tolerances defined somewhere together
    rather than the current check of having them wherever they are used.
""" 

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

from continuation import Continuation

def continuation_test(fmag, Uw, Fl, h, solver, vib_sys, cont_config):
    
    test_failed = False
    
    ###############################################################################
    ####### 2. Solve HBM at point                                          #######
    ###############################################################################
    
    fun = lambda U : vib_sys.hbm_res(np.hstack((U, Uw[-1])), \
                                     fmag*Fl, h, Nt=128, aft_tol=1e-7)[0:2]
    
    X, R, dRdX, sol = solver.nsolve(fun, fmag*Uw[:-1])
    
    R_fun, dRdX_fun = fun(X)
    
    print('At resonance: Max(abs(U - Ulinear0))=', np.abs(X - fmag*Uw[:-1].reshape((-1))).max())
    
    if fmag < 1e-6:
        # Only check against linear resonance amplitude for light forcing 
        # (when it is near linear response)
        test_failed = test_failed or np.abs(X - fmag*Uw[:-1].reshape((-1))).max() > 1e-10
    
    ###############################################################################
    ####### 3. Verify Continuation Residual                                 #######
    ###############################################################################
    
    Uw0 = np.hstack((X, Uw[-1])) # Solution from previous HBM solve.
    
    CtoP = hutils.harmonic_wise_conditioning(Uw0, Ndof, h, delta=1e-3*fmag)
        
    ds = 0.01
    
    # Generate Continuation Model
    cont_solver = Continuation(solver, ds0=ds, CtoP=CtoP, config=cont_config)
    
    fun = lambda Uw : vib_sys.hbm_res(Uw, fmag*Fl, h, Nt=128, aft_tol=1e-7)
    
    ########### Continuation Residual at Initial Point:
    XlamC = Uw0 / CtoP
    XlamC0 = Uw0 / CtoP
    
    # Require a predictor: 
    XlamPprev = np.copy(Uw0)
    XlamPprev[-1] = XlamPprev[-1] - 1
    dirC = cont_solver.predict(fun, Uw0, XlamPprev)
    
    Raug, dRaugdXlamC = cont_solver.correct_res(fun, XlamC, XlamC0, ds, dirC)
    
    print('Max(non arc length Residual): ', np.max(np.abs(Raug[:-1])) )
    test_failed = test_failed or np.max(np.abs(Raug[:-1])) > 1e-5
    
    print('abs(Arc Residual) - 1 (should be 0): ', np.abs(Raug[-1]) - 1 )
    test_failed = test_failed or np.abs(np.abs(Raug[-1]) - 1) > 1e-14
    
    
    ###############################################################################
    ####### 4. Gradient of Augmented Equations                              #######
    ###############################################################################
    
    fun_aug = lambda XlamC : cont_solver.correct_res(fun, XlamC, XlamC0, ds, dirC)
    
    grad_passed = vutils.check_grad(fun_aug, XlamC, atol=0.0, rtol=1e-6)
    
    test_failed = test_failed or grad_passed
    
    
    ###############################################################################
    ####### 5. Forward Stepping Satisfyies Length Residual                  #######
    ###############################################################################
    
    dXlamPprev = CtoP*XlamC
    dXlamPprev[-1] -= CtoP[-1]*ds # Set to increasing frequency
    
    dirC = cont_solver.predict(fun, CtoP*XlamC, dXlamPprev)
    
    print('Frequency direction (should be positive):', np.sign(dirC[-1]))
    
    
    test_failed = test_failed or (not (np.sign(dirC[-1]) == 1.0))
    
    Raug, dRaugdXlamC = cont_solver.correct_res(fun, XlamC + ds*dirC, XlamC, ds, dirC)
    
    print('Arc Length Residual (should be 0):', Raug[-1])
    
    test_failed = test_failed or np.abs(Raug[-1]) > 1e-14
    
    return test_failed

###############################################################################
####### 1. Setup Nonlinear HBM Model                                    #######
###############################################################################

failed_flag = False

###########################
# Setup Nonlinear Force

# Simple Mapping to spring displacements
Q = np.array([[-1.0, 1.0, 0.0]])

# Weighted / integrated mapping back for testing purposes
# MATLAB implementation only supported T = Q.T for instantaneous forcing.
T = np.array([[-0.5], \
              [0.5], \
              [0.0] ])

kalpha = np.array([3.2])

duff_force = CubicForce(Q, T, kalpha)

###########################
# Setup Vibration System

M = np.array([[6.12, 3.33, 4.14],
              [3.33, 4.69, 3.42],
              [4.14, 3.42, 3.7 ]])

K = np.array([[3.0, 0.77, 1.8 ],
               [0.77, 2.48, 1.71],
               [1.8 , 1.71, 2.51]])


ab_damp = [0.0001, 0.0003]
C = ab_damp[0]*M + ab_damp[1]*K

vib_sys = VibrationSystem(M, K, C)

# Verify Mass and Stiffness Matrices are Appropriate
solver = NonlinearSolver

# lam,V = solver.eigs(M) # M must be positive definite.
# lam,V = solver.eigs(K) # K should be at least positive semi-definite.
lam,V = solver.eigs(K, M)

vib_sys.add_nl_force(duff_force)


###########################
# Solution Initial Guess
lam,V = solver.eigs(vib_sys.K, vib_sys.M)


h = np.array([0, 1, 2, 3, 4, 5]) 

Nhc = hutils.Nhc(h)
Ndof = M.shape[0]

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

qlinear = (vi @ Fl[1*Ndof:2*Ndof]) / np.sqrt( (wn**2 - w**2)**2 + (2*zeta*w*wn)**2)

# 90 deg. phase lag near resonance.
Uw[2*Ndof:3*Ndof] = qlinear * vi


###############################################################################
####### 0. Test Configurations                                          #######
###############################################################################

psuedo_config = {'corrector': 'Pseudo'}
ortho_config = {'corrector': 'ortho'}

###############################################################################
####### A. Linear Test                                                  #######
###############################################################################

print('Testing Essentially Linear System, Peusdo Arc Length:')

fmag = 0.0000001

test_failed = continuation_test(fmag, Uw, Fl, h, solver, vib_sys, psuedo_config)

failed_flag = failed_flag or test_failed

###############################################################################
####### B. Nonlinear Test                                               #######
###############################################################################

print('\n\nTesting Nonlinear System, Peusdo Arc Length:')

fmag = 1.0

# Augmented Jacobian Fails at the Nonlinear DOFS (1 and 2) at the First and 
# third harmonics (sine and cosine) in 2x2 blocks
# It is only wrong when doing the conditioning. Otherwise it is okay.

test_failed = continuation_test(fmag, Uw, Fl, h, solver, vib_sys, psuedo_config)

failed_flag = failed_flag or test_failed


###############################################################################
####### A. Linear Test                                                  #######
###############################################################################

print('\n\nTesting Essentially Linear System, Orthogonal Corrector:')

fmag = 0.0000001

test_failed = continuation_test(fmag, Uw, Fl, h, solver, vib_sys, ortho_config)

failed_flag = failed_flag or test_failed


###############################################################################
####### B. Nonlinear Test                                               #######
###############################################################################

print('\n\nTesting Nonlinear System, Orthogonal Corrector:')

fmag = 1.0

# Augmented Jacobian Fails at the Nonlinear DOFS (1 and 2) at the First and 
# third harmonics (sine and cosine) in 2x2 blocks
# It is only wrong when doing the conditioning. Otherwise it is okay.

test_failed = continuation_test(fmag, Uw, Fl, h, solver, vib_sys, ortho_config)

failed_flag = failed_flag or test_failed

###############################################################################
####### 6. Full Continuation Run                                        #######
###############################################################################

print('\n\nRunning a full linear continuation (pseudo corrector):')

fmag = 1.0 
lam0 = 0.2
lam1 = 3

# Linear system
vib_sys = VibrationSystem(M, K, ab=ab_damp)

kalpha = np.array([0.0])
duff_force = CubicForce(Q, T, kalpha)

vib_sys.add_nl_force(duff_force)

# Forcing
Fl = np.zeros((Nhc*Ndof,))
Fl[Ndof] = 1

# Solution at initial point
fun = lambda U : vib_sys.hbm_res(np.hstack((U, lam0)), fmag*Fl, h)[0:2]

U0stat = np.linalg.solve(vib_sys.K, Fl[Ndof:2*Ndof])
U0 = np.zeros_like(Fl)
U0[Ndof:2*Ndof] = U0stat

X, R, dRdX, sol = solver.nsolve(fun, fmag*U0)

Uw0 = np.hstack((U0, lam0))

continue_config = {'DynamicCtoP': True, 
                   'TargetNfev' : 200,
                   'MaxSteps'   : 2000,
                   'dsmin'      : 0.005,
                   'verbose'    : False,
                   'xtol'       : 5e-8*Uw0.shape[0], 
                   'corrector'  : 'Pseudo'}

CtoP = hutils.harmonic_wise_conditioning(Uw0, Ndof, h, delta=1e-3*fmag)
CtoP[-1] = lam1 # so steps are approximately ds/sqrt(2) of lam1

# print('Currently have all conditioning turned off.')
cont_solver = Continuation(solver, ds0=0.05, CtoP=CtoP, config=continue_config)

fun = lambda Uw : vib_sys.hbm_res(Uw, fmag*Fl, h)

XlamP_full = cont_solver.continuation(fun, Uw0, lam0, lam1)

# Compare Results to the Linear FRF:

Xwlinear = vib_sys.linear_frf(XlamP_full[:, -1], Fl[Ndof:2*Ndof], solver, 3)

error = np.max(np.abs(XlamP_full[:, Ndof:3*Ndof] - Xwlinear[:, :-1]))

print('Maximum Error between linear FRF and arc length: %.4e \n(Expected 2.85e-4, less than 1e-6 away from resonance.)' % (error))
# This check includes points near resonance and thus has error of 0.80043
# Plotting can be done to verify that the error is usually low
#    import matplotlib.pyplot as plt
#    plt.plot(np.log10(np.abs(XlamP_full[:, Ndof:3*Ndof] - Xwlinear[:, :-1])))
#    plt.show()

away_from_resonance_mask = np.max(np.sqrt(Xwlinear[:, :3]**2 + Xwlinear[:, 3:6]**2), axis=1) < 100

error = np.max(np.abs(XlamP_full[away_from_resonance_mask, Ndof:3*Ndof] \
                      - Xwlinear[away_from_resonance_mask, :-1]))
    
print('Away from resonance error is {:.4e} (expected less than 1e-5)'.format(error))

failed_flag = failed_flag or error > 1e-5

###############################################################################
####### 6. Full Continuation Run (Orthogonal)                           #######
###############################################################################


print('\n\nRunning a full linear continuation (orthogonal corrector):')

continue_config = {'DynamicCtoP': True, 
                   'TargetNfev' : 200,
                   'MaxSteps'   : 2000,
                   'dsmin'      : 0.005,
                   'verbose'    : False,
                   'xtol'       : 5e-8*Uw0.shape[0], 
                   'corrector'  : 'Ortho'}

cont_solver = Continuation(solver, ds0=0.05, CtoP=CtoP, config=continue_config)

XlamP_full = cont_solver.continuation(fun, Uw0, lam0, lam1)

Xwlinear = vib_sys.linear_frf(XlamP_full[:, -1], Fl[Ndof:2*Ndof], solver, 3)

error = np.max(np.abs(XlamP_full[:, Ndof:3*Ndof] - Xwlinear[:, :-1]))

print('Maximum Error between linear FRF and arc length: %.4e \n(Expected 8.35e-5, less than 1e-6 away from resonance.)' % (error))

failed_flag = failed_flag or error > 1e-4


if failed_flag:
    print('\n\nTest FAILED, investigate results further!\n')
else:
    print('\n\nTest passed.\n')
    


# ############# Plot to see what is going on.
# import matplotlib as mpl
# mpl.rcParams['lines.linewidth'] = 2

# import matplotlib.pyplot as plt

# dof = 0

# x1h1mag_lin = np.sqrt(Xwlinear[:, dof]**2 + Xwlinear[:, Ndof+dof]**2)

# x1h1mag = np.sqrt(XlamP_full[:, Ndof+dof]**2 + XlamP_full[:, 2*Ndof+dof]**2)
# x1h3mag = np.sqrt(XlamP_full[:, 5*Ndof+dof]**2 + XlamP_full[:, 5*Ndof+Ndof+dof]**2)

# plt.plot(XlamP_full[:, -1], np.log10(x1h1mag), label='Harmonic 1')
# plt.plot(Xwlinear[:, -1], np.log10(x1h1mag_lin), label='Linear 1', linestyle='--')
# plt.plot(XlamP_full[:, -1], np.log10(x1h3mag), label='Harmonic 3')
# plt.legend()
# plt.show()

