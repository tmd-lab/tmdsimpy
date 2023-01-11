# Steps to Verify
#   5. Linear system check
#   6. Run a 2+ DOF Jenkins and Duffing Backbone and check some peak points 
#       against HBM continuations

import sys
import numpy as np

# Path to Harmonic balance / vibration system 
sys.path.append('../')
sys.path.append('../NL_FORCES')

from cubic_stiffness import CubicForce
from vector_jenkins import VectorJenkins
from vibration_system import VibrationSystem

from solvers import NonlinearSolver
from continuation import Continuation

import harmonic_utils as hutils
import verification_utils as vutils


###############################################################################
###### EPMC Continuation Function                                        ######
###############################################################################

def epmc_cont(vib_sys, a0, a1, h):
    
    solver = NonlinearSolver 
    
    #### Select Mode of Interest + Initial State
    # eigvals,eigvecs = solver.eigs(vib_sys.K, vib_sys.M)
    eigvals,eigvecs = solver.eigs(vib_sys.K, M=vib_sys.M, subset_by_index=[0,0])
    
    Ndof = eigvecs.shape[0]
    Nhc = hutils.Nhc(h)
    
    U = np.zeros(Ndof*Nhc)
    h0 = 1*(h[0] == 0)
    U[h0*Ndof:(1+h0)*Ndof] = eigvecs[:, 0]
    
    lin_mode = eigvecs[:, 0]
    
    Uwx0 = np.hstack((U, np.sqrt(eigvals[0]), vib_sys.ab[0]))
    
    # Phase constraint consistent with placement of eigenvector
    Fl = np.zeros(Ndof*hutils.Nhc(h))
    Fl[(h0+1)*Ndof:(h0+2)*Ndof] = 1
    
    #### Solve for an initial step
    fun = lambda Uwx : vib_sys.epmc_res(np.hstack((Uwx, a0)), \
                                     Fl, h, Nt=1<<10, aft_tol=1e-7)[0:2]
        
    X, R, dRdX, sol = solver.nsolve(fun, Uwx0)
    
    Uwxa0 = np.hstack((X, a0))
    
    #### Continuation Run
    
    CtoP = hutils.harmonic_wise_conditioning(Uwxa0, Ndof, h, delta=1e-3)
    
    cont_config = {'DynamicCtoP': True, 
                    'TargetNfev' : 200,
                    'MaxSteps'   : 2000,
                    'dsmin'      : 0.001,
                    'verbose'    : False,
                    'corrector'  : 'Ortho'}
    
    # Generate Continuation Model
    cont_solver = Continuation(solver, ds0=0.01, CtoP=CtoP, config=cont_config)
    
    fun = lambda Uwxa : vib_sys.epmc_res(Uwxa, Fl, h, Nt=1<<10, aft_tol=1e-7)
    
    Uwxa_full = cont_solver.continuation(fun, Uwxa0, a0, a1)
    
    return Uwxa_full,lin_mode



###############################################################################
###### Build Systems                                                     ######
###############################################################################

####### Parameters
m = 1 # kg
c = 0.01 # kg/s
k = 1 # N/m

# Duffing Parameters
knl = 1 # N/m^3
knl2dof = 0.5 # N/m^3 - Matches the typical example problem

# Jenkins Parameters
kt = 0.25 # N/m
Fs = 0.2 # N

ab_damp = [c/m, 0]


####### SDOF
# Nonlinear Force
Q = np.array([[1.0]])
T = np.array([[1.0]])

duff_force = CubicForce(Q, T, knl)
vector_jenkins_force = VectorJenkins(Q, T, kt, Fs)


# Setup Vibration System
M = np.array([[m]])

K = np.array([[k]])

ab_damp = [c/m, 0]

sdof_duffing = VibrationSystem(M, K, ab=ab_damp)
sdof_duffing.add_nl_force(duff_force)

sdof_jenkins = VibrationSystem(M, K, ab=ab_damp)
sdof_jenkins.add_nl_force(vector_jenkins_force)

####### 2-DOF Systems
# Nonlinear Force
Q = np.array([[0.0, 1.0]])
T = np.array([[0.0], [1.0]])

duff_force = CubicForce(Q, T, knl2dof)
vector_jenkins_force = VectorJenkins(Q, T, kt, Fs)


# Setup Vibration System
M = np.array([[1.0, 0.0], [0.0, 1.0]])

K = np.array([[2.0, -1.0], [-1.0, 2.0]])

ab_damp = [0.01, 0]

mdof_linear = VibrationSystem(M, K, ab=ab_damp)

mdof_duffing = VibrationSystem(M, K, ab=ab_damp)
mdof_duffing.add_nl_force(duff_force)

mdof_jenkins = VibrationSystem(M, K, ab=ab_damp)
mdof_jenkins.add_nl_force(vector_jenkins_force)

all_systems = [sdof_duffing, sdof_jenkins, mdof_duffing, mdof_jenkins]

###############################################################################
###### Basic Consistency + Correctness Checks                            ######
###############################################################################

######### With Zeroth Harmonic

print('\nChecking all systems with zeroth harmonic included.')

h = np.array([0, 1, 2, 3, 4, 5, 6, 7])
# h = np.array([0, 1, 2, 3])

Nhc = hutils.Nhc(h)

for i in range(len(all_systems)):
    
    sys = all_systems[i]
    
    # loop over systems
    Ndof = sys.M.shape[0]
        
    rng = np.random.default_rng(seed=42)
    Uwxa = rng.random((Ndof*Nhc+3))-0.5
    Uwxa[-3] = 1.4
    Uwxa[-2] = 0.01
    Uwxa[-1] = 0.1
    
    Fl = np.zeros(Ndof*Nhc)
    Fl[Ndof:2*Ndof] = 1.0
    
    # R, dRdUwx, dRda = sys.epmc_res(Uwxa, Fl, h)
    
    # print('\nDisplacement Gradient:')
    fun = lambda Uwx : sys.epmc_res(np.hstack((Uwx, Uwxa[-1])), Fl, h, Nt=128, aft_tol=1e-7)[0:2]
    vutils.check_grad(fun, Uwxa[:-1], verbose=False, atol=2e-8)
    
    
    # print('Amplitude Gradient:')
    fun = lambda a : sys.epmc_res(np.hstack((Uwxa[:-1], a)), Fl, h, Nt=128, aft_tol=1e-7)[0:3:2]
    vutils.check_grad(fun, np.atleast_1d(Uwxa[-1]), verbose=False, atol=2e-8)


print('Finished gradient check with zeroth harmonic.')

######### Without Zeroth Harmonic

print('\nChecking all systems without zeroth harmonic included.')

h = np.array([1, 2, 3, 4, 5, 6, 7])
# h = np.array([1, 2, 3])

Nhc = hutils.Nhc(h)

for i in range(len(all_systems)):
    
    sys = all_systems[i]
    
    # loop over systems
    Ndof = sys.M.shape[0]
        
    rng = np.random.default_rng(seed=42)
    Uwxa = rng.random((Ndof*Nhc+3))-0.5
    Uwxa[-3] = 1.4
    Uwxa[-2] = 0.01
    Uwxa[-1] = 0.1
    
    Fl = np.zeros(Ndof*Nhc)
    Fl[Ndof:2*Ndof] = 1.0
    
    # R, dRdUwx, dRda = sys.epmc_res(Uwxa, Fl, h)
    
    # print('\nDisplacement Gradient:')
    fun = lambda Uwx : sys.epmc_res(np.hstack((Uwx, Uwxa[-1])), Fl, h, Nt=128, aft_tol=1e-7)[0:2]
    vutils.check_grad(fun, Uwxa[:-1], verbose=False, atol=2e-8)
    
    
    # print('Amplitude Gradient:')
    fun = lambda a : sys.epmc_res(np.hstack((Uwxa[:-1], a)), Fl, h, Nt=128, aft_tol=1e-7)[0:3:2]
    vutils.check_grad(fun, np.atleast_1d(Uwxa[-1]), verbose=False, atol=2e-8)


print('Finished gradient check without zeroth harmonic.')

###############################################################################
###### Static Force Correctness                                          ######
###############################################################################

print('\nStarting zeroth harmonic residual checking: ')

h = np.array([0, 1, 2, 3, 4, 5, 6, 7])

Nhc = hutils.Nhc(h)

Ndof = 1
Fl = np.zeros(Ndof*Nhc)

Fl[0] = 2.0 # Static Force
Fl[Ndof:2*Ndof] = 1 # Phase Constraint

rng = np.random.default_rng(seed=42)
Uwxa = np.zeros(Ndof*Nhc+3)
Uwxa[1:2] = 5
Uwxa[-3] = 1.4
Uwxa[-2] = 0.01
Uwxa[-1] = 0.1

# Exact static displacement expected (Jenkins)
Uwxa[0] = Fl[0] / k

R, dRdUwx, dRda = sdof_jenkins.epmc_res(Uwxa, Fl, h)

if not R[0] == 0.0:
    print('Incorrect static displacement / zeroth harmonic residual.')

#### 2 DOF Check

h = np.array([0, 1, 2, 3, 4, 5, 6, 7])

Nhc = hutils.Nhc(h)

Ndof = 2
Fl = np.zeros(Ndof*Nhc)

Fl[0] = 2.0 # Static Force
Fl[1] = 1.5 # Static Force
Fl[Ndof:2*Ndof] = 1 # Phase Constraint

Uwxa = np.zeros(Ndof*Nhc+3)
Uwxa[Ndof*1:2*Ndof] = 5.0
Uwxa[Ndof*5:6*Ndof] = 0.1
Uwxa[-3] = 1.4
Uwxa[-2] = 0.01
Uwxa[-1] = 0.1

# Exact static displacement expected (Jenkins)
Uwxa[0:Ndof] = np.linalg.solve(mdof_jenkins.K, Fl[0:Ndof])

# # Can use this line to verify that the test really breaks when put in the wrong
# # answer
# Uwxa[0] = Uwxa[0] + 1e-5

R, dRdUwx, dRda = mdof_jenkins.epmc_res(Uwxa, Fl, h)

if np.abs(R[0] + R[1]) > 1e-14:
    print('Incorrect static displacement / zeroth harmonic residual - 2 DOF.')


print('Finished zeroth harmonic residual checking.')


###############################################################################
###### Linear System EPMC Verification                                   ######
###############################################################################

print('\nStarting Check on Linear System')

a0 = 0.05
a1 = 5

h = np.array([0,1,2,3,4,5])
Ndof = 2

Uwxa_full,lin_mode = epmc_cont(mdof_linear, a0, a1, h)

# Verify Linear Solution solves the problem
error = np.copy(Uwxa_full[:, :-1])
error[:, Ndof:2*Ndof] -= lin_mode
error[:, -2] -= 1.0 # Linear Frequency
error[:, -1] -= mdof_linear.ab[0] # Mass prop damping

max_err = np.max(np.abs(error))

if max_err > 1e-15:
    print('EPMC Failed on Linear System, error of {:.3e}'.format(max_err))

print('Finished Check on Linear System')

###############################################################################
###### EPMC for Nonlinear Systems v. Some Analytical Sanity Checks       ######
###############################################################################

print('\nStarting SDOF frequency checks.')

###### Duffing sanity check - sdof_duffing

h = np.array([0, 1, 2, 3])
Nhc = hutils.Nhc(h)
h0 = 1*(h[0] == 0)

vib_sys = sdof_duffing

a0 = -2
a1 = 2

Uwxa_full,lin_mode = epmc_cont(sdof_duffing, a0, a1, h)

# 1 and 3 Harmonic approximation of duffing oscillator
amp = 10.0 ** Uwxa_full[:, -1]
analytical_w = np.sqrt((sdof_duffing.K \
                        + 0.75*sdof_duffing.nonlinear_forces[0].kalpha*amp**2)\
                       /sdof_duffing.M)[0]
    
analytical_w - Uwxa_full[:, -3]

error = (analytical_w - Uwxa_full[:, -3]) / analytical_w

accept_error = np.max(np.abs(error[:20])) < 1e-4 and np.max(np.abs(error)) < 3e-2

if not accept_error:
    print('\nDuffing Oscillator has unexpected errors in frequency.\n')


###### Jenkins sanity check - sdof_jenkins

h = np.array([0, 1, 2, 3])
Nhc = hutils.Nhc(h)
h0 = 1*(h[0] == 0)

vib_sys = sdof_duffing

a0 = -2
a1 = 5

Uwxa_full,lin_mode = epmc_cont(sdof_jenkins, a0, a1, h)

low_amp_freq = np.sqrt((sdof_jenkins.K + sdof_jenkins.nonlinear_forces[0].kt)\
                       /sdof_jenkins.M)[0,0]

high_amp_freq = np.sqrt(sdof_jenkins.K/sdof_jenkins.M)[0,0]


accept_error = np.abs(Uwxa_full[0, -3] - low_amp_freq) < 1e-12 \
                and np.abs(Uwxa_full[-1, -3] - high_amp_freq) < 1e-7

if not accept_error:
    print('\nJenkins Oscillator has unexpected errors in frequency.\n')


print('\nFinished SDOF frequency checks.')
