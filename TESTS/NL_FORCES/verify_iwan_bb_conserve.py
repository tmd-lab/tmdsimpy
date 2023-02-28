"""
Verification of the AFT implementation(s).
Currently:
    -Instantaneous Force w/ Softening Nonlinearity Based on Iwan Model
    
    
failed_flag = False, changes to true if a test fails at any point 

"""
# TODO:
# 1. Do Some analytical checking of the AFT in fully slipped / fully stuck regimes
# 2. Plot the backbone to verify.
# 3. Verify Derivatives in both stuck and slipped regimes

import sys
import numpy as np

sys.path.append('../')
import verification_utils as vutils

sys.path.append('../../ROUTINES')
import harmonic_utils as hutils

sys.path.append('../../ROUTINES/NL_FORCES')
from iwan_bb_conserve import ConservativeIwanBB

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
import matplotlib.pyplot as plt

"""
System (all cubic springs, fixed boundaries):

    /|        + ----+        + ----+        |\
    /|---k1---| M1  |---k2---| M2  |---k3---|\
    /|        +-----+        +-----+        |\

"""

###############################################################################
###### Test Parameters                                                   ######
###############################################################################

failed_flag = False

analytical_tol_stuck = 1e-17 # Tolerance against analytical stuck
analytical_tol_slip = 1e-9 # Fully slipped state tolerance

rtol_grad = 1e-7 # Relative gradient tolerance

high_amp_grad_rtol = 3e-5 # Relative tolerance for a specific case


###############################################################################
###### System                                                            ######
###############################################################################

# Simple Mapping to spring displacements
Q = np.array([[1.0, 0], \
              [0, 1.0]])

# Weighted / integrated mapping back for testing purposes
T = np.array([[1.0, 0.0], \
              [0.0, 1.0]])

kt = 2.0
Fs = 3.0
chi = 0.0
beta = 0.0

softening_force = ConservativeIwanBB(Q, T, kt, Fs, chi, beta)

###############################################################################
######## Plot Nonlinear Stiffness to verify it makes sense             ########
###############################################################################

umax = 4*Fs/kt

uplot = np.linspace(-umax, umax, 1001)

fnlplot = softening_force.local_force_history(uplot, 0*uplot)[0]

plt.plot(uplot, fnlplot/Fs)
plt.xlabel('Displacement')
plt.ylabel('Force/Fs')
plt.show()

###############################################################################
######## Verify Limits of Fourier Coefficients                         ########
###############################################################################


########
# Stuck Regime
########

softening_force.kt = 2 
softening_force.Fs = 1e16

h = np.array([0, 1, 2, 3, 4, 5, 6, 7]) 
Nhc = hutils.Nhc(h)
Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components

Nd = Q.shape[1]

U = np.zeros((Nd*Nhc, 1))

# First DOF, Cosine Term, Fundamental
U[Nd+0, 0] = 1e-2

# Second DOF, Sine Term, Fundamental
U[2*Nd+1, 0] = 1e-2

w = 1 # Test for various w

FnlH = softening_force.aft(U, w, h, Nt=1<<17)[0]

FnlH_analytical = np.zeros_like(FnlH)

# Cosine Term
FnlH_analytical[Nd+0]    = softening_force.kt*U[Nd+0]

# Sine Term
FnlH_analytical[2*Nd+1]  = softening_force.kt*U[2*Nd+1] # 1st

error =  np.linalg.norm(FnlH - FnlH_analytical)
failed_flag = failed_flag or error > analytical_tol_stuck

print('Force Difference Between numerical and analytical (stuck):', error)

# np.vstack((FnlH, FnlH_analytical)).T

########
# Square Wave Limit
########

softening_force.kt = 1e16 
softening_force.Fs = 0.1

h = np.array([0, 1, 2, 3, 4, 5, 6, 7]) 
Nhc = hutils.Nhc(h)
Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components

Nd = Q.shape[1]

U = np.zeros((Nd*Nhc, 1))

# First DOF, Cosine Term, Fundamental
U[Nd+0, 0] = 1e16

# Second DOF, Sine Term, Fundamental
U[2*Nd+1, 0] = 1e16

w = 1 # Test for various w

FnlH = softening_force.aft(U, w, h, Nt=1<<17)[0]

FnlH_analytical = np.zeros_like(FnlH)

# Cosine Term
FnlH_analytical[Nd+0]    = 4*softening_force.Fs/np.pi # 1st
FnlH_analytical[5*Nd+0]  = -4/3*softening_force.Fs/np.pi # 3rd
FnlH_analytical[9*Nd+0]  = 4/5*softening_force.Fs/np.pi # 5th 
FnlH_analytical[13*Nd+0] = -4/7*softening_force.Fs/np.pi # 7th 

# Sine Term
FnlH_analytical[2*Nd+1]  = 4*softening_force.Fs/np.pi # 1st
FnlH_analytical[6*Nd+1]  = 4/3*softening_force.Fs/np.pi # 3rd
FnlH_analytical[10*Nd+1] = 4/5*softening_force.Fs/np.pi # 5th 
FnlH_analytical[14*Nd+1] = 4/7*softening_force.Fs/np.pi # 7th 


error =  np.linalg.norm(FnlH - FnlH_analytical)
failed_flag = failed_flag or error > analytical_tol_slip

print('Force Difference Between numerical and analytical (slipped):', error)

# np.vstack((FnlH, FnlH_analytical)).T


###############################################################################
######## Check Derivatives (if multiple regimes)                       ########
###############################################################################


# Simple Mapping to spring displacements
Q = np.array([[1.0, 0], \
              [-1.0, 1.0], \
              [0, 1.0]])

# Weighted / integrated mapping back for testing purposes
T = np.array([[1.0, 0.25, 0.0], \
              [0.0, 0.25, 1.0]])

kt = 2.0
Fs = 3.0
chi = 0.0
beta = 0.0

softening_force = ConservativeIwanBB(Q, T, kt, Fs, chi, beta)


"""
Case 1: 
    -Fix Center DOF
    -Only move the first Harmonic
"""
# h = np.array([0, 1, 2, 3]) # Manual Checking expansion / debugging
h = np.array([0, 1, 2, 3, 4, 5, 6, 7]) # Automate Checking with this
Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components

Nd = Q.shape[1]

U = np.zeros((Nd*Nhc, 1))

# First DOF, Cosine Term, Fundamental
U[Nd+0, 0] = 4

# Second DOF, Sine Term, Fundamental
U[2*Nd+1, 0] = 3

w = 1 # Test for various w



#######################
# First Derivative Check

print('\nTesting Simple First Harmonic Motion:')

# np.hstack((Fnl, Fnl_analytical)).round(3)

print('Mid Amplitude:')

# Numerically Verify Gradient
fun = lambda U: softening_force.aft(U, w, h)[0:2]
grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=rtol_grad)
failed_flag = failed_flag or grad_failed

# Numerically Verify Frequency Gradient
fun = lambda w: softening_force.aft(U, w[0], h)[0::2]
grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, rtol=rtol_grad)
failed_flag = failed_flag or grad_failed


print('High Amplitude:')

softening_force.Fs = 1e-2*Fs

# Numerically Verify Gradient
fun = lambda U: softening_force.aft(U, w, h)[0:2]
grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=high_amp_grad_rtol)
failed_flag = failed_flag or grad_failed

# Numerically Verify Frequency Gradient
fun = lambda w: softening_force.aft(U, w[0], h)[0::2]
grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, rtol=rtol_grad)
failed_flag = failed_flag or grad_failed


softening_force.Fs = Fs


print('Low Amplitude:')

softening_force.Fs = 1e5*Fs

# Numerically Verify Gradient
fun = lambda U: softening_force.aft(U, w, h)[0:2]
grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=rtol_grad)
failed_flag = failed_flag or grad_failed

# Numerically Verify Frequency Gradient
fun = lambda w: softening_force.aft(U, w[0], h)[0::2]
grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, rtol=rtol_grad)
failed_flag = failed_flag or grad_failed

softening_force.Fs = Fs

#######################
# Verify Zeroth Harmonic at a constant (both DOFs, all non-zero kalpha)

print('Test with Zeroth Harmonic, skipping 4th:')

# np.random.seed(42)
np.random.seed(1023)
# np.random.seed(0)

h = np.array([0, 1, 2, 3, 5, 6, 7]) # Automate Checking with this

kt = 2.0
Fs = 3.0
chi = 0.0
beta = 0

# Weighted / integrated mapping back for testing purposes
T = np.array([[1.0, 0.25, 0.0], \
              [0.0, 0.25, 0.5]])

Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
Nd = Q.shape[1]

U = np.random.rand(Nd*Nhc, 1)


softening_force = ConservativeIwanBB(Q, T, kt, Fs, chi, beta)

fun = lambda U: softening_force.aft(U, w, h)[0:2]
grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=rtol_grad)
failed_flag = failed_flag or grad_failed


# Numerically Verify Frequency Gradient
fun = lambda w: softening_force.aft(U, w[0], h)[0::2]
grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, rtol=rtol_grad)
failed_flag = failed_flag or grad_failed


######################
# Test without zeroth harmonic
print('Test without Zeroth Harmonic, skipping 4th:')
h = np.array([1, 2, 3, 5, 6, 7]) # Automate Checking with this

Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
Nd = Q.shape[1]

U = np.random.rand(Nd*Nhc, 1)

fun = lambda U: softening_force.aft(U, w, h)[0:2]
grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=rtol_grad)
failed_flag = failed_flag or grad_failed

# Numerically Verify Frequency Gradient
fun = lambda w: softening_force.aft(U, w[0], h)[0::2]
grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, rtol=rtol_grad)
failed_flag = failed_flag or grad_failed

###############################################################################
#### Test Result                                                           ####
###############################################################################

if failed_flag:
    print('\n\nTest FAILED, investigate results further!\n')
else:
    print('\n\nTest passed.\n')