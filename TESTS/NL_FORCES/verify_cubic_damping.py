"""
Verification of the AFT implementation.
Currently:
    -Instantaneous Force w/ Cubic Damping Nonlinearity
    
    
failed_flag = False, changes to true if a test fails at any point 
"""

import sys
import numpy as np

# Python Utilities
sys.path.append('../')
sys.path.append('../../')
import verification_utils as vutils

sys.path.append('../../NL_FORCES')

from cubic_damping import CubicDamping


"""
System (all cubic dampers, fixed boundaries):
    Only checks AFT, so don't need the system to be physical by adding springs.
    /|        + ----+        + ----+        |\
    /|---c1---| M1  |---c2---| M2  |---c3---|\
    /|        +-----+        +-----+        |\

"""

#####################

failed_flag = False

analytical_sol_tol = 1e-12 # Tolerance comparing to analytical solution

rtol_grad = 1e-10 # Relative gradient tolerance


#####################
# Simple Mapping to spring displacements
Q = np.array([[1.0, 0], \
              [-1.0, 1.0], \
              [0, 1.0]])

# Weighted / integrated mapping back for testing purposes
T = np.array([[1.0, 0.25, 0.0], \
              [0.0, 0.25, 1.0]])

calpha = np.array([3, 0, 7])

nl_force = CubicDamping(Q, T, calpha)

"""
Case 1: 
    -Fix Center DOF
    -Only move the first Harmonic
    -Compare to analytical expansion of cos^3/sin^3
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

w = 1.732 # Test for various w, not 1.0 so velocity is clearly different from displacement.

print('Testing Simple First Harmonic Motion:')

Fnl, dFnldU = nl_force.aft(U, w, h)

# # Analytically Verify Force expansion:
# # X^3*cos^3(x) = X^3*( 3/4*cos(x) + 1/4*cos(3x) )
# # X^3*sin^3(x) = X^3*(3/4*sin(x) - 1/4*sin(3x)
# # With Velocity shift
# #     cos -> -sin
# #     sin -> cos
Fnl_analytical = np.zeros_like(Fnl) 
Fnl_analytical[2*Nd+0] = -T[0,0]*( 0.75*calpha[0]*(Q[0,0]*U[Nd+0]*w)**3 )
Fnl_analytical[6*Nd+0] = T[0,0]*( 0.25*calpha[0]*(Q[0,0]*U[Nd+0]*w)**3 )

Fnl_analytical[1*Nd+1] = T[1,2]*( 0.75*calpha[2]*(Q[2,1]*U[2*Nd+1]*w)**3 )
Fnl_analytical[5*Nd+1] = T[1,2]*( 0.25*calpha[2]*(Q[2,1]*U[2*Nd+1]*w)**3 )

analytical_error = np.linalg.norm(Fnl - Fnl_analytical)
failed_flag = failed_flag or analytical_error > analytical_sol_tol

print('Difference Between numerical and analytical:', analytical_error)
# np.hstack((Fnl, Fnl_analytical)).round(3)

# Numerically Verify Gradient
fun = lambda U: nl_force.aft(U, w, h)
grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=rtol_grad)
failed_flag = failed_flag or grad_failed


#######################
# Verify Zeroth Harmonic at a constant (both DOFs, all non-zero calpha)

print('Test with Zeroth Harmonic, skipping 4th:')

# np.random.seed(42)
np.random.seed(1023)
# np.random.seed(0)

h = np.array([0, 1, 2, 3, 5, 6, 7]) # Automate Checking with this
calpha = np.array([3, 5, 7])

# Weighted / integrated mapping back for testing purposes
T = np.array([[1.0, 0.25, 0.0], \
              [0.0, 0.25, 0.5]])

Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
Nd = Q.shape[1]

U = np.random.rand(Nd*Nhc, 1)


nl_force = CubicDamping(Q, T, calpha)

fun = lambda U: nl_force.aft(U, w, h)
grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=rtol_grad)
failed_flag = failed_flag or grad_failed



######################
# Test without zeroth harmonic
print('Test without Zeroth Harmonic, skipping 4th:')
h = np.array([1, 2, 3, 5, 6, 7]) # Automate Checking with this

Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
Nd = Q.shape[1]

U = np.random.rand(Nd*Nhc, 1)

fun = lambda U: nl_force.aft(U, w, h)
grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=rtol_grad)
failed_flag = failed_flag or grad_failed

#######################
# Test Result

if failed_flag:
    print('\n\nTest FAILED, investigate results further!\n')
else:
    print('\n\nTest passed.\n')