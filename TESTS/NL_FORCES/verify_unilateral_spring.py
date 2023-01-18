"""
Verification of the AFT implementation(s).
Currently:
    -Instantaneous Force w/ Unilateral Spring Contact
    

failed_flag = False, changes to true if a test fails at any point 
"""

import sys
import numpy as np

sys.path.append('../')
sys.path.append('../../')
import verification_utils as vutils
import harmonic_utils as hutils

sys.path.append('../../ROUTINES/')
sys.path.append('../../ROUTINES/NL_FORCES')
from unilateral_spring import UnilateralSpring

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
import matplotlib.pyplot as plt


###############################################################################
##### Test Parameters                                                     #####
###############################################################################

failed_flag = False

rtol_grad = 1e-9
atol_grad = 1e-9

###############################################################################
##### Numerical Implementation provides desired force displacement response ###
###############################################################################


# Simple Mapping to spring displacements
Q = np.array([[1.0]])
T = np.array([[1.0]])

# Cases are: 
#   1. Simple unilateral spring
#   2. Offset by delta unilateral spring
#   3. Normal preload, not offset
#   4. Normal preload and offset
#   5. Offset, no preload, impact.

kuni  = np.array([1.0, 1.0, 1.0, 1.0, 3.0])
Npre  = np.array([0.0, 0.0, 2.0, 2.0,  0.0])
delta = np.array([0.0, 0.3, 0.0, 0.3,  0.3])

uni_springs = 5*[None]

for i in range(len(uni_springs)):
    uni_springs[i] = UnilateralSpring(Q, T, kuni[i], Npreload=Npre[i], delta=delta[i])
    
umax = 3*delta.max()

uplot = np.linspace(-umax, umax, 1000)

for i in range(len(uni_springs)):
    legend_name = 'k={:.2f}, Np={:.2f}, delta={:.2f}'.format(kuni[i], Npre[i], delta[i])
    
    fplot = uni_springs[i].local_force_history(uplot, np.zeros_like(uplot))[0]

    plt.plot(uplot, fplot, label=legend_name)
    
plt.xlabel('Displacement')
plt.ylabel('Force')
plt.legend()
plt.title('Local Force Function')
plt.show()



###############################################################################
##### Test Derivative Accuracy                                            #####
###############################################################################

Nd = 1

h = np.array([0, 1, 2, 3, 4, 5, 6, 7]) 
w = 1.0

Nhc = hutils.Nhc(h)

U = np.zeros((Nd*Nhc, 1))

# np.random.seed(42)
np.random.seed(1023)
# np.random.seed(0)

# Test several different values of U on different length scales for each spring type
U = np.random.rand(Nd*Nhc, 10)

U = U*np.array([[0.1, 0.5, 1.0, 2.0, 3.0, 10.0, 20.0, 50.0, 100.0, 0.01]])


for i in range(len(uni_springs)):
    print('Testing unilateral spring number {:}'.format(i))
    
    for j in range(U.shape[1]):
        
        fun = lambda U: uni_springs[i].aft(U[:, j], w, h)
        grad_failed = vutils.check_grad(fun, U, verbose=False, atol=atol_grad, rtol=rtol_grad)
        
        failed_flag = failed_flag or grad_failed
        
        
###############################################################################
##### Test Results                                                        #####
###############################################################################

if failed_flag:
    print('\n\nTest FAILED, investigate results further!\n')
else:
    print('\n\nTest passed. Manually check figure for correctness.\n')