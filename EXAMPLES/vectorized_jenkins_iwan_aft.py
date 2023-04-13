"""
This script demonstrates the speedup of the vectorized Jenkins and Iwan models

The vectorized models allow for parallel computation of different times in AFT.

These algorithms, still give exactly the same results as the normal 
implementations. Comparisons of the accuracy to the normal implementation
is included in tests under 'TESTS/NL_FORCES'

This script also shows the convergence for Jenkins w.r.t. number of time steps
in the AFT procedure for a case of approaching a square wave due to lots of 
slip

"""

import sys
import numpy as np
import time
import matplotlib.pyplot as plt

# Python Utilities
sys.path.append('../ROUTINES/')
sys.path.append('../ROUTINES/NL_FORCES')
import harmonic_utils as hutils

from jenkins_element import JenkinsForce
from vector_jenkins import VectorJenkins

from iwan4_element import Iwan4Force 
from vector_iwan4 import VectorIwan4



###############################################################################
###  Jenkins Speed Tests                                                    ###
###############################################################################

print('\n\nVectorized v. Serial Jenkins Implementation:')

# Simple Mapping to displacements
Q = np.array([[1.0]])
T = np.array([[1.0]])

kt = 2.0
Fs = 3.0
umax = 5.0
freq = 1.4 # rad/s

vector_jenkins_force = VectorJenkins(Q, T, kt, Fs)
jenkins_force = JenkinsForce(Q, T, kt, Fs)

# vector_jenkins_force.init_history(unlth0=0)

Nt = 1 << 10

w = 1.7

h = np.array([0, 1, 2, 3])
Unl = 5*np.array([[0.75, 2.0, 1.3, 0.0, 0.0, 1.0, 0.0]]).T
unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl

num_times = 1000

start_time =  time.perf_counter()
for i in range(num_times):
    Fnl, dFnldU = vector_jenkins_force.aft(Unl, w, h, Nt=Nt)[0:2]
end_time =  time.perf_counter()
vec_time = (end_time - start_time)/num_times
print('Vector Time: {:.4e} sec'.format(vec_time))

start_time =  time.perf_counter()
for i in range(num_times):
    Fnl, dFnldU = jenkins_force.aft(Unl, w, h, Nt=Nt)[0:2]
end_time =  time.perf_counter()
serial_time = (end_time - start_time)/num_times

print('Averaged over {:} runs.'.format(num_times))
print('Serial Time: {:.4e} sec'.format(serial_time))
print('Speedup: {:.4f}'.format(serial_time/vec_time))

###############################################################################
###     AFT Convergence for Jenkins                                         ###
###############################################################################


h = np.array([0, 1, 2, 3])
Unl = np.array([[0, 10000000000.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
Fs = 1
kt = 100000000

vector_jenkins_force = VectorJenkins(Q, T, kt, Fs)

Nhc = hutils.Nhc(h)

Nt_test = np.arange(7,21,1)
Fnl_test = np.zeros((Nt_test.shape[0], Nhc))

for i in range(Nt_test.shape[0]):
    Fnl_test[i, :] = vector_jenkins_force.aft(Unl, w, h, Nt=1<<Nt_test[i])[0]

########### First Harmonic

plt.plot(Nt_test, Fnl_test[:, 1]/vector_jenkins_force.Fs, label='Harmonic 1, cosine')
# plt.plot(Nt_test, Fnl_test[:, 2]/vector_jenkins_force.Fs, label='Harmonic 1, sine')

plt.ylabel('Harmonic Force Coefficient/Fs')
plt.xlabel('log2(Nt)')
plt.yscale('log')  
# plt.ylim((0, 3.5))
plt.legend()
# plt.savefig('./Figs/.png', format='png', dpi=300)
plt.show()

########### Third Harmonic

plt.plot(Nt_test, Fnl_test[:, 5]/vector_jenkins_force.Fs, label='Harmonic 3, cosine')
# plt.plot(Nt_test, Fnl_test[:, 6]/vector_jenkins_force.Fs, label='Harmonic 3, sine')

plt.ylabel('Harmonic Force Coefficient/Fs')
plt.xlabel('log2(Nt)')
plt.yscale('log')  
# plt.ylim((0, 3.5))
plt.legend()
# plt.savefig('./Figs/.png', format='png', dpi=300)
plt.show()


###############################################################################
###  Iwan Speed Tests                                                       ###
###############################################################################


# Simple Mapping to spring displacements
Q = np.array([[1.0]])
T = np.array([[1.0]])

kt = 2.0
Fs = 3.0
chi = -0.1
beta = 0.1

Nsliders = 100

iwan_force   = Iwan4Force(Q, T, kt, Fs, chi, beta, Nsliders=Nsliders, alphasliders=1.0)
vector_force = VectorIwan4(Q, T, kt, Fs, chi, beta, Nsliders=Nsliders, alphasliders=1.0)

phimax = iwan_force.phimax

##########################################
###### Compare Time Series                                   

print('\n\nVectorized v. Serial Iwan Implementation:')

Nt = 1 << 10

w = 1.7

h = np.array([0, 1, 2, 3])
Unl = 5*phimax*np.array([[0.75, 2.0, 1.3, 0.0, 0.0, 1.0, 0.0]]).T


num_times = 200

start_time =  time.perf_counter()
for i in range(num_times):
    Fnl, dFnldU = vector_force.aft(Unl, w, h, Nt=Nt)[0:2]

end_time =  time.perf_counter()
vec_time = (end_time - start_time)/num_times
print('Vector Time: {:.4e} sec'.format(vec_time))

start_time =  time.perf_counter()
for i in range(num_times):
    Fnl, dFnldU = iwan_force.aft(Unl, w, h, Nt=Nt)[0:2]

end_time =  time.perf_counter()
serial_time = (end_time - start_time)/num_times

print('Nt={:}, averaged over {:} runs.'.format(Nt, num_times))
print('Serial Time: {:.4e} sec'.format(serial_time))
print('Speedup: {:.4f}'.format(serial_time/vec_time))
    
