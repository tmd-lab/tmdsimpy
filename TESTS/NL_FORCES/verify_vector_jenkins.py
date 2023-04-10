"""
Verification of vectorized Jenkins algorithm

This script compares outputs to the non-vectorized Jenkins algorithm for 
verification. Thus gradients do not need to be numerically approximated here.

This script also does timing of the speedup between the algorithms. Default is 
to only average over 1 time so limit computational effort on a test.


failed_flag = False, changes to true if a test fails at any point 
"""

import sys
import numpy as np
import time
import matplotlib.pyplot as plt

sys.path.append('../.')
import verification_utils as vutils

# Python Utilities
sys.path.append('../../ROUTINES/')
sys.path.append('../../ROUTINES/NL_FORCES')
import harmonic_utils as hutils

from jenkins_element import JenkinsForce
from vector_jenkins import VectorJenkins


###############################################################################
###     Test Parameters                                                     ###
###############################################################################

failed_flag = False

force_tol = 5e-15 # All should be exactly equal
df_tol = 1e-14 # rounding error on the derivatives
dfdw_tol = 1e-16 # everything should have zero derivative w.r.t. frequency

###############################################################################
###     Testing Function                                                    ###
###############################################################################

# Useful functions for testing: 
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


###############################################################################
###     Modeling                                                            ###
###############################################################################

# Vectorized Jenkins Algorithm


# Simple Mapping to displacements
Q = np.array([[1.0]])
T = np.array([[1.0]])

kt = 2.0
Fs = 3.0
umax = 5.0
freq = 1.4 # rad/s
Nt = 1<<8

vector_jenkins_force = VectorJenkins(Q, T, kt, Fs)
jenkins_force = JenkinsForce(Q, T, kt, Fs)

vector_jenkins_force.init_history(unlth0=0)



Nt = 1 << 14

w = 1.7

h = np.array([0, 1, 2, 3])
Unl = 5*np.array([[0.75, 2.0, 1.3, 0.0, 0.0, 1.0, 0.0]]).T
unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl

num_times = 1

start_time =  time.perf_counter()
for i in range(num_times):
    fnl_vec, dfduh_vec = time_series_forces(Unl, h, Nt, w, vector_jenkins_force)
    # Fnl, dFnldU = vector_jenkins_force.aft(Unl, w, h, Nt=Nt)[0:2]
end_time =  time.perf_counter()
vec_time = (end_time - start_time)/num_times
print('Vector Time: {:.4e} sec'.format(vec_time))

start_time =  time.perf_counter()
for i in range(num_times):
    fnl, dfduh = time_series_forces(Unl, h, Nt, w, jenkins_force)
    # Fnl, dFnldU = jenkins_force.aft(Unl, w, h, Nt=Nt)[0:2]
end_time =  time.perf_counter()
serial_time = (end_time - start_time)/num_times

print('Averaged over {:} runs.'.format(num_times))
print('Serial Time: {:.4e} sec'.format(serial_time))
print('Speedup: {:.4f}'.format(serial_time/vec_time))


force_error = np.max(np.abs(fnl-fnl_vec))
df_error = np.max(np.abs(dfduh-dfduh_vec))
failed_flag = failed_flag or force_error > force_tol or df_error > df_tol

print('Force error: {:.4e} and derivative error: {:.4e}'.format(force_error, df_error))



###############################################################################
###     AFT Verification - Run more tests                                   ###
###############################################################################

Nt = 1 << 10

w = 1.7

##### Verification 1

h = np.array([0, 1, 2, 3])
Unl = 5*np.array([[0.75, 2.0, 1.3, 0.0, 0.0, 1.0, 0.0]]).T
unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl


fnl_vec, dfduh_vec = time_series_forces(Unl, h, Nt, w, vector_jenkins_force)
FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_jenkins_force.aft(Unl, w, h, Nt=Nt)

fnl, dfduh = time_series_forces(Unl, h, Nt, w, jenkins_force)
FnlH, dFnldUH, dFnldw = jenkins_force.aft(Unl, w, h, Nt=Nt)

print('\nVerification 1')

force_error = np.max(np.abs(fnl-fnl_vec))
df_error = np.max(np.abs(dfduh-dfduh_vec))
failed_flag = failed_flag or force_error > force_tol or df_error > df_tol


print('Force error: {:.4e} and derivative error: {:.4e}'.format(force_error, df_error))

FH_error = np.max(np.abs(FnlH_vec-FnlH_vec))
dFH_error = np.max(np.abs(dFnldUH-dFnldUH_vec))
failed_flag = failed_flag or FH_error > force_tol or dFH_error > df_tol

print('Harmonic Force error: {:.4e} and derivative error: {:.4e}'\
      .format(FH_error, dFH_error))

dfdw_error = np.max(np.abs(dFnldw - dFnldw_vec))
failed_flag = failed_flag or dfdw_error > dfdw_tol

print('')


##### Verification 2

h = np.array([0, 1, 2, 3])
Unl = 0.2*np.array([[0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.0]]).T
unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl


fnl_vec, dfduh_vec = time_series_forces(Unl, h, Nt, w, vector_jenkins_force)
FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_jenkins_force.aft(Unl, w, h, Nt=Nt)

fnl, dfduh = time_series_forces(Unl, h, Nt, w, jenkins_force)
FnlH, dFnldUH, dFnldw = jenkins_force.aft(Unl, w, h, Nt=Nt)

print('\nVerification 2')

force_error = np.max(np.abs(fnl-fnl_vec))
df_error = np.max(np.abs(dfduh-dfduh_vec))
failed_flag = failed_flag or force_error > force_tol or df_error > df_tol

print('Force error: {:.4e} and derivative error: {:.4e}'.format(force_error, df_error))

FH_error = np.max(np.abs(FnlH_vec-FnlH_vec))
dFH_error = np.max(np.abs(dFnldUH-dFnldUH_vec))
failed_flag = failed_flag or FH_error > force_tol or dFH_error > df_tol

print('Harmonic Force error: {:.4e} and derivative error: {:.4e}'\
      .format(FH_error, dFH_error))

dfdw_error = np.max(np.abs(dFnldw - dFnldw_vec))
failed_flag = failed_flag or dfdw_error > dfdw_tol

print('')


##### Verification 3

h = np.array([0, 1, 2, 3])
Unl = 10000000000.0*np.array([[0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.0]]).T
unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl


fnl_vec, dfduh_vec = time_series_forces(Unl, h, Nt, w, vector_jenkins_force)
FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_jenkins_force.aft(Unl, w, h, Nt=Nt)

fnl, dfduh = time_series_forces(Unl, h, Nt, w, jenkins_force)
FnlH, dFnldUH, dFnldw = jenkins_force.aft(Unl, w, h, Nt=Nt)

print('\nVerification 3')

force_error = np.max(np.abs(fnl-fnl_vec))
df_error = np.max(np.abs(dfduh-dfduh_vec))
failed_flag = failed_flag or force_error > force_tol or df_error > df_tol

print('Force error: {:.4e} and derivative error: {:.4e}'.format(force_error, df_error))

FH_error = np.max(np.abs(FnlH_vec-FnlH_vec))
dFH_error = np.max(np.abs(dFnldUH-dFnldUH_vec))
failed_flag = failed_flag or FH_error > force_tol or dFH_error > df_tol

print('Harmonic Force error: {:.4e} and derivative error: {:.4e}'\
      .format(FH_error, dFH_error))

dfdw_error = np.max(np.abs(dFnldw - dFnldw_vec))
failed_flag = failed_flag or dfdw_error > dfdw_tol

print('')


##### Verification 5

h = np.array([0, 1, 2, 3])
Unl = 5*np.array([[0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.0]]).T
unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl


fnl_vec, dfduh_vec = time_series_forces(Unl, h, Nt, w, vector_jenkins_force)
FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_jenkins_force.aft(Unl, w, h, Nt=Nt)

fnl, dfduh = time_series_forces(Unl, h, Nt, w, jenkins_force)
FnlH, dFnldUH, dFnldw = jenkins_force.aft(Unl, w, h, Nt=Nt)

print('\nVerification 4')

force_error = np.max(np.abs(fnl-fnl_vec))
df_error = np.max(np.abs(dfduh-dfduh_vec))
failed_flag = failed_flag or force_error > force_tol or df_error > df_tol

print('Force error: {:.4e} and derivative error: {:.4e}'.format(force_error, df_error))


FH_error = np.max(np.abs(FnlH_vec-FnlH_vec))
dFH_error = np.max(np.abs(dFnldUH-dFnldUH_vec))
failed_flag = failed_flag or FH_error > force_tol or dFH_error > df_tol

print('Harmonic Force error: {:.4e} and derivative error: {:.4e}'\
      .format(FH_error, dFH_error))

dfdw_error = np.max(np.abs(dFnldw - dFnldw_vec))
failed_flag = failed_flag or dfdw_error > dfdw_tol

print('')


##### Verification 6

h = np.array([0, 1, 2, 3])
Unl = 5*np.array([[4.29653115, 4.29165565, 2.8307871 , 4.17186848, 3.37441948,\
       0.80543152, 3.55638299]]).T
unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl


fnl_vec, dfduh_vec = time_series_forces(Unl, h, Nt, w, vector_jenkins_force)
FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_jenkins_force.aft(Unl, w, h, Nt=Nt)

fnl, dfduh = time_series_forces(Unl, h, Nt, w, jenkins_force)
FnlH, dFnldUH, dFnldw = jenkins_force.aft(Unl, w, h, Nt=Nt)

print('\nVerification 5')

force_error = np.max(np.abs(fnl-fnl_vec))
df_error = np.max(np.abs(dfduh-dfduh_vec))
failed_flag = failed_flag or force_error > force_tol or df_error > df_tol

print('Force error: {:.4e} and derivative error: {:.4e}'.format(force_error, df_error))


FH_error = np.max(np.abs(FnlH_vec-FnlH_vec))
dFH_error = np.max(np.abs(dFnldUH-dFnldUH_vec))
failed_flag = failed_flag or FH_error > force_tol or dFH_error > df_tol

print('Harmonic Force error: {:.4e} and derivative error: {:.4e}'\
      .format(FH_error, dFH_error))

dfdw_error = np.max(np.abs(dFnldw - dFnldw_vec))
failed_flag = failed_flag or dfdw_error > dfdw_tol

print('')

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
###     Test Results                                                        ###
###############################################################################

if failed_flag:
    print('\n\nTest FAILED, investigate results further!\n')
else:
    print('\n\nTest passed.\n')