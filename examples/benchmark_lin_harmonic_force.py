"""
Comparison between two different algorithms for 
tmdsimpy.harmonic_utils.harmonic_stiffness.
The original implementation using numpy.kron to create dense matrices and 
sum components without any loops or if statements. However, this may scale
poorly to large matrices and many harmonics. Alternative avoids adding as many
zeros.

Original implementation utilizing an expensive friction model, around 850 DOFs 
times 7 harmonic components (0, 1c, 1s, 2c, 2s, 3c, 3s) suggested that 14%
of total continuation computation time may be spent in this routine. 

"""


import sys
import numpy as np
import timeit

sys.path.append('..')
import tmdsimpy.harmonic_utils as hutils


h_max = 3
w = 1.732 # Deliberately not a nice number



h = np.array(range(h_max+1))


Ndof = 850 # consider 1, 10, 850
num_time = 10

np.random.seed(1023)

M = np.random.rand(Ndof, Ndof)
C = np.random.rand(Ndof, Ndof)
K = np.random.rand(Ndof, Ndof)


time_small = timeit.timeit(lambda : hutils.harmonic_stiffness(M, C, K, w, h),
                           number=num_time)


time_large = timeit.timeit(lambda : \
                             hutils.harmonic_stiffness_many_dof(M, C, K, w, h),
                             number=num_time)

print(('Ndof={: 4d}, baseline (for small) : {: 6.4e} s,'\
      +' alternative (for large) {: 6.4e}').format(Ndof, time_small, time_large))

reference_res = hutils.harmonic_stiffness(M, C, K, w, h)
new_res = hutils.harmonic_stiffness(M, C, K, w, h)

print('(E-E)/norm(E): {: 6.4e}, (dEdw-dEdw)/norm(dEdw): {: 6.4e}'.format(
    np.linalg.norm(reference_res[0] - new_res[0])/np.linalg.norm(new_res[0]),
    np.linalg.norm(reference_res[1] - new_res[1])/np.linalg.norm(new_res[1])))