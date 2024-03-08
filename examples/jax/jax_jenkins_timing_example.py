"""
Example of using JAX with the Jenkins implementation that uses JAX and 
Just In Time Complication (JIT)
"""

# Standard imports
import numpy as np
import sys
import timeit

# Imports of Custom Functions and Classes
sys.path.append('../..')
import tmdsimpy.harmonic_utils as hutils

# JAX version w/o vectorization
from tmdsimpy.jax.nlforces.jenkins_element import JenkinsForce

# Compare to vectorized (non JAX version)
from tmdsimpy.nlforces.vector_jenkins import VectorJenkins

# Also compare JAX Vector Jenkins
from tmdsimpy.jax.nlforces.vector_jenkins import VectorJenkins as JV_Jenkins


###############################################################################
###  Create Jenkins Model                                                   ###
###############################################################################

print('\n\nRunning JAX Jenkins implementation')

# Simple Mapping to displacements
Q = np.array([[1.0]])
T = np.array([[1.0]])

kt = 2.0
Fs = 3.0
umax = 5.0
freq = 1.4 # rad/s

vector_jenkins_force = VectorJenkins(Q, T, kt, Fs)
jenkins_force = JenkinsForce(Q, T, kt, Fs, u0=None)

jax_vec_jenkins = JV_Jenkins(Q, T, kt, Fs, u0=None)


###############################################################################
###  Timing Comparisons                                                     ###
###############################################################################

# Times are all similar at Nt=1<<10
# JAX slows down a lot for Nt=1<<14, but shows clearer differences with vector algorithm
Nt = 1 << 10

w = 1.7

h = np.array([0, 1, 2, 3])
Unl = 5*np.array([[0.75, 2.0, 1.3, 0.0, 0.0, 1.0, 0.0]]).T
unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl

num_times = 100

print('Averaging over {:} runs with Nt={:}'.format(num_times, Nt))

vec_time = timeit.timeit(lambda : vector_jenkins_force.aft(Unl, w, h, Nt=Nt), \
                         number=num_times)


compile_time = timeit.timeit(lambda : jenkins_force.aft(Unl, w, h, Nt=Nt), \
                             number=1)
    
    
jax_time = timeit.timeit(lambda : jenkins_force.aft(Unl, w, h, Nt=Nt), \
                             number=num_times)

    
vec_compile_time = timeit.timeit(lambda : jax_vec_jenkins.aft(Unl, w, h, Nt=Nt), \
                             number=1)
    
vec_jax_time = timeit.timeit(lambda : jax_vec_jenkins.aft(Unl, w, h, Nt=Nt), \
                             number=num_times)

print('Vector Time: {:.4e} sec'.format(vec_time/num_times))
print('\nJIT Compile Time: {:.4e} sec'.format(compile_time))
print('JAX Time (after compile): {:.4e} sec'.format(jax_time/num_times))

print('\nJIT Compile Time (Vector): {:.4e} sec'.format(vec_compile_time))
print('Vector JAX Time (after compile): {:.4e} sec'.format(vec_jax_time/num_times))



