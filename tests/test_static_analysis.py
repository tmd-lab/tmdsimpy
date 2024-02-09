"""
Test Static Analysis Routines. 

System: 
    1. Consider a 2 DOF system with a Jenkins Slider
    
Tests Functions: 
    ===Test 1=================================
    1. Init History function to all zeros
    2. Static Analysis with Jenkins element having non-zero Fs and non-Zero 
        Fexternal
    3. Update History
    4. Look at solution for Jenkins with no applied external load. 
    5. Call init history again
    6. Verify that Jenkins with no load returns to zero displacement
    ===Test 2=====================================================
    1. Call Init again
    2. Set prestress mu to zero with vib_sys function
    3. Check new prestress solution
    4. Set mu back to non-zero
    5. Check new prestress solution
    
At each step of checking a solution, also verify that the gradients returned are
correct.
"""

"""
Related work to complete for full updates:
    1. Elastic Dry Friction Force function w/ tests (tests in the force test file)
"""


import sys
import numpy as np
import unittest

import verification_utils as vutils

sys.path.append('..')
from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.nlforces.vector_jenkins import VectorJenkins



class TestStaticAnalysis(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Create nonlinear systems in addition to the normal unittest 
        initialization

        Returns
        -------
        None.

        """
        super(TestStaticAnalysis, self).__init__(*args, **kwargs)
        
        M = np.eye(2)
        K = np.array([[4, -1], [-1, 1]])
        
        vib_sys = VibrationSystem(M, K, ab=[0.01, 0.0])
        
        
        Q = np.array([[1.0, -1.0]])
        T = np.array([[1.0], [-1.0]])
        kt = 2.0
        Fs = 1.0
        
        nl_force = VectorJenkins(Q, T, kt, Fs)
        
        vib_sys.add_nl_force(nl_force)
        
        self.vib_sys = vib_sys
        
        self.rtol = 1e-11
        self.atol = 1e-10
        
    def test_basic_static(self):
        """
        Test static residual for vibration system with Jenkins element

        ===Test 1=================================
        1. Init History function to all zeros
        2. Static Analysis with Jenkins element having non-zero Fs and non-Zero 
            Fexternal
        3. Update History
        4. Look at solution for Jenkins with no applied external load. 
        5. Call init history again
        6. Verify that Jenkins with no load returns to zero displacement

        """
        
        vib_sys = self.vib_sys
        
        vib_sys.init_force_history()
        
        Fext = np.array([0.0, 5.0])
        
        ##################
        # Known Residual Value
        
        U0 = np.zeros(2)
        
        R, dRdU = vib_sys.static_res(U0, Fext)
        
        self.assertLess(np.linalg.norm(Fext + R), 1e-12, 
                        'Residual should be external force for zero displacement.')
        
        fun = lambda U : vib_sys.static_res(U, Fext)
            
        grad_failed = vutils.check_grad(fun, U0, verbose=False, rtol=self.rtol, atol=self.atol)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient')
        
        ##################
        # Solution
        
        Usol = np.array([5.0/3.0, 5.0/3.0+4.0])
        
        
        R, dRdU = vib_sys.static_res(Usol, Fext)
        self.assertLess(np.linalg.norm(R), 1e-12, 
                        'Residual should be zero for this solution.')
        
        fun = lambda U : vib_sys.static_res(U, Fext)
            
        grad_failed = vutils.check_grad(fun, Usol, verbose=False, rtol=self.rtol, atol=self.atol)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient')
        
        ###############
        # Update History
        
        vib_sys.update_force_history(Usol)
        
        Fs = vib_sys.nonlinear_forces[0].Fs
        kt = vib_sys.nonlinear_forces[0].kt
        
        # This sets the Jenkins element to it's anchor position
        Usol2 = np.array([0.0, Usol[1] - Usol[0] - Fs/kt]) 
        
        # This cancels the linear forces from the linear springs at that point
        Fext2  = np.array([-3.5, 3.5])
        
        R, dRdU = vib_sys.static_res(Usol2, Fext2)
        
        # import pdb; pdb.set_trace()
        
        self.assertLess(np.linalg.norm(R), 1e-12, 
                        'Residual should be zero for this solution.')
        
        fun = lambda U : vib_sys.static_res(U, Fext2)
            
        grad_failed = vutils.check_grad(fun, Usol2, verbose=False, rtol=self.rtol, atol=self.atol)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient')
        
        ###############
        # Reinitialize History
        
        vib_sys.init_force_history()
        
        R, dRdU = vib_sys.static_res(0.0*Usol2, 0.0*Fext2)
                
        self.assertLess(np.linalg.norm(R), 1e-12, 
                        'Residual should be zero for this solution.')
        
        fun = lambda U : vib_sys.static_res(U, 0.0*Fext2)
            
        grad_failed = vutils.check_grad(fun, 0.0*Usol2, verbose=False, rtol=self.rtol, atol=self.atol)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient')
        
        return
        
    def test_mu0_static(self):
        """
        Test static with friction set to zero.
            
        ===Test 2=====================================================
        1. Call Init again
        2. Set prestress mu to zero with vib_sys function
        3. Check new prestress solution
        4. Set mu back to non-zero
        5. Check new prestress solution
        """
        
        
        ###############
        # Reinitialize History / set prestress mu
        vib_sys = self.vib_sys
        
        vib_sys.init_force_history()
        
        vib_sys.set_prestress_mu()
        
        ###############
        # Solution with no friction response
        
        Fext = np.array([0.0, 5.0])
        
        Usol = np.array([5.0/3.0, 5.0/3.0+5.0])
        
        R, dRdU = vib_sys.static_res(Usol, Fext)
        self.assertLess(np.linalg.norm(R), 1e-12, 
                        'Wrong solution with friction turned off.')
        
        
        fun = lambda U : vib_sys.static_res(U, Fext)
            
        grad_failed = vutils.check_grad(fun, Usol, verbose=False, rtol=self.rtol, atol=self.atol)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient')
        
        ###############
        # Solution with with friction turned back on
        
        vib_sys.reset_real_mu()
        
        Fext = np.array([0.0, 5.0])
        
        Usol = np.array([5.0/3.0, 5.0/3.0+4.0])
        
        R, dRdU = vib_sys.static_res(Usol, Fext)
        self.assertLess(np.linalg.norm(R), 1e-12, 
                        'Wrong solution with friction turned back on.')
        
        fun = lambda U : vib_sys.static_res(U, Fext)
            
        grad_failed = vutils.check_grad(fun, Usol, verbose=False, rtol=self.rtol, atol=self.atol)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient')
        
        return

if __name__ == '__main__':
    unittest.main()

