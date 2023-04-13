"""
Test of accuracy for unilateral spring AFT implementation
    
"""

import sys
import numpy as np
import unittest

sys.path.append('../')
import verification_utils as vutils

sys.path.append('../../ROUTINES/')
sys.path.append('../../ROUTINES/NL_FORCES')
import harmonic_utils as hutils
from unilateral_spring import UnilateralSpring


class TestUniSpring(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Define tolerances here for all the tests

        Returns
        -------
        None.

        """
        super(TestUniSpring, self).__init__(*args, **kwargs)
        
        self.rtol_grad = 1e-9
        self.atol_grad = 1e-9

        ####################
        # Create the nonlinear forces to be tested
        
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
            

        self.uni_springs = uni_springs
        
        
        # Code for an example plotting what each setting looks like
        """
        import matplotlib as mpl
        mpl.rcParams['lines.linewidth'] = 2
        import matplotlib.pyplot as plt

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
        """

    def test_force_values(self):
        """
        Calculate force values and verify that expected values are produced

        Returns
        -------
        None.

        """

        uni_springs = self.uni_springs
        
        for i in range(len(uni_springs)):
            
            delta = uni_springs[i].delta
            Npre = uni_springs[i].Npreload
            kuni = uni_springs[i].k
            
            
            utest = np.array([-5.0, delta, delta+1.0])
            fexpect = np.array([-Npre, -Npre, -Npre+1.0*kuni])
                        
            ftest = uni_springs[i].local_force_history(utest, np.zeros_like(utest))[0]
            
            self.assertLess(np.linalg.norm(ftest-fexpect), 1e-16, 
                            'Unilateral spring case {} gives unexpected force values'.format(i))
            
            
            
    def test_derivative(self):
        """
        Test the derivatives from AFT for the unilateral spring

        Returns
        -------
        None.

        """
                
        uni_springs = self.uni_springs
        
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
            
            for j in range(U.shape[1]):
                
                fun = lambda U: uni_springs[i].aft(U[:, j], w, h)[0:2]
                grad_failed = vutils.check_grad(fun, U, verbose=False, 
                                                atol=self.atol_grad, 
                                                rtol=self.rtol_grad)
                
                self.assertFalse(grad_failed, 'Incorrect displacement gradient.')
                                
                # Numerically Verify Frequency Gradient
                fun = lambda w: uni_springs[i].aft(U[:, j], w[0], h)[0::2]
                grad_failed = vutils.check_grad(fun, np.array([w]), 
                                                verbose=False, 
                                                rtol=self.rtol_grad)
                
                self.assertFalse(grad_failed, 'Incorrect frequency gradient.')
                


if __name__ == '__main__':
    unittest.main()