"""
Unit Tests for continuation postprocessing functions
"""

import sys
import numpy as np
import unittest

sys.path.append('../..')
import tmdsimpy.postprocess.continuation as cpost



class TestContinuationPost(unittest.TestCase):
        
    def test_hermite_upsample_cubic(self):
        """
        Test hermite upsample interpolation using a simple polynomial case
        
        Returns
        -------
        None.

        """
        
        # let t be the index of the row and be used to parameterize the 
        # interpolation problem
        # the values to be interpolated are then
        # [t**3, t**2, 5, t**3-t**2+t-2, t]
        
        tmax = 10
        
        t = 1.0*np.array(range(tmax+1))
        t = t.reshape(-1, 1)
        
        XlamP_fun = lambda t : np.hstack((t**3, 
                                           t**2, 
                                           5*np.ones_like(t), 
                                           t**3-t**2+t-2, 
                                           t))
        
        XlamP_full = XlamP_fun(t)
        
        XlamP_grad_full = np.hstack((3*t**2, 
                                       2*t, 
                                       np.zeros_like(t), 
                                       3*t**2-2*t+1, 
                                       np.ones_like(t)))
        
        ######### Upsample with default rate
        
        XlamP_interp = cpost.hermite_upsample(XlamP_full, XlamP_grad_full)
        
        t_expect = np.linspace(0, tmax, (XlamP_full.shape[0]-1)*10+1)
        XlamP_expect = XlamP_fun(t_expect.reshape(-1,1))
        
        self.assertLess(np.linalg.norm(XlamP_interp - XlamP_expect), 1e-12,
                        'Default cubic hermite spline interpolation failed.')
        
        ######### Upsample at rate 7
        
        XlamP_interp = cpost.hermite_upsample(XlamP_full, XlamP_grad_full,
                                              upsample_freq=7)
        
        t_expect = np.linspace(0, tmax, (XlamP_full.shape[0]-1)*7+1)
        XlamP_expect = XlamP_fun(t_expect.reshape(-1,1))
        
        self.assertLess(np.linalg.norm(XlamP_interp - XlamP_expect), 1e-12,
                        '7 point cubic hermite spline interpolation failed.')
        
        
        ######### Interpolate to specific points
        
        step_interp = np.array([0.5, 1.87, 3.215])
        
        
        XlamP_interp = cpost.hermite_upsample(XlamP_full, XlamP_grad_full,
                                              new_lams=step_interp)
        
        XlamP_expect = XlamP_fun(step_interp.reshape(-1,1))
        
        self.assertLess(np.linalg.norm(XlamP_interp - XlamP_expect), 1e-12,
                        'Specific point cubic hermite spline interpolation failed.')
        
        ######### Upsample with default rate + altered gradients
        
        XlamP_grad_full[0, :] = 0.13 * XlamP_grad_full[0, :]
        XlamP_grad_full[2, :] = 2.0 * XlamP_grad_full[2, :]
        XlamP_grad_full[3, :] = 1.75 * XlamP_grad_full[3, :]
                
        XlamP_interp = cpost.hermite_upsample(XlamP_full, XlamP_grad_full)
        
        t_expect = np.linspace(0, tmax, (XlamP_full.shape[0]-1)*10+1)
        XlamP_expect = XlamP_fun(t_expect.reshape(-1,1))
        
        self.assertLess(np.linalg.norm(XlamP_interp - XlamP_expect), 1e-12,
                        'Upsample failed with randomly adjusted gradients.')
        
        return
        
    def test_hermite_interp_errors(self):
        """
        Test the cases that should throw errors for hermite interpolation
        along the last variable. These are
        
        1. Non-monotonic XlamP_full[:, -1] 
        2. Negative values of XlamP_grad_full[:, -1]
        3. Both of the previous being okay, but the cubic of lam still being
        non-monotonic
        4. Value asked for is out of bounds
        

        Returns
        -------
        None.

        """
        
        XlamP_full = np.ones((5, 4))
        XlamP_full[:, -1] = np.array(range(5))
        
        XlamP_grad_full = np.ones_like(XlamP_full)
        
        
        ##############
        # Out of bounds low
        out_low = lambda : cpost.hermite_interp(XlamP_full, XlamP_grad_full,
                                                np.array([-1.0]))
        
        self.assertRaises(AssertionError, out_low)
        
        ################
        # Out of bounds high
        out_high = lambda : cpost.hermite_interp(XlamP_full, XlamP_grad_full,
                                                np.array([5.0]))
        
        self.assertRaises(AssertionError, out_high)
        
        ################
        # Fail with repeated entry 
        XlamP_repeat = np.copy(XlamP_full)
        XlamP_repeat[-1, -1] = XlamP_repeat[-2, -1]
        
        err_repeat = lambda : cpost.hermite_interp(XlamP_repeat, XlamP_grad_full,
                                                np.array([1.0]))
        
        
        self.assertRaises(AssertionError, err_repeat)
        
        ################
        # Fail with grad being negative
        XlamP_grad_neg = np.copy(XlamP_grad_full)
        XlamP_grad_neg[-2, -1] = -1.0
        
        err_grad = lambda : cpost.hermite_interp(XlamP_full, XlamP_grad_neg,
                                                np.array([1.0]))
        
        self.assertRaises(AssertionError, err_grad)
        
        return
    
    def test_hermite_interp_values(self):
        """
        Test interpolation to specific values of lam parameter

        Returns
        -------
        None.

        """
        
        tp = np.array([0, 1, 2, 3, 4, 5])
        
        # Define several other variables that are at most cubic. 
        XlamP_fun = lambda tcol : np.hstack((tcol**3, 
                                            tcol**2, 
                                            tcol, 
                                            np.ones_like(tcol),
                                            tcol**3-tcol**2+tcol-2, 
                                            tcol))
        
        XlamP_grad_fun = lambda tcol : np.hstack((3*tcol**2, 
                                            2*tcol, 
                                            np.ones_like(tcol), 
                                            np.zeros_like(tcol),
                                            3*tcol**2-2*tcol+1, 
                                            np.ones_like(tcol)))
        
        XlamP_full = XlamP_fun(tp.reshape(-1,1))
        XlamP_grad_full = XlamP_grad_fun(tp.reshape(-1,1))
        
        step_interp = np.array([0.0, 0.5, 1.87, 2.0, 3.215, 4.99, 5.0])
        
        # Choose several random fractional steps and some whole number steps
        # and use upsample to get those values
        # Also include some exact points including ends to make sure it is robust
        XlamP_interp_ref = cpost.hermite_upsample(XlamP_full, XlamP_grad_full,
                                                  new_lams=step_interp)
        
        # Use results of upsample to pass in values of lam corresponding to 
        # known points
        XlamP_interp_lam = cpost.hermite_interp(XlamP_full, XlamP_grad_full, 
                                                XlamP_interp_ref[:, -1])
        
        # Check Solutions
        
        self.assertLess(np.linalg.norm(XlamP_interp_lam - XlamP_interp_ref), 
                        1e-12,
                        'Default cubic hermite spline interpolation failed.')
        
        return
    
    def test_linear_interp_values(self):
        """
        Test linear interpolation calculations
        """
        
        rng = np.random.default_rng(seed=1023)
        
        XlamP_full = rng.random((5, 7))
        XlamP_full[:, -1] = np.sort(XlamP_full[:, -1])
        XlamP_full[0, -1] = 0.0
        XlamP_full[-1, -1] = 1.0
        
        reference_values = 2*XlamP_full[:, -1] + 1
        
        new_values = rng.random(8)
        
        # 1. Test for errors in user inputs
        fun = lambda : cpost.linear_interp(XlamP_full, new_values, 
                                       reference_values=np.array([-1, 1, 0]))
        
        self.assertRaises(AssertionError, fun)
        
        # 2. Correct return shape for a single interpolation point
        XlamP_interp = cpost.linear_interp(XlamP_full, 0.5)
        
        self.assertEqual(XlamP_interp.shape, (1,7),
             'Interpolation to a single point does not give expected shape.')
        
        # 3. Handling out of bounds as np.nan on top and bottom
        XlamP_interp = cpost.linear_interp(XlamP_full, 
                                           np.array([-1.0, 0.5, 1.5]))
        
        self.assertTrue(np.all(np.isnan(XlamP_interp[0])), 
                        'Failed to return NaN for out of bounds on lower side')
        
        self.assertTrue(np.all(np.isnan(XlamP_interp[-1])), 
                        'Failed to return NaN for out of bounds on upper side')

        # 4. Correctly interpolating all columns for no provided reference
        XlamP_interp = cpost.linear_interp(XlamP_full, new_values)
        
        XlamP_ref = np.zeros((new_values.shape[0], XlamP_full.shape[1]))
        
        for col in range(XlamP_full.shape[1]):
            
            XlamP_ref[:, col] = np.interp(new_values, XlamP_full[:, -1], 
                                          XlamP_full[:, col])
        
        self.assertLess(np.linalg.norm(XlamP_interp - XlamP_ref), 1e-12, 
                        'Incorrect interpolation results.')
        
        
        # 5. Correctly interpolating for a provided reference
        new_vals5 = 2*new_values+1
        XlamP_interp = cpost.linear_interp(XlamP_full, new_vals5,
                                           reference_values=reference_values)
        
        XlamP_ref = np.zeros((new_values.shape[0], XlamP_full.shape[1]))
        
        for col in range(XlamP_full.shape[1]):
            
            XlamP_ref[:, col] = np.interp(new_vals5, reference_values, 
                                          XlamP_full[:, col])
        
        self.assertLess(np.linalg.norm(XlamP_interp - XlamP_ref), 1e-12, 
                    'Incorrect interpolation results for provided reference.')
                
        return
    
if __name__ == '__main__':
    unittest.main()
