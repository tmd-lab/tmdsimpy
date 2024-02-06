"""
Unit Tests for continuation postprocessing functions
"""

import sys
import numpy as np
import unittest

sys.path.append('../..')
import tmdsimpy.postprocess.continuation_post as cpost



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
        
        t = np.array(range(tmax+1))
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
                                              new_points=step_interp)
        
        XlamP_expect = XlamP_fun(step_interp.reshape(-1,1))
        
        self.assertLess(np.linalg.norm(XlamP_interp - XlamP_expect), 1e-12,
                        'Specific point cubic hermite spline interpolation failed.')
        
        return
        
    def test_hermite_upsample_backtrack(self):
        """
        Test hermite upsample interpolation using a more complicated
        parametric function that backtracks in the second variable
        
        Points are
        t | x | y 
        0 | 0 | 1
        1 | 3 | 3
        2 | 2 | 2
        3 | 4 | 0
        
        Returns
        -------
        None.

        """
        
        x_fun = lambda t : 7*t**3/6.0 - 11*t**2/2.0 + 22*t/3.0
        dxdt_fun = lambda t : 21*t**2/6.0 - 22*t/2.0 + 22.0/3

        y_fun = lambda t : t**3/3.0 - 5*t**2/2.0 + 25*t/6.0 + 1
        dydt_fun = lambda t : 3*t**2/3.0 - 10*t/2.0 + 25/6.0
        
        """
        # Plotting example - use previous functions as well
        
        t = np.linspace(0, 3, 100)
        tp = np.array([0, 1, 2, 3])

        x = x_fun(t)
        y = y_fun(t)

        xp = x_fun(tp)
        yp = y_fun(tp)

        import matplotlib.pyplot as plt
        plt.plot(x, y)
        plt.plot(xp, yp, 'o')
        plt.show()
        """
        
        # Construct interpolation data
        tp = np.array([0, 1, 2, 3]).reshape(-1,1)
        
        XlamP_fun = lambda t : np.hstack((x_fun(t), y_fun(t)))
        
        XlamP_full = XlamP_fun(tp)
        XlamP_grad_full = np.hstack((dxdt_fun(tp), dydt_fun(tp)))
        
        ####### Test default up sampling
        
        XlamP_interp = cpost.hermite_upsample(XlamP_full, XlamP_grad_full)

        t_expect = np.linspace(0, XlamP_full.shape[0]-1, 
                               (XlamP_full.shape[0]-1)*10+1)
        
        XlamP_expect = XlamP_fun(t_expect.reshape(-1,1))
        
        self.assertLess(np.linalg.norm(XlamP_interp - XlamP_expect), 1e-12,
                        'Default cubic hermite spline interpolation failed.')
                
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
        
        ################
        # Fail with hidden double root in an interval
        
        # Points to construct polynomial of lam:
        # (0,0), (1, 3), (2, 2), (3, 4)
        # Only first and last point will be passed to interp function
        # value of 2.5 should have multiple real roots
        
        lam_fun = lambda tcol : 7*tcol**3/6.0 - 11*tcol**2/2.0 + 22*tcol/3.0
        lam_grad_fun = lambda tcol : 21*tcol**2/6.0 - 22*tcol/2.0 + 22/3.0
        
        tcol = np.array([0, 3])
        
        XlamP_full = np.ones((2, 5))
        XlamP_full[:, -1] = lam_fun(tcol)
        
        XlamP_grad_full = np.ones((2, 5))
        XlamP_grad_full[:, -1] = lam_grad_fun(tcol)
        
        err_roots = lambda : cpost.hermite_interp(XlamP_full, XlamP_grad_full,
                                                  np.array([2.5]))
        breakpoint()
        self.assertRaises(AssertionError, err_roots)
        
        return
    
    def test_hermite_interp_values(self):
        """
        Test interpolation to specific values of lam parameter

        Returns
        -------
        None.

        """
        
        tp = np.array([0, 1, 2, 3, 4, 5])
        
        dlamdt_fun = lambda t : t**2 + 5*t + 1 # always positive slope for t > 0
        lam_fun = lambda t : t**3/3.0 + 2.5*t**2 + t + 3.0
        
        # Define several other variables that are at most cubic. 
        XlamP_fun = lambda tcol : np.hstack((tcol**3, 
                                            tcol**2, 
                                            tcol, 
                                            np.ones_like(tcol),
                                            tcol**3-tcol**2+tcol-2, 
                                            lam_fun(tcol)))
        
        XlamP_grad_fun = lambda tcol : np.hstack((3*tcol**2, 
                                            2*tcol, 
                                            np.ones_like(tcol), 
                                            np.zeros_like(tcol),
                                            3*tcol**2-2*tcol+1, 
                                            dlamdt_fun(tcol)))
        
        XlamP_full = XlamP_fun(tp.reshape(-1,1))
        XlamP_grad_full = XlamP_grad_fun(tp.reshape(-1,1))
        
        step_interp = np.array([0.0, 0.5, 1.87, 2.0, 3.215, 4.99, 5.0])
        
        # Choose several random fractional steps and some whole number steps
        # and use upsample to get those values
        # Also include some exact points including ends to make sure it is robust
        XlamP_interp_ref = cpost.hermite_upsample(XlamP_full, XlamP_grad_full,
                                                  new_points=step_interp)
        
        # Use results of upsample to pass in values of lam corresponding to 
        # known points
        XlamP_interp_lam = cpost.hermite_interp(XlamP_full, XlamP_grad_full, 
                                                XlamP_interp_ref[:, -1])
        
        # Check Solutions
        
        self.assertLess(np.linalg.norm(XlamP_interp_lam - XlamP_interp_ref), 
                        1e-12,
                        'Default cubic hermite spline interpolation failed.')
        
        return