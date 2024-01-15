"""
Verify that the normal asperity contact model matches the previous MATLAB 
version

Steps:
    1. Verify some force displacement against MATLAB
    2. Verify autodiff grads at each point
    3. Verify double going to maximum point and get the correct unloading grad
    4. Test vectorizing the call to normal loading asperity functions

"""


# Standard imports
import numpy as np
import sys
import unittest

# Python Utilities
sys.path.append('../..')

from tmdsimpy.jax.nlforces.roughcontact.rough_contact import RoughContactFriction 
import tmdsimpy.harmonic_utils as hutils


sys.path.append('..')
import verification_utils as vutils

# Import for the reference data
import yaml



###############################################################################
###     Testing Class                                                       ###
###############################################################################




class TestRoughContact(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Define tolerances here for all the tests

        Returns
        -------
        None.

        """

        super(TestRoughContact, self).__init__(*args, **kwargs)     
        
        # Tolerances
        self.rel_force_tol = 1e-6 # Error on force evaluation
        self.atol_grad = 1e-16
        self.rtol_grad = 1e-8
        
        ###############
        # Load Reference Data from old Rough Contact Model
        yaml_file = './reference/normal_asperity.yaml'
        
        with open(yaml_file, 'r') as file:
            ref_dict = yaml.safe_load(file)
            
        # Construct New Rough Contact Models to Match the old ones
        Q = np.eye(3)
        T = np.eye(3)
        
        normal_asp_models = (len(ref_dict['E']) )* [None]
        mu = 1e20
        
        for ind in range(len(ref_dict['E'])):
        
            curr_model = RoughContactFriction(Q, T, ref_dict['E'][ind], 
                                              ref_dict['nu'][ind], 
                                              ref_dict['R'][ind], 
                                              ref_dict['Et'][ind], 
                                              ref_dict['Sys'][ind], 
                                              mu, u0=0, meso_gap=0, 
                                              gaps=np.array([0.0]), 
                                              gap_weights=np.array([1.0]))
            
            normal_asp_models[ind] = curr_model
            
        self.normal_asp_models = normal_asp_models
        
        self.normal_asp_un = ref_dict['normal_disp']
        self.normal_asp_fn = ref_dict['normal_force']
        self.normal_asp_rad = ref_dict['contact_radius']
        
        self.normal_asp_ux = ref_dict['x_disp']
        self.normal_asp_fx = ref_dict['x_force']
        self.normal_asp_uy = ref_dict['y_disp']
        self.normal_asp_fy = ref_dict['y_force']
            
        ###############
        # Create a model that uses the repeated parameters from ind=[0,1], 
        # but uses a list of gaps and adds the forces.
        offset = np.array(ref_dict['normal_disp'])[1, 0] - np.array(ref_dict['normal_disp'])[0, 0]
        
        two_asp_model = RoughContactFriction(Q, T, ref_dict['E'][1], 
                                          ref_dict['nu'][1], 
                                          ref_dict['R'][1], 
                                          ref_dict['Et'][1], 
                                          ref_dict['Sys'][1], 
                                          mu, u0=0, meso_gap=0, 
                                          gaps=np.array([offset, 0.0, 0.0]), 
                                          gap_weights=np.array([1.0, 0.5, 0.0]))
        
        self.two_asp_model = two_asp_model
        self.two_asp_un = ref_dict['normal_disp'][1]
        self.two_asp_ux = ref_dict['x_disp'][1]
        self.two_asp_uy = ref_dict['y_disp'][1]
        
        
        self.two_asp_fn = np.array(ref_dict['normal_force'][0]) \
                                + 0.5*np.array(ref_dict['normal_force'][1])
        
        self.two_asp_fx = np.array(ref_dict['x_force'][0]) \
                                + 0.5*np.array(ref_dict['x_force'][1])
                                
        self.two_asp_fy = np.array(ref_dict['y_force'][0]) \
                                + 0.5*np.array(ref_dict['y_force'][1])
                                
                                
        ###############
        yaml_file = './reference/element_cycle_tractions.yaml'
        
        with open(yaml_file, 'r') as file:
            ref_dict = yaml.safe_load(file)
            
            
        element_model = RoughContactFriction(Q, T, ref_dict['E'], 
                                          ref_dict['nu'], 
                                          ref_dict['R'], 
                                          ref_dict['Et'], 
                                          ref_dict['Sys'], 
                                          ref_dict['mu'], 
                                          u0=0, meso_gap=0, 
                                          gaps=np.array(ref_dict['gap_values']), 
                                          gap_weights=np.array(ref_dict['gap_weights']))
        
        element_uxyn = [np.array(val).T for val in ref_dict['uxyn']]
        element_txyn = [np.array(val).T for val in ref_dict['txyn']]
        
        self.element_model = element_model
        self.element_uxyn = element_uxyn
        self.element_txyn = element_txyn
        
    def test_normal_asperity(self):
        """
        Test a series of applied normal displacements for different material 
        properties
        
        At each displacement, check the gradient. 

        Returns
        -------
        None.

        """
        
        for ind in range(len(self.normal_asp_models)):
            
            model = self.normal_asp_models[ind]
            
            model.init_history()
            
            ux = self.normal_asp_ux[ind]
            uy = self.normal_asp_uy[ind]
            un = self.normal_asp_un[ind]
            
            fx = self.normal_asp_fx[ind]
            fy = self.normal_asp_fy[ind]
            fn = self.normal_asp_fn[ind]
            contact_rad = self.normal_asp_rad[ind]
            
            for j in range(len(un)):
                
                # Evaluate Force
                uxyn = np.zeros((3))
                uxyn[0] = ux[j]
                uxyn[1] = uy[j]
                uxyn[-1] = un[j]
                
                fxyn_ref = np.array([fx[j], fy[j], fn[j]])
                
                F, dFdX = model.force(uxyn, update_hist=False)
                                
                self.assertLessEqual(np.linalg.norm(F - fxyn_ref), 
                                     np.linalg.norm(fxyn_ref)*self.rel_force_tol, 
                                     'Asperity Model: {}, load index: {}'.format(ind, j))
                
                # Verify Gradient
                fun = lambda X : model.force(X, update_hist=False)
                
                grad_failed = vutils.check_grad(fun, uxyn, verbose=False, 
                                            atol=self.atol_grad,
                                            rtol=self.rtol_grad,
                                            h=1e-5*un[-1])
                
                self.assertFalse(grad_failed, 
                                 'Incorrect Gradient w.r.t. Uxyn, Asperity Model: {}, load index: {}'.format(ind, j))
                
                # Update history
                F, dFdX, aux = model.force(uxyn*np.array([0,0,1]), 
                                           update_hist=True, return_aux=True)
                new_contact_rad = aux[4]
                
                self.assertLessEqual(np.abs(new_contact_rad - contact_rad[j]), 
                                     np.abs(contact_rad[j])*self.rel_force_tol, 
                                     'Asperity Model: {}, load index: {}'.format(ind, j))
                
            
    def test_two_normal_asp(self):
        """
        Test vectorized version of two asperities with normal contact

        Returns
        -------
        None.

        """
        
        model = self.two_asp_model
        
        model.init_history()
        
        ux = self.two_asp_ux
        uy = self.two_asp_uy
        un = self.two_asp_un
        
        fx = self.two_asp_fx
        fy = self.two_asp_fy
        fn = self.two_asp_fn
        
        for j in range(len(un)):
            
            # Evaluate Force
            uxyn = np.array([ux[j], uy[j], un[j]])
            fxyn_ref = np.array([fx[j], fy[j], fn[j]])
            
            F, dFdX = model.force(uxyn, update_hist=False)
            
            self.assertLessEqual(np.linalg.norm(F - fxyn_ref), 
                                 np.linalg.norm(fxyn_ref)*self.rel_force_tol, 
                                 'Two Asperity Model, load index: {}'.format(j))
                
            # Verify Gradient
            fun = lambda X : model.force(X, update_hist=False)
            
            grad_failed = vutils.check_grad(fun, uxyn, verbose=False, 
                                        atol=self.atol_grad,
                                        rtol=self.rtol_grad,
                                        h=1e-5*un[-1])
            
            self.assertFalse(grad_failed, 
                             'Incorrect Gradient w.r.t. Uxyn, Two Asperity Model, load index: {}'.format(j))
            
            # Update history
            F, dFdX = model.force(uxyn*np.array([0,0,1]), update_hist=True)
              
    def test_unload_grad_asp(self):
        """
        Test gradient at maximum displacement to verify that it takes the 
        unloading gradient.

        Returns
        -------
        None.

        """
        
        model = self.normal_asp_models[0]
        
        model.init_history()
        
        un = np.max(np.array(self.normal_asp_un[0]))
        
        uxyn = np.array([0.0, 0.0, un])
        
        F_load, dFdX_load = model.force(uxyn, update_hist=True)
        
        
        F_unload, dFdX_unload = model.force(uxyn, update_hist=False)
        
        h = 1e-5*un
        uxyn_left = np.array([0.0, 0.0, un - h])
        
        F_left, dFdX_left = model.force(uxyn_left, update_hist=False)
        
        dFdX_unload_num = (F_unload[-1] - F_left[-1]) / h
        
        self.assertLess(np.abs(dFdX_unload[-1, -1] - dFdX_unload_num), 
                        np.abs(dFdX_unload[-1, -1] - dFdX_load[-1, -1])*1e-4, 
                        'Gradient for repeated normal displacement wrongly using loading branch')
        
    def test_cycle_forces(self):
        """
        Verify that the local history of element tractions is correct against
        reference data from the previous implementation 
        (covers a full cycle of stick and slip and reversals)

        Returns
        -------
        None.

        """
        element_model = self.element_model
        element_uxyn = self.element_uxyn
        element_txyn = self.element_txyn
        
        for ind in range(len(element_uxyn)):
            
            unlt = element_uxyn[ind]
            
            # 1. Calculate forces over a cycle and compare against a saved reference
            fxyn_curr = element_model.local_force_history(unlt, 0, 0, 0, 
                                                       np.array([0,0,0]), 
                                                       max_repeats=1)
            
            self.assertLess(np.linalg.norm(fxyn_curr - element_txyn[ind]), 
                            np.linalg.norm(element_txyn[ind])*self.rel_force_tol,
                            'Combined element tractions do not match expected values over a cycle.')
        
            if ind == 5:
                # 2. Repeat over two cycles and check that the first entry may
                # in general change
                fxyn_steady = element_model.local_force_history(unlt, 0, 0, 0, 
                                                       np.array([0,0,0]), 
                                                       max_repeats=2)
                
                self.assertGreater(np.abs(fxyn_steady[0,0] - fxyn_curr[0,0]), 
                                   10.0, 'Case should differ on second cycle')
                
                self.assertLess(np.abs(fxyn_steady[3,0] - fxyn_curr[3,0]), 
                                   1e-15, 'Case should be steady-state here.')
                
    def test_aft(self): 
        """
        Test the AFT implementation of the rough contact model.
        
        1. Test gradients for several cases
        
        Returns
        -------
        None.

        """
        
        element_model = self.element_model
        
        U_list = [
                  np.array([0, .5e-5, 2e-5, 
                            0.1e-5, 0.2e-5, 0.05e-5, 
                            0.3e-5, 0.2e-5, 0.05e-5, 
                            0.1e-5, 0.1e-5, 0.02e-5]),
                  np.array([0, .5e-5, 2e-5, 
                          0.1e-5, 0.2e-5, 0.05e-5, 
                          0.3e-5, 0.2e-5, 0.05e-5, 
                          0.0, 0.0, 0.0, 
                          0.0, 0.0, 0.0,
                          0.1e-5, 0.1e-5, 0.02e-5]),
                  np.array([0, .5e-5, 1e-5, 
                          0.1e-5, 0.2e-5, 0.05e-5, 
                          0.3e-5, 0.2e-5, 0.05e-5, 
                          0.1e-5, 0.1e-5, 0.02e-5]),
                  np.array([0, .5e-5, .1e-5, 
                          0.1e-5, 0.2e-5, 0.05e-5, 
                          0.3e-5, 0.2e-5, 0.1e-5, 
                          0.1e-5, 0.1e-5, 0.05e-5]),
                  np.array([0, .5e-5, -.1e-5, 
                          0.1e-5, 0.2e-5, -0.1e-5, 
                          0.3e-5, 0.2e-5, -0.1e-5, 
                          0.1e-5, 0.1e-5, -0.5e-5]),
                  np.array([0, .5e-5, 2e-5, 
                          0.0e-5, 0.0e-5, -0.0e-5, 
                          0.4e-5, 0.4e-5, 0.08e-5, 
                          0.0e-5, 0.0e-5, -0.0e-5,
                          0.1e-5, 0.1e-5, 0.02e-5]),
                  np.array([0, .5e-5, 2e-5, 
                          0.0e-5, 0.0e-5, 0.0e-5, 
                          0.4e-5, 0.4e-5, 0.08e-5, 
                          0.0e-5, 0.0e-5, 0.0e-5,
                          0.0e-5, 0.0e-5, 0.0e-5,
                          0.0e-5, 0.0e-5, 0.0e-5,
                          0.1e-5, 0.1e-5, 0.02e-5]),
                ]
        
        w = 1.35
        h = np.array(range(5+1))
        Nt = 1 << 7
        
        Ndof = 3
        Nhc = hutils.Nhc(h)

        # import pdb; pdb.set_trace()

        for j in range(len(U_list)):
            
            
            U = np.zeros(Ndof * Nhc)
            U[:U_list[j].shape[0]] = U_list[j]
            
            fun = lambda U : element_model.aft(U, w, h, Nt=Nt, max_repeats=2)[0:2]
            
            loosen_grad = 1
            if j == 6:
                loosen_grad = 1e3
            
            grad_failed = vutils.check_grad(fun, U, verbose=False, 
                                        atol=self.atol_grad,
                                        rtol=self.rtol_grad*loosen_grad,
                                        h=1e-5*np.linalg.norm(U))
            
            self.assertFalse(grad_failed, 
                             'Incorrect Gradient w.r.t. U for AFT, set: {}'.format(j))
            
            dFdw = element_model.aft(U, w, h, Nt=Nt, max_repeats=2)[2]
            
            self.assertLess(np.linalg.norm(dFdw), 1e-12, 
                            'Gradient w.r.t. w should be zero.')
   
    def test_aft_shapes(self): 
        """
        Test the AFT implementation of the rough contact model.
        
        1. Look at force coefficients and make sure that they look correct
        2. Test flag for not returning gradient.
        
        These cases show a lot more harmonic contributions in the tangent 
        direction than may be expected. However, if the first harmonic component
        in the normal direction is set to zero, then roughly the expected 
        distribution of only odd harmonics are observed in the tangent directions.
        
        Returns
        -------
        None.

        """
        
        element_model = self.element_model
        
        U_sub = np.array([0, 0, 2e-5, 
                        3.0e-5, 1.0e-5, 0.5e-5,])
        
        w = 1.35
        h = np.array(range(5+1))
        Nt = 1 << 7
        
        Ndof = 3
        Nhc = hutils.Nhc(h)


        U = np.zeros(Ndof * Nhc)
        U[:U_sub.shape[0]] = U_sub
        
        # Only Normal displacement
        U_n = np.copy(U)
        U_n[0::3] = 0
        U_n[1::3] = 0
        
        # Normal and X
        U_x = np.copy(U)
        U_x[1::3] = 0
        
        # All Three
        U_y = U
        
        fun = lambda U : element_model.aft(U, w, h, Nt=Nt, max_repeats=2)[0]
        
        F_n = fun(U_n)
        F_x = fun(U_x)
        F_y = fun(U_y)
        
        # Normal motion should not result in any tangent coefficients
        self.assertEqual(np.linalg.norm(F_n[0::3]), 0, 
                         'Normal displacement should only cause normal forces.')
            
        self.assertEqual(np.linalg.norm(F_n[1::3]), 0, 
                         'Normal displacement should only cause normal forces.')
        
        # Normal and X should not cause any Y forces and should match the 
        # normal forces from before
                            
        self.assertLess(np.linalg.norm(F_x[2::3] - F_n[2::3]), 1e-12, 
                         'Normal and tangent should not change normal.')
        
        self.assertEqual(np.linalg.norm(F_x[1::3]), 0, 
                         'Normal and x should not cause y.')
        
        # Adding y should not change x and normal
        self.assertLess(np.linalg.norm(F_y[2::3] - F_n[2::3]), 1e-12, 
                         'Adding Y should not change normal direction loads.')
        
        self.assertLess(np.linalg.norm(F_y[0::3] - F_x[0::3]), 1e-12, 
                         'Adding Y should not change X direction loads.')
        
        
        res_default = element_model.aft(U, w, h, Nt=Nt, max_repeats=2)
        res_no_grad = element_model.aft(U, w, h, Nt=Nt, max_repeats=2, calc_grad=False)
        res_true_grad = element_model.aft(U, w, h, Nt=Nt, max_repeats=2, calc_grad=True)
        
        self.assertEqual(len(res_default), 3, 'Default AFT returns wrong number of outputs')
        self.assertEqual(len(res_no_grad), 1, 'No Grad AFT returns wrong number of outputs')
        self.assertEqual(len(res_true_grad), 3, 'True Grad AFT returns wrong number of outputs')
        
        # Should be exact since the calculation is the same, just 
        # returning different outputs
        self.assertEqual(np.linalg.norm(res_default[0] - res_no_grad[0]), 0.0)
                
if __name__ == '__main__':
    unittest.main()