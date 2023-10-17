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
        
        # Load Reference Data from old Rough Contact Model
        yaml_file = './reference/normal_asperity.yaml'
        
        with open(yaml_file, 'r') as file:
            ref_dict = yaml.safe_load(file)
            
        # Construct New Rough Contact Models to Match the old ones
        Q = np.eye(3)
        T = np.eye(3)
        
        normal_asp_models = (len(ref_dict['E']) )* [None]
        mu = 1.0
        
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
        self.normal_asp_disp = ref_dict['normal_disp']
        self.normal_asp_force = ref_dict['normal_force']
        self.normal_asp_rad = ref_dict['contact_radius']
            
        # Create a model that uses the repeated parameters from ind=[0,1], 
        # but uses a list of gaps and adds the forces.
        offset = np.array(ref_dict['normal_disp'])[1, 0] - np.array(ref_dict['normal_disp'])[0, 0]
        
        two_asp_model = RoughContactFriction(Q, T, ref_dict['E'][1], 
                                          ref_dict['nu'][1], 
                                          ref_dict['R'][1], 
                                          ref_dict['Et'][1], 
                                          ref_dict['Sys'][1], 
                                          mu, u0=0, meso_gap=0, 
                                          gaps=np.array([offset, 0.0]), 
                                          gap_weights=np.array([1.0, 0.5]))
        
        self.two_asp_model = two_asp_model
        self.two_asp_disp = ref_dict['normal_disp'][1]
        self.two_asp_force = np.array(ref_dict['normal_force'][0]) \
                                + 0.5*np.array(ref_dict['normal_force'][1])
        
        
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
            
            un = self.normal_asp_disp[ind]
            fn = self.normal_asp_force[ind]
            contact_rad = self.normal_asp_rad[ind]
            
            for j in range(len(un)):
                
                # Evaluate Force
                uxyn = np.zeros((3))
                uxyn[-1] = un[j]
                
                F, dFdX = model.force(uxyn, update_hist=False)
                
                # if j ==4:
                #     import pdb; pdb.set_trace()
                
                self.assertLessEqual(np.abs(F[-1] - fn[j]), 
                                     np.abs(fn[j])*self.rel_force_tol, 
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
                F, dFdX, aux = model.force(uxyn, update_hist=True, return_aux=True)
                new_contact_rad = aux[3]
                
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
        
        un = self.two_asp_disp
        fn = self.two_asp_force
        
        for j in range(len(un)):
            
            # Evaluate Force
            uxyn = np.zeros((3))
            uxyn[-1] = un[j]
            
            F, dFdX = model.force(uxyn, update_hist=False)
            
            # if j ==4:
            #     import pdb; pdb.set_trace()
            
            self.assertLessEqual(np.abs(F[-1] - fn[j]), 
                                 np.abs(fn[j])*self.rel_force_tol, 
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
            F, dFdX = model.force(uxyn, update_hist=True)
            
            
if __name__ == '__main__':
    unittest.main()