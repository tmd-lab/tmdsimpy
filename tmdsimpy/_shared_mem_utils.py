"""
Helper functions for evaluating nonlinear forces with shared memory. 

To prevent errors, these need to be in a separate module file than the calls 
to these functions
"""

import numpy as np
import warnings


def _single_aft(U, w, h, Nt, aft_tol, nlforce):
    """
    Simple wrapper function to call AFT on single nonlinear force to enable
    parallelism
    """
    res = nlforce.aft(U, w, h, Nt, aft_tol, return_local=True)
    
    return res
    
def combine_aft(res1, res2):
    """
    Simple helper function to do add reduction across AFT results
    """
    return (res1[0]+res2[0], res1[1]+res2[1], res1[2]+res2[2])

def divide_list(items, num_parts):
    """
    There should be some easy tests to write for this
    """
    
    warnings.warn('Shared memory divide list does not have a test.')

    if num_parts > len(items):
        num_parts = len(items)
    
    divided = num_parts*[None]
    
    sub_length = len(items) // num_parts # Floor with integer division    
    extra_items = len(items) % num_parts
    
    for i in range(num_parts):
        
        start_offset = np.minimum(i, extra_items)
        end_offset   = np.minimum(i+1, extra_items)
        
        divided[i] = items[(i*sub_length + start_offset):((i+1)*sub_length + end_offset)]
        
    
    return divided