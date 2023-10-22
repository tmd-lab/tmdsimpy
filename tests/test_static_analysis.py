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






