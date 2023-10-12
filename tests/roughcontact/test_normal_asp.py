"""
Verify that the normal asperity contact model matches the previous MATLAB 
version

Steps:
    1. Verify some force displacement against MATLAB
    2. Verify autodiff grads at each point
    3. Verify double going to maximum point and get the correct unloading grad
    4. Test vectorizing the call to normal loading asperity functions

"""

import yaml

yaml_file = './reference/normal_asperity.yaml'

with open(yaml_file, 'r') as file:
    ref_dict = yaml.safe_load(file)
    