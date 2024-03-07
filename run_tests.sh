cd tests
python3 -m unittest discover

cd nlforces
python3 -m unittest discover

cd ../postprocess/
python3 -m unittest discover

# If jax is installed, also check those tests
cd ../jax
python3 -m unittest discover

# These also require jax
cd ../roughcontact/
python3 -m unittest discover

# Return to top level
cd ../..
