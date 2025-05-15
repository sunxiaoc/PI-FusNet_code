# Description

- 'PI-FusNet_code' is a Python module that implements a three-dimensional inversion algorithm based on physics-informed networks. This algorithm is described in the 2025 paper titled 'Three-dimensional inversion method based on multi-source fused physical information networks for leachate distribution in landfills' by SUN et al. The PI-FusNet model achieves the fusion of resistivity and self-potential data. 
- 'main.py' is the main function of the program, which includes code for modules such as network invocation, loss function construction, and network training and testing.
- 'net_modules.py' is the network structure module, which includes components such as multi-scale residual blocks, invertible blocks, and reconstruction blocks.
- 'self_potential_imaging.py' is an algorithm module for self-potential probability density imaging, designed to transform one-dimensional self-potential (SP) measurements into two-dimensional charge probability density distributions.

# Requirements

- pandas
- numpy
- pytorch
- sklearn
- matplotlib
- os
- time
- random
- argparse
- json
