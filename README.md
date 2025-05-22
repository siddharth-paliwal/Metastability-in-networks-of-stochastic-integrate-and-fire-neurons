## Metastability in networks of stochastic integrate-and-fire neurons

This repository contains code to generate figures associated with the above paper.

Each file in the folder 'Figures' named 'Figure *.ipynb' is a jupyter file that can be run to generate each figure. The data generated from simulations and theory is saved in the Data folder which has folders within for each figure.

The simulations can be run by running 'main.py' in the 'Code' folder. Be sure to change parameters associated with the simulation being run in 'parameters.py'. The details for what each file does is given below:

main.py: The main file to run simulations for different parameter sets as required. Simulations may be run for pre-defined parameter sets by uncommenting the variables 'var1' or 'var2', etc., listed in main.py, or by specifying parameters in parameters.py. The default code in this repository draws parameters from parameters.py.

parameters.py: Contains the parameters to be used for simulations.

phi.py: Contains functions for various classes of nonlinearities that can be used in the model.

sim_network_EI.py: Contains code to run network simulations for different network architectures.

utility_functions.py: Contains some miscellaneous functions called when running the code aove.

weight_matrix.py: Contains code to generate weight matrices for different network architectures.
