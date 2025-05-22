import os
from parameters import *
from weight_matrix import *
from sim_network_EI import *
from plot_simulations import *
import pickle as pickle

# Create Path for saving Plots and Data
pathData = '../Data/EI Symmetric/'
if not os.path.exists(pathData):
    os.mkdir('../Data/EI Symmetric/')

pathPlot = '../Plot/'
if not os.path.exists(pathPlot):
    os.mkdir('../Plot/')

# Get network parameters
parameters = network_params()

var1 = np.array([4.0])
# var2 = np.array([1.0])

# var1 = np.arange(2.70, 3.21, 0.10).round(2)
var2 = np.arange(40, 410, 40)
var3 = 0.0 * np.ones((parameters.n_runs,))

# spkTrain = np.zeros((len(RAll), parameters.n_runs, parameters.Nt, 2*parameters.numClusters))
# memVol = np.zeros((len(RAll), parameters.n_runs, parameters.Nt, 2*parameters.numClusters))
# gSyn = np.zeros((len(RAll), parameters.n_runs, parameters.Nt, 2*parameters.numClusters))

for i in range(len(var1)):

    parameters.wEE = var1[i]

    for j in range(len(var2)):

        parameters.N = var2[j]
        parameters.NE = int(parameters.nE * parameters.N)
        parameters.NI = int(parameters.nI * parameters.N)

        # spkTrain = np.zeros((parameters.n_runs, parameters.Nt, parameters.N))
        # memVol = np.zeros((parameters.n_runs, parameters.Nt, parameters.N))
        # gSyn = np.zeros((parameters.n_runs, parameters.Nt, parameters.N))
        # spkTimes = np.zeros((parameters.n_runs, int(parameters.maxSpikes), 2))
        W = np.zeros((parameters.n_runs, parameters.N, parameters.N))
        # count = np.zeros((parameters.n_runs))
        spkCount = np.zeros((parameters.n_runs, parameters.Nt))

        for k in range(parameters.n_runs):
            parameters.stim = var3[k]
            # Get the weight matrix for the network
            optionW = {0: get_weight_matrix_Exc_Cluster, 1: get_weight_matrix_ExcInh_Cluster, 2: get_weight_matrix_Exc, 3: get_weight_matrix_Exc_N, 4: get_weight_matrix_Exc_Inh_N}
            np.random.seed(6565492)
            W[k, :, :] = optionW[4](parameters)

            # Simulate the network and get spiking data, membrane potential and synaptic current
            option = {0: simulate_network_EI, 1: simulate_network_Homogeneous, 2: simulate_network_MF, 3: simulate_network_deterministic}
            np.random.seed()
            # spkTrain[k, :, :], memVol[k, :, :], gSyn[k, :, :], spkTimes[k, :, :], count[k] = option[0](parameters, W[k, :, :])
            spkTemp, _, _, _, _ = option[0](parameters, W[k, :, :])
            spkCount[k, :] = np.sum(spkTemp, axis=1)


        print('var2'+str(j))
        # spkTrain = spkTrain.squeeze()
        # memVol = memVol.squeeze()
        # gSyn = gSyn.squeeze()
        # spkTimes = spkTimes.squeeze()
        W = W.squeeze()
        spkCount = spkCount.squeeze()

        # Save Data
        with open(pathData + str(parameters.N) + '_par', 'wb') as handle:
            pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # np.savez(pathData + str(parameters.N)  + '_R_' + str(var2[j]) + '_JP_' + str(var1[i]), spkTrain, memVol, gSyn, spkTimes, W, count)
        # np.savez(pathData + str(parameters.N)  + '_J_' + str(var1[i]), spkTrain, memVol, spkTimes, W)
        np.savez(pathData + str(parameters.N)  + '_J_' + str(var1[i]), spkCount, W)
    print('var1'+str(i))
