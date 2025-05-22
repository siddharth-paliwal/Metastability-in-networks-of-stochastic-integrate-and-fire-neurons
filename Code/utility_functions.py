import numpy as np

def get_mean_memVol(data, parameters):

    meanData = np.zeros((data.shape[0], data.shape[1], data.shape[2], parameters.numClusters))

    for i in range(parameters.numClusters):

        meanData[:, :, :, i] = np.mean(data[:, :, :, i*parameters.EClusterSize:(i + 1) * parameters.EClusterSize], axis=3)

    return meanData

def get_psth(data, window_size, parameters):

    total_time = data.shape[0]
    N = data.shape[1]
    psth = np.zeros((int(total_time/window_size), N))
    i = 0
    for t in range(0, total_time, window_size):
        psth[i, :] = np.sum(data[t:t + window_size, :].flatten()) / window_size
        i += 1

    return psth

def get_spikeTrainVariability(data, parameters):

    f = get_psth(data, 200, parameters)
    fi = np.zeros((f.shape[0], parameters.numClusters))

    for i in range(parameters.numClusters):
        fi[:, i] = np.mean(f[:, i * parameters.EClusterSize:(i + 1) * parameters.EClusterSize], axis=1)

    spkTrVar = 1 / parameters.numClusters * np.sum(np.std(fi, axis=1))

    return spkTrVar


def get_W_MF(W, parameters):

    W1 = np.zeros((2 * parameters.numClusters, 2 * parameters.numClusters))
    for i in range(parameters.numClusters):
        for j in range(parameters.numClusters):
            W1[i, j] = np.mean(W[i * parameters.EClusterSize: (i + 1) * parameters.EClusterSize,
                               j * parameters.EClusterSize:(j + 1) * parameters.EClusterSize])
            W1[i, j + parameters.numClusters] = np.mean(W[i * parameters.EClusterSize:(i + 1) * parameters.EClusterSize,
                                                        parameters.NE + j * parameters.IClusterSize:parameters.NE + (
                                                                    j + 1) * parameters.IClusterSize])
            W1[i + parameters.numClusters, j] = np.mean(
                W[parameters.NE + i * parameters.IClusterSize:parameters.NE + (i + 1) * parameters.IClusterSize,
                j * parameters.EClusterSize:(j + 1) * parameters.EClusterSize])
            W1[i + parameters.numClusters, j + parameters.numClusters] = np.mean(
                W[parameters.NE + i * parameters.IClusterSize:parameters.NE + (i + 1) * parameters.IClusterSize,
                parameters.NE + j * parameters.IClusterSize:parameters.NE + (j + 1) * parameters.IClusterSize])

    return W1


def get_noise_MF(noise, parameters):

    noise1 = np.zeros((2 * parameters.numClusters,))
    for i in range(parameters.numClusters):
        noise1[i] = np.mean(noise[i * parameters.EClusterSize:(i + 1) * parameters.EClusterSize])
        noise1[i + parameters.numClusters] = np.mean(noise[
                                                     parameters.numClusters + i * parameters.IClusterSize:parameters.numClusters + (
                                                                 i + 1) * parameters.IClusterSize])

    return noise1
