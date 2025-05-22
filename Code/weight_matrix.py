import numpy as np

def get_weight_matrix_Exc_Cluster(parameters):

    WRatio = parameters.WRatio  # Ratio of Win / Wout(synaptic weight of within group to neurons outside the group)
    REE = parameters.REE

    wEE = 1.40 / np.sqrt(parameters.N)  # 1.70
    wEI = 2.60 / np.sqrt(parameters.N)  # 1.50
    wIE = 1.20 / np.sqrt(parameters.N)  # 2.20
    wII = 4.50 / np.sqrt(parameters.N)  # 4.50
    wSE = 0.5*0.03 * np.sqrt(parameters.N)
    wSI = 0.5*0.12 * np.sqrt(parameters.N)

    if parameters.numClusters != 1:
        wEEsub = WRatio * wEE
        pEEsub = REE * parameters.pEE
        pEE = parameters.pEE * (parameters.numClusters - REE) / (parameters.numClusters - 1)
        wEE = wEE * (parameters.numClusters - WRatio) / (parameters.numClusters - 1)  # Average weight for sub - clusters

        wIEsub = WRatio * wIE
        wIE = wIE * (parameters.numClusters - WRatio) / (parameters.numClusters - 1)  # Average weight for sub - clusters
        pIEsub = REE * parameters.pXX
        pIE = parameters.pXX * (parameters.numClusters - REE) / (parameters.numClusters - 1)
    else:
        wEEsub = wEE
        wIEsub = wIE
        pIE = parameters.pXX
        pEE = parameters.pEE
        pEE = pEEsub
        pIEsub = pIE

    weightsEI = np.random.binomial(1, parameters.pXX, (parameters.NE, parameters.NI))      # Weight matrix of inhibitory to excitatory LIF cells
    weightsEI = wEI * weightsEI

    weightsIE = np.random.binomial(1, parameters.pXX, (parameters.NI, parameters.NE))     # Weight matrix of excitatory to inhibitory cells
    weightsIE = wIE * weightsIE

    weightsII = np.random.binomial(1, pIE, (parameters.NI, parameters.NI))     # Weight matrix of inhibitory to inhibitory cells
    weightsII = wII * weightsII

    weightsEE = np.random.binomial(1, pEE, (parameters.NE, parameters.NE))     # Weight matrix of excitatory to excitatory cells
    weightsEE = wEE * weightsEE

    # Create the group weight matrices and update the total weight matrix
    for i in range(parameters.numClusters):
        weightsEEsub = np.random.binomial(1, pEEsub, (parameters.EClusterSize, parameters.EClusterSize))
        weightsEEsub = wEEsub * weightsEEsub
        weightsEE[i * parameters.EClusterSize:(i+1) * parameters.EClusterSize, i * parameters.EClusterSize:(i+1) * parameters.EClusterSize] = weightsEEsub

    # # Create the group weight matrices for Exc to Inh and update the total weight matrix
    for i in range(parameters.numClusters):
        weightsIEsub = np.random.binomial(1, pIEsub, (parameters.IClusterSize, parameters.EClusterSize))
        weightsIEsub = wIEsub * weightsIEsub
        weightsIE[i * parameters.IClusterSize:(i + 1) * parameters.IClusterSize, i * parameters.EClusterSize:(i + 1) * parameters.EClusterSize] = weightsIEsub

    # Ensure the diagonals are zero
    np.fill_diagonal(weightsII, parameters.nI * wSI)
    np.fill_diagonal(weightsEE, parameters.nE * wSE)

    W = np.zeros((parameters.N, parameters.N))
    W[:parameters.NE, :parameters.NE] = weightsEE
    W[parameters.NE:, parameters.NE:] = -weightsII
    W[parameters.NE:, :parameters.NE] = weightsIE
    W[:parameters.NE, parameters.NE:] = -weightsEI

    return W

def get_weight_matrix_ExcInh_Cluster(parameters):

    WRatioE = parameters.WRatioE  # Ratio of Win / Wout(synaptic weight of within group to neurons outside the group)
    WRatioI = 1 + parameters.R * (WRatioE - 1)
    REE = parameters.REE
    RII = 1 + parameters.R * (REE - 1)
    scale = 1.0

    wEE = scale * 1.40 / np.sqrt(parameters.N)  # 1.68 1.72 # 1.40
    wEI = scale * 2.70 / np.sqrt(parameters.N)  # 1.31 1.52 # 2.70
    wIE = scale * 1.10 / np.sqrt(parameters.N)  # 2.20 2.20 # 1.15
    wII = scale * 4.70 / np.sqrt(parameters.N)  # 3.80 4.50 # 3.80
    # wSE = 0.011 * np.sqrt(parameters.N)
    # wSI = 4 * 0.011 * np.sqrt(parameters.N)

    if parameters.numClusters != 1:
        wEEsub = WRatioE * wEE
        pEEsub = REE * parameters.pEE
        wEE = wEE * (parameters.numClusters - WRatioE) / (parameters.numClusters - 1)  # Average weight for sub - clusters
        pEE = parameters.pEE * (parameters.numClusters - REE) / (parameters.numClusters - 1)

        wIEsub = WRatioE * wIE
        pIEsub = REE * parameters.pXX
        wIE = wIE * (parameters.numClusters - WRatioE) / (parameters.numClusters - 1)  # Average weight for sub - clusters
        pIE = parameters.pXX * (parameters.numClusters - REE) / (parameters.numClusters - 1)

        wEIsub = WRatioI * wEI
        pEIsub = RII * parameters.pXX
        wEI = wEI * (parameters.numClusters - WRatioI) / (parameters.numClusters - 1)  # Average weight for sub - clusters
        pEI = parameters.pXX * (parameters.numClusters - RII) / (parameters.numClusters - 1)

        wIIsub = WRatioI * wII
        pIIsub = RII * parameters.pXX
        wII = wII * (parameters.numClusters - WRatioI) / (parameters.numClusters - 1)  # Average weight for sub - clusters
        pII = parameters.pXX * (parameters.numClusters - RII) / (parameters.numClusters - 1)
    else:
        wEEsub = wEE
        wIEsub = wIE
        wEIsub = wEI
        wIIsub = wII

    weightsEI = np.random.binomial(1, pEI, (parameters.NE, parameters.NI))         # Weight matrix of inhibitory to single compartment excitatory LIF units
    weightsEI = wEI * weightsEI

    weightsIE = np.random.binomial(1, pIE, (parameters.NI, parameters.NE))         # Weight matrix of excitatory to inhibitory cells
    weightsIE = wIE * weightsIE

    weightsII = np.random.binomial(1, pII, (parameters.NI, parameters.NI))         # Weight matrix of inhibitory to inhibitory cells
    weightsII = wII * weightsII

    weightsEE = np.random.binomial(1, pEE, (parameters.NE, parameters.NE))         # Weight matrix of excitatory to excitatory cells
    weightsEE = wEE * weightsEE

    # Create the group weight matrices and update the total weight matrix
    for i in range(parameters.numClusters):
        weightsEEsub = np.random.binomial(1, pEEsub, (parameters.EClusterSize, parameters.EClusterSize))
        weightsEEsub = wEEsub * weightsEEsub
        weightsEE[i * parameters.EClusterSize:(i + 1) * parameters.EClusterSize, i * parameters.EClusterSize:(i + 1) * parameters.EClusterSize] = weightsEEsub

    # Create the group weight matrices for Exc to Inh and update the total weight matrix
    for i in range(parameters.numClusters):
        weightsIEsub = np.random.binomial(1, pIEsub, (parameters.IClusterSize, parameters.EClusterSize))
        weightsIEsub = wIEsub * weightsIEsub
        weightsIE[i * parameters.IClusterSize:(i+1) * parameters.IClusterSize, i * parameters.EClusterSize:(i+1) * parameters.EClusterSize] = weightsIEsub

    # Create the group weight matrices for Inh to Exc and update the total weight matrix
    for i in range(parameters.numClusters):
        weightsEIsub = np.random.binomial(1, pEIsub, (parameters.EClusterSize, parameters.IClusterSize))
        weightsEIsub = wEIsub * weightsEIsub
        weightsEI[i * parameters.EClusterSize:(i+1) * parameters.EClusterSize, i * parameters.IClusterSize:(i+1) * parameters.IClusterSize] = weightsEIsub

    # Create the group weight matrices and update the total weight matrix
    for i in range(parameters.numClusters):
        weightsIIsub = np.random.binomial(1, pIIsub, (parameters.IClusterSize, parameters.IClusterSize))
        weightsIIsub = wIIsub * weightsIIsub
        weightsII[i * parameters.IClusterSize:(i + 1) * parameters.IClusterSize, i * parameters.IClusterSize:(i + 1) * parameters.IClusterSize] = weightsIIsub

    # Ensure the diagonals are zero
    # np.fill_diagonal(weightsII, parameters.nI * wSI)
    # np.fill_diagonal(weightsEE, parameters.nE * -wSE)

    np.fill_diagonal(weightsII, 0.0 * wII)
    np.fill_diagonal(weightsEE, -0.0 * wII)

    W = np.zeros((parameters.N, parameters.N))
    W[:parameters.NE, :parameters.NE] = weightsEE
    W[parameters.NE:, parameters.NE:] = -weightsII
    W[parameters.NE:, :parameters.NE] = weightsIE
    W[:parameters.NE, parameters.NE:] = -weightsEI

    return W

def get_weight_matrix_Exc(parameters):

    wEE = 0.01 / np.sqrt(parameters.N)

    W = np.random.normal(wEE, 0.1 * wEE, (parameters.N, parameters.N))  # Weight matrix of excitatory to excitatory cells
    # W = W * np.random.binomial(1, parameters.pEE, (parameters.N, parameters.N))

    np.fill_diagonal(W, -24.0*wEE)

    return W

def get_weight_matrix_Exc_N(parameters):

    N = parameters.N
    
    pEE  = parameters.pEE
    
    wEE = parameters.wEE / (pEE * N)

    W = wEE * np.random.binomial(1, pEE, (N, N))

    np.fill_diagonal(W, 0.)

    return W

def get_weight_matrix_Exc_Inh_N(parameters):

    N = parameters.N
    NE = parameters.NE
    NI = parameters.NI
    
    pEE  = parameters.pEE
    pEI  = parameters.pEI
    pIE  = parameters.pIE
    pII  = parameters.pII

    # Change this in the transiton rate paper to NE/NI (cocnsistent with the original paper)
    wEE = parameters.wEE / (pEE * NE)
    wEI = parameters.wEI / (pEI * NI)
    wIE = parameters.wIE / (pIE * NE)
    wII = parameters.wII / (pII * NI)

    weightsEE = wEE * np.random.binomial(1, pEE, (NE, NE))
    weightsEI = wEI * np.random.binomial(1, pEI, (NE, NI))
    weightsIE = wIE * np.random.binomial(1, pIE, (NI, NE))
    weightsII = wII * np.random.binomial(1, pII, (NI, NI))

    W = np.zeros((parameters.N, parameters.N))
    W[:NE, :NE] = weightsEE
    W[NE:, NE:] = -weightsII
    W[NE:, :NE] = weightsIE
    W[:NE, NE:] = -weightsEI

    np.fill_diagonal(W, 0.)

    return W
