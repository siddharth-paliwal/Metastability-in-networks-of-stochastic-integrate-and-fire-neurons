import numpy as np
from phi import *
from utility_functions import *

def simulate_network_EI(parameters, W):

    # Unpackage Parameters
    dt = parameters.dt
    gain = parameters.gain
    N = parameters.N
    Etr = parameters.Etr
    trans = parameters.trans
    power = parameters.power
    thres = parameters.Vthres
    Nt = parameters.Nt
    simPhi = parameters.simPhi
    VReset = parameters.Vreset

    # Simulation variables
    t = 0
    numSpikes = 0
    maxSpikes = parameters.maxSpikes  # 500 Hz / neuron
    spkTimes = np.zeros((int(maxSpikes), 2))  # store spike times and neuron labels
    memVol = np.zeros((Nt, parameters.N))
    spkTrain = np.zeros((Nt, N))
    gSyn = np.zeros((Nt, parameters.N))

    # Initialization
    V = np.random.uniform(0, 2, size=(N,))
    noise = np.concatenate( (np.ones((parameters.NE, )) * parameters.IeE, np.ones((parameters.NI, )) * parameters.IeI) )
    tauMem = np.concatenate(
        (1 / parameters.tauE * np.ones(int(parameters.NE), ), 1 / parameters.tauI * np.ones(int(parameters.NI), )),
        axis=0)
    spks = np.zeros((N, ))

    count = 0

    for i in range(0, Nt - 1, 1):

        t += dt
        V += tauMem * ( dt * (-V + noise) + np.dot(W, spks) ) - V * spks

        # Refractory Period
        # for j in range(N):
        #     if (lastAP[j] + Etr / dt) >= (i + 1):
        #         memVol[i + 1, j] = 0

        # Decide if each neuron spikes, update synaptic output of spiking neurons each neuron's rate is phi(g)
        if simPhi ==0:
            r = threshold_power_law(V, gain, power)
        elif simPhi == 1:
            r = exponential(V, gain, thres)
        
        r[r > 1/dt] = 1/dt
            
        try:
            # spkTrain[i + 1, :] = np.random.poisson(r * dt, size=(N,))
            spks = np.random.binomial(n=1, p=dt*r)
        except:
            break

        count += sum(1 * (spks > 1))  # Count to check how many >1 spike incidents occur

        if t > trans:
            for j in range(N):
                if spks[j] >= 1 and numSpikes < maxSpikes:
                    spkTimes[numSpikes, 0] = t  # Save time at which spike occurred
                    spkTimes[numSpikes, 1] = j  # Save the neuron which spiked

                    numSpikes += 1

        memVol[i+1, :] = V.copy()
        spkTrain[i+1,:] = spks.copy()
        gSyn[i+1,:] = noise.copy()

    return spkTrain, memVol, gSyn, spkTimes, count


def simulate_network_Homogeneous(parameters, W):

    # Unpackage Parameters
    dt = parameters.dt
    gain = parameters.gain
    N = parameters.N
    Etr = parameters.Etr
    trans = parameters.trans
    power = parameters.power
    thres = parameters.Vthres
    Nt = parameters.Nt
    simPhi = parameters.simPhi
    VReset = parameters.Vreset

    # Simulation variables
    t = 0
    numSpikes = 0
    maxSpikes = parameters.maxSpikes  # 500 Hz / neuron
    spkTimes = np.zeros((int(maxSpikes), 2))  # store spike times and neuron labels
    memVol = np.zeros((Nt, parameters.N))
    spkTrain = np.zeros((Nt, N))
    gSyn = np.zeros((Nt, parameters.N))

    # Initialization
    V = np.random.uniform(2, 4, size=(N,))
    noise = np.ones((N, )) * parameters.IeE
    tauMem = 1 / parameters.tauE * np.ones((N, ))
    spks = np.zeros((N, ))

    count = 0

    for i in range(0, Nt - 1, 1):

        t += dt

        if t <= trans:
            E = noise + parameters.stim
        else:
            E = noise

        V += tauMem * ( dt * (-V + E) + np.dot(W, spks) ) - V * spks

        # Refractory Period
        # for j in range(N):
        #     if (lastAP[j] + Etr / dt) >= (i + 1):
        #         memVol[i + 1, j] = 0

        # Decide if each neuron spikes, update synaptic output of spiking neurons each neuron's rate is phi(g)
        if simPhi ==0:
            r = threshold_power_law(V, gain, power)
        elif simPhi == 1:
            r = exponential(V, gain, thres)
        
        r[r > 1/dt] = 1/dt
            
        try:
            # spkTrain[i + 1, :] = np.random.poisson(r * dt, size=(N,))
            spks = np.random.binomial(n=1, p=dt*r)
        except:
            break

        count += sum(1 * (spks > 1))  # Count to check how many >1 spike incidents occur

        if t > trans:
            for j in range(N):
                if spks[j] >= 1 and numSpikes < maxSpikes:
                    spkTimes[numSpikes, 0] = t  # Save time at which spike occurred
                    spkTimes[numSpikes, 1] = j  # Save the neuron which spiked

                    numSpikes += 1

        memVol[i+1, :] = V.copy()
        spkTrain[i+1,:] = spks.copy()
        gSyn[i+1,:] = np.dot(W, spks)


    return spkTrain, memVol, gSyn, spkTimes, count


def sim_network_Perturb(parameters, W, t_start_perturb1, t_start_perturb2, t_end_perturb1, t_end_perturb2, perturb_amp):

    # Unpackage Parameters
    dt = parameters.dt
    gain = parameters.gain
    N = parameters.N
    Etr = parameters.Etr
    trans = parameters.trans
    power = parameters.power
    thres = parameters.Vthres
    Nt = parameters.Nt
    simPhi = parameters.simPhi
    VReset = parameters.Vreset

    # Simulation variables
    t = 0
    numSpikes = 0
    maxSpikes = parameters.maxSpikes  # 500 Hz / neuron
    spkTimes = np.zeros((int(maxSpikes), 2))  # store spike times and neuron labels
    memVol = np.zeros((Nt, N))
    gSyn = np.zeros((Nt, N))
    spkTrain = np.zeros((Nt, N))

    # Initialization
    V = np.random.uniform(0, 2, size=(N,))
    noise = np.ones((N, )) * parameters.IeE
    spks = np.zeros((N, ))
    tauMem = 1 / parameters.tauE * np.ones(int(parameters.N), )
    memVol[0,:] = V.copy()
    spkTrain[0,:] = spks.copy()
    gSyn[0,:] = noise.copy()

    count = 0

    for i in range(0, Nt - 1, 1):

        t += dt
        
        if (t >= t_start_perturb1) and (t < t_end_perturb1):
            E = noise + perturb_amp
        elif (t >= t_start_perturb2) and (t < t_end_perturb2):
            E = noise - perturb_amp
        else:
            E = noise
        
        V += tauMem * ( dt * (-V + E) + np.dot(W, spks) ) - V * spks

        # Refractory Period
        # for j in range(N):
        #     if (lastAP[j] + Etr / dt) >= (i + 1):
        #         memVol[i + 1, j] = 0

        # Decide if each neuron spikes, update synaptic output of spiking neurons each neuron's rate is phi(g)
        if simPhi == 0:
            r = threshold_power_law(V, gain, power)
        elif simPhi == 1:
            r = exponential(V, gain, thres)
        
        r[r > 1/dt] = 1/dt
            
        try:
            # spkTrain[i + 1, :] = np.random.poisson(r * dt, size=(N,))
            spks = np.random.binomial(n=1, p=dt*r)
        except:
            break

        idx = np.where(spkTrain[i + 1, :] >= 1)

        count += sum(1 * (spks > 1))  # Count to check how many >1 spike incidents occur

        if t > trans:
            for j in range(N):
                if spks[j] >= 1 and numSpikes < maxSpikes:
                    spkTimes[numSpikes, 0] = t  # Save time at which spike occurred
                    spkTimes[numSpikes, 1] = j  # Save the neuron which spiked

                    numSpikes += 1

        memVol[i+1, :] = V.copy()
        spkTrain[i+1,:] = spks.copy()
        gSyn[i+1,:] = E.copy()


    return spkTrain, memVol, gSyn, spkTimes, count





def simulate_network_MF(parameters, W):
    # Unpackage Parameters
    dt = parameters.dt
    gain = parameters.gain
    N = parameters.N
    trans = parameters.trans

    # Simulation variables
    Nt = parameters.Nt
    t = 0

    maxSpikes = parameters.maxSpikes  # 500 Hz / neuron
    spkTimes = np.zeros((int(maxSpikes), 2))  # store spike times and neuron labels

    # Simulate Entire Population
    # tauMem = np.concatenate(
    #     (1 / parameters.tauE * np.ones(int(parameters.NE), ), 1 / parameters.tauI * np.ones(int(parameters.NI), )),
    #     axis=0)
    # tauSyn = np.concatenate(
    #     ((1 / parameters.t_EE) * np.ones(int(parameters.NE), ), (1 / parameters.t_II) * np.ones(int(parameters.NI), )),
    #     axis=0)
    # memVol = 0.01 * np.random.uniform(0, 1, size=(N, Nt)).T  # why does this not start with random number?
    # memVolS = 0.01 * np.random.uniform(0, 1, size=(N, Nt)).T
    # gSyn = np.zeros((Nt, N))
    # spkTrain = np.zeros((Nt, N))

    noise = np.concatenate(
        (parameters.NE * parameters.pEE * parameters.IeE / np.sqrt(N) + parameters.IeE_var * np.random.uniform(
            size=(int(parameters.NE),)),
         parameters.NE * parameters.pEE * parameters.IeI / np.sqrt(N) + parameters.IeI_var * np.random.uniform(
             size=(int(parameters.NI),))),
        axis=0)

    # noise = np.concatenate(
    #     (np.random.normal(parameters.NE * parameters.pEE * parameters.IeE / np.sqrt(N), parameters.IeE_var,
    #                       size=(int(parameters.NE),)),
    #      np.random.normal(parameters.NE * parameters.pEE * parameters.IeI / np.sqrt(N), parameters.IeI_var,
    #                       size=(int(parameters.NI),))), axis=0)

    # Simulate Clusters
    tauMem = np.concatenate(
        (1 / parameters.tauE * np.ones(int(parameters.numClusters), ),
         1 / parameters.tauI * np.ones(int(parameters.numClusters), )),
        axis=0)
    tauSyn = np.concatenate(
        ((1 / parameters.t_EE) * np.ones(int(parameters.numClusters), ),
         (1 / parameters.t_II) * np.ones(int(parameters.numClusters), )),
        axis=0)
    memVol = 0.01 * np.random.uniform(0, 1, size=(2 * parameters.numClusters, Nt)).T  # why does this not start with random number?
    memVolS = 0.01 * np.random.uniform(0, 1, size=(2 * parameters.numClusters, Nt)).T
    gSyn = np.zeros((Nt, 2 * parameters.numClusters))
    spkTrain = np.zeros((Nt, 2 * parameters.numClusters))

    W_MF = get_W_MF(W, parameters)
    noise_MF = get_noise_MF(noise, parameters)
    count = 0

    for i in range(0, Nt - 1, 1):

        t += dt
        memVol[i + 1, :] = memVol[i, :] + dt * tauMem * (-memVol[i, :] + noise_MF) + dt * gSyn[i, :] - sigmoid(memVol[i + 1, :], gain) * (memVol[i, :] - parameters.jSelf)  # + np.sqrt(2 * 0.01 * dt) * np.random.normal(0, 1, size=(N, ))
        gSyn[i + 1, :] = gSyn[i, :] - (gSyn[i, :] * tauSyn) * dt + np.dot(W_MF, dt * sigmoid(memVol[i + 1, :], gain)) * tauSyn

        # memVol[i + 1, :] = memVol[i, :] + dt * tauMem * (-memVol[i, :] + noise) + np.dot(W, dt * sigmoid_MF(memVol[i, :], memVolS[i, :], gain))
        # memVolS[i + 1, :] = memVolS[i, :] + 2 * tauMem * np.exp(-2 * t / tauMem) + np.dot(W @ W, dt * sigmoid_MF(memVol[i, :], memVolS[i, :], gain))

    return spkTrain, memVol, memVolS, spkTimes, count


def simulate_network_deterministic(parameters, W):
    # Unpackage Parameters
    dt = parameters.dt
    gain = parameters.gain
    N = parameters.N
    Etr = parameters.Etr
    trans = parameters.trans
    Vthres = parameters.Vthres
    Vreset = parameters.Vreset

    # Simulation variables
    Nt = parameters.Nt
    t = 0
    numSpikes = 0

    maxSpikes = parameters.maxSpikes  # 500 Hz / neuron
    spkTimes = np.zeros((int(maxSpikes), 2))  # store spike times and neuron labels

    tauMem = np.concatenate(
        (1 / parameters.tauE * np.ones(int(parameters.NE), ), 1 / parameters.tauI * np.ones(int(parameters.NI), )),
        axis=0)
    tauSyn = np.concatenate(
        ((1 / parameters.t_EE) * np.ones(int(parameters.NE), ), (1 / parameters.t_II) * np.ones(int(parameters.NI), )),
        axis=0)
    memVol = 0.1 * np.random.uniform(0, 1, size=(parameters.N, Nt)).T  # why does this not start with random number?
    # memVol = np.zeros((parameters.N, Nt)).T  # why does this not start with random number?
    gSyn = np.zeros((parameters.N, Nt)).T
    spkTrain = np.zeros((Nt, parameters.N))

    count = 0

    # noise = np.concatenate(
    #     (parameters.NE * parameters.pEE * parameters.IeE / np.sqrt(N) + parameters.IeE_var * np.random.uniform(
    #         size=(int(parameters.NE),)),
    #      parameters.NE * parameters.pEE * parameters.IeI / np.sqrt(N) + parameters.IeI_var * np.random.uniform(
    #          size=(int(parameters.NI),))),
    #     axis=0)

    noise = np.concatenate(
        (np.random.normal(parameters.NE * parameters.pEE * parameters.IeE / np.sqrt(N), parameters.IeE_var,
                          size=(int(parameters.NE),)),
         np.random.normal(parameters.NE * parameters.pEE * parameters.IeI / np.sqrt(N), parameters.IeI_var,
                          size=(int(parameters.NI),))), axis=0)
    lastAP = -int(parameters.Etr / parameters.dt) * np.ones((parameters.N,))

    for i in range(0, Nt - 1, 1):

        t += dt
        memVol[i + 1, :] = memVol[i, :] + dt * tauMem * (-memVol[i, :] + noise) + dt * gSyn[i, :]

        for j in range(N):

            # Refractory Period
            if (lastAP[j] + Etr / dt) >= (i + 1):
                memVol[i + 1, j] = Vreset

            # Deterministic Network
            if memVol[i + 1, j] > Vthres:
                memVol[i + 1, j] = Vthres
                spkTrain[i + 1, j] = 1
                lastAP[j] = i + 1

        gSyn[i + 1, :] = gSyn[i, :] - (gSyn[i, :] * tauSyn) * dt + np.dot(W, spkTrain[i + 1, :].T) * tauSyn

        count += sum(1 * (spkTrain[i + 1, :] > 1))  # Count to check how many >1 spike incidents occur

        if t > trans:
            for j in range(N):
                if spkTrain[i + 1, j] >= 1 and numSpikes < maxSpikes:
                    spkTimes[numSpikes, 0] = t  # Save time at which spike occurred
                    spkTimes[numSpikes, 1] = j  # Save the neuron which spiked
                    numSpikes += 1

    return spkTrain, memVol, gSyn, spkTimes, count
