from scipy.spatial.distance import pdist,squareform
import numpy as np
import time
import pandas as pd
import numpy.matlib
from math import exp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score,precision_score,auc
from sklearn.metrics import accuracy_score,recall_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC,LinearSVC
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class fps_analysis(object):
    def __init__(self):
        pass

    def paramspec(self, data):
        Buffersize = 1000
        PreStd = []
        P_Summary = []
        PFS = []
        T = round(np.shape(data)[0] / Buffersize)
        return Buffersize, P_Summary, T, PFS, PreStd

    def distance_cal(self, data):
        D = pdist(data)
        Dist = squareform(D)
        return Dist

    def fitness_cal(self, sample, pop, stdData, gamma):
        Ns = np.shape(sample)[0]
        Np = np.shape(pop)[0]
        Newsample = np.concatenate([sample, pop])
        Dist = self.distance_cal(Newsample)
        fitness = []
        for i in range(Np):
            # distArray = np.power(Dist[i + Ns, 0:Ns], 2)
            # distArray = np.power(Dist[i + Ns, 0:Ns], 1)
            distArray = Dist[i + Ns, 0:Ns]
            temp = np.power(np.exp(-distArray / stdData), gamma)
            fitness.append(np.sum(temp))
        return fitness

    def fitness_update(self, P_Summary, Current, fitness, PreStd, gamma, stdData):
        [N, dim] = np.shape(Current)
        t_I = len(PreStd)
        NewFit = fitness
        if len(P_Summary) > 0:
            PreFit = P_Summary[:, dim]
            PreP = P_Summary[:, 0:dim]
            OldStd = PreStd[t_I - 1]
            for i in range(N):
                fitin = 0
                for j in range(np.shape(PreP)[0]):
                    if np.linalg.norm(Current[i][:] - PreP[j][:]) < 0.01:
                        fitin = PreFit[j]
                        break
                    else:
                        d = np.linalg.norm(Current[i][:] - PreP[j][:])
                        # fitin += (exp(-d ** 2 / stdData) ** gamma) * (PreFit[j] ** (OldStd / stdData))
                        fitin += (exp(-d / stdData) ** gamma) * (PreFit[j] ** (OldStd / stdData))
                NewFit[i] = fitness[i] + fitin
        return NewFit

    def popinitial(self, sample, PreMu, PreStd, Buffersize):
        [N, L] = np.shape(sample)
        pop_Size = round(1 * N)
        # Compute the statistics of the current data chunk
        minLimit = np.min(sample, axis=0)
        meanData = np.mean(sample, axis=0)
        maxLimit = np.max(sample, axis=0)
        # Update the statistics of the data stream
        meanData = self.updatemean(PreMu, meanData, Buffersize)
        PreMu.append(meanData)
        # Compute the standard deviation of the current data chunk
        MD = np.matlib.repmat(meanData, N, 1)
        tempSum = np.sum(np.sum((MD - sample) ** 2, axis=1))
        stdData = tempSum / N
        stdData = stdData**0.5
        # Update the standard deviation of the data stream
        stdData = self.stdupdate(stdData, PreStd, Buffersize)
        # Randonmly Initialize the population indices from the data chunk
        pop_Index = np.arange(0, N)
        pop = sample[pop_Index, :]
        # Calculate the initial niche radius
        radius = numpy.linalg.norm((maxLimit - minLimit)) * 0.4  # 0.6

        return [stdData, pop_Index, pop, radius, PreMu, PreStd]

    def updatemean(self, PreMu, meanData, BufferSize):
        # Num of the processed data chunk
        t_P = len(PreMu)
        # Update the mean of the data stream as new data chunk arrives
        if t_P == 0:
            newMu = meanData
        else:
            oldMu = PreMu[t_P - 1][:]
            newMu = (meanData + oldMu * t_P) / (t_P + 1)
        return newMu

    def stdupdate(self, Std, PreStd, BufferSize):
        # Num of the processed data chunk
        t_P = len(PreStd)
        # Update the variance of the data stream as new data chunk arrives
        if t_P == 0:
            newStd = Std
        else:
            oldStd = PreStd[t_P - 1]
            newStd = (Std + oldStd * t_P) / (t_P + 1)
        return newStd

    # ------------------------Parameter Estimation----------------------------#
    def cca(self, sample, stdData, Dist):
        m = 1
        gamma = 5
        ep = 0.998
        N = np.shape(sample)[0]
        while 1:
            den1 = []
            den2 = []
            for i in range(N - 1):
                # Diff = np.power(Dist[i, :], 2)
                # Diff = np.power(Dist[i, :], 1)
                Diff = Dist[i, :]
                temp1 = np.power(np.exp(-Diff / stdData), gamma * m)
                temp2 = np.power(np.exp(-Diff / stdData), gamma * (m + 1))
                den1.append(np.sum(temp1))
                den2.append(np.sum(temp2))
            y = np.corrcoef(den1, den2)[0, 1]
            if y > ep:
                break
            m = m + 1
        return m * gamma

    def dcca(self, sample, stdData, P_Summary, gamma, dim):
        P_Center = P_Summary[:, 0:dim]
        P_F = P_Summary[:, dim]
        gam1 = gamma
        N1 = np.shape(sample)[0]
        N2 = np.shape(P_Center)[0]
        ep = 0.998
        N = N1 + N2
        temp = np.concatenate([sample, P_Center], axis=0)
        Dist = self.distance_cal(temp)
        while 1:
            gam2 = gam1 + 5
            den1 = []
            den2 = []
            for i in range(N):
                # Diff = np.power(Dist[i, 0:N1], 2)
                # Diff = np.power(Dist[i, 0:N1], 1)
                Diff = Dist[i, 0:N1]
                temp1 = np.power(np.exp(-Diff / stdData), gam1)
                temp2 = np.power(np.exp(-Diff / stdData), gam2)
                sum1 = np.sum(temp1)
                sum2 = np.sum(temp2)
                if i < N1:
                    T1 = 0
                    T2 = 0
                    for j in range(N2):
                        T1 += P_F[j] ** (gam1 / gamma)
                        T2 += P_F[j] ** (gam2 / gamma)
                    s1 = sum1 + T1
                    s2 = sum2 + T2
                else:
                    s1 = sum1 + P_F[i - N1] ** (gam1 / gamma)
                    s2 = sum2 + P_F[i - N1] ** (gam2 / gamma)
                den1.append(s1)
                den2.append(s2)
            y = np.corrcoef(den1, den2)[0, 1]
            if y > ep:
                break
            gam1 = gam2
        return gam1

    def tpc_search(self, Dist, Pop_Index, Pop, radius, fitness):
        # Extract the size of the population
        [N, dim] = np.shape(Pop)
        P = []  # Initialize the Peak Vector
        P_fitness = []
        # i = 1
        marked = []
        co = []
        OriginalIndice = Pop_Index
        while 1:
            # -------------Search for the local maximum-----------------#
            SortIndice = np.argsort(fitness)
            NewIndice = SortIndice[::-1]

            Pop = Pop[NewIndice, :]
            fitness = fitness[NewIndice]
            OriginalIndice = OriginalIndice[NewIndice]

            P.append(Pop[0, :])

            P_fitness.append(fitness[0])
            P_Indice = OriginalIndice[0]

            Ind = self.assigntopeaks(Pop, Pop_Index, P, P_Indice, marked, radius, Dist)

            marked.append(Ind)
            marked.append(NewIndice[0])

            if not Ind:
                Ind = [NewIndice[0]]

            co.append(len(Ind))
            TempFit = fitness
            sum1 = 0
            for j in range(len(Ind)):
                sum1 += fitness[np.where(OriginalIndice == Ind[j])]
            for th in range(len(Ind)):
                TempFit[np.where(OriginalIndice == Ind[th])] = fitness[np.where(OriginalIndice == Ind[th])] / sum1
            fitness = TempFit
            if np.sum(co) >= N:
                P = np.asarray(P)
                P_fitness = np.asarray(P_fitness)
                break

        return P, P_fitness

    def mergeinchunk(self, P, P_fitness, sample, gamma, stdData):
        """Perform the Merge of TPCs witnin each data chunk
        """
        # Num of TPCs
        [Nc, dim] = np.shape(P)
        NewP = []
        NewP_fitness = []
        marked = []
        unmarked = []
        Com = []

        # Num of TPCs
        Nc = np.shape(P)[0]
        for i in range(Nc):
            MinDist = np.inf
            MinIndice = 100000
            if i not in marked:
                for j in range(Nc):
                    if j != i and j not in marked:
                        d = np.linalg.norm(P[j, :] - P[i, :])
                        if d < MinDist:
                            MinDist = d
                            MinIndice = j
                if MinIndice <= Nc:
                    MinIndice = int(MinIndice)
                    Merge = True
                    Neighbor = P[MinIndice][:]
                    X = (Neighbor + P[i, :]) / 2
                    X = np.reshape(X, (1, np.shape(P)[1]))
                    fitX = self.fitness_cal(sample, X, stdData, gamma)
                    fitP = P_fitness[i]
                    fitN = P_fitness[MinIndice]
                    if fitX < 0.85 * min(fitN, fitP):
                        Merge = False
                    if Merge:
                        Com.append([i, MinIndice])
                        marked.append(MinIndice)
                        marked.append(i)
                    else:
                        unmarked.append(i)
        Com = np.asarray(Com)
        # Number of Possible Merges:
        Nm = np.shape(Com)[0]
        for k in range(Nm):
            if P_fitness[Com[k, 0]] >= P_fitness[Com[k, 1]]:
                NewP.append(P[Com[k, 0], :])
                NewP_fitness.append(P_fitness[Com[k, 0]])
            else:
                NewP.append(P[Com[k, 1], :])
                NewP_fitness.append(P_fitness[Com[k, 1]])
        # Add Unmerged TPCs to the NewP
        for n in range(Nc):
            if n not in Com:
                NewP.append(P[n, :])
                NewP_fitness.append(P_fitness[n])
        NewP = np.asarray(NewP)
        NewP_fitness = np.asarray(NewP_fitness)
        return NewP, NewP_fitness

    def mergeonline(self, P, P_fitness, P_summary, PreStd, sample, gamma, stdData):
        """Perform the Merge of Clusters Between Historical and New Clusters
        """
        # Num of TPCs
        [Nc, dim] = np.shape(P)
        NewP = []
        NewP_fitness = []
        marked = []
        unmarked = []
        Com = []

        for i in range(Nc):
            MinDist = np.inf
            MinIndice = 100000
            if i not in marked:
                for j in range(Nc):
                    if j != i and j not in marked:
                        d = np.linalg.norm(P[j, :] - P[i, :])
                        if d < MinDist:
                            MinDist = d
                            MinIndice = j
                if MinIndice < Nc:
                    Merge = True
                    Neighbor = P[MinIndice][:]
                    X = (Neighbor + P[i][:]) / 2
                    X = np.reshape(X, (1, np.shape(P)[1]))
                    RfitX = self.fitness_cal(sample, X, stdData, gamma)
                    fitX = self.fitness_update(P_summary, X, RfitX, PreStd, gamma, stdData)
                    fitP = P_fitness[i]
                    fitN = P_fitness[MinIndice]
                    if fitX < 0.85 * min(fitN, fitP):
                        Merge = False
                    if Merge:
                        Com.append([i, MinIndice])
                        marked.append(MinIndice)
                        marked.append(i)
                    else:
                        unmarked.append(i)
        Com = np.asarray(Com)
        # Number of Possible Merges:
        Nm = np.shape(Com)[0]
        for k in range(Nm):
            if P_fitness[Com[k, 0]] >= P_fitness[Com[k, 1]]:
                NewP.append(P[Com[k, 0]][:])
                NewP_fitness.append(P_fitness[Com[k, 0]])
            else:
                NewP.append(P[Com[k, 1]][:])
                NewP_fitness.append(P_fitness[Com[k, 1]])
        # Add Unmerged TPCs to the NewP
        for n in range(Nc):
            if n not in Com:
                NewP.append(P[n][:])
                NewP_fitness.append(P_fitness[n])
        NewP = np.asarray(NewP)
        NewP_fitness = np.asarray(NewP_fitness)
        return NewP, NewP_fitness

    def ce_inchunk(self, sample, P, P_fitness, stdData, gamma):
        while 1:
            HistP = P
            #        HistPF = P_fitness
            P, P_fitness = self.mergeinchunk(P, P_fitness, sample, gamma, stdData)
            if np.shape(P)[0] == np.shape(HistP)[0]:
                break
        return P, P_fitness

    def ce_online(self, sample, P_Summary, P, P_fitness, stdData, gamma, PreStd):
        dim = np.shape(P)[1]

        # Concatenate the historical and new clusters together
        PC = np.concatenate([P_Summary[:, 0:dim], P])
        RPF = self.fitness_cal(sample, PC, stdData, gamma)
        PF = self.fitness_update(P_Summary, PC, RPF, PreStd, gamma, stdData)

        while 1:
            HistPC = PC
            #        HistPF = PF
            PC, PF = self.mergeonline(PC, PF, P_Summary, PreStd, sample, gamma, stdData)
            RPF = self.fitness_cal(sample, PC, stdData, gamma)
            PF = self.fitness_update(P_Summary, PC, RPF, PreStd, gamma, stdData)
            if np.shape(PC)[0] == np.shape(HistPC)[0]:
                break
        return PC, PF

    def clustervalidation(self, sample, P):
        while 1:
            NewP = []
            PreP = P
            [R_d, RIndice] = self.cluster_assign(sample, P)

            for i in range(np.shape(P)[0]):
                Temp = np.where(RIndice == i)
                Temp = np.asarray(Temp)
                if np.shape(Temp)[1] > 2:
                    NewP.append(P[i][:])
            P = NewP
            if np.shape(P)[0] == np.shape(PreP)[0]:
                break
        return np.asarray(P)

    def clustersummary(self, P, PF, P_Summary, sample):
        dim = np.shape(sample)[1]
        Rp = self.averagedist(P, P_Summary, sample, dim)
        P = np.asarray(P)
        PF = [PF]
        PF = np.asarray(PF)
        Rp = np.reshape(Rp, (np.shape(P)[0], 1))
        PCluster = np.concatenate([P, PF.T], axis=1)
        PCluster = np.concatenate([PCluster, Rp], axis=1)
        P_Summary = PCluster
        return P_Summary

    def storeinf(self, PF, PFS, PreStd, stdData):
        PreStd.append(stdData)
        PFS.append(PF)
        return PreStd, PFS

    # --------------------Cluster Radius Computation and Update--------------------#
    def averagedist(self, P, P_Summary, sample, dim):
        P = P
        # Obtain the assignment of clusters
        [distance, indices] = self.cluster_assign(sample, P)
        rad1 = []
        # if the summary of clusters is not empty
        if len(P_Summary) > 0:

            PreP = P_Summary[:, 0:dim]  # Hstorical Cluster Center vector
            PreR = P_Summary[:, dim + 1]
            for i in range(np.shape(P)[0]):
                if np.shape(np.where(indices == i))[1] > 1:
                    SumD1 = 0
                    Count1 = 0
                    for j in range(np.shape(sample)[0]):
                        if indices[j] == i:
                            SumD1 += distance[j]
                            Count1 += 1
                    rad1.append(SumD1 / Count1)
                else:
                    C_d = []
                    for k in range(np.shape(PreP)[0]):
                        C_d.append(np.linalg.norm(P[i][:] - PreP[k][:]))
                    CI = np.argmin(C_d)
                    rad1.append(PreR[CI])
        elif not P_Summary:
            for i in range(np.shape(P)[0]):
                SumD1 = 0
                Count1 = 0
                for j in range(np.shape(sample)[0]):
                    if indices[j] == i:
                        SumD1 += distance[j]
                        Count1 += 1
                rad1.append(SumD1 / Count1)
        return np.asarray(rad1)

    def assigntopeaks(self, pop, pop_index, P, P_I, marked, radius, Dist):
        temp = []
        [N, L] = np.shape(pop)
        for i in range(N):
            distance = Dist[i, P_I]
            # marked = np.array(marked)
            if not np.isin(marked, pop_index[i]):
                if distance < radius:
                    temp.append(pop_index[i])
        indices = temp
        return indices

    def cluster_assign(self, sample, P):
        # Number of samples
        N = np.shape(sample)[0]
        # Number of Clusters at t
        Np = np.shape(P)[0]
        MinDist = []
        MinIndice = []
        for i in range(N):
            d = []
            for j in range(Np):
                d.append(np.linalg.norm(sample[i][:] - P[j][:]))
            if len(d) <= 1:
                tempD = d
                tempI = 0
            else:
                tempD = np.min(d)
                tempI = np.argmin(d)
            MinDist.append(tempD)
            MinIndice.append(tempI)
        MinDist = np.asarray(MinDist)
        MinIndice = np.asarray(MinIndice)
        return MinDist, MinIndice

    def compute_radius(self, MinDist, ClusterIndice):
        cluster = np.unique(ClusterIndice)
        nc = len(cluster)
        cluster_rad = []
        for i in range(nc):
            currentcluster = np.where(ClusterIndice == cluster[i])[0]
            cluster_rad.append(np.mean(MinDist[currentcluster]))
        return cluster_rad

    def predict(self, data):
        dim = np.shape(data)[1]
        [BufferSize, P_Summary, T, PFS, PreStd] = self.paramspec(data)
        T = int(T)
        gammaHist = []
        PFS = []
        PreMu = []

        for t in range(T):
            if t < T - 1:
                sample = data[t * BufferSize:(t + 1) * BufferSize, :]
            else:
                sample = data[t * BufferSize:np.shape(data)[0]]
            if t == 0:
                AccSample = sample
            else:
                AccSample = np.concatenate([AccSample, sample])

            [stdData, pop_index, pop, radius, PreMu, PreStd] = self.popinitial(sample, PreMu, PreStd, BufferSize)
            # Initialize the fitness vector
            fitness = np.zeros((len(pop_index), 1))
            # Initialize the indices vector
            indices = np.zeros((len(pop_index), 1))
            Dist = self.distance_cal(sample)
            # print(Dist)
            if PreStd:
                if PreStd[len(PreStd) - 1] > stdData:
                    P = P_Summary[:, 0:dim]
                    localFit = self.fitness_cal(sample, P, stdData, gamma)
                    PF = self.fitness_update(P_Summary, P, localFit, PreStd, gamma, stdData)
                    P_Summary = self.clustersummary(P, PF, P_Summary, sample)
                    PFS.append(PF)
                    PreStd.append(stdData)
                    clustercenter = P
                    [Assign, clusterindex] = self.cluster_assign(AccSample, P)
                    continue
            else:
                gamma = self.cca(sample, stdData, Dist)
            gammaHist.append(gamma)
            fitness = self.fitness_cal(sample, pop, stdData, gamma)
            fitness = np.array(fitness)
            P, P_fitness = self.tpc_search(Dist, pop_index, pop, radius, fitness)
            P, P_fitness = self.ce_inchunk(sample, P, P_fitness, stdData, gamma)
            P_fitness = self.fitness_cal(sample, P, stdData, gamma)
            P_fitness = self.fitness_update(P_Summary, P, P_fitness, PreStd, gamma, stdData)
            # print('Processing Data Chunk ' + str(t))
            if t == 0:
                P = P
                PF = np.asarray(P_fitness)
            else:
                P, P_fitness = self.ce_online(sample, P_Summary, P, P_fitness, stdData, gamma, PreStd)
                PF = np.asarray(P_fitness)
            P_Summary = self.clustersummary(P, PF, P_Summary, sample)
            PreStd, PFS = self.storeinf(PF, PFS, PreStd, stdData)
        # Clustering procedure finishes
        [MinDist, ClusterIndice] = self.cluster_assign(AccSample, P)
        return P, ClusterIndice, MinDist
