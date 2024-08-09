# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:53:01 2022

@author: Xuyang
"""

import torch
from math import exp
import numpy as np
from .strategy import Strategy
from .fps_clustering import fps_analysis
from sklearn.cluster import MiniBatchKMeans, KMeans

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

class DECALF(Strategy):
  def __init__(self, dataset, net, args_input, args_task):
    super(DECALF, self).__init__(dataset, net, args_input, args_task)

  def diversityfetch1(self, candidate_fet1, current, priority1, interd1, dth, fetchsize):
    fetch1 = []
    num_center = fetchsize
    chunked_dist1 = interd1[candidate_fet1]
    chunked_dist = chunked_dist1[:, candidate_fet1]
    for i in range(num_center):
      top_idx = np.argmax(priority1)
      fetch1.append(current[candidate_fet1[top_idx]])
      neighbordist = chunked_dist[top_idx, :]
      neighboridx = np.where(neighbordist <= dth)[0]
      priority1[top_idx] = priority1[top_idx] / (1 + 200 * np.sum(priority1[neighboridx]))
      priority1[neighboridx] = priority1[neighboridx] / (1 + 1 * np.sum(priority1[neighboridx]))
    fetch1 = np.asarray(fetch1)
    fetch1 = fetch1.astype(int)
    return fetch1

  def diversityfetch2(self, candidate_fet2, current, priority2, interd1, dth, fetchsize):
    fetch2 = []
    num_border = fetchsize
    chunked_dist1 = interd1[candidate_fet2]
    chunked_dist = chunked_dist1[:, candidate_fet2]
    for i in range(num_border):
      top_idx = np.argmax(priority2)
      fetch2.append(current[candidate_fet2[top_idx]])
      neighbordist = chunked_dist[top_idx][:]
      neighboridx = np.where(neighbordist <= dth)[0]
      priority2[top_idx] = priority2[top_idx] / (1 + 200 * np.sum(priority2[neighboridx]))
      priority2[neighboridx] = priority2[neighboridx] / (1 + 1 * np.sum(priority2[neighboridx]))
    fetch2 = np.asarray(fetch2)
    fetch2 = fetch2.astype(int)
    return fetch2

  def active_query(self, samples, cluster_centers, cluster_idx, label_budget):
    query_idx = []
    dist_cluster = squareform(pdist(cluster_centers))

    for i in range(np.shape(cluster_centers)[0]):
      curr_cluster = np.where(cluster_idx == i)[0]
      curr_dist = squareform(pdist(samples[curr_cluster]))
      num_queries = round(label_budget * len(curr_dist) / np.shape(samples)[0])
      num_nei = 3
      knei_dist, query_priority = [], []
      temp_interdist = dist_cluster[i, :]
      rng = np.random.default_rng()
      random_selection = rng.choice(curr_cluster, size=num_queries, replace=False)
      query_idx = np.append(query_idx, random_selection)
      # if len(curr_cluster) < 2:
      #   # query_idx = np.append(query_idx, curr_cluster)
      #   continue
      # temp_neigh1 = cluster_centers[np.argsort(temp_interdist)[0], :]
      # temp_neigh2 = cluster_centers[np.argsort(temp_interdist)[1], :]
      # for j in range(len(curr_cluster)):
      #   query_priority.append(1 + exp(-np.linalg.norm(samples[curr_cluster[j], :] - cluster_centers[i, :])**2))
      #   knei_dist.append(np.mean(np.sort(curr_dist[j, :])[1:num_nei+1]))
      # sortIndex1 = np.argsort(query_priority)
      # sortIndex1 = sortIndex1[::-1]
      # dth = 0.01*np.mean(knei_dist) # 0.0001
      # query_priority = np.array(query_priority)
      # # fet1 = curr_cluster[sortIndex1[:round(num_queries * 0.5)]]
      # fet1 = self.diversityfetch1(sortIndex1[:round(len(query_priority) / 2)],
      #                             curr_cluster,
      #                             query_priority[sortIndex1[:round(len(query_priority) / 2)]],
      #                             curr_dist, dth, round(num_queries * 0.5)) # 0.5
      # fil_index = sortIndex1[round(num_queries * 0.5):]
      # d2 = []
      # inter_dist = squareform(pdist(cluster_centers))
      # center_priority = []
      # for i_2 in range(np.shape(cluster_centers)[0]):
      #   center_priority.append(np.sum(1+np.exp(-inter_dist[i_2,:])))
      # center_priority = np.array(center_priority)
      # global_center = cluster_centers[np.argmax(center_priority)]
      # temp_neigh1 = global_center
      # temp_neigh2 = cluster_centers[np.argsort(temp_interdist)[1],:]
      # for k in range(len(fil_index)):
      #   temp_d1 = np.linalg.norm(samples[curr_cluster[fil_index[k]], :] - temp_neigh1)
      #   temp_d2 = np.linalg.norm(samples[curr_cluster[fil_index[k]], :] - temp_neigh2)
      #   # d2.append(1 + exp(-abs((temp_d1 + temp_d2)/2 - np.linalg.norm(temp_neigh1 - temp_neigh2)/2)))
        
      #   temp_sum1 = (temp_d1 + temp_d2)/np.linalg.norm(temp_neigh1 - temp_neigh2)
      #   temp_ratio1 = max(temp_d1, temp_d2) / min(temp_d1, temp_d2)
      #   bi_criteria = temp_sum1 * temp_ratio1
      #   d2.append(1 + exp(-bi_criteria))
      #   # d2.append(temp_ratio1)
      # d2 = np.array(d2)
      # sortIndex2 = np.argsort(d2)
      # sortIndex2 = sortIndex2[::-1]
      # # fet2 = curr_cluster[fil_index[sortIndex2[:round(num_queries * 0.5)]]]
      # fet2 = self.diversityfetch2(fil_index, curr_cluster,
      #                             d2, curr_dist, dth,
      #                             round(num_queries * 0.5))
      # # sortIndex2 = np.argsort(d2)
      # # candidate_fet2 = fil_index[sortIndex2[:int(round(num_queries * 1))]] # 0.8
      # # candidate_fet2 = fil_index
      # # sum_dist = []
      # # for ii in range(len(candidate_fet2)):
      # #   candidate_d1 = np.linalg.norm(samples[curr_cluster[candidate_fet2[ii]], :] - temp_neigh1)
      # #   candidate_d2 = np.linalg.norm(samples[curr_cluster[candidate_fet2[ii]], :] - temp_neigh1)
      # #   # sum_dist.append(1 + 1 / (1 + candidate_d1 + candidate_d2))
      # #   sum_dist.append(candidate_d1 + candidate_d2)
      # # sum_dist = np.array(sum_dist)
      # # fet2 = curr_cluster[candidate_fet2[np.argsort(sum_dist)[-round(num_queries * 0.5):]]]
      # # fet2 = self.diversityfetch2(candidate_fet2, curr_cluster,
      # #                             sum_dist, curr_dist, dth,
      # #                             round(num_queries * 0.5))
      
      # query_idx = np.append(query_idx, fet1)
      # query_idx = np.append(query_idx, fet2)
    print('No of unique idxs:', len(np.unique(query_idx)))
    return query_idx

  def query(self, label_budget):
    unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
    embedding_unlabeled = self.get_embeddings(unlabeled_data).numpy()
    # unlabeled_raw = self.get_raw_embeddings(unlabeled_data)
    # embedding_unlabeled = unlabeled_raw
    num_clusters = 50
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters)
        km.fit_predict(embedding_unlabeled)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000)
        km.fit_predict(embedding_unlabeled)
    cluster_centers, cluster_idx = km.cluster_centers_, km.labels_
    # clustering_model = fps_analysis()
    # cluster_centers, cluster_idx, cluster_dist = clustering_model.predict(embedding_unlabeled)
    print("clustering stage finish!", np.shape(cluster_centers)[0])
    query_idx = self.active_query(embedding_unlabeled, cluster_centers, cluster_idx, label_budget)
    query_idx = query_idx.astype(int)
    return unlabeled_idxs[query_idx]
