# Wyatt Verchere 2020

# Extract potentially interesting features from microstate data
# some of these are from the microstate kit and some of these are requested and discussed features in lab.

import scipy.io as sio
import scipy.signal as sig
import numpy as np
from eeg_microstates import *
import os
import scipy.stats as st

# load cached features if appropraite,otherwise, generate new ones
def load_feature_data(X,downsized):

    if downsized == False:
        if not os.path.exists("abstr/gvs_feats_matrix.npy"):
            X = np.array([feats_from_sequence(x,500) for x in X])

            np.save("abstr/gvs_feats_matrix.npy",X)
        return np.load('abstr/gvs_feats_matrix.npy')
    else:
        if not os.path.exists("abstr/gvs_feats_matrix_downsized.npy"):
            X = np.array([feats_from_sequence(x,500) for x in X])

            np.save("abstr/gvs_feats_matrix_downsized.npy",X)
        return np.load('abstr/gvs_feats_matrix_downsized.npy')

# Names of main set of features
def feat_names():
    feat_names = []
    feat_names.extend(["p_empirical_"+str(i) for i in range(4)]) #0
    # feat_names.append("GFP Peaks per second.")
    feat_names.extend(["Transition Matrix["+str(i)+","+str(j)+"]" for i in range(4) for j in range(4)]) #4-20
    feat_names.extend(["Empirical entropy","Empirical entropy rate","Theoretical MC entropy rate"])
    feat_names.extend(["p-value of 0-Markovity","p-value of 1-Markovity","p-value of 2-Markovity"])
    feat_names.extend(["geoTest for Map"+str(i) for i in range(4)])
    feat_names.extend(["conditional homogeneity"])
    feat_names.extend(["symmetry test"])
    feat_names.append("Maximum AIF value in 27-37")
    feat_names.append("Location of Maximum AIF value after 15")
    feat_names.append("Number of Maxima in AIF between 0-50")
    feat_names.append("Number of Minima in AIF between 0-50")
    feat_names.append("Number of samples")
    feat_names.append("MeanAIFInBand - Alpha") # log mean over mean
    feat_names.append("MeanAIFInBand - Beta") # log mean over mean
    feat_names.append("MeanAIFInBand - Gamma") # log mean over mean
    # feat_names.extend(["ICA of AIF - "+str(i) for i in range(n_components)])

    return feat_names

# Extract features from sequence
# x is n_sam array, -1 indices are *NOT COUNTED*
def feats_from_sequence(x,fs):
    feats =[]
    n_maps = len(np.unique(x))

    # Most basic features- how often does a state occur, how likely from a given state is it likely to transition into any other state
    p_hat = p_empirical(x, n_maps)
    T_hat = T_empirical(x, n_maps)

    # p_hat, pps
    feats.extend(p_hat[0:4])

    # transition matrix values
    feats.extend(T_hat.flatten())

    # empiral entropy, entropy rate, entropy max
    h_hat = H_1(x, n_maps)
    h_max = max_entropy(n_maps)
    h_rate, _ = excess_entropy_rate(x, n_maps, kmax=8, doplot=True)
    h_mc = mc_entropy_rate(p_hat, T_hat)
    feats.extend(np.array([h_hat,h_rate,h_mc]))

    # The statistical Markovianity tests
    # p_M0 => p-value for test of whether P(X_n) is dependent on P(X_(n-1)) (closer to zero, there is a stronger dependency)
    # p_M1 => p-value for test of whether P(X_n) independent of P(X_(n-2)) 
    alpha = 0.01
    p_M0 = testMarkov0(x, n_maps, alpha)
    p_M1 = testMarkov1(x, n_maps, alpha)
    p_M2 = testMarkov2(x, n_maps, alpha)
    p_vals_geo = geoTest(x, n_maps, 1000./fs, alpha)
    feats.extend(np.array([p_M0,p_M1,p_M2]))
    feats.extend(np.array([p for p in p_vals_geo]))

    # conditional homogeneity test - lower means a greater tendency for transition matrix to change over time 
    L = 5000
    p3 = conditionalHomogeneityTest(x, n_maps, L, alpha)
    feats.append(p3)

    # symmetry test
    p4 = symmetryTest(x, n_maps, alpha)
    feats.append(p4)

    
    # We examine the complicated AIF values
    # AIF -> essentially for each value AIF[t], how predictlbe n+t is from ANY sample n
    # This produces very interesting strange results with our dataset.
    # Parkinson's patients are often in 'brain loops' (due to dopamine),w hich may be related to this
    # aif values
    l_max = 100
    n_mc = 10
    aif = mutinf(x, n_maps, l_max)
    # max
    aif_np = aif[27:37]
    aif_np_max = np.max(aif_np)
    feats.append(aif_np_max)
    # argmax
    aif_np = aif[15:]
    aif_np_argmax = np.argmax(aif_np)
    feats.append(aif_np_argmax)
    # num maxima
    aif_np = aif[0:50]
    extrema = sig.argrelextrema(aif_np, np.greater)[0]
    aif_np_maxima = len(extrema)
    feats.append(aif_np_maxima)
    # num minima
    aif_np = aif[0:50]
    extrema = sig.argrelextrema(aif_np, np.less)[0]
    aif_np_minima = len(extrema)
    feats.append(aif_np_minima)
    # # ica
    # aif_np = aif[10:60]
    # n_components = 2
    # ica=FastICA(n_components=n_components,random_state=0)#
    # aif_np_ica = ica.fit_transform(aif_np)
    # feats.extend(aif_np_ica.tolist())

    # This is more relevant for the downsized sample set then the standardized one
    feats.append(len(x))

    # Alpha
    start_freq_in_samples = fs/8
    end_freq_in_samples = fs/13
    aif_seg = aif
    logmean = np.mean(aif_seg[int(end_freq_in_samples):int(start_freq_in_samples)])
    feats.append(logmean/np.mean(aif_seg))
    # Beta
    start_freq_in_samples = fs/13
    end_freq_in_samples = fs/30
    aif_seg = aif
    logmean = np.mean(aif_seg[int(end_freq_in_samples):int(start_freq_in_samples)])
    feats.append(logmean/np.mean(aif_seg))
    # Gamma
    start_freq_in_samples = fs/30
    end_freq_in_samples = fs/50
    aif_seg = aif
    logmean = np.mean(aif_seg[int(end_freq_in_samples):int(start_freq_in_samples)])
    feats.append(logmean/np.mean(aif_seg))
    
      

    return np.array(feats)


def metastate_features(microstate_x):
    def window(a, w = 4, o = 2, copy = False):
        sh = (a.size - w + 1, w)
        st = a.strides * 2
        view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
        if copy:
            return view.copy()
        else:
            return view
    num_pts = microstate_x.shape[0]
        
    window_size = 500
    window_stride = 50
    microstates_alpha_windows_x = np.array([window(x,w=window_size,o=window_stride) for x in microstate_x]).reshape(-1,500)

    microstates_alpha_windows_x.shape

    x_feats = np.array([np.r_[p_empirical(x,4).flatten(),T_empirical(x,4).flatten()] for x in microstates_alpha_windows_x])

    X = x_feats

    from sklearn.decomposition import PCA
    pca_components = 5
    pca = PCA(n_components=pca_components)
    X_new = pca.fit_transform(X)
    
    X_new = X_new.reshape(    num_pts ,-1,pca_components)
    return np.mean(X_new,axis=1)

def meta_feature_names():
    return ["Mean meta-state component 1","Mean meta-state component 2","Mean meta-state component 3","Mean meta-state component 4","Mean meta-state component 5"]

def time_feature_names():
    return ["Mean time in state1",
    "Max time in state1",
    "Std of time in state1",
    "Skew of time in state1",
    "Number of distinct occurrences of state1",
    "Median time in state1",

    "Mean time in state2",
    "Max time in state2",
    "Std of time in state2",
    "Skew of time in state2",
    "Number of distinct occurrences of state2",
    "Median time in state2",

    "Mean time in state3",
    "Max time in state3",
    "Std of time in state3",
    "Skew of time in state3",
    "Number of distinct occurrences of state3",
    "Median time in state3",

    "Mean time in state4",
    "Max time in state4",
    "Std of time in state4",
    "Skew of time in state4",
    "Number of distinct occurrences of state4",
    "Median time in state4",
    ]


def time_feats(times):
    ret = []
    for v in range(len(times)):
        ret.append(np.mean(np.array(times[v])))
        ret.append(np.max(np.array(times[v])))
        ret.append(np.std(np.array(times[v])))
        ret.append(st.skew(np.array(times[v])))
        ret.append(len(times[v]))
        ret.append(np.median(np.array(times[v])))
    return np.array(ret)


# Input: sequence of states
# fs = frequency
# Output: sequence of states, only one state each (reduces A A A B B B C C C to A B C)
# - list of lists set of *times* per states, in ms
# if keep_indices is on, returned sequence has -1 appened to keep same size
def downsize_sequence(seq,fs):
    from itertools import groupby
    time_sets = [[] for _ in np.unique(seq)]
    ret_seq = []
    def analyze_group(g):
        time_sets[g[0]].append(len(g)/fs)
        ret_seq.append(g[0])

    for k,g in groupby(seq):
        analyze_group(list(g))

    return np.array(ret_seq), time_sets


