# Wyatt Verchere 2020

# Helper functions to load and format GVS subject data from previous studies

import scipy.io as sio
import numpy as np

# GVS matrix of whether or not that patient *improved*
# feat is a number from 0 to (number of feats)

# Returns:
# num_patient x num_stim binary matrix
#       - 1 = improved, 0 = did not improve
def get_improved_binary_matrix(feat):
    taskImprove = sio.loadmat("taskImprove.mat")
    unordered_list = [] # final matrix as unordered list (not correct order of classes)
    for class_id in range(3): # PD-offmed, PD-onmed, and HC in order in file (So will need to be reordered)
        subj_count = len(taskImprove["stim3"][0,class_id][:,0]) # pick arbitrary stim and feature for number of subjects
        class_mat = np.zeros((subj_count,7)) # 7 stim

        for stim in range(3,10): # stim matrix names  range from 3-9 inclusive
            class_mat[:,stim-3] = taskImprove["stim"+str(stim)][0,class_id][:,feat] # extract feature which is column index 'feat'

        unordered_list.append(class_mat)

    # Re-order list and turn to matrix (done the long way so its changeable easily)
    ret_matrix = np.r_[
        unordered_list[2],
        unordered_list[1],
        unordered_list[0]
    ]

    return ret_matrix

def get_genders():
    genderImprove = sio.loadmat("gvsparticipantinfo.mat")
    g = np.r_[genderImprove['hcinfo'],genderImprove['pdinfo'],genderImprove['pdinfo']]
    return g[:,1]
def get_ages():
    genderImprove = sio.loadmat("gvsparticipantinfo.mat")
    a = np.r_[genderImprove['hcinfo'],genderImprove['pdinfo'],genderImprove['pdinfo']]
    return a[:,0]
def get_updrs():
    u = [
        26,17,11, 37, 29,
        8, 39, 35, 12, 30,
        13, 31, 34, 32, 14,
        21, 29, 20, 18, 14
    ];
    return np.r_[np.zeros(shape=[22,]),u,u]


# array([array(['Grip Strength'], dtype='<U13'),        => 0 in 0-indexed
#        array(['Strength Velocity'], dtype='<U17'),    => 4 in 0-indexed
#        array(['Movement Time (ms)'], dtype='<U18'),   => 6 in 0-indexed
#        array(['Squeeze Time (ms)'], dtype='<U17'),    => 7 in 0-indexed
#        array(['Reaction Time (ms)'], dtype='<U18'),   => 12 in 0-indexed
#        array(['Peak Time (ms)'], dtype='<U14')], dtype=object) => 11 in 0-indexed
def get_improved_feats():
    taskImprove = sio.loadmat("taskImprove.mat")
    return taskImprove['featurename'][0]



def get_improved_amount_matrix():
    def load_feat_means(filename):
        temp = sio.loadmat(filename)

        hcname='hcoffmed'
        pdonname='pdonmed'
        pdoffname='pdoffmed'
        print(filename)
        if filename.endswith("8.mat"):
            hcname = 'hcoffmed_gvs8'
            pdonname = 'pdonmed_gvs8'
            pdoffname = 'pdoffmed_gvs8'

        temp_combined = np.r_[
            np.array([np.array(t) for t in temp[hcname][0]]),
            np.array([np.array(t) for t in temp[pdonname][0]]),
            np.array([np.array(t) for t in temp[pdoffname][0]])
        ]

        # only load variables that are selected in get_improved_feats
        choose_features = np.array([0,4,6,7,12,11])

        temp_combined = temp_combined[:,:,choose_features]

        # maybe shouldn't get mean over trials.......?
        return np.nanmean(temp_combined,axis=1)

    sham = load_feat_means("task_gvssham.mat")
    stats_7 = load_feat_means("task_gvsstim7.mat")
    stats_8 = load_feat_means("task_gvsstim8.mat")

    improvement_7 = stats_7 - sham
    improvement_8 = stats_8 - sham

    return np.stack((improvement_7,improvement_8))



# We are provided with a set of data that shows the change in reaction in only 3 features
# This is identical to get_improved_amount_matrix but:
# - has only those 3 features
# - has it for *all stim* (not just the ones for whose EEG data we've collected)
def get_threefeat_improved_feats():
    temp = sio.loadmat("BehaviourData/behav_gvsstim1(sham).mat")

    return [i[0] for i in temp['featurename'][0]]

def get_threefeat_improved_amount_matrix():
    def load_feat_means(filename):
        def correct_nan_shape(dataset):
            dataset = np.array(dataset)
            if dataset.shape[0] == 1 or dataset.shape[1] == 1:
                return np.empty(shape=(10,3))
            return dataset
        temp = sio.loadmat(filename)

        hcname='hc'
        pdonname='pdonmed'
        pdoffname='pdoffmed'

        # this is needed as nan patients for this dataset are the wrong size


        temp_combined = np.r_[
            np.array([correct_nan_shape(t) for t in temp[hcname][0]]),
            np.array([correct_nan_shape(t) for t in temp[pdonname][0]]),
            np.array([correct_nan_shape(t) for t in temp[pdoffname][0]])
        ]
        # maybe shouldn't get mean over trials.......?
        return np.nanmean(temp_combined,axis=1)

    sham = load_feat_means("BehaviourData/behav_gvsstim1(sham).mat")
    stats_2 = load_feat_means("BehaviourData/behav_gvsstim2(noisy).mat")
    stats_3 = load_feat_means("BehaviourData/behav_gvsstim3.mat")
    stats_4 = load_feat_means("BehaviourData/behav_gvsstim4.mat")
    stats_5 = load_feat_means("BehaviourData/behav_gvsstim5.mat")
    stats_6 = load_feat_means("BehaviourData/behav_gvsstim6.mat")
    stats_7 = load_feat_means("BehaviourData/behav_gvsstim7.mat")
    stats_8 = load_feat_means("BehaviourData/behav_gvsstim8.mat")
    stats_9 = load_feat_means("BehaviourData/behav_gvsstim9.mat")

    return np.stack((
        # stats_2 - sham,
        # # stats_3 - sham,
        # stats_4 - sham,
        # stats_5 - sham,
        # stats_6 - sham,
        stats_7 - sham,
        stats_8 - sham,
        # stats_9 - sham,
    ))
