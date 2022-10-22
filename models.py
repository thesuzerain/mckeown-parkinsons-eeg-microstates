# Wyatt Verchere, 2020

import numpy as np
from termcolor import colored
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import scipy.stats as st

ITERATIONS = 50

# classes = y for classification
# classes maybe != y for regression, but should be same shape

# Randomly downsample an unbalanced dataset
# Returns downsampled data where each value of 'classes' has the same number of occurrences of each element
# 'classes' are identifiers that can be used to group data that may be different to 'y'
# (ie: each subject's ID, in a situation where 'y' is some other label)
# 'classes' defaults to 'y'
def downsample(X,y,classes=None):
    if classes is None:
        classes = y

    class_ids, class_sizes = np.unique(classes,return_counts=True)
    smallest_cs = np.amin(class_sizes) # smallest class => value to downsample to

    for c, cs in zip(class_ids, class_sizes): # class, class_size
        diff = cs - smallest_cs
        if diff <= 0:
            continue

        # 'diff' samples from 1:cs
        ra = np.random.choice(cs, diff, replace=False).astype(int)
        class_indices = np.where(classes == c)[0]
        z= np.arange(len(X))
        # Drop random samples of bigger classes
        classes = np.delete(classes,class_indices[ra],0)
        y = np.delete(y,class_indices[ra],0)
        X = np.delete(X,class_indices[ra],0)
    # Return X matrix in correct shape (cannot be 1D)
    return X.reshape(X.shape[0],-1), y

# classes = y for classification
# classes maybe != y for regression, but should be same shape
# also => removes nan subjects from the list


# Restricts x samples to usable data
# -> Removes all rows with a NAN in the row or in the corresponding label
# -> Removes all rows with a class not in allowed_classes (if applicable)
# 'classes' are identifiers that can be used to group data that may be different to 'y'
# (ie: each subject's ID, in a situation where 'y' is some other label)
# 'classes' defaults to 'y'
def restrict_to_allowed(X,y,classes=None,allowed_classes=0):
    if classes is None:
        classes = y

    if allowed_classes == 0:
        allowed_classes = np.unique(classes)
    else:
        allowed_classes = np.array(allowed_classes)

    nanX = np.any(np.isnan(dim2(X)),axis=1)
    nany = np.isnan(y)
    isin = np.isin(classes,allowed_classes)

    X = dim2(X)[np.logical_and.reduce((isin,~nanX,~nany))]
    y = y[np.logical_and.reduce((isin,~nanX,~nany))]

    return X,y

# Force data to be at least 2d
def dim2(d):
    if d.ndim<2:
        return d.reshape(-1,1)
    return d


def compare_classifier_generic(classifier_func,scoring_method,X,y,classes=None,allowed_classes=0):
    if np.all(np.isnan(X)) or np.all(np.isnan(y)):
        print("Dataset is entirely NaN, skipping.")
        return 0,0
    
    # Sanity check
    X, y = restrict_to_allowed(X,y,classes=classes,allowed_classes=allowed_classes)
    
    # Normalize
    for i in range(X.shape[1]):
        X[:,i] -= np.mean(X[:,i])
        X[:,i] /= np.std(X[:,i])

    scores = []
    # Downsamples randomly repeatedly
    for i in range(ITERATIONS):
        if classes is None:
            X_ = X; y_ = y
        else:
            X_, y_ = downsample(X,y,classes)

        model = classifier_func()
        scores.extend(cross_val_score(model,dim2(X_),y_,scoring=scoring_method))

    scores = np.array(scores)
    return np.mean(scores), np.std(scores)


# Uses simple SVM to get the classifiability of y from X
# => returns mean, std of repeated downsampling and cross validation
def compare_svm(X,y,allowed_classes=0):
    mean, std = compare_classifier_generic(SVC,'accuracy',X,y,allowed_classes=0)
    return mean, std

# Uses simple logistic regression to get the classifiability of y from X
# => returns mean, std of repeated downsampling and cross validation
def compare_lr(X,y,allowed_classes=0):
    mean, std = compare_classifier_generic(LogisticRegression(),X,y,allowed_classes=0)
    return mean, std

# Uses a random forest to get the predicability of y from X
# allowed_classes are what values of y are used
# ignore repeats => do not run more than once
# => returns mean, std of repeated cross_val
def compare_rf(X,y,allowed_classes=0,ignore_repeats=False):
    mean, std = compare_classifier_generic(RandomForestClassifier(),'accuracy',X,y,allowed_classes=0)
    return mean, std

# Uses a simple Linear regression to get the correlation of y from X
# allowed_classes are what values of y are used
# ignore repeats => do not run more than once
# => returns mean, std of repeated cross_val
def regress_linear(X,y,classes=None,allowed_classes=0):
    mean, std = compare_classifier_generic(LinearRegression(),'r2',X,y,allowed_classes=0)
    return mean, std

# Uses a Random Forest to get the correlation of y from X
# allowed_classes are what values of y are used
# ignore repeats => do not run more than once
# => returns mean, std of repeated cross_val
def regress_rf(X,y,classes=None,allowed_classes=0,ignore_repeats=False):
    mean, std = compare_classifier_generic(RandomForestRegressor(),'r2',X,y,allowed_classes=0)
    return mean, std


def find_relevant_features_classification(X,y,thresh=-1,type="SVM",allowed_classes=0):
    ret = np.zeros((len(X.T))) # return array is length=features
    # if threshold unset, set to a little bit over random chance
    if thresh<0:
        thresh = 0.1+ 1.0/len(np.unique(y)) # eg: 0.6 for 2 classes, 0.4 for 3...
    for x_, x in enumerate(X.T):
        # if the classification accuracy is better than thresh, mark this feature as relevant
        if type=="SVM" and compare_svm(x.T,y,allowed_classes=allowed_classes)[0] >= thresh:
            ret[x_] = 1
        if type=="LogisticRegression" and compare_lr(x.T,y,allowed_classes=allowed_classes)[0] >= thresh:
            ret[x_] = 1
        if type=="RandomForest" and compare_rf(x.T,y,ignore_repeats=True,allowed_classes=allowed_classes)[0] >= thresh:
            ret[x_] = 1
    return ret


def close_to_zero(values):
    values[np.isclose(values,0)]=0
    return values

def color_for_class(c):
    color_order = 'bkrbkr'
    return color_order[c]


def display_ttest(dataset, class_1_mask, class_2_mask, comparison_name):
    _,p = st.ttest_ind(dataset[class_1_mask],dataset[class_2_mask])
    print_ttest(p,comparison_name)



def plot_classes_for_feat(dataset, divider,feat_name, legend_class_names, label_class_names):

    patients_so_far = 0

    range_num = len(label_class_names)
    for i in range(range_num):
        set_count = np.count_nonzero(divider==i)
        plt.scatter(patients_so_far + np.arange(set_count),dataset[divider==i].reshape(-1,1),c=color_for_class(i),label=legend_class_names[i])
        patients_so_far += set_count
        if i < 5:
            plt.axvline(x=patients_so_far, color='k')
            

#     plt.xlabel("HC-          PDON-      PDOFF-        HC+        PDON+        PDOFF+",horizontalalignment='right', position=(1,25))
#     if only_three:
#         plt.xlabel("HC-                           PDON-                        PDOFF-",horizontalalignment='right', position=(1,25))
    plt.ylabel(feat_name)
    plt.legend()
    plt.show()


def print_ttest(p,class_text,num_digits = 3):
    strengthcolor="white"
    if p < 0.1:
        strengthcolor="red"
    if p < 0.05:
        strengthcolor="yellow"
    if p < 0.01:
        strengthcolor="blue"
    if p < 0.001:
        strengthcolor="green"
    print(colored(class_text+f": p={p:.4f}",strengthcolor))

def print_crossval(mean,std,unique_classes,feat_text):
    goodness_rating = (mean-(1/unique_classes))/(1-(1/unique_classes))
    strengthcolor="white"
    if goodness_rating > 0.2:
        strengthcolor="red"
    if goodness_rating > 0.4:
        strengthcolor="yellow"
    if goodness_rating > 0.6:
        strengthcolor="blue"
    if goodness_rating > 0.8:
        strengthcolor="green"
    print(colored("Cross validation score for "+ feat_text+ f": {mean:.4f} ±{std:.4f}",strengthcolor))

    # print(colored("Cross validation score for "+ feat_text+": "+str(mean)+" ±"+str(std),strengthcolor))


def print_crossval_regress(mean,std,feat_text):
    goodness_rating = mean
    strengthcolor="white"
    if goodness_rating > 0.2:
        strengthcolor="red"
    if goodness_rating > 0.4:
        strengthcolor="yellow"
    if goodness_rating > 0.6:
        strengthcolor="blue"
    if goodness_rating > 0.8:
        strengthcolor="green"

    # print(colored("Cross validation score for "+ class_text+": R2="+" ±"+str(std),strengthcolor))
    print(colored(f'Cross validation score for {feat_text:} : R2= {mean:.2f}±{std:.2f}',strengthcolor))
