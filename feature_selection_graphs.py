# Course: MSc Data Science and Machine Learning
# Module: COMP0158 (MSc Thesis)
# Submission By: BMALH02

#The following file generates hyperparameter tuning graphs for feature selection methods

import pymrmr
import numpy as np
import pandas as pd
from turtle import color
from matplotlib import pyplot as plt
from scipy.stats import uniform
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier

#Loading data
X_train = pd.read_csv('X_train.csv', index_col=None, header=0) 
X_test = pd.read_csv('X_test.csv', index_col=None, header=0) 
y_train = pd.read_csv('y_train.csv', index_col=None, header=0) 
y_test = pd.read_csv('y_test.csv', index_col=None, header=0) 

#-----------SelectKBest----------
print('Generating Graph for SelectKBest')
kbest_train_scores = []
kbest_val_scores = []
num_feats = np.arange(1,41)
for f in range(1, 41):
    print('Computing for K = ', f)
    kbest_selector = SelectKBest(f_classif, k = f) #top f features
    kbest_selector.fit(X_train, y_train.values.ravel())
    feature_index = kbest_selector.get_support(indices=True)
    kbest_selected_features = X_train.columns[feature_index]
    X_kbest_train = X_train[kbest_selected_features]
    rf = RandomForestClassifier(max_depth=30, bootstrap = True, max_features =2, min_samples_leaf=1, min_samples_split= 3, n_estimators= 75)
    scores = cross_validate(rf, X_kbest_train, y_train.values.ravel(), cv=10, scoring='f1', return_train_score=True)
    kbest_train_scores.append(np.mean(scores['train_score']))
    kbest_val_scores.append(np.mean(scores['test_score']))

plt.plot(num_feats, kbest_train_scores, label = "Training Score", color = 'deepskyblue')
plt.plot(num_feats, kbest_val_scores, label = "Validation Score using 10-Fold CV", color = 'royalblue')
plt.legend()
plt.xlabel('Number of Features (SelectKBest)')
plt.ylabel('F1 Score')
plt.grid()
plt.savefig("kbest_graph.png", format="PNG", bbox_inches = "tight")
plt.clf()
print('Graph Saved.')

#-----------Maximum Relevance — Minimum Redundancy (mRMR)-----------------
print('Generating Graph for mRMR')
mrmr_train_scores = []
mrmr_val_scores = []
num_feats = np.arange(1,41)
for f in range(1, 41):
    print('Computing for K = ', f)
    mrmr_input = pd.concat([y_train, X_train], axis=1)
    mrmr_features = pymrmr.mRMR(mrmr_input, 'MIQ', f) #top f features
    X_mrmr_train = X_train[mrmr_features]
    rf = RandomForestClassifier(max_depth=30, bootstrap = True, max_features =2, min_samples_leaf=1, min_samples_split= 3, n_estimators= 75)
    scores = cross_validate(rf, X_mrmr_train, np.array(y_train['cdi_label']), cv=10, scoring='f1', return_train_score=True)
    mrmr_train_scores.append(np.mean(scores['train_score']))
    mrmr_val_scores.append(np.mean(scores['test_score']))

plt.plot(num_feats, mrmr_train_scores, label = "Training Score", color = 'deepskyblue')
plt.plot(num_feats, mrmr_val_scores, label = "Validation Score using 10-Fold CV", color = 'royalblue')
plt.legend()
plt.xlabel('Number of Features (mRMR)')
plt.ylabel('F1 Score')
plt.grid()
plt.savefig("mrmr_graph.png", format="PNG", bbox_inches = "tight")
plt.clf()
print('Graph Saved.')