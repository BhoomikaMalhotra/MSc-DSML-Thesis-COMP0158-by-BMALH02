# Course: MSc Data Science and Machine Learning
# Module: COMP0158 (MSc Thesis)
# Candidate Number: SBRT4

#The following file generates hyperparameter tuning graphs for Random Forest Trained using All Features

import numpy as np
import pandas as pd
from turtle import color
from pandas import read_excel
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier

#Loading data
X_train = pd.read_csv('X_train.csv', index_col=None, header=0) 
X_test = pd.read_csv('X_test.csv', index_col=None, header=0) 
y_train = pd.read_csv('y_train.csv', index_col=None, header=0) 
y_test = pd.read_csv('y_test.csv', index_col=None, header=0) 

#--------Random Forest + All Features---------
#Hyper-Parameter Tuning
#max_depth = represents the number of levels
max_depth = np.linspace(8, 40, num = 25, dtype = int)
#n_estimators = Number of trees 
n_estimators =  np.linspace(start = 10, stop = 300, num = 25, dtype = int)
#max_features = maximum features provided to each tree
max_features = list(range(2, 31,2))
#min_samples_leaf = minimum samples required at each leaf node
min_samples_leaf = np.linspace(1, 4, num = 4, dtype = int)
#min_samples_split = represents the minimum samples required for a split to occur
min_samples_split = np.arange(2, 11)
bootstrap = [True]

#N-Esimators
n_estim_train_results = []
n_estim_test_results = []
for estimator in n_estimators:
   print('Computing for n_estimators = ', estimator)
   rf = RandomForestClassifier(n_estimators = estimator)
   scores = cross_validate(rf, X_train, y_train.values.ravel(), cv=10, scoring='f1', return_train_score=True)
   n_estim_train_results.append(np.mean(scores['train_score']))
   n_estim_test_results.append(np.mean(scores['test_score']))

plt.plot(n_estimators, n_estim_train_results, label = "Training Score", color = 'deepskyblue')
plt.plot(n_estimators, n_estim_test_results, label = "Validation Score using 10-Fold CV", color = 'royalblue')
plt.legend()
plt.xlabel('Number of Trees')
plt.ylabel('F1 Score')
plt.grid()
plt.savefig("n_estim_rf.png", format="PNG", bbox_inches = "tight")
plt.clf()

#Max-Depth
depth_train_results = []
depth_test_results = []
for depth in max_depth:
   print('Computing for depth = ', depth)
   rf = RandomForestClassifier(max_depth = depth)
   scores = cross_validate(rf, X_train, y_train.values.ravel(), cv=10, scoring='f1', return_train_score=True)
   depth_train_results.append(np.mean(scores['train_score']))
   depth_test_results.append(np.mean(scores['test_score']))

plt.plot(max_depth, depth_train_results, label = "Training Score", color = 'deepskyblue')
plt.plot(max_depth, depth_test_results, label = "Validation Score using 10-Fold CV", color = 'royalblue')
plt.legend()
plt.xlabel('Maximum Depth')
plt.ylabel('F1 Score')
plt.grid()
plt.savefig("max_depth_rf.png", format="PNG", bbox_inches = "tight")
plt.clf()

#Max-Features
feats_train_results = []
feats_test_results = []
for feat in max_features:
   print('Computing for max_features = ', feat)
   rf = RandomForestClassifier(max_features = feat)
   scores = cross_validate(rf, X_train, y_train.values.ravel(), cv=10, scoring='f1', return_train_score=True)
   feats_train_results.append(np.mean(scores['train_score']))
   feats_test_results.append(np.mean(scores['test_score']))

plt.plot(max_features, feats_train_results, label = "Training Score", color = 'deepskyblue')
plt.plot(max_features, feats_test_results, label = "Validation Score using 10-Fold CV", color = 'royalblue')
plt.legend()
plt.xlabel('Maximum Number of Features')
plt.ylabel('F1 Score')
plt.grid()
plt.savefig("max_features_rf.png", format="PNG", bbox_inches = "tight")
plt.clf()

#Min-Sample-Split
split_train_results = []
split_test_results = []
for split in min_samples_split:
   print('Computing for min_samples_split = ', split)
   rf = RandomForestClassifier(min_samples_split = split)
   scores = cross_validate(rf, X_train, y_train.values.ravel(), cv=10, scoring='f1', return_train_score=True)
   split_train_results.append(np.mean(scores['train_score']))
   split_test_results.append(np.mean(scores['test_score']))

plt.plot(min_samples_split, split_train_results, label = "Training Score", color = 'deepskyblue')
plt.plot(min_samples_split, split_test_results, label = "Validation Score using 10-Fold CV", color = 'royalblue')
plt.legend()
plt.xlabel('Minimum Number of Samples Required for a Split')
plt.ylabel('F1 Score')
plt.grid()
plt.savefig("min_split_rf.png", format="PNG", bbox_inches = "tight")
plt.clf()

print('Graphs Saved.')