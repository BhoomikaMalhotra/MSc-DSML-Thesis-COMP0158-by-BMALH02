# Course: MSc Data Science and Machine Learning
# Module: COMP0158 (MSc Thesis)
# Candidate Number: SBRT4

import re
import shap
import pymrmr
import numpy as np
import pandas as pd
import xgboost as xgb
from turtle import color
from sklearn import tree
from boruta import BorutaPy
from sklearn import metrics
from pandas import read_excel
from scipy.stats import uniform
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split,RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier

#If the xlsx data file does not download, uncomment the following 2 lines.
# import pip
# pip.main(["install", "openpyxl"])

#Loading data
data = pd.read_excel('psych_data1.xlsx', index_col=None, header=0)  
print('Dataset Downloaded')
print('')
#Removing empty rows
data.drop(['VAR00006', 'VAR00001', 'VAR00002', 'VAR00003', 'VAR00004'], axis=1, inplace=True)

#Counting number of missing values
data = data.replace(np.nan, int(999)) #Replacing NaN to 999
cols = data.columns.tolist()
total_missing_vals = 0
for c in cols:
    total_missing_vals += (data[c] == 999).sum()
print('Total Missing Values Are =', total_missing_vals)

#Combining stressful Life Events and their perceived impact
events = []
stress_levels = []
for ques_num in range(1,61):
    events.append("eev"+str(ques_num)+"imp")
    stress_levels.append("eev"+str(ques_num)+"fre")

for event, stress in zip(events , stress_levels):
    data[event] = data[event] * data[stress]

for stress_l in stress_levels:
    data.drop(stress_l, axis=1, inplace=True)

data = data.replace(int(998001), int(999))
cols = data.columns.tolist()
total_missing_vals_1 = 0
for c in cols:
    total_missing_vals_1 += (data[c] == 999).sum()

print('After combining stressful life events and their perceived impact, total missing values are =', total_missing_vals_1)

#Manual selection of Miscellaneous Personal Questionnaire
all_cs_data = data.loc[:, :'tdeleit']
all_cs_data.drop(['institution_name', 'dob_known', 'institution_time','school_name','number_residents','siblings_institution', 
'Frequency_family_contact', 'occupation','Earnings','work_enjoyability'	,'skip_school_for_work'], axis=1, inplace=True)

all_cs_data.replace(999, np.nan, inplace=True) #Replacing 999 back to NA
all_cs_data.drop(all_cs_data.loc[:, 'Group_Type':'working'].columns, axis=1, inplace=True)

#Complete cases for CDI (all 27 questions)
depression = all_cs_data.loc[:, ['Id','cdi1','cdi2','cdi3','cdi4','cdi5','cdi6','cdi7','cdi8','cdi9','cdi10','cdi11','cdi12','cdi13','cdi14','cdi15',
'cdi16','cdi17','cdi18','cdi19','cdi20','cdi21','cdi22','cdi23','cdi24','cdi25','cdi26','cdi27']]
complete_depression = depression.dropna(axis = 0, how = 'any', inplace = False)
print('Total labeled cases = ', len(complete_depression))

#Cross sectional dataset for complete CDI cases
all_cs_data.drop(all_cs_data.loc[:, 'cdi1':'cdi27'].columns, axis=1, inplace=True)
complete_cases = pd.merge(complete_depression , all_cs_data, on = 'Id') #merging by ID

#Data Encoding (Combining psychological psychological tools)
group_data = complete_cases

#Child Depression Inventory (CDI)
group_data['cdi_score'] = complete_cases.loc[:,'cdi1':'cdi27'].sum(axis = 1) 
group_data.drop(['cdi1','cdi2','cdi3','cdi4','cdi5','cdi6','cdi7','cdi8','cdi9','cdi10','cdi11','cdi12','cdi13','cdi14','cdi15',
'cdi16','cdi17','cdi18','cdi19','cdi20','cdi21','cdi22','cdi23','cdi24','cdi25','cdi26','cdi27'], axis=1, inplace=True)

#Assigning labels (1-Moderately/Severely Depressed, 0-Not Depressed/onset)
depression_labels =[]
for cdi_c in group_data['cdi_score']:
    if cdi_c > 19:
        depression_labels.append(1)
    else:
        depression_labels.append(0)
group_data['cdi_label'] = depression_labels
group_data.drop(['cdi_score'], axis=1, inplace=True)
group_data.drop(['Id'], axis=1, inplace=True)
print('')
print('Label Assignment Complete.')
print('')
print('Total missing values prior to imputation are = ', group_data.isna().sum().sum())

t_dep_counter = 0
t_not_dep_counter = 0

for cl in group_data['cdi_label']:
    if cl == 0:
        t_not_dep_counter += 1
    else:
        t_dep_counter += 1
print('')
print('Total Candidates who are Moderately/Severely Depressed in Entire Dataset = ', t_dep_counter)
print('Total Candidates who are Not Depressed/ At Onset of Depression in Entire Dataset =', t_not_dep_counter)

#Standardising data
standardisation_scalar = MinMaxScaler()
group_data = pd.DataFrame(standardisation_scalar.fit_transform(group_data), columns = group_data.columns)

#Splitting into Train and Test sets in 0.75:0.25 ratio
cdi_output = group_data['cdi_label']
group_data.drop(['cdi_label'], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(group_data.index, cdi_output, test_size = 0.25, random_state=123, stratify=cdi_output)

X_train = group_data.iloc[X_train]
X_test = group_data.iloc[X_test]
X_train = pd.concat([X_train, y_train], axis = 1)

dep_counter = 0
not_dep_counter = 0

for cl in X_train['cdi_label']:
    if cl == 0:
        not_dep_counter += 1
    else:
        dep_counter += 1
print('')
print('Total Candidates who are Moderately/Severely Depressed in Training Dataset = ', dep_counter)
print('Total Candidates who are Not Depressed/ At Onset of Depression in Training Dataset =', not_dep_counter)

# #Balancing dataset by oversampling 
not_dep_data = X_train[X_train['cdi_label'] == 0].sample(n=638, replace=False)
dep_data = X_train[X_train['cdi_label']  == 1.0].sample(n=638, replace=True)
X_train = pd.concat([not_dep_data, dep_data], ignore_index=True).sample(frac=1) #Shuffling data
X_train = X_train.reset_index(drop = True)

b_dep_counter = 0
b_not_dep_counter = 0

for cl in X_train['cdi_label']:
    if cl == 0:
        b_not_dep_counter += 1
    else:
        b_dep_counter += 1
print('')
print('After Oversampling: Total Candidates who are Moderately/Severely Depressed = ', b_dep_counter)
print('After Oversampling: Total Candidates who are Not Depressed/ At Onset of Depression =', b_not_dep_counter)

y_train = X_train['cdi_label']
X_train.drop(['cdi_label'], axis=1, inplace=True)

# #K-NN imputation hyperparameter tuning
# #------------PLEASE UNCOMMENT THE DOLLOWING SECTION OF CODE TO GENERATE GRAPH---------------
# #K-Fold CV to determine the best value of k in KNN Imputation using only training data
# k_values = [k for k in range(46) if k % 2 != 0]
# evals = list()
# print('')
# print('Generating Graph for Hyperparameter Tuning in K-NN Imputation')
# for k in k_values:
# 	impute_pip = Pipeline(steps=[('i', KNNImputer(n_neighbors=k)), ('m', tree.DecisionTreeClassifier(max_depth = 6))])
# 	kfold_cv = RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=1)
# 	cv_scores = cross_val_score(impute_pip, X_train, y_train, scoring='f1_macro', cv=kfold_cv, n_jobs=-1)
# 	evals.append(np.mean(cv_scores))  
# plt.plot(k_values, evals, color = 'blue')
# plt.xlabel('Value of K')
# plt.ylabel('F1 Score of Decision Tree')
# plt.grid()
# plt.savefig("knn_imputation.png", format="PNG", bbox_inches = "tight")
# plt.clf()
# print('Graph Saved.')
# print('')
# #--------------------------------------------------------------------------

#Training final imputer on X_train and applying it on X_test
final_knn_imputer = KNNImputer(n_neighbors=13) #K = 13 in K-NN Imputation
X_train = pd.DataFrame(final_knn_imputer.fit_transform(X_train),columns = X_train.columns)
X_test = pd.DataFrame(final_knn_imputer.transform(X_test),columns = X_test.columns)
print('')
print('K-NN Imputation executed.')

#Multidimensional Life Satisfaction
#Self- Questions 1,6,11,15,20,25,29,35,40,44
X_train['lf_self'] = X_train['emsv1']+X_train['emsv6']+X_train['emsv11']+X_train['emsv15']+X_train['emsv20']+X_train['emsv25']+X_train['emsv29']+X_train['emsv35']+X_train['emsv40']+X_train['emsv44']
X_test['lf_self'] = X_test['emsv1']+X_test['emsv6']+X_test['emsv11']+X_test['emsv15']+X_test['emsv20']+X_test['emsv25']+X_test['emsv29']+X_test['emsv35']+X_test['emsv40']+X_test['emsv44']
#Comparative Self- Questions 2,7,12,16,21,30,36,45
X_train['lf_comp_self'] = X_train['emsv2']+X_train['emsv7']+X_train['emsv12']+X_train['emsv16']+X_train['emsv21']+X_train['emsv30']+X_train['emsv36']+X_train['emsv45']
X_test['lf_comp_self'] = X_test['emsv2']+X_test['emsv7']+X_test['emsv12']+X_test['emsv16']+X_test['emsv21']+X_test['emsv30']+X_test['emsv36']+X_test['emsv45']
#Non-Violence- Questions 8,22,31,47
X_train['lf_nonviolence'] = X_train['emsv8']+X_train['emsv22']+X_train['emsv31']+X_train['emsv47']
X_test['lf_nonviolence'] = X_test['emsv8']+X_test['emsv22']+X_test['emsv31']+X_test['emsv47']
#Family- Questions 3,9,13,17,23,26,32,37,41,46,50
X_train['lf_family'] = X_train['emsv3']+X_train['emsv9']+X_train['emsv13']+X_train['emsv17']+X_train['emsv23']+X_train['emsv26']+X_train['emsv32']+X_train['emsv37']+X_train['emsv41']+X_train['emsv46']+X_train['emsv50']
X_test['lf_family'] = X_test['emsv3']+X_test['emsv9']+X_test['emsv13']+X_test['emsv17']+X_test['emsv23']+X_test['emsv26']+X_test['emsv32']+X_test['emsv37']+X_test['emsv41']+X_test['emsv46']+X_test['emsv50']
#School- Questions 5,14,19,28,34,43,49
X_train['lf_school'] = X_train['emsv5']+X_train['emsv14']+X_train['emsv19']+X_train['emsv28']+X_train['emsv34']+X_train['emsv43']+X_train['emsv49']
X_test['lf_school'] = X_test['emsv5']+X_test['emsv14']+X_test['emsv19']+X_test['emsv28']+X_test['emsv34']+X_test['emsv43']+X_test['emsv49']
#Friednship- Questions 4,10,18,24,27,33,38,39,42,48
X_train['lf_friendship'] = X_train['emsv4']+X_train['emsv10']+X_train['emsv18']+X_train['emsv24']+X_train['emsv27']+X_train['emsv33']+X_train['emsv38']+X_train['emsv39']+X_train['emsv42']+X_train['emsv48']
X_test['lf_friendship'] = X_test['emsv4']+X_test['emsv10']+X_test['emsv18']+X_test['emsv24']+X_test['emsv27']+X_test['emsv33']+X_test['emsv38']+X_test['emsv39']+X_test['emsv42']+X_test['emsv48']
X_train.drop(X_train.loc[:, 'emsv1':'emsv50'].columns, axis=1, inplace=True)
X_test.drop(X_test.loc[:, 'emsv1':'emsv50'].columns, axis=1, inplace=True)

#PANAS
#Positive- Questions 1,3,4,5,7,8,10,11,12,14,15,18,21,22,23,24,29,35,36,39
X_train['panas_positive'] = X_train['ea1']+X_train['ea3']+X_train['ea4']+X_train['ea5']+X_train['ea7']+X_train['ea8']+X_train['ea10']+X_train['ea11']+X_train['ea12']+X_train['ea14']+X_train['ea15']+X_train['ea18']+X_train['ea21']+X_train['ea22']+X_train['ea23']+X_train['ea24']+X_train['ea29']+X_train['ea35']+X_train['ea36']+X_train['ea39']
X_test['panas_positive'] = X_test['ea1']+X_test['ea3']+X_test['ea4']+X_test['ea5']+X_test['ea7']+X_test['ea8']+X_test['ea10']+X_test['ea11']+X_test['ea12']+X_test['ea14']+X_test['ea15']+X_test['ea18']+X_test['ea21']+X_test['ea22']+X_test['ea23']+X_test['ea24']+X_test['ea29']+X_test['ea35']+X_test['ea36']+X_test['ea39']
#Negative- Questions 2,6,9,13,16,17,19,20,25,26,27,28,30,31,32,33,34,37,38,40
X_train['panas_negative'] = X_train['ea2']+X_train['ea6']+X_train['ea9']+X_train['ea13']+X_train['ea16']+X_train['ea17']+X_train['ea19']+X_train['ea20']+X_train['ea25']+X_train['ea26']+X_train['ea27']+X_train['ea28']+X_train['ea30']+X_train['ea31']+X_train['ea32']+X_train['ea33']+X_train['ea34']+X_train['ea37']+X_train['ea38']+X_train['ea40']
X_test['panas_negative'] = X_test['ea2']+X_test['ea6']+X_test['ea9']+X_test['ea13']+X_test['ea16']+X_test['ea17']+X_test['ea19']+X_test['ea20']+X_test['ea25']+X_test['ea26']+X_test['ea27']+X_test['ea28']+X_test['ea30']+X_test['ea31']+X_test['ea32']+X_test['ea33']+X_test['ea34']+X_test['ea37']+X_test['ea38']+X_test['ea40']
X_train.drop(X_train.loc[:, 'ea1':'ea40'].columns, axis=1, inplace=True)
X_test.drop(X_test.loc[:, 'ea1':'ea40'].columns, axis=1, inplace=True)

#Normalising training and testing datasets
norm_scalar = StandardScaler()
X_train = pd.DataFrame(norm_scalar.fit_transform(X_train),columns = X_train.columns)
X_test = pd.DataFrame(norm_scalar.transform(X_test),columns = X_test.columns)

#Saving datasets for future use
X_train.to_csv('X_train.csv', index = False, encoding='utf-8')
X_test.to_csv('X_test.csv', index = False, encoding='utf-8')
y_train.to_csv('y_train.csv', index = False, encoding='utf-8')
y_test.to_csv('y_test.csv', index = False, encoding='utf-8')

print('')
print('Pre-processing complete.')

#Function to compute evaluation metrics
def eval_metrics(y_test_labels, y_pred_labels):
    '''
    Calculates evaluation metrics for a model.
    Input: True and predicted labels in array form.
    Output: Confusion Matrix, Sensitivity, Specificity, Precision, F1 Score and ROC-AUC. 
    '''
    auc = metrics.roc_auc_score(y_test_labels, y_pred_labels)
    cm = metrics.confusion_matrix(y_test_labels, y_pred_labels)
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    specificity = (TN / float(TN + FP))
    sensitivity = (TP / float(TP + FN))
    precision = (TP / float(TP + FP))
    f1_sc = f1_score(y_test_labels, y_pred_labels, average='macro')
    #Print commands
    print('---Evaluation Metrics----')
    print('Precision = ', precision)
    print('Sensitivity = ', sensitivity)
    print('Specificity = ', specificity)
    print('F1 Score = ', f1_sc)
    print('Area Under the Curve (AUC) = ', auc)
    print('---Confusion Matrix----')
    print('True Positive', TP)
    print('True Negative', TN)
    print('False Positive', FP)
    print('False Negative', FN)

#Loading data
X_train = pd.read_csv('X_train.csv', index_col=None, header=0) 
X_test = pd.read_csv('X_test.csv', index_col=None, header=0) 
y_train = pd.read_csv('y_train.csv', index_col=None, header=0) 
y_test = pd.read_csv('y_test.csv', index_col=None, header=0) 

#----Models With All Features------

print('')
print('Beginning Predictive Modelling.')

#--Random Forest + All Features--
#Hyper-Parameter Tuning- Step 1 - Random Grid Search
#max_depth = represents the number of levels
max_depth = np.linspace(8, 40, num = 25, dtype = int)
#n_estimators = Number of trees 
n_estimators =  np.linspace(start = 10, stop = 300, num = 25, dtype = int)
#max_features = maximum features provided to each tree
max_features = list(range(2, 31, 2))
#min_samples_split = represents the minimum samples required for a split to occur
min_samples_split = np.arange(2, 11)
bootstrap = [True]

rf_random_grid_search_vals = {'max_depth': max_depth, 'n_estimators': n_estimators, 'max_features': max_features, 'min_samples_split': min_samples_split, 'bootstrap': bootstrap}

#Fitting a baseline model
rf_basic = RandomForestClassifier()
rf_random_grid_search = RandomizedSearchCV(estimator = rf_basic, param_distributions = rf_random_grid_search_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
rf_random_grid_search.fit(X_train, y_train.values.ravel())
print('')
print('Random Forest: Random Grid Search Results are :')
print(rf_random_grid_search.best_params_)

#Step 2: Exhaustive Grid Search on narrowed down hyper-parameters
rf_final_gs_vals = {
    'bootstrap': [True],
    'max_depth': [15, 18, 20, 22, 25],
    'max_features': [2, 3, 4],
    'min_samples_split': [2, 3, 4],
    'n_estimators': [180, 200, 220, 240]
}
rf_basic = RandomForestClassifier()
rf_final_grid_search = GridSearchCV(estimator = rf_basic, param_grid = rf_final_gs_vals, cv = 5, verbose=2, n_jobs = -1)
rf_final_grid_search.fit(X_train, y_train.values.ravel())
print('')
print('Random Forest: Final Grid Search Results are :')
print(rf_final_grid_search.best_params_)

rf_pruned = RandomForestClassifier(max_depth=20, bootstrap = True, max_features =3, min_samples_split= 3, n_estimators= 200)
rf_pruned.fit(X_train, y_train.values.ravel()) 
rf_y_pred = rf_pruned.predict(X_test)
print("Results for Pruned Random Forest Model with All Features are = ")
eval_metrics(y_test, rf_y_pred)

rf_all_features = np.argsort(rf_pruned.feature_importances_)[::-1]
rf_all_features = rf_all_features[0:10][::-1]
plt.barh(X_train.columns.values[rf_all_features], rf_pruned.feature_importances_[rf_all_features], color = 'deepskyblue')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("rf_all_feat.png", format="PNG", bbox_inches = "tight")
plt.clf()

#--------AdaBoost + All Features----------
adaboost_n_estimators =  np.linspace(start = 50, stop = 500, num = 50, dtype = int)
adaboost_learn_rate = [0.001, 0.05, 0.1, 0.5, 0.75, 1]
adaboost_random_gs_vals = {
    'n_estimators': adaboost_n_estimators,
    'learning_rate': adaboost_learn_rate
    }
adaboost_basic = AdaBoostClassifier()
adaboost_random_grid_search = RandomizedSearchCV(estimator = adaboost_basic, param_distributions = adaboost_random_gs_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
adaboost_random_grid_search.fit(X_train, y_train.values.ravel())
print('')
print('Adaboost Random Grid Search Results are :')
print(adaboost_random_grid_search.best_params_)

#Grid Search on narrowed down hyper-parameters
adaboost_final_gs_vals = {
    'n_estimators': [425, 450, 475, 500],
    'learning_rate': [ 0.7, 0.8, 0.9, 1]
}
adaboost_basic = AdaBoostClassifier()
adaboost_final_grid_search = GridSearchCV(estimator = adaboost_basic, param_grid = adaboost_final_gs_vals, cv = 5, verbose=2, n_jobs = -1)
adaboost_final_grid_search.fit(X_train, y_train.values.ravel())
print('')
print('Adaboost: Final Grid Search Results are :')
print(adaboost_final_grid_search.best_params_)

adaboost_tuned = AdaBoostClassifier(n_estimators = 475, learning_rate = 0.9)
adaboost_tuned.fit(X_train, y_train.values.ravel()) 
adaboost_y_pred = adaboost_tuned.predict(X_test)
print("Results for Tuned Adaboost Model with All Features are = ")
eval_metrics(y_test, adaboost_y_pred)

adaboost_all_features = np.argsort(adaboost_tuned.feature_importances_)[::-1]
adaboost_all_features = adaboost_all_features[0:10][::-1]
plt.barh(X_train.columns.values[adaboost_all_features], adaboost_tuned.feature_importances_[adaboost_all_features], color = 'cadetblue')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("adaboost_all_feat.png", format="PNG", bbox_inches = "tight")
plt.clf()

#----------GBM + All Features--------
#Random Grid Search
#learning_rate = shrinks contribution of each tree by lr
gb_learning_rate = [0.0001, 0.001, 0.05, 0.1, 0.5, 0.75, 1]
#n_estimators = Number of trees 
gb_n_estimators =  np.linspace(start = 1, stop = 200, num = 50, dtype = int)
#max_depth = represents the number of levels
gb_max_depth = np.linspace(5, 50, num = 10, dtype = int)
#min_samples_split = represents the minimum samples required for a split to occur
gb_min_samples_split = np.arange(2, 11)
#max_features = maximum features provided to each tree
gb_max_features = list(range(1, 31, 2))

gb_random_grid_search_vals = {'max_depth': gb_max_depth, 'n_estimators': gb_n_estimators, 'max_features': gb_max_features, 'min_samples_split': gb_min_samples_split, 'learning_rate': gb_learning_rate}

#Fitting a baseline GB model
gb_basic = GradientBoostingClassifier()
gb_random_grid_search = RandomizedSearchCV(estimator = gb_basic, param_distributions = gb_random_grid_search_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
gb_random_grid_search.fit(X_train, y_train.values.ravel())
print('')
print('GBM + All Features: Random Grid Search Results are :')
print(gb_random_grid_search.best_params_)

#Grid Search on narrowed down hyper-parameters
gb_final_gs_vals = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [10, 12, 15, 18],
    'max_features': [3, 6, 9],
    'min_samples_split': [4, 6, 8],
    'n_estimators': [175, 185, 195]
}

gb_final_grid_search = GridSearchCV(estimator = gb_basic, param_grid = gb_final_gs_vals, cv = 3, verbose=2, n_jobs = -1)
gb_final_grid_search.fit(X_train, y_train.values.ravel())
print('')
print('GBM + All Features: Final Grid Search Results are :')
print(gb_final_grid_search.best_params_)

gb_tuned = GradientBoostingClassifier(max_depth=15, learning_rate = 0.05, max_features =3, min_samples_split= 8, n_estimators= 175)
gb_tuned.fit(X_train, y_train.values.ravel()) 
gb_y_pred = gb_tuned.predict(X_test)
print("Results for Tuned Gradient Boosting Model with All Features are = ")
eval_metrics(y_test, gb_y_pred)

gb_all_features = np.argsort(gb_tuned.feature_importances_)[::-1]
gb_all_features = gb_all_features[0:10][::-1]
plt.barh(X_train.columns.values[gb_all_features], gb_tuned.feature_importances_[gb_all_features], color = 'cornflowerblue')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("gb_all_feat.png", format="PNG", bbox_inches = "tight")
plt.clf()


# --------- XGBoost + All Features--------
xgb_max_depth = np.linspace(5, 50, num = 10, dtype = int)
xgb_n_estimators = np.linspace(start = 1, stop = 200, num = 50, dtype = int)
xgb_learning_rate = [0.001, 0.05, 0.1, 0.5, 0.75, 1]
xgb_subsample = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
xgb_colsample_bytree = uniform(0.8, 0.2)

xgb_random_grid_search_vals = {'max_depth': xgb_max_depth, 'n_estimators': xgb_n_estimators, 'learning_rate': xgb_learning_rate, 'subsample': xgb_subsample, 'colsample_bytree': xgb_colsample_bytree}

xgb_basic = xgb.XGBClassifier(objective = "binary:logistic")
xgb_random_grid_search = RandomizedSearchCV(xgb_basic, param_distributions = xgb_random_grid_search_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
xgb_random_grid_search.fit(X_train, y_train.values.ravel())
print('')
print('Xtreme Gradient Boosting: Random Grid Search Results are :')
print(xgb_random_grid_search.best_params_)

#Grid Search on narrowed down hyper-parameters
xgb_final_gs_vals = {
    'learning_rate': [0.05, 0.07, 0.1],
    'max_depth': [45, 50, 55],
    'n_estimators': [185, 200, 215],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.8, 0.85, 0.9],
}

xgb_basic = xgb.XGBClassifier(objective = "binary:logistic")
xgb_final_grid_search = GridSearchCV(estimator = xgb_basic, param_grid = xgb_final_gs_vals, cv = 5, verbose=2, n_jobs = -1)
xgb_final_grid_search.fit(X_train, y_train.values.ravel())
print('')
print('Xtreme Gradient Boosting: Final Grid Search Results are :')
print(xgb_final_grid_search.best_params_)

xgb_tuned = xgb.XGBClassifier(objective = "binary:logistic", colsample_bytree = 0.9, learning_rate = 0.07, max_depth = 45, n_estimators = 185, subsample = 0.8)
xgb_tuned.fit(X_train, y_train.values.ravel()) 
xgb_y_pred = xgb_tuned.predict(X_test)
print("Results for Tuned XGBoost Model with All Features are = ")
eval_metrics(y_test, xgb_y_pred)

xgb_all_features = np.argsort(xgb_tuned.feature_importances_)[::-1]
xgb_all_features = xgb_all_features[0:10][::-1]
plt.barh(X_train.columns.values[xgb_all_features], xgb_tuned.feature_importances_[xgb_all_features], color = 'slateblue')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("xgb_all_feat.png", format="PNG", bbox_inches = "tight")
plt.clf()

#----------- Bagging + All Features ----------
#Hyperparameters
bag_n_estimators = np.linspace(start = 1, stop = 200, num = 50, dtype = int)
bag_max_samples = np.linspace(start = 1, stop = X_train.shape[1], num = 50, dtype = int)
bag_bootstrap = [True]

bag_random_grid_search_vals = {'n_estimators': bag_n_estimators, 'max_samples': bag_max_samples, 'bootstrap': bag_bootstrap }

#Fitting a baseline Bagging model
bag_basic = BaggingClassifier()
bag_random_grid_search = RandomizedSearchCV(estimator = bag_basic, param_distributions = bag_random_grid_search_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
bag_random_grid_search.fit(X_train, y_train.values.ravel())
print('')
print('Bagging: Random Grid Search Results are :')
print(bag_random_grid_search.best_params_)

#Grid Search on narrowed down hyper-parameters
bag_final_gs_vals = {
    'n_estimators': [85, 95, 105],
    'max_samples': [65, 50, 75],
    'bootstrap': [True]
}

bag_basic = BaggingClassifier()
bag_final_grid_search = GridSearchCV(estimator = bag_basic, param_grid = bag_final_gs_vals, cv = 5, verbose=2, n_jobs = -1)
bag_final_grid_search.fit(X_train, y_train.values.ravel())
print('')
print('Bagging: Final Grid Search Results are :')
print(bag_final_grid_search.best_params_)

bag_tuned = BaggingClassifier(n_estimators = 105, max_samples = 75, max_features = 73, bootstrap = False)
bag_tuned.fit(X_train, y_train.values.ravel()) 
bag_y_pred = bag_tuned.predict(X_test)
print("Results for Tuned Bagging Model with All Features are = ")
eval_metrics(y_test, bag_y_pred)

bag_all_features = np.mean([tree.feature_importances_ for tree in bag_tuned.estimators_], axis=0)
bag_imps = {fn: x for fn, x in zip(X_train.columns.values, bag_all_features)}
bag_sort_imps = sorted(bag_imps.items(), key=lambda x: x[1], reverse=False)
bag_sort_imps = bag_sort_imps[0:10]
names = []
vals = []
for imp_feat in bag_sort_imps:
    names.append(imp_feat[0])
    vals.append(imp_feat[1])
plt.barh(names, vals, color = 'lightseagreen')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("bag_all_feat.png", format="PNG", bbox_inches = "tight")
plt.clf()

#----------Feature Selection----------
#----------BORUTA-----------
#Using the pruned random forest model for boruta implementation
rf_pruned_v2 = RandomForestClassifier(max_depth=20, bootstrap = True, max_features =3, min_samples_split= 3, n_estimators= 200)
boruta_feature_selector = BorutaPy(rf_pruned_v2, n_estimators='auto', verbose=2, random_state=1)
boruta_feature_selector.fit(np.array(X_train), y_train.values.ravel())

#printing selected features
boruta_selected_features = X_train.columns[boruta_feature_selector.support_].to_list()
print('Features selected using Boruta are = ', boruta_selected_features)

#Boruta selected features only as dataframe
X_boruta_train = X_train[boruta_selected_features]
X_boruta_test = X_test[boruta_selected_features]

# -----------Maximum Relevance — Minimum Redundancy (mRMR)-----------------
mrmr_input = pd.concat([y_train, X_train], axis=1)
mrmr_features = pymrmr.mRMR(mrmr_input, 'MIQ', 17) #top 17 features
print('Features selected using mRMR (k = 17) are ', mrmr_features)

#mRMR selected features only as dataframe
X_mrmr_train = X_train[mrmr_features]
X_mrmr_test = X_test[mrmr_features]

#-----------Select-K-Best----------------------------
kbest_feature_selector = SelectKBest(f_classif, k = 22)
kbest_feature_selector.fit(X_train, y_train.values.ravel())
#printing selected features
feature_index = kbest_feature_selector.get_support(indices=True)
kbest_selected_features = X_train.columns[feature_index]
print("Features selected using SelectKBest (k = 22) are = ")
print(kbest_selected_features)
#SelectKBest selected features only as dataframe
X_kbest_train = X_train[kbest_selected_features]
X_kbest_test = X_test[kbest_selected_features]

# ------Models with Features Selected by Boruta------

#-------- Random Forest + Boruta Features ----------
#Hyper-Parameter Tuning- Step 1 - Random Grid Search
#max_depth = represents the number of levels
max_depth = np.linspace(8, 40, num = 25, dtype = int)
#n_estimators = Number of trees 
n_estimators =  np.linspace(start = 10, stop = 300, num = 25, dtype = int)
#max_features = maximum features provided to each tree
max_features = list(range(2, 31, 2))
#min_samples_split = represents the minimum samples required for a split to occur
min_samples_split = np.arange(2, 11)
bootstrap = [True]

rf_boruta_random_grid_search_vals = {'max_depth': max_depth, 'n_estimators': n_estimators, 'max_features': max_features, 'min_samples_split': min_samples_split, 'bootstrap': bootstrap}

#Fitting a baseline model
rf_basic_v2 = RandomForestClassifier()
rf_random_grid_search = RandomizedSearchCV(estimator = rf_basic_v2, param_distributions = rf_boruta_random_grid_search_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
rf_random_grid_search.fit(X_boruta_train, y_train.values.ravel())
print('')
print('Random Forest + Boruta: Random Grid Search Results are :')
print(rf_random_grid_search.best_params_)

#Step 2: Exhaustive Grid Search on narrowed down hyper-parameters
rf_boruta_final_gs_vals = {
    'bootstrap': [True],
    'max_depth': [14, 16, 18],
    'max_features': [2, 3, 4],
    'min_samples_split': [4, 5, 6],
    'n_estimators': [65, 70, 75]
}

rf_basic_v2 = RandomForestClassifier()
rf_boruta_final_grid_search = GridSearchCV(estimator = rf_basic_v2, param_grid = rf_boruta_final_gs_vals, cv = 5, verbose=2, n_jobs = -1)
rf_boruta_final_grid_search.fit(X_boruta_train, y_train.values.ravel())
print('')
print('Random Forest + Boruta: Final Grid Search Results are :')
print(rf_boruta_final_grid_search.best_params_)

rf_boruta_pruned = RandomForestClassifier(max_depth=18, bootstrap = True, max_features =3, min_samples_split= 4, n_estimators= 65)
rf_boruta_pruned.fit(X_boruta_train, y_train.values.ravel()) 
rf_boruta_y_pred = rf_boruta_pruned.predict(X_boruta_test)
print("Results for Random Forest + Boruta are = ")
eval_metrics(y_test, rf_boruta_y_pred)

rf_boruta_features = np.argsort(rf_boruta_pruned.feature_importances_)[::-1]
rf_boruta_features = rf_boruta_features[0:10][::-1]
plt.barh(X_boruta_train.columns.values[rf_boruta_features], rf_boruta_pruned.feature_importances_[rf_boruta_features], color = 'deepskyblue')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("rf_boruta.png", format="PNG", bbox_inches = "tight")
plt.clf()

#--------AdaBoost + Boruta Features----------
adaboost_n_estimators =  np.linspace(start = 50, stop = 500, num = 50, dtype = int)
adaboost_learn_rate = [0.001, 0.05, 0.1, 0.5, 0.75, 1]
adaboost_boruta_random_gs_vals = {
    'n_estimators': adaboost_n_estimators,
    'learning_rate': adaboost_learn_rate
    }
adaboost_basic_v2 = AdaBoostClassifier()
adaboost_boruta_random_grid_search = RandomizedSearchCV(estimator = adaboost_basic_v2, param_distributions = adaboost_boruta_random_gs_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
adaboost_boruta_random_grid_search.fit(X_boruta_train, y_train.values.ravel())
print('')
print('Adaboost + Boruta: Random Grid Search Results are :')
print(adaboost_boruta_random_grid_search.best_params_)

#Grid Search on narrowed down hyper-parameters
adaboost_boruta_final_gs_vals = {
    'n_estimators': [ 465, 475, 485],
    'learning_rate': [0.8, 0.9, 1]
}

adaboost_basic_v2 = AdaBoostClassifier()
adaboost_boruta_final_grid_search = GridSearchCV(estimator = adaboost_basic_v2, param_grid = adaboost_boruta_final_gs_vals, cv = 5, verbose=2, n_jobs = -1)
adaboost_boruta_final_grid_search.fit(X_boruta_train, y_train.values.ravel())
print('')
print('Adaboost + Boruta: Final Grid Search Results are :')
print(adaboost_boruta_final_grid_search.best_params_)

adaboost_boruta_tuned = AdaBoostClassifier(n_estimators = 475, learning_rate = 1)
adaboost_boruta_tuned.fit(X_boruta_train, y_train.values.ravel()) 
adaboost_boruta_y_pred = adaboost_boruta_tuned.predict(X_boruta_test)
print("Results for Adaboost with Boruta Features are = ")
eval_metrics(y_test, adaboost_boruta_y_pred)

adaboost_boruta_features = np.argsort(adaboost_boruta_tuned.feature_importances_)[::-1]
adaboost_boruta_features = adaboost_boruta_features[0:10][::-1]
plt.barh(X_boruta_train.columns.values[adaboost_boruta_features], adaboost_boruta_tuned.feature_importances_[adaboost_boruta_features], color = 'cadetblue')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("adaboost_boruta.png", format="PNG", bbox_inches = "tight")
plt.clf()

#----------GBM + Boruta Features--------
#Random Grid Search
#learning_rate = shrinks contribution of each tree by lr
gb_learning_rate = [0.0001, 0.001, 0.05, 0.1, 0.5, 0.75, 1]
#n_estimators = Number of trees 
gb_n_estimators =  np.linspace(start = 1, stop = 200, num = 50, dtype = int)
#max_depth = represents the number of levels
gb_max_depth = np.linspace(5, 50, num = 10, dtype = int)
#min_samples_split = represents the minimum samples required for a split to occur
gb_min_samples_split = np.arange(2, 11)
#max_features = maximum features provided to each tree
gb_max_features = list(range(1, 31, 2))

gb_boruta_random_grid_search_vals = {'max_depth': gb_max_depth, 'n_estimators': gb_n_estimators, 'max_features': gb_max_features, 'min_samples_split': gb_min_samples_split, 'learning_rate': gb_learning_rate}

#Fitting a baseline GB model
gb_basic = GradientBoostingClassifier()
gb_boruta_random_grid_search = RandomizedSearchCV(estimator = gb_basic, param_distributions = gb_boruta_random_grid_search_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
gb_boruta_random_grid_search.fit(X_boruta_train, y_train.values.ravel())
print('')
print('GBM + Boruta: Random Grid Search Results are :')
print(gb_boruta_random_grid_search.best_params_)

#Grid Search on narrowed down hyper-parameters
gb_boruta_final_gs_vals = {
    'learning_rate': [0.05, 0.1, 0.3],
    'max_depth': [12, 14, 16],
    'max_features': [4, 6, 8],
    'min_samples_split': [2, 3, 4],
    'n_estimators': [175, 185, 195]
}
gb_basic = GradientBoostingClassifier()
gb_boruta_final_grid_search = GridSearchCV(estimator = gb_basic, param_grid = gb_boruta_final_gs_vals, cv = 5, verbose=2, n_jobs = -1)
gb_boruta_final_grid_search.fit(X_boruta_train, y_train.values.ravel())
print('')
print('GBM + Boruta: Final Grid Search Results are :')
print(gb_boruta_final_grid_search.best_params_)

gb_boruta_tuned = GradientBoostingClassifier(max_depth=14, learning_rate = 0.1, max_features =4, min_samples_split= 4, n_estimators= 185)
gb_boruta_tuned.fit(X_boruta_train, y_train.values.ravel()) 
gb_boruta_y_pred = gb_boruta_tuned.predict(X_boruta_test)
print("Results for Tuned Gradient Boosting Model with Boruta Features are = ")
eval_metrics(y_test, gb_boruta_y_pred)

gb_boruta_features = np.argsort(gb_boruta_tuned.feature_importances_)[::-1]
gb_boruta_features = gb_boruta_features[0:10][::-1]
plt.barh(X_boruta_train.columns.values[gb_boruta_features], gb_boruta_tuned.feature_importances_[gb_boruta_features], color = 'cornflowerblue')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("gb_boruta.png", format="PNG", bbox_inches = "tight")
plt.clf()


# --------- XGBoost + Boruta Features--------
xgb_max_depth = np.linspace(5, 50, num = 10, dtype = int)
xgb_n_estimators = np.linspace(start = 1, stop = 200, num = 50, dtype = int)
xgb_learning_rate = [0.001, 0.05, 0.1, 0.5, 0.75, 1]
xgb_subsample = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
xgb_colsample_bytree = uniform(0.8, 0.2)

xgb_boruta_random_grid_search_vals = {'max_depth': xgb_max_depth, 'n_estimators': xgb_n_estimators, 'learning_rate': xgb_learning_rate, 'subsample': xgb_subsample, 'colsample_bytree': xgb_colsample_bytree}

xgb_basic = xgb.XGBClassifier(objective = "binary:logistic")
xgb_boruta_random_grid_search = RandomizedSearchCV(xgb_basic, param_distributions = xgb_boruta_random_grid_search_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
xgb_boruta_random_grid_search.fit(X_boruta_train, y_train.values.ravel())
print('')
print('XGB + Boruta: Random Grid Search Results are :')
print(xgb_boruta_random_grid_search.best_params_)

#Grid Search on narrowed down hyper-parameters
xgb_boruta_final_gs_vals = {
    'learning_rate': [0.05, 0.07, 0.1],
    'max_depth': [30, 35, 40],
    'n_estimators': [120, 130, 140],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.8, 0.85, 0.9],
}

xgb_basic = xgb.XGBClassifier(objective = "binary:logistic")
xgb_boruta_final_grid_search = GridSearchCV(estimator = xgb_basic, param_grid = xgb_boruta_final_gs_vals, cv = 5, verbose=2, n_jobs = -1)
xgb_boruta_final_grid_search.fit(X_boruta_train, y_train.values.ravel())
print('')
print('XGB + Boruta: Final Grid Search Results are :')
print(xgb_boruta_final_grid_search.best_params_)

xgb_boruta_tuned = xgb.XGBClassifier(objective = "binary:logistic", colsample_bytree = 0.9, learning_rate = 0.1, max_depth = 30, n_estimators = 140, subsample = 0.7)
xgb_boruta_tuned.fit(X_boruta_train, y_train.values.ravel()) 
xgb_boruta_y_pred = xgb_boruta_tuned.predict(X_boruta_test)
print("Results for Tuned XGBoost Model with Boruta Features are = ")
eval_metrics(y_test, xgb_boruta_y_pred)

xgb_boruta_features = np.argsort(xgb_boruta_tuned.feature_importances_)[::-1]
xgb_boruta_features = xgb_boruta_features[0:10][::-1]
plt.barh(X_boruta_train.columns.values[xgb_boruta_features], xgb_boruta_tuned.feature_importances_[xgb_boruta_features], color = 'slateblue')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("xgb_boruta.png", format="PNG", bbox_inches = "tight")
plt.clf()

#----------- Bagging + Boruta Features ----------
#Hyperparameters
bag_n_estimators = np.linspace(start = 1, stop = 200, num = 50, dtype = int)
bag_max_samples = np.linspace(start = 1, stop = X_train.shape[1], num = 50, dtype = int)
bag_bootstrap = [True]

bag_boruta_random_grid_search_vals = {'n_estimators': bag_n_estimators, 'max_samples': bag_max_samples, 'bootstrap': bag_bootstrap }

#Fitting a baseline Bagging model
bag_basic = BaggingClassifier()
bag_boruta_random_grid_search = RandomizedSearchCV(estimator = bag_basic, param_distributions = bag_boruta_random_grid_search_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
bag_boruta_random_grid_search.fit(X_boruta_train, y_train.values.ravel())
print('')
print('Bagging + Boruta: Random Grid Search Results are :')
print(bag_boruta_random_grid_search.best_params_)

#Grid Search on narrowed down hyper-parameters
bag_boruta_final_gs_vals = {
    'n_estimators': [145, 150, 155],
    'max_samples': [65, 70, 75],
    'bootstrap': [True]
}

bag_basic = BaggingClassifier()
bag_boruta_final_grid_search = GridSearchCV(estimator = bag_basic, param_grid = bag_boruta_final_gs_vals, cv = 5, verbose=2, n_jobs = -1)
bag_boruta_final_grid_search.fit(X_boruta_train, y_train.values.ravel())
print('')
print('Bagging + Boruta: Final Grid Search Results are :')
print(bag_boruta_final_grid_search.best_params_)

bag_boruta_tuned = BaggingClassifier(n_estimators = 145, max_samples = 75, max_features = 14, bootstrap = False)
bag_boruta_tuned.fit(X_boruta_train, y_train.values.ravel()) 
bag_boruta_y_pred = bag_boruta_tuned.predict(X_boruta_test)
print("Results for Tuned Bagging Model with Boruta Features are = ")
eval_metrics(y_test, bag_boruta_y_pred)

bag_boruta_features = np.mean([tree.feature_importances_ for tree in bag_boruta_tuned.estimators_], axis=0)
bag_boruta_imps = {fn: x for fn, x in zip(X_boruta_train.columns.values, bag_boruta_features)}
bag_sort_imps = sorted(bag_boruta_imps.items(), key=lambda x: x[1], reverse=False)
bag_sort_imps = bag_sort_imps[0:10]
names = []
vals = []
for imp_feat in bag_sort_imps:
    names.append(imp_feat[0])
    vals.append(imp_feat[1])
plt.barh(names, vals, color = 'lightseagreen')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("bag_boruta.png", format="PNG", bbox_inches = "tight")
plt.clf()


#-------- Random Forest + mRMR Features ----------
#Hyper-Parameter Tuning- Step 1 - Random Grid Search
#max_depth = represents the number of levels
max_depth = np.linspace(8, 40, num = 25, dtype = int)
#n_estimators = Number of trees 
n_estimators =  np.linspace(start = 10, stop = 300, num = 25, dtype = int)
#max_features = maximum features provided to each tree
max_features = list(range(2, 31, 2))
#min_samples_split = represents the minimum samples required for a split to occur
min_samples_split = np.arange(2, 11)
bootstrap = [True]

rf_mrmr_random_grid_search_vals = {'max_depth': max_depth, 'n_estimators': n_estimators, 'max_features': max_features, 'min_samples_split': min_samples_split, 'bootstrap': bootstrap}

#Fitting a baseline model
rf_basic_v2 = RandomForestClassifier()
rf_mrmr_random_grid_search = RandomizedSearchCV(estimator = rf_basic_v2, param_distributions = rf_mrmr_random_grid_search_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
rf_mrmr_random_grid_search.fit(X_mrmr_train, y_train.values.ravel())
print('')
print('Random Forest + mRMR: Random Grid Search Results are :')
print(rf_mrmr_random_grid_search.best_params_)

#Step 2: Exhaustive Grid Search on narrowed down hyper-parameters
rf_mrmr_final_gs_vals = {
    'bootstrap': [True],
    'max_depth': [22, 25, 27],
    'max_features': [4, 5, 6],
    'min_samples_split': [4, 5, 6],
    'n_estimators': [135, 140, 145]
}

rf_basic_v2 = RandomForestClassifier()
rf_mrmr_final_grid_search = GridSearchCV(estimator = rf_basic_v2, param_grid = rf_mrmr_final_gs_vals, cv = 5, verbose=2, n_jobs = -1)
rf_mrmr_final_grid_search.fit(X_mrmr_train, y_train.values.ravel())
print('')
print('Random Forest + mRMR: Final Grid Search Results are :')
print(rf_mrmr_final_grid_search.best_params_)

rf_mrmr_pruned = RandomForestClassifier(max_depth=27, bootstrap = True, max_features =4, min_samples_split= 4, n_estimators= 145)
rf_mrmr_pruned.fit(X_mrmr_train, y_train.values.ravel()) 
rf_mrmr_y_pred = rf_mrmr_pruned.predict(X_mrmr_test)
print("Results for Random Forest + mRMR are = ")
eval_metrics(y_test, rf_mrmr_y_pred)

rf_mrmr_features = np.argsort(rf_mrmr_pruned.feature_importances_)[::-1]
rf_mrmr_features = rf_mrmr_features[0:10][::-1]
plt.barh(X_mrmr_train.columns.values[rf_mrmr_features], rf_mrmr_pruned.feature_importances_[rf_mrmr_features], color = 'deepskyblue')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("rf_mrmr.png", format="PNG", bbox_inches = "tight")
plt.clf()

#--------AdaBoost +  mRMR Features----------
adaboost_n_estimators =  np.linspace(start = 50, stop = 500, num = 50, dtype = int)
adaboost_learn_rate = [0.001, 0.05, 0.1, 0.5, 0.75, 1]
adaboost_mrmr_random_gs_vals = {
    'n_estimators': adaboost_n_estimators,
    'learning_rate': adaboost_learn_rate
    }
adaboost_basic_v2 = AdaBoostClassifier()
adaboost_mrmr_random_grid_search = RandomizedSearchCV(estimator = adaboost_basic_v2, param_distributions = adaboost_mrmr_random_gs_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
adaboost_mrmr_random_grid_search.fit(X_mrmr_train, y_train.values.ravel())
print('')
print('Adaboost + mRMR: Random Grid Search Results are :')
print(adaboost_mrmr_random_grid_search.best_params_)

#Grid Search on narrowed down hyper-parameters
adaboost_mrmr_final_gs_vals = {
    'n_estimators': [ 465, 475, 485],
    'learning_rate': [0.8, 0.9, 1]
}

adaboost_basic_v2 = AdaBoostClassifier()
adaboost_mrmr_final_grid_search = GridSearchCV(estimator = adaboost_basic_v2, param_grid = adaboost_mrmr_final_gs_vals, cv = 5, verbose=2, n_jobs = -1)
adaboost_mrmr_final_grid_search.fit(X_mrmr_train, y_train.values.ravel())
print('')
print('Adaboost + mRMR: Final Grid Search Results are :')
print(adaboost_mrmr_final_grid_search.best_params_)

adaboost_mrmr_tuned = AdaBoostClassifier(n_estimators = 485, learning_rate = 1)
adaboost_mrmr_tuned.fit(X_mrmr_train, y_train.values.ravel()) 
adaboost_mrmr_y_pred = adaboost_mrmr_tuned.predict(X_mrmr_test)
print("Results for Adaboost with mRMR Features are = ")
eval_metrics(y_test, adaboost_mrmr_y_pred)

adaboost_mrmr_features = np.argsort(adaboost_mrmr_tuned.feature_importances_)[::-1]
adaboost_mrmr_features = adaboost_mrmr_features[0:10][::-1]
plt.barh(X_mrmr_train.columns.values[adaboost_mrmr_features], adaboost_mrmr_tuned.feature_importances_[adaboost_mrmr_features], color = 'cadetblue')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("adaboost_mrmr.png", format="PNG", bbox_inches = "tight")
plt.clf()

#----------GBM + mRMR Features--------
#Random Grid Search
#learning_rate = shrinks contribution of each tree by lr
gb_learning_rate = [0.0001, 0.001, 0.05, 0.1, 0.5, 0.75, 1]
#n_estimators = Number of trees 
gb_n_estimators =  np.linspace(start = 1, stop = 200, num = 50, dtype = int)
#max_depth = represents the number of levels
gb_max_depth = np.linspace(5, 50, num = 10, dtype = int)
#min_samples_split = represents the minimum samples required for a split to occur
gb_min_samples_split = np.arange(2, 11)
#max_features = maximum features provided to each tree
gb_max_features = list(range(1, 31, 2))

gb_mrmr_random_grid_search_vals = {'max_depth': gb_max_depth, 'n_estimators': gb_n_estimators, 'max_features': gb_max_features, 'min_samples_split': gb_min_samples_split, 'learning_rate': gb_learning_rate}

#Fitting a baseline GB model
gb_basic = GradientBoostingClassifier()
gb_mrmr_random_grid_search = RandomizedSearchCV(estimator = gb_basic, param_distributions = gb_mrmr_random_grid_search_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
gb_mrmr_random_grid_search.fit(X_mrmr_train, y_train.values.ravel())
print('')
print('GBM + mRMR: Random Grid Search Results are :')
print(gb_mrmr_random_grid_search.best_params_)

#Grid Search on narrowed down hyper-parameters
gb_mrmr_final_gs_vals = {
    'learning_rate': [0.1, 0.3, 0.5],
    'max_depth': [35, 40, 45],
    'max_features': [3, 4, 5],
    'min_samples_split': [4, 6, 8],
    'n_estimators': [70, 75, 80]
}
gb_basic = GradientBoostingClassifier()
gb_mrmr_final_grid_search = GridSearchCV(estimator = gb_basic, param_grid = gb_mrmr_final_gs_vals, cv = 5, verbose=2, n_jobs = -1)
gb_mrmr_final_grid_search.fit(X_mrmr_train, y_train.values.ravel())
print('')
print('GBM + mRMR: Final Grid Search Results are :')
print(gb_mrmr_final_grid_search.best_params_)

gb_mrmr_tuned = GradientBoostingClassifier(max_depth=40, learning_rate = 0.3, max_features =4, min_samples_split= 6, n_estimators= 75)
gb_mrmr_tuned.fit(X_mrmr_train, y_train.values.ravel()) 
gb_mrmr_y_pred = gb_mrmr_tuned.predict(X_mrmr_test)
print("Results for Tuned Gradient Boosting Model with mRMR Features are = ")
eval_metrics(y_test, gb_mrmr_y_pred)

gb_mrmr_features = np.argsort(gb_mrmr_tuned.feature_importances_)[::-1]
gb_mrmr_features = gb_mrmr_features[0:10][::-1]
plt.barh(X_mrmr_train.columns.values[gb_mrmr_features], gb_mrmr_tuned.feature_importances_[gb_mrmr_features], color = 'cornflowerblue')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("gb_mrmr.png", format="PNG", bbox_inches = "tight")
plt.clf()

# --------- XGBoost + mRMR Features--------
xgb_max_depth = np.linspace(5, 50, num = 10, dtype = int)
xgb_n_estimators = np.linspace(start = 1, stop = 200, num = 50, dtype = int)
xgb_learning_rate = [0.001, 0.05, 0.1, 0.5, 0.75, 1]
xgb_subsample = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
xgb_colsample_bytree = uniform(0.8, 0.2)

xgb_mrmr_random_grid_search_vals = {'max_depth': xgb_max_depth, 'n_estimators': xgb_n_estimators, 'learning_rate': xgb_learning_rate, 'subsample': xgb_subsample, 'colsample_bytree': xgb_colsample_bytree}

xgb_basic = xgb.XGBClassifier(objective = "binary:logistic")
xgb_mrmr_random_grid_search = RandomizedSearchCV(xgb_basic, param_distributions = xgb_mrmr_random_grid_search_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
xgb_mrmr_random_grid_search.fit(X_mrmr_train, y_train.values.ravel())
print('')
print('XGB + mRMR: Random Grid Search Results are :')
print(xgb_mrmr_random_grid_search.best_params_)

#Grid Search on narrowed down hyper-parameters
xgb_mrmr_final_gs_vals = {
    'learning_rate': [0.01, 0.05, 0.07],
    'max_depth': [30, 35, 40],
    'n_estimators': [185, 200, 215],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.8, 0.85, 0.9],
}

xgb_basic = xgb.XGBClassifier(objective = "binary:logistic")
xgb_mrmr_final_grid_search = GridSearchCV(estimator = xgb_basic, param_grid = xgb_mrmr_final_gs_vals, cv = 5, verbose=2, n_jobs = -1)
xgb_mrmr_final_grid_search.fit(X_mrmr_train, y_train.values.ravel())
print('')
print('XGB + mRMR: Final Grid Search Results are :')
print(xgb_mrmr_final_grid_search.best_params_)

xgb_mrmr_tuned = xgb.XGBClassifier(objective = "binary:logistic", colsample_bytree = 0.8, learning_rate = 0.01, max_depth = 35, n_estimators = 215, subsample = 0.9)
xgb_mrmr_tuned.fit(X_mrmr_train, y_train.values.ravel()) 
xgb_mrmr_y_pred = xgb_mrmr_tuned.predict(X_mrmr_test)
print("Results for Tuned XGBoost Model with mRMR Features are = ")
eval_metrics(y_test, xgb_mrmr_y_pred)

xgb_mrmr_features = np.argsort(xgb_mrmr_tuned.feature_importances_)[::-1]
xgb_mrmr_features = xgb_mrmr_features[0:10][::-1]
plt.barh(X_mrmr_train.columns.values[xgb_mrmr_features], xgb_mrmr_tuned.feature_importances_[xgb_mrmr_features], color = 'slateblue')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("xgb_mrmr.png", format="PNG", bbox_inches = "tight")
plt.clf()

#----------- Bagging + mRMR Features ----------
#Hyperparameters
bag_n_estimators = np.linspace(start = 1, stop = 200, num = 50, dtype = int)
bag_max_samples = np.linspace(start = 1, stop = X_train.shape[1], num = 50, dtype = int)
bag_bootstrap = [True]

bag_mrmr_random_grid_search_vals = {'n_estimators': bag_n_estimators, 'max_samples': bag_max_samples, 'bootstrap': bag_bootstrap }

#Fitting a baseline Bagging model
bag_basic = BaggingClassifier()
bag_mrmr_random_grid_search = RandomizedSearchCV(estimator = bag_basic, param_distributions = bag_mrmr_random_grid_search_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
bag_mrmr_random_grid_search.fit(X_mrmr_train, y_train.values.ravel())
print('')
print('Bagging + mRMR: Random Grid Search Results are :')
print(bag_mrmr_random_grid_search.best_params_)

#Grid Search on narrowed down hyper-parameters
bag_mrmr_final_gs_vals = {
    'n_estimators': [90, 95, 100],
    'max_samples': [60, 65, 70],
    'bootstrap': [True]
}

bag_basic = BaggingClassifier()
bag_mrmr_final_grid_search = GridSearchCV(estimator = bag_basic, param_grid = bag_mrmr_final_gs_vals, cv = 5, verbose=2, n_jobs = -1)
bag_mrmr_final_grid_search.fit(X_mrmr_train, y_train.values.ravel())
print('')
print('Bagging + mRMR: Final Grid Search Results are :')
print(bag_mrmr_final_grid_search.best_params_)

bag_mrmr_tuned = BaggingClassifier(n_estimators = 95, max_samples = 70, max_features = 17, bootstrap = False)
bag_mrmr_tuned.fit(X_mrmr_train, y_train.values.ravel()) 
bag_mrmr_y_pred = bag_mrmr_tuned.predict(X_mrmr_test)
print("Results for Tuned Bagging Model with mRMR Features are = ")
eval_metrics(y_test, bag_mrmr_y_pred)

bag_mrmr_features = np.mean([tree.feature_importances_ for tree in bag_mrmr_tuned.estimators_], axis=0)
bag_mrmr_imps = {fn: x for fn, x in zip(X_mrmr_train.columns.values, bag_mrmr_features)}
bag_sort_imps = sorted(bag_mrmr_imps.items(), key=lambda x: x[1], reverse=False)
bag_sort_imps = bag_sort_imps[0:10]
names = []
vals = []
for imp_feat in bag_sort_imps:
    names.append(imp_feat[0])
    vals.append(imp_feat[1])
plt.barh(names, vals, color = 'lightseagreen')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("bag_mrmr.png", format="PNG", bbox_inches = "tight")
plt.clf()

#-------- Random Forest + SelectKBest Features ----------
#Hyper-Parameter Tuning- Step 1 - Random Grid Search
#max_depth = represents the number of levels
max_depth = np.linspace(8, 40, num = 25, dtype = int)
#n_estimators = Number of trees 
n_estimators =  np.linspace(start = 10, stop = 300, num = 25, dtype = int)
#max_features = maximum features provided to each tree
max_features = list(range(2, 31, 2))
#min_samples_split = represents the minimum samples required for a split to occur
min_samples_split = np.arange(2, 11)
bootstrap = [True]

rf_kbest_random_grid_search_vals = {'max_depth': max_depth, 'n_estimators': n_estimators, 'max_features': max_features, 'min_samples_split': min_samples_split, 'bootstrap': bootstrap}

#Fitting a baseline model
rf_basic_v2 = RandomForestClassifier()
rf_kbest_random_grid_search = RandomizedSearchCV(estimator = rf_basic_v2, param_distributions = rf_kbest_random_grid_search_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
rf_kbest_random_grid_search.fit(X_kbest_train, y_train.values.ravel())
print('')
print('Random Forest + SelectKBest: Random Grid Search Results are :')
print(rf_kbest_random_grid_search.best_params_)

#Step 2: Exhaustive Grid Search on narrowed down hyper-parameters
rf_kbest_final_gs_vals = {
    'bootstrap': [True],
    'max_depth': [22, 25, 27],
    'max_features': [4, 5, 6],
    'min_samples_split': [4, 5, 6],
    'n_estimators': [140, 145, 150]
}

rf_basic_v2 = RandomForestClassifier()
rf_kbest_final_grid_search = GridSearchCV(estimator = rf_basic_v2, param_grid = rf_kbest_final_gs_vals, cv = 5, verbose=2, n_jobs = -1)
rf_kbest_final_grid_search.fit(X_kbest_train, y_train.values.ravel())
print('')
print('Random Forest + SekectKBest: Final Grid Search Results are :')
print(rf_kbest_final_grid_search.best_params_)

rf_kbest_pruned = RandomForestClassifier(max_depth=25, bootstrap = True, max_features =4, min_samples_split= 5, n_estimators= 150)
rf_kbest_pruned.fit(X_kbest_train, y_train.values.ravel()) 
rf_kbest_y_pred = rf_kbest_pruned.predict(X_kbest_test)
print("Results for Random Forest + SekectKBest are = ")
eval_metrics(y_test, rf_kbest_y_pred)

rf_kbest_features = np.argsort(rf_kbest_pruned.feature_importances_)[::-1]
rf_kbest_features = rf_kbest_features[0:10][::-1]
plt.barh(X_kbest_train.columns.values[rf_kbest_features], rf_kbest_pruned.feature_importances_[rf_kbest_features], color = 'deepskyblue')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("rf_kbest.png", format="PNG", bbox_inches = "tight")
plt.clf()

#--------AdaBoost +  SelectKBest Features----------
adaboost_n_estimators =  np.linspace(start = 50, stop = 500, num = 50, dtype = int)
adaboost_learn_rate = [0.001, 0.05, 0.1, 0.5, 0.75, 1]
adaboost_kbest_random_gs_vals = {
    'n_estimators': adaboost_n_estimators,
    'learning_rate': adaboost_learn_rate
    }
adaboost_basic_v2 = AdaBoostClassifier()
adaboost_kbest_random_grid_search = RandomizedSearchCV(estimator = adaboost_basic_v2, param_distributions = adaboost_kbest_random_gs_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
adaboost_kbest_random_grid_search.fit(X_kbest_train, y_train.values.ravel())
print('')
print('Adaboost + SelectKBest: Random Grid Search Results are :')
print(adaboost_kbest_random_grid_search.best_params_)

#Grid Search on narrowed down hyper-parameters
adaboost_kbest_final_gs_vals = {
    'n_estimators': [ 430, 445, 460],
    'learning_rate': [0.9, 0.95, 1]
}

adaboost_basic_v2 = AdaBoostClassifier()
adaboost_kbest_final_grid_search = GridSearchCV(estimator = adaboost_basic_v2, param_grid = adaboost_kbest_final_gs_vals, cv = 5, verbose=2, n_jobs = -1)
adaboost_kbest_final_grid_search.fit(X_kbest_train, y_train.values.ravel())
print('')
print('Adaboost + SelectKBest: Final Grid Search Results are :')
print(adaboost_kbest_final_grid_search.best_params_)

adaboost_kbest_tuned = AdaBoostClassifier(n_estimators = 445, learning_rate = 1)
adaboost_kbest_tuned.fit(X_kbest_train, y_train.values.ravel()) 
adaboost_kbest_y_pred = adaboost_kbest_tuned.predict(X_kbest_test)
print("Results for Adaboost with SelectKBest Features are = ")
eval_metrics(y_test, adaboost_kbest_y_pred)

adaboost_kbest_features = np.argsort(adaboost_kbest_tuned.feature_importances_)[::-1]
adaboost_kbest_features = adaboost_kbest_features[0:10][::-1]
plt.barh(X_kbest_train.columns.values[adaboost_kbest_features], adaboost_kbest_tuned.feature_importances_[adaboost_kbest_features], color = 'cadetblue')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("adaboost_kbest.png", format="PNG", bbox_inches = "tight")
plt.clf()

#----------GBM + SelectKBest Features--------
#Random Grid Search
#learning_rate = shrinks contribution of each tree by lr
gb_learning_rate = [0.0001, 0.001, 0.05, 0.1, 0.5, 0.75, 1]
#n_estimators = Number of trees 
gb_n_estimators =  np.linspace(start = 1, stop = 200, num = 50, dtype = int)
#max_depth = represents the number of levels
gb_max_depth = np.linspace(5, 50, num = 10, dtype = int)
#min_samples_split = represents the minimum samples required for a split to occur
gb_min_samples_split = np.arange(2, 11)
#max_features = maximum features provided to each tree
gb_max_features = list(range(1, 31, 2))

gb_kbest_random_grid_search_vals = {'max_depth': gb_max_depth, 'n_estimators': gb_n_estimators, 'max_features': gb_max_features, 'min_samples_split': gb_min_samples_split, 'learning_rate': gb_learning_rate}

#Fitting a baseline GB model
gb_basic = GradientBoostingClassifier()
gb_kbest_random_grid_search = RandomizedSearchCV(estimator = gb_basic, param_distributions = gb_kbest_random_grid_search_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
gb_kbest_random_grid_search.fit(X_kbest_train, y_train.values.ravel())
print('')
print('GBM + SelectKBest: Random Grid Search Results are :')
print(gb_kbest_random_grid_search.best_params_)

#Grid Search on narrowed down hyper-parameters
gb_kbest_final_gs_vals = {
    'learning_rate': [0.01, 0.05, 0.07],
    'max_depth': [12, 15, 18],
    'max_features': [4, 6, 8],
    'min_samples_split': [2, 3, 4],
    'n_estimators': [160, 165, 170]
}
gb_basic = GradientBoostingClassifier()
gb_kbest_final_grid_search = GridSearchCV(estimator = gb_basic, param_grid = gb_kbest_final_gs_vals, cv = 3, verbose=2, n_jobs = -1)
gb_kbest_final_grid_search.fit(X_kbest_train, y_train.values.ravel())
print('')
print('GBM + SelectKBest: Final Grid Search Results are :')
print(gb_kbest_final_grid_search.best_params_)

gb_kbest_tuned = GradientBoostingClassifier(max_depth=15, learning_rate = 0.05, max_features = 6, min_samples_split= 2, n_estimators= 165)
gb_kbest_tuned.fit(X_kbest_train, y_train.values.ravel()) 
gb_kbest_y_pred = gb_kbest_tuned.predict(X_kbest_test)
print("Results for Tuned Gradient Boosting Model with SelectKBest Features are = ")
eval_metrics(y_test, gb_kbest_y_pred)

gb_kbest_features = np.argsort(gb_kbest_tuned.feature_importances_)[::-1]
gb_kbest_features = gb_kbest_features[0:10][::-1]
plt.barh(X_kbest_train.columns.values[gb_kbest_features], gb_kbest_tuned.feature_importances_[gb_kbest_features], color = 'cornflowerblue')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("gb_kbest.png", format="PNG", bbox_inches = "tight")
plt.clf()

# --------- XGBoost + SelectKBest Features--------
xgb_max_depth = np.linspace(5, 50, num = 10, dtype = int)
xgb_n_estimators = np.linspace(start = 1, stop = 200, num = 50, dtype = int)
xgb_learning_rate = [0.001, 0.05, 0.1, 0.5, 0.75, 1]
xgb_subsample = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
xgb_colsample_bytree = uniform(0.8, 0.2)

xgb_kbest_random_grid_search_vals = {'max_depth': xgb_max_depth, 'n_estimators': xgb_n_estimators, 'learning_rate': xgb_learning_rate, 'subsample': xgb_subsample, 'colsample_bytree': xgb_colsample_bytree}

xgb_basic = xgb.XGBClassifier(objective = "binary:logistic")
xgb_kbest_random_grid_search = RandomizedSearchCV(xgb_basic, param_distributions = xgb_kbest_random_grid_search_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
xgb_kbest_random_grid_search.fit(X_kbest_train, y_train.values.ravel())
print('')
print('XGB + SelectKBest: Random Grid Search Results are :')
print(xgb_kbest_random_grid_search.best_params_)

#Grid Search on narrowed down hyper-parameters
xgb_kbest_final_gs_vals = {
    'learning_rate': [0.01, 0.05, 0.07],
    'max_depth': [20, 25, 30],
    'n_estimators': [25, 30, 35],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.8, 0.85, 0.9],
}

xgb_basic = xgb.XGBClassifier(objective = "binary:logistic")
xgb_kbest_final_grid_search = GridSearchCV(estimator = xgb_basic, param_grid = xgb_kbest_final_gs_vals, cv = 5, verbose=2, n_jobs = -1)
xgb_kbest_final_grid_search.fit(X_kbest_train, y_train.values.ravel())
print('')
print('XGB + SelectKBest: Final Grid Search Results are :')
print(xgb_kbest_final_grid_search.best_params_)

xgb_kbest_tuned = xgb.XGBClassifier(objective = "binary:logistic", colsample_bytree = 0.9, learning_rate = 0.07, max_depth = 20, n_estimators = 35, subsample = 0.9)
xgb_kbest_tuned.fit(X_kbest_train, y_train.values.ravel()) 
xgb_kbest_y_pred = xgb_kbest_tuned.predict(X_kbest_test)
print("Results for Tuned XGBoost Model with SelectKBest Features are = ")
eval_metrics(y_test, xgb_kbest_y_pred)

xgb_kbest_features = np.argsort(xgb_kbest_tuned.feature_importances_)[::-1]
xgb_kbest_features = xgb_kbest_features[0:10][::-1]
plt.barh(X_kbest_train.columns.values[xgb_kbest_features], xgb_kbest_tuned.feature_importances_[xgb_kbest_features], color = 'slateblue')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("xgb_kbest.png", format="PNG", bbox_inches = "tight")
plt.clf()

#----------- Bagging + SelectKBest Features ----------
#Hyperparameters
bag_n_estimators = np.linspace(start = 1, stop = 200, num = 50, dtype = int)
bag_max_samples = np.linspace(start = 1, stop = X_train.shape[1], num = 50, dtype = int)
bag_bootstrap = [True]

bag_kbest_random_grid_search_vals = {'n_estimators': bag_n_estimators, 'max_samples': bag_max_samples, 'bootstrap': bag_bootstrap }

#Fitting a baseline Bagging model
bag_basic = BaggingClassifier()
bag_kbest_random_grid_search = RandomizedSearchCV(estimator = bag_basic, param_distributions = bag_kbest_random_grid_search_vals, n_iter = 75, cv = 5, verbose=2, random_state=123, n_jobs = -1)
bag_kbest_random_grid_search.fit(X_kbest_train, y_train.values.ravel())
print('')
print('Bagging + SelectKBest: Random Grid Search Results are :')
print(bag_kbest_random_grid_search.best_params_)

#Grid Search on narrowed down hyper-parameters
bag_kbest_final_gs_vals = {
    'n_estimators': [160, 165, 170],
    'max_samples': [60, 65, 70],
    'bootstrap': [True]
}

bag_basic = BaggingClassifier()
bag_kbest_final_grid_search = GridSearchCV(estimator = bag_basic, param_grid = bag_kbest_final_gs_vals, cv = 5, verbose=2, n_jobs = -1)
bag_kbest_final_grid_search.fit(X_kbest_train, y_train.values.ravel())
print('')
print('Bagging + SelectKBest: Final Grid Search Results are :')
print(bag_kbest_final_grid_search.best_params_)

bag_kbest_tuned = BaggingClassifier(n_estimators = 165, max_samples = 70, max_features = 22, bootstrap = False)
bag_kbest_tuned.fit(X_kbest_train, y_train.values.ravel()) 
bag_kbest_y_pred = bag_kbest_tuned.predict(X_kbest_test)
print("Results for Tuned Bagging Model with SelectKBest Features are = ")
eval_metrics(y_test, bag_kbest_y_pred)

bag_kbest_features = np.mean([tree.feature_importances_ for tree in bag_kbest_tuned.estimators_], axis=0)
bag_kbest_imps = {fn: x for fn, x in zip(X_kbest_train.columns.values, bag_kbest_features)}
bag_sort_imps = sorted(bag_kbest_imps.items(), key=lambda x: x[1], reverse=False)
bag_sort_imps = bag_sort_imps[0:10]
names = []
vals = []
for imp_feat in bag_sort_imps:
    names.append(imp_feat[0])
    vals.append(imp_feat[1])
plt.barh(names, vals, color = 'lightseagreen')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature Importance')
plt.savefig("bag_kbest.png", format="PNG", bbox_inches = "tight")
plt.clf()

# Best Performing models using Shap:
# XGBoost + All Features
xgb_all_explainer = shap.Explainer(xgb_tuned.predict, X_test)
xgb_shap_values = xgb_all_explainer(X_test)
f = plt.figure()
shap.plots.beeswarm(xgb_shap_values, max_display=11)
f.savefig("xgb_all_features_shap.png", format="PNG", bbox_inches = "tight")

# XGBoost + SelectKBest Features
xgb_kbest_explainer = shap.Explainer(xgb_kbest_tuned.predict, X_kbest_test)
xgb_shap_values_kbest = xgb_kbest_explainer(X_kbest_test)
f1 = plt.figure()
shap.plots.beeswarm(xgb_shap_values_kbest, max_display=11)
f1.savefig("xgb_kbest_features_shap.png", format="PNG", bbox_inches = "tight")

# Random Forest + Boruta Features
rf_boruta_explainer = shap.Explainer(rf_boruta_pruned.predict, X_boruta_test)
rf_shap_values_boruta = rf_boruta_explainer(X_boruta_test)
f2 = plt.figure()
shap.plots.beeswarm(rf_shap_values_boruta, max_display=11)
f2.savefig("rf_boruta_features_shap.png", format="PNG", bbox_inches = "tight")

# XGBoost + mRMR Features
xgb_mrmr_explainer = shap.Explainer(xgb_mrmr_tuned.predict, X_mrmr_test)
xgb_shap_values_mrmr = xgb_mrmr_explainer(X_mrmr_test)
f3 = plt.figure()
shap.plots.beeswarm(xgb_shap_values_mrmr, max_display=11)
plt.rcParams.update({'font.size': 12})
f3.savefig("xgb_mrmr_features_shap.png", format="PNG", bbox_inches = "tight")

print('Execution Complete.')