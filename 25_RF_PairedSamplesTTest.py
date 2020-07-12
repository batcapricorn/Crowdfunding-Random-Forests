#--------------------- Random Forests - Bachelor Thesis --------------------#

#---------------------------- Paired-Samples T-Test ------------------------#

#In the following scrypt, a paired-samples t-test with different models that 
#use distinct features will be computed.

#---------------------------------- Code -----------------------------------#

#------------ General Part ------------#
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from scipy import stats

#Variables
wd_main = os.getcwd()
os.mkdir('25_RF_PairedSamplesTTest_Export')
wd_data = wd_main + '\\12_DataProcessing_Export'
wd_export = wd_main + '\\25_RF_PairedSamplesTTest_Export'

#Data
os.chdir(wd_data)
df = pickle.load(open('model_data.pkl', 'rb'))
X = df.iloc[:, 1:-2]
y_cla = df.loc[:, 'successful']
y_reg = df.loc[:, 'funding_rate']

#Feature groups
feature_groups = {'basic_information': ['goal_usd', 'duration'],
                  'descriptive_information': ['name_length', 'blurb_gradelevel',
                                              'category'],
                  'geographic_information': ['country', 'state', 'city']}

#------------- Classifier -------------#
os.chdir(wd_export)

    #subsets
for group in feature_groups.keys():
    export_name = 'cla_' + group + '.pkl'
    cla = RandomForestClassifier(n_estimators=400, criterion='gini', max_features='sqrt')
    cv_results = cross_validate(cla, X[feature_groups[group]], y_cla, 
                                cv=30, scoring='accuracy')
    pickle.dump(cv_results, open(export_name, 'wb'))
    
    #all features
cla = RandomForestClassifier(n_estimators=400, criterion='gini', max_features='sqrt')
cv_results = cross_validate(cla, X, y_cla, cv=30, scoring='accuracy')
pickle.dump(cv_results, open('cla_all_information.pkl', 'wb'))

#------------- Regression -------------# 
#R2
    #subsets
for group in feature_groups.keys():
    export_name = 'reg_r2_' + group + '.pkl'
    reg = RandomForestRegressor(n_estimators=400, criterion='mse', max_features='sqrt')
    cv_results = cross_validate(reg, X[feature_groups[group]], y_reg, 
                                cv=30, scoring='r2')
    pickle.dump(cv_results, open(export_name, 'wb'))    

    #all features
reg = RandomForestRegressor(n_estimators=400, criterion='mse', max_features='sqrt')
cv_results = cross_validate(reg, X, y_reg, cv=30, scoring='r2')
pickle.dump(cv_results, open('reg_r2_all_information.pkl', 'wb'))

#Log. MSE
    #subsets
for group in feature_groups.keys():
    export_name = 'reg_logmse_' + group + '.pkl'
    reg = RandomForestRegressor(n_estimators=400, criterion='mse', max_features='sqrt')
    cv_results = cross_validate(reg, X[feature_groups[group]], y_reg, 
                                cv=30, scoring='neg_mean_squared_log_error')
    pickle.dump(cv_results, open(export_name, 'wb'))    

    #all features
reg = RandomForestRegressor(n_estimators=400, criterion='mse', max_features='sqrt')
cv_results = cross_validate(reg, X, y_reg, cv=30, scoring='neg_mean_squared_log_error')
pickle.dump(cv_results, open('reg_logmse_all_information.pkl', 'wb'))

#Evaluation
cv_results_li = glob.glob('*.pkl')
cv_results = {}
for item in cv_results_li:
    name = item.split('.')[0]
    dict_cv = pickle.load(open(item, 'rb'))
    cv_results[name] = dict_cv['test_score']
    
    #Classifier (own plot)
sns.set(palette='viridis')
sns.kdeplot(cv_results['cla_all_information'], label='All Information')
sns.kdeplot(cv_results['cla_basic_information'], label='Basic Information')
sns.kdeplot(cv_results['cla_descriptive_information'], label='Descriptive Information')
sns.kdeplot(cv_results['cla_geographic_information'], label='Geographic Information')
plt.xlabel('Accuracy')
plt.title('Classifier')

    #Classifier (subplot)
fig = plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
sns.set(palette='viridis')
sns.kdeplot(cv_results['cla_all_information'], label='All Information')
sns.kdeplot(cv_results['cla_basic_information'], label='Basic Information')
sns.kdeplot(cv_results['cla_descriptive_information'], label='Descriptive Information')
sns.kdeplot(cv_results['cla_geographic_information'], label='Geographic Information')
plt.xlabel('Accuracy')
plt.title('Classifier')


    #Regressor (subplot)
plt.subplot(1,2,2)
sns.set(palette='viridis')
sns.kdeplot(cv_results['reg_r2_all_information'], label='All Information')
sns.kdeplot(cv_results['reg_r2_basic_information'], label='Basic Information')
sns.kdeplot(cv_results['reg_r2_descriptive_information'], label='Descriptive Information')
sns.kdeplot(cv_results['reg_r2_geographic_information'], label='Geographic Information')
plt.xlabel('R2')
plt.title('Regressor')

plt.show()
plt.close()

#T-Test
    #Classifier
classifier_li = ['cla_all_information', 'cla_basic_information', 
                 'cla_descriptive_information', 'cla_geographic_information']

samplestest_classifier = pd.DataFrame()
means = [cv_results[x].mean() for x in classifier_li]
samplestest_classifier['mean'] = means
samplestest_classifier.index = classifier_li
for item in classifier_li:
    samplestest_classifier[item] = np.nan
    
for left_item in classifier_li:
    for right_item in classifier_li:
        if left_item == right_item:
            continue
        else:
            t_value, p_value = stats.ttest_ind(cv_results[left_item],
                                               cv_results[right_item])
            if p_value <= 0.01:
                result = str(t_value) + '***'
            elif p_value <= 0.05:
                result = str(t_value) + '**'
            elif p_value <= 0.1:
                result = str(t_value) + '*'
            else:
                result = t_value
            samplestest_classifier.loc[left_item, right_item] = result
        
samplestest_classifier.to_excel('samplestest_classifier.xlsx')

    #Regression
regression_li = ['reg_r2_all_information', 'reg_r2_basic_information',
                 'reg_r2_descriptive_information', 'reg_r2_geographic_information']

samplestest_regression = pd.DataFrame()
means = [cv_results[x].mean() for x in regression_li]
samplestest_regression['mean'] = means
samplestest_regression.index = regression_li
for item in regression_li:
    samplestest_regression[item] = np.nan
    
for left_item in regression_li:
    for right_item in regression_li:
        if left_item == right_item:
            continue
        else:
            t_value, p_value = stats.ttest_ind(cv_results[left_item],
                                               cv_results[right_item])
            if p_value <= 0.01:
                result = str(t_value) + '***'
            elif p_value <= 0.05:
                result = str(t_value) + '**'
            elif p_value <= 0.1:
                result = str(t_value) + '*'
            else:
                result = t_value
            samplestest_regression.loc[left_item, right_item] = result
        
samplestest_regression.to_excel('samplestest_regression.xlsx')