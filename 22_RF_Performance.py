#--------------------- Random Forests - Bachelor Thesis --------------------#

#-------------------------------- Performance ------------------------------#

#In the following scrypt, the performance of the tuned random forests will
#be evaluated.

#---------------------------------- Code -----------------------------------#

#------------ General Part ------------#
#Modules
import numpy as np 
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

#Variables
wd_main = os.getcwd()
os.mkdir('22_RF_Performance_Export')
wd_data_model = wd_main + '\\21_RF_Tuning_Export'
wd_data = wd_main + '\\12_DataProcessing_Export'
wd_export = wd_main + '\\22_RF_Performance_Export'

#------------- Classifier -------------#
os.chdir(wd_data_model)
cla = pickle.load(open('rf_classifier_tunded.sav', 'rb'))
os.chdir(wd_data)
df = pickle.load(open('model_data.pkl', 'rb'))

    #% OOB score
oob_error_cla = cla.oob_score_ #0.7270518728425301

    #% ROC
cla_proba = [x[1] for x in cla.oob_decision_function_]
auc_cla = roc_auc_score(df.loc[:, 'successful'], cla_proba) #0.8158472295105472

def get_prediction(proba, threshold=0.5):
    '''Function that determines whether a project was classified successful or not according
    to the votes.'''
    if proba >= threshold:
        return 1
    else:
        return 0
f1_cla = f1_score(df.loc[:, 'successful'], [get_prediction(x) for x in cla_proba]) #0.7535335555790205
recall_cla = recall_score(df.loc[:, 'successful'], [get_prediction(x) for x in cla_proba]) #0.7744085821473972
precision_cla = precision_score(df.loc[:, 'successful'], [get_prediction(x) for x in cla_proba]) #0.7337544065804935

#-------------- Regression --------------#
os.chdir(wd_data_model)
reg = pickle.load(open('rf_regression_tunded.sav', 'rb'))

    #% OOB score
oob_error_reg = reg.oob_score_ #0.11878565124816465

    #% ROC
def success_dummy(funding_rate):
    '''Function that determines whether a project was successful or not according
    to its funding rate.'''
    if funding_rate >= 1:
        return 1
    else:
        return 0
    
y_dummy_reg = [success_dummy(x) for x in reg.oob_prediction_]

error_rate_reg = accuracy_score(df.loc[:, 'successful'], y_dummy_reg, normalize=True) #0.6944504780268069
auc_reg = roc_auc_score(df.loc[:, 'successful'], y_dummy_reg) #0.6971785750989925
f1_reg = f1_score(df.loc[:, 'successful'], y_dummy_reg) #0.7001960745739149
recall_reg = recall_score(df.loc[:, 'successful'], y_dummy_reg) #0.6621089511053235
precision_reg = precision_score(df.loc[:, 'successful'], y_dummy_reg) #0.7429324872496017


#-------------- ROC Curve --------------#
fpr_cla, tpr_cla, thresholds_cla = roc_curve(df.loc[:, 'successful'], cla_proba)
fpr_reg, tpr_reg, thresholds_reg = roc_curve(df.loc[:, 'successful'], reg.oob_prediction_)

sns.set(palette='viridis')
ax = plt.plot(fpr_cla, tpr_cla, label='Classifier (AUC = %0.2f)' % auc_cla)
ax = plt.plot(fpr_reg, tpr_reg, '--', label='Regression (AUC = %0.2f)' % auc_reg)
plt.legend(loc = 'lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.close()

#----------------- Stats ----------------#
stats_dict = {'oob_score_rough': [oob_error_cla, oob_error_reg],
              'oob_score_classifier': [oob_error_cla, error_rate_reg],
              'auc_score': [auc_cla, auc_reg],
              'f1': [f1_cla, f1_reg],
              'recall': [recall_cla, recall_reg],
              'precision': [precision_cla, precision_reg]}

stats_df = pd.DataFrame(stats_dict)
stats_df.index = ['Classifier', 'Regression']
os.chdir(wd_export)
stats_df.to_excel('performance_stats.xlsx')