# Random Forests - Bachelor Thesis

"""
Feature Importances
In the following scrypt, the feature importances of the implemented models 
will be evaluated.
"""

# General Part
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import pickle

# Variables
wd_main = os.getcwd()
os.mkdir("23_RF_FeatureImportances_Export")
wd_model = wd_main + "\\21_RF_Tuning_Export"
wd_data = wd_main + "\\12_DataProcessing_Export"
wd_export = wd_main + "\\23_RF_FeatureImportances_Export"

# Data
os.chdir(wd_data)
df = pickle.load(open("model_data.pkl", "rb"))
feature_names = list(df.columns[1:-2])
feature_names = [
    "Name (Word Count)",
    "Blurb (Grade Level)",
    "Category",
    "Country",
    "State",
    "City",
    "Goal in $",
    "Duration",
]

# Classifier
os.chdir(wd_model)
cla = pickle.load(open("rf_classifier_tunded.sav", "rb"))
fi_cla = cla.feature_importances_

# Regression
reg = pickle.load(open("rf_regression_tunded.sav", "rb"))
fi_reg = reg.feature_importances_

# Graph
fig = plt.figure(figsize=(15, 7))
sns.set(palette="viridis", color_codes=True)

plt.subplot(121)
plt.title("Classifier", fontweight="bold")
plt.barh(np.array(feature_names)[np.argsort(fi_cla)], fi_cla[np.argsort(fi_cla)])
plt.xlabel("Feature Importance (Gini)")

plt.subplot(122)
plt.title("Regressor", fontweight="bold")
plt.barh(np.array(feature_names)[np.argsort(fi_reg)], fi_reg[np.argsort(fi_reg)])
plt.xlabel("Feature Importance (MSE)")

plt.show()
plt.close()
