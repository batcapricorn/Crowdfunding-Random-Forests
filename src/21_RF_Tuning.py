# Random Forests - Bachelor Thesis

"""
Tuning
In the following scrypt, the employed random forest models will be
implemented and tuned.
"""

# General Part
# Modules
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid

# Variables
wd_main = os.getcwd()
os.mkdir("21_RF_Tuning_Export")
wd_data = wd_main + "\\12_DataProcessing_Export"
wd_export = wd_main + "\\21_RF_Tuning_Export"

# Data
os.chdir(wd_data)
df = pickle.load(open("model_data.pkl", "rb"))
os.chdir(wd_export)
X = df.loc[
    :,
    [
        "name_length",
        "blurb_gradelevel",
        "category",
        "country",
        "state",
        "city",
        "goal_usd",
        "duration",
    ],
]
y_classifier = df.loc[:, "successful"]
y_regressor = df.loc[:, "funding_rate"]

# Tuning
# Classifier
#% Grid Search
grid_classifier = {
    "n_estimators": [100, 200, 300, 400],
    "max_features": ["sqrt", "log2"],
    "criterion": ["gini", "entropy"],
}

best_score = 0
oob_scores_cla = []
for g in ParameterGrid(grid_classifier):
    rf = RandomForestClassifier(oob_score=True)
    rf.set_params(**g)
    rf.fit(X, y_classifier)
    oob_scores_cla.append(
        [g["n_estimators"], g["max_features"], g["criterion"], rf.oob_score_]
    )
    # save if best
    if rf.oob_score_ > best_score:
        best_score = rf.oob_score_
        best_grid = g

    #% Train best model (params=[400, sqrt, gini])
clf = RandomForestClassifier(oob_score=True)
clf.set_params(**best_grid)
clf.fit(X, y_classifier)

#% Save model and stats
pickle.dump(clf, open("rf_classifier_tunded.sav", "wb"))
pickle.dump(oob_scores_cla, open("rf_classifier_tuningstats.pkl", "wb"))


# Regressor
#% Grid Search
grid_regressor = {
    "n_estimators": [100, 200, 300, 400],
    "max_features": ["auto", "sqrt", "log2"],
    "criterion": ["mse"],
}

best_score = 0
oob_scores_reg = []
for g in ParameterGrid(grid_regressor):
    rf = RandomForestRegressor(oob_score=True)
    rf.set_params(**g)
    rf.fit(X, y_regressor)
    oob_scores_reg.append(
        [g["n_estimators"], g["max_features"], g["criterion"], rf.oob_score_]
    )
    # save if best
    if rf.oob_score_ > best_score:
        best_score = rf.oob_score_
        best_grid = g

    #% Train best model (params=[400, 'sqrt', 'mse'], score=0.11878565124816465)
reg = RandomForestRegressor(oob_score=True)
reg.set_params(**best_grid)
reg.fit(X, y_regressor)

#% Save model and stats
pickle.dump(reg, open("rf_regression_tunded.sav", "wb"))
pickle.dump(oob_scores_reg, open("rf_regressor_tuningstats.pkl", "wb"))
