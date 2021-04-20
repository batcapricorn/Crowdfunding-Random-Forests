# Random Forests - Bachelor Thesis

"""
Permutation Importances
In the following scrypt, the permutation importances of the implemented models 
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
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Variables
wd_main = os.getcwd()
os.mkdir("24_RF_PermutationImportances_Export")
wd_model = wd_main + "\\21_RF_Tuning_Export"
wd_data = wd_main + "\\12_DataProcessing_Export"
wd_export = wd_main + "\\24_RF_PermutationImportances_Export"

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
X = df.iloc[:, 1:-2]
y_cla = df.loc[:, "successful"]
y_reg = df.loc[:, "funding_rate"]

# Permutation Function
def permutation_importance(model, X, y, repititions=5):
    """Function that computes the permutation importance for each
    feature of the given data set."""
    reference_score = model.score(X, y)
    permutation_importances = {}
    for feature in X.columns:
        X_shuffled = X.copy()
        feature_permutation_scores = []
        for i in range(repititions):
            X_shuffled.loc[:, feature] = list(shuffle(X_shuffled[feature]))
            permutation_score = model.score(X_shuffled, y)
            feature_permutation_scores.append(permutation_score)
        permutation_importances[feature] = reference_score - (1 / repititions) * np.sum(
            feature_permutation_scores
        )
    return permutation_importances


# Classifier
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cla, test_size=0.1, random_state=42
)

cla = RandomForestClassifier(n_estimators=400, max_features="sqrt")
cla.fit(X_train, y_train)

pi_cla = permutation_importance(cla, X_test, y_test, repititions=5)
pi_cla = pd.DataFrame(pi_cla.items(), columns=["feature", "score"])

os.chdir(wd_export)
pi_cla.to_excel("permutation_importances_classifier.xlsx")

# Regression
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.1, random_state=42
)

reg = RandomForestRegressor(n_estimators=400, max_features="sqrt")
reg.fit(X_train, y_train)

pi_reg = permutation_importance(reg, X_test, y_test, repititions=5)
pi_reg = pd.DataFrame(pi_reg.items(), columns=["feature", "score"])
pi_reg.to_excel("permutation_importances_regression.xlsx")

# Graph
fig = plt.figure(figsize=(15, 7))
sns.set(palette="viridis", color_codes=True)
feature_names = np.array(feature_names)

plt.subplot(121)
plt.title("Classifier", fontweight="bold")
pi_cla_scores = np.array(pi_cla["score"])
plt.barh(
    feature_names[np.argsort(pi_cla_scores)], pi_cla_scores[np.argsort(pi_cla_scores)]
)
plt.xlabel("Permutation Importance")

plt.subplot(122)
plt.title("Regressor", fontweight="bold")
pi_reg_scores = np.array(pi_reg["score"])
plt.barh(
    feature_names[np.argsort(pi_reg_scores)], pi_reg_scores[np.argsort(pi_reg_scores)]
)
plt.xlabel("Permutation Importance")

plt.show()
plt.close()
