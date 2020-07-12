#--------------------- Random Forests - Bachelor Thesis --------------------#

#----------------------------- Decsion Trees -------------------------------#

#In the following scrypt, I will implement a visualisation of a 
#decision tree. It will be used in my paper to demonstrate the 
#functionality of this technique.  


#---------------------------------- Code -----------------------------------#

#------------ General Part ------------#
#Modules
import numpy as np
import pandas as pd
from datetime import timedelta
import os
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz 
import graphviz

#Directory
wd_main = os.getcwd() 

#Dataset
os.mkdir('02_Decision_Trees_Export')
os.chdir(wd_main + '\\01_Import_Data_Export\\03_Main_Data')
df = pd.read_csv('crowdfunding_rough_draft_v2.csv')

#------------ Decsion Tree ------------#
#Input and Output
    #% Target
y = df['usd_pledged'] / df['usd_goal']
y = y[y < np.percentile(y, 99)]
X = df[['usd_goal', 'duration']]
X['duration'] = (pd.to_datetime(df['deadline']) - pd.to_datetime(df['launched_at'])).dt.days
X.columns = ['usd_goal', 'duration_days']
X = X.loc[y.index,:]


#Fitting
dtree = tree.DecisionTreeRegressor(max_depth=2)
dtree.fit(X, y)

#Visualisation
os.chdir(wd_main + '\\02_Decision_Trees_Export')
dot_data = export_graphviz(dtree, out_file =None, rounded=True,
                           special_characters=True, filled=True,
                           feature_names=['Funding Goal in USD', 'Duration in Days'])
graph = graphviz.Source(dot_data)
graph.format = "png"
graph.view()