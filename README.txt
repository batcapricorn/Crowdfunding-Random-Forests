#--------------------- Random Forests - Bachelor Thesis ---------------------#

#----------------------------------- READ ME -------------------------------#

This folder contains the code utilzed to examine crowdfunding projects with 
the aid of random forests. It does not include the corresponding export files.
Nevertheless, the results could be replicated easily with the provided scrpyts.
Eventually, the working directories would need to be adapted. Some scrypts 
are computationally very intense and correspond to a running time above 10 
hours.

The Python-Project contains the following files:

Phase 1
----------------------------------------------------------------------------
01_Import_Data:
	Web crawl that concated the data sets provided by webrobots.io

02_Decision_Trees:
	Visualization of an exemplary decision tree

Phase 2
----------------------------------------------------------------------------
11_EDA:
	Exploratory Data Analysis of the underlying data set

12_DataProcessing:
	Feature Engineering and Data Processing of the underlying data set

Phase 3
----------------------------------------------------------------------------
21_RF_Tuning:
	Tuning of the implemented models via grid search

22_RF_Performance:
	Computation of numerous performance metrics

23_RF_FeatureImportances:
	Computation of the feature importance values

24_RF_PermutationIMportances:
	Computation of the permutation importance values

25_RF_PairedSamplesTTest:
	Paired-Samples T-Test in order to analyze the impact of feature subgroups