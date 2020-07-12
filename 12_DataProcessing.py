#--------------------- Random Forests - Bachelor Thesis --------------------#

#----------------------------- Data Processing -----------------------------#

#In the following scrypt, the underlying data for my study will be processed. 


#---------------------------------- Code -----------------------------------#

#------------ General Part ------------#
#Modules
import numpy as np 
import pandas as pd
import os
import pickle
import textstat
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from num2words import num2words
from sklearn import preprocessing

#Variables
wd_main = os.getcwd()
os.mkdir('12_DataProcessing_Export')
wd_data = wd_main + '\\01_Import_Data_Export\\03_Main_Data'
wd_export = wd_main + '\\12_DataProcessing_Export'
df_model_data = pd.DataFrame()

#Read data
os.chdir(wd_data)
df = pickle.load(open('crowdfunding_rough_draft_v2.pkl', 'rb'))
os.chdir(wd_export)

#---------- Data Processing -----------#

#Targets
df['funding_rate'] =  df['usd_pledged'] / df['usd_goal']
df['success_dummy'] = df.loc[:, 'state']
df['success_dummy'] = df['success_dummy'].replace('failed', 0).replace('successful', 1)

#Exclude campaigns with extremely small funding goals
df = df[df['usd_goal']>100]

#NaN-values
nan_stats = df.isnull().sum()
df['location_state'] = df['location_state'].fillna('missing')
df['location_country'] = df['location_country'].fillna('missing')
df['city'] = df['city'].fillna('missing')
df['name'] = df['name'].fillna('')
df['blurb'] = df['blurb'].fillna('')

#Data types
dtypes = df.dtypes
df['main_category'] = df['main_category'].astype('category') #160 categories
df['location_country'] = df['location_country'].astype('category') #205 countries
df['location_state'] = df['location_state'].astype('category') #1327 states
df['city'] = df['city'].astype('category') #15934 cities

#Save backup
pickle.dump(df, open('crowdfunding_processed_rough_draft.pkl', 'wb'))

#Grade Level
gl = [textstat.flesch_kincaid_grade(blurb) for blurb in list(df.loc[:, 'blurb'])]
df_model_data['id'] = df.loc[:, 'id']
df_model_data['blurb_gradelevel'] = gl

df_model_data['name_length'] = [str(x).strip("!-?/\,.*^@'#:+~").replace('-', '') \
                                .replace(':', '') for x in df.loc[:, 'name']]
df_model_data['name_length'] = [[y for y in x.split(' ') if y!=''] for x in df_model_data.loc[:, 'name_length']]    
df_model_data['name_length'] = [len(x) for x in df_model_data.loc[:, 'name_length']]

#Catgorical Values
df_model_data['category'] = df.loc[:, 'main_category'].astype('category')
df_model_data['country'] = df.loc[:, 'location_country'].astype('category')
df_model_data['state'] = df.loc[:, 'location_state'].astype('category')
df_model_data['city'] = df.loc[:, 'city'].astype('category')

le = preprocessing.LabelEncoder()
for category in ['category', 'country', 'state', 'city']:
    df_model_data.loc[:,category] = le.fit_transform([str(x) for x in df_model_data.loc[:,category]])

#Numerical values
df_model_data['goal_usd'] = df.loc[:, 'usd_goal'].astype('float')

#Time dependend values
df_model_data['duration'] = df.loc[:, 'duration'].dt.days

#Target values
df_model_data['funding_rate'] = df.loc[:, 'funding_rate'].astype('float')
df_model_data['successful'] = df.loc[:, 'success_dummy'].replace('successful', 1).replace('failed', 0)

#Save data set
column_names = ['id', 'blurb_gradelevel', 'name_length', 'category', 'country', 'state', 'city',
                'goal_usd', 'duration', 'funding_rate', 'successful']
df_model_data.columns = column_names
columns_order = ['id', 'name_length', 'blurb_gradelevel', 'category', 'country', 'state', 'city',
                'goal_usd', 'duration', 'funding_rate', 'successful']
df_model_data = df_model_data[columns_order]
pickle.dump(df_model_data, open('model_data.pkl', 'wb'))
stats = df_model_data.describe()
stats.to_excel('data_stats.xlsx')