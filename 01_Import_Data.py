#--------------------- Random Forests - Bachelor Thesis --------------------#

#----------------------------- Import of Data ------------------------------#

#In the following scrypt, all Kickstarter datasets provided by the web 
#scraping service "webrobots.io" will be imported, concated and cleaned. 


#---------------------------------- Code -----------------------------------#

#------------ General Part ------------#
#Required moduls
import numpy as np
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
import urllib.request
import glob
import zipfile
from datetime import datetime
import json
import ast
import pickle

#Variables
main_path = os.getcwd()
os.mkdir('01_Import_Data_Export')
export_path = main_path + '\\01_Import_Data_Export'
webrobot_path = 'https://webrobots.io/kickstarter-datasets/'

#------------- Web Crawl --------------#
#Request data from webrobot.io
webrobot_response = requests.get(webrobot_path)
soup = BeautifulSoup(webrobot_response.text, 'html.parser')

#Extract required links for csv-files
links = []
for link in soup.find_all('a'):
    if ('Kickstarter' in str(link)) and ('json' not in str(link)) and ('zip' in str(link)):
        links.append(str(link['href']))

#Create directory for export
os.chdir(export_path)
export_file_zip = '01_Export_Webrobot'
export_directory_zip = export_path + '\\' + export_file_zip
os.mkdir(export_file_zip)

#Download zip files
os.chdir(export_directory_zip)
for link in links: 
    zip_filename = link[63:73] + '.zip'
    urllib.request.urlretrieve(link, zip_filename)

#------------ Data Import -------------#
#Create directory for csv files to prevent an memory exception
os.chdir(export_path)
export_file_csv = '02_Export_Webrobot_CSV'
export_directory_csv = export_path + '\\' + export_file_csv
os.mkdir(export_file_csv)
os.chdir(export_directory_zip)

#Retrieve the corresponding csv-files 
zip_files = glob.glob('*.zip')
files_columns = {}
file_counter = 0
for file in zip_files: 
    try:
        zf = zipfile.ZipFile(file)
        items = [item.filename for item in zf.infolist()]
        counter = 1
        for item in items:
            if counter == 1:
                df_file = pd.read_csv(zf.open(item))
                counter += 1
            else:
                df_file = pd.concat([df_file, pd.read_csv(zf.open(item))], ignore_index=True)
                counter += 1
        os.chdir(export_directory_csv)
        df_file.to_csv((str(file).split('.')[0] + '.csv'), header=True)
        os.chdir(export_directory_zip)
        files_columns[file] = list(df_file.columns)
        file_counter += 1
        print('Progress: %d of %d' %(file_counter, len(zip_files)))
    except:
        file_counter += 1
        print(file)
        continue
    
#Delete temporary ZIP-files
os.chdir(export_directory_zip)
for zip_file in zip_files:
    os.remove(zip_file)
    
#------------- Clean Data -------------#
#Inspect files_columns dictionary
    #% Result: Drop '2015-10-22.zip' due to different format (it does not contain sufficient information)
files_columns.pop('2015-10-22.zip')

#Concat DataFrames on required columns
columns = ['id','name','blurb','goal','pledged','state','currency',
           'deadline','state_changed_at','created_at','launched_at','backers_count','static_usd_rate','usd_pledged',
           'creator','location','category','profile','urls'] 
os.chdir(export_directory_csv)
counter = 1
files = list(files_columns.keys())
for file in files:
    if counter == 1:
        file_csv = str(file).split('.')[0] + '.csv'
        df_rough = pd.read_csv(file_csv, header=0)
        df_rough = df_rough[df_rough['state'].isin(['successful', 'failed'])] #only successful/failed projects are completed
        df_rough = df_rough.drop_duplicates(subset=['id'])
        counter += 1
    else:
        file_csv = str(file).split('.')[0] + '.csv'
        df_file = pd.read_csv(file_csv, header=0)
        df_file = df_file[df_file['state'].isin(['successful', 'failed'])]
        df_file = df_file.drop_duplicates(subset=['id'])
        df_rough = pd.concat([df_rough, df_file], ignore_index=True)
        df_rough = df_rough.drop_duplicates(subset=['id'])
        counter += 1
    print('Progress: %d of %d' %(counter, len(files)))
df_rough = df_rough[columns]

#Convert unix timestamp
def convert_unix_timestamp(x):
    '''Function that can convert a unix timestamp to a datetime object'''
    return datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')
unix_timestamps = ['created_at', 'deadline', 'launched_at', 'state_changed_at']
for column in unix_timestamps:    
    df_rough[column] = pd.to_datetime(df_rough[column].apply(lambda x: convert_unix_timestamp(x)))
    
#Convert json formats to dictionaries   
def convert_strings_to_dictionaries(x):
    '''Function that can convert, if possible, an variable (e.g. in json format) to a python dictionary'''
    try:
        try:
            return json.loads(x)
        except:
            parsed = ast.parse(x, mode='eval')
            fixed = ast.fix_missing_locations(parsed)
            compiled = compile(fixed, '<string>', 'eval')
            return eval(compiled)
    except:
        return x
    
dict_columns = ['category', 'creator', 'profile', 'urls', 'location']
for column in dict_columns:
    df_rough[column] = df_rough[column].apply(lambda x: convert_strings_to_dictionaries(x))
    
#Extract additional information from dictionaries  
df_rough['user_profile'] = np.nan #Link to Kickstarter profile of entrepreneur
df_rough['location_country'] = np.nan #Country
df_rough['location_state'] = np.nan #Geographical State
df_rough['city'] = np.nan #City
df_rough['main_category'] = np.nan #Main category of project
df_rough['project_link'] = np.nan #Link to the crowdunding project
df_rough['reward_link'] = np.nan #Link to the crowdunding project  
  
def extract_additional_data(x, column):
    '''Function that can extract additional data from columns which were origninally in json format'''
    if column == 'user_profile':
        try:
            return x['urls']['web']['user']
        except:
            pass
    if column == 'location_country':
        try:
            return x['country']
        except:
            pass
    if column == 'location_state':
        try:
            return x['state']
        except:
            pass
    if column == 'city':
        try:
            return x['name']
        except:
            pass
    if column == 'main_category':
        try:
            return x['name']
        except:
            pass
    if column == 'project_link':
        try:
            return x['web']['project']
        except:
            pass
    if column == 'reward_link':
        try:
            return x['web']['rewards']
        except:
            pass

df_rough['user_profile'] = df_rough['creator'].apply(lambda x: extract_additional_data(x, 'user_profile'))
df_rough['location_country'] = df_rough['location'].apply(lambda x: extract_additional_data(x, 'location_country'))
df_rough['location_state'] = df_rough['location'].apply(lambda x: extract_additional_data(x, 'location_state'))
df_rough['city'] = df_rough['location'].apply(lambda x: extract_additional_data(x, 'city'))
df_rough['main_category'] = df_rough['category'].apply(lambda x: extract_additional_data(x, 'main_category'))
df_rough['project_link'] = df_rough['urls'].apply(lambda x: extract_additional_data(x, 'project_link'))
df_rough['reward_link'] = df_rough['urls'].apply(lambda x: extract_additional_data(x, 'reward_link'))

df_rough['usd_goal'] = df_rough['goal'] * df_rough['static_usd_rate']
df_rough['duration'] = pd.to_datetime(df_rough['deadline']) - pd.to_datetime(df_rough['launched_at'])

#Delete temporary CSV-files
os.chdir(export_directory_csv)
for csv_file in glob.glob('*.csv'):
    os.remove(csv_file)

#Save backup
os.chdir(export_path)
os.mkdir('03_Main_Data')
export_backup = export_path + '\\03_Main_Data'
os.chdir(export_backup)
df_rough.to_csv('crowdfunding_rough_draft_v1.csv', index=False, header=True, encoding='unicode-escape')
pickle.dump(df_rough, open('crowdfunding_rough_draft_v1.pkl', 'wb'))
    
#Drop unnecessary columns
selected_columns = ['id', 'main_category', 'name', 'blurb', 
                    'goal', 'usd_goal', 'pledged', 'usd_pledged', 'currency', 'static_usd_rate', 'state', 'backers_count',
                    'deadline', 'state_changed_at', 'created_at', 'launched_at', 'duration',
                    'location_country', 'location_state', 'city',
                    'user_profile', 'project_link', 'reward_link']
df_tidy = df_rough[selected_columns]
    
#Save adapted backup
df_tidy.to_csv('crowdfunding_rough_draft_v2.csv', index=False, header=True)
pickle.dump(df_tidy, open('crowdfunding_rough_draft_v2.pkl', 'wb'))   