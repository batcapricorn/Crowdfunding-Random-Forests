#--------------------- Random Forest - Bachelor Thesis ---------------------#

#----------------------------------- EDA -----------------------------------#

#In the following scrypt, the underlying data will be analyzed with the aid of
#common statistical and graphical tools.  


#---------------------------------- Code -----------------------------------#

#------------ General Part ------------#
#Modules
import numpy as np 
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

#Variables
wd_main = os.getcwd()
os.mkdir('11_EDA_Export')
wd_data = wd_main + '\\01_Import_Data_Export\\03_Main_Data'
wd_export = wd_main + '\\11_EDA_Export'

#Read data
os.chdir(wd_data)
df = pickle.load(open('crowdfunding_rough_draft_v2.pkl', 'rb'))
os.chdir(wd_export)

#---------------- EDA ----------------#

#------ Basics ------# 
#Head and Tail
header_20 = df.head(20)
tail_20 = df.tail(20)

#Descriptive statistics
df['funding_rate'] =  df['usd_pledged'] / df['usd_goal']
df['success_dummy'] = df.loc[:, 'state']
df['success_dummy'] = df['success_dummy'].replace('failed', 0).replace('successful', 1)
stats = df.describe()

    #% Check extreme funding rates
obs = df[df['funding_rate']>np.percentile(df['funding_rate'], 99)]

    #% Exclude campaigns with extremely small funding goals
df = df[df['usd_goal']>100]

    #% Update stats
stats = df.describe()

    #% Save
stats.to_excel('stats_rough_data.xlsx', header=True)

#NaN-values
nan_stats = df.isnull().sum()
df['location_state'] = df['location_state'].fillna('missing')
df['location_country'] = df['location_country'].fillna('missing')
df['city'] = df['city'].fillna('missing')

#Data types
dtypes = df.dtypes
df['main_category'] = df['main_category'].astype('category') #160 categories
df['location_country'] = df['location_country'].astype('category') #205 countries
df['location_state'] = df['location_state'].astype('category') #1327 states
df['city'] = df['city'].astype('category') #15934 cities

#Pairplot
sns.pairplot(df)

#----- Features -----#

#-- Countries --#
country_stats = df['location_country'].value_counts()
country_stats = pd.DataFrame(country_stats)
country_stats['percentage'] = country_stats['location_country'] / country_stats.sum()[0]
country_stats = country_stats.reset_index()
country_stats.columns = ['location_country', 'country_count', 'percentage']

    #% Average stats
obs = df.groupby('location_country').mean()[['funding_rate', 'success_dummy', 
                                    'usd_pledged', 'usd_goal']].reset_index()
country_stats = country_stats.merge(obs, on='location_country', how='left')
country_stats.columns = ['location_country', 'country_count', 'percentage', 
                         'avg_funding_rate', 'avg_success_rate', 
                         'avg_usd_pledged', 'avg_usd_goal']

    #% Std stats
obs = df.groupby('location_country').std()[['funding_rate',
                                    'usd_pledged', 'usd_goal']].reset_index()
obs.columns = ['location_country', 'std_funding_rate',
               'std_usd_pledged', 'std_usd_goal']
country_stats = country_stats.merge(obs, on='location_country', how='left')

    #% Save
country_stats.to_excel('country_stats_rough_data.xlsx', header=True)

    #% Visualization
        #% All countries
sns.set(palette='viridis')
ax = plt.scatter(np.log(country_stats['avg_usd_pledged'].replace(0, 1)),
                 np.log(country_stats['avg_usd_goal']),
                 alpha=0.6,
                 c=country_stats['avg_success_rate'],
                 cmap='viridis',
                 vmin=0, vmax=1)
plt.xlabel('Log Avg. Pledged in $')
plt.ylabel('Log Avg. Goal in $')
plt.colorbar(ax)
plt.show()
plt.close()

        #% Top 20
country_stats_top20 = country_stats.loc[:20,:]
sns.set(palette='viridis')
ax = plt.scatter(np.log(country_stats_top20['avg_usd_pledged'].replace(0, 1)),
                 np.log(country_stats_top20['avg_usd_goal']),
                 s=20000*country_stats_top20['percentage'],
                 alpha=0.6,
                 c=country_stats_top20['avg_success_rate'],
                 cmap='viridis',
                 vmin=0, vmax=1)
plt.xlabel('Log Avg. Pledged in $')
plt.ylabel('Log Avg. Goal in $')
plt.colorbar(ax)
for country in country_stats_top20['location_country']:
    plt.annotate(s=country, 
                 xy=(np.log(country_stats_top20[country_stats_top20['location_country']==country]['avg_usd_pledged']),
                     np.log(country_stats_top20[country_stats_top20['location_country']==country]['avg_usd_goal'])),
                     fontsize='small')
plt.show()
plt.close()


#-- Time series --#
years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
df_time = df.loc[:,:].set_index('launched_at')
df_time = df_time.sort_index()

    #% Number of campaigns per year
time_counts = df_time.resample('Y').count()
time_stats = pd.DataFrame()
time_stats['campaign_count'] = time_counts['id']
time_stats['relative_campaign_count'] = time_stats['campaign_count'] / time_stats['campaign_count'].sum()
time_stats.index = years
    #% Annual means 
time_avg = df_time.resample('Y').mean()
time_avg = time_avg[['usd_goal', 'usd_pledged', 'funding_rate', 'success_dummy']]
time_avg.columns = ['avg_usd_goal', 'avg_usd_pledged', 'avg_funding_rate', 'success_rate']
time_avg.index = years
time_stats = time_stats.merge(time_avg, left_index=True, right_index=True)
    #% Annual standard deviation
time_std = df_time.resample('Y').std()
time_std = time_std[['usd_goal', 'usd_pledged', 'funding_rate']]
time_std.columns = ['std_usd_goal', 'std_usd_pledged', 'std_funding_rate']
time_std.index = years
time_stats = time_stats.merge(time_std, left_index=True, right_index=True)
    #% Save
time_stats.to_excel('time_stats_rough_data.xlsx', header=True)
    #% Visualization
sns.set(palette='viridis')
ax = plt.plot(time_stats.index, np.log(time_stats['campaign_count']),
                 label='N° of Campaigns')
ax = plt.plot(time_stats.index, 
                 np.log(time_stats['campaign_count']*time_stats['success_rate']),
                 label='N° of Successful Campaigns')
ax = plt.plot(time_stats.index, np.log(time_stats['avg_usd_goal']),
                 label='Avg. Funding Goal in $', dashes=(2,2))
ax = plt.plot(time_stats.index, np.log(time_stats['avg_usd_pledged']),
                 label='Avg. Pledged in $', dashes=(2,2))
plt.xlabel('Year')
plt.ylabel('Log Values')
plt.legend(loc='lower right')
plt.show()
plt.close()


#-- Category --#
category_stats = df['main_category'].value_counts()
category_stats = pd.DataFrame(category_stats)
category_stats['percentage'] = category_stats['main_category'] / category_stats.sum()[0]
category_stats = category_stats.reset_index()
category_stats.columns = ['main_category', 'category_count', 'percentage']

    #% Average stats
obs = df.groupby('main_category').mean()[['funding_rate', 'success_dummy', 
                                          'usd_pledged', 'usd_goal']].reset_index()
category_stats = category_stats.merge(obs, on='main_category', how='left')
category_stats.columns = ['main_category', 'category_count', 'percentage', 
                         'avg_funding_rate', 'avg_success_rate', 
                         'avg_usd_pledged', 'avg_usd_goal']

    #% Std stats
obs = df.groupby('main_category').std()[['funding_rate',
                                         'usd_pledged', 'usd_goal']].reset_index()
obs.columns = ['main_category', 'std_funding_rate',
               'std_usd_pledged', 'std_usd_goal']
category_stats = category_stats.merge(obs, on='main_category', how='left')

    #% Save
category_stats.to_excel('category_stats_rough_data.xlsx', header=True)

    #% Visualization
sns.set(palette='viridis')
ax = plt.scatter(np.log(category_stats['avg_usd_pledged'].replace(0, 1)),
                 np.log(category_stats['avg_usd_goal']),
                 s=20000*category_stats['percentage'],
                 alpha=0.6,
                 c=category_stats['avg_success_rate'],
                 cmap='viridis',
                 vmin=0, vmax=1)
plt.xlabel('Log Avg. Pledged in $')
plt.ylabel('Log Avg. Goal in $')
plt.colorbar(ax)
for category in category_stats.loc[:10, 'main_category']:
    plt.annotate(s=category, 
                 xy=(np.log(category_stats[category_stats['main_category']==category]['avg_usd_pledged']),
                     np.log(category_stats[category_stats['main_category']==category]['avg_usd_goal'])),
                     fontsize='small')
plt.show()
plt.close()

#-- Funding Rate --#
funding_rate = df['funding_rate']
funding_rate = funding_rate[funding_rate<1.5]
sns.set(palette='viridis')
sns.kdeplot(funding_rate, legend=False)
plt.xlabel('Funding Rate')
plt.show()
plt.close()