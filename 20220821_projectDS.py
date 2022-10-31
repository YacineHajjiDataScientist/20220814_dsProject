#!/usr/bin/env python
# coding: utf-8

# # To do

# + complete template with variables update completion
# + Start encoding variables
# + Merger les bases
# + VCramer on merged dataset
# + Faire un modèle basique
# + Parmi les variables gardées, faire une relation entre celles qui sont fortement liées dans la base de données finale (e.g. node features graph)

# # --------------------------------------Session--------------------------------------

# In[6]:


# install modules
pip install dill
pip install xgboost


# In[ ]:


# Option 2
python -m install dill


# In[1]:


# import modules
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import dill
import datetime
import math

from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
import statsmodels.api

from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression
import xgboost as xgb

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# functions
def V_cramer(tab, n):
    # Initiating objects
    nrow, ncol = tab.shape
    resultats_test = chi2_contingency(tab)
    statistique = resultats_test[0]
    # Computing objects
    r = ncol - (((ncol - 1) **  2) / (n - 1))
    k = nrow - (((nrow - 1) **  2) / (n - 1))
    phi_squared = max(0, ((statistique / n) - (((ncol - 1) * (nrow - 1)) / (n - 1))))
    V = math.sqrt((phi_squared / (min(k - 1, r - 1))))
    return V


# In[3]:


##### Defining directory
os.chdir('C:\\Users\\Megaport\\20220814_projectDS')
os.chdir('C:\\Users\\Megaport\\Desktop\\jupyterNotebook')
os.getcwd()


# In[ ]:


# import session
# dill.load_session('notebook_env.db')


# In[ ]:


# save session
# dill.dump_session('notebook_env.db')


# # --------------------------------------Import--------------------------------------

# ##### Unique datasets

# In[10]:


##### Import of tables into dataframes
dfLieux = pd.read_csv('20220906_table_lieux.csv', sep=',')
dfUsagers = pd.read_csv('20220906_table_usagers.csv', sep=',')
dfVehicules = pd.read_csv('20220906_table_vehicules.csv', sep=',')
dfCarac = pd.read_csv('20220906_table_caracteristiques.csv', sep=',')

##### Additional dataframes
dfJoursFeriesMetropole = pd.read_csv('20221009_table_joursFeriesMetropole.csv', sep=';')


# In[6]:


print('dfLieux dimensions:', dfLieux.shape)
print('dfUsagers dimensions:', dfUsagers.shape)
print('dfVehicules dimensions:', dfVehicules.shape)
print('dfCarac dimensions:', dfCarac.shape)


# ##### Pooled datasets

# In[4]:


##### Import of tables into dataframes
dfPool = pd.read_csv('20221024_table_poolPostDataManagement_YAH_BPA.csv', sep=',')


# In[5]:


print('dfPool dimensions:', dfPool.shape)


# # --------------------------------------Data-management--------------------------------------

# ##### Computing new variables

# In[11]:


# Computing date variable
dfCarac['date'] = dfCarac['jour'].astype(str) + '-' + dfCarac['mois'].astype(str) + '-' + dfCarac['an'].astype(str)
dfCarac['date'] = pd.to_datetime(dfCarac['date']);

# Computing months with categorical labels
dfCarac['mois_label'] = dfCarac['mois']
dfCarac['mois_label'] = dfCarac['mois_label'].replace(to_replace=np.arange(1, 13, 1), 
                                                      value=['jan', 'fev', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])

# Days of the week
dfCarac['weekday'] = dfCarac['date'].dt.weekday

# Hour of the day
dfCarac['hrmn'] = dfCarac['hrmn'].replace('\:', '', regex=True).astype(int)
dfCarac['hour'] = dfCarac['hrmn']//100

# Year of accident
dfCarac['year'] = dfCarac['date'].dt.year

# Adding the year variable to dfUsagers dataframe
dfUsagers = dfUsagers.merge(right=dfCarac[['Num_Acc', 'year']], on='Num_Acc')

# Age of people during the accident (removing ages above 99, could be completion issues and there are very few values)
dfUsagers['age'] = dfUsagers.year - dfUsagers.an_nais
dfUsagers.loc[dfUsagers['age'] > 99, 'age'] = np.nan

# Largeur de la route assignée au trafic
dfLieux.larrout = dfLieux.larrout.replace('\,', '.', regex=True).astype('float64')
dfLieux.lartpc = dfLieux.lartpc.replace('\,', '.', regex=True).astype('float64')


# ##### Refining variables before Merging datasets

# In[12]:


### dfCarac
## hourGrp: nuit (22h - 6h) - jour heures creuses (10h-16h) - jour heures de pointe (7-9h, 17-21h)
hourConditions = [((dfCarac["hour"]>=22) | (dfCarac["hour"]<=6)),
                  (((dfCarac["hour"]>=7) & (dfCarac["hour"]<=9)) | ((dfCarac["hour"]>=17) & (dfCarac["hour"]<=21))),
                  ((dfCarac["hour"]>=10) | (dfCarac["hour"]<=16))]
hourChoices = ["nuit", "heure de pointe", "journee"]
dfCarac["hourGrp"] = np.select(hourConditions, hourChoices)
## atm: passer en NA les valeurs -1 et 9 (other) qui sont difficilement interprétables dans un modèle de ML
dfCarac['atm'] = dfCarac['atm'].replace([-1, 9], [np.nan, np.nan])
## Date feriée/weekend/feriée ou weekend
dateFerie = list(map(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y').strftime('%Y-%m-%d'), dfJoursFeriesMetropole['date']))
dfDateFerie = pd.DataFrame({'dateFerie': dateFerie})
dfCarac['dateFerie'] = np.where((dfCarac.date.isin(dfDateFerie.dateFerie)), 1, 0)
dfCarac['dateWeekend'] = np.where((dfCarac.weekday>=5), 1, 0)
dfCarac['dateFerieAndWeekend'] = np.where((dfCarac.date.isin(dfDateFerie.dateFerie) | (dfCarac.weekday>=5)), 1, 0)

### dfLieux
## nbvGrp: 0/1/2/3/4+, avec -1 et 9+ en NA
nbvConditions = [((dfLieux["nbv"]>=9) | (dfLieux["nbv"]==-1)),
                (dfLieux["nbv"]==0),
                (dfLieux["nbv"]==1),
                (dfLieux["nbv"]==2),
                (dfLieux["nbv"]==3),
                (dfLieux["nbv"]>=4),]
nbvChoices = [np.nan, '0', '1', '2', '3', '4+']
dfLieux['nbvGrp'] = np.select(nbvConditions, nbvChoices)
## vostGrp: présence yes/no d'une voie réservée
dfLieux['vospGrp'] = dfLieux['vosp'].replace([-1, 0, 1, 2, 3], [np.nan, 0, 1, 1, 1])
## profGrp: -1 et 0 en NA
dfLieux['prof'] = dfLieux['prof'].replace([-1, 0], [np.nan, np.nan])
## planGrp: en binaire not straight vs straight (yes/no), les -1 et 0 en NA
dfLieux['planGrp'] = dfLieux['plan'].replace([-1, 0, 1, 2, 3, 4], [np.nan, np.nan, 0, 1, 1, 1])
## lartpcGrp: 0/1/2/3/4+, avec -1 et 9+ en NA
lartpcConditions = [((dfLieux["lartpc"]==0.0)),
                    ((dfLieux["lartpc"]>=20)),
                    ((dfLieux["lartpc"]>0) & (dfLieux["lartpc"]<5)),
                    ((dfLieux["lartpc"]>=5) & (dfLieux["lartpc"]<10)),
                    ((dfLieux["lartpc"]>=10) & (dfLieux["lartpc"]<15)),
                    ((dfLieux["lartpc"]>=15) & (dfLieux["lartpc"]<20))]
lartpcChoices = [np.nan, np.nan, 1, 2, 3, 4]
dfLieux['lartpcGrp'] = np.select(lartpcConditions, lartpcChoices)
dfLieux['lartpcGrp'] = dfLieux['lartpcGrp'].replace([0, 1, 2, 3, 4], [np.nan, '0-5', '5-10', '10-15', '15-20'])
## larroutGrp: 0/1/2/3/4+, avec -1 et 9+ en NA
larroutConditions = [((dfLieux["larrout"]==0.0)),
                    ((dfLieux["larrout"]>=200)),
                    ((dfLieux["larrout"]>0) & (dfLieux["larrout"]<50)),
                    ((dfLieux["larrout"]>=50) & (dfLieux["larrout"]<100)),
                    ((dfLieux["larrout"]>=100) & (dfLieux["larrout"]<150)),
                    ((dfLieux["larrout"]>=150) & (dfLieux["larrout"]<200))]
larroutChoices = [np.nan, np.nan, 1, 2, 3, 4]
dfLieux['larroutGrp'] = np.select(larroutConditions, larroutChoices)
dfLieux['larroutGrp'] = dfLieux['larroutGrp'].replace([0, 1, 2, 3, 4], [np.nan, '0-50', '50-100', '100-150', '150-200'])

## surf: transformation des -1, 0 et 9 en NA
dfLieux['surf'] = dfLieux['surf'].replace([-1, 0, 9], [np.nan, np.nan, np.nan])
## situ: transformation des -1, 0 et 9 en  NA
dfLieux['situ'] = dfLieux['situ'].replace([-1, 0], [np.nan, np.nan])

### dfUsagers
## Does a gravity of type X exist for an accident
dfUsagers['grav34exists'] = np.where(dfUsagers.grav2>=3, 1, 0)
dfUsagers['grav4exists'] = np.where(dfUsagers.grav2==4, 1, 0)
dfUsagers['grav3exists'] = np.where(dfUsagers.grav2==3, 1, 0)
dfUsagers['grav2exists'] = np.where(dfUsagers.grav2==2, 1, 0)
## place: transformation des 0 en NA
dfUsagers['place'] = dfUsagers['place'].replace([0], [np.nan])
## actp: harmonization des valeurs et transformation des -1 en NA
dfUsagers['actp'] = dfUsagers['actp'].replace({'0.0':0, '0':0, 0:0,
                                              '-1.0':np.nan, '-1':np.nan, ' -1':np.nan, -1:np.nan,
                                              '1.0':1, '1':1, 1:1,
                                              '2.0':2, '2':2, 2:2,
                                              '3.0':3, '3':3, 3:3,
                                              '4.0':4, '4':4, 4:4,
                                              '5.0':5, '5':5, 5:5,
                                              '6.0':6, '6':6, 6:6,
                                              '7.0':7, '7':7, 7:7,
                                              '8.0':8, '8':8, 8:8,
                                              '9.0':9, '9':9, 9:9
                                              })
## etatp: transformation des -1 en NA et nombre de piétons seuls dans l'accident
dfUsagers['etatp'] = dfUsagers['etatp'].replace([-1], [np.nan])
dfUsagers['etatp_pieton_alone_exists'] = np.where((dfUsagers['etatp']==1), 1, 0)
## locp: transformation des 0 en NA et nombre de piétons en fonction de leur position pendant l'accident
dfUsagers['locp'] = dfUsagers['locp'].replace([-1], [np.nan])
dfUsagers['locp_pieton_1_exists'] = np.where(((dfUsagers.locp==1)), 1, 0)
dfUsagers['locp_pieton_3_exists'] = np.where(((dfUsagers.locp==3)), 1, 0)
dfUsagers['locp_pieton_6_exists'] = np.where(((dfUsagers.locp==6)), 1, 0)
## Number of pietons in catu variable (or catu_conductor)
dfUsagers['catu_pieton_exists'] = np.where(((dfUsagers.catu==3) | (dfUsagers.catu==4)), 1, 0)
dfUsagers['catu_conductor_exists'] = np.where(((dfUsagers.catu==1)), 1, 0)
## Number of men/women conductor
dfUsagers['sexe_male_conductor_exists'] = np.where(((dfUsagers.sexe==1) & (dfUsagers.catu==1)), 1, 0)
dfUsagers['sexe_female_conductor_exists'] = np.where(((dfUsagers.sexe==2) & (dfUsagers.catu==1)), 1, 0)
## Number of conductor going to courses/promenade (3 & 5)
dfUsagers['trajet_coursesPromenade_conductor_exists'] = np.where((((dfUsagers.trajet==3) & (dfUsagers.catu==1)) | 
                                                           ((dfUsagers.trajet==5) & (dfUsagers.catu==1))), 1, 0)
## Mean age of conductors and nonCoductors by accident
# Preliminary dataFrames with mean age of Conductors/nonConductors by accident
dfAgeMeanConductors = dfUsagers[(dfUsagers['catu_conductor_exists']==1)][['Num_Acc', 'age']].groupby(['Num_Acc']).mean().rename({'age':'ageMeanConductors'}, axis=1)
dfAgeMeanNonConductors = dfUsagers[(dfUsagers['catu_conductor_exists']==0)][['Num_Acc', 'age']].groupby(['Num_Acc']).mean().rename({'age':'ageMeanNonConductors'}, axis=1)
# New variable 'Num_Acc' for merging
dfAgeMeanConductors['Num_Acc'] = dfAgeMeanConductors.index
dfAgeMeanNonConductors['Num_Acc'] = dfAgeMeanNonConductors.index
# Change index so there is no ambiguity while merging
dfAgeMeanConductors.index = np.arange(1, len(dfAgeMeanConductors) + 1)
dfAgeMeanNonConductors.index = np.arange(1, len(dfAgeMeanNonConductors) + 1)
# Merging new variables
dfUsagers = dfUsagers.merge(right=dfAgeMeanConductors, how='left', on='Num_Acc')
dfUsagers = dfUsagers.merge(right=dfAgeMeanNonConductors, how='left', on='Num_Acc')

### Computeing all variables as 'is there at least one of'
dfAtLeastOneByAccident = pd.DataFrame({
                                      # event exists yes/no by accident
              'Num_Acc':  dfUsagers.groupby('Num_Acc')['grav4exists'].sum().index, 
              'gravGrp_23_4': np.where(dfUsagers.groupby('Num_Acc')['grav4exists'].sum()>=1, 1, 0), 
              'gravGrp_2_34': np.where(dfUsagers.groupby('Num_Acc')['grav34exists'].sum()>=1, 1, 0), 
              'catu_pieton': np.where(dfUsagers.groupby('Num_Acc')['catu_pieton_exists'].sum()>=1, 1, 0), 
              'sexe_male_conductor': np.where(dfUsagers.groupby('Num_Acc')['sexe_male_conductor_exists'].sum()>=1, 1, 0), 
              'sexe_female_conductor': np.where(dfUsagers.groupby('Num_Acc')['sexe_female_conductor_exists'].sum()>=1, 1, 0), 
              'trajet_coursesPromenade_conductor': np.where(dfUsagers.groupby('Num_Acc')['trajet_coursesPromenade_conductor_exists'].sum()>=1, 1, 0), 
              'etatpGrp_pieton_alone': np.where(dfUsagers.groupby('Num_Acc')['etatp_pieton_alone_exists'].sum()>=1, 1, 0),
              'locpGrp_pieton_1': np.where(dfUsagers.groupby('Num_Acc')['locp_pieton_1_exists'].sum()>=1, 1, 0),
              'locpGrp_pieton_3': np.where(dfUsagers.groupby('Num_Acc')['locp_pieton_3_exists'].sum()>=1, 1, 0),
              'locpGrp_pieton_6': np.where(dfUsagers.groupby('Num_Acc')['locp_pieton_6_exists'].sum()>=1, 1, 0),
                   
                                       # count event variable by accident
              'nb_grav4_by_acc': dfUsagers.groupby('Num_Acc')['grav4exists'].sum(),
              'nb_grav3_by_acc': dfUsagers.groupby('Num_Acc')['grav3exists'].sum(), 
              'nb_catu_pieton': dfUsagers.groupby('Num_Acc')['catu_pieton_exists'].sum(), 
              'nb_sexe_male_conductor': dfUsagers.groupby('Num_Acc')['sexe_male_conductor_exists'].sum(), 
              'nb_sexe_female_conductor': dfUsagers.groupby('Num_Acc')['sexe_female_conductor_exists'].sum(), 
              'nb_trajet_coursesPromenade_conductor': dfUsagers.groupby('Num_Acc')['trajet_coursesPromenade_conductor_exists'].sum(), 
              'nb_etatpGrp_pieton_alone': dfUsagers.groupby('Num_Acc')['etatp_pieton_alone_exists'].sum(), 
              'nb_locpGrp_pieton_1': dfUsagers.groupby('Num_Acc')['locp_pieton_1_exists'].sum(), 
              'nb_locpGrp_pieton_3': dfUsagers.groupby('Num_Acc')['locp_pieton_3_exists'].sum(), 
              'nb_locpGrp_pieton_6': dfUsagers.groupby('Num_Acc')['locp_pieton_6_exists'].sum(), 
    
                                        # mean of variable by accident
              'ageMeanConductors': dfUsagers.groupby('Num_Acc')['ageMeanConductors'].mean(), 
              'ageMeanNonConductors': dfUsagers.groupby('Num_Acc')['ageMeanNonConductors'].mean()})

### Change index so there is no ambiguity while merging
dfAtLeastOneByAccident.index = np.arange(1, len(dfAtLeastOneByAccident) + 1)


# ##### Merging dataFrames post-DataManagement

# In[20]:


##### Merging of tables into 1 pooled dataframe post-DataManagement (2 steps required)
dfPoolPostDataManagementTemp = pd.merge(dfLieux, dfCarac, on="Num_Acc")
dfPoolPostDataManagement = pd.merge(dfPoolPostDataManagementTemp, dfAtLeastOneByAccident, on="Num_Acc")


# In[21]:


##### Removing latest variables
dfPoolPostDataManagement = dfPoolPostDataManagement.drop(['Unnamed: 0.1_x', 'Unnamed: 0_x'], axis=1)


# In[24]:


##### Export dataframe
pathExport = 'D:\\jupyterDatasets\\'
dfPoolPostDataManagement.to_csv(pathExport+'20221022_table_poolPostDataManagement_YAH.csv', index=False, sep=';')


# ##### Verification transformation variables (Quality Check)

# In[159]:


# pd.crosstab(dfCarac["hour"], dfCarac["hourGrp"])
# dfCarac['atm'].value_counts()
# print(dfCarac.dateFerie.value_counts(normalize=True))
# print(dfCarac.dateWeekend.value_counts(normalize=True))
# print(dfCarac.dateFerieAndWeekend.value_counts(normalize=True))
# pd.crosstab(dfLieux["nbv"], dfLieux["nbvGrp"])
# pd.crosstab(dfLieux["vosp"], dfLieux["vospGrp"])
# dfCarac['prof'].value_counts()
# pd.crosstab(dfLieux["plan"], dfLieux["planGrp"])
# dfCarac['surf'].value_counts()
# dfCarac['situ'].value_counts()
# dfLieux['lartpcGrp'].value_counts()
# dfLieux['larroutGrp'].value_counts()
# dfAtLeastOneByAccident.sexe_male_conductor.value_counts()
# dfAtLeastOneByAccident.sexe_female_conductor.value_counts()
# dfAtLeastOneByAccident.trajet_coursesPromenade_conductor.value_counts()


# # --------------------------------------Descriptive statistics--------------------------------------
# ### Mapping of variables
# In this section, we describe each table

# ### -Table Carac-

# In[12]:


dfCarac.head(3)


# In[4]:


dfCarac.info()


# In[5]:


### Proportion of NA by variable
dfCarac.isnull().sum() * 100 / len(dfCarac)


# In[6]:


# Number of modalities by variable
print('Jours:', len(dfCarac.jour.value_counts()))
print('Mois:', len(dfCarac.mois.value_counts()))
print('An:', len(dfCarac.an.value_counts()))
print('hrmn:', len(dfCarac.hrmn.value_counts()))
print('lum:', len(dfCarac.lum.value_counts()))
print('atm:', len(dfCarac.atm.value_counts()))
print('col:', len(dfCarac.col.value_counts()))
print('agg:', len(dfCarac['agg'].value_counts()))


# In[7]:


# Description of values of each variable
print(dfCarac.an.value_counts())
print(dfCarac.lum.value_counts())
print(dfCarac.atm.value_counts())
print(dfCarac.col.value_counts())
print(dfCarac.hrmn.value_counts())
print(dfCarac['agg'].value_counts())


# In[9]:


dfCarac[['jour', 'mois', 'an', 'lum', 'atm', 'col', 'agg']].hist(figsize=(20, 8), layout=(2, 4));


# ### -Table Lieux-

# In[15]:


dfLieux.head(3)


# In[12]:


dfLieux[['nbv', 'vosp', 'prof', 'pr', 'pr1', 'plan', 'lartpc', 'larrout', 'surf', 'infra', 'situ', 'env1']].describe()


# In[5]:


dfLieux.info()


# In[11]:


### Proportion of NA by variable
dfLieux.isnull().sum() * 100 / len(dfLieux)


# In[58]:


# Unique modalities by variable
print('nbv:', len(dfLieux.nbv.value_counts()))
print('vosp:', len(dfLieux.vosp.value_counts()))
print('prof:', len(dfLieux.vosp.value_counts()))
print('plan:', len(dfLieux.plan.value_counts()))
print('pr:', len(dfLieux.pr.value_counts()))
print('pr1:', len(dfLieux.pr1.value_counts()))
print('lartpc:', len(dfLieux.lartpc.value_counts()))
print('larrout:', len(dfLieux.larrout.value_counts()))
print('surf:', len(dfLieux.surf.value_counts()))
print('infra:', len(dfLieux.infra.value_counts()))
print('situ:', len(dfLieux.situ.value_counts()))
print('env1:', len(dfLieux.env1.value_counts()))


# In[24]:


# Values by variable
print(dfLieux.nbv.value_counts())
print(dfLieux.vosp.value_counts())
print(dfLieux.prof.value_counts())
print(dfLieux.plan.value_counts())
print(dfLieux.pr.value_counts())
print(dfLieux.pr1.value_counts())
print(dfLieux.lartpc.value_counts())
print(dfLieux.larrout.value_counts())
print(dfLieux.surf.value_counts())
print(dfLieux.infra.value_counts())
print(dfLieux.situ.value_counts())
print(dfLieux.env1.value_counts())


# In[47]:


# Equilibre variables
print(dfLieux.vosp.value_counts(normalize=True))
print(dfLieux.env1.value_counts(normalize=True))


# In[10]:


dfLieux[['vosp', 'prof', 'plan', 'surf', 'infra', 'situ', 'env1', 'grav']].hist(figsize=(20, 8), layout=(2, 4));


# ##### -Table Usagers-

# In[20]:


dfUsagers.head(3)


# In[14]:


### Proportion of NA by variable
dfUsagers.isnull().sum() * 100 / len(dfUsagers)


# ##### -Table Vehicles-

# In[21]:


dfVehicules.head(3)


# ### Graphs

# In[5]:


# Gravity variable in Carac dataframe
pd.DataFrame({'prop':dfCarac.grav.value_counts(normalize=True),
              'count':dfCarac.grav.value_counts()})


# In[26]:


# Gravity variable in Lieux dataframe
pd.DataFrame({'prop':dfLieux.grav.value_counts(normalize=True),
              'count':dfLieux.grav.value_counts()})


# $\color{#0005FF}{\text{Both dataframes Carac and Lieux have the same amount of accidents, they also have the same accident gravity distribution}}$

# ##### -Table Carac-

# ### Year

# In[51]:


# Display plots
plt.figure(figsize=(10, 4))
sns.countplot(dfCarac['year'], palette=['#9D9D9D'])
plt.hlines(y=len(dfCarac['year'])/16, xmin=-0.5, xmax=15.5, color='blue', alpha=0.4)
plt.title("Nombre d'accident par année");
# It seems that the number of accident never stops decreasing year after year
# The observable large decreases seem to be during 2007-2008, 2011-2012 and 2019-2020
# The number of accident seemed to be stable between 2013 and 2019


# In[52]:


# Initiating dataframe grouped by month
dfCaracGpByYear = (dfCarac.groupby(['year'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plotx
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="year", y="percentage", hue="grav", data=dfCaracGpByYear, 
             palette=['#C8C8C8','#F4B650','#F45050']);
# It seems that the gravity is less important during 2018 to 2020


# In[91]:


# data-management
dfYearGrav = pd.crosstab(dfCarac['year'], dfCarac['grav'], normalize=0).sort_values(by=4, ascending=False)
dfYearGravRaw = pd.crosstab(dfCarac['year'], dfCarac['grav']).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(dfYearGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfYearGrav.apply(lambda x: x/dfCarac['grav'].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# Even though 2018 to 2020 have the largest proportions of accident gravity 3, they also have the lowest gravity 3 ones
# It seems that the state has focused on reducing the overall number of accident but not the gravity of accidents


# ### Months

# In[54]:


# Display plots
plt.figure(figsize=(10, 4))
sns.countplot(dfCarac['mois'], 
             palette=['#96CED7', '#96CED7', 
                   '#96D7A2', '#96D7A2', '#96D7A2', 
                   '#D79696', '#D79696', '#D79696', 
                   '#D7CF96', '#D7CF96', '#D7CF96', 
                   '#96CED7'])
plt.hlines(y=len(dfCarac['mois'])/12, xmin=-0.5, xmax=11.5, color='blue', alpha=0.4)
plt.title("Nombre d'accident par mois");
# On peut observer que les mois de juin, juillet, septembre et octobre semblent avoir le plus d'accidents
# On peut observer que le mois de février compte le moins d'accidents mais il comporte aussi 28 jours


# In[55]:


# Initiating dataframe grouped by month
dfCaracGpByMonth = (dfCarac.groupby(['mois'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plotx
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="mois", y="percentage", hue="grav", data=dfCaracGpByMonth, 
             palette=['#C8C8C8','#F4B650','#F45050']);
# It seems that the gravity of accident is larger during the weekend compared to the week


# In[92]:


# data-management
dfMonthGrav = pd.crosstab(dfCarac['mois'], dfCarac['grav'], normalize=0).sort_values(by=4, ascending=False)
dfMonthGravRaw = pd.crosstab(dfCarac['mois'], dfCarac['grav']).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(dfMonthGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfMonthGrav.apply(lambda x: x/dfCarac['grav'].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# It seems that the largest proportion of accident gravity 2 & 3 happen during august and july


# ### Month day

# In[60]:


plt.figure(figsize=(20, 5))
sns.countplot(x=dfCarac['jour'], color='grey')
plt.hlines(y=len(dfCarac['jour'])/(365/12), xmin=-0.5, xmax=30.5, color='blue', alpha=0.4)
plt.title("Nombre d'accident par jour du mois");
# With no surprise, day 31 has twice as less accidents as other days of the month because it only occurs 1 months out of 2


# In[61]:


# Initiating dataframe grouped by weekday
dfCaracGpByMonthday = (dfCarac.groupby(['jour'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plotx
fig, ax = plt.subplots(figsize=(20, 4))
sns.barplot(x="jour", y="percentage", hue="grav", data=dfCaracGpByMonthday, 
             palette=['#C8C8C8','#F4B650','#F45050']);
# Hard to read this figure but no trend seems to be seen


# In[93]:


# data-management
dfMonthdayGrav = pd.crosstab(dfCarac['jour'], dfCarac['grav'], normalize=0).sort_values(by=4, ascending=False)
dfMonthdayGravRaw = pd.crosstab(dfCarac['jour'], dfCarac['grav']).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 8))
sns.heatmap(dfMonthdayGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfMonthdayGrav.apply(lambda x: x/dfCarac['grav'].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# Geniunly no trend drawn


# ### Weekday

# In[63]:


sns.countplot(x=dfCarac['weekday'], 
             palette=['#A0A491', '#A0A491', '#A0A491', '#A0A491', '#A0A491', '#E17441', '#E17441'])
plt.hlines(y=len(dfCarac['weekday'])/7, xmin=-0.5, xmax=6.5, color='blue', alpha=0.4);
plt.xticks(np.arange(7), ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
plt.title("Nombre d'accident par jour de la semaine");
# It seems that the friday is the accident day


# In[64]:


# Initiating dataframe grouped by weekday
dfCaracGpByWeekday = (dfCarac.groupby(['weekday'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plotx
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="weekday", y="percentage", hue="grav", data=dfCaracGpByWeekday, 
             palette=['#C8C8C8','#F4B650','#F45050'])
ax.set_xticklabels(['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']);
# It seems that the gravity of accident is larger during the weekend compared to the week


# In[94]:


# data-management
dfWeekdayGrav = pd.crosstab(dfCarac['weekday'], dfCarac['grav'], normalize=0).sort_values(by=4, ascending=False)
dfWeekdayGravRaw = pd.crosstab(dfCarac['weekday'], dfCarac['grav']).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(dfWeekdayGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfWeekdayGrav.apply(lambda x: x/dfCarac['grav'].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# Actually, the largest proportion of accident gravity 2 is during sunday then saturday


# ### Hour of the day

# In[75]:


sns.countplot(x=dfCarac['hour'], 
             palette=['#090F23', '#03060F', '#040D29', '#484743', '#999588', '#CDC5A9', 
                     '#DDD3B0', '#F5E7B1', '#FFECA4', '#FFDEA4', '#FFD2A4', '#FFCBA4', 
                     '#FFBDA4', '#FFAFA4', '#FFA4A4', '#FFA4D6', '#EDA4FF', '#C2A4FF', 
                     '#A4AAFF', '#839AE5', '#5B71B8', '#3C508F', '#26366A', '#152043'])
plt.xticks([0, 3, 6, 9, 12, 15, 18, 21], ['Minuit', '3h', '6h', '9h', 'Midi', '15h', '18h', '21h'])
plt.title("Nombre d'accident par heure de la journée")
plt.hlines(y=len(dfCarac['hour'])/24, xmin=-0.5, xmax=23.5, color='blue', alpha=0.4);
# It seems that most accident happen between 4pm and 7pm which is when people usually go back home and the sun goes down
# At 7am, the number of accident drastically increase and really goes down after 8pm


# In[33]:


# Initiating dataframe grouped by hour
dfCaracGpByHour = (dfCarac.groupby(['hour'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plotx
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="hour", y="percentage", hue="grav", data=dfCaracGpByHour, 
             palette=['#C8C8C8','#F4B650','#F45050']);
# Wow, it seems that the gravity of accidents is worst during the night (22pm-6am)
# More than 5% gravity 2 during the night against less than 4% during full day


# In[95]:


# data-management
dfHourGrav = pd.crosstab(dfCarac['hour'], dfCarac['grav'], normalize=0).sort_values(by=4, ascending=False)
dfHourGravRaw = pd.crosstab(dfCarac['hour'], dfCarac['grav']).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 8))
sns.heatmap(dfHourGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfHourGrav.apply(lambda x: x/dfCarac['grav'].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# Proposition: creating a full night variable [0-6am] (yes/no)


# In[141]:


# Initiating gravity proportion of CCA hour variable
propGrav = dfCarac['grav'].value_counts(normalize=True)

# Initiating dataframe grouped by hour
dfCaracGpByHour = (dfCarac.groupby(['hour'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    # 1st plot
sns.barplot(x="hour", y="percentage", hue="grav", data=dfCaracGpByHour, 
             palette=['#C8C8C8','#F4B650','#F45050'], ax=ax[0])
        # text outside the plot
# ax[0].set_xticks(np.arange(0, 5, 1))
# ax[0].set_xticklabels(['day', 'dawn', 'night\nwo light', 'night\nwi light not lit', 'night\n wi light lit'])
ax[0].set_title('Gravité des accidents en fonction de l\'heure de la journée')
ax[0].set_xlabel('Hour')
ax[0].set_ylabel('%', rotation=0)
        # adding horizontal overall proportion by gravity
ax[0].axhline(y=propGrav.loc[2]*100, color='#C8C8C8', linestyle='--')
ax[0].axhline(y=propGrav.loc[3]*100, color='#F4B650', linestyle='--')
ax[0].axhline(y=propGrav.loc[4]*100, color='#F45050', linestyle='--')
    # 2nd plot
sns.heatmap(dfHourGrav.apply(lambda x: x/dfCarac['grav'].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1])
        # text outside the plot
ax[1].set_xticks([0.5, 1.5, 2.5])
ax[1].set_xticklabels(['léger', 'hostpitalisé', 'tué'])
ax[1].set_title('Fold de gravité des accidents en fonction de l\'heure de la journée')
ax[1].set_xlabel('')
ax[1].set_ylabel('');


# ### Lum

# In[74]:


sns.countplot(x=dfCarac['lum'][(dfCarac['lum']!=-1)], 
             palette=['#FF5D5D', '#5774B8', '#000000', '#000000', '#FDEC8B'])
plt.title("Nombre d'accident par condition luminaire")
plt.hlines(y=len(dfCarac['lum'][(dfCarac['lum']!=-1)])/5, xmin=-0.5, xmax=4.5, color='blue', alpha=0.4);
# It seems that most accident happen during the full day


# In[75]:


# Initiating dataframe grouped by hour
dfCaracGpByLum = (dfCarac[(dfCarac['lum']!=-1)].groupby(['lum'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plots
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="lum", y="percentage", hue="grav", data=dfCaracGpByLum, 
             palette=['#C8C8C8','#F4B650','#F45050']);
# Wow, it seems that the gravity of accidents is worst during the night (22pm-6am)
# More than 5% gravity 2 during the night against less than 4% during full day


# In[96]:


# data-management
dfLumGrav = pd.crosstab(dfCarac['lum'][(dfCarac['lum']!=-1)], dfCarac['grav'][(dfCarac['lum']!=-1)], normalize=0).sort_values(by=4, ascending=False)
dfLumGravRaw = pd.crosstab(dfCarac['lum'][(dfCarac['lum']!=-1)], dfCarac['grav'][(dfCarac['lum']!=-1)]).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfLumGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfLumGrav.apply(lambda x: x/dfCarac['grav'][(dfCarac['lum']!=-1)].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# The night without public lightning seems to have a drastic increase of gravity 2 and 3 accidents rate (10% and 44%)!
# Then the two other cases where no much light is on have interesting gravity 2 increase accident rates


# In[136]:


# Initiating gravity proportion of CCA lum variable
propGrav = dfCarac['grav'][(dfCarac['lum']!=-1)].value_counts(normalize=True)

# Initiating dataframe grouped by hour
dfCaracGpByLum = (dfCarac[(dfCarac['lum']!=-1)].groupby(['lum'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(18, 4))
    # 1st plot
sns.barplot(x="lum", y="percentage", hue="grav", data=dfCaracGpByLum, 
             palette=['#C8C8C8','#F4B650','#F45050'], ax=ax[0])
        # adding horizontal overall proportion by gravity
ax[0].axhline(y=propGrav.loc[2]*100, color='#C8C8C8', linestyle='--')
ax[0].axhline(y=propGrav.loc[3]*100, color='#F4B650', linestyle='--')
ax[0].axhline(y=propGrav.loc[4]*100, color='#F45050', linestyle='--')
        # text outside the plot
ax[0].set_xticks(np.arange(0, 5, 1))
ax[0].set_xticklabels(['day', 'dawn', 'night\nwo light', 'night\nwi light not lit', 'night\n wi light lit'])
ax[0].set_title('Gravité des accidents en fonction de la luminosité')
ax[0].set_xlabel('')
ax[0].set_ylabel('%', rotation=0)
    # 2nd plot
sns.heatmap(dfLumGrav.apply(lambda x: x/dfCarac['grav'][(dfCarac['lum']!=-1)].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1])
        # text outside the plot
ax[1].set_xticks([0.5, 1.5, 2.5])
ax[1].set_xticklabels(['léger', 'hostpitalisé', 'tué'])
ax[1].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
ax[1].set_yticklabels(['night\nwo light', 'night\nwi light not lit', 'dawn', 'day', 'night\n wi light lit'], rotation=0)
ax[1].set_title('Fold de gravité des accidents en fonction de la luminosité')
ax[1].set_xlabel('')
ax[1].set_ylabel('');


# ### Atm

# In[54]:


sns.countplot(x=dfCarac['atm'][(dfCarac['atm']!=-1)], 
             palette=['#8A8A8A', '#090F23', '#090F23', '#090F23', '#090F23', '#090F23', 
                     '#090F23', '#090F23', '#090F23'])
plt.title("Nombre d'accident par condition athmosphérique")
plt.hlines(y=len(dfCarac['atm'][(dfCarac['atm']!=-1)])/9, xmin=-0.5, xmax=8.5, color='blue', alpha=0.4);
# It seems that most accident happen with normal atmospheric conditions, then light rain


# In[35]:


# Initiating dataframe grouped by hour
dfCaracGpByAtm = (dfCarac[(dfCarac['atm']!=-1)].groupby(['atm'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plotx
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="atm", y="percentage", hue="grav", data=dfCaracGpByAtm, 
             palette=['#C8C8C8','#F4B650','#F45050']);
# Wow, it seems that the gravity of accidents is worst during fog/smoke, strong wind/storm, dazzling weather and 'other'


# In[97]:


# data-management
dfAtmGrav = pd.crosstab(dfCarac['atm'][(dfCarac['atm']!=-1)], dfCarac['grav'][(dfCarac['atm']!=-1)], normalize=0).sort_values(by=4, ascending=False)
dfAtmGravRaw = pd.crosstab(dfCarac['atm'][(dfCarac['atm']!=-1)], dfCarac['grav'][(dfCarac['atm']!=-1)]).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfAtmGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfAtmGrav.apply(lambda x: x/dfCarac['grav'][(dfCarac['atm']!=-1)].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# These graphs confirm that both gravity 2 and 3 are increase for groups 5, 6, 7 and 9


# ### Col

# In[89]:


sns.countplot(x=dfCarac['col'][(dfCarac['col']!=-1)], 
             palette=['#090F23', '#090F23', '#090F23', '#090F23', '#090F23', '#090F23', '#090F23'])
plt.title("Nombre d'accident par type de collision")
plt.hlines(y=len(dfCarac['col'][(dfCarac['col']!=-1)])/7, xmin=-0.5, xmax=6.5, color='blue', alpha=0.4);
# Other collision and by the side are the most counted
# It is quite disturbing to see that the most filled class is the group 'other'
# There is a feeling that this variable was not well defined or filled


# In[36]:


# Initiating dataframe grouped by hour
dfCaracGpByCol = (dfCarac[(dfCarac['col']!=-1)].groupby(['col'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plotx
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="col", y="percentage", hue="grav", data=dfCaracGpByCol, 
             palette=['#C8C8C8','#F4B650','#F45050']);
# Les groupes 2, 3 et 4 sont très peu impactés en termes de gravité alors que les groupes 1, 6 et 7 semblent impactants


# In[98]:


# data-management
dfColGrav = pd.crosstab(dfCarac['col'][(dfCarac['col']!=-1)], dfCarac['grav'][(dfCarac['col']!=-1)], normalize=0).sort_values(by=4, ascending=False)
dfColGravRaw = pd.crosstab(dfCarac['col'][(dfCarac['col']!=-1)], dfCarac['grav'][(dfCarac['col']!=-1)]).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfColGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfColGrav.apply(lambda x: x/dfCarac['grav'][(dfCarac['col']!=-1)].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# La collision de type 1 est celle qui maximise les accidents de gravité 3 avec un fort taux de gravité 2
# Les collisions de type 6 et 7 sont celles qui maximisent les accidents de gravité 2


# ### Date

# In[40]:


# Initiating variable
varDate = dfCarac.date.value_counts().sort_index()

# Display plot
plt.figure(figsize=(20, 5))
plt.plot(varDate.index, varDate, color='#CE5E7D')
plt.axhline(y=varDate.mean(), color='k', linestyle='--')
plt.title('Number of accident through the time')
plt.ylim([0, 400]);


# In[41]:


# Distribution nombre d'accidents par gravité
plt.boxplot([varDateGrav2, varDateGrav3, varDateGrav4])
plt.xticks(ticks=[1, 2, 3], labels=['2', '3', '4']);

# Mean accidents by gravity
print(varDateGrav2.mean())
print(varDateGrav3.mean())
print(varDateGrav4.mean())


# In[46]:


# Initiating variables
varDateGrav2 = dfCarac[(dfCarac.grav==2)].date.value_counts().sort_index()
varDateGrav3 = dfCarac[(dfCarac.grav==3)].date.value_counts().sort_index()
varDateGrav4 = dfCarac[(dfCarac.grav==4)].date.value_counts().sort_index()

# Display plots by gravity
plt.figure(figsize=(20, 5))
plt.subplot(131)
plt.plot(varDateGrav2.index, varDateGrav2, color='#C8C8C8')
plt.axhline(y=varDateGrav2.mean(), color='k', linestyle='--')
plt.title('Number of accident through the time (gravity 2)')
plt.ylim([0, 250]);
plt.subplot(132)
plt.plot(varDateGrav3.index, varDateGrav3, color='#F4B650')
plt.axhline(y=varDateGrav3.mean(), color='k', linestyle='--')
plt.title('Number of accident through the time (gravity 3)')
plt.ylim([0, 175]);
plt.subplot(133)
plt.plot(varDateGrav4.index, varDateGrav4, color='#F45050')
plt.axhline(y=varDateGrav4.mean(), color='k', linestyle='--')
plt.title('Number of accident through the time (gravity 4)')
plt.ylim([0, 35]);


# In[ ]:


# Initiating folds
dfDateGrav = pd.crosstab(dfCarac['date'], dfCarac['grav'], normalize=0).sort_values(by=4, ascending=False)
dfDateGravLambda = dfDateGrav.apply(lambda x: x/dfCarac['grav'].value_counts(normalize=True), axis=1)
dfDateGravLambdaGrav2 = dfDateGravLambda[2].sort_index()
dfDateGravLambdaGrav3 = dfDateGravLambda[3].sort_index()
dfDateGravLambdaGrav4 = dfDateGravLambda[4].sort_index()


# In[56]:


# Display folds by gravity
plt.figure(figsize=(20, 5))
plt.subplot(131)
plt.plot(dfDateGravLambdaGrav2.index, dfDateGravLambdaGrav2, color='#C8C8C8')
plt.axhline(y=dfDateGravLambdaGrav2.mean(), color='k', linestyle='--')
plt.title('Fold of gravity 2 accidents')
plt.ylim([0, 5]);
plt.subplot(132)
plt.plot(dfDateGravLambdaGrav3.index, dfDateGravLambdaGrav3, color='#F4B650')
plt.axhline(y=dfDateGravLambdaGrav3.mean(), color='k', linestyle='--')
plt.title('Fold of gravity 3 accidents')
plt.ylim([0, 5]);
plt.subplot(133)
plt.plot(dfDateGravLambdaGrav4.index, dfDateGravLambdaGrav4, color='#F45050')
plt.axhline(y=dfDateGravLambdaGrav4.mean(), color='k', linestyle='--')
plt.title('Fold of gravity 4 accidents')
plt.ylim([0, 5]);
# There is a clear time-related event impacting gravity 4 accidents


# In[42]:


# Distribution analysis
sns.kdeplot(varDate, shade=True);


# ### Agg

# In[39]:


print(dfCarac['agg'].value_counts())
print(dfCarac['agg'].value_counts(normalize=True))


# In[35]:


sns.countplot(x=dfCarac['agg'], color='grey')
plt.title("Nombre d'accident en agglo/hors agglo")
plt.hlines(y=len(dfCarac['agg'])/2, xmin=-0.5, xmax=1.5, color='blue', alpha=0.4);
# Many accidents when there are 2 route tracks


# In[36]:


# Initiating dataframe grouped
dfCaracGpByAgg = (dfCarac.groupby(['agg'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plotx
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="agg", y="percentage", hue="grav", data=dfCaracGpByAgg, 
             palette=['#C8C8C8','#F4B650','#F45050']);
# Les accidents semblent plus graves hors agglomération


# In[126]:


# data-management
dfAggGrav = pd.crosstab(dfCarac['agg'], dfCarac['grav'], normalize=0).sort_values(by=4, ascending=False)
dfAggGravRaw = pd.crosstab(dfCarac['agg'], dfCarac['grav']).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfAggGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfAggGrav.apply(lambda x: x/dfCarac['grav'].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# 2 fois plus d'accidents de gravité 2 hors agglomération


# ### nbv

# In[78]:


dfLieux.nbv[(dfLieux.nbv<8) & (dfLieux.nbv>-1)].value_counts()


# In[83]:


sns.countplot(x=dfLieux.nbv[(dfLieux.nbv<7) & (dfLieux.nbv>-1)], color='grey')
plt.title("Nombre d'accident par nombre de voies")
plt.hlines(y=len(dfLieux['nbv'][(dfLieux.nbv<7) & (dfLieux.nbv>-1)])/7, xmin=-0.5, xmax=6.5, color='blue', alpha=0.4);
# Many accidents when there are 2 route tracks


# In[37]:


# Initiating dataframe grouped
dfCaracGpByNbv = (dfLieux[(dfLieux.nbv<7) & (dfLieux.nbv>-1)].groupby(['nbv'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plotx
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="nbv", y="percentage", hue="grav", data=dfCaracGpByNbv, 
             palette=['#C8C8C8','#F4B650','#F45050']);
# Les groupes 0 et 2 semblent avoir un taux élevé d'accidents gravité 2 et 3


# In[99]:


# data-management
dfNbvGrav = pd.crosstab(dfLieux.nbv[(dfLieux.nbv<7) & (dfLieux.nbv>-1)], dfCarac['grav'][(dfLieux.nbv<7) & (dfLieux.nbv>-1)], normalize=0).sort_values(by=4, ascending=False)
dfNbvGravRaw = pd.crosstab(dfLieux.nbv[(dfLieux.nbv<7) & (dfLieux.nbv>-1)], dfCarac['grav'][(dfLieux.nbv<7) & (dfLieux.nbv>-1)]).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfNbvGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfNbvGrav.apply(lambda x: x/dfLieux['grav'][(dfLieux.nbv<7) & (dfLieux.nbv>-1)].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# Les groupes 0 et 2 semblent avoir un taux élevé d'accidents gravité 2 et 3


# ### vosp

# In[88]:


dfLieux.vosp[(dfLieux.vosp>-1)].value_counts(normalize=True)


# In[56]:


sns.countplot(x=dfLieux.vosp[(dfLieux.vosp>-1)], color='grey')
plt.title("Nombre d'accident par présence de voie")
plt.hlines(y=len(dfLieux['vosp'][(dfLieux.vosp>-1)])/4, xmin=-0.5, xmax=3.5, color='blue', alpha=0.4);
# Many accidents when there are no additional reserved track


# In[38]:


# Initiating dataframe grouped
dfCaracGpByVosp = (dfLieux[(dfLieux.vosp>-1)].groupby(['vosp'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plot
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="vosp", y="percentage", hue="grav", data=dfCaracGpByVosp, 
             palette=['#C8C8C8','#F4B650','#F45050']);
# XXX


# In[100]:


# data-management
dfVospGrav = pd.crosstab(dfLieux['vosp'][(dfLieux['vosp']!=-1)], dfLieux['grav'][(dfLieux['vosp']!=-1)], normalize=0).sort_values(by=4, ascending=False)
dfVospGravRaw = pd.crosstab(dfLieux['vosp'][(dfLieux['vosp']!=-1)], dfLieux['grav'][(dfLieux['vosp']!=-1)]).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfVospGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfVospGrav.apply(lambda x: x/dfLieux['grav'][(dfLieux['vosp']!=-1)].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# XXX


# ### prof

# In[65]:


dfLieux.prof[(dfLieux.prof>-1)].value_counts()


# In[82]:


sns.countplot(x=dfLieux.prof[(dfLieux.prof>-1)], color='grey')
plt.title("Nombre d'accident par pente de la route")
plt.hlines(y=len(dfLieux['prof'][(dfLieux.prof>-1)])/5, xmin=-0.5, xmax=4.5, color='blue', alpha=0.4);
# Many accidents when there is a dish track


# In[39]:


# Initiating dataframe grouped
dfCaracGpByProf = (dfLieux[(dfLieux.prof>-1)].groupby(['prof'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plot
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="prof", y="percentage", hue="grav", data=dfCaracGpByProf, 
             palette=['#C8C8C8','#F4B650','#F45050']);
# XXX


# In[101]:


# data-management
dfProfGrav = pd.crosstab(dfLieux['prof'][(dfLieux['prof']!=-1)], dfLieux['grav'][(dfLieux['prof']!=-1)], normalize=0).sort_values(by=4, ascending=False)
dfProfGravRaw = pd.crosstab(dfLieux['prof'][(dfLieux['prof']!=-1)], dfLieux['grav'][(dfLieux['prof']!=-1)]).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfProfGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfProfGrav.apply(lambda x: x/dfLieux['grav'][(dfLieux['prof']!=-1)].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# XXX


# ### plan

# In[67]:


dfLieux.plan[(dfLieux.plan>-1)].value_counts()


# In[84]:


sns.countplot(x=dfLieux.plan[(dfLieux.plan>-1)], color='grey')
plt.title("Nombre d'accident par plan de route")
plt.hlines(y=len(dfLieux['plan'][(dfLieux.plan>-1)])/5, xmin=-0.5, xmax=4.5, color='blue', alpha=0.4);
# Many accidents when there are straight part


# In[40]:


# Initiating dataframe grouped
dfCaracGpByPlan = (dfLieux[(dfLieux.plan>-1)].groupby(['plan'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plot
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="plan", y="percentage", hue="grav", data=dfCaracGpByPlan, 
             palette=['#C8C8C8','#F4B650','#F45050']);
# XXX


# In[102]:


# data-management
dfPlanGrav = pd.crosstab(dfLieux['plan'][(dfLieux['plan']!=-1)], dfLieux['grav'][(dfLieux['plan']!=-1)], normalize=0).sort_values(by=4, ascending=False)
dfPlanGravRaw = pd.crosstab(dfLieux['plan'][(dfLieux['plan']!=-1)], dfLieux['grav'][(dfLieux['plan']!=-1)]).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfPlanGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfPlanGrav.apply(lambda x: x/dfLieux['grav'][(dfLieux['plan']!=-1)].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# XXX


# ### lartpc

# In[7]:


dfLieux['lartpc'].describe()


# In[10]:


# Display distributions
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.kdeplot(dfLieux['lartpc'], shade=True, ax=ax[0])
sns.kdeplot(dfLieux['lartpc'][(dfLieux['lartpc']<20)], shade=True, ax=ax[1])
sns.kdeplot(dfLieux['lartpc'][(dfLieux['lartpc']<20) & (dfLieux['lartpc']>0)], shade=True, ax=ax[2]);
print('mean=', round(dfLieux.lartpc.mean()))
print('mean=', round(dfLieux.lartpc[(dfLieux['lartpc']<20)].mean()))
# There are mainly 0 values and values around a mean of 5m but values around 15, 10 and 5 (mean of 1 when outliers removed)


# In[25]:


# Distribution by gravity
sns.kdeplot(
   data=dfLieux[(dfLieux['lartpc']<20) & (dfLieux['lartpc']>0)], x="lartpc", hue="grav",
   fill=True, common_norm=False, palette=['#C8C8C8','#F4B650','#F45050'],
   alpha=.5, linewidth=2
);


# In[36]:


sns.kdeplot(
   data=dfLieux[(dfLieux['lartpc']<20) & (dfLieux['lartpc']>0) & (dfLieux['grav']==2)], x="lartpc",
   fill=True, common_norm=False, color=['#C8C8C8'],
   alpha=.5, linewidth=2)
plt.ylim(0, 0.6);


# In[37]:


sns.kdeplot(
   data=dfLieux[(dfLieux['lartpc']<20) & (dfLieux['lartpc']>0) & (dfLieux['grav']==3)], x="lartpc",
   fill=True, common_norm=False, color=['#F4B650'],
   alpha=.5, linewidth=2)
plt.ylim(0, 0.6);


# In[39]:


sns.kdeplot(
   data=dfLieux[(dfLieux['lartpc']<20) & (dfLieux['lartpc']>0) & (dfLieux['grav']==4)], x="lartpc",
   fill=True, common_norm=False, color=['#F45050'],
   alpha=.5, linewidth=2)
plt.ylim(0, 0.6);


# ### larrout

# In[91]:


dfLieux.larrout[(dfLieux.larrout>-1)].describe()


# In[100]:


# Display distributions
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.kdeplot(dfLieux.larrout[(dfLieux.larrout>-1)], shade=True, ax=ax[0])
sns.kdeplot(dfLieux.larrout[(dfLieux.larrout>-1) & (dfLieux.larrout<250)], shade=True, ax=ax[1]);
print('mean=', round(dfLieux.larrout[(dfLieux.larrout>-1)].mean()))
print('mean=', round(dfLieux.larrout[(dfLieux.larrout>-1) & (dfLieux.larrout<250)].mean()))
# It seems that most values are 0 and a mean around 58m (54m without outliers)


# In[42]:


# Distribution by gravity
sns.kdeplot(
   data=dfLieux[(dfLieux.larrout>-1) & (dfLieux.larrout<250)], x="larrout", hue="grav",
   fill=True, common_norm=False, palette=['#C8C8C8','#F4B650','#F45050'],
   alpha=.5, linewidth=2,
);


# In[49]:


sns.kdeplot(
   data=dfLieux[(dfLieux.larrout>-1) & (dfLieux.larrout<250) & (dfLieux.grav==2)], x="larrout", hue="grav",
   fill=True, common_norm=False, palette=['#C8C8C8'],
   alpha=.5, linewidth=2)
plt.ylim(0, 0.04);


# In[51]:


sns.kdeplot(
   data=dfLieux[(dfLieux.larrout>-1) & (dfLieux.larrout<250) & (dfLieux.grav==3)], x="larrout", hue="grav",
   fill=True, common_norm=False, palette=['#F4B650'],
   alpha=.5, linewidth=2)
plt.ylim(0, 0.04);


# In[50]:


sns.kdeplot(
   data=dfLieux[(dfLieux.larrout>-1) & (dfLieux.larrout<250) & (dfLieux.grav==4)], x="larrout", hue="grav",
   fill=True, common_norm=False, palette=['#F45050'],
   alpha=.5, linewidth=2)
plt.ylim(0, 0.04);


# ### surf

# In[86]:


#cat
dfLieux.surf[(dfLieux.surf>-1)].value_counts()


# In[91]:


sns.countplot(x=dfLieux.surf[(dfLieux.surf>-1)], color='grey')
plt.title("Nombre d'accident par surface de la route (météo)")
plt.hlines(y=len(dfLieux['surf'][(dfLieux.vosp>-1)])/10, xmin=-0.5, xmax=9.5, color='blue', alpha=0.4);
# Many accidents when there is normal or wet meteo


# In[43]:


# Initiating dataframe grouped
dfCaracGpBySurf = (dfLieux[(dfLieux.surf>-1)].groupby(['surf'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plot
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="surf", y="percentage", hue="grav", data=dfCaracGpBySurf, 
             palette=['#C8C8C8','#F4B650','#F45050']);
# XXX


# In[103]:


# data-management
dfSurfGrav = pd.crosstab(dfLieux['surf'][(dfLieux['surf']!=-1)], dfLieux['grav'][(dfLieux['surf']!=-1)], normalize=0).sort_values(by=4, ascending=False)
dfSurfGravRaw = pd.crosstab(dfLieux['surf'][(dfLieux['surf']!=-1)], dfLieux['grav'][(dfLieux['surf']!=-1)]).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfSurfGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfSurfGrav.apply(lambda x: x/dfLieux['grav'][(dfLieux['surf']!=-1)].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# XXX


# ### infra

# In[93]:


#cat
dfLieux.infra[(dfLieux.infra>-1)].value_counts()


# In[95]:


sns.countplot(x=dfLieux.infra[(dfLieux.infra>-1)], color='grey')
plt.title("Nombre d'accident en fonction des infrastuctures supplémentaires")
plt.hlines(y=len(dfLieux['infra'][(dfLieux.infra>-1)])/10, xmin=-0.5, xmax=9.5, color='blue', alpha=0.4);
# Many accidents when there are no additional infrastructures


# In[44]:


# Initiating dataframe grouped
dfCaracGpByInfra = (dfLieux[(dfLieux.infra>-1)].groupby(['infra'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plot
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="infra", y="percentage", hue="grav", data=dfCaracGpByInfra, 
             palette=['#C8C8C8','#F4B650','#F45050']);
# XXX


# In[104]:


# data-management
dfInfraGrav = pd.crosstab(dfLieux['infra'][(dfLieux['infra']!=-1)], dfLieux['grav'][(dfLieux['infra']!=-1)], normalize=0).sort_values(by=4, ascending=False)
dfInfraGravRaw = pd.crosstab(dfLieux['infra'][(dfLieux['infra']!=-1)], dfLieux['grav'][(dfLieux['infra']!=-1)]).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfInfraGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfInfraGrav.apply(lambda x: x/dfLieux['grav'][(dfLieux['infra']!=-1)].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# XXX


# ### situ

# In[78]:


#cat
dfLieux.situ[(dfLieux.situ>-1)].value_counts()


# In[98]:


sns.countplot(x=dfLieux.situ[(dfLieux.situ>-1)], color='grey')
plt.title("Nombre d'accident par zone d'accident")
plt.hlines(y=len(dfLieux['situ'][(dfLieux.situ>-1)])/8, xmin=-0.5, xmax=7.5, color='blue', alpha=0.4);
# Many accidents happen on the road


# In[45]:


# Initiating dataframe grouped
dfCaracGpBySitu = (dfLieux[(dfLieux.situ>-1)].groupby(['situ'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plot
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="situ", y="percentage", hue="grav", data=dfCaracGpBySitu, 
             palette=['#C8C8C8','#F4B650','#F45050']);
# XXX


# In[105]:


# data-management
dfSituGrav = pd.crosstab(dfLieux['situ'][(dfLieux['situ']!=-1)], dfLieux['grav'][(dfLieux['situ']!=-1)], normalize=0).sort_values(by=4, ascending=False)
dfSituGravRaw = pd.crosstab(dfLieux['situ'][(dfLieux['situ']!=-1)], dfLieux['grav'][(dfLieux['situ']!=-1)]).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfSituGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfSituGrav.apply(lambda x: x/dfLieux['grav'][(dfLieux['situ']!=-1)].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# XXX


# In[48]:


plt.plot(dfSituGrav.apply(lambda x: x/dfLieux['grav'][(dfLieux['situ']!=-1)].value_counts(normalize=True), axis=1)[4], 'o', color='#F45050')
plt.plot(dfSituGrav.apply(lambda x: x/dfLieux['grav'][(dfLieux['situ']!=-1)].value_counts(normalize=True), axis=1)[3], 'o', color='#F4B650')
plt.plot(dfSituGrav.apply(lambda x: x/dfLieux['grav'][(dfLieux['situ']!=-1)].value_counts(normalize=True), axis=1)[2], 'o', color='#C8C8C8')
plt.axhline(y=1, color='k', linestyle='--');


# In[49]:


# Data-management
dfSituGravFold = dfSituGrav.apply(lambda x: x/dfLieux['grav'][(dfLieux['situ']!=-1)].value_counts(normalize=True), axis=1).stack().reset_index()

# Rename columns
dfSituGravFold.rename(columns={'level_1':'grav', 0:'fold'}, inplace=True)

# Display plot
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x='situ', y='fold', hue='grav', data=dfSituGravFold, 
             palette=['#C8C8C8','#F4B650','#F45050']);


# ### env1

# In[79]:


#cat
dfLieux.env1.value_counts()


# In[101]:


sns.countplot(x=dfLieux.env1, color='grey')
plt.title("Nombre d'accident par présence d'école à proximité")
plt.hlines(y=len(dfLieux['env1'])/3, xmin=-0.5, xmax=2.5, color='blue', alpha=0.4);
# Many accidents happen when there are no school near


# In[50]:


# Initiating dataframe grouped
dfCaracGpByEnv1 = (dfLieux.groupby(['env1'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plot
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="env1", y="percentage", hue="grav", data=dfCaracGpByEnv1, 
             palette=['#C8C8C8','#F4B650','#F45050']);
# XXX


# In[106]:


# data-management
dfEnv1Grav = pd.crosstab(dfLieux['env1'], dfLieux['grav'], normalize=0).sort_values(by=4, ascending=False)
dfEnv1GravRaw = pd.crosstab(dfLieux['env1'], dfLieux['grav']).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfEnv1Grav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfEnv1Grav.apply(lambda x: x/dfLieux['grav'].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# XXX


# ### catu

# In[53]:


dfUsagers.grav.value_counts(normalize=True)


# In[18]:


dfUsagers.catu.value_counts(normalize=True)


# In[50]:


sns.countplot(x=dfUsagers.catu, color='grey')
plt.title("Nombre d'accident par présence d'école à proximité")
plt.hlines(y=len(dfUsagers['catu'])/4, xmin=-0.5, xmax=3.5, color='blue', alpha=0.4);
# XXX


# In[39]:


# Initiating dataframe grouped
dfUsagersGpByCatu = (dfUsagers.groupby(['catu'])['grav2']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav2'))

# Display plot
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="catu", y="percentage", hue="grav2", data=dfUsagersGpByCatu, 
             palette=['grey', '#C8C8C8', '#F4B650', '#F45050']);
# Les piétons semblent avoir plus d'hospitalisations
# Les piétons pedestres semblent avoir plus d'accidents de gravité 4


# In[107]:


# data-management
dfCatuGrav = pd.crosstab(dfUsagers['catu'], dfUsagers['grav2'], normalize=0).sort_values(by=4, ascending=False)
dfCatuGravRaw = pd.crosstab(dfUsagers['catu'], dfUsagers['grav2']).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfCatuGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfCatuGrav.apply(lambda x: x/dfUsagers['grav2'].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# XXX


# ### grav

# In[17]:


dfUsagers.grav.value_counts(normalize=True)


# In[31]:


sns.countplot(x=dfUsagers.grav2, palette=['grey', '#C8C8C8', '#F4B650', '#F45050'])
plt.title("Gravité de l'accident de chaque victime")
plt.hlines(y=len(dfUsagers['grav'])/4, xmin=-0.5, xmax=3.5, color='blue', alpha=0.4)
plt.xticks(ticks=np.arange(0, 4, 1), labels=['indemne', 'léger', 'hospitalisé', 'tué'])
plt.xlabel('Gravité accident');
# Cette variable devrait être présentée


# ### sexe

# In[16]:


dfUsagers.sexe.value_counts(normalize=True)


# In[158]:


sns.countplot(x=dfUsagers.sexe, color='grey')
plt.title("Nombre d'accident par Sexe")
plt.hlines(y=len(dfUsagers['sexe'])/2, xmin=-0.5, xmax=1.5, color='blue', alpha=0.4);
# XXX


# In[34]:


# Initiating dataframe grouped
dfUsagersGpBySexe = (dfUsagers.groupby(['sexe'])['grav2']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav2'))

# Display plot
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="sexe", y="percentage", hue="grav2", data=dfUsagersGpBySexe, 
             palette=['grey', '#C8C8C8', '#F4B650', '#F45050']);
# Plus de tués chez les hommes mais plus de blessés indemnes également, paradoxalement


# In[108]:


# data-management
dfSexeGrav = pd.crosstab(dfUsagers['sexe'], dfUsagers['grav2'], normalize=0).sort_values(by=4, ascending=False)
dfSexeGravRaw = pd.crosstab(dfUsagers['sexe'], dfUsagers['grav2']).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfSexeGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfSexeGrav.apply(lambda x: x/dfUsagers['grav2'].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# XXX


# ### trajet

# In[19]:


dfUsagers.trajet.value_counts(normalize=True)


# In[24]:


sns.countplot(x=dfUsagers.trajet[(dfUsagers.trajet>0)], color='grey')
plt.title("Nombre d'accident par type de trajet")
plt.hlines(y=len(dfUsagers['trajet'][(dfUsagers.trajet>0)])/6, xmin=-0.5, xmax=5.5, color='blue', alpha=0.4);
# XXX


# In[47]:


# Initiating dataframe grouped
dfUsagersGpByTrajet = (dfUsagers[(dfUsagers.trajet>0)].groupby(['trajet'])['grav2']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav2'))

# Display plot
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="trajet", y="percentage", hue="grav2", data=dfUsagersGpByTrajet, 
             palette=['grey', '#C8C8C8', '#F4B650', '#F45050']);
# Plus de tués chez les hommes mais plus de blessés indemnes également, paradoxalement


# In[109]:


# data-management
dfTrajetGrav = pd.crosstab(dfUsagers['trajet'][(dfUsagers.trajet>0)], dfUsagers['grav2'][(dfUsagers.trajet>0)], normalize=0).sort_values(by=4, ascending=False)
dfTrajetGravRaw = pd.crosstab(dfUsagers['trajet'][(dfUsagers.trajet>0)], dfUsagers['grav2'][(dfUsagers.trajet>0)]).sort_values(by=4, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfTrajetGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfTrajetGrav.apply(lambda x: x/dfUsagers['grav2'][(dfUsagers.trajet>0)].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# XXX


# ### Inferential analysis

# In[10]:


# Proportions
propUsagersGrav = round(dfUsagers.grav2.value_counts(normalize=True), 2)
propCaracGrav = round(dfCarac.grav.value_counts(normalize=True), 2)
print(propUsagersGrav)
print(propCaracGrav)

# Display plot
plt.figure(figsize=(10, 5))
    # plot
plt.bar([1, 2, 3, 4], dfUsagers.grav2.value_counts(normalize=True), label='Exemple 1', color=['grey', '#C8C8C8', '#F4B650', '#F45050'])
plt.bar([6, 7, 8], dfCarac.grav.value_counts(normalize=True), label='Exemple 2', color=['#C8C8C8', '#F4B650', '#F45050'])
plt.plot([4.5, 4.5], [0, 0.8], color='k')
    # text around the plot
plt.ylim([0, 0.8])
plt.xticks(ticks=np.arange(1, 9, 1), labels=['indemne', 'léger', 'hospitalisé', 'tué', 'indemne', 'léger', 'hospitalisé', 'tué'])
plt.title('Gravité des accidents, avant vs après avoir raffiné la variable')
plt.ylabel('%', rotation=0)
    # text inside the plot
plt.text(2, 0.7, 'Avant', weight='bold', fontsize=20)
plt.text(6, 0.7, 'Après', weight='bold', fontsize=20)
plt.text(0.75, propUsagersGrav.loc[1]+0.025, propUsagersGrav.loc[1], weight='bold')
for i in np.arange(2, 5):
    plt.text([1.8, 2.9, 3.8][i-2], propUsagersGrav.loc[i]+0.02, propUsagersGrav.loc[i], weight='bold')
    plt.text([5.8, 6.9, 7.8][i-2], propCaracGrav.loc[i]+0.02, propCaracGrav.loc[i], weight='bold');


# In[102]:


### V Cramer score & p-value
# Function
def vCramerChisqPvalue(varname, var, contingtableRaw):
    res = varname, round(V_cramer(contingtableRaw, len(var)), 2), round(chi2_contingency(contingtableRaw)[1], 4)
    return res

# Filling table
dfVcramerChisqPvalue = pd.DataFrame([vCramerChisqPvalue('year', dfCarac['year'], dfYearGravRaw), 
              vCramerChisqPvalue('mois', dfCarac['mois'], dfMonthGravRaw), 
              vCramerChisqPvalue('jour', dfCarac['jour'], dfMonthdayGravRaw), 
              vCramerChisqPvalue('weekday', dfCarac['weekday'], dfWeekdayGravRaw), 
              vCramerChisqPvalue('hour', dfCarac['hour'], dfHourGravRaw), 
              vCramerChisqPvalue('lum', dfCarac['lum'], dfLumGravRaw), 
              vCramerChisqPvalue('atm', dfCarac['atm'], dfAtmGravRaw), 
              vCramerChisqPvalue('col', dfCarac['col'], dfColGravRaw), 
              vCramerChisqPvalue('agg', dfCarac['agg'], dfAggGravRaw), 
              vCramerChisqPvalue('nbv', dfLieux['nbv'], dfNbvGravRaw), 
              vCramerChisqPvalue('vosp', dfLieux['vosp'], dfVospGravRaw), 
              vCramerChisqPvalue('prof', dfLieux['prof'], dfProfGravRaw), 
              vCramerChisqPvalue('plan', dfLieux['plan'], dfPlanGravRaw), 
              vCramerChisqPvalue('surf', dfLieux['surf'], dfSurfGravRaw), 
              vCramerChisqPvalue('infra', dfLieux['infra'], dfInfraGravRaw), 
              vCramerChisqPvalue('situ', dfLieux['situ'], dfSituGravRaw), 
              vCramerChisqPvalue('env1', dfLieux['env1'], dfEnv1GravRaw), 
              vCramerChisqPvalue('catu', dfUsagers['catu'], dfCatuGravRaw), 
              vCramerChisqPvalue('sexe', dfUsagers['sexe'], dfSexeGravRaw), 
              vCramerChisqPvalue('trajet', dfUsagers['trajet'], dfTrajetGravRaw)])
dfVcramerChisqPvalue = dfVcramerChisqPvalue.rename({0:'Variable', 1:'V Cramer', 2:'pvalue'}, axis=1)

# Display table
dfVcramerChisqPvalue.sort_values(by='V Cramer', ascending=False)
# dfVcramerChisqPvalue


# In[13]:


##### Vcramer between variables
### Initiating objects
varList1Carac = ['year', 'mois', 'jour', 'weekday', 'hour', 'lum', 'atm', 'col', 'agg', 'int']
varList2Carac = ['year', 'mois', 'jour', 'weekday', 'hour', 'lum', 'atm', 'col', 'agg', 'int']
varList1Lieux = ['catr', 'circ', 'nbv', 'vosp', 'prof', 'plan', 'surf', 'infra', 'situ', 'env1']
varList2Lieux = ['catr', 'circ', 'nbv', 'vosp', 'prof', 'plan', 'surf', 'infra', 'situ', 'env1']
varList1Usagers = ['place', 'catu', 'sexe', 'trajet', 'secu', 'locp', 'actp', 'etatp']
varList2Usagers = ['place', 'catu', 'sexe', 'trajet', 'secu', 'locp', 'actp', 'etatp']
varList1Vehicules = ['choc', 'manv', 'motor', 'senc', 'catv', 'obs', 'obsm']
varList2Vehicules = ['choc', 'manv', 'motor', 'senc', 'catv', 'obs', 'obsm']
resMatrixCarac = pd.DataFrame(np.zeros(shape=(len(varList1Carac), len(varList2Carac))), index=varList1Carac, columns=varList2Carac)
resMatrixLieux = pd.DataFrame(np.zeros(shape=(len(varList1Lieux), len(varList2Lieux))), index=varList1Lieux, columns=varList2Lieux)
resMatrixUsagers = pd.DataFrame(np.zeros(shape=(len(varList1Usagers), len(varList2Usagers))), index=varList1Usagers, columns=varList2Usagers)
resMatrixVehicules = pd.DataFrame(np.zeros(shape=(len(varList1Vehicules), len(varList2Vehicules))), index=varList1Vehicules, columns=varList2Vehicules)

### Filling dataframe (Carac)
for i in varList1Carac:
    for j in varList2Carac:
        tab = pd.crosstab(dfCarac[i], dfCarac[j])
        resMatrixCarac[j][i] = round(V_cramer(tab, tab.sum().sum()), 2)
### Filling dataframe (Lieux)
for i in varList1Lieux:
    for j in varList2Lieux:
        tab = pd.crosstab(dfLieux[i], dfLieux[j])
        resMatrixLieux[j][i] = round(V_cramer(tab, tab.sum().sum()), 2)
### Filling dataframe (Usagers)
for i in varList1Usagers:
    for j in varList2Usagers:
        tab = pd.crosstab(dfUsagers[i], dfUsagers[j])
        resMatrixUsagers[j][i] = round(V_cramer(tab, tab.sum().sum()), 2)
### Filling dataframe (Vehicules)
for i in varList1Vehicules:
    for j in varList2Vehicules:
        tab = pd.crosstab(dfVehicules[i], dfVehicules[j])
        resMatrixVehicules[j][i] = round(V_cramer(tab, tab.sum().sum()), 2)


# In[24]:


# Display VCramer dataframes - Unique datasets
fig, ax = plt.subplots(2, 2, figsize=(20, 15))
sns.heatmap(resMatrixCarac, ax=ax[0, 0])
ax[0, 0].set_title('Caracteristicts table')
sns.heatmap(resMatrixLieux, ax=ax[0, 1])
ax[0, 1].set_title('Places table')
sns.heatmap(resMatrixUsagers, ax=ax[1, 0])
ax[1, 0].set_title('Users table');
sns.heatmap(resMatrixVehicules, ax=ax[1, 1])
ax[1, 1].set_title('Vehicules table');

# Display raw VCramer dataframes
print(resMatrixCarac)
print(resMatrixLieux)
print(resMatrixUsagers)
print(resMatrixVehicules)


# In[5]:


##### Describing dfPool
dfPool.head(2)


# In[6]:


dfPool.apply(pd.Series.value_counts)


# In[5]:


dfPool[dfPool.eq(-1).any(1)]
# dfPool[dfPool.eq(-1).any(1)].iloc[:, 41: 50]


# In[18]:


dfPool.columns


# ##### DataFrame with NA prevalence and p-value against gravGrp_2_34

# In[9]:


a = dfPool.iloc[:,38]
print(len(a.unique()))
a.value_counts()


# In[6]:


# Nouveau dataFrame sans NA
dfPoolCCA = dfPool.dropna()

# Verification du nombre de NA
print('NA count:', (dfPoolCCA.isnull().sum() * 100 / len(dfPoolCCA)).sum())
print('Dim:', dfPoolCCA.shape)


# In[7]:


# Création du dataFrame qui permettra de faire la sélection de variables
dfDescVarExpl = pd.DataFrame({'propNA':dfPool.isnull().sum() * 100 / len(dfPool), 
                              'lenUnique':dfPool.apply(lambda x: len(x.unique()), axis=0),
                              'type':['nonInf', 'nonInf', 'cat', 'cat', 'cat', 'cat', 
                                      'cat', 'cat', 'cat', 'target', 'cat', 'bin', 
                                      'bin', 'cat', 'cat', 'cat', 'nonInf', 'cat', 
                                      'bin', 'cat', 'cat', 'cat', 'nonInf', 'nonInf', 
                                      'target', 'nonInf', 'cat', 'cat', 'cat', 'nonInf', 
                                      'cat', 'bin', 'bin', 'bin', 'target', 'target', 
                                      'bin', 'bin', 'bin', 'bin', 'target', 'target', 
                                      'num', 'num', 'num', 'num', 'num', 'num', 
                                      'num', 'num', 'num', 'num', 'bin', 'bin', 
                                      'bin', 'bin', 'num', 'bin', 'num', 'cat', 
                                      'cat', 'num']})


# In[12]:


# Création de la variable p-value
dfDescVarExpl['pvalue'] = np.nan


# In[13]:


### loop p-values over all variables
for i in dfDescVarExpl.index:
    varCateg = dfPoolCCA[i]
    if dfDescVarExpl.type.loc[i] in ['bin', 'cat']:
        table = pd.crosstab(dfPoolCCA['gravGrp_2_34'], varCateg)
        dfDescVarExpl.pvalue.loc[i] = chi2_contingency(table)[1:2][0]
    if dfDescVarExpl.type.loc[i] in ['num']:
        result = statsmodels.formula.api.ols(i + ' ~ gravGrp_2_34', data=dfPoolCCA).fit()
        dfDescVarExpl.pvalue.loc[i] = statsmodels.api.stats.anova_lm(result).loc['gravGrp_2_34'][4]


# In[14]:


dfDescVarExpl.sort_values('propNA', ascending=False).head(10)


# In[15]:


dfDescVarExpl.sort_values('lenUnique', ascending=False).head(10)


# In[ ]:


# Create a dataframe specific for ML dfPoolML where variables that can't be used won't be there
# Create a new dataframe
# Create xl file with quick definition of each variables that can be used for ML: what is it, categories, type (cat/num), corr
# Il reste des données -1 (e.g. int)


# In[92]:


dfDescVarExpl[dfDescVarExpl.type=='num'].index


# In[147]:


dfDescVarExpl.loc[['num_veh', 'obsGrp', 'col']]


# In[142]:


dfPool['com'].value_counts()


# In[8]:


### Adding missing variables
dfPool['etatpGrp_pieton_alone'] = np.where(dfPool.groupby('Num_Acc')['nb_etatpGrp_pieton_alone'].sum()>=1, 1, 0)
dfPool['locpGrp_pieton_1'] = np.where(dfPool.groupby('Num_Acc')['nb_locpGrp_pieton_1'].sum()>=1, 1, 0)
dfPool['locpGrp_pieton_3'] = np.where(dfPool.groupby('Num_Acc')['nb_locpGrp_pieton_3'].sum()>=1, 1, 0)
dfPool['locpGrp_pieton_6'] = np.where(dfPool.groupby('Num_Acc')['nb_locpGrp_pieton_6'].sum()>=1, 1, 0)


# In[18]:


##### Vcramer between variables - pooled dataset
### Initiating objects
# varList1Pool = ['prof', 'planGrp', 'surf', 'atm', 'situ', 
#                 'vospGrp', 
#                 'larroutGrp', 'lartpcGrp', 
#                 'env1', 'catv_EPD_exist', 'catv_PL_exist', 
#                 'obsGrp', 'trajet_coursesPromenade_conductor', 
#                 'sexe_male_conductor', 'sexe_female_conductor', 
#                 'int', 'intGrp', 'catv_train_exist', 'infra', 'agg', 'catr', 'hour', 'hourGrp', 'lum', 'com', 'dep', 'circ', 'nbvGrp', 
#                 'catv_2_roues_exist', 'nbVeh', 'catu_pieton', 'col', 'populationGrp', 
#                 'year', 'mois_label', 'jour', 
#                 'weekday', 'dateWeekend', 'dateFerieAndWeekend', 'dateFerie']

# Id: 'Num_Acc', 
# Outcome: 'gravGrp_23_4', 'gravGrp_2_34', 'nb_grav4_by_acc', 'nb_grav3_by_acc', 

# Removed (nonInformative): 'Unnamed: 0', 'date', 'dep', 'jour', 'grav_x', 'grav_y', 'year', 'mois', 'com', 
# Removed (tooManyNA): 'lartpcGrp', 'ageMeanNonConductors', 'larroutGrp', 'env1', 
# Removed (correlated): 'int', 'agg', 'catu_pieton', 'weekday', 'dateWeekend', 

# Numeric removed (correlated): 'nb_sexe_male_conductor', 'nb_sexe_female_conductor', 'nb_catu_pieton', 
#'hour', 'population_tot', 'nb_trajet_coursesPromenade_conductor', 
#'nb_etatpGrp_pieton_alone', 'nb_locpGrp_pieton_1', 'nb_locpGrp_pieton_3', 'nb_locpGrp_pieton_6', 

# Numeric kept: , 'choc_cote', 'ageMeanConductors', 'num_veh', 
# Categorical kept:
#                 'prof', 'planGrp', 'surf', 'atm', 
#                 'vospGrp', 
#                 'catv_EPD_exist', 'catv_PL_exist', 
#                 'trajet_coursesPromenade_conductor',
#                 'sexe_male_conductor', 'sexe_female_conductor', 
#                 'intGrp', 'catv_train_exist', 'infra', 'catr', 'hourGrp', 'lum', 'circ', 'nbvGrp', 
#                 'catv_2_roues_exist', 'col', 'obsGrp', 'situ', 'populationGrp', 
#                 'mois_label', 'dateFerieAndWeekend', 'dateFerie',
#                 'etatpGrp_pieton_alone', 'locpGrp_pieton_1', 'locpGrp_pieton_3', 'locpGrp_pieton_6'

varList1Pool = ['prof', 'planGrp', 'surf', 'atm', 
                'vospGrp', 
                'catv_EPD_exist', 'catv_PL_exist', 
                'trajet_coursesPromenade_conductor',
                'sexe_male_conductor', 'sexe_female_conductor', 
                'intGrp', 'catv_train_exist', 'infra', 'catr', 'hourGrp', 'lum', 'circ', 'nbvGrp', 
                'catv_2_roues_exist', 'col', 'obsGrp', 'situ', 'populationGrp', 
                'mois_label', 'dateFerieAndWeekend', 'dateFerie',
                'etatpGrp_pieton_alone', 'locpGrp_pieton_1', 'locpGrp_pieton_3', 'locpGrp_pieton_6'
                ]
varList2Pool = varList1Pool
resMatrixPool = pd.DataFrame(np.zeros(shape=(len(varList1Pool), len(varList2Pool))), index=varList1Pool, columns=varList2Pool)

### Filling dataframe (Pool)
for i in varList1Pool:
    for j in varList2Pool:
        tab = pd.crosstab(dfPool[i], dfPool[j])
        resMatrixPool[j][i] = round(V_cramer(tab, tab.sum().sum()), 2)


# In[19]:


# Display VCramer dataframes
fig, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(resMatrixPool, ax=ax);


# In[9]:


##### Uptdates before export
### Removing -1 values
dfPool = dfPool.replace(-1, np.nan)


# In[ ]:


### Adding missing variables
dfPool['etatpGrp_pieton_alone'] = np.where(dfPool.groupby('Num_Acc')['nb_etatpGrp_pieton_alone'].sum()>=1, 1, 0)
dfPool['locpGrp_pieton_1'] = np.where(dfPool.groupby('Num_Acc')['nb_locpGrp_pieton_1'].sum()>=1, 1, 0)
dfPool['locpGrp_pieton_3'] = np.where(dfPool.groupby('Num_Acc')['nb_locpGrp_pieton_3'].sum()>=1, 1, 0)
dfPool['locpGrp_pieton_6'] = np.where(dfPool.groupby('Num_Acc')['nb_locpGrp_pieton_6'].sum()>=1, 1, 0)


# In[10]:


### Modifying index
dfPool = dfPool.set_index('Num_Acc')


# In[11]:


##### DataFrame for ML
### Variables selection
dfPoolML = dfPool[[
                        # Variable à expliquer
                    'gravGrp_2_34', 
    
                        # Variables explicatives
                    'choc_cote', 'ageMeanConductors', 'num_veh', 
                    'prof', 'planGrp', 'surf', 'atm', 
                    'vospGrp', 
                    'catv_EPD_exist', 'catv_PL_exist', 
                    'trajet_coursesPromenade_conductor',
                    'sexe_male_conductor', 'sexe_female_conductor', 
                    'intGrp', 'catv_train_exist', 'infra', 'catr', 'hourGrp', 'lum', 'circ', 'nbvGrp', 
                    'catv_2_roues_exist', 'col', 'obsGrp', 'situ', 'populationGrp', 
                    'mois_label', 'dateFerieAndWeekend', 'dateFerie',
                    'etatpGrp_pieton_alone', 'locpGrp_pieton_1', 'locpGrp_pieton_3', 'locpGrp_pieton_6']]

### Removing NA values
dfPoolMLCCA = dfPoolML.dropna()


# In[45]:


##### Export dataframe
pathExport = 'D:\\jupyterDatasets\\'
dfPoolMLCCA.to_csv(pathExport+'20221031_table_dfPoolMLCCA.csv', index=False, sep=';')


# In[43]:


dfPoolML.shape


# In[44]:


dfPoolMLCCA.shape


# # --------------------------------------Pre-processing--------------------------------------
# ### Preprocessing of datasets for specific methods to be used

# In[72]:


##### Création de data-frames en fonction de problématiques différentes
### All variables for gravGrp2_34 (e.g. xgboost)
# removed: 'dep'
dfPoolML2_34 = dfPool[['gravGrp_2_34', 'prof', 'planGrp', 'surf', 'atm', 'situ', 
                            'vospGrp', 
                            'larroutGrp', 'lartpcGrp', 
                            'env1', 'catv_EPD_exist', 'catv_PL_exist', 
                            'obsGrp', 'trajet_coursesPromenade_conductor', 
                            'sexe_male_conductor', 'sexe_female_conductor', 
                            'int', 'intGrp', 'catv_train_exist', 'infra', 'agg', 'catr', 
                            'hour', 'hourGrp', 'lum', 'com', 'circ', 'nbvGrp', 
                            'catv_2_roues_exist', 'nbVeh', 'catu_pieton', 'col', 'populationGrp', 
                            'year', 'mois_label', 'jour', 
                            'weekday', 'dateWeekend', 'dateFerieAndWeekend', 'dateFerie']]
##### All variables without strong Cramer for gravGrp2_34 (e.g. regression)
### toupdate
# removed: 'dep', 'surf', 'lartpcGrp', 'catv_EPD_exist', 'vospGrp', 'int', 'catv_train_exist', 'sexe_female_conductor', 
# 'catr', 'hour', 'hourGrp', 'circ', 'catv_2_roues_exist', 'nbVeh', 'catu_pieton', 'populationGrp', 'larroutGrp', 'obsGrp', 
# 'dateWeekend', 'dateFerie', 
dfPoolML2_34noCorr = dfPool[['gravGrp_2_34', 'prof', 'planGrp', 'atm', 'situ', 
                            
                            
                            'env1', 'catv_PL_exist', 
                            'trajet_coursesPromenade_conductor', 
                            'sexe_male_conductor', 
                            'intGrp', 'infra', 'agg', 
                            'lum', 'com', 'nbvGrp', 
                            'col', 
                            'year', 'mois_label', 'jour', 
                            'weekday', 'dateFerieAndWeekend']]


# In[73]:


### Suppression des lignes comportant des NA pour les algorithmes nécessaires
dfPoolML2_34noCorrNoNA = dfPoolML2_34noCorr.dropna()


# In[63]:


### Verification of number of NA
max(dfPoolML2_34noCorrNoNA.isnull().sum() * 100 / len(dfPoolML2_34noCorrNoNA))


# # --------------------------------------Modelling--------------------------------------
# ### XGBoost
# - Immune to multi-collinearity
# - Works with NA values

# In[12]:


# Defining target and features
target_xgb = dfPoolMLCCA.gravGrp_2_34
features = dfPoolMLCCA.drop('gravGrp_2_34', axis=1)
features_matrix = pd.get_dummies(features, drop_first=True)


# In[14]:


### Features
features_matrix.columns


# In[13]:


# Verification of features length
print(len(set(features_matrix.columns)))
print(len(features_matrix.columns))


# In[15]:


# Checking if any duplicate feature in the matrix and if so which ones
duplicate_columns = features_matrix.columns[features_matrix.columns.duplicated()]
duplicate_columns


# In[16]:


### Splitting into train & test
X_train, X_test, y_train, y_test = model_selection.train_test_split(features_matrix, target_xgb, test_size=0.2, random_state=1)


# In[17]:


X_train.columns


# In[18]:


print(X_train.shape)
print(X_test.shape)


# In[19]:


train = xgb.DMatrix(data=X_train, label=y_train)
test = xgb.DMatrix(data=X_test, label=y_test)


# In[20]:


params = {'booster' : 'gbtree', 
          'learning_rate' : 1, 
          'objective' : 'binary:logistic'}
xgb1 = xgb.train(params=params, dtrain=train, num_boost_round=50, evals=[(train, 'train'), (test, 'eval')])


# In[21]:


types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']

for f in types:
    xgb.plot_importance(xgb1 ,max_num_features=15, importance_type=f, title='importance: '+f);


# In[71]:


xgb_preds_train


# In[63]:


# Train
preds_train = xgb1.predict(train)
xgb_preds_train = pd.Series(np.where(preds_train >= 0.5, 1, 0))
# Test
preds_test = xgb1.predict(test)
xgb_preds_test = pd.Series(np.where(preds_test >= 0.5, 1, 0))


# In[74]:


### From probabilities to binary
# Train
xgb_preds_train = []

for i in preds_train:
    if i>=0.5:
        xgb_preds_train.append(1)
    if i<0.5:
        xgb_preds_train.append(0)

# Test
xgb_preds_test = []
for i in preds_test:
    if i>=0.5:
        xgb_preds_test.append(1)
    if i<0.5:
        xgb_preds_test.append(0)


# In[75]:


# Train contingency table
pd.crosstab(y_train, xgb_preds_train, colnames=['xgb_pred_train'], normalize=True)


# In[77]:


# Test contingency table
pd.crosstab(y_test, xgb_preds_test, colnames=['xgb_pred_test'], normalize=True)


# In[82]:


# Performance criteria
print(classification_report(y_train, xgb_preds_train))
print(classification_report(y_test, xgb_preds_test))


# In[83]:


rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
print("RMSE train: %f" % (rmse_train))
print("RMSE test : %f" % (rmse_test))


# In[87]:


# AUC
fpr_xgb, tpr_xgb, seuils = roc_curve(y_test, xgb_preds_test, pos_label=1)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
roc_auc_xgb


# In[89]:


# ROC curve
plt.figure(figsize=(4, 4))
plt.plot(fpr_xgb, tpr_xgb, color='orange', lw=2, label=round(roc_auc_xgb, 2))
plt.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), 'b--', label='0.50')
plt.ylabel('Taux vrais positifs')
plt.xlabel('Taux faux positifs')
plt.title('Courbe ROC')
plt.legend(loc='lower right');


# ### Logistic regression
# - Sensitivte to multi-collinearity
# - Doesn't work with NA values
# - Only floats for predictions so a pd.get_dummies is required
# - Standardization helps to not mislead variables range with their weight

# In[90]:


# Defining target and features
target_lr = dfPoolMLCCA.gravGrp_2_34
features_lr = dfPoolMLCCA.drop('gravGrp_2_34', axis=1)
features_matrix_lr = pd.get_dummies(features_lr, drop_first=True)


# In[91]:


### Splitting into train & test
X_train, X_test, y_train, y_test = model_selection.train_test_split(features_matrix_lr, target_lr, test_size=0.2, random_state=1)


# In[92]:


# LR model
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred_train = lr.predict(X_train)
lr_pred_test = lr.predict(X_test)


# In[93]:


pd.crosstab(y_train, lr_pred_train>=0.5, normalize=True)


# In[94]:


pd.crosstab(y_test, lr_pred_test>=0.5, normalize=True)


# In[95]:


# Performance criteria
print(classification_report(y_train, lr_pred_train>=0.5))
print(classification_report(y_test, lr_pred_test>=0.5))


# In[96]:


coeffs = list(lr.coef_)
coeffs.insert(0, lr.intercept_)

feats = list(X_train.columns)
feats.insert(0, 'intercept')

pd.DataFrame({'valeur estimée': coeffs}, index=feats)


# In[97]:


print(lr.score(X_train, y_train))
print(model_selection.cross_val_score(lr, X_train, y_train).mean())


# In[98]:


# AUC
fpr, tpr, seuils = roc_curve(y_test, lr_pred_test, pos_label=1)
roc_auc = auc(fpr, tpr)
roc_auc


# In[99]:


# ROC curve
plt.figure(figsize=(4, 4))
plt.plot(fpr, tpr, color='orange', lw=2, label=round(roc_auc, 2))
plt.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), 'b--', label='0.50')
plt.ylabel('Taux vrais positifs')
plt.xlabel('Taux faux positifs')
plt.title('Courbe ROC')
plt.legend(loc='lower right');

