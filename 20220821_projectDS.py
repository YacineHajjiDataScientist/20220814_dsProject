#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# To do
+ find a way to create hour variable
+ plot variables alone (count) and vs gravity
+ find a way to plot proportions by group
+ fill template


# In[74]:


time = pd.DataFrame(data={'a':['1200', '1845', '00:31']})
time


# In[80]:


[i.replace(":", "") for i in time['a']]


# In[85]:


dfCarac['hrmn'].replace(':', '')
# [i.replace(":", "") for i in dfCarac['hrmn']]


# # Session

# In[8]:


# install modules
pip install dill


# In[1]:


# import modules
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dill

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


##### Defining directory
os.chdir('C:\\Users\\Megaport\\20220814_projectDS')
os.chdir('C:\\Users\\Megaport\\Desktop\\jupyterNotebook')
os.getcwd()


# In[3]:


# import session
dill.load_session('notebook_env.db')


# In[59]:


# save session
dill.dump_session('notebook_env.db')


# # Import

# In[7]:


##### Import of tables into dataframes
dfLieux = pd.read_csv('20220817_table_lieux.csv', sep=',')
dfUsagers = pd.read_csv('20220814_table_usagers.csv', sep=',')
dfVehicules = pd.read_csv('20220817_table_vehicules.csv', sep=',')
dfCarac = pd.read_csv('20220817_table_caracteristiques.csv', sep=',')

##### Merging of tables into 1 pooled dataframe
# dfPool = pd.merge(dfLieux, dfUsagers, dfVehicules, dfCarac, on="Num_Acc")


# In[11]:


print('dfLieux dimensions:', dfLieux.shape)
print('dfUsagers dimensions:', dfUsagers.shape)
print('dfVehicules dimensions:', dfVehicules.shape)
print('dfCarac dimensions:', dfCarac.shape)
# print('dfPool dimensions:', dfPool.shape)


# # Data-management

# In[33]:


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
dfCarac['hour'] = dfCarac['hrmn']//100


# In[13]:


dfCarac['date'].value_counts().sort_index()


# In[48]:



dfCarac['hrmn'].value_counts()


# # Descriptive statistics
# ### Mapping of variables
# In this section, we describe each table

# In[12]:


dfCarac.head(3)


# In[58]:


print('Jours:', len(dfCarac.jour.value_counts()))
print('Mois:', len(dfCarac.mois.value_counts()))
print('An:', len(dfCarac.an.value_counts()))
print('hrmn:', len(dfCarac.hrmn.value_counts()))
print('lum:', len(dfCarac.lum.value_counts()))
print('atm:', len(dfCarac.atm.value_counts()))
print('col:', len(dfCarac.col.value_counts()))
print('agg:', len(dfCarac['agg'].value_counts()))
print(dfCarac.an.value_counts())
print(dfCarac.lum.value_counts())
print(dfCarac.atm.value_counts())
print(dfCarac.col.value_counts())
print(dfCarac.hrmn.value_counts())
print(dfCarac['agg'].value_counts())


# In[37]:


dfCarac.info()


# In[22]:


### Proportion of NA
dfCarac.isnull().sum() * 100 / len(dfCarac)


# In[13]:


dfLieux.head(3)


# In[38]:


dfLieux.info()


# In[23]:


### Proportion of NA
dfLieux.isnull().sum() * 100 / len(dfLieux)


# In[39]:


dfLieux[['nbv', 'vosp', 'prof', 'pr', 'pr1', 'plan', 'lartpc', 'larrout', 'surf', 'infra', 'situ', 'env1']].describe()


# In[6]:


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
print('hrmn:', len(dfLieux.larrout.value_counts()))
print(dfLieux.vosp.value_counts())
print(dfLieux.pr.value_counts())
print(dfLieux.pr1.value_counts())
print(dfLieux.lartpc.value_counts())
print(dfLieux.env1.value_counts())
print(dfLieux.larrout.value_counts())


# In[14]:


dfUsagers.head(3)


# In[28]:


dfVehicules.head(3)


# ### Graphs
# ##### Time-related graphs

# In[56]:


fig, ax = plt.subplots(2, 1)
sns.countplot(dfCarac['mois'], 
             palette=['#D7F1F5', '#D7F1F5', 
                   '#E3F5D7', '#E3F5D7', '#E3F5D7', 
                   '#FEE7B9', '#FEE7B9', '#FEE7B9', 
                   '#FEBEB9', '#FEBEB9', '#FEBEB9', 
                   '#D7F1F5'], ax=ax[0])
sns.countplot(x=dfCarac['mois'], hue=dfCarac['grav'], 
             palette=['#F45050','#F4B650','#C8C8C8'], ax=ax[1]);


# In[87]:


plt.figure(figsize=(10, 4))
sns.countplot(dfCarac['mois'], 
             palette=['#D7F1F5', '#D7F1F5', 
                   '#E3F5D7', '#E3F5D7', '#E3F5D7', 
                   '#FEE7B9', '#FEE7B9', '#FEE7B9', 
                   '#FEBEB9', '#FEBEB9', '#FEBEB9', 
                   '#D7F1F5'])
plt.hlines(y=len(dfCarac['mois'])/12, xmin=0, xmax=11, color='blue', alpha=0.4);
# On peut observer que les mois de juin, juillet, septembre et octobre semblent avoir le plus d'accidents
# On peut observer que le mois de février compte le moins d'accidents mais il comporte aussi 28 jours


# In[93]:


plt.figure(figsize=(10, 4))
sns.countplot(x=dfCarac['mois'], hue=dfCarac['grav'], 
             palette=['#F45050','#F4B650','#C8C8C8']);

# Faire un graphique où on affiche la proportion de gravité par mois
# idem par heures du jour (24h au total)


# In[31]:


sns.countplot(x=dfCarac['weekday'], 
             palette=['#A0A491', '#A0A491', '#A0A491', '#A0A491', '#A0A491', '#E17441', '#E17441'])
plt.hlines(y=len(dfCarac['weekday'])/7, xmin=0, xmax=6, color='blue', alpha=0.4);
plt.xticks(np.arange(7), ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
plt.title("Nombre d'accident par jour de la semaine");

# It seems that the friday is the accident day


# In[127]:


pd.crosstab(dfCarac['weekday'], dfCarac['grav'], normalize=0)


# In[142]:


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
             palette=['#F45050','#F4B650','#C8C8C8'])
ax.set_xticklabels(['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']);
# It seems that the gravity of accident is larger during the weekend compared to the week


# In[146]:


sns.histplot(dfCarac, x="weekday", hue="grav", stat="probability", multiple="fill", shrink=8);


# In[32]:


plt.figure(figsize=(10, 4))
sns.countplot(x=dfCarac['weekday'], hue=dfCarac['grav'], 
             palette=['#F45050','#F4B650','#C8C8C8']);


# In[ ]:


sns.countplot(x=dfCarac['hour'])
plt.xticks([0, 6, 12, 18, 24], ['Minuit', '6h', 'Midi', '18h', 'Minuit'])
plt.title("Nombre d'accident par heure de la journée");


# In[89]:


dfCarac['hrmn']


# In[5]:


dfLieux.grav.value_counts()

