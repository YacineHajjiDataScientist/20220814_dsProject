#!/usr/bin/env python
# coding: utf-8

# # To do

# + fill template
# + plot distribution of numeric variables
# + compute V cramer and R² of each variable
# + plot ordered V cramer and R² of each variable

# # --------------------------------------Session--------------------------------------

# In[ ]:


# install modules
pip install dill


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

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


##### Defining directory
os.chdir('C:\\Users\\Megaport\\20220814_projectDS')
os.chdir('C:\\Users\\Megaport\\Desktop\\jupyterNotebook')
os.getcwd()


# In[3]:


# import session
dill.load_session('notebook_env.db')


# In[98]:


# save session
dill.dump_session('notebook_env.db')


# # --------------------------------------Import--------------------------------------

# In[7]:


##### Import of tables into dataframes
dfLieux = pd.read_csv('20220817_table_lieux.csv', sep=',')
dfUsagers = pd.read_csv('20220814_table_usagers.csv', sep=',')
dfVehicules = pd.read_csv('20220817_table_vehicules.csv', sep=',')
dfCarac = pd.read_csv('20220817_table_caracteristiques.csv', sep=',')

##### Merging of tables into 1 pooled dataframe
# dfPool = pd.merge(dfLieux, dfUsagers, dfVehicules, dfCarac, on="Num_Acc")


# In[8]:


print('dfLieux dimensions:', dfLieux.shape)
print('dfUsagers dimensions:', dfUsagers.shape)
print('dfVehicules dimensions:', dfVehicules.shape)
print('dfCarac dimensions:', dfCarac.shape)
# print('dfPool dimensions:', dfPool.shape)


# # --------------------------------------Data-management--------------------------------------

# In[9]:


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


# In[10]:


dfCarac['date'].value_counts().sort_index()


# In[11]:


dfCarac['year'].value_counts().sort_index()


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


# In[16]:


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


# In[59]:


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


# ##### -Table Vehicles-

# In[21]:


dfVehicules.head(3)


# ### Graphs

# In[22]:


# Gravity variable in Carac dataframe
pd.DataFrame({'prop':dfCarac.grav.value_counts(normalize=True),
              'count':dfCarac.grav.value_counts()})


# In[23]:


# Gravity variable in Lieux dataframe
pd.DataFrame({'prop':dfLieux.grav.value_counts(normalize=True),
              'count':dfLieux.grav.value_counts()})


# $\color{#0005FF}{\text{Both dataframes Carac and Lieux have the same amount of accidents, they also have the same accident gravity distribution}}$

# ##### -Table Carac-

# ### Year

# In[73]:


# Display plots
plt.figure(figsize=(10, 4))
sns.countplot(dfCarac['year'], palette=['#9D9D9D'])
plt.hlines(y=len(dfCarac['year'])/16, xmin=-0.5, xmax=15.5, color='blue', alpha=0.4)
plt.title("Nombre d'accident par année");
# It seems that the number of accident never stops decreasing year after year
# The observable large decreases seem to be during 2007-2008, 2011-2012 and 2019-2020
# The number of accident seemed to be stable between 2013 and 2019


# In[25]:


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
             palette=['#F45050','#F4B650','#C8C8C8']);
# It seems that the gravity is less important during 2018 to 2020


# In[47]:


# data-management
dfYearGrav = pd.crosstab(dfCarac['year'], dfCarac['grav'], normalize=0).sort_values(by=2, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(dfYearGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfYearGrav.apply(lambda x: x/dfCarac['grav'].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# Even though 2018 to 2020 have the largest proportions of accident gravity 3, they also have the lowest gravity 3 ones
# It seems that the state has focused on reducing the overall number of accident but not the gravity of accidents


# ### Months

# In[72]:


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


# In[28]:


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
             palette=['#F45050','#F4B650','#C8C8C8']);
# It seems that the gravity of accident is larger during the weekend compared to the week


# In[46]:


# data-management
dfMonthGrav = pd.crosstab(dfCarac['mois'], dfCarac['grav'], normalize=0).sort_values(by=2, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(dfMonthGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfMonthGrav.apply(lambda x: x/dfCarac['grav'].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# It seems that the largest proportion of accident gravity 2 & 3 happen during august and july


# ### Month day

# In[29]:


plt.figure(figsize=(20, 5))
sns.countplot(x=dfCarac['jour'], color='grey')
plt.hlines(y=len(dfCarac['jour'])/(365/12), xmin=-0.5, xmax=30.5, color='blue', alpha=0.4)
plt.title("Nombre d'accident par jour du mois");
# With no surprise, day 31 has twice as less accidents as other days of the month because it only occurs 1 months out of 2


# In[31]:


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
             palette=['#F45050','#F4B650','#C8C8C8']);
# Hard to read this figure but no trend seems to be seen


# In[45]:


# data-management
dfMonthdayGrav = pd.crosstab(dfCarac['jour'], dfCarac['grav'], normalize=0).sort_values(by=2, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 8))
sns.heatmap(dfMonthdayGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfMonthdayGrav.apply(lambda x: x/dfCarac['grav'].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# Geniunly no trend drawn


# ### Weekday

# In[30]:


sns.countplot(x=dfCarac['weekday'], 
             palette=['#A0A491', '#A0A491', '#A0A491', '#A0A491', '#A0A491', '#E17441', '#E17441'])
plt.hlines(y=len(dfCarac['weekday'])/7, xmin=-0.5, xmax=6.5, color='blue', alpha=0.4);
plt.xticks(np.arange(7), ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
plt.title("Nombre d'accident par jour de la semaine");
# It seems that the friday is the accident day


# In[31]:


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


# In[44]:


# data-management
dfWeekdayGrav = pd.crosstab(dfCarac['weekday'], dfCarac['grav'], normalize=0).sort_values(by=2, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(dfWeekdayGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfWeekdayGrav.apply(lambda x: x/dfCarac['grav'].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# Actually, the largest proportion of accident gravity 2 is during sunday then saturday


# ### Hour of the day

# In[33]:


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


# In[34]:


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
             palette=['#F45050','#F4B650','#C8C8C8']);
# Wow, it seems that the gravity of accidents is worst during the night (22pm-6am)
# More than 5% gravity 2 during the night against less than 4% during full day


# In[43]:


# data-management
dfHourGrav = pd.crosstab(dfCarac['hour'], dfCarac['grav'], normalize=0).sort_values(by=2, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 8))
sns.heatmap(dfHourGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfHourGrav.apply(lambda x: x/dfCarac['grav'].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# Proposition: creating a full night variable [0-6am] (yes/no)


# ### Lum

# In[74]:


sns.countplot(x=dfCarac['lum'][(dfCarac['lum']!=-1)], 
             palette=['#FF5D5D', '#5774B8', '#000000', '#000000', '#FDEC8B'])
plt.title("Nombre d'accident par condition luminaire")
plt.hlines(y=len(dfCarac['lum'][(dfCarac['lum']!=-1)])/5, xmin=-0.5, xmax=4.5, color='blue', alpha=0.4);
# It seems that most accident happen during the full day


# In[77]:


# Initiating dataframe grouped by hour
dfCaracGpByLum = (dfCarac[(dfCarac['lum']!=-1)].groupby(['lum'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plotx
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="lum", y="percentage", hue="grav", data=dfCaracGpByLum, 
             palette=['#F45050','#F4B650','#C8C8C8']);
# Wow, it seems that the gravity of accidents is worst during the night (22pm-6am)
# More than 5% gravity 2 during the night against less than 4% during full day


# In[42]:


# data-management
dfLumGrav = pd.crosstab(dfCarac['lum'][(dfCarac['lum']!=-1)], dfCarac['grav'][(dfCarac['lum']!=-1)], normalize=0).sort_values(by=2, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfLumGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfLumGrav.apply(lambda x: x/dfCarac['grav'][(dfCarac['lum']!=-1)].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# The night without public lightning seems to have a drastic increase of gravity 2 and 3 accidents rate (10% and 44%)!
# Then the two other cases where no much light is on have interesting gravity 2 increase accident rates


# ### Atm

# In[54]:


sns.countplot(x=dfCarac['atm'][(dfCarac['atm']!=-1)], 
             palette=['#8A8A8A', '#090F23', '#090F23', '#090F23', '#090F23', '#090F23', 
                     '#090F23', '#090F23', '#090F23'])
plt.title("Nombre d'accident par condition athmosphérique")
plt.hlines(y=len(dfCarac['atm'][(dfCarac['atm']!=-1)])/9, xmin=-0.5, xmax=8.5, color='blue', alpha=0.4);
# It seems that most accident happen with normal atmospheric conditions, then light rain


# In[84]:


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
             palette=['#F45050','#F4B650','#C8C8C8']);
# Wow, it seems that the gravity of accidents is worst during fog/smoke, strong wind/storm, dazzling weather and 'other'


# In[41]:


# data-management
dfAtmGrav = pd.crosstab(dfCarac['atm'][(dfCarac['atm']!=-1)], dfCarac['grav'][(dfCarac['atm']!=-1)], normalize=0).sort_values(by=2, ascending=False)

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


# In[90]:


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
             palette=['#F45050','#F4B650','#C8C8C8']);
# Les groupes 2, 3 et 4 sont très peu impactés en termes de gravité alors que les groupes 1, 6 et 7 semblent impactants


# In[40]:


# data-management
dfColGrav = pd.crosstab(dfCarac['col'][(dfCarac['col']!=-1)], dfCarac['grav'][(dfCarac['col']!=-1)], normalize=0).sort_values(by=2, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfColGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfColGrav.apply(lambda x: x/dfCarac['grav'][(dfCarac['col']!=-1)].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# La collision de type 1 est celle qui maximise les accidents de gravité 3 avec un fort taux de gravité 2
# Les collisions de type 6 et 7 sont celles qui maximisent les accidents de gravité 2


# In[57]:


# To update
for i in np.arange(1, 8):
    plt.plot(dfColGrav.div(dfColGrav.max(axis=0), axis=1).loc[i], label=i)
plt.legend();


# ### nbv

# In[78]:


dfLieux.nbv[(dfLieux.nbv<8) & (dfLieux.nbv>-1)].value_counts()


# In[83]:


sns.countplot(x=dfLieux.nbv[(dfLieux.nbv<7) & (dfLieux.nbv>-1)], color='grey')
plt.title("Nombre d'accident par nombre de voies")
plt.hlines(y=len(dfLieux['nbv'][(dfLieux.nbv<7) & (dfLieux.nbv>-1)])/7, xmin=-0.5, xmax=6.5, color='blue', alpha=0.4);
# Many accidents when there are 2 route tracks


# In[88]:


# Initiating dataframe grouped by hour
dfCaracGpByNbv = (dfLieux[(dfLieux.nbv<7) & (dfLieux.nbv>-1)].groupby(['nbv'])['grav']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('grav'))

# Display plotx
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="nbv", y="percentage", hue="grav", data=dfCaracGpByNbv, 
             palette=['#F45050','#F4B650','#C8C8C8']);
# Les groupes 0 et 2 semblent avoir un taux élevé d'accidents gravité 2 et 3


# In[50]:


# data-management
dfNbvGrav = pd.crosstab(dfLieux.nbv[(dfLieux.nbv<7) & (dfLieux.nbv>-1)], dfCarac['grav'][(dfLieux.nbv<7) & (dfLieux.nbv>-1)], normalize=0).sort_values(by=2, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfNbvGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfNbvGrav.apply(lambda x: x/dfLieux['grav'][(dfLieux.nbv<7) & (dfLieux.nbv>-1)].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# Les groupes 0 et 2 semblent avoir un taux élevé d'accidents gravité 2 et 3


# ### vosp

# In[60]:


dfLieux.vosp[(dfLieux.vosp>-1)].value_counts()


# In[56]:


sns.countplot(x=dfLieux.vosp[(dfLieux.vosp>-1)], color='grey')
plt.title("Nombre d'accident par présence de voie")
plt.hlines(y=len(dfLieux['vosp'][(dfLieux.vosp>-1)])/4, xmin=-0.5, xmax=3.5, color='blue', alpha=0.4);
# Many accidents when there are no additional reserved track


# In[103]:


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
             palette=['#F45050','#F4B650','#C8C8C8']);
# XXX


# In[111]:


# data-management
dfVospGrav = pd.crosstab(dfLieux['vosp'][(dfLieux['vosp']!=-1)], dfLieux['grav'][(dfLieux['vosp']!=-1)], normalize=0).sort_values(by=2, ascending=False)

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


# In[104]:


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
             palette=['#F45050','#F4B650','#C8C8C8']);
# XXX


# In[112]:


# data-management
dfProfGrav = pd.crosstab(dfLieux['prof'][(dfLieux['prof']!=-1)], dfLieux['grav'][(dfLieux['prof']!=-1)], normalize=0).sort_values(by=2, ascending=False)

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


# In[105]:


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
             palette=['#F45050','#F4B650','#C8C8C8']);
# XXX


# In[114]:


# data-management
dfPlanGrav = pd.crosstab(dfLieux['plan'][(dfLieux['plan']!=-1)], dfLieux['grav'][(dfLieux['plan']!=-1)], normalize=0).sort_values(by=2, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfPlanGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfPlanGrav.apply(lambda x: x/dfLieux['grav'][(dfLieux['plan']!=-1)].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# XXX


# ### pr

# In[70]:


# cat
dfLieux.pr


# In[132]:


print(type(dfLieux.pr1))


# ### pr1

# In[152]:


#num
# plt.hist(dfLieux.pr1.dropna(), bins=5, rwidth=0.5, orientation='horizontal')
# Boxplots by gravity level
# sns.catplot(y='pr1', x='grav', data=dfLieux, kind='box')
# sns.catplot(y='pr1', x='grav', data=dfLieux, kind='box')
# plt.yscale('log');


# In[ ]:





# In[ ]:





# ### lartpc

# In[141]:


#num
dfLieux.lartpc
sns.kdeplot(dfLieux['lartpc'], shade=True)


# In[ ]:





# In[150]:


# dfLieux['larrout'].astype('numerical')


# ### larrout

# In[151]:


#num
# dfLieux.larrout
# sns.kdeplot(dfLieux['larrout'], shade=True)


# In[ ]:





# In[ ]:





# ### surf

# In[86]:


#cat
dfLieux.surf[(dfLieux.surf>-1)].value_counts()


# In[91]:


sns.countplot(x=dfLieux.surf[(dfLieux.surf>-1)], color='grey')
plt.title("Nombre d'accident par surface de la route (météo)")
plt.hlines(y=len(dfLieux['surf'][(dfLieux.vosp>-1)])/10, xmin=-0.5, xmax=9.5, color='blue', alpha=0.4);
# Many accidents when there is normal or wet meteo


# In[106]:


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
             palette=['#F45050','#F4B650','#C8C8C8']);
# XXX


# In[115]:


# data-management
dfSurfGrav = pd.crosstab(dfLieux['surf'][(dfLieux['surf']!=-1)], dfLieux['grav'][(dfLieux['surf']!=-1)], normalize=0).sort_values(by=2, ascending=False)

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


# In[107]:


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
             palette=['#F45050','#F4B650','#C8C8C8']);
# XXX


# In[116]:


# data-management
dfInfraGrav = pd.crosstab(dfLieux['infra'][(dfLieux['infra']!=-1)], dfLieux['grav'][(dfLieux['infra']!=-1)], normalize=0).sort_values(by=2, ascending=False)

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


# In[108]:


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
             palette=['#F45050','#F4B650','#C8C8C8']);
# XXX


# In[117]:


# data-management
dfSituGrav = pd.crosstab(dfLieux['situ'][(dfLieux['situ']!=-1)], dfLieux['grav'][(dfLieux['situ']!=-1)], normalize=0).sort_values(by=2, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfSituGrav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfSituGrav.apply(lambda x: x/dfLieux['grav'][(dfLieux['situ']!=-1)].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# XXX


# ### env1

# In[79]:


#cat
dfLieux.env1.value_counts()


# In[101]:


sns.countplot(x=dfLieux.env1, color='grey')
plt.title("Nombre d'accident par présence d'école à proximité")
plt.hlines(y=len(dfLieux['env1'])/3, xmin=-0.5, xmax=2.5, color='blue', alpha=0.4);
# Many accidents happen when there are no school near


# In[109]:


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
             palette=['#F45050','#F4B650','#C8C8C8']);
# XXX


# In[149]:


# data-management
dfEnv1Grav = pd.crosstab(dfLieux['env1'], dfLieux['grav'], normalize=0).sort_values(by=2, ascending=False)

# Display plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(dfEnv1Grav, annot=True, cmap='cubehelix', ax=ax[0])
sns.heatmap(dfEnv1Grav.apply(lambda x: x/dfLieux['grav'].value_counts(normalize=True), axis=1), annot=True, cmap='magma_r', ax=ax[1]);
fig.show()
# XXX


# ### XXX

# In[ ]:





# In[ ]:





# In[ ]:





# ### XXX

# In[ ]:





# In[ ]:





# In[ ]:





# ### XXX

# In[ ]:





# In[ ]:





# In[ ]:




