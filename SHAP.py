#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install shap


# In[1]:


### Import libraries
import pandas as pd

from sklearn import model_selection
import xgboost as xgb

import joblib as jb

import shap


# In[2]:


### Import data
feature_matrix = pd.read_pickle('D:\\jupyterDatasets\\20221112_table_feature_matrix.csv')
target = pd.read_pickle('D:\\jupyterDatasets\\20221119_table_target.csv')

print(feature_matrix.shape)
print(target.shape)


# In[3]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(feature_matrix, target, test_size=0.2, random_state=1)
train = xgb.DMatrix(data=X_train, label=y_train)
test = xgb.DMatrix(data=X_test, label=y_test)

print('Prevalence y train:', round(sum(y_train) / len(y_train), 4))
print('Prevalence y test:', round(sum(y_test) / len(y_test), 4))


# In[4]:


# Import model
xgb_clf = jb.load('C:\\Users\\Megaport\\Desktop\\jupyterNotebook\\grid_search\\optimal_xgb.joblib')


# In[5]:


# SHAP
explainer =  shap.TreeExplainer(xgb_clf)
get_ipython().run_line_magic('time', 'shap_values = explainer.shap_values(test)')


# In[6]:


shap_values[1]


# In[27]:


### SHAP plot - Relation of variables
shap.summary_plot(shap_values, X_test, plot_type='dot')


# In[29]:


### SHAP plot - Relation target with 2 variables
shap.dependence_plot("ageMeanConductors", shap_values, X_test, interaction_index= 'sexe_female_conductor_1')


# In[13]:


shap.force_plot(explainer.expected_value, shap_values[:1000], X_test[:1000])


# In[8]:


### SHAP plot - Impact of each modality for one person
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :])


# In[7]:


types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']

for f in types:
    xgb.plot_importance(xgb_clf, max_num_features=15, importance_type=f, title='importance: '+f);

