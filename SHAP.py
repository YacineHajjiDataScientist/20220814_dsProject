#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install shap


# In[1]:


### Import libraries
import pandas as pd

from sklearn.preprocessing import StandardScaler

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


# In[23]:


### Train/test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(feature_matrix, target, test_size=0.2, random_state=1)

### Scaling for Lasso
scaler = StandardScaler()
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)
train_scaled_df = pd.DataFrame(train_scaled, columns = X_train.columns)
test_scaled_df = pd.DataFrame(test_scaled, columns = X_test.columns)

### XGBoost datasets
train = xgb.DMatrix(data=X_train, label=y_train)
test = xgb.DMatrix(data=X_test, label=y_test)

print('Prevalence y train:', round(sum(y_train) / len(y_train), 4))
print('Prevalence y test:', round(sum(y_test) / len(y_test), 4))


# In[9]:


# Import models
xgb_clf = jb.load('C:\\Users\\Megaport\\Desktop\\jupyterNotebook\\grid_search\\optimal_xgb.joblib')
en_clf = jb.load('C:\\Users\\Megaport\\Desktop\\jupyterNotebook\\grid_search\\optimal_model_en.joblib')


# In[16]:


# SHAP en
explainer_en = shap.LinearExplainer(en_clf, train_scaled, link=shap.links.logit)
get_ipython().run_line_magic('time', 'shap_values_en = explainer_en.shap_values(test_scaled)')


# In[17]:


# SHAP xgb
explainer_xgb =  shap.TreeExplainer(xgb_clf)
get_ipython().run_line_magic('time', 'shap_values_xgb = explainer_xgb.shap_values(test)')


# In[18]:


shap_values_en[1]


# In[24]:


### SHAP plot en - Relation of variables
shap.summary_plot(shap_values_en, test_scaled_df, plot_type='dot')


# In[22]:


### SHAP plot xgb - Relation of variables
shap.summary_plot(shap_values_xgb, X_test, plot_type='dot')


# In[29]:


### SHAP plot xgb - Relation target with 2 variables
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

