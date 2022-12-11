#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install sage-importance


# In[ ]:


pip install shap


# In[7]:


get_ipython().system('pip install lightgbm')


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


# In[3]:


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


# In[4]:


# Import models
en_clf = jb.load('C:\\Users\\Megaport\\Desktop\\jupyterNotebook\\grid_search\\optimal_model_en.joblib')
xgb_clf = jb.load('C:\\Users\\Megaport\\Desktop\\jupyterNotebook\\grid_search\\optimal_xgb.joblib')
lgbm_clf = jb.load('C:\\Users\\Megaport\\Desktop\\jupyterNotebook\\grid_search\\LGBM.joblib')


# In[5]:


# SHAP en
explainer_en = shap.LinearExplainer(en_clf, train_scaled, link=shap.links.logit)
get_ipython().run_line_magic('time', 'shap_values_en = explainer_en.shap_values(test_scaled)')


# In[6]:


# SHAP xgb
explainer_xgb =  shap.TreeExplainer(xgb_clf)
get_ipython().run_line_magic('time', 'shap_values_xgb = explainer_xgb.shap_values(test)')


# In[7]:


# SHAP lgbm
explainer_lgbm =  shap.TreeExplainer(lgbm_clf)
get_ipython().run_line_magic('time', 'shap_values_lgbm = explainer_xgb.shap_values(test_scaled)')


# In[24]:


### SHAP plot en - Relation of variables
shap.summary_plot(shap_values_en, test_scaled_df, plot_type='dot')


# In[22]:


### SHAP plot xgb - Relation of variables
shap.summary_plot(shap_values_xgb, X_test, plot_type='dot')


# In[8]:


### SHAP plot lgbm - Relation of variables
shap.summary_plot(shap_values_lgbm, X_test, plot_type='dot')


# In[35]:


### SHAP plot xgb - Relation target with 2 variables
shap.dependence_plot("ageMeanConductors", shap_values_xgb, X_test, interaction_index= 'sexe_female_conductor_1')


# In[36]:


### SHAP plot xgb - Relation target with 2 variables
shap.dependence_plot("nbVeh", shap_values_xgb, X_test, interaction_index= 'sexe_female_conductor_1')


# In[40]:


### SHAP plot - Sample order by similarity
shap.initjs()
shap.force_plot(explainer_xgb.expected_value, shap_values_xgb[:1000], X_test[:1000])


# In[41]:


### SHAP plot - Impact of each modality for one person
shap.initjs()
shap.force_plot(explainer_xgb.expected_value, shap_values_xgb[0, :], X_test.iloc[0, :])


# ### Waterfall plots

# In[ ]:


from shap import Explainer, Explanation


# In[ ]:


explainer = Explainer(en_clf)
sv = explainer(train)


# In[ ]:


sv.base_values


# In[ ]:


### Waterfall plot for observation 1
explainer = Explainer(xgb_clf)
sv = explainer(X_train)

exp = Explanation(sv[:,:,6], sv.base_values[:,6], X_train, feature_names=None)
idx = 7 # datapoint to explain
waterfall_plot(exp[idx])


# In[ ]:


### Waterfall plot for observation 1
explainer = Explainer(xgb_clf)
sv = explainer(X_train)

exp = Explanation(sv[:,6], sv.base_values, X_train, feature_names=None)
idx = 7 # datapoint to explain
waterfall(exp[idx])


# In[ ]:


shap.plots.waterfall(exp[idx])


# In[ ]:


### Waterfall plot for observation 1
shap.plots.waterfall(shap_values_xgb[0])


# ### Overall performances (by feature importance)

# In[7]:


### Overall feature importance
types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']

for f in types:
    xgb.plot_importance(xgb_clf, max_num_features=15, importance_type=f, title='importance: '+f);


# ### SAGE

# In[ ]:


import sage

feature_names = X_test.columns

# Set up an imputer to handle missing features
imputer = sage.MarginalImputer(xgb_clf, X_test[:512])

# Set up an estimator
estimator = sage.PermutationEstimator(imputer, 'mse')

# Calculate SAGE values
sage_values = estimator(X_test, y_test)
sage_values.plot(feature_names)

