#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install sage-importance


# In[ ]:


pip install shap


# In[ ]:


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


# # User case: how to calibrate it?

# ##### Case: Discussion au gouvernement
# 
# Le président vient d'entrer en fonction et il voudrait savoir si cela vaut le coup de dépenser de l'argent pour réduire les accidents graves sur la route ou s'il faut qu'il garde cet argent pour d'autres projets.  
# 
# <span style="background-color: #D67676">
# <b>Le ministre des transports affirme qu'il faut absolument dépenser plus car il voit une véritable opportunité de réduire les accidents graves, il sait d'ailleurs qu'il peut prédire à l'avance la majorité des accidents graves, mais au coût de mal prédire les accidents bénins.</b></span>
# 
# <span style="background-color: #76D680"><b>Le ministre de l'économie appuie le fait qu'il faut exclusivement se baser sur le profil des accidents graves pour lesquels nous avons une probabilité élevée de bien prédire afin que l'argent soit en grande majorité bien dépensée dans les infrastructures.</b></span>
# 
# <span style="background-color: #76C3D6"><b>L'expert analyste, lui, propose que l'enveloppe soit divisée en deux afin que le ministère des transports puisse bénéficier de dépenses sur son projet mais à condition qu'il prédise le mieux possible à la fois les accidents graves et ceux moins graves afin de ne pas créer trop d'infrastructures inutiles.</b></span>
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# The objective is to target most factors that could represent potential severe accidents so that the French state could spend their money in the right infrastructures.
# 
# For this purpore, we focus on the detection of True Positives, this means that we must let the model predict more people as positives. It will increase the recall of positives but also increase the number of negatives predicted as positives (decrease the precision of positives).
# 
# In order to increase the positive recall, we must use a lower cutoff than 0.5.

# ### Predictions with XGBoost

# In[39]:


### Import libraries
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

### Binarisation of predictions
# Train
xgb_preds_train = xgb_clf.predict(train)
xgb_preds_train_bin03 = np.where(xgb_preds_train >= 0.3, 1, 0)
xgb_preds_train_bin05 = np.where(xgb_preds_train >= 0.5, 1, 0)
xgb_preds_train_bin07 = np.where(xgb_preds_train >= 0.7, 1, 0)
# Test
xgb_preds_test = xgb_clf.predict(test)
xgb_preds_test_bin03 = np.where(xgb_preds_test >= 0.3, 1, 0)
xgb_preds_test_bin05 = np.where(xgb_preds_test >= 0.5, 1, 0)
xgb_preds_test_bin07 = np.where(xgb_preds_test >= 0.7, 1, 0)


# #### Training set

# In[18]:


### Train contingency table at 0.3 cutoff
pd.crosstab(y_train, xgb_preds_train_bin05, colnames=['xgb_pred_train'], normalize=True)


# In[19]:


### Train contingency table at 0.3 cutoff
pd.crosstab(y_train, xgb_preds_train_bin03, colnames=['xgb_pred_train'], normalize=True)


# We can see above that more patients are predicted as positive.

# In[20]:


### Performance criteria at 0.5 cutoff
print(classification_report(y_train.astype('int'), xgb_preds_train_bin05))


# In[21]:


### Performance criteria at 0.3 cutoff
print(classification_report(y_train.astype('int'), xgb_preds_train_bin03))


# As expected, we have increased positive recall while reducing negative precision.

# #### Test set

# In[22]:


### Test contingency table at 0.3 cutoff
pd.crosstab(y_test, xgb_preds_test_bin03, colnames=['xgb_pred_test'], normalize=True)


# In[23]:


### Performance criteria at 0.3 cutoff
print(classification_report(y_test.astype('int'), xgb_preds_test_bin03))


# In[40]:


### Performance criteria at 0.5 cutoff
print(classification_report(y_test.astype('int'), xgb_preds_test_bin05))


# In[41]:


### Performance criteria at 0.7 cutoff
print(classification_report(y_test.astype('int'), xgb_preds_test_bin07))


# In[54]:


### Function defining Youden index
def cutoff_youdens_j(fpr_xgb, tpr_xgb, seuils):
    j_scores = tpr_xgb - fpr_xgb
    j_ordered = sorted(zip(j_scores, seuils))
    return j_ordered[-1][1]

### Finding Youden cutoff
youden_cutoff = cutoff_youdens_j(fpr_xgb, tpr_xgb, seuils)

### Performance criteria at Youden cutoff
print(classification_report(y_test.astype('int'), np.where(xgb_preds_test >= youden_cutoff, 1, 0)))
print('Youden index=', round(youden_cutoff, 2))


# ##### ROC curve

# 

# In[67]:


### AUC
fpr_xgb, tpr_xgb, seuils = roc_curve(y_test.astype('int'), xgb_preds_test, pos_label=1)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
roc_auc_xgb

### ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr_xgb, tpr_xgb, color='black', lw=2, label=round(roc_auc_xgb, 2))
plt.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), 'b--', label='0.50')
plt.ylabel('Taux vrais positifs')
plt.xlabel('Taux faux positifs')
plt.title('Courbe ROC')
plt.legend(loc='lower right')
plt.annotate('High positive recall (0.3)', xy=(1-0.57, 0.82), xytext=(1-0.47, 0.81), arrowprops={'facecolor' : '#D67676'})
plt.annotate('Standard cutoff (0.5)', xy=(1-0.84, 0.60), xytext=(1-0.74, 0.59), arrowprops={'facecolor' : 'grey'})
plt.annotate('High negative recall (0.7)', xy=(1-0.94, 0.40), xytext=(1-0.84, 0.39), arrowprops={'facecolor' : '#76D680'})
plt.annotate('Youden index (0.43)', xy=(1-0.78, 0.67), xytext=(1-0.68, 0.66), arrowprops={'facecolor' : '#76C3D6'});

