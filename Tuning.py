#!/usr/bin/env python
# coding: utf-8

# # Tuning models optimization

# ##### Modules

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import dill
import datetime
import math
import itertools

from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
import statsmodels.api

from sklearn import model_selection
from sklearn.metrics import mean_squared_error, classification_report, roc_curve, auc, plot_confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
import xgboost as xgb

get_ipython().run_line_magic('matplotlib', 'inline')


# ##### Define workspace

# In[2]:


os.chdir('C:\\Users\\Megaport\\Desktop\\jupyterNotebook')
os.getcwd()


# ##### Import

# In[3]:


# dfPoolMLCCA = pd.read_pickle('D:\\jupyterDatasets\\20221031_table_dfPoolMLCCA.csv')
feature_matrix = pd.read_pickle('D:\\jupyterDatasets\\20221112_table_feature_matrix.csv')
target = pd.read_pickle('D:\\jupyterDatasets\\20221119_table_target.csv')

# print(dfPoolMLCCA.shape)
print(feature_matrix.shape)
print(target.shape)


# In[4]:


feature_matrix.columns


# ##### Splitting into train/test sets

# In[5]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(feature_matrix, target, test_size=0.2, random_state=1)


# In[6]:


train = xgb.DMatrix(data=X_train, label=y_train)
test = xgb.DMatrix(data=X_test, label=y_test)


# In[8]:


print('Prevalence y train:', round(sum(y_train) / len(y_train), 4))
print('Prevalence y test:', round(sum(y_test) / len(y_test), 4))


# ### Models

# ### 1- XGBoost
# **Les paramètres généraux**  
# - booster : Le type de booster utilisé (par défaut gbtree).  
# - nthread : Le nombre de coeurs à utiliser pour le calcul parallèle (par défaut tous les coeurs disponibles sont utilisés).  
# 
# **Les paramètres du booster (on se limitera ici au cas des arbres)**  
# - num_boost_round : Le nombre maximum d'itérations/d'arbres construits (vaut 100 par défaut).  
# - learning_rate : Contrôle le 'taux d’apprentissage'. A chaque étape de boosting, on introduit une constante dans la formule de mise à jour des modèles. Elle réduit le poids obtenu par rapport aux performances pour prévenir l'overfitting. Une valeur faible entraîne un modèle plus robuste au sur-apprentissage, mais un calcul et une convergence plus lents. Pensez à augmenter le nombre d'arbres lorsque learning_rate est faible (vaut 0.3 par défaut, et doit être compris entre 0 et 1).  
# - min_split_loss : Réduction de perte minimale requise pour effectuer une partition supplémentaire sur un nœud de l'arbre. Plus il est grand, plus l'algorithme sera conservateur.  
# - max_depth : Contrôle la profondeur des arbres. Plus les arbres sont profonds, plus le modèle est complèxe et plus grandes sont les chances d'overfitting (vaut 6 par défaut).  
# - min_child_weight : Critère d'arrêt relatif à la taille minimum du nombre d'observation dans un noeud (vaut 1 par défaut).  
# - subsample : Permet d'utiliser un sous-échantillon du dataset d'entraînement pour chaque arbre (vaut 1 par défaut, pas de sous-échantillonnage ; et doit être compris entre 0 et 1).  
# - colsample_bytree : Permet d'utiliser un certain nombre de variables parmi celles d'origine (vaut 1 par défaut, toutes les variables sont séléctionnées ; et doit être compris entre 0 et 1).  
# - reg_lambda et reg_alpha : contrôlent respectivement la régularisation L1 et L2 sur les poids (équivalent aux régression Ridge et Lasso).  
# 
# - Fonction objectif à utiliser
#     - binary:logistic pour la classification binaire. Retourne les probabilités pour chaque classe.
#     - reg:linear pour la régression.
#     - multi:softmax pour la classification multiple en utilisant la fonction softmax. Retourne les labels prédits.
#     - multi:softprob pour la classification multiple en utilisant la fonction softmax. Retourne les probabilités pour chaque classe.
# - eval_metric : Métrique d'évaluation (par défaut l'erreur de prédiction pour la classification, le RMSE pour la régression).
# Les métriques disponibles sont : mae (Mean Absolute Error), Logloss, AUC, RMSE, error mologloss, etc...
# - early_stopping_rounds : pour arrêter l'apprentissage quand l'évaluation sur l'ensemble de test ne s'améliore plus durant un certain nombre d'itérations. L'erreur de validation doit diminuer au moins tous les early_stopping_rounds pour continuer l'entraînement.

# ##### Initial model

# In[17]:


### Initiating parameters for basic XGBoost
# params = {'objective' : 'binary:logistic', 
#           'booster' : 'gbtree', 
#           'learning_rate' : 1,
#           'eval_metric' : 'AUC'}
params = [
    ('objective', 'binary:logistic'),
    ('max_depth', 6),
    ('eval_metric', 'auc'),
    ('early_stopping_rounds', 10)
]

### Basic XGBoost without tuning
clf_xgb = xgb.train(params=params, dtrain=train, 
                    num_boost_round=50, 
                    evals=[(train, 'train'), (test, 'eval')])


# In[47]:


xgb_pred_train = clf_xgb.predict(X_train)
xgb_pred_test = clf_xgb.predict(X_test)


# In[48]:


pd.crosstab(y_train, xgb_pred_train, colnames=['xgb_pred_train'], normalize=True)


# In[49]:


print(classification_report(y_train, xgb_pred_train))


# In[50]:


pd.crosstab(y_test, xgb_pred_test, colnames=['xgb_pred_test'], normalize=True)


# In[51]:


print(classification_report(y_test, xgb_pred_test))


# In[55]:


# AUC train
fpr_xgb_train, tpr_xgb_train, seuils = roc_curve(y_train, xgb_pred_train, pos_label=1)
roc_auc_xgb_train = auc(fpr_xgb_train, tpr_xgb_train)
# AUC test
fpr_xgb_test, tpr_xgb_test, seuils = roc_curve(y_test, xgb_pred_test, pos_label=1)
roc_auc_xgb_test = auc(fpr_xgb_test, tpr_xgb_test)

print(roc_auc_xgb_train)
print(roc_auc_xgb_test)


# In[ ]:


# ROC curve
plt.figure(figsize=(4, 4))
plt.plot(fpr_xgb_test, tpr_xgb_test, color='orange', lw=2, label=round(roc_auc_xgb_test, 2))
plt.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), 'b--', label='0.50')
plt.ylabel('Taux vrais positifs')
plt.xlabel('Taux faux positifs')
plt.title('Courbe ROC')
plt.legend(loc='lower right');


# ##### Tuning

# In[57]:


param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.05, 0.01],
    'gamma': [0, 0.25, 1.0],
    'reg_lambda': [0, 1.0, 10.0],
    'scale_pos_weight': [1, 3, 5]
}


# In[60]:


optimal_params = GridSearchCV(
    estimator=clf_xgb,
    param_grid=param_grid,
    scoring='roc_auc',
    n_jobs=10,
    cv=3)


# In[ ]:


optimal_params.fit(X_train, 
                   y_train,
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)])


# In[ ]:


print(optimal_params.best_params_)


# In[45]:


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

estimator = XGBClassifier(
    objective= 'binary:logistic',
    num_boost_round=50,
    seed=1
)

parameters = {
    'max_depth': [2, 3, 4]
}

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = -1,
    cv = 5,
    verbose=True
)

grid_search.fit(X_train, y_train)


# In[46]:


grid_search.best_estimator_


# In[51]:


grid_search.cv_results_

