#!/usr/bin/env python
# coding: utf-8

# ### Le code est découpé en plusieurs parties
# 
# - **Import**: Import des modules et des données pour la modélisation. Définition workspace, découpage des données entre train/test sets, standardization données et transformation de la feature matrix pour l'utilisation du XGBoost  
# - **Modelling**:   
# 1- Modélisation sans tunning de l'Elastic Net/Lasso, modélisation avec tunning (l1 & alpha parameters) avec un gridSearch, garder le modèle optimisant l'auc test et export de ce dernier  
# 2- Modélisation sans tunning spécifique du XGBoost et affichage des résultats test à battre, modélisation avec tunning en plusieurs étapes:   
#     * a) jouer sur max_depth, gamma et reg_lambda  
#     * b) augmenter max_depth à gamma et reg_lambda fixé  
#     * c) max_depth plus grand et plus bas, gamma, min_child_weight et n_estimators  
#     * d) max_depth faible, gamma, min_child_weight et n_estimators plus grand  
#     * e) max_depth, lambda, alpha et n_estimators  
#     * f) max_depth, lambda plus grand, alpha plus grand et n_estimators fixé  
#     * g) max_depth, lambda fixé, alpha plus grand et n_estimators plus grand  
#     * h) learning_rate qui baisse sur le modèle qui a eu les performances les plus intéressantes, dernière étape  
#     * Puis export du modèle optimal sélectionné  
# Il y a 2 parties exploration dans le script: tentatives d'entrainement de différents modèles linéaires et affichage de leurs coefficients (basic/ridge/lasso/ElasticNet), et à la fin l'affichage d'un arbre décisionnel du XGBoost

# # Tuning models optimization

# # ------ Import ------

# ##### Modules

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

import dill
import datetime
import math
import itertools

import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
import statsmodels.api

from sklearn import model_selection
from sklearn.metrics import mean_squared_error, classification_report, roc_curve, auc, plot_confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, KFold

from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, lasso_path, LassoCV, ElasticNetCV, ElasticNet
from sklearn.feature_selection import f_regression, SelectKBest, SelectFromModel
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
import xgboost as xgb
from xgboost import XGBClassifier

import joblib
from joblib import dump, load

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


# Checking all features remaining
feature_matrix.columns


# ##### Splitting into train/test sets

# In[4]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(feature_matrix, target, test_size=0.2, random_state=1)


# In[5]:


### Scaling for Lasso
scaler = preprocessing.StandardScaler()
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)

### XGBoost datasets
train = xgb.DMatrix(data=X_train, label=y_train)
test = xgb.DMatrix(data=X_test, label=y_test)


# In[7]:


print('Prevalence y train:', round(sum(y_train) / len(y_train), 4))
print('Prevalence y test:', round(sum(y_test) / len(y_test), 4))


# # ------ Modelling ------

# ### 1- Lasso/ElasticNet
# *Les paramètres généraux*
# - alpha: float, default=1.0
# Constant that multiplies the penalty terms. Defaults to 1.0. See the notes for the exact mathematical meaning of this parameter. alpha = 0 is equivalent to an ordinary least square, solved by the LinearRegression object. For numerical reasons, using alpha = 0 with the Lasso object is not advised. Given this, you should use the LinearRegression object.
# - l1_ratio: float, default=0.5
# The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

# <span style="color:red">Exploration of coefficients for different linear models---</span>

# In[10]:


# Launching basic linear, Ridge, Lasso regressions
linreg = LinearRegression()
ridgeregcv = RidgeCV(alphas=(0.001, 0.01, 0.1, 0.25, 0.5, 0.9), cv=10)
lassoregcv = LassoCV(alphas=(0.001, 0.01, 0.1, 0.25, 0.5, 0.9), cv=10)
lassoreg = Lasso(alpha=0.25)
linreg.fit(train_scaled, y_train)
ridgeregcv.fit(train_scaled, y_train)
lassoregcv.fit(train_scaled, y_train)
lassoreg.fit(train_scaled, y_train)


# In[13]:


# Comparing basic linear, Ridge, Lasso regressions coefficients
feats = list(X_train.columns)
feats.insert(0, 'intercept')
coeffs1 = list(linreg.coef_)
coeffs1.insert(0, linreg.intercept_)
coeffs2 = list(ridgeregcv.coef_)
coeffs2.insert(0, ridgeregcv.intercept_)
coeffs3 = list(lassoregcv.coef_)
coeffs3.insert(0, lassoregcv.intercept_)
coeffs4 = list(lassoreg.coef_)
coeffs4.insert(0, lassoreg.intercept_)
pd.DataFrame({'coeflinreg': coeffs1, 
             'coefridgereg': coeffs2, 
             'coeflassoreg': coeffs3, 
             'coeflasso0.25reg': coeffs4}, index=feats).transpose()


# In[19]:


# Affichage graphique des features gardées par le modèle en fonction de la pénalisation
alpha_path, coefs_lasso, _ = lasso_path(train_scaled, y_train, alphas=(0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1));

plt.figure(figsize=(20, 10))
for i in range(coefs_lasso.shape[0]):
    plt.plot(alpha_path, coefs_lasso[i, :], '-', label=X_train.columns[i])
plt.legend();


# In[21]:


plt.figure(figsize=(20, 30))
plt.barh(y=X_train.columns, width=lassoregcv.coef_);


# In[22]:


# Fitting an Elastic Net through a Cross-Validation
model_en = ElasticNetCV(cv=8, 
                        l1_ratio=(0.1, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99), 
                        alphas=(0.001, 0.01, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0))
model_en.fit(train_scaled, y_train)


# <span style="color:red">---end of exploration</span>

# ##### Tunning the Elastic Net (modifying alpha an l1 penality values)

# In[7]:


##### Initiating basic ElasticNet()
eNet = ElasticNet()

parameters = {
#     'max_iter': [1, 5, 10],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'l1_ratio': np.arange(0.1, 1.0, 0.1)
}

kf = KFold(n_splits=3, shuffle=True, random_state=1)

### GridSearchCV
grid_search_eNet = GridSearchCV(
    estimator=eNet,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 1,
    cv = kf,
    verbose=3,
    return_train_score=True
)

grid_search_eNet.fit(train_scaled, y_train)

print(grid_search_eNet.best_params_)


# In[14]:


##### Launching optimal ElasticNet
params = [
    ('alpha', 0.0001),
    ('l1_ratio', 0.1)
]

en_clf = ElasticNet(alpha=0.0001, 
                    l1_ratio=0.1)

en_clf.fit(train_scaled, y_train)


# In[15]:


# Export model
dump(en_clf, 'optimal_model_en.joblib')


# ### 2- XGBoost
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

# In[175]:


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


# In[176]:


# Getting probability predictions of XGBoost
xgb_pred_train = clf_xgb.predict(train)
xgb_pred_test = clf_xgb.predict(test)


# In[177]:


### Export of prediction of initial XGBoost
# pd.DataFrame(xgb_pred_train).to_pickle('D:\\jupyterDatasets\\20221121_xgb_pred_train.csv')
# pd.DataFrame(xgb_pred_test).to_pickle('D:\\jupyterDatasets\\20221121_xgb_pred_test.csv')


# In[20]:


# Transforming probability predictions into binary outcome
xgb_pred_train_bin = np.where(xgb_pred_train>=0.5, 1, 0)
xgb_pred_test_bin = np.where(xgb_pred_test>=0.5, 1, 0)


# In[21]:


# Contingency table of training observations against predictions
pd.crosstab(y_train, xgb_pred_train_bin, colnames=['xgb_pred_train'], normalize=True)


# In[22]:


# Basic training performances without tuning
print(classification_report(y_train, xgb_pred_train_bin))


# In[23]:


# Contingency table of test observations against predictions
pd.crosstab(y_test, xgb_pred_test_bin, colnames=['xgb_pred_test'], normalize=True)


# In[24]:


# Basic test performances without tuning
print(classification_report(y_test, xgb_pred_test_bin))


# In[25]:


# AUC train
fpr_xgb_train, tpr_xgb_train, seuils = roc_curve(y_train, xgb_pred_train, pos_label=1)
roc_auc_xgb_train = auc(fpr_xgb_train, tpr_xgb_train)
# AUC test
fpr_xgb_test, tpr_xgb_test, seuils = roc_curve(y_test, xgb_pred_test, pos_label=1)
roc_auc_xgb_test = auc(fpr_xgb_test, tpr_xgb_test)

print(roc_auc_xgb_train)
print(roc_auc_xgb_test)


# In[26]:


# ROC curve
plt.figure(figsize=(4, 4))
plt.plot(fpr_xgb_test, tpr_xgb_test, color='orange', lw=2, label=round(roc_auc_xgb_test, 2))
plt.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), 'b--', label='0.50')
plt.ylabel('Taux vrais positifs')
plt.xlabel('Taux faux positifs')
plt.title('Courbe ROC')
plt.legend(loc='lower right');

#-> AUC à battre: 0.79


# ##### Tuning

# ##### 1ère tentative (max_depth, gamma, reg_lambda)

# In[109]:


##### Initiating basic XGBoost
estimator = XGBClassifier(
    objective= 'binary:logistic',
    num_boost_round=50,
    seed=1
)

parameters = {
    'max_depth': [4, 6],
    'gamma': [0, 0.25, 1],
    'reg_lambda': [0, 1.0, 10.0],
}

#     'learning_rate': [0.1, 0.05, 0.03, 0.01]
        
kf = KFold(n_splits=5, shuffle=True, random_state=1)

### GridSearchCV
grid_search_1 = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = -1,
    cv = kf,
    verbose=1,
    return_train_score=True
)

grid_search_1.fit(X_train, y_train)

print(grid_search_1.best_params_)
### Best parameters: {'gamma': 0.25, 'max_depth': 6, 'reg_lambda': 10.0}
### Decision: we keep gamma=0.25, we go beyond max_depth=6 and beyond reg_lambda=10


# In[108]:


dfGridSearchCV_1 = pd.DataFrame(grid_search_1.cv_results_)[['params', 'rank_test_score', 'mean_test_score', 'mean_train_score', 'std_test_score', 'std_train_score']]
dfGridSearchCV_1.sort_values('rank_test_score')
### Max_depth=6 always was better in test cases
### 3 best models had reg_lambda=10.0
### We can still question the gamma value


# In[133]:


##### Plot GridSearchCV1 train vs test
results = ['mean_test_score', 'mean_train_score', 'std_test_score', 'std_train_score']

train_scores = grid_search_1.cv_results_['mean_train_score']
test_scores = grid_search_1.cv_results_['mean_test_score']

plt.plot(train_scores, label='train', color='#8940B8')
plt.plot(test_scores, label='test', color='#10CB8A')
plt.xticks(range(18))
plt.ylim((0.75, 0.85))
plt.legend(loc='best')
plt.show()

#-> Conclusion: Pas de valeur ajoutée au tunning


# In[131]:


dfGridSearchCV_1.params


# ##### 2ème tentative (augmenter max_depth, gamma fixé, reg_lambda fixé)

# In[142]:


##### GridSearchCV 2nd step
parameters = {
    'max_depth': [4, 6, 8, 10],
    'gamma': [0.25],
    'reg_lambda': [10.0]
}

kf = KFold(n_splits=5, shuffle=True, random_state=1)

### GridSearchCV
grid_search_2 = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = -1,
    cv = kf,
    verbose=1,
    return_train_score=True
)

grid_search_2.fit(X_train, y_train)

print(grid_search_2.best_params_)
### Best parameters: {'gamma': 0.25, 'max_depth': 6, 'reg_lambda': 10.0}
### Decision: we keep gamma=0.25, we go beyond max_depth=6 and beyond reg_lambda=10


# In[143]:


dfGridSearchCV_2 = pd.DataFrame(grid_search_2.cv_results_)[['params', 'rank_test_score', 'mean_test_score', 'mean_train_score', 'std_test_score', 'std_train_score']]
dfGridSearchCV_2.sort_values('rank_test_score')
### We need either 4/6 max depth or regularization to avoid overfitting


# In[153]:


##### Plot GridSearchCV2 train vs test
results = ['mean_test_score', 'mean_train_score', 'std_test_score', 'std_train_score']

train_scores = grid_search_2.cv_results_['mean_train_score']
test_scores = grid_search_2.cv_results_['mean_test_score']

plt.plot(train_scores, label='train', color='#8940B8')
plt.plot(test_scores, label='test', color='#10CB8A')
plt.xticks([0, 1, 2, 3], [4, 6, 8, 10])
plt.xlabel('max_depth')
plt.ylim((0.75, 0.85))
plt.legend(loc='best')
plt.show()

#-> Conclusion: aucune valeur ajoutée du tuning


# In[155]:


dfGridSearchCV_2.params


# In[168]:


##### Launching optimal XGBoost
params = [
    ('objective', 'binary:logistic'),
    ('max_depth', 6),
    ('gamma', 0.25),
    ('reg_lambda', 10.0),
    ('eval_metric', 'auc'),
    ('early_stopping_rounds', 10)
]

optimal_xgb = xgb.train(params=params, dtrain=train, 
                    num_boost_round=100, 
                    evals=[(train, 'train'), (test, 'eval')])


# In[172]:


### Prediction values of optimal XGBoost
xgb_opti_pred_train = optimal_xgb.predict(train)
xgb_opti_pred_test = optimal_xgb.predict(test)


# In[174]:


### Export of prediction of optimal XGBoost
# pd.DataFrame(xgb_opti_pred_train).to_pickle('D:\\jupyterDatasets\\20221121_xgb_opti_pred_train.csv')
# pd.DataFrame(xgb_opti_pred_test).to_pickle('D:\\jupyterDatasets\\20221121_xgb_opti_pred_test.csv')


# ##### 3ème tentative (max_depth plus grand et plus bas, gamma, min_child_weight et n_estimators)

# In[24]:


##### Initiating basic XGBoost
estimator = XGBClassifier(
    objective= 'binary:logistic',
    seed=1
)

parameters = {
    'max_depth': [3, 4, 6, 8, 10, 12],
    'gamma': [0, 0.25, 1],
    'min_child_weight': [0, 0.25, 1],
    'n_estimators': [50, 100, 150]
}

        
kf = KFold(n_splits=5, shuffle=True, random_state=1)

### GridSearchCV
grid_search_3 = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 1,
    cv = kf,
    verbose=3,
    return_train_score=True
)

grid_search_3.fit(X_train, y_train)

print(grid_search_3.best_params_)
### Best parameters: {'gamma': 0, 'max_depth': 6, 'min_child_weight': 0, 'n_estimators': 150}
### Decision: 


# In[28]:


joblib.dump(grid_search_3, 'grid_search_3.pkl')


# In[32]:


##### Plot GridSearchCV3 train vs test
results = ['mean_test_score', 'mean_train_score', 'std_test_score', 'std_train_score']

train_scores = grid_search_3.cv_results_['mean_train_score']
test_scores = grid_search_3.cv_results_['mean_test_score']

plt.plot(train_scores, label='train', color='#8940B8')
plt.plot(test_scores, label='test', color='#10CB8A')
plt.ylim((0.75, 0.95))
plt.legend(loc='best')
plt.show()

#-> Conclusion: pas de valeur ajoutée au tunning


# ##### 4ème tentative (max_depth faible, gamma, min_child_weight et n_estimators plus grand)

# In[36]:


##### Initiating basic XGBoost
estimator = XGBClassifier(
    objective= 'binary:logistic',
    seed=1
)

parameters = {
    'max_depth': [3, 4],
    'gamma': [0.25, 1],
    'min_child_weight': [0.25, 1],
    'n_estimators': [250, 350]
}

        
kf = KFold(n_splits=3, shuffle=True, random_state=1)

### GridSearchCV
grid_search_4 = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 1,
    cv = kf,
    verbose=3,
    return_train_score=True
)

grid_search_4.fit(X_train, y_train)

print(grid_search_4.best_params_)
### Best parameters: {'gamma': 0.25, 'max_depth': 4, 'min_child_weight': 0.25, 'n_estimators': 350}
### Decision: 


# In[37]:


joblib.dump(grid_search_4, 'grid_search_4.pkl')

#-> Conclusion: pas de valeur ajoutée au tunning


# ##### 5ème tentative (max_depth, lambda, alpha et n_estimators)

# In[8]:


##### Initiating basic XGBoost
estimator = XGBClassifier(
    objective= 'binary:logistic',
    seed=1
)

parameters = {
    'max_depth': [3, 4, 6],
    'lambda': [2, 4, 7, 15],
    'alpha': [2, 4, 7, 15],
    'n_estimators': [250, 350, 450]
}

        
kf = KFold(n_splits=3, shuffle=True, random_state=1)

### GridSearchCV
grid_search_5 = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 1,
    cv = kf,
    verbose=3,
    return_train_score=True
)

grid_search_5.fit(X_train, y_train)

print(grid_search_5.best_params_)
### Best parameters: {'alpha': 15, 'lambda': 4, 'max_depth': 4, 'n_estimators': 450}
### Decision: 


# In[9]:


joblib.dump(grid_search_5, 'grid_search_5.pkl')

#-> Conclusion: pas de valeur ajoutée au tunning


# ##### 6ème tentative (max_depth, lambda plus grand, alpha plus grand et n_estimators fixé)

# In[10]:


##### Initiating basic XGBoost
estimator = XGBClassifier(
    objective= 'binary:logistic',
    seed=1
)

parameters = {
    'max_depth': [6, 10],
    'lambda': [15, 30, 50],
    'alpha': [15, 30, 50],
    'n_estimators': [450]
}

        
kf = KFold(n_splits=3, shuffle=True, random_state=1)

### GridSearchCV
grid_search_6 = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 1,
    cv = kf,
    verbose=3,
    return_train_score=True
)

grid_search_6.fit(X_train, y_train)

print(grid_search_6.best_params_)
### Best parameters: {'alpha': 30, 'lambda': 50, 'max_depth': 6, 'n_estimators': 450}
### Decision: 


# In[11]:


joblib.dump(grid_search_6, 'grid_search_6.pkl')

#-> Conclusion: pas de valeur ajoutée du tunning


# ##### 7ème tentative (max_depth, lambda fixé, alpha plus grand et n_estimators plus grand)

# In[12]:


##### Initiating basic XGBoost
estimator = XGBClassifier(
    objective= 'binary:logistic',
    seed=1
)

parameters = {
    'max_depth': [4, 6],
    'lambda': [4],
    'alpha': [25, 50, 100],
    'n_estimators': [450, 600, 800]
}

        
kf = KFold(n_splits=3, shuffle=True, random_state=1)

### GridSearchCV
grid_search_7 = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 1,
    cv = kf,
    verbose=3,
    return_train_score=True
)

grid_search_7.fit(X_train, y_train)

print(grid_search_7.best_params_)
### Best parameters: {'alpha': 25, 'lambda': 4, 'max_depth': 6, 'n_estimators': 450}
### Decision: 


# In[ ]:


joblib.dump(grid_search_7, 'grid_search_7.pkl')

#-> Conclusion: pas de valeur ajoutée du tuning


# ##### 8ème tentative (learning_rate qui baisse sur le modèle qui a eu les performances les plus intéressantes, dernière étape)

# In[13]:


##### Initiating basic XGBoost
estimator = XGBClassifier(
    objective= 'binary:logistic',
    seed=1
)

parameters = {
    'max_depth': [6],
    'lambda': [15],
    'alpha': [15],
    'n_estimators': [450],
    'learning_rate': [0.3, 0.1, 0.05, 0.01]
}

        
kf = KFold(n_splits=3, shuffle=True, random_state=1)

### GridSearchCV
grid_search_8 = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 1,
    cv = kf,
    verbose=3,
    return_train_score=True
)

grid_search_8.fit(X_train, y_train)

print(grid_search_8.best_params_)
### Best parameters: {'alpha': 15, 'lambda': 15, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 450}
### Decision: 


# In[14]:


joblib.dump(grid_search_8, 'grid_search_8.pkl')

#-> Conclusion: Un learning_rate de 0.1 a été choisi car pas de valeur ajoutée des autres (overfitting probablement)


# ##### Optimal Model

# In[9]:


##### Launching optimal XGBoost
params = [
    ('objective', 'binary:logistic'),
    ('max_depth', 6),
    ('eval_metric', 'auc'),
    ('early_stopping_rounds', 10),
    ('learning_rate', 0.1)
]

### Optimal XGBoost after tuning
optimal_xgb = xgb.train(params=params, dtrain=train, 
                    num_boost_round=500, 
                    evals=[(train, 'train'), (test, 'eval')])


# In[19]:


# Export model
dump(optimal_xgb, 'optimal_xgb.joblib')


# In[20]:


# Import model
loaded_model = load('C:\\Users\\Megaport\\Desktop\\jupyterNotebook\\grid_search\\optimal_xgb.joblib')
# loaded_model.predict(test)


# <span style="color:red">Exploration: tentative d'afficher un arbre décisionnel issu du XGBoost---</span>

# In[141]:


scores = grid_search_1.cv_results_['mean_test_score'].reshape(3, 3, 2)
scores


# In[ ]:


conda install graphviz python-graphviz


# In[102]:


fig, ax = plt.subplots(figsize=(200, 200))
xgb.plot_tree(clf_xgb, num_trees=4, ax=ax, 
              yes_color='#00cc00', no_color='#FF000', 
              condition_node_params={'shape': 'box', 'style': 'solid'})
plt.show()
#plt.savefig('D:\\jupyterDatasets\\xgboost_tree_graph.png')
#plt.savefig("D:\\jupyterDatasets\\xgboost_tree_graph.pdf")


# In[44]:


grid_search_1.best_estimator_


# In[32]:


grid_search_1.cv_results_


# <span style="color:red">---fin de l'exploration</span>
