# 20220814_dsProject
DS project 'Accidents de la route'

Nécessite l'installation des packages suivants :
pandas : 1.5.2
numpy : 1.21.5
shap :  0.41.0
sage : 0.0.4
scikit-learn : 1.2.1
joblib : 1.1.1 
matplotlib : 3.6.2
seaborn : 0.12.2
optuna : 3.0.4
plotly : 5.10.0
scipy : 1.10.0
xgboost : 1.7.4
lightgbm : 3.3.3
tensorflow : 2.11.0


Dans ce projet, nous avons décomposé notre travail en plusieurs parties. 

Le fichier "fonctions.py" comportent toutes les fonctions qui ont été utilisées dans l'étude. 

1 - dataPreprocessing.py

inputs : tables usagers / caracteristiques / lieux / vehicules et tables exogènes jours féries / population et densité
Ce programme permet de transformer les tables initiales en table comportant toutes lers features retenues pour nos travaux.
outputs : table avec le pool de variables retenues, une tableau features_matrix et une table target

2 - explorationAndDataVisualisation.ipynb

inputs : tables initiales et table 
Ce notebook présente toutes les visualiations que nous avons effectuées (hors cartographique qui prenait trop de place sur GIT (les fonctions permettant de les faire sont cependant bien dans le fichier fonctions.py)) et les travaux d'exploration des données 
outputs : visualisations du rapport 

3 - modelTuning.py

inputs : table features_matrix et table target
Ce notebook présente la construction des modèles initiaux (RF / LogisticRegression / ElastiNet / LGBM / XGBoost /MLP) et le tuning pour les quelques modèles retenus (XGBoost, LGBM, RF, EN) avant leur enregistrement.
ouputs : modèles optimisés

4 - modelInterpretability.ipynb

inputs : modèles optimisés enregistrés. 
Ce notebook présente nos travaux avec la librairie SHAP pour expliquer les prédictions de nos modèles et présenter les features ayant la plus haute importance. 
outputs : visualisations SHAP

5 - vulgarisation.ipynb

inputs : modèles optimisés
Ce notebook présente notre case study et l'utilisation des modèles dans chacun des scénarios construits.
outputs : résultats des modèles pour chaque scénario.



