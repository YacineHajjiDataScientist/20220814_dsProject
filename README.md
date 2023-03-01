# 20220814_dsProject
DS project 'Accidents de la route'

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

4 - interpretability.ipynb


