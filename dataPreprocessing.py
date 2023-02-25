from fonctions import *


##### Pour effectuer le préprocessing, nous commençons par importer chaque table et la formater

#### Pour la table Carac nous avons besoin de 2 sources externes, la table des jours féries en Métropole et
# la table de la population par commune


dfCarac = formatage_table_carac(
    "20230225_table_caracteristiques.csv",
    "20221009_table_joursFeriesMetropole.csv",
    "pop_commune.csv",
)

## Formatage des autres tables disqponibles
dfLieux = formatage_table_lieux("20230225_table_lieux.csv")
dfUsagers = formatage_table_usagers("20230225_table_usagers.csv", dfCarac)
dfVehicules = formatage_table_vehicules("20230225_table_vehicules.csv")

## Création de nouvelles variables à partir des tables Véhicules et Usagers  
dfVarSupp = construction_variables_supp(dfVehicules, dfUsagers)


# Merge des différentes table pour obtenir notre base de données finale
dfPool = construction_table_travail(dfLieux, dfCarac, dfVarSupp)
# Sélection des variables utilisées ensuite dans les modèles

selection_features_ML(dfPool)
