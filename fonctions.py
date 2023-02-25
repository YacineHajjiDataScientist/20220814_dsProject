import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime
import math
from scipy.stats import chi2_contingency
import plotly.express as px
import geojson


def mise_au_format_dep(df):
    # Normalisation de la variable département dans un dataframe
    df["dep"] = df["dep"].astype(str)

    # S'il y a 3 caractères avec un 0 en dernière position, on conserve les deux premiers
    df["dep"] = df["dep"].apply(lambda x: x[:2] if (len(x) == 3 and x[2] == "0") else x)

    # Nous retirons de le premier caractère si c'est un 0
    df["dep"] = df["dep"].apply(lambda x: x[1:] if (x[0] == "0") else x)
    # Nous remettons le 0 pour les départements avec un seul chiffre
    df["dep"] = df["dep"].apply(lambda x: "0" + x if (len(x) == 1) else x)

    # Pour les départements corses, nous utilisons 2A et 2B
    df["dep"] = df["dep"].replace("201", "2A")
    df["dep"] = df["dep"].replace("202", "2B")


def regroupement_population(x):

    # Fonction permettant de regrouper dans une variable catégorielle les villes selon leur catégorie
    if x == np.NAN:
        return "Z"
    elif x < 2000:
        return "Village"
    elif x < 5000:
        return "Bourg"
    elif x < 20000:
        return "Petite Ville"
    elif x < 50000:
        return "Ville Moyenne"
    elif x < 200000:
        return "Grande Ville"
    else:
        return "Métropole"


def regroupement_intersection(x):
    # Fonction permettant de regrouper dans une variable catégorielle les intersections selon leur catégorie

    if x == 1:
        return "Hors intersection"
    elif x in [2, 3, 4]:
        return "Croisement de deux routes"
    elif x in [5, 6, 7]:
        return "Croisement circulaire"
    elif x == 8:
        return "Passage à niveau"
    else:
        return "Autres"


def regroupement_cat_veh(x):

    # Fonction permettant de regrouper dans une variable catégorielle les types de véhicules

    cat1 = [
        1,
        2,
        4,
        5,
        6,
        30,
        31,
        32,
        33,
        34,
        80,
        35,
        36,
        41,
        42,
        43,
    ]  # 2, 3 roues et quads
    cat2 = [3, 7, 8, 9, 10, 11, 12]  # VL et VUL
    cat3 = [13, 14, 15, 16, 17, 20, 21, 37, 38, 18]  # PL
    cat4 = [39]  # Trains
    cat5 = [40, 19, 99, 0]  # Tramways
    cat9 = [50, 60]
    if x in cat1:
        return "2,3 roues & quads"
    elif x in cat2:
        return "VL & VUL"
    elif x in cat3:
        return "PL"
    elif x in cat4:
        return "Train"
    elif x in cat5:
        return "Autres"
    elif x in cat9:
        return "EPD"


def regroupement_obstacles(x):
    # Fonction permettant de regrouper les types d'obstacles dans une variable catégorielle
    if x == 0:
        return "Pas d'Obstacle"
    elif x == -1:
        return "Z"
    else:
        return "Obstacle"


def regroupement_nb_vehicules(x):
    # Fonction permettant de catégoriser le nombre de véhicules impliqués dans l'accident
    if x == 1:
        return "1 véhicule"
    elif x == 2:
        return "2 véhicules"
    elif x == 3:
        return "3 véhicules"
    elif x > 3 and x < 10:
        return "entre 4 et 10 véhicules"
    else:
        return "+ de 10 véhicules"


def fonction_choc_personne(x):
    # Fonction permettant de déterminer de quel côté du véhicule était situé la personne
    if x.choc in (1, 2, 3) and x.place in (1, 6, 2):
        return "Avant"
    elif x.choc in (3, 6, 8) and x.place in (1, 7, 4):
        return "Gauche"
    elif x.choc in (2, 5, 7) and x.place in (3, 9, 2):
        return "Droite"
    elif x.choc in (4, 5, 6) and x.place in (4, 5, 3):
        return "Arrière"
    else:
        return "Z"


def cat(x):
    # Fonction indicatrice si différent de non renseigné
    if x != "Z":
        return 1
    else:
        return 0


def retraitement_pop_ville(x):
    # Fonction pour transformer en NA pour certaines communes pour lesquelles trop d'accidents semble renseignés vu la population
    if (
        (x.nom_commune == "Bornel")
        | (x.nom_commune == "Betz")
        | (x.nom_commune == "Auneuil")
    ):
        return np.NaN
    else:
        return x.population_tot


def formatage_table_carac(
    nom_fichier_carac, nom_fichier_jours_feries, nom_fichier_communes
):
    ##### Fonction permettant d'importer la table Caractéristiques puis de la formater (ajout de variables, conversion de types etc...)

    dfCarac = pd.read_csv(nom_fichier_carac, sep=",", index_col=0, low_memory=False)

    # Dataframes additionnels pour le formatage de la table Carac

    dfJoursFeriesMetropole = pd.read_csv(nom_fichier_jours_feries, sep=";")
    dfCommunes = pd.read_csv(nom_fichier_communes, sep=";")

    # Suppression des variables que l'on utilisera pas, la latitude, la longitude, l'adresse car comportant trop de possibilités
    # et avec des risques d'erreurs importants
    dfCarac = dfCarac.drop(["lat", "long", "gps", "adr"], axis=1)

    # Retraitement de l'année de XX à 20XX
    dfCarac["year"] = dfCarac["an"].apply(lambda x: (2000 + x) if x < 2000 else x)
    # Construction de la variable date à partir du jour, du mois et de l'an
    dfCarac["date"] = dfCarac.apply(
        lambda x: datetime(x["year"], x["mois"], x["jour"]), axis=1
    )

    dfCarac["date"] = pd.to_datetime(dfCarac["date"])
    # Transformation en variable catégorielle du mois
    dfCarac["mois_label"] = dfCarac["mois"]
    dfCarac["mois_label"] = dfCarac["mois_label"].replace(
        to_replace=np.arange(1, 13, 1),
        value=[
            "jan",
            "fev",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ],
    )

    # Construction de la variable jour de la semaine
    dfCarac["weekday"] = dfCarac["date"].apply(lambda x: x.weekday())
    # Construction de la variable heure de la journée
    dfCarac["hrmn"] = dfCarac["hrmn"].replace("\:", "", regex=True).astype(int)
    dfCarac["hour"] = dfCarac["hrmn"] // 100

    # Construction de la variable hourGrp: nuit (22h - 6h) - jour heures creuses (10h-16h) - jour heures de pointe (7-9h, 17-21h)
    hourConditions = [
        ((dfCarac["hour"] >= 22) | (dfCarac["hour"] <= 6)),
        (
            ((dfCarac["hour"] >= 7) & (dfCarac["hour"] <= 9))
            | ((dfCarac["hour"] >= 17) & (dfCarac["hour"] <= 21))
        ),
        ((dfCarac["hour"] >= 10) | (dfCarac["hour"] <= 16)),
    ]
    hourChoices = ["nuit", "heure de pointe", "journee"]
    dfCarac["hourGrp"] = np.select(hourConditions, hourChoices)

    # Transformation de la variable atm: passer en NA les valeurs -1 et 9 (other) qui sont difficilement interprétables dans un modèle de ML
    dfCarac["atm"] = dfCarac["atm"].replace([-1, 9], [np.nan, np.nan])

    # Récupération des dates feriée/non fériée et semaine/weekend
    dateFerie = list(
        map(
            lambda x: datetime.strptime(x, "%d/%m/%Y").strftime("%Y-%m-%d"),
            dfJoursFeriesMetropole["date"],
        )
    )
    dfDateFerie = pd.DataFrame({"dateFerie": dateFerie})
    dfCarac["dateFerie"] = np.where((dfCarac.date.isin(dfDateFerie.dateFerie)), 1, 0)
    dfCarac["dateWeekend"] = np.where((dfCarac["weekday"] >= 5), 1, 0)
    dfCarac["dateFerieAndWeekend"] = np.where(
        (dfCarac.date.isin(dfDateFerie.dateFerie) | (dfCarac["weekday"] >= 5)), 1, 0
    )

    # On utilise la fonction mise_au_format_dep pour normaliser les départements dans les deux tables dfCarac et dfCommunes

    mise_au_format_dep(dfCarac)
    mise_au_format_dep(dfCommunes)

    # Normalisation de la variable "com" (numéro de la commune) entre les 2 tables)
    dfCarac["com"] = pd.to_numeric(dfCarac["com"], errors="coerce")
    dfCarac["com"] = dfCarac["com"].apply(lambda x: x % 100 if x > 1000 else x)
    dfCommunes["com"] = pd.to_numeric(dfCommunes["com"], errors="coerce")

    # Récupération de la population totale par communes (regroupement des villes avec plusieurs arrondissements) pour la table communes
    dfCommunes = dfCommunes.drop(
        [
            "code_region",
            "code_arrondissement",
            "code_canton",
            "population_mun",
            "population_part",
        ],
        axis=1,
    )
    dfCommunes = dfCommunes.drop_duplicates(subset=["dep", "com"], keep="first")

    # Récupération de la population de la ville où a eu lieu l'accident
    dfCarac = dfCarac.merge(
        dfCommunes[["dep", "com", "population_tot", "nom_commune"]],
        on=["dep", "com"],
        how="left",
    )

    dfCarac["population_tot"] = dfCarac["population_tot"].str.replace(" ", "")
    dfCarac["population_tot"] = pd.to_numeric(dfCarac["population_tot"])

    # Utilisation de la fonction retraitement_pop_ville pour remplir en NA des villes mal renseignées
    dfCarac["population_tot"] = dfCarac.apply(retraitement_pop_ville, axis=1)
    # Fonction pour récupérer la catégorie de la ville selon sa population
    dfCarac["populationGrp"] = dfCarac["population_tot"].apply(
        lambda x: regroupement_population(x)
    )

    # Fonction pour récupérer la catégorie d'intersection
    dfCarac["intGrp"] = dfCarac.int.apply(regroupement_intersection)

    # Transformation pour lum  des -1,en  NA
    dfCarac["lum"] = dfCarac["lum"].replace([-1], [np.nan])
    # Transformation pour col:  des -1,en  NA
    dfCarac["col"] = dfCarac["col"].replace([-1], [np.nan])

    # Retrait des variables inutiles
    dfCarac = dfCarac.drop(["an", "hrmn"], axis=1)

    return dfCarac


def formatage_table_lieux(nom_fichier_lieux):

    dfLieux = pd.read_csv(nom_fichier_lieux, sep=",", index_col=0, low_memory=False)

    # Retrait des variables inutiles, v1 et v2 qui indiquent l'emplacement précis de l'accident (trop spécifique)
    # et pr et pr1 les bornes de rattachement
    dfLieux = dfLieux.drop(["voie", "v1", "v2", "pr", "pr1"], axis=1)

    # Formatage de la largeur de la route assignée au trafic
    dfLieux.larrout = dfLieux.larrout.replace("\,", ".", regex=True).astype("float64")
    dfLieux.lartpc = dfLieux.lartpc.replace("\,", ".", regex=True).astype("float64")

    # Codage de la variable nbvGrp: 0/1/2/3/4+, avec -1 et 9+ en NA, qui catégorise selon le nombre de voies
    nbvConditions = [
        ((dfLieux["nbv"] >= 9) | (dfLieux["nbv"] == -1)),
        (dfLieux["nbv"] == 0),
        (dfLieux["nbv"] == 1),
        (dfLieux["nbv"] == 2),
        (dfLieux["nbv"] == 3),
        (dfLieux["nbv"] >= 4),
    ]
    nbvChoices = [np.nan, "0", "1", "2", "3", "4+"]
    dfLieux["nbvGrp"] = np.select(nbvConditions, nbvChoices)
    dfLieux["nbvGrp"] = dfLieux["nbvGrp"].replace(["nan"], [np.nan])

    # Codage de la variable lartpcGrp: 0/1/2/3/4+, avec -1 et 9+ en NA
    lartpcConditions = [
        ((dfLieux["lartpc"] == 0.0)),
        ((dfLieux["lartpc"] >= 20)),
        ((dfLieux["lartpc"] > 0) & (dfLieux["lartpc"] < 5)),
        ((dfLieux["lartpc"] >= 5) & (dfLieux["lartpc"] < 10)),
        ((dfLieux["lartpc"] >= 10) & (dfLieux["lartpc"] < 15)),
        ((dfLieux["lartpc"] >= 15) & (dfLieux["lartpc"] < 20)),
    ]
    lartpcChoices = [np.nan, np.nan, 1, 2, 3, 4]
    dfLieux["lartpcGrp"] = np.select(lartpcConditions, lartpcChoices)
    dfLieux["lartpcGrp"] = dfLieux["lartpcGrp"].replace(
        [0, 1, 2, 3, 4], [np.nan, "0-5", "5-10", "10-15", "15-20"]
    )

    # Codage de la variable larroutGrp: 0/1/2/3/4+, avec -1 et 9+ en NA
    larroutConditions = [
        ((dfLieux["larrout"] == 0.0)),
        ((dfLieux["larrout"] >= 200)),
        ((dfLieux["larrout"] > 0) & (dfLieux["larrout"] < 50)),
        ((dfLieux["larrout"] >= 50) & (dfLieux["larrout"] < 100)),
        ((dfLieux["larrout"] >= 100) & (dfLieux["larrout"] < 150)),
        ((dfLieux["larrout"] >= 150) & (dfLieux["larrout"] < 200)),
    ]
    larroutChoices = [np.nan, np.nan, 1, 2, 3, 4]
    dfLieux["larroutGrp"] = np.select(larroutConditions, larroutChoices)
    dfLieux["larroutGrp"] = dfLieux["larroutGrp"].replace(
        [0, 1, 2, 3, 4], [np.nan, "0-50", "50-100", "100-150", "150-200"]
    )

    # Recodage de certaines variables en binaire
    dfLieux["vospGrp"] = dfLieux["vosp"].replace([-1, 0, 1, 2, 3], [np.nan, 0, 1, 1, 1])
    dfLieux["planGrp"] = dfLieux["plan"].replace(
        [-1, 0, 1, 2, 3, 4], [np.nan, np.nan, 0, 1, 1, 1]
    )

    # Transformation en NA pour des modalités de certaines variables
    dfLieux["prof"] = dfLieux["prof"].replace([-1, 0], [np.nan, np.nan])
    dfLieux["circ"] = dfLieux["circ"].replace([-1, 0], [np.nan, np.nan])
    dfLieux["surf"] = dfLieux["surf"].replace([-1, 0, 9], [np.nan, np.nan, np.nan])
    dfLieux["situ"] = dfLieux["situ"].replace([-1, 0], [np.nan, np.nan])
    dfLieux["infra"] = dfLieux["infra"].replace([-1], [np.nan])

    # Retrait des variables que l'on ne souhaite pas utiliser par la suite
    dfLieux = dfLieux.drop(["nbv", "vosp", "plan", "vma"], axis=1)

    return dfLieux


def formatage_table_usagers(nom_fichier_usagers, table_carac):

    dfUsagers = pd.read_csv(nom_fichier_usagers, sep=",", index_col=0, low_memory=False)

    # Ajout de l'année de l'accident (via la table Caractéristique), à la table usagers
    dfUsagers = dfUsagers.merge(right=table_carac[["Num_Acc", "year"]], on="Num_Acc")

    # Calcul de l'âge de la personne au moment de l'accident (borne max à 99 ans)
    dfUsagers["age"] = dfUsagers.year - dfUsagers.an_nais
    dfUsagers.loc[dfUsagers["age"] > 99, "age"] = np.nan

    # Construction d'indicatrices de gravité de l'accident
    dfUsagers["grav34exists"] = np.where(dfUsagers.grav2 >= 3, 1, 0)
    dfUsagers["grav4exists"] = np.where(dfUsagers.grav2 == 4, 1, 0)
    dfUsagers["grav3exists"] = np.where(dfUsagers.grav2 == 3, 1, 0)
    dfUsagers["grav2exists"] = np.where(dfUsagers.grav2 == 2, 1, 0)
    dfUsagers["place"] = dfUsagers["place"].replace([0], [np.nan])

    # Formatage de la variable actp:
    dfUsagers["actp"] = dfUsagers["actp"].replace(
        {
            "0.0": 0,
            "0": 0,
            0: 0,
            "-1.0": np.nan,
            "-1": np.nan,
            " -1": np.nan,
            -1: np.nan,
            "1.0": 1,
            "1": 1,
            1: 1,
            "2.0": 2,
            "2": 2,
            2: 2,
            "3.0": 3,
            "3": 3,
            3: 3,
            "4.0": 4,
            "4": 4,
            4: 4,
            "5.0": 5,
            "5": 5,
            5: 5,
            "6.0": 6,
            "6": 6,
            6: 6,
            "7.0": 7,
            "7": 7,
            7: 7,
            "8.0": 8,
            "8": 8,
            8: 8,
            "9.0": 9,
            "9": 9,
            9: 9,
        }
    )
    # Formatage de la variable etatp: transformation des -1 en NA et nombre de piétons seuls dans l'accident
    dfUsagers["etatp"] = dfUsagers["etatp"].replace([-1], [np.nan])
    dfUsagers["etatp_pieton_alone_exists"] = np.where((dfUsagers["etatp"] == 1), 1, 0)
    # Formatage de la variable locp et construction d'indicatrices :
    # transformation des 0 en NA et nombre de piétons en fonction de leur position pendant l'accident
    dfUsagers["locp"] = dfUsagers["locp"].replace([-1], [np.nan])
    dfUsagers["locp_pieton_1_exists"] = np.where(((dfUsagers.locp == 1)), 1, 0)
    dfUsagers["locp_pieton_3_exists"] = np.where(((dfUsagers.locp == 3)), 1, 0)
    dfUsagers["locp_pieton_6_exists"] = np.where(((dfUsagers.locp == 6)), 1, 0)

    # Construction d'indicatrices piéton / conducteurs
    dfUsagers["catu_pieton_exists"] = np.where(
        ((dfUsagers.catu == 3) | (dfUsagers.catu == 4)), 1, 0
    )
    dfUsagers["catu_conductor_exists"] = np.where(((dfUsagers.catu == 1)), 1, 0)

    # Construction d'indicatrices hommes / femmes
    dfUsagers["sexe_male_conductor_exists"] = np.where(
        ((dfUsagers.sexe == 1) & (dfUsagers.catu == 1)), 1, 0
    )
    dfUsagers["sexe_female_conductor_exists"] = np.where(
        ((dfUsagers.sexe == 2) & (dfUsagers.catu == 1)), 1, 0
    )

    # Construction indicatrice trajet Promenade pour les conducteurs
    dfUsagers["trajet_coursesPromenade_conductor_exists"] = np.where(
        (
            ((dfUsagers.trajet == 3) & (dfUsagers.catu == 1))
            | ((dfUsagers.trajet == 5) & (dfUsagers.catu == 1))
        ),
        1,
        0,
    )

    ## Calcul de l'âge moyen des conducteurs et des non-conducteurs de l'accident
    # Construction d'un dataframe par accident
    dfAgeMeanConductors = (
        dfUsagers[(dfUsagers["catu_conductor_exists"] == 1)][["Num_Acc", "age"]]
        .groupby(["Num_Acc"])
        .mean()
        .rename({"age": "ageMeanConductors"}, axis=1)
        .reset_index()
    )
    dfAgeMeanNonConductors = (
        dfUsagers[(dfUsagers["catu_conductor_exists"] == 0)][["Num_Acc", "age"]]
        .groupby(["Num_Acc"])
        .mean()
        .rename({"age": "ageMeanNonConductors"}, axis=1)
        .reset_index()
    )

    # Rapatriement de l'information sur chaque ligne
    dfUsagers = dfUsagers.merge(right=dfAgeMeanConductors, how="left", on="Num_Acc")
    dfUsagers = dfUsagers.merge(right=dfAgeMeanNonConductors, how="left", on="Num_Acc")

    return dfUsagers


def formatage_table_vehicules(nom_fichier_vehicules):

    dfVehicules = pd.read_csv(
        nom_fichier_vehicules, sep=",", index_col=0, low_memory=False
    )

    # Utilisation de la fonction 'regroupement_obstacles" pour catégoriser le type d'obstacle
    dfVehicules["obsGrp"] = dfVehicules.obs.apply(regroupement_obstacles)
    dfVehicules["obsGrp"] = dfVehicules["obsGrp"].replace(["Z"], [np.nan])

    # Construction de la variable catégorie de véhicules et d'indicatrices liés à des types intéressants
    dfVehicules["catvGrp"] = dfVehicules.catv.apply(regroupement_cat_veh)
    dfVehicules["catv_train_exist"] = np.where(dfVehicules.catvGrp == "Train", 1, 0)
    dfVehicules["catv_PL_exist"] = np.where(dfVehicules.catvGrp == "PL", 1, 0)
    dfVehicules["catv_2_roues_exist"] = np.where(
        dfVehicules.catvGrp == "2,3 roues & quads", 1, 0
    )
    dfVehicules["catv_EPD_exist"] = np.where(dfVehicules.catvGrp == "EPD", 1, 0)

    return dfVehicules


def construction_variables_supp(dfVehicules, dfUsagers):

    # Lien entre les différentes tables pour construire de nouvelles variables / puis obtenir notre table de travail

    # Création d'une table pour garder la présence d'obstacles ou non
    dfVehicules_obs = dfVehicules.sort_values(by="obs", ascending=True)
    dfVehicules_obs = dfVehicules_obs[["Num_Acc", "obsGrp"]].drop_duplicates(
        subset=["Num_Acc"], keep="first"
    )

    # Construction de la variable nombre de véhicules puis de sa catégorisation
    dfNombreVehicule = (
        dfVehicules[["Num_Acc", "num_veh"]].groupby(["Num_Acc"]).count().reset_index()
    )
    dfNombreVehicule = dfNombreVehicule.sort_values(by="num_veh", ascending=False)
    dfNombreVehicule["nbVeh"] = dfNombreVehicule.num_veh.apply(
        regroupement_nb_vehicules
    )
    dfNombreVehicule = dfNombreVehicule.drop("nbVeh", axis=1)

    # Construction des indicatrices au niveau de l'accident et non plus du véhicule
    dfVeh_type_veh = (
        dfVehicules[
            [
                "Num_Acc",
                "catv_train_exist",
                "catv_PL_exist",
                "catv_2_roues_exist",
                "catv_EPD_exist",
            ]
        ]
        .groupby("Num_Acc")
        .sum()
        .reset_index()
    )
    dfVeh_type_veh["catv_train_exist"] = dfVeh_type_veh["catv_train_exist"].apply(
        lambda x: 1 if x >= 1 else 0
    )
    dfVeh_type_veh["catv_PL_exist"] = dfVeh_type_veh["catv_PL_exist"].apply(
        lambda x: 1 if x >= 1 else 0
    )
    dfVeh_type_veh["catv_2_roues_exist"] = dfVeh_type_veh["catv_2_roues_exist"].apply(
        lambda x: 1 if x >= 1 else 0
    )
    dfVeh_type_veh["catv_EPD_exist"] = dfVeh_type_veh["catv_EPD_exist"].apply(
        lambda x: 1 if x >= 1 else 0
    )

    # Regroupement des tables Usagers et Véhicules pour récupérer le nombre de personnes dans les véhicules positionnées du côté du choc
    dfMergeUsVeh = dfUsagers.merge(dfVehicules, how="left", on=["Num_Acc", "num_veh"])
    dfMergeUsVeh["choc_place"] = dfMergeUsVeh.apply(
        lambda x: fonction_choc_personne(x), axis=1
    )
    dfMergeUsVeh["choc_cote"] = dfMergeUsVeh["choc_place"].apply(cat)

    dfMergeUsVeh = (
        dfMergeUsVeh[["Num_Acc", "choc_cote"]].groupby(["Num_Acc"]).sum().reset_index()
    )

    # Construction d'un dataframe d'indicatrices et de compteurs par accident à partir de la table Usagers
    dfAtLeastOneByAccident = pd.DataFrame(
        {
            # event exists yes/no by accident
            "gravGrp_23_4": np.where(
                dfUsagers.groupby("Num_Acc")["grav4exists"].sum() >= 1, 1, 0
            ),
            "gravGrp_2_34": np.where(
                dfUsagers.groupby("Num_Acc")["grav34exists"].sum() >= 1, 1, 0
            ),
            "catu_pieton": np.where(
                dfUsagers.groupby("Num_Acc")["catu_pieton_exists"].sum() >= 1, 1, 0
            ),
            "sexe_male_conductor": np.where(
                dfUsagers.groupby("Num_Acc")["sexe_male_conductor_exists"].sum() >= 1,
                1,
                0,
            ),
            "sexe_female_conductor": np.where(
                dfUsagers.groupby("Num_Acc")["sexe_female_conductor_exists"].sum() >= 1,
                1,
                0,
            ),
            "trajet_coursesPromenade_conductor": np.where(
                dfUsagers.groupby("Num_Acc")[
                    "trajet_coursesPromenade_conductor_exists"
                ].sum()
                >= 1,
                1,
                0,
            ),
            # count event variable by accident
            "nb_grav4_by_acc": dfUsagers.groupby("Num_Acc")["grav4exists"].sum(),
            "nb_grav3_by_acc": dfUsagers.groupby("Num_Acc")["grav3exists"].sum(),
            "nb_catu_pieton": dfUsagers.groupby("Num_Acc")["catu_pieton_exists"].sum(),
            "nb_sexe_male_conductor": dfUsagers.groupby("Num_Acc")[
                "sexe_male_conductor_exists"
            ].sum(),
            "nb_sexe_female_conductor": dfUsagers.groupby("Num_Acc")[
                "sexe_female_conductor_exists"
            ].sum(),
            "nb_trajet_coursesPromenade_conductor": dfUsagers.groupby("Num_Acc")[
                "trajet_coursesPromenade_conductor_exists"
            ].sum(),
            "nb_etatpGrp_pieton_alone": dfUsagers.groupby("Num_Acc")[
                "etatp_pieton_alone_exists"
            ].sum(),
            "nb_locpGrp_pieton_1": dfUsagers.groupby("Num_Acc")[
                "locp_pieton_1_exists"
            ].sum(),
            "nb_locpGrp_pieton_3": dfUsagers.groupby("Num_Acc")[
                "locp_pieton_3_exists"
            ].sum(),
            "nb_locpGrp_pieton_6": dfUsagers.groupby("Num_Acc")[
                "locp_pieton_6_exists"
            ].sum(),
            # mean of variable by accident
            "ageMeanConductors": dfUsagers.groupby("Num_Acc")[
                "ageMeanConductors"
            ].mean(),
            "ageMeanNonConductors": dfUsagers.groupby("Num_Acc")[
                "ageMeanNonConductors"
            ].mean(),
        }
    ).reset_index()

    dfAtLeastOneByAccident["etatpGrp_pieton_alone"] = np.where(
        dfAtLeastOneByAccident.groupby("Num_Acc")["nb_etatpGrp_pieton_alone"].sum()
        >= 1,
        1,
        0,
    )
    dfAtLeastOneByAccident["locpGrp_pieton_1"] = np.where(
        dfAtLeastOneByAccident.groupby("Num_Acc")["nb_locpGrp_pieton_1"].sum() >= 1,
        1,
        0,
    )
    dfAtLeastOneByAccident["locpGrp_pieton_3"] = np.where(
        dfAtLeastOneByAccident.groupby("Num_Acc")["nb_locpGrp_pieton_3"].sum() >= 1,
        1,
        0,
    )
    dfAtLeastOneByAccident["locpGrp_pieton_6"] = np.where(
        dfAtLeastOneByAccident.groupby("Num_Acc")["nb_locpGrp_pieton_6"].sum() >= 1,
        1,
        0,
    )

    # Construction d'un dataframe regroupant l'ensemble des variables supplémentaires constituées
    dfVarSuppVeh = dfVeh_type_veh.merge(dfNombreVehicule, on="Num_Acc")
    dfVarSuppVeh = dfVarSuppVeh.merge(dfVehicules_obs, on="Num_Acc")
    dfVarSuppVeh = dfVarSuppVeh.merge(dfMergeUsVeh, on="Num_Acc", how="left")
    dfVarSuppVeh = dfVarSuppVeh.merge(dfAtLeastOneByAccident, on="Num_Acc", how="left")

    return dfVarSuppVeh


def construction_table_travail(dfLieux, dfCarac, dfVarSupp):
    ## Fonction permettant de construire la table finale

    # Merge des différentes tables formatées et construites
    dfPool = pd.merge(dfLieux, dfCarac, on="Num_Acc")
    dfPool = pd.merge(dfPool, dfVarSupp, on="Num_Acc")

    # Transformation des -1 en NA
    dfPool = dfPool.replace(-1, np.nan)

    ### Modification des types de variables
    dfPool[
        [
            "etatpGrp_pieton_alone",
            "prof",
            "circ",
            "planGrp",
            "surf",
            "atm",
            "vospGrp",
            "catv_EPD_exist",
            "catv_PL_exist",
            "trajet_coursesPromenade_conductor",
            "sexe_male_conductor",
            "sexe_female_conductor",
            "catv_train_exist",
            "infra",
            "catr",
            "lum",
            "catv_2_roues_exist",
            "col",
            "situ",
            "dateFerieAndWeekend",
            "dateFerie",
            "locpGrp_pieton_1",
            "locpGrp_pieton_3",
            "locpGrp_pieton_6",
        ]
    ] = dfPool[
        [
            "etatpGrp_pieton_alone",
            "prof",
            "circ",
            "planGrp",
            "surf",
            "atm",
            "vospGrp",
            "catv_EPD_exist",
            "catv_PL_exist",
            "trajet_coursesPromenade_conductor",
            "sexe_male_conductor",
            "sexe_female_conductor",
            "catv_train_exist",
            "infra",
            "catr",
            "lum",
            "catv_2_roues_exist",
            "col",
            "situ",
            "dateFerieAndWeekend",
            "dateFerie",
            "locpGrp_pieton_1",
            "locpGrp_pieton_3",
            "locpGrp_pieton_6",
        ]
    ].astype(
        object
    )

    # Renommage des variables
    dfPool = dfPool.rename(columns={"num_veh": "nbVeh"})

    # Modification de l'index
    dfPool = dfPool.set_index("Num_Acc")

    return dfPool


def selection_features_ML(dfPool):
    # Fonction permettant de ne retenir que les variables qui seront utilisées dans les algorithmes testés par la suite et génère 2 .csv
    # L'un pour les variables explicatives, l'un pour la cible
    dfPoolML = dfPool[
        [
            # Variable à expliquer
            "gravGrp_2_34",
            # Variables explicatives
            "choc_cote",
            "ageMeanConductors",
            "nbVeh",
            "prof",
            "planGrp",
            "surf",
            "atm",
            "vospGrp",
            "catv_EPD_exist",
            "catv_PL_exist",
            "trajet_coursesPromenade_conductor",
            "sexe_male_conductor",
            "sexe_female_conductor",
            "intGrp",
            "catv_train_exist",
            "infra",
            "catr",
            "hourGrp",
            "lum",
            "circ",
            "nbvGrp",
            "catv_2_roues_exist",
            "col",
            "obsGrp",
            "situ",
            "populationGrp",
            "mois_label",
            "dateFerieAndWeekend",
            "dateFerie",
            "etatpGrp_pieton_alone",
            "locpGrp_pieton_1",
            "locpGrp_pieton_3",
            "locpGrp_pieton_6",
        ]
    ]

    ### Retrait des NA
    dfPoolMLCCA = dfPoolML.dropna()

    ### Définition des variables explicatives et de la feature
    target = dfPoolMLCCA.gravGrp_2_34
    features = dfPoolMLCCA.drop("gravGrp_2_34", axis=1)
    features_matrix = pd.get_dummies(features, drop_first=True)

    ### Retrait des 70 features les moins intéressantes
    # (based on xgboost weight/gain/cover/total_gain/total_cover informations)
    features_matrix = features_matrix.drop(
        [
            "atm_4.0",
            "mois_label_nov",
            "locpGrp_pieton_6_1",
            "surf_3.0",
            "catr_5.0",
            "surf_4.0",
            "infra_4.0",
            "infra_2.0",
            "situ_5.0",
            "infra_1.0",
            "infra_6.0",
            "mois_label_may",
            "dateFerie_1",
            "infra_8.0",
            "lum_4.0",
            "prof_4.0",
            "mois_label_jun",
            "hourGrp_journee",
            "mois_label_sep",
            "surf_5.0",
            "catr_7.0",
            "catr_6.0",
            "surf_6.0",
            "surf_7.0",
            "dateFerieAndWeekend_1",
            "locpGrp_pieton_1_1",
            "atm_6.0",
        ],
        axis=1,
    )

    ### Export
    # DataFrame post data-processing

    # DataFrame for Machine Learning (all 97 features), removing variable from dfPool
    dfPoolMLCCA.to_pickle("20230225_table_dfPoolMLCCA.csv")

    # DataFrame containing all features kept and target variable
    features_matrix.to_pickle("20230225_table_feature_matrix.csv")
    print(features_matrix.shape)
    target.to_pickle("20230225_table_target.csv")
    print(target.shape)


def countplot_base(table, variable, title, palette, xticks=[], label_xticks=[]):
    # Fonction permettant d'afficher un countplot titré d'une variable pour une table donnée
    plt.figure(figsize=(8, 6))
    sns.countplot(data=table, x=variable, palette=palette)
    if xticks != []:
        plt.xticks(xticks, label_xticks)
    plt.hlines(
        y=len(table[variable]) / len(table[variable].unique()),
        xmin=-0.5,
        xmax=len(table[variable].unique()) - 0.5,
        color="blue",
        alpha=0.4,
    )
    plt.title(title)


def barplot_variable(
    table, feature, target, title, xticks=[], label_xticks=[], table_usager=0
):
    # Fonction permettant d'afficher un barplot titré de la variable cible selon une feature (catégorielle)
    tableGby = (
        table.groupby([feature])[target]
        .value_counts(normalize=True)
        .rename("percentage")
        .mul(100)
        .reset_index()
        .sort_values(target)
    )

    # Display plotx
    fig, ax = plt.subplots(figsize=(8, 6))

    if table_usager == 0:
        sns.barplot(
            x=feature,
            y="percentage",
            hue=target,
            data=tableGby,
            palette=["#C8C8C8", "#F4B650", "#F45050"],
        )
        if xticks != []:
            plt.xticks(xticks, label_xticks)
    else:
        sns.barplot(
            x=feature,
            y="percentage",
            hue=target,
            data=tableGby,
            palette=["grey", "#C8C8C8", "#F4B650", "#F45050"],
        )

    plt.title(title)


def heatmap_crosstable(table, feature, target, title):
    # Fonction permettant d'afficher une heatmap de la variable explicative avec la variable cible
    # pour trouver les déséquilibres de répartition de la cible
    dfCrosstab = pd.crosstab(table[feature], table[target], normalize=0).sort_values(
        by=4, ascending=False
    )

    # Display plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    sns.heatmap(dfCrosstab, annot=True, cmap="cubehelix", ax=ax[0])
    sns.heatmap(
        dfCrosstab.apply(
            lambda x: x / table[target].value_counts(normalize=True), axis=1
        ),
        annot=True,
        cmap="magma_r",
        ax=ax[1],
    )
    plt.title(title)


def kdeplot_variable(table, feature, target, borne_inf, borne_sup, title, palette):
    # Fonction permettant d'afficher la distribution d'une variable
    sns.kdeplot(
        data=table[(table[feature] < borne_sup) & (table[feature] > borne_inf)],
        x=feature,
        hue=target,
        fill=True,
        common_norm=False,
        palette=palette,
        alpha=0.5,
        linewidth=2,
    )
    plt.title(title)


def barplot_heatmap_associated(
    table,
    feature,
    target,
    title_countplot,
    title_heatmap,
    xticks_labels_cntplot=[],
    xticks_labels_heatmap=[],
    x_ticks_label=0,
):
    # Fonction permettant d'afficher un countplot et une heatmap de l'étude d'une feature sur deux graphiques à côté

    # Initiating gravity proportion of CCA lum variable
    dfTarget = table[target].value_counts(normalize=True)

    dfFeatureCrosstab = pd.crosstab(
        table[feature],
        table[target],
        normalize=0,
    ).sort_values(by=4, ascending=False)

    # Initiating dataframe grouped by hour
    dfTargetGby = (
        table.groupby([feature])[target]
        .value_counts(normalize=True)
        .rename("percentage")
        .mul(100)
        .reset_index()
        .sort_values(target)
    )

    # Display plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    # 1st plot
    sns.barplot(
        x=feature,
        y="percentage",
        hue=target,
        data=dfTargetGby,
        palette=["#C8C8C8", "#F4B650", "#F45050"],
        ax=ax[0],
    )
    # adding horizontal overall proportion by gravity
    ax[0].axhline(y=dfTarget.loc[2] * 100, color="#C8C8C8", linestyle="--")
    ax[0].axhline(y=dfTarget.loc[3] * 100, color="#F4B650", linestyle="--")
    ax[0].axhline(y=dfTarget.loc[4] * 100, color="#F45050", linestyle="--")

    # text outside the plot
    if x_ticks_label == 1:
        ax[0].set_xticks(np.arange(0, len(xticks_labels_cntplot), 1))
        ax[0].set_xticklabels(xticks_labels_cntplot)
    ax[0].set_title(title_countplot)
    ax[0].set_xlabel("")
    ax[0].set_ylabel("%", rotation=0)
    # 2nd plot
    sns.heatmap(
        dfFeatureCrosstab.apply(
            lambda x: x / table[target].value_counts(normalize=True),
            axis=1,
        ),
        annot=True,
        cmap="magma_r",
        ax=ax[1],
    )
    # text outside the plot
    ax[1].set_xticks([0.5, 1.5, 2.5])
    ax[1].set_xticklabels(xticks_labels_heatmap)
    if x_ticks_label == 1:
        ax[1].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
        ax[1].set_yticklabels(
            xticks_labels_cntplot,
            rotation=0,
        )
    ax[1].set_title(title_heatmap)
    ax[1].set_xlabel("")
    ax[1].set_ylabel("")


def plot_std(table, feature, title, color):
    # Fonction permettant d'afficher une variable
    varDate = table[feature].value_counts().sort_index()
    # Display plot
    plt.figure(figsize=(10, 5))
    plt.plot(varDate.index, varDate, color=color)
    plt.axhline(y=varDate.mean(), color="k", linestyle="--")
    plt.title(title)

    plt.ylim([0, varDate.max()])


def V_cramer(tab, n):
    # Fonction permettant de calculer le V de Cramer entre les variables d'une table
    # Initiating objects
    nrow, ncol = tab.shape
    resultats_test = chi2_contingency(tab)
    statistique = resultats_test[0]
    # Computing objects
    r = ncol - (((ncol - 1) ** 2) / (n - 1))
    k = nrow - (((nrow - 1) ** 2) / (n - 1))
    phi_squared = max(0, ((statistique / n) - (((ncol - 1) * (nrow - 1)) / (n - 1))))
    V = math.sqrt((phi_squared / (min(k - 1, r - 1))))
    return V


def vCramer_table(df, listeVar, title):

    # Fonction permettant d'afficher le heatmap des V de Cramer d'une table
    resMatrixCarac = pd.DataFrame(
        np.zeros(shape=(len(listeVar), len(listeVar))), index=listeVar, columns=listeVar
    )
    ### Filling dataframe (Carac)
    for i in listeVar:
        for j in listeVar:
            tab = pd.crosstab(df[i], df[j])
            resMatrixCarac[j][i] = round(V_cramer(tab, tab.sum().sum()), 2)
    sns.heatmap(resMatrixCarac)
    plt.title(title)


def carto(table, geojson, feature, target, title, maximum=1000):
    # Fonction permettant d'afficher une carte d'une feature par rapport à sa target
    fig3 = px.choropleth_mapbox(
        table,
        locations=feature,
        geojson=geojson,
        color=target,
        color_continuous_scale=["green", "orange", "red"],
        range_color=[min(table[target]), min(max(table[target]),maximum)],
        title=title,
        mapbox_style="open-street-map",
        center={"lat": 46, "lon": 2},
        zoom=4,
        opacity=0.6,
    )

    fig3.show()


def formatage_geojson(fichier):
    with open(fichier, encoding="UTF-8") as dep:
        departement = geojson.load(dep)
    for feature in departement["features"]:
        feature["id"] = feature["properties"]["code"]
    return departement



def carte_densite_pop(title):
    # Affichage de la densité de population par département
    departement = formatage_geojson("departements.geojson")
    densite = pd.read_csv("densite_pop.csv", sep=";")
    densite["code"] = densite["code"].apply(
        lambda x: "0" + x if len(x.strip()) == 1 else x
    )
    densite["densite"] = densite["densite"].fillna('0')
    densite["densite"] = densite['densite'].apply(lambda x: str(x).replace(',','.'))
    
    densite["densite"] = pd.to_numeric(densite["densite"])
    carto(densite, departement, "code", "densite", title,maximum=250)


def carte_gravite_moyenne(table, feature, target, title):
    # Fonction permettant d'afficher la carte de la gravité moyenne d'accident par département
    # On récupère une la position des départements  français
    departement = formatage_geojson("departements.geojson")
    gravite = table[[target, feature]].groupby(feature).mean().reset_index()
    gravite.rename(columns={feature: "code"}, inplace=True)
    carto(gravite, departement, "code", "grav", title)

