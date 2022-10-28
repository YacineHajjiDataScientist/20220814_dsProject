import pandas as pd
import numpy as np
import datetime
##### Import of tables into dataframes
dfLieux = pd.read_csv('20220906_table_lieux.csv', sep=',')
dfUsagers = pd.read_csv('20220906_table_usagers.csv', sep=',')
dfVehicules = pd.read_csv('20220906_table_vehicules.csv', sep=',')
dfCarac = pd.read_csv('20220906_table_caracteristiques.csv', sep=',')

dfLieux = dfLieux.drop(["Unnamed: 0.1","Unnamed: 0", "voie", "v1", "v2","pr", "pr1"], axis = 1)
dfUsagers = dfUsagers.drop(["Unnamed: 0.1","Unnamed: 0"], axis = 1)
dfVehicules = dfVehicules.drop(["Unnamed: 0.1","Unnamed: 0"], axis = 1)
dfCarac = dfCarac.drop(["Unnamed: 0.1","Unnamed: 0", "lat", "long", "gps", "adr"], axis = 1)

##### Additional dataframes
dfJoursFeriesMetropole = pd.read_csv('20221009_table_joursFeriesMetropole.csv', sep=';')
dfCommunes = pd.read_csv("pop_commune.csv", sep=";")

######### ############

# Computing date variable
dfCarac['date'] = dfCarac['jour'].astype(str) + '-' + dfCarac['mois'].astype(str) + '-' + dfCarac['an'].astype(str)
dfCarac['date'] = pd.to_datetime(dfCarac['date']);

# Computing months with categorical labels
dfCarac['mois_label'] = dfCarac['mois']
dfCarac['mois_label'] = dfCarac['mois_label'].replace(to_replace=np.arange(1, 13, 1), 
                                                      value=['jan', 'fev', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
# Days of the week
dfCarac['weekday'] = dfCarac['date'].dt.weekday

# Hour of the day
dfCarac['hrmn'] = dfCarac['hrmn'].replace('\:', '', regex=True).astype(int)
dfCarac['hour'] = dfCarac['hrmn']//100

# Year of accident
dfCarac['year'] = dfCarac['date'].dt.year

# Adding the year variable to dfUsagers dataframe
dfUsagers = dfUsagers.merge(right=dfCarac[['Num_Acc', 'year']], on='Num_Acc')

# Age of people during the accident (removing ages above 99, could be completion issues and there are very few values)
dfUsagers['age'] = dfUsagers.year - dfUsagers.an_nais
dfUsagers.loc[dfUsagers['age'] > 99, 'age'] = np.nan

# Largeur de la route assignée au trafic
dfLieux.larrout = dfLieux.larrout.replace('\,', '.', regex=True).astype('float64')
dfLieux.lartpc = dfLieux.lartpc.replace('\,', '.', regex=True).astype('float64')

### dfCarac
# hourGrp: nuit (22h - 6h) - jour heures creuses (10h-16h) - jour heures de pointe (7-9h, 17-21h)
hourConditions = [((dfCarac["hour"]>=22) | (dfCarac["hour"]<=6)),
                  (((dfCarac["hour"]>=7) & (dfCarac["hour"]<=9)) | ((dfCarac["hour"]>=17) & (dfCarac["hour"]<=21))),
                  ((dfCarac["hour"]>=10) | (dfCarac["hour"]<=16))]
hourChoices = ["nuit", "heure de pointe", "journee"]
dfCarac["hourGrp"] = np.select(hourConditions, hourChoices)
# atm: passer en NA les valeurs -1 et 9 (other) qui sont difficilement interprétables dans un modèle de ML
dfCarac['atm'] = dfCarac['atm'].replace([-1, 9], [np.nan, np.nan])
# Date feriée/weekend/feriée ou weekend
dateFerie = list(map(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y').strftime('%Y-%m-%d'), dfJoursFeriesMetropole['date']))
dfDateFerie = pd.DataFrame({'dateFerie': dateFerie})
dfCarac['dateFerie'] = np.where((dfCarac.date.isin(dfDateFerie.dateFerie)), 1, 0)
dfCarac['dateWeekend'] = np.where((dfCarac.weekday>=5), 1, 0)
dfCarac['dateFerieAndWeekend'] = np.where((dfCarac.date.isin(dfDateFerie.dateFerie) | (dfCarac.weekday>=5)), 1, 0)

### dfLieux
# nbvGrp: 0/1/2/3/4+, avec -1 et 9+ en NA
nbvConditions = [((dfLieux["nbv"]>=9) | (dfLieux["nbv"]==-1)),
                (dfLieux["nbv"]==0),
                (dfLieux["nbv"]==1),
                (dfLieux["nbv"]==2),
                (dfLieux["nbv"]==3),
                (dfLieux["nbv"]>=4),]
nbvChoices = [np.nan, '0', '1', '2', '3', '4+']
dfLieux['nbvGrp'] = np.select(nbvConditions, nbvChoices)
# vostGrp: présence yes/no d'une voie réservée
dfLieux['vospGrp'] = dfLieux['vosp'].replace([-1, 0, 1, 2, 3], [np.nan, 0, 1, 1, 1])
# profGrp: -1 et 0 en NA
dfLieux['prof'] = dfLieux['prof'].replace([-1, 0], [np.nan, np.nan])
# circ: -1 et 0 en NA
dfLieux['circ'] = dfLieux['circ'].replace([-1, 0], [np.nan, np.nan])


# planGrp: en binaire not straight vs straight (yes/no), les -1 et 0 en NA
dfLieux['planGrp'] = dfLieux['plan'].replace([-1, 0, 1, 2, 3, 4], [np.nan, np.nan, 0, 1, 1, 1])
# lartpcGrp: 0/1/2/3/4+, avec -1 et 9+ en NA
lartpcConditions = [((dfLieux["lartpc"]==0.0)),
                    ((dfLieux["lartpc"]>=20)),
                    ((dfLieux["lartpc"]>0) & (dfLieux["lartpc"]<5)),
                    ((dfLieux["lartpc"]>=5) & (dfLieux["lartpc"]<10)),
                    ((dfLieux["lartpc"]>=10) & (dfLieux["lartpc"]<15)),
                    ((dfLieux["lartpc"]>=15) & (dfLieux["lartpc"]<20))]
lartpcChoices = [np.nan, np.nan, 1, 2, 3, 4]
dfLieux['lartpcGrp'] = np.select(lartpcConditions, lartpcChoices)
dfLieux['lartpcGrp'] = dfLieux['lartpcGrp'].replace([0, 1, 2, 3, 4], [np.nan, '0-5', '5-10', '10-15', '15-20'])
# larroutGrp: 0/1/2/3/4+, avec -1 et 9+ en NA
larroutConditions = [((dfLieux["larrout"]==0.0)),
                    ((dfLieux["larrout"]>=200)),
                    ((dfLieux["larrout"]>0) & (dfLieux["larrout"]<50)),
                    ((dfLieux["larrout"]>=50) & (dfLieux["larrout"]<100)),
                    ((dfLieux["larrout"]>=100) & (dfLieux["larrout"]<150)),
                    ((dfLieux["larrout"]>=150) & (dfLieux["larrout"]<200))]
larroutChoices = [np.nan, np.nan, 1, 2, 3, 4]
dfLieux['larroutGrp'] = np.select(larroutConditions, larroutChoices)
dfLieux['larroutGrp'] = dfLieux['larroutGrp'].replace([0, 1, 2, 3, 4], [np.nan, '0-50', '50-100', '100-150', '150-200'])

# surf: transformation des -1, 0 et 9 en  NA
dfLieux['surf'] = dfLieux['surf'].replace([-1, 0, 9], [np.nan, np.nan, np.nan])
# situ: transformation des -1, 0 et 9 en  NA
dfLieux['situ'] = dfLieux['situ'].replace([-1, 0], [np.nan, np.nan])
# infra: transformation des -1,en  NA
dfLieux['infra'] = dfLieux['infra'].replace([-1], [np.nan])

### dfUsagers
# Does a gravity of type X exist for an accident
dfUsagers['grav34exists'] = np.where(dfUsagers.grav2>=3, 1, 0)
dfUsagers['grav4exists'] = np.where(dfUsagers.grav2==4, 1, 0)
dfUsagers['grav3exists'] = np.where(dfUsagers.grav2==3, 1, 0)
dfUsagers['grav2exists'] = np.where(dfUsagers.grav2==2, 1, 0)
dfUsagers['place'] = dfUsagers['place'].replace([0], [np.nan])
## actp: harmonization des valeurs et transformation des -1 en NA
dfUsagers['actp'] = dfUsagers['actp'].replace({'0.0':0, '0':0, 0:0,
                                              '-1.0':np.nan, '-1':np.nan, ' -1':np.nan, -1:np.nan,
                                              '1.0':1, '1':1, 1:1,
                                              '2.0':2, '2':2, 2:2,
                                              '3.0':3, '3':3, 3:3,
                                              '4.0':4, '4':4, 4:4,
                                              '5.0':5, '5':5, 5:5,
                                              '6.0':6, '6':6, 6:6,
                                              '7.0':7, '7':7, 7:7,
                                              '8.0':8, '8':8, 8:8,
                                              '9.0':9, '9':9, 9:9
                                              })
## etatp: transformation des -1 en NA et nombre de piétons seuls dans l'accident
dfUsagers['etatp'] = dfUsagers['etatp'].replace([-1], [np.nan])
dfUsagers['etatp_pieton_alone_exists'] = np.where((dfUsagers['etatp']==1), 1, 0)
## locp: transformation des 0 en NA et nombre de piétons en fonction de leur position pendant l'accident
dfUsagers['locp'] = dfUsagers['locp'].replace([-1], [np.nan])
dfUsagers['locp_pieton_1_exists'] = np.where(((dfUsagers.locp==1)), 1, 0)
dfUsagers['locp_pieton_3_exists'] = np.where(((dfUsagers.locp==3)), 1, 0)
dfUsagers['locp_pieton_6_exists'] = np.where(((dfUsagers.locp==6)), 1, 0)
# Number of pietons in catu variable
dfUsagers['catu_pieton_exists'] = np.where(((dfUsagers.catu==3) | (dfUsagers.catu==4)), 1, 0)
dfUsagers['catu_conductor_exists'] = np.where(((dfUsagers.catu==1)), 1, 0)

# Number of men/women conductor
dfUsagers['sexe_male_conductor_exists'] = np.where(((dfUsagers.sexe==1) & (dfUsagers.catu==1)), 1, 0)
dfUsagers['sexe_female_conductor_exists'] = np.where(((dfUsagers.sexe==2) & (dfUsagers.catu==1)), 1, 0)
# Number of conductor going to courses/promenade (3 & 5)
dfUsagers['trajet_coursesPromenade_conductor_exists'] = np.where((((dfUsagers.trajet==3) & (dfUsagers.catu==1)) | 
                                                           ((dfUsagers.trajet==5) & (dfUsagers.catu==1))), 1, 0)
## Mean age of conductors and nonCoductors by accident
# Preliminary dataFrames with mean age of Conductors/nonConductors by accident
dfAgeMeanConductors = dfUsagers[(dfUsagers['catu_conductor_exists']==1)][['Num_Acc', 'age']].groupby(['Num_Acc']).mean().rename({'age':'ageMeanConductors'}, axis=1)
dfAgeMeanNonConductors = dfUsagers[(dfUsagers['catu_conductor_exists']==0)][['Num_Acc', 'age']].groupby(['Num_Acc']).mean().rename({'age':'ageMeanNonConductors'}, axis=1)
# New variable 'Num_Acc' for merging
dfAgeMeanConductors['Num_Acc'] = dfAgeMeanConductors.index
dfAgeMeanNonConductors['Num_Acc'] = dfAgeMeanNonConductors.index
# Change index so there is no ambiguity while merging
dfAgeMeanConductors.index = np.arange(1, len(dfAgeMeanConductors) + 1)
dfAgeMeanNonConductors.index = np.arange(1, len(dfAgeMeanNonConductors) + 1)
# Merging new variables
dfUsagers = dfUsagers.merge(right=dfAgeMeanConductors, how='left', on='Num_Acc')
dfUsagers = dfUsagers.merge(right=dfAgeMeanNonConductors, how='left', on='Num_Acc')
# Computeing all variables as 'is there at least one of'
dfAtLeastOneByAccident = pd.DataFrame({
                                      # event exists yes/no by accident
              'Num_Acc':  dfUsagers.groupby('Num_Acc')['grav4exists'].sum().index, 
              'gravGrp_23_4': np.where(dfUsagers.groupby('Num_Acc')['grav4exists'].sum()>=1, 1, 0), 
              'gravGrp_2_34': np.where(dfUsagers.groupby('Num_Acc')['grav34exists'].sum()>=1, 1, 0), 
              'catu_pieton': np.where(dfUsagers.groupby('Num_Acc')['catu_pieton_exists'].sum()>=1, 1, 0), 
              'sexe_male_conductor': np.where(dfUsagers.groupby('Num_Acc')['sexe_male_conductor_exists'].sum()>=1, 1, 0), 
              'sexe_female_conductor': np.where(dfUsagers.groupby('Num_Acc')['sexe_female_conductor_exists'].sum()>=1, 1, 0), 
              'trajet_coursesPromenade_conductor': np.where(dfUsagers.groupby('Num_Acc')['trajet_coursesPromenade_conductor_exists'].sum()>=1, 1, 0), 
                    
                                       # count event variable by accident
              'nb_grav4_by_acc': dfUsagers.groupby('Num_Acc')['grav4exists'].sum(),
              'nb_grav3_by_acc': dfUsagers.groupby('Num_Acc')['grav3exists'].sum(), 
              'nb_catu_pieton': dfUsagers.groupby('Num_Acc')['catu_pieton_exists'].sum(), 
              'nb_sexe_male_conductor': dfUsagers.groupby('Num_Acc')['sexe_male_conductor_exists'].sum(), 
              'nb_sexe_female_conductor': dfUsagers.groupby('Num_Acc')['sexe_female_conductor_exists'].sum(), 
              'nb_trajet_coursesPromenade_conductor': dfUsagers.groupby('Num_Acc')['trajet_coursesPromenade_conductor_exists'].sum(),
              'nb_etatpGrp_pieton_alone': dfUsagers.groupby('Num_Acc')['etatp_pieton_alone_exists'].sum(), 
              'nb_locpGrp_pieton_1': dfUsagers.groupby('Num_Acc')['locp_pieton_1_exists'].sum(), 
              'nb_locpGrp_pieton_3': dfUsagers.groupby('Num_Acc')['locp_pieton_3_exists'].sum(), 
              'nb_locpGrp_pieton_6': dfUsagers.groupby('Num_Acc')['locp_pieton_6_exists'].sum(), 
    
                                        # mean of variable by accident
              'ageMeanConductors': dfUsagers.groupby('Num_Acc')['ageMeanConductors'].mean(), 
              'ageMeanNonConductors': dfUsagers.groupby('Num_Acc')['ageMeanNonConductors'].mean()})

### Change index so there is no ambiguity while merging
dfAtLeastOneByAccident.index = np.arange(1, len(dfAtLeastOneByAccident) + 1)

# Création des nouvelles variables pour l'étude de la gravité des accidents de la route 

# Définition fonction de mise au format département
def mise_au_format_dep(df) : 
    df["dep"] = df["dep"].astype(str)
    df["dep"] = df["dep"].apply(lambda x: x[:2] if (len(x) == 3 and x[2] == "0") else x)
    df["dep"] = df["dep"].apply(lambda x: x[1:] if (x[0] == "0") else x)
    df["dep"] = df["dep"].apply(lambda x: "0" + x if (len(x) == 1) else x)
    df["dep"] = df["dep"].replace("201", "2A")
    df["dep"] = df["dep"].replace("202", "2B")

# Application aux tables dfCarac et dfCommunes
mise_au_format_dep(dfCarac)
mise_au_format_dep(dfCommunes)

# Travail sur la colonne "com" de la table dfCarac pour la mettre au format
dfCarac["com"] = pd.to_numeric(dfCarac["com"], errors= "coerce")

dfCarac["com"] = dfCarac["com"].apply(lambda x: x % 100 if x > 1000 else x)
dfCommunes["com"] = pd.to_numeric(dfCommunes["com"], errors= "coerce")
dfCommunes = dfCommunes.drop(["code_region", "code_arrondissement", "code_canton", "population_mun", 'population_part'], axis = 1)
dfCommunes = dfCommunes.drop_duplicates(subset= ["dep", "com"], keep = 'first')
dfCarac_BPA = dfCarac.merge(dfCommunes, on = ["dep", "com"], how = "left")
dfCarac_BPA['population_tot'] = dfCarac_BPA['population_tot'].str.replace(" ", "")
dfCarac_BPA['population_tot'] = pd.to_numeric(dfCarac_BPA['population_tot'])
# lum: transformation des -1,en  NA
dfCarac['lum'] = dfCarac['lum'].replace([-1], [np.nan])
# col: transformation des -1,en  NA
dfCarac['col'] = dfCarac['col'].replace([-1], [np.nan])

def regroupement_population(x) :
    if x < 2000 :
        return 'Village'
    elif x < 5000 :
        return 'Bourg'
    elif x < 20000 :
        return 'Petite Ville'
    elif x < 50000 :
        return 'Ville Moyenne'
    elif x < 200000 :
        return 'Grande Ville'
    else :
        return "Métropole"
    
dfCarac_BPA['populationGrp'] = dfCarac_BPA['population_tot'].apply(lambda x: regroupement_population(x))
def regroupement_intersection(x) :
    if x == 1 :
        return "Hors intersection"
    elif x in [2,3,4] :
        return "Croisement de deux routes"
    elif x in [5,6,7] :
        return "Croisement circulaire" 
    elif x == 8 :
        return "Passage à niveau"
    else : 
        return "Autres"

dfCarac_BPA["intGrp"] = dfCarac_BPA.int.apply(regroupement_intersection)
# On garde la liste des variables suivantes 
# On filtre les accidents qui ne se sont pas produits à Bornel, Betz ou Auneuil qui sont dans le TOP30 des villes les plus accidentées 
# peu crédible

dfCarac_filtre = dfCarac_BPA[(dfCarac_BPA.nom_commune != "Bornel") & (dfCarac_BPA.nom_commune != "Betz") & (dfCarac_BPA.nom_commune != "Auneuil")]
variable_conservees = ["Num_Acc", "population_tot", "populationGrp", "intGrp"]
dfCarac_filtre = dfCarac_filtre[variable_conservees]

# Travaux sur la table véhicules

def regroupement_cat_veh(x) :
    cat1 = [1, 2, 4, 5, 6, 30, 31, 32, 33,34, 80, 35, 36, 41, 42, 43] # 2, 3 roues et quads
    cat2 = [3, 7, 8, 9, 10, 11, 12] # VL et VUL
    cat3 = [13, 14, 15, 16, 17, 20, 21, 37, 38, 18] # PL 
    cat4 = [39] # Trains
    cat5 = [40, 19, 99, 0] # Tramways
    cat9 = [50,60]
    if x in cat1 :
        return "2,3 roues & quads"
    elif x in cat2 :
        return "VL & VUL"
    elif x in cat3 :
        return "PL"
    elif x in cat4 :
        return "Train"
    elif x in cat5:
        return "Autres"
    elif x in cat9 :
        return "EPD"
    
dfVehicules["catvGrp"] = dfVehicules.catv.apply(regroupement_cat_veh)

def regroupement_obstacles(x) :
    if x == 0 :
        return "Pas d'Obstacle"
    elif x == - 1 :
        return "Z"
    else : 
        return "Obstacle" 

dfVehicules["obsGrp"] = dfVehicules.obs.apply(regroupement_obstacles)
dfVehicules['obsGrp'] = dfVehicules['obsGrp'].replace(["Z"], [np.nan])

dfVehicules_obs = dfVehicules.sort_values(by = 'obs', ascending = True)
dfVehicules_obs = dfVehicules_obs.drop_duplicates(subset= ["Num_Acc"], keep= 'first')
dfVehicules_obs_filtre = dfVehicules_obs[['Num_Acc', "obsGrp"]]
# Construction de la variable nombre de véhicules
dfNombreVehicule = dfVehicules[['Num_Acc',"num_veh"]].groupby(['Num_Acc']).count().reset_index()
dfNombreVehicule = dfNombreVehicule.sort_values(by='num_veh', ascending = False) 

def regroupement_nb_vehicules(x) :
    if x == 1 :
        return "1 véhicule"
    elif x == 2 :
        return "2 véhicules"
    elif x == 3 :
        return "3 véhicules"
    elif x > 3 and x < 10 :
        return "entre 4 et 10 véhicules"
    else : 
        return "+ de 10 véhicules" 

dfNombreVehicule["nbVeh"] = dfNombreVehicule.num_veh.apply(regroupement_nb_vehicules)
dfNombreVehicule = dfNombreVehicule.drop("nbVeh", axis = 1)
dfVehicules['catv_train_exist'] = np.where(dfVehicules.catvGrp =="Train", 1,0)
dfVehicules['catv_PL_exist'] = np.where(dfVehicules.catvGrp =="PL", 1,0)
dfVehicules['catv_2_roues_exist'] = np.where(dfVehicules.catvGrp =="2,3 roues & quads", 1,0)
dfVehicules['catv_EPD_exist'] = np.where(dfVehicules.catvGrp =="EPD", 1,0)

dfVeh_type_veh = dfVehicules[['Num_Acc', 'catv_train_exist', "catv_PL_exist", "catv_2_roues_exist", "catv_EPD_exist"]].groupby("Num_Acc").sum().reset_index()

dfVeh_type_veh["catv_train_exist"] = dfVeh_type_veh["catv_train_exist"].apply(lambda x : 1 if x >=1 else 0)
dfVeh_type_veh["catv_PL_exist"] = dfVeh_type_veh["catv_PL_exist"].apply(lambda x : 1 if x >=1 else 0)
dfVeh_type_veh["catv_2_roues_exist"] = dfVeh_type_veh["catv_2_roues_exist"].apply(lambda x : 1 if x >=1 else 0)
dfVeh_type_veh["catv_EPD_exist"] = dfVeh_type_veh["catv_EPD_exist"].apply(lambda x : 1 if x >=1 else 0)


dfMerge = dfUsagers.merge(dfVehicules, how='left', on= ['Num_Acc', 'num_veh'])

def fonction_choc_personne(x) :
    if x.choc in (1,2,3) and x.place in (1,6,2):
        return "Avant"
    elif x.choc in (3,6,8) and x.place in (1,7,4):
        return "Gauche"
    elif x.choc in (2,5,7) and x.place in (3,9,2):
        return "Droite"
    elif x.choc in (4,5,6) and x.place in (4,5,3):
        return "Arrière"
    else :
        return "Z"
    
dfMerge["choc_place"] = dfMerge.apply(lambda x : fonction_choc_personne(x), axis = 1)
def cat(x) :
    if x != "Z" :
        return 1
    else :
        return 0

dfMerge['choc_cote'] =dfMerge['choc_place'].apply(cat) 
dfTemp = dfMerge[['Num_Acc', 'choc_cote']].groupby(['Num_Acc' ]).sum().reset_index()


df_pool_var_BPA = dfVeh_type_veh.merge(dfNombreVehicule, on = "Num_Acc")
df_pool_var_BPA = df_pool_var_BPA.merge(dfVehicules_obs_filtre, on = "Num_Acc")
df_pool_var_BPA = df_pool_var_BPA.merge(dfCarac_filtre, on = "Num_Acc", how='left')
df_pool_var_BPA = df_pool_var_BPA.merge(dfTemp, on = "Num_Acc", how='left')

##### Merging of tables into 1 pooled dataframe post-DataManagement (2 steps required)
dfLieux = dfLieux.drop(['nbv', 'vosp','plan', 'lartpc','larrout','vma'], axis = 1)
dfCarac = dfCarac.drop(['an','hrmn'], axis = 1)

dfPoolPostDataManagementTemp = pd.merge(dfLieux, dfCarac, on="Num_Acc")
dfPoolPostDataManagement = pd.merge(dfPoolPostDataManagementTemp, dfAtLeastOneByAccident, on="Num_Acc")
dfPoolPostDataManagement = pd.merge(dfPoolPostDataManagement, df_pool_var_BPA, on="Num_Acc", how= 'left')

dfPoolPostDataManagement.to_csv("20221024_table_poolPostDataManagement_YAH_BPA.csv")