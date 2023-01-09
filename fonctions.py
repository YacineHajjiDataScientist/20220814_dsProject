from sklearn.metrics import (
    roc_curve,
    auc,
    recall_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def courbe_roc(model, X_test, y_test):
    y_probas2 = model.predict_proba(X_test)
    y_probas3 = y_probas2[:, 1]
    fpr1, tpr1, _ = roc_curve(y_test, y_probas3)
    roc_auc = auc(fpr1, tpr1)
    # Sortir liste des features avec coef en valeur absolue
    plt.title("Courbe ROC :")

    plt.plot(fpr1, tpr1, label="AUC = %0.2f" % roc_auc)

    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")


def get_optimized_cutoff(model, X_train, y_train, recall_objectif):
    model_proba_test = model.predict_proba(X_train)

    for borne in reversed(np.arange(0, 1, 0.01)):
        model_pred_test = np.where(model_proba_test[:, 1] >= borne, 1, 0)

        recall = recall_score(y_train, model_pred_test)
        if recall >= recall_objectif:
            print(classification_report(y_train, model_pred_test))
            return borne, model_pred_test


def recall_features(X_test, y_test, y_pred):
    
    liste_features = ['prof_2.0', 'prof_3.0',
       'planGrp_1.0', 'surf_2.0', 'surf_8.0', 'atm_2.0', 'atm_3.0', 'atm_5.0',
       'atm_7.0', 'atm_8.0', 'vospGrp_1.0', 'catv_EPD_exist_1',
       'catv_PL_exist_1', 'trajet_coursesPromenade_conductor_1',
       'sexe_male_conductor_1', 'sexe_female_conductor_1',
       'intGrp_Croisement circulaire', 'intGrp_Croisement de deux routes',
       'intGrp_Hors intersection', 'intGrp_Passage à niveau',
       'catv_train_exist_1', 'infra_3.0', 'infra_5.0', 'infra_7.0',
       'infra_9.0', 'catr_2.0', 'catr_3.0', 'catr_4.0', 'catr_9.0',
       'hourGrp_nuit', 'lum_2.0', 'lum_3.0', 'lum_5.0', 'circ_2.0', 'circ_3.0',
       'circ_4.0', 'nbvGrp_1', 'nbvGrp_2', 'nbvGrp_3', 'nbvGrp_4+',
       'catv_2_roues_exist_1', 'col_2.0', 'col_3.0', 'col_4.0', 'col_5.0',
       'col_6.0', 'col_7.0', "obsGrp_Pas d'Obstacle", 'situ_2.0', 'situ_3.0',
       'situ_4.0', 'situ_6.0', 'situ_8.0', 'populationGrp_Grande Ville',
       'populationGrp_Métropole', 'populationGrp_Petite Ville',
       'populationGrp_Village', 'populationGrp_Ville Moyenne',
       'mois_label_aug', 'mois_label_dec', 'mois_label_fev', 'mois_label_jan',
       'mois_label_jul', 'mois_label_mar', 'mois_label_oct',
       'etatpGrp_pieton_alone_1', 'locpGrp_pieton_3_1']
    X_test["reel"] = y_test
    X_test["pred"] = y_pred
    liste_resultats = []
    for feature in liste_features :
        vrais_positifs = X_test[feature][(X_test.reel == 1) & (X_test.pred == 1)].sum()
        faux_positifs = X_test[feature][(X_test.reel == 0) & (X_test.pred == 1)].sum()
        vrais_negatifs = X_test[feature][(X_test.reel == 0) & (X_test.pred == 0)].sum()
        faux_negatifs = X_test[feature][(X_test.reel == 1) & (X_test.pred == 0)].sum()
        recall_feature = vrais_positifs / (vrais_positifs+faux_negatifs)
        liste_resultats.append({'feature' : feature, "recall" : recall_feature})
    
    df_results = pd.DataFrame(liste_resultats).sort_values(by='recall', ascending = False)
    return df_results
        