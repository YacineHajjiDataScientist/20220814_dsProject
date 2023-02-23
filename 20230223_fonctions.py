from sklearn.metrics import (
    roc_curve, auc,
    recall_score, precision_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



def courbe_roc(model, X_test, y_test):
    probs = model.predict_proba(X_test)
    
    fpr1, tpr1, seuils = roc_curve(y_test, probs[:, 1], pos_label=1)
    roc_auc = auc(fpr1, tpr1)
    # Sortir liste des features avec coef en valeur absolue
    plt.title("Courbe ROC :")

    plt.plot(fpr1, tpr1, label="AUC = %0.2f" % roc_auc)

    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate.")
    plt.xlabel("False Positive Rate")
    plt.annotate('High cutoff (0.73)', xy=(1-0.92, 0.40), xytext=(1-0.84, 0.39), arrowprops={'facecolor' : '#76D680'})




def get_optimized_cutoff(model, X_train, y_train, precision_objectif):
    model_proba_test = model.predict_proba(X_train)

    for borne in np.arange(0, 1, 0.01):
        model_pred_test = np.where(model_proba_test[:, 1] >= borne, 1, 0)

        precision = precision_score(y_train, model_pred_test)

        # on suppose que la precision est monotone pour avoir l'optimum
        if precision >= precision_objectif:
            print(classification_report(y_train, model_pred_test))
            return borne, model_pred_test


def recall_features(X_test, y_test, y_pred):
    
    liste_features = ['atm_7.0', 'catv_EPD_exist_1', 'catv_train_exist_1', 'infra_7.0',
       'catr_2.0', 'catr_3.0', 'lum_3.0', 'nbvGrp_4+', 'col_2.0', 'col_4.0',
       'situ_6.0', 'populationGrp_Grande Ville', 'populationGrp_Métropole',
       'populationGrp_Petite Ville', 'populationGrp_Ville Moyenne']

    # liste_features = X_test.columns
    X_test["reel"] = y_test
    X_test["pred"] = y_pred
    liste_resultats = []
    for feature in liste_features :
        vrais_positifs = X_test[feature][(X_test.reel == 1) & (X_test.pred == 1)].sum()
        faux_positifs  = X_test[feature][(X_test.reel == 0) & (X_test.pred == 1)].sum()
        vrais_negatifs = X_test[feature][(X_test.reel == 0) & (X_test.pred == 0)].sum()
        faux_negatifs  = X_test[feature][(X_test.reel == 1) & (X_test.pred == 0)].sum()
        total          = vrais_positifs+vrais_negatifs+faux_positifs+faux_negatifs

# Le ministre de l'économie cherche à limiter les interventions inutiles
# Donc maximise le recall_feature

        recall_feature    = vrais_positifs / (vrais_positifs+faux_negatifs)
        precision_feature = vrais_positifs / (vrais_positifs+faux_positifs)
        accuracy_feature  = (vrais_negatifs +vrais_positifs)/ total

        liste_resultats.append({'feature'   : feature, 
                                'precision' : precision_feature,
                                'recall'    : recall_feature,
                                'accuracy'  : accuracy_feature })
    
    df_results = pd.DataFrame(liste_resultats).sort_values(by='precision', ascending = False)
    return df_results
        