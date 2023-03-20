import streamlit as st
from joblib import dump, load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# from sklearn.model_selection import train_test_split
# from fonctions import plot_density_model

model_xgboost = load("optimal_xgb.joblib")

# feature_matrix = pd.read_pickle('./import/20230225_table_feature_matrix.csv')
# target = pd.read_pickle('./import/20230225_table_target.csv')
# X_train, X_test, y_train, y_test = train_test_split(feature_matrix, target, test_size=0.2, random_state=1)

# st.header('Rapport final - Démo Streamlit')
st.title("Accidents Routiers en France")
st.write(
    """
Le Président de la République Française vient de prendre ses fonctions. Il souhaiterait mettre en place une politique volontariste 
en matière de Sécurité Routière. \n
Pour ce faire, il s'appuie sur 2 de ses ministres qui défendent leur intérêts: le ministre des Transports et le ministre des Finances. \n
Il fait également appel à un Expert Data dont l'objectif est de chercher un compromis entre les visions des ministres.
"""
)


# Compo de la SIDE BARRE
st.sidebar.title("Variables du modèle")

population = st.sidebar.selectbox(
    "Taille de la ville",
    ["Village", "Petite Ville", "Ville Moyenne", "Grande Ville", "Métropole", "Autre"],
)
categorie_route = st.sidebar.selectbox(
    "Type de route",
    [
        "Autres",
        "Nationales",
        "Départementales",
        "Communales",
    ],
)
conditions_atmospheriques = st.sidebar.selectbox(
    "Conditions Météo",
    ["Pluie légère", "Pluie forte", "Brouillard", "Eblouissant", "Couvert", "Autre"],
)
nb_vehicule = st.sidebar.slider("Nombre de véhicules", 0, 10, 2)
luminosité = st.sidebar.selectbox(
    "Luminosité",
    [
        "crépuscule ou aube",
        "nuit sans éclairage public",
        "nuit avec éclairage public allumé",
        "Autre",
    ],
)

# Conteneur des variables non visibles
with st.sidebar.expander("Autres variables du modèle", expanded=False):

    age_moyen_conducteurs = st.slider("Age Moyen des conducteurs", 18, 100, 30)
    nb_personnes_cote_choc = st.slider("Nombre de personnes du côté du choc", 0, 10, 1)

    regime_circulation = st.selectbox(
        "Régime de Circulation",
        [
            "Autre",
            "Bidirectionnelle",
            "avec Chaussées Séparées",
            "avec Voies d'affectation variable ",
        ],
    )
    type_collision = st.selectbox(
        "Type de collision",
        [
            "Autre",
            "3 véhicules et + - multiples",
            "2 véhicules - arrière",
            "2 véhicules - côté",
            "3 véhicules et + - en chaîne ",
            "sans collision",
        ],
    )
    infra = st.selectbox(
        "Infrastructure",
        [
            "Autre",
            "bretelle d'échangeur",
            "carrefour aménagé",
            "zone de péage",
        ],
    )
    intersection = st.selectbox(
        "Type d'intersection",
        [
            "Autre",
            "passage à niveau" "croisement circulaire",
            "croisement de 2 routes",
            "hors intersection",
        ],
    )

    mois = st.selectbox(
        "Mois",
        [
            "mars",
            "janvier",
            "février",
            "juillet",
            "août",
            "octobre",
            "décembre",
            "Autre",
        ],
    )

    profil = st.selectbox("Profil de la route", ["Pente", "Sommet de côte", "Autre"])
    situation = st.selectbox(
        "Situation de l'accident",
        [
            "Autres",
            "sur bande d'arrêt d'urgence",
            "sur accôtement",
            "sur trottoir",
            "sur autre voie spéciale",
        ],
    )
    surface = st.selectbox(
        "Surface de la route",
        ["Autres", "présence d'un corps gras - huile", "mouillée"],
    )
    pres_2roues = st.checkbox("présence d'un deux roues")
    pres_EPD = st.checkbox("présence d'un engin personnel de déplacement")
    pres_PL = st.checkbox("présence d'un poids lourd")
    pres_train = st.checkbox("présence d'un train")
    pres_pieton = st.checkbox("présence d'un piéton seul")
    abs_obstacle = st.checkbox("Absence d'obstacles")
    loc_pieton = st.checkbox("Piéton sur passage piéton sans signalisation lumineuse")
    nuit = st.checkbox("Accident de nuit")
    route_rectiligne = st.checkbox("Route rectiligne")
    pres_homme_volant = st.checkbox("Présence d'un homme au volant")
    pres_femme_volant = st.checkbox("Présence d'une femme au volant")
    trajet_promenade = st.checkbox("Le but du trajet était une promenade")
    pres_piste_cyclabe = st.checkbox("présence d'une piste cyclable")
    # ... Fin du conteneur

result = st.button("Effectuer la prédiction")
if result:
    X_test = pd.DataFrame(
        columns=[
            "choc_cote",
            "ageMeanConductors",
            "nbVeh",
            "prof_2.0",
            "prof_3.0",
            "planGrp_1.0",
            "surf_2.0",
            "surf_8.0",
            "atm_2.0",
            "atm_3.0",
            "atm_5.0",
            "atm_7.0",
            "atm_8.0",
            "vospGrp_1.0",
            "catv_EPD_exist_1",
            "catv_PL_exist_1",
            "trajet_coursesPromenade_conductor_1",
            "sexe_male_conductor_1",
            "sexe_female_conductor_1",
            "intGrp_Croisement circulaire",
            "intGrp_Croisement de deux routes",
            "intGrp_Hors intersection",
            "intGrp_Passage à niveau",
            "catv_train_exist_1",
            "infra_3.0",
            "infra_5.0",
            "infra_7.0",
            "infra_9.0",
            "catr_2.0",
            "catr_3.0",
            "catr_4.0",
            "catr_9.0",
            "hourGrp_nuit",
            "lum_2.0",
            "lum_3.0",
            "lum_5.0",
            "circ_2.0",
            "circ_3.0",
            "circ_4.0",
            "nbvGrp_1",
            "nbvGrp_2",
            "nbvGrp_3",
            "nbvGrp_4+",
            "catv_2_roues_exist_1",
            "col_2.0",
            "col_3.0",
            "col_4.0",
            "col_5.0",
            "col_6.0",
            "col_7.0",
            "obsGrp_Pas d'Obstacle",
            "situ_2.0",
            "situ_3.0",
            "situ_4.0",
            "situ_6.0",
            "situ_8.0",
            "populationGrp_Grande Ville",
            "populationGrp_Métropole",
            "populationGrp_Petite Ville",
            "populationGrp_Village",
            "populationGrp_Ville Moyenne",
            "mois_label_aug",
            "mois_label_dec",
            "mois_label_fev",
            "mois_label_jan",
            "mois_label_jul",
            "mois_label_mar",
            "mois_label_oct",
            "etatpGrp_pieton_alone_1",
            "locpGrp_pieton_3_1",
        ],
        dtype="int",
    )

    X_test.loc[0, "choc_cote"] = nb_personnes_cote_choc
    X_test.loc[0, "ageMeanConductors"] = age_moyen_conducteurs
    X_test.loc[0, "nbVeh"] = nb_vehicule

    X_test.loc[0, "prof_2.0"] = 1 if profil == "Pente" else 0
    X_test.loc[0, "prof_3.0"] = 1 if profil == "Sommet de côte" else 0

    X_test.loc[0, "planGrp_1.0"] = 1 if route_rectiligne else 0

    X_test.loc[0, "surf_2.0"] = 1 if surface == "mouillée" else 0
    X_test.loc[0, "surf_8.0"] = (
        1 if surface == "présence d'un corps gras - huile" else 0
    )

    X_test.loc[0, "atm_2.0"] = 1 if conditions_atmospheriques == "Pluie légère" else 0
    X_test.loc[0, "atm_3.0"] = 1 if conditions_atmospheriques == "Pluie forte" else 0
    X_test.loc[0, "atm_5.0"] = 1 if conditions_atmospheriques == "Brouillard" else 0
    X_test.loc[0, "atm_7.0"] = 1 if conditions_atmospheriques == "Eblouissant" else 0
    X_test.loc[0, "atm_8.0"] = 1 if conditions_atmospheriques == "Couvert" else 0

    X_test.loc[0, "vospGrp_1.0"] = 1 if pres_piste_cyclabe else 0

    X_test.loc[0, "catv_EPD_exist_1"] = 1 if pres_EPD else 0
    X_test.loc[0, "catv_PL_exist_1"] = 1 if pres_PL else 0

    X_test.loc[0, "trajet_coursesPromenade_conductor_1"] = 1 if trajet_promenade else 0

    X_test.loc[0, "sexe_male_conductor_1"] = 1 if pres_homme_volant else 0
    X_test.loc[0, "sexe_female_conductor_1"] = 1 if pres_femme_volant else 0

    X_test.loc[0, "intGrp_Croisement circulaire"] = (
        1 if intersection == "croisement circulaire" else 0
    )
    X_test.loc[0, "intGrp_Croisement de deux routes"] = (
        1 if intersection == "croisement de 2 routes" else 0
    )
    X_test.loc[0, "intGrp_Hors intersection"] = (
        1 if intersection == "hors intersection" else 0
    )
    X_test.loc[0, "intGrp_Passage à niveau"] = (
        1 if intersection == "passage à niveau" else 0
    )

    X_test.loc[0, "catv_train_exist_1"] = 1 if pres_train else 0

    X_test.loc[0, "infra_3.0"] = 1 if infra == "bretelle d'échangeur" else 0
    X_test.loc[0, "infra_5.0"] = 1 if infra == "carrefour aménagé" else 0
    X_test.loc[0, "infra_7.0"] = 1 if infra == "zone de péage" else 0
    X_test.loc[0, "infra_9.0"] = 1 if infra == "Autre" else 0

    X_test.loc[0, "catr_2.0"] = 1 if categorie_route == "Nationales" else 0
    X_test.loc[0, "catr_3.0"] = 1 if categorie_route == "Départementales" else 0
    X_test.loc[0, "catr_4.0"] = 1 if categorie_route == "Communales" else 0
    X_test.loc[0, "catr_9.0"] = 1 if categorie_route == "Autre" else 0

    X_test.loc[0, "hourGrp_nuit"] = 1 if nuit else 0

    X_test.loc[0, "lum_2.0"] = 1 if luminosité == "crépuscule ou aube" else 0
    X_test.loc[0, "lum_3.0"] = 1 if luminosité == "nuit sans éclairage public" else 0
    X_test.loc[0, "lum_5.0"] = (
        1 if luminosité == "nuit avec éclairage public allumé" else 0
    )

    X_test.loc[0, "circ_2.0"] = 1 if regime_circulation == "Bidirectionnelle" else 0
    X_test.loc[0, "circ_3.0"] = (
        1 if regime_circulation == "avec Chaussées Séparées" else 0
    )
    X_test.loc[0, "circ_4.0"] = (
        1 if regime_circulation == "avec Voies d'affectation variable" else 0
    )

    X_test.loc[0, "nbvGrp_1"] = 1 if nb_vehicule == 1 else 0
    X_test.loc[0, "nbvGrp_2"] = 1 if nb_vehicule == 2 else 0
    X_test.loc[0, "nbvGrp_3"] = 1 if nb_vehicule == 3 else 0
    X_test.loc[0, "nbvGrp_4+"] = 1 if nb_vehicule >= 4 else 0

    X_test.loc[0, "catv_2_roues_exist_1"] = 1 if pres_2roues else 0

    X_test.loc[0, "col_2.0"] = 1 if type_collision == "2 véhicules - arrière" else 0
    X_test.loc[0, "col_3.0"] = 1 if type_collision == "2 véhicules - côté" else 0
    X_test.loc[0, "col_4.0"] = (
        1 if type_collision == "3 véhicules et + - en chaîne " else 0
    )
    X_test.loc[0, "col_5.0"] = (
        1 if type_collision == "3 véhicules et + - multiples" else 0
    )
    X_test.loc[0, "col_6.0"] = 1 if type_collision == "sans collision" else 0
    X_test.loc[0, "col_7.0"] = 1 if type_collision == "Autres" else 0

    X_test.loc[0, "obsGrp_Pas d'Obstacle"] = 1 if abs_obstacle else 0

    X_test.loc[0, "situ_2.0"] = 1 if situation == "sur bande d'arrêt d'urgence" else 0
    X_test.loc[0, "situ_3.0"] = 1 if situation == "sur accôtement" else 0
    X_test.loc[0, "situ_4.0"] = 1 if situation == "sur trottoir" else 0
    X_test.loc[0, "situ_6.0"] = 1 if situation == "sur autre voie spéciale" else 0
    X_test.loc[0, "situ_8.0"] = 1 if situation == "Autres" else 0

    X_test.loc[0, "populationGrp_Grande Ville"] = (
        1 if population == "Grande Ville" else 0
    )
    X_test.loc[0, "populationGrp_Métropole"] = 1 if population == "Métropole" else 0
    X_test.loc[0, "populationGrp_Petite Ville"] = (
        1 if population == "Petite Ville" else 0
    )
    X_test.loc[0, "populationGrp_Village"] = 1 if population == "Village" else 0
    X_test.loc[0, "populationGrp_Ville Moyenne"] = (
        1 if population == "Ville Moyenne" else 0
    )

    X_test.loc[0, "mois_label_aug"] = 1 if mois == "août" else 0
    X_test.loc[0, "mois_label_dec"] = 1 if mois == "décembre" else 0
    X_test.loc[0, "mois_label_fev"] = 1 if mois == "février" else 0
    X_test.loc[0, "mois_label_jan"] = 1 if mois == "janvier" else 0
    X_test.loc[0, "mois_label_jul"] = 1 if mois == "juillet" else 0
    X_test.loc[0, "mois_label_mar"] = 1 if mois == "mars" else 0
    X_test.loc[0, "mois_label_oct"] = 1 if mois == "octobre" else 0

    X_test.loc[0, "etatpGrp_pieton_alone_1"] = 1 if pres_pieton else 0

    X_test.loc[0, "locpGrp_pieton_3_1"] = 1 if loc_pieton else 0

    proba = model_xgboost.predict_proba(X_test)

    limits = [32, 43, 63, 100]
    data_to_plot = ("Probabilité de gravité", round(proba[0, 1] * 100, 0))
    palette = sns.color_palette("Blues", len(limits))
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_yticks([1])
    ax.set_yticklabels([data_to_plot[0]])

    prev_limit = 0
    for idx, lim in enumerate(limits):
        ax.barh([1], lim - prev_limit, left=prev_limit, height=15, color=palette[idx])
        prev_limit = lim

    ax.barh([1], data_to_plot[1], color="black", height=5)
    plt.annotate(
        "Ministre des \n Transports",
        xy=(32, 9),
        xytext=(27, 14),
        arrowprops={
            "facecolor": "blue",
            "linewidth": 0.5,
            "headwidth": 10,
            "headlength": 4,
        },
        size=6,
    )
    plt.annotate(
        "Expert \n Data",
        xy=(43, 9),
        xytext=(40, 14),
        arrowprops={
            "facecolor": "blue",
            "linewidth": 0.5,
            "headwidth": 10,
            "headlength": 4,
        },
        size=6,
    )
    plt.annotate(
        "Ministre de \n l'Economie",
        xy=(63, 9),
        xytext=(58, 14),
        arrowprops={
            "facecolor": "blue",
            "linewidth": 0.5,
            "headwidth": 10,
            "headlength": 4,
        },
        size=6,
    )
    plt.tight_layout()
    st.pyplot(fig=fig)


### Besoin dans le streamlit d'afficher les paramètres


def bases_streamlit():
    st.title("Demo streamlit : MC JUIN22CDS")
    # texte
    st.text("Ceci est du texte")
    # header \ subheader
    st.header("This is a Header")
    st.subheader("This a Subheader")
    # MARKDOWN
    st.markdown("- This is a markdown")
    # Link
    st.markdown("[Cliquez ici pour accéder à google](https://google.com)")

    # Write HTML
    st.write("You can write HTML")
    html_page = """
    <div style="background-color:green;padding:100px">
        <p style="font-size:50px">Streamlit is very awesome</p>
    </div>
    """
    st.markdown(html_page, unsafe_allow_html=True)

    # html_form = """
    # <div>
    #    <form>
    #    <input type="text" name="firstname"/>
    #    </form>
    # </div>
    # """
    # st.markdown(html_form, unsafe_allow_html=True)

    # Alert text
    st.write("Alert text")
    st.success("Success!")
    st.info("Information")
    st.warning("Attention, ces données ne représent que l'année 2021")
    st.error("Une erreur")

    ## MEDIA
    # Image
    # import Image function
    st.write("ouverture d'une image:")
    # open an image
    # img = Image.open("OIP (2).jpeg")

    # Plot the image
    # st.image(img, caption="Logo DataScientest")

    # Audio
    # audio_file = open('name_of_file.ext', "rb")
    # audio_bytes = audio_file.read()
    # st.audio(audio_bytes, format="audio/mp3")

    # Video with URL
    st.subheader("une vidéo directement de YouTube:")
    st.video(data="https://www.youtube.com/watch?v=SNNK6z03TaA")

    ### WIDGET
    st.subheader("Let's talk about widgets")
    # Bouton
    st.button("Appuyez")

    # Other button

    result = st.button("press me please : bouton")
    if result:
        st.text("vous avez réussi top ! ")

    # getting interaction button
    if st.button("Press Me again"):
        st.success("this is a success!")
        st.error("Erreur")

    # Checkbox
    if st.checkbox("Hide & seek"):
        st.success("showing")

    # Radio
    gender_list = ["Man", "Woman"]
    gender = st.radio("Sélectionner un genre", gender_list)
    if gender == gender_list[0]:
        st.info("Man")
    else:
        st.info("Woman")

    # Select
    location = st.selectbox("Your Job", ["Data Scientist", "Dentist", "Doctor"])
    if location == "Data Scientist":
        st.info("Vous êtes Data Scientist")

    # Multiselect
    liste_course = st.multiselect(
        "liste de course", ["tomates", "dentifrice", "écouteurs"]
    )

    # Text imput
    name = st.text_input("your name", "your name here")
    st.text(name)
    st.text(name[:2])

    # Number input
    age = st.number_input("Age", 5, 100)

    # text area
    message = st.text_area("Enter your message")

    # Slider
    niveau = st.slider("select the level", 0, 100)

    st.write(niveau * 2)

    # Ballons
    if st.button("Press me again"):
        st.write("Yesss, you'r ready!")
        st.balloons()
