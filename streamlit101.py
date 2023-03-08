import streamlit as st
import pandas as pd
import numpy as np

st.title('Accidents routiers en France ')
st.header('Rapport final')
st.write('contenu du rapport')
st.caption('style de caption')

st.write('mon premier DF dans streamlit')
df=pd.DataFrame({
    'Col1': [1,2,3,4,5],
    'Col2': [10,20,30,40,50]}
    )
st.write(df)

st.sidebar.write('Type de curseurs')
x=st.sidebar.slider('x',0,100)
st.write('x= ',x)


st.sidebar.write('Type de cases à cocher')
if st.sidebar.checkbox('Cases à cocher'):
    st.balloons()

nom=st.sidebar.text_input('Nom:')
date_jour=st.sidebar.date_input('Date:')
st.write(nom)
st.write(date_jour)
nombre= st.number_input('Entrez un nombre entre 10 et 80',10,80)
opt=st.selectbox('Choisissez une ligne:',[1,2,3,4,5])
st.sidebar.write('Vous avez choisi la ligne n°',opt, 'et entré le nombre: ',nombre)

# Presentations: en 3 colonnes:
gauche, centre, droit=st.columns(3)
gauche.write('Je suis à gauche ----->')
gauche.write('Je suis à gauche ----->')
gauche.write('Je suis à gauche ----->')

with centre:
    st.write("<---- J'écris au centre --->")
    st.write("<---- J'écris au centre --->")
    st.write("<---- J'écris au centre --->")

droit.write('<----- Je suis à droite')
droit.write('<----- Je suis à droite')
droit.write('<----- Je suis à droite')


# Presentations en Onglets:
tab1, tab2, tab3 =st.tabs(['Onglet 1', 'Onglet 2', 'Onglet 3'])
with tab1:
    st.header('Mon 1er Onglet')
    st.write("Scénario du Ministre de l'économie")

with tab2:
    st.header('Mon 2ème Onglet')
    st.write("Scénario du Ministre du travail")
    
with tab3:
    st.header('Mon 3ème Onglet')
    st.write("Avis de  l'expert")
    

#  Barre de progression
import time
der_iter=st.empty()
bar=st.progress(0)
for i in range(100):
    der_iter.text(f'Iteration {i+1}')
    bar.progress(i+1)
    time.sleep(0.1)



