import streamlit as st
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
import numpy as np
import pandas as pd

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

@st.cache
def get_data():
    billets = pd.read_csv ('data/billets.csv', delimiter=';')
    return billets

    

with header:
    st.title('Projet 9: Détecter des faux billets avec Python')
    st.text("Cette institution a pour objectif de mettre en place des méthodes d’identification des contrefaçons des billets en euros. Ils font donc appel à vous, spécialiste de la data, pour mettre en place une modélisation qui serait capable d’identifier automatiquement les vrais des faux billets. Et ce à partir simplement de certaines dimensions du billet et des éléments qui le composent.")



with dataset:
    st.header('Dataset')
    billets = get_data()
    billets = billets.dropna(axis=0, thresh=7)
    st.write(billets.head(5))

    


with features:
    st.header('User Input Parameters')

    def user_input_features():
        diagonal = st.sidebar.slider('diagonal',171.04, 173.01, 171.81)	
        height_left	= st.sidebar.slider('height_left', 103.14, 104.88,104.86)
        height_right = st.sidebar.slider('height_right', 102.82, 104.95, 104.95)	
        margin_low = st.sidebar.slider('margin_low', 2.98, 6.9, 4.52)	
        margin_up = st.sidebar.slider('margin_up', 2.27  , 3.91, 2.89)	
        length = st.sidebar.slider('length', 109.49, 114.44, 112.83)
        data = {'diagonal': diagonal,
                'height_left': height_left,
                'height_right': height_right,
                'margin_low': margin_low,
                'margin_up': margin_up,
                'length': length}
        features = pd.DataFrame(data, index=[0])
        return features
    

    df = user_input_features()
    st.write(df)

with model_training:
    st.header("Prediction via regression logistique")

    x = billets[['diagonal','height_left','height_right','margin_low','margin_up','length']]
    y = billets['is_genuine']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Create a Logistic Regression Object, perform Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    y_pred = log_reg.predict(x_test)
    st.write(log_reg.predict(df))

    st.subheader('mesure de performance du modèle via une Confusion Matrix')
    st.write(confusion_matrix(y_test, y_pred))
