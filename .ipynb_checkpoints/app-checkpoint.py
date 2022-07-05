import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
from sklearn import svm
import streamlit as st


# Path del modelo preentrenado
MODEL_PATH = 'models/pickle_model.pkl'


# Se recibe la imagen y el modelo, devuelve la predicción
def model_prediction(x_in, model):

    x = np.asarray(x_in).reshape(1,-1)
    preds=model.predict(x)

    return preds


def main():
    
    model=''

    # Se carga el modelo
    if model=='':
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
    
    # Título
    html_temp = """
    <h1 style="color:#181082;text-align:center;">Nivel de Conocimiento de una Materia </h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    # Lecctura de datos
    #Datos = st.text_input("Ingrese los valores : N P K Temp Hum pH lluvia:")
    STG = st.text_input("Asistencia de la Materia:")
    SCG = st.text_input("Participación en la Materia:")
    STR = st.text_input("Cumplimiento de actividades:")
    LPR = st.text_input("Rendimiento en las practicas:")
    PEG = st.text_input("Rendimiento en el examen:")
    
    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción :"): 
        #x_in = list(np.float_((Datos.title().split('\t'))))
        x_in =[np.float_(STG.title()),
                    np.float_(SCG.title()),
                    np.float_(STR.title()),
                    np.float_(LPR.title()),
                    np.float_(PEG.title())]
        predictS = model_prediction(x_in, model)
        st.success('Su Nivel de Conocimiento es: {}'.format(predictS[0]).upper())

if __name__ == '__main__':
    main()
