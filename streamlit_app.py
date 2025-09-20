# app_recommender.py

import streamlit as st
import joblib
import pandas as pd
import os

# --- Título y descripción de la aplicación ---
st.title("📚 Sistema de Recomendación de Libros")
st.markdown("Ingresa un título de libro para obtener recomendaciones basadas en su contenido.")

# --- Cargar el modelo de recomendación optimizado y los datos ---
model_filename = 'book_recommender_model.joblib'
data_filename = 'book_data.joblib'

@st.cache_resource(show_spinner="Cargando modelo de recomendación...")
def load_model_data():
    """
    Carga el modelo de recomendación y los datos optimizados.
    Utiliza el caché de Streamlit para evitar recargar en cada interacción.
    """
    if not os.path.exists(model_filename) or not os.path.exists(data_filename):
        st.error(f"Error: No se encontraron los archivos '{model_filename}' y/o '{data_filename}'. "
                 "Asegúrate de que estén en el mismo directorio que este script.")
        return None, None, None

    try:
        sparse_sim_dict = joblib.load(model_filename)
        df = joblib.load(data_filename)
        # Mapeo de títulos a índices para una búsqueda rápida
        indices = pd.Series(df.index, index=df['title']).drop_duplicates()
        return sparse_sim_dict, df, indices
    except Exception as e:
        st.error(f"Error al cargar los archivos: {e}")
        return None, None, None

sparse_sim_dict, df, indices = load_model_data()

# --- Función para obtener las recomendaciones ---
def get_content_based_recommendations(title, sim_dict, data_frame, indices_series):
    """
    Devuelve las 5 mejores recomendaciones de libros a partir de un título.
    """
    if title not in indices_series:
        return f"El libro '{title}' no se encontró en la base de datos."

    # Obtener el índice del libro
    idx = indices_series[title]

    # Obtener las puntuaciones de similitud del diccionario pre-calculado
    sim_scores = sim_dict.get(idx, [])

    # Obtener los títulos de los libros recomendados
    recommended_titles = []
    for score in sim_scores:
        book_index = score[0]
        recommended_titles.append(data_frame.iloc[book_index]['title'])

    if not recommended_titles:
        return "No se encontraron libros similares para recomendar."
    
    return recommended_titles

# --- Interfaz de usuario principal ---
if sparse_sim_dict is not None and df is not None and indices is not None:
    # Obtener la lista de títulos para el autocompletado del usuario
    book_titles = df['title'].tolist()
    
    # Campo de entrada de texto
    user_input = st.selectbox(
        "Selecciona o escribe el título del libro:",
        options=book_titles,
        index=None,
        placeholder="Ej. 'The Lord of the Rings'"
    )

    if user_input:
        st.markdown("---")
        st.subheader("Resultados de la Recomendación")
        
        # Obtener las recomendaciones
        recommendations = get_content_based_recommendations(user_input, sparse_sim_dict, df, indices)
        
        if isinstance(recommendations, str):
            st.warning(recommendations)
        else:
            st.success(f"Hemos encontrado las siguientes recomendaciones para **'{user_input}'**:")
            for i, rec_title in enumerate(recommendations, 1):
                st.write(f"**{i}.** {rec_title}")

else:
    st.warning("No se pudo cargar el modelo. Por favor, asegúrate de que los archivos 'book_recommender_model.joblib' y 'book_data.joblib' estén en el directorio correcto.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
    **💡 Cómo funciona:**
    Este script carga un modelo de recomendación basado en contenido
    pre-calculado. Al ingresar el título de un libro, busca en el
    modelo los libros con características similares para
    ofrecerte sugerencias.
""")
