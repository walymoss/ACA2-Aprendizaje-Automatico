
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

st.set_page_config(
    page_title="Análisis del Rendimiento Académico",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# CARGA DE RECURSOS
# --------------------------------------------------

@st.cache_resource
def cargar_modelos():
    modelo = tf.keras.models.load_model(
        "modelo_rendimiento_academico.keras"
    )
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")

    return modelo, encoder, scaler


@st.cache_data
def cargar_datos():
    df = pd.read_csv("StudentsPerformance.csv")
    df["average_score"] = df[
        ["math score", "reading score", "writing score"]
    ].mean(axis=1)

    return df


modelo, encoder, scaler = cargar_modelos()
df = cargar_datos()

# --------------------------------------------------
# ENCABEZADO
# --------------------------------------------------

st.title("🎓 Análisis del Rendimiento Académico")

st.write(
    """
    Aplicación desarrollada mediante una red neuronal artificial
    de tipo Perceptrón Multicapa (MLP), para explorar los factores
    asociados al rendimiento académico y estimar promedios.
    """
)

st.caption(
    "Proyecto académico | Especialización en Inteligencia Artificial"
)

# --------------------------------------------------
# PESTAÑAS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "🔮 Predicción",
    "📊 Análisis exploratorio",
    "🧠 Evaluación del modelo"
])

# ==================================================
# TAB 1: PREDICCIÓN
# ==================================================

with tab1:

    st.subheader("Estimación del rendimiento académico")

    st.write(
        "Selecciona las características del estudiante "
        "para obtener una estimación de su promedio."
    )

    categorias = encoder.categories_

    with st.form("formulario_prediccion"):

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Género",
                categorias[0].tolist()
            )

            race = st.selectbox(
                "Grupo étnico",
                categorias[1].tolist()
            )

            education = st.selectbox(
                "Nivel educativo de los padres",
                categorias[2].tolist()
            )

        with col2:
            lunch = st.selectbox(
                "Tipo de almuerzo",
                categorias[3].tolist()
            )

            preparation = st.selectbox(
                "Curso de preparación",
                categorias[4].tolist()
            )

        submitted = st.form_submit_button(
            "Predecir rendimiento",
            use_container_width=True
        )

    if submitted:

        datos = pd.DataFrame([{
            "gender": gender,
            "race/ethnicity": race,
            "parental level of education": education,
            "lunch": lunch,
            "test preparation course": preparation
        }])

        datos_encoded = encoder.transform(datos)
        datos_scaled = scaler.transform(datos_encoded)

        prediccion = modelo.predict(
            datos_scaled,
            verbose=0
        )[0][0]

        st.divider()

        st.metric(
            "Promedio académico estimado",
            f"{prediccion:.2f} / 100"
        )

        st.warning(
            """
            Esta estimación es orientativa. Durante la evaluación,
            el modelo obtuvo un R² negativo, por lo que su capacidad
            predictiva es limitada. El resultado no debe interpretarse
            como una calificación real o una predicción confiable
            del desempeño individual.
            """
        )

# ==================================================
# TAB 2: ANÁLISIS EXPLORATORIO
# ==================================================

with tab2:

    st.subheader("Exploración de los datos")

    st.write(
        "Resumen estadístico y visualización de las variables "
        "del conjunto de datos."
    )

    # Indicadores generales

    col1, col2, col3 = st.columns(3)

    col1.metric("Estudiantes", f"{len(df):,}")
    col2.metric("Variables originales", "8")
    col3.metric(
        "Promedio general",
        f"{df['average_score'].mean():.2f}"
    )

    st.divider()

    # Distribución de promedios

    st.markdown("### Distribución del rendimiento académico")

    st.bar_chart(
        df["average_score"].value_counts(
            bins=10,
            sort=False
        ).sort_index()
    )

    # Comparación por preparación

    st.markdown("### Rendimiento según preparación para el examen")

    preparacion = (
        df.groupby("test preparation course")[
            ["math score", "reading score", "writing score"]
        ]
        .mean()
        .round(2)
    )

    st.bar_chart(preparacion)

    # Comparación por nivel educativo

    st.markdown("### Rendimiento según educación de los padres")

    educacion = (
        df.groupby("parental level of education")[
            "average_score"
        ]
        .mean()
        .sort_values(ascending=False)
        .round(2)
    )

    st.bar_chart(educacion)

    # Comparación por género

    st.markdown("### Rendimiento promedio por género")

    genero = (
        df.groupby("gender")["average_score"]
        .mean()
        .round(2)
    )

    st.bar_chart(genero)

    st.info(
        """
        Los gráficos presentan asociaciones descriptivas observadas
        en los datos. No demuestran relaciones causales entre las
        características y el rendimiento académico.
        """
    )

# ==================================================
# TAB 3: EVALUACIÓN DEL MODELO
# ==================================================

with tab3:

    st.subheader("Desempeño de la red neuronal")

    st.write(
        "Resultados obtenidos durante la evaluación con los "
        "200 registros del conjunto de prueba."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", "11.976")
    col2.metric("RMSE", "15.291")
    col3.metric("R²", "-0.091")

    st.divider()

    st.markdown("### Interpretación de las métricas")

    st.markdown(
        """
        - **MAE:** el error absoluto promedio es de aproximadamente
          11.98 puntos.
        - **RMSE:** el error cuadrático medio, expresado en la escala
          original, es de aproximadamente 15.29 puntos.
        - **R²:** el valor negativo indica que el modelo tiene un
          desempeño inferior al de una predicción basada en el
          promedio de los valores reales del conjunto de prueba.
        """
    )

    st.error(
        """
        El modelo presenta limitaciones para predecir el promedio
        académico a partir de las cinco variables seleccionadas.
        Los resultados se presentan de forma transparente y permiten
        identificar oportunidades de mejora.
        """
    )

    st.markdown("### Características de la arquitectura")

    arquitectura = pd.DataFrame({
        "Componente": [
            "Tipo de red",
            "Variables de entrada",
            "Primera capa oculta",
            "Segunda capa oculta",
            "Capa de salida",
            "Parámetros entrenables"
        ],
        "Configuración": [
            "Perceptrón Multicapa (MLP)",
            "17 características codificadas",
            "32 neuronas - ReLU",
            "16 neuronas - ReLU",
            "1 neurona - Lineal",
            "1.121"
        ]
    })

    st.dataframe(
        arquitectura,
        hide_index=True,
        use_container_width=True
    )

# --------------------------------------------------
# PIE DE PÁGINA
# --------------------------------------------------

st.divider()

st.caption(
    "Proyecto académico de Aprendizaje Automático | "
    "Modelo de predicción del rendimiento académico"
)
