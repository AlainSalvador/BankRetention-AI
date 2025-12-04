import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px


#Configuración de la página
st.set_page_config(
    page_title="BankRetention AI",
    page_icon="🏦",
    layout="wide"
)

# 2. Cargar el modelo (Usamos cache para que no lo cargue cada vez que tocas un botón)
@st.cache_resource
def load_model():
    model = joblib.load('C:/Users/dassl/Desktop/PythonSamsung/Módulo IA/BankRetention_Project/Models/model_churn.pkl')
    return model

@st.cache_data
def load_data():
    data = pd.read_csv('C:/Users/dassl/Desktop/PythonSamsung/Módulo IA/BankRetention_Project/Data/Churn_Modelling.csv')
    return data

try:
    model = load_model()
    df_raw = load_data()
except FileNotFoundError:
    st.error("Error: No se encontró el archivo 'model_churn.pkl'. Verifica la carpeta 'Models'.")
    st.stop()

#Título y Descripción
st.title("🏦 BankRetention AI: Simulador de Fuga de Clientes")
st.markdown("""
Esta herramienta utiliza Inteligencia Artificial para predecir la probabilidad de que un cliente abandone el banco.
Modifica los valores en el panel lateral para simular diferentes escenarios.
""")

tab1, tab2 = st.tabs(["Dashboard General", "Simulador de riesgos"])

with tab1:
    st.header("Situación actual del Banco")

    total_clientes = len(df_raw)
    num_fugas = len(df_raw[df_raw['Exited'] == 1])
    tasa_fuga = (num_fugas / total_clientes) * 100
    dinero_fuga = df_raw[df_raw['Exited'] == 1]['Balance'].sum()

    #Filas de metricas
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("Total clientes", f"{total_clientes:,}")
    col_kpi2.metric("Tasa de fuga (Churn rate)", f"{tasa_fuga:.2f}%", "-Objetivo < 15%")
    col_kpi3.metric("Capital de riesgo (Perdido)", f"{dinero_fuga:,.0f}")

    st.markdown("---")

    c1, c2, = st.columns(2)

    with c1:
        st.subheader("Distribucion de fuga")
        fig_pie = px.pie(df_raw, names='Exited', title='Clientes Retenidos (0) vs Fugados (1)', 
                         color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Fuga por Pais")
        fig_bar = px.histogram(df_raw, x="Geography", color="Exited", barmode="group",
                               color_discrete_sequence=['#2ecc71', '#e74c3c'],
                               title="Comparativa por Región")
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.header("Predicción Individual por cliente")
    
    # Panel Lateral (Inputs del Usuario)
    st.sidebar.header("Datos del Cliente")

    with st.sidebar:
        def user_input_features():
            # Variables numéricas
            credit_score = st.sidebar.slider('Puntaje de Crédito (Credit Score)', 300, 850, 600)
            age = st.sidebar.slider('Edad', 18, 92, 40)
            tenure = st.sidebar.slider('Tenencia (Años en el banco)', 0, 10, 3)
            balance = st.sidebar.number_input('Saldo en Cuenta ($)', min_value=0.0, value=60000.0)
            estimated_salary = st.sidebar.number_input('Salario Estimado ($)', min_value=0.0, value=50000.0)
            num_of_products = st.sidebar.slider('Número de Productos', 1, 4, 2)
    
            # Variables categóricas (Deben coincidir con las opciones del dataset original)
            geography = st.sidebar.selectbox('País', ('France', 'Germany', 'Spain'))
            gender = st.sidebar.selectbox('Género', ('Female', 'Male'))
            has_cr_card = st.sidebar.selectbox('¿Tiene Tarjeta de Crédito?', (1, 0), format_func=lambda x: 'Sí' if x == 1 else 'No')
            is_active_member = st.sidebar.selectbox('¿Es Miembro Activo?', (1, 0), format_func=lambda x: 'Sí' if x == 1 else 'No')

            # Crear el DataFrame con los nombres EXACTOS de las columnas que usó el modelo
            data = {
                'CreditScore': credit_score,
                'Geography': geography,
                'Gender': gender,
                'Age': age,
                'Tenure': tenure,
                'Balance': balance,
                'NumOfProducts': num_of_products,
                'HasCrCard': has_cr_card,
                'IsActiveMember': is_active_member,
                'EstimatedSalary': estimated_salary
            }
            features = pd.DataFrame(data, index=[0])
            return features

        input_df = user_input_features()

    #Panel Principal - Predicción
    st.subheader("Análisis de Riesgo en Tiempo Real")

    # Crear dos columnas para mostrar resultados
    col1, col2 = st.columns([2, 1])

    with col1:
        # Obtener predicción (0 o 1) y probabilidad (0% a 100%)
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        churn_probability = prediction_proba[0][1] # Probabilidad de que sea 1 (Churn)

    # Gráfico de Velocímetro (Gauge Chart)
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = churn_probability * 100,
        title = {'text': "Probabilidad de Fuga (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': churn_probability * 100}
        }
    ))
    st.plotly_chart(fig)

    with col2:
        st.write("### Diagnóstico:")
        
        if churn_probability > 0.7:
            st.error("🔴 **ALTO RIESGO**")
            st.write("El cliente tiene una alta probabilidad de abandonar el banco.")
            st.markdown("**Acción recomendada:** Contactar inmediatamente y ofrecer incentivos de retención.")
        elif churn_probability > 0.3:
            st.warning("🟡 **RIESGO MEDIO**")
            st.write("El cliente está en zona de incertidumbre.")
            st.markdown("**Acción recomendada:** Monitorear actividad y mejorar la satisfacción.")
        else:
            st.success("🟢 **CLIENTE SEGURO**")
        st.write("El cliente parece leal y estable.")

    #Mostrar los datos ingresados (Para depuración/confirmación)
    with st.expander("Ver datos crudos del cliente"):
        st.write(input_df)

    st.markdown("---")
    st.subheader("📊 ¿Qué factores están influyendo más en este modelo?")

    # Extraer la importancia de las variables del modelo
    importances = model.named_steps['classifier'].feature_importances_

    #Obtener los nombres de las columnas
    feature_names = ['Score Crédito', 'País', 'Género', 'Edad', 'Tenencia', 'Balance', 'Num Productos', 'Tarjeta Crédito', 'Miembro Activo', 'Salario']

    feature_imp_df = pd.DataFrame({
        'Factor': feature_names[:len(importances)], # Ajuste seguro
        'Importancia': importances[:len(feature_names)]
    }).sort_values(by='Importancia', ascending=True)

    #Crear el gráfico de barras horizontal
    fig_imp = go.Figure(go.Bar(
        x=feature_imp_df['Importancia'],
        y=feature_imp_df['Factor'],
        orientation='h',
        marker=dict(color='rgba(50, 171, 96, 0.6)', line=dict(color='rgba(50, 171, 96, 1.0)', width=1))
    ))

    fig_imp.update_layout(
        title="Peso de cada variable en la decisión de la IA",
        xaxis_title="Nivel de Importancia",
        yaxis_title="Factor", 
        height=400
    )

    st.plotly_chart(fig_imp, use_container_width=True)
    st.info("Utilizar barra lateral izquierda para modificar los datos del cliente")