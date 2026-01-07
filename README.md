# BankRetention AI: Sistema Inteligente de Fidelización

> **Herramienta de soporte a la toma de decisiones para reducir la fuga de clientes bancarios (Churn) utilizando Machine Learning.**

---
## Descripción del Proyecto
**BankRetention AI** es una aplicación web interactiva diseñada para ayudar a las instituciones financieras a identificar proactivamente a los clientes en riesgo de abandonar el banco.

A diferencia de los reportes estáticos tradicionales, esta herramienta permite a los ejecutivos bancarios:
1.  **Diagnosticar** el estado actual de la cartera de clientes (KPIs).
2.  **Predecir** la probabilidad de fuga de un cliente específico en tiempo real.
3.  **Simular** estrategias de retención (ej. "¿Qué pasa si aumentamos la satisfacción o reducimos el número de productos?").

## Contexto de Negocio
Conseguir un cliente nuevo cuesta **5 veces más** que retener a uno existente. Los bancos pierden millones anualmente debido a fugas silenciosas que podrían prevenirse. Este proyecto utiliza datos históricos para detectar patrones complejos de comportamiento y alertar antes de que el cliente cierre su cuenta.

## Características Clave
* **Dashboard Gerencial:** Vista macro con métricas clave como Tasa de Fuga (Churn Rate), Capital en Riesgo y distribución geográfica.
* **Motor de IA (Random Forest):** Modelo predictivo entrenado con +10,000 registros históricos, alcanzando una precisión del ~86%.
* **Simulador Interactivo:** Interfaz para modificar variables del cliente (Edad, Saldo, Productos) y ver cómo cambia su riesgo en tiempo real.
* **Explicabilidad (XAI):** Gráficos de "Importancia de Características" que explican *por qué* la IA tomó esa decisión (ej. Edad, Nivel de Actividad).

## Tecnologías Utilizadas (Stack)
* **Lenguaje:** Python 3.10+
* **Machine Learning:** Scikit-Learn (Random Forest Classifier), Joblib (Persistencia).
* **Procesamiento de Datos:** Pandas, NumPy.
* **Visualización:** Plotly Express & Graph Objects (Gráficos interactivos).
* **Interfaz Web:** Streamlit.

## Estructura del Proyecto
```text
BankRetention_Project/
│
├── Data/                   # Dataset original (Churn_Modelling.csv)
├── Models/                 # Modelos entrenados (.pkl) y pipelines
├── Notebooks/              # Jupyter Notebooks para análisis exploratorio (EDA) y entrenamiento
├── venv/                   # Entorno virtual
│
├── app.py                  # Aplicación principal
├── requirements.txt        # Lista de dependencias del proyecto
└── README.md               # Documentación del proyecto
