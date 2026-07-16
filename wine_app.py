import os
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from google import genai

# O 503 "high demand" atinge qualquer modelo de forma intermitente, entao a lista
# abaixo e fallback, nao um modelo "imune". Retry e o que realmente resolve.
GEMINI_MODELS = ('gemini-2.5-flash-lite', 'gemini-2.5-flash')
MAX_RETRIES = 3  # por modelo; pior caso = 6 tentativas, ~6s de espera total

# 1. Page configuration
st.set_page_config(page_title="Wine Score Predictor", page_icon="🍷", layout="centered")

st.title("🍷 Wine Quality Regressor")
st.write("Enter the wine characteristics below to predict its continuous quality score.")

# 2. Load the trained REGRESSION model
@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent / 'wine_model.pkl')

model = load_model()

# 3. Create user inputs (Only the 11 original features)
st.subheader("Wine Features")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", min_value=4.6, max_value=15.9, value=8.3, step=0.1)
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.12, max_value=1.58, value=0.52, step=0.01)
    citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.27, step=0.01)
    residual_sugar = st.number_input("Residual Sugar", min_value=0.9, max_value=15.5, value=2.5, step=0.1)
    chlorides = st.number_input("Chlorides", min_value=0.012, max_value=0.611, value=0.087, step=0.001)
    free_sulfur = st.number_input("Free Sulfur Dioxide", min_value=1.0, max_value=72.0, value=15.0, step=1.0)

with col2:
    total_sulfur = st.number_input("Total Sulfur Dioxide", min_value=6.0, max_value=289.0, value=46.0, step=1.0)
    density = st.number_input("Density", min_value=0.99007, max_value=1.00369, value=0.99675, format="%.5f", step=0.0001)
    ph = st.number_input("pH", min_value=2.74, max_value=4.01, value=3.31, step=0.01)
    sulphates = st.number_input("Sulphates", min_value=0.33, max_value=2.0, value=0.65, step=0.01)
    alcohol = st.number_input("Alcohol (%)", min_value=8.4, max_value=14.9, value=10.4, step=0.1)

# 4. Automatic Feature Engineering (Must match the training DataFrame exactly)
acidity_ratio = volatile_acidity / fixed_acidity
sulfur_ratio = free_sulfur / total_sulfur
alcohol_density = alcohol / density

# 5. Organize into the exact column sequence your model trained on (14 features)
input_data = pd.DataFrame([{
    'fixed acidity': fixed_acidity,
    'volatile acidity': volatile_acidity,
    'citric acid': citric_acid,
    'residual sugar': residual_sugar,
    'chlorides': chlorides,
    'free sulfur dioxide': free_sulfur,
    'total sulfur dioxide': total_sulfur,
    'density': density,
    'pH': ph,
    'sulphates': sulphates,
    'alcohol': alcohol,
    'acidity_ratio': acidity_ratio,
    'sulfur_ratio': sulfur_ratio,
    'alcohol_density': alcohol_density
}])

# 6. Prediction Button
if st.button("📈 Predict Wine Score", use_container_width=True, type="primary"):
    predicted_score = model.predict(input_data)[0] # Pegando o valor float bruto
    
    st.markdown("---")
    status_col, metric_col = st.columns(2)
    
    with metric_col:
        st.metric(label="Predicted Quality", value=f"{predicted_score:.2f} / 10.0")
    
    with status_col:
        st.markdown("### 🍷 Chemical Assessment")
        if predicted_score <= 5.0:
            st.markdown("<p style='color:#B22222; font-weight:bold; font-size:18px;'>This wine is predicted to have Below Average quality.</p>", unsafe_allow_html=True)
        elif predicted_score <= 6.5:
            st.markdown("<p style='color:#D2691E; font-weight:bold; font-size:18px;'>This wine is predicted to have Average/Medium quality.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#006400; font-weight:bold; font-size:18px;'>This wine is predicted to have Premium/Good quality!</p>", unsafe_allow_html=True)

    # 7. SHAP Explainer
    st.markdown("---")
    st.subheader("🔍 Top 3 Prediction Influencers")

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(input_data)
    shap_values = booster.predict(dmatrix, pred_contribs=True)[0] # Pegando a primeira linha
    
    feature_contribs = shap_values[:-1]
    
    impact_df = pd.DataFrame({
        'Feature': input_data.columns,
        'Impact': feature_contribs,
        'Absolute_Impact': np.abs(feature_contribs)
    })
    
    top_3 = impact_df.sort_values(by='Absolute_Impact', ascending=False).head(3)
    
    # Criar uma string limpa dos top influenciadores para passar pro Gemini
    influencers_text = ""
    for _, row in top_3.iterrows():
        direction = "pushed the score UP" if row['Impact'] > 0 else "dragged the score DOWN"
        influencers_text += f"- {row['Feature']}: {direction} by {row['Impact']:.2f}\n"

    # Plotly Rendering (Invertendo a ordem apenas para exibição visual do gráfico)
    top_3_plot = top_3.iloc[::-1]
    colors = ['#A62B2B' if val < 0 else '#00CC96' for val in top_3_plot['Impact']]
    
    fig = go.Figure(go.Bar(
        x=top_3_plot['Impact'],
        y=[col.title().replace('_', ' ') for col in top_3_plot['Feature']],
        orientation='h',
        marker_color=colors,
        text=[f" {val:+.2f}" for val in top_3_plot['Impact']],
        textposition='outside'
    ))
    
    fig.update_layout(
        xaxis_title="Impact on Score", yaxis_title="",
        margin=dict(l=20, r=40, t=10, b=10), height=220,
        showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#2C1A1A")
    )
    fig.add_shape(type="line", x0=0, y0=-0.5, x1=0, y1=2.5, line=dict(color="gray", width=1, dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

    # 8. NEW: AI Sommelier Review (Chamada do Gemini)
    st.markdown("---")
    st.subheader("🤖 AI Sommelier Review")
    
    # Verifica se a chave de API está configurada
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Modo de teste local (caso você tenha rodado export GEMINI_API_KEY="sua_chave" no terminal)
        api_key = os.environ.get("GEMINI_API_KEY")


    if not api_key:
        st.warning("⚠️ Please configure your GEMINI_API_KEY in streamlit secrets or environment variables to see the Sommelier review.")
    else:
        with st.spinner("The AI Sommelier is tasting and evaluating the wine..."):
            # Inicializa o cliente com o novo SDK do Google GenAI
            client = genai.Client(api_key=api_key)


            # Engenharia de prompt detalhada mesclando dados químicos e SHAP
            prompt = f"""
            You are an elite, sarcastic, poetic, yet technically brilliant Wine Sommelier.
            A machine learning model just predicted a wine quality score of {predicted_score:.2f} out of 10.0.
                
            Here are the chemical features of this wine:
            {input_data.to_dict(orient='records')[0]}
                
            According to our SHAP values, the top 3 mathematical drivers that dictated this score are:
            {influencers_text}
                
            Write a brilliant, short sommelier review (3-4 sentences max) explaining WHY the wine got this score based on these chemical realities. 
            - Adopt a sophisticated, storytelling tone.
            - If the score is low, explain poetically how those negative drivers ruined the harmony (e.g., high volatile acidity masking the fruit, or low alcohol making it thin).
            - If the score is high, praise how the chemistry created a masterpiece.
            - Directly mention at least 1 or 2 of the top influencers.
            - Write the final review in Portuguese.
            """

            # O 503 (high demand) e intermitente em qualquer modelo, entao retenta
            # cada um ate MAX_RETRIES antes de cair pro proximo da lista.
            review, last_error = None, None
            for model_id in GEMINI_MODELS:
                for attempt in range(MAX_RETRIES):
                    try:
                        response = client.models.generate_content(
                            model=model_id,
                            contents=prompt
                        )
                        review = response.text
                        break
                    except Exception as e:
                        last_error = e
                        # Nao espera depois da ultima tentativa: 1s, 2s, desiste.
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(attempt + 1)
                if review:
                    break

            if review:
                st.info(review)
            else:
                st.error(f"Could not generate AI review. Error: {last_error}")
