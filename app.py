#!pip install streamlit
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle


# Configuração da página
st.set_page_config(page_title = 'Global Solution',
                   page_icon = './images/logo_fiap.png',
                   layout='wide',
                   initial_sidebar_state='expanded')

st.title('Global Solution 01 - Front End & Mobile Development')
st.write('Grupo: Caio Salesi, Gustavo Agostinho, Leonardo Sisilio')
with st.sidebar:
    st.write('Com esse app você pode simular o ambiente de um recife de corais para prever os riscos do branqueamento, um efeito devastador que ocorre quando as microalgas simbióticas responsáveis pelas cores e pela energia vital dos corais, são expulsas devido ao estresse ambiental, deixando-os pálidos e vulneráveis.')
    st.write('')
    st.write('')
    database = st.radio('Fonte dos dados de entrada (X):', ('CSV', 'Online'))
    if database == 'CSV':
        st.info('Upload do CSV')
        file = st.file_uploader('Selecione o arquivo CSV', type='csv')

# Modelo
# Carregando o modelo escolhido
# Descompactando porque o arquivo era muito pesado para o GitHub e Streamlit Cloud
import zipfile
with zipfile.ZipFile('rf_regressor_model.zip', 'r') as f:
    f.extractall()

with open('rf_regressor_model.pkl', 'rb') as f:
    rf_regressor_model = pickle.load(f)


# Caso o usuário escolha inserir dados manualmente
if database == 'Online':
    st.title('Insira os dados manualmente')

    # Colunas pra preenchimento das features
    col_ocean_exposure_distance, col_turbidity_cyclone_depth_temperature, col_windspeed_ssta_tsa = st.columns(3)
    
    with col_ocean_exposure_distance:
        ocean = st.selectbox('Oceano', ['Ocean_Arabian Gulf', 'Ocean_Atlantic', 'Ocean_Indian', 'Ocean_Pacific', 'Ocean_Red Sea'])
        exposure          = st.selectbox('Exposure', ['Exposure_Exposed', 'Exposure_Sheltered', 'Exposure_Sometimes'])
        distance_to_shore = st.number_input('Distance to Shore', min_value=0.0, max_value=50000.00, format="%.2f")

    with col_turbidity_cyclone_depth_temperature:
        turbidity    = st.number_input('Turbidity', min_value=0.0, max_value=100.00, format="%.4f")
        cyclone_freq = st.number_input('Cyclone Frequency', min_value=0.0, max_value=1000.00, format="%.2f")
        depth        = st.number_input('Depth (m)', min_value=0.0, max_value=100.00, format="%.2f")
        temperature  = st.number_input('Temperature (Kelvin)', min_value=0.0, max_value=500.00, format="%.2f")

    with col_windspeed_ssta_tsa:
        windspeed = st.number_input('Windspeed (m/h)', min_value=0.0, max_value=50.00, format="%.1f")
        ssta      = st.number_input('SSTA', max_value=50.00, format="%.2f")
        tsa       = st.number_input('TSA', max_value=50.00, format="%.2f")

    st.subheader('Threshold')
    threshold = st.slider('Escolha o threshold', min_value=1, max_value=100, step=1, value=50)

    st.subheader('Fazer a predição')
    input_data = pd.DataFrame({
        # ocean     (ajustado depois)
        # exposure  (ajustado depois)
        'Distance_to_Shore': [distance_to_shore],
        'Turbidity': [turbidity],
        'Cyclone_Frequency': [cyclone_freq],
        'Depth_m': [depth],
        'Temperature_Kelvin': [temperature],
        'Windspeed': [windspeed],
        'SSTA': [ssta],
        'TSA': [tsa]
    })

    # Ajustando 'Ocean' e 'Exposure' por ter OneHotEncoding
    ocean_labels = ['Ocean_Arabian Gulf', 'Ocean_Atlantic', 'Ocean_Indian', 'Ocean_Pacific', 'Ocean_Red Sea']
    for ocean_option in ocean_labels:
        input_data[ocean_option] = 1 if ocean_option == ocean else 0

    exposure_labels = ['Exposure_Exposed', 'Exposure_Sheltered', 'Exposure_Sometimes']
    for exposure_option in exposure_labels:
        input_data[exposure_option] = 1 if exposure_option == exposure else 0

    if st.button('Prever'):      
        ypred = rf_regressor_model.predict(input_data)
        # Verificando se a porcentagem prevista ultrapassa o threshold
        y_pred_binary = (ypred > threshold).astype(int)
        st.subheader('Resultado:')
        if y_pred_binary == 1:
            st.error('Corais nesse ambiente provavelmente estão sofrendo com branqueamento :(')
        else:
            st.success('Corais nesse ambiente provavelmente estão saudáveis!')

elif database == 'CSV':
    if file:
        X_test = pd.read_csv(file)
        ypred = rf_regressor_model.predict(X_test)

        slider_treshold = st.slider('Threshold',
                            min_value = 1, max_value = 100, step = 1, value = 50)
        qtd_over_threshold = sum(1 for pred in ypred if pred > slider_treshold)

        # Resultados
        col_unhealthy, col_healthy = st.columns(2)
        with col_unhealthy:
            st.metric('Corais nesse ambiente provavelmente estão sofrendo com branqueamento :(', value = qtd_over_threshold)
        with col_healthy:
            st.metric('Corais nesse ambiente provavelmente estão saudáveis!', value = len(ypred) - qtd_over_threshold)

    else:
        st.warning('Arquivo CSV não foi enviado')