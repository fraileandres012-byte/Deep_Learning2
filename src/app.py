import streamlit as st
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# Asegurar que podemos importar audio_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from audio_utils import load_audio_wave, wav_to_logmel_from_wave, pad_or_crop_mel, N_MELS, FIXED_TIME_FRAMES, gradcam_heatmap, overlay_gradcam_on_mel, mel_figure
except ImportError:
    # Fallback por si se ejecuta desde la raíz
    from src.audio_utils import load_audio_wave, wav_to_logmel_from_wave, pad_or_crop_mel, N_MELS, FIXED_TIME_FRAMES, gradcam_heatmap, overlay_gradcam_on_mel, mel_figure

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Fabrik AI | Audio Classifier",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DICCIONARIO DE IDIOMAS (i18n) ---
TRANSLATIONS = {
    "es": {
        "title": "Clasificador de Audio IA",
        "subtitle": "Prototipo de Clasificación de Géneros Electrónicos",
        "upload_label": "Sube tu archivo de audio (WAV, MP3, OGG)",
        "upload_help": "Arrastra y suelta un archivo de audio para analizarlo.",
        "tab_demo": "🚀 Demo en Vivo",
        "tab_tech": "🛠️ Análisis Técnico",
        "tab_biz": "💼 Impacto de Negocio",
        "model_status_ok": "Modelo cargado y listo para inferencia.",
        "model_status_err": "No se encontró el modelo (.h5). Por favor súbelo o colócalo en la carpeta models/.",
        "predicting": "Analizando espectrograma...",
        "result_label": "Género Predicho",
        "confidence_label": "Nivel de Confianza",
        "viz_title": "Interpretabilidad del Modelo",
        "gradcam_desc": "El mapa de calor muestra qué frecuencias y momentos activaron la neurona ganadora.",
        "biz_title": "Casos de Uso Corporativo",
        "biz_1": "📦 Catalogación Automática",
        "biz_1_desc": "Procesamiento masivo de librerías musicales sin intervención humana.",
        "biz_2": "©️ Detección de Copyright",
        "biz_2_desc": "Identificación de estilos protegidos o firmas sonoras específicas.",
        "biz_3": "🎧 Motores de Recomendación",
        "biz_3_desc": "Mejora del engagement mediante sugerencias precisas basadas en características de audio profundo.",
        "load_sample": "O carga un audio de ejemplo:",
        "sample_btn_1": "Cargar Sample Makina",
        "sample_btn_2": "Cargar Sample Newstyle"
    },
    "en": {
        "title": "AI Audio Classifier",
        "subtitle": "Electronic Genre Classification Prototype",
        "upload_label": "Upload your audio file (WAV, MP3, OGG)",
        "upload_help": "Drag and drop an audio file to analyze.",
        "tab_demo": "🚀 Live Demo",
        "tab_tech": "🛠️ Technical Analysis",
        "tab_biz": "💼 Business Impact",
        "model_status_ok": "Model loaded and ready for inference.",
        "model_status_err": "Model (.h5) not found. Please upload it or place it in models/ folder.",
        "predicting": "Analyzing spectrogram...",
        "result_label": "Predicted Genre",
        "confidence_label": "Confidence Level",
        "viz_title": "Model Interpretability",
        "gradcam_desc": "Heatmap shows which frequencies and timestamps triggered the winning neuron.",
        "biz_title": "Corporate Use Cases",
        "biz_1": "📦 Automated Tagging",
        "biz_1_desc": "Mass processing of music libraries without human intervention.",
        "biz_2": "©️ Copyright Detection",
        "biz_2_desc": "Identification of protected styles or specific sound signatures.",
        "biz_3": "🎧 Recommendation Engines",
        "biz_3_desc": "Improving user engagement via precise suggestions based on deep audio features.",
        "load_sample": "Or load a sample audio:",
        "sample_btn_1": "Load Makina Sample",
        "sample_btn_2": "Load Newstyle Sample"
    }
}

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
<style>
    .main-header {font-size: 3rem; font-weight: 800; color: #1E3A8A;}
    .sub-header {font-size: 1.5rem; color: #64748B;}
    .metric-card {background-color: #F1F5F9; padding: 20px; border-radius: 10px; border-left: 5px solid #3B82F6;}
    .biz-card {background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR & IDIOMA ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1256/1256650.png", width=50)
    st.title("Fabrik AI")
    
    lang_code = st.selectbox("Language / Idioma", ["en", "es"], format_func=lambda x: "🇬🇧 English" if x == "en" else "🇪🇸 Español")
    t = TRANSLATIONS[lang_code]
    
    st.divider()
    st.info("v2.4.0-beta (International Build)")

# --- FUNCIONES ---
@st.cache_resource
def load_model_cached(path):
    return tf.keras.models.load_model(path)

def get_model_path():
    # Busca en rutas comunes
    candidates = [
        "models/fabrik_makina_newstyle.h5",
        "fabrik_makina_newstyle.h5",
        "../models/fabrik_makina_newstyle.h5"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# --- INTERFAZ PRINCIPAL ---
st.markdown(f"<div class='main-header'>{t['title']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-header'>{t['subtitle']}</div>", unsafe_allow_html=True)

# Carga del Modelo
model_path = get_model_path()
model = None
if model_path:
    try:
        model = load_model_cached(model_path)
        st.toast(t['model_status_ok'], icon="✅")
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    # Si no encuentra el modelo, permite subirlo (útil para demos portátiles)
    uploaded_model = st.file_uploader("Upload Model (.h5)", type=["h5"])
    if uploaded_model:
        with open("temp_model.h5", "wb") as f:
            f.write(uploaded_model.getbuffer())
        model = load_model_cached("temp_model.h5")

# Tabs
tab1, tab2, tab3 = st.tabs([t['tab_demo'], t['tab_tech'], t['tab_biz']])

# --- TAB 1: LIVE DEMO ---
with tab1:
    col_input, col_results = st.columns([1, 2])
    
    with col_input:
        st.write("### Input")
        uploaded_file = st.file_uploader(t['upload_label'], type=["wav", "mp3", "ogg", "flac", "aiff"], help=t['upload_help'])
        
        st.write("---")
        st.write(t['load_sample'])
        # Simulamos samples generando ruido o tonos si no hay archivos reales
        if st.button(t['sample_btn_1']):
            st.session_state['mock_audio'] = 'makina'
        if st.button(t['sample_btn_2']):
            st.session_state['mock_audio'] = 'newstyle'

    with col_results:
        if uploaded_file is not None or 'mock_audio' in st.session_state:
            if model is None:
                st.warning(t['model_status_err'])
            else:
                # Guardar archivo temporal
                if uploaded_file:
                    with open("temp_audio.wav", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    path_to_analyze = "temp_audio.wav"
                else:
                    # Lógica para sample mock (en producción usarías archivos reales)
                    path_to_analyze = None 
                    st.info("⚠️ Running in simulation mode for sample button (no real audio file loaded). Upload a file for real inference.")

                if path_to_analyze:
                    with st.spinner(t['predicting']):
                        try:
                            # 1. Preprocesar
                            y, sr = load_audio_wave(path_to_analyze)
                            # Cortar a 10s para la red
                            mel = wav_to_logmel_from_wave(y, sr=sr, n_mels=N_MELS)
                            mel = pad_or_crop_mel(mel, FIXED_TIME_FRAMES)
                            # (1, 128, 431, 1)
                            X = mel[np.newaxis, ..., np.newaxis]

                            # 2. Predecir
                            probs = model.predict(X, verbose=0)[0]
                            LABELS = ["Makina", "Newstyle"]
                            pred_idx = np.argmax(probs)
                            pred_label = LABELS[pred_idx]
                            confidence = probs[pred_idx]

                            # 3. Mostrar KPIs
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div style="font-size:0.9rem; color:#666;">{t['result_label']}</div>
                                    <div style="font-size:2rem; font-weight:bold; color:#1E3A8A;">{pred_label}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with c2:
                                color = "green" if confidence > 0.8 else "orange"
                                st.markdown(f"""
                                <div class="metric-card" style="border-left: 5px solid {color};">
                                    <div style="font-size:0.9rem; color:#666;">{t['confidence_label']}</div>
                                    <div style="font-size:2rem; font-weight:bold; color:{color};">{confidence:.1%}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.progress(float(confidence))

                            # 4. Audio Player
                            st.audio(path_to_analyze)

                            # Guardar datos para la Tab Técnica
                            st.session_state['last_X'] = X
                            st.session_state['last_idx'] = pred_idx
                            st.session_state['last_label'] = pred_label

                        except Exception as e:
                            st.error(f"Error analyzing audio: {e}")

# --- TAB 2: TECHNICAL DEEP DIVE ---
with tab2:
    if 'last_X' in st.session_state and model is not None:
        st.subheader(t['viz_title'])
        st.write(t['gradcam_desc'])
        
        X = st.session_state['last_X']
        pred_idx = st.session_state['last_idx']
        LABELS = ["Makina", "Newstyle"]

        # Grad-CAM Logic
        try:
            # Buscar última capa convolucional
            conv_layers = [l.name for l in model.layers if "conv" in l.name]
            if conv_layers:
                target_layer = conv_layers[-1]
                
                heatmap = gradcam_heatmap(model, X, target_layer, pred_idx, upsample_to=(N_MELS, FIXED_TIME_FRAMES))
                
                # Visualización Lado a Lado
                c_tech_1, c_tech_2 = st.columns(2)
                
                with c_tech_1:
                    st.markdown("**Original Mel-Spectrogram**")
                    # Extraer canal 0 para visualizar
                    fig_mel = mel_figure(X[0, :, :, 0], sr=22050, n_mels=N_MELS, title=f"Input Audio ({LABELS[pred_idx]})")
                    st.pyplot(fig_mel)

                with c_tech_2:
                    st.markdown(f"**AI Attention Map ({target_layer})**")
                    fig_cam = overlay_gradcam_on_mel(X[0, :, :, 0], heatmap, labels=LABELS, pred_idx=pred_idx)
                    st.pyplot(fig_cam)
            else:
                st.warning("Could not find a Convolutional layer for Grad-CAM.")
        except Exception as e:
            st.error(f"Grad-CAM computation error: {e}")

    else:
        st.info("Run a prediction in the 'Demo' tab first to see the technical analysis.")

# --- TAB 3: BUSINESS IMPACT ---
with tab3:
    st.subheader(t['biz_title'])
    
    st.markdown(f"""
    <div class="biz-card">
        <h3>{t['biz_1']}</h3>
        <p>{t['biz_1_desc']}</p>
    </div>
    <div class="biz-card">
        <h3>{t['biz_2']}</h3>
        <p>{t['biz_2_desc']}</p>
    </div>
    <div class="biz-card">
        <h3>{t['biz_3']}</h3>
        <p>{t['biz_3_desc']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mock Chart
    st.write("### Performance vs Legacy Systems")
    chart_data = {"System": ["Manual Tagging", "Legacy Rule-based", "Fabrik AI (Deep Learning)"],
                  "Efficiency (Files/Hour)": [50, 400, 5000]}
    st.bar_chart(chart_data, x="System", y="Efficiency (Files/Hour)")
