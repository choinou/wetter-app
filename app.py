import streamlit as st
import requests
from PIL import Image
import traceback  # Neu: Hilft uns, den genauen Fehler zu finden

st.set_page_config(page_title="Himmel-Scanner", page_icon="🌦️", layout="centered")

st.title("🌦️ KI Himmels-Scanner (Fehlersuche)")
st.write("Füttere die KI mit einem Bild vom Himmel.")

# --- DEIN HUGGING FACE TOKEN HIER EINTRAGEN ---
HF_TOKEN = "DEIN_HIER_EINGEBEN" 

API_URL = "https://api-inference.huggingface.co/models/raccor/google-vit-base-patch16-224-weather-classification"

def query_weather_api(image_bytes):
    headers = {}
    if HF_TOKEN and HF_TOKEN != "DEIN_HIER_EINGEBEN" and HF_TOKEN != "":
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    return response

st.markdown("---")

st.subheader("Wie möchtest du das Foto bereitstellen?")
tab1, tab2 = st.tabs(["📁 Foto hochladen", "📸 Foto machen (Live-Kamera)"])

img_file = None

with tab1:
    uploaded_file = st.file_uploader("Wähle ein Bild aus:", type=["jpg", "jpeg", "png"], key="uploader")
    if uploaded_file:
        img_file = uploaded_file

with tab2:
    camera_file = st.camera_input("Nimm den aktuellen Himmel auf:", key="camera")
    if camera_file:
        img_file = camera_file

if img_file is not None:
    st.markdown("---")
    st.subheader("🧠 KI-Analyse läuft...")
    
    with st.spinner("Das Foto wird verarbeitet..."):
        try:
            # 1. Bild-Bytes holen
            image_bytes = img_file.getvalue()
            
            # 2. API aufrufen
            response = query_weather_api(image_bytes)
            
            # 3. Status-Code prüfen
            st.write(f"🌐 Server-Antwort-Code: `{response.status_code}`")
            
            if response.status_code == 200:
                predictions = response.json()
                
                if isinstance(predictions, dict) and "estimated_time" in predictions:
                    st.warning(f"Das Modell lädt noch auf den HF-Servern. Bitte warte {int(predictions['estimated_time'])} Sekunden und lade das Bild erneut hoch.")
                else:
                    top_result = predictions[0]
                    weather_label = top_result['label'].lower()
                    confidence = top_result['score'] * 100

                    st.success(f"**Ergebnis:** {weather_label.capitalize()} ({confidence:.1f}%)")
            else:
                st.error(f"Hugging Face meldet einen Fehler (Code {response.status_code})")
                st.code(response.text) # Zeigt den echten Fehlertext von Hugging Face an

        except Exception as e:
            # DAS HIER ZEIGT UNS JETZT DEN ECHTEN FEHLER!
            st.error("🚨 Ein interner Python-Fehler ist aufgetreten:")
            st.exception(e)  # Druckt den exakten Fehler auf den Bildschirm
