import streamlit as st
import requests
from PIL import Image
import io

# 1. App-Konfiguration & Styling
st.set_page_config(page_title="Himmel-Scanner", page_icon="🌦️", layout="centered")

st.title("🌦️ KI Himmels-Scanner")
st.write("Füttere die KI mit einem Bild vom Himmel, um zu sehen, ob du einen Regenschirm brauchst.")

# Die Adresse zur kostenlosen Hugging Face API für genau dein Wunsch-Modell
API_URL = "https://api-inference.huggingface.co/models/raccor/google-vit-base-patch16-224-weather-classification"

# Funktion, die das Bild per Internet an Hugging Face schickt und das Ergebnis holt
def query_weather_api(image_bytes):
    response = requests.post(API_URL, data=image_bytes)
    return response.json()

st.markdown("---")

# 2. Auswahlmodus: Hochladen ODER Kamera via Tabs
st.subheader("Wie möchtest du das Foto bereitstellen?")
tab1, tab2 = st.tabs(["📁 Foto hochladen", "📸 Foto machen (Live-Kamera)"])

img_file = None

with tab1:
    uploaded_file = st.file_uploader(
        "Wähle ein Bild vom Himmel oder den Wolken aus:", 
        type=["jpg", "jpeg", "png"],
        key="uploader"
    )
    if uploaded_file:
        img_file = uploaded_file

with tab2:
    camera_file = st.camera_input("Nimm den aktuellen Himmel auf:", key="camera")
    if camera_file:
        img_file = camera_file

# 3. Auswertung und Logik
if img_file is not None:
    # Wir wandeln das hochgeladene Bild in "Bytes" um, damit wir es senden können
    image_bytes = img_file.getvalue()
    
    st.markdown("---")
    st.subheader("🧠 KI-Analyse läuft...")
    
    with st.spinner("Das Foto wird an die Hugging Face API gesendet..."):
        try:
            # API abfragen
            predictions = query_weather_api(image_bytes)
            
            # Fehler abfangen (z.B. falls die API das Bild nicht lesen konnte oder das Modell lädt)
            if isinstance(predictions, dict) and "error" in predictions:
                st.warning("Die KI wird auf den Servern gerade gestartet. Bitte warte 10 Sekunden und versuche es noch einmal!")
            else:
                # Das beste Ergebnis herausfiltern
                top_result = predictions[0]
                weather_label = top_result['label'].lower()
                confidence = top_result['score'] * 100

                # Anzeige des KI-Ergebnisses
                st.info(f"**Ergebnis der KI:** {weather_label.capitalize()} ({confidence:.1f}% Sicherheit)")

                # 4. Regenschirm-Logik (Problemorientierung & Alltagsbezug)
                st.subheader("☂️ Deine Empfehlung:")

                # Wenn Wolken oder Regen erkannt werden -> Regenschirm mitnehmen
                if "rain" in weather_label or "cloudy" in weather_label:
                    st.error("🚨 **Nimm einen Regenschirm mit!**")
                    st.write(
                        "Die KI erkennt dichte Wolken oder bereits einsetzenden Regen. "
                        "Selbst wenn es jetzt in Lübeck noch trocken ist: Die Wahrscheinlichkeit ist hoch, "
                        "dass du später im Regen stehst. Jacke und Regenschirm sind Pflicht!"
                    )
                    
                # Wenn es sonnig oder klar ist -> Kein Regenschirm nötig
                elif "sunny" in weather_label or "clear" in weather_label or "sunrise" in weather_label:
                    st.success("😎 **Kein Regenschirm nötig!**")
                    st.write(
                        "Der Himmel sieht super aus! Die KI erkennt viel freie Sicht und Sonne. "
                        "Du wirst heute höchstwahrscheinlich trocken bleiben. Genieß das gute Wetter!"
                    )
                    
                # Für Nebel oder unklare Wetterlagen
                else:
                    st.warning("⚠️ **Vorsicht geboten / Unsichere Wetterlage**")
                    st.write(
                        "Die KI erkennt eine unklare Wetterlage (z. B. Nebel oder starken Dunst). "
                        "Es droht zwar akut kein schwerer Schauer, aber eine wetterfeste Jacke schadet heute sicher nicht."
                    )
        except Exception as e:
            st.error("Es gab ein Problem bei der Verbindung mit der KI. Bitte probiere es noch einmal.")
