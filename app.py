import streamlit as st
from transformers import pipeline
from PIL import Image
import os

# 1. App-Konfiguration
st.set_page_config(page_title="Himmel-Scanner", page_icon="🌦️", layout="centered")

st.title("🌦️ KI Himmels-Scanner")
st.write("Diese App nutzt ein echtes Hugging-Face-Wettermodell direkt auf dem Server.")

# 2. Modell sicher herunterladen und lokal cachen (Verhindert Internet-Fehler nach dem Start)
@st.cache_resource
def load_weather_model():
    # Wir zwingen die App, das Modell in den lokalen Ordner zu sichern
    local_model_dir = "./mein_wetter_modell"
    
    if not os.path.exists(local_model_dir):
        with st.spinner("Modell wird das erste Mal von Hugging Face geladen... Bitte ca. 1 Minute Geduld!"):
            # Holt das spezialisierte Wetter-Modell
            pipe = pipeline("image-classification", model="raccor/google-vit-base-patch16-224-weather-classification")
            # Speichert es lokal ab, damit kein Internet mehr gebraucht wird
            pipe.save_pretrained(local_model_dir)
            return pipe
    else:
        # Wenn es schon existiert, laden wir es blitzschnell offline aus dem Ordner
        return pipeline("image-classification", model=local_model_dir)

try:
    classifier = load_weather_model()
except Exception as e:
    st.error(f"Fehler beim Starten des Modells: {e}")
    st.write("Tipp: Falls Streamlit Cloud das Internet sperrt, starte die App einmal lokal auf deinem Computer, damit das Modell heruntergeladen wird!")

st.markdown("---")

# 3. Auswahlmodus: Hochladen ODER Kamera
st.subheader("Wie möchtest du das Foto bereitstellen?")
tab1, tab2 = st.tabs(["📁 Foto hochladen", "📸 Foto machen (Live-Kamera)"])

img_file = None

with tab1:
    uploaded_file = st.file_uploader("Wähle ein Bild vom Himmel oder den Wolken aus:", type=["jpg", "jpeg", "png"], key="uploader")
    if uploaded_file:
        img_file = uploaded_file

with tab2:
    camera_file = st.camera_input("Nimm den aktuellen Himmel auf:", key="camera")
    if camera_file:
        img_file = camera_file

# 4. Auswertung und Wetter-Logik
if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    
    st.markdown("---")
    st.subheader("🧠 KI-Analyse läuft...")
    
    with st.spinner("Das Hugging-Face-Modell analysiert die Wolken..."):
        # Vorhersage mit dem spezialisierten Wetter-Modell treffen
        predictions = classifier(image)
        
        # Das beste Ergebnis extrahieren
        top_result = predictions[0]
        weather_label = top_result['label'].lower()
        confidence = top_result['score'] * 100

    # Anzeige der genauen Klassen von Hugging Face
    st.info(f"**Erkannte Wetterlage:** `{weather_label.capitalize()}` ({confidence:.1f}% Sicherheit)")

    # 5. Problemorientierung & Alltagsbezug: Regenschirm-Entscheidung
    st.subheader("☂️ Deine Empfehlung:")

    # Das Hugging Face Modell kennt exakt diese Klassen: "rainy", "cloudy", "sunny", "foggy"
    if "rain" in weather_label:
        st.error("🚨 **Schauer-Alarm! Nimm UNBEDINGT einen Regenschirm mit.**")
        st.write(
            "Die KI hat eindeutig Regen oder extrem nasse Schauerwolken erkannt. "
            "Wenn du jetzt ohne Schirm oder Regenjacke rausgehst, wirst du nass!"
        )
    elif "cloudy" in weather_label:
        st.warning("⚠️ **Vorsicht: Grauer Himmel / Späterer Regen möglich.**")
        st.write(
            "Die KI erkennt dichte Bewölkung (Cloudy). Auch wenn es in Lübeck gerade "
            "noch trocken sein sollte: Das Wetter kann schnell umschlagen. "
            "Nimm zur Sicherheit lieber eine Jacke oder einen kleinen Schirm mit!"
        )
    elif "sunny" in weather_label or "clear" in weather_label:
        st.success("😎 **Kein Regenschirm nötig!**")
        st.write(
            "Die KI meldet Sonnenschein oder einen klaren, freundlichen Himmel. "
            "Du wirst heute trocken bleiben. Lass den Schirm zu Hause und genieß den Tag!"
        )
    else:
        # Für "foggy" (Nebel) oder unklare Ergebnisse
        st.warning("🌫️ **Sichtbehinderung / Nebelig**")
        st.write(
            "Die KI erkennt Nebel oder starken Dunst. Es ist zwar kein akuter Wolkenbruch "
            "zu sehen, aber es könnte feucht und kühl sein. Pack dich warm ein!"
        )
