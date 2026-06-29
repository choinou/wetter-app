import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. App-Konfiguration & Styling
st.set_page_config(page_title="KI Wetter-Scanner", page_icon="🌦️", layout="centered")

st.title("🌦️ KI Himmels-Scanner")
st.write("Diese App nutzt ein spezialisiertes, neuronales Netzwerk, das exklusiv mit Wetter- und Wolkenbildern trainiert wurde.")

# 2. Reines Wetter-Modell von Hugging Face laden
@st.cache_resource
def load_weather_model():
    # Dieses Modell wurde ausschließlich auf die Erkennung von Wetterlagen trainiert
    return pipeline("image-classification", model="alibidaran/vit-base-weather-classification")

with st.spinner("🤖 Spezialisiertes Wetter-Modell wird geladen... Bitte kurz warten."):
    try:
        classifier = load_weather_model()
        st.success("✅ Wetter-KI erfolgreich gestartet!")
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")

st.markdown("---")

# 3. GUI: Foto machen oder hochladen
st.subheader("1. Himmel fotografieren oder hochladen")
tab1, tab2 = st.tabs(["📸 Foto machen (Live-Kamera)", "📁 Foto hochladen"])

img_file = None
with tab1:
    camera_file = st.camera_input("Nimm den aktuellen Himmel auf:", key="camera")
    if camera_file:
        img_file = camera_file
with tab2:
    uploaded_file = st.file_uploader("Oder wähle ein Wetterbild aus:", type=["jpg", "jpeg", "png"], key="uploader")
    if uploaded_file:
        img_file = uploaded_file

# 4. KI-Vorhersage & Regenschirm-Logik
if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    
    st.markdown("---")
    st.subheader("🧠 KI-Analyse läuft...")
    
    with st.spinner("Das Wetter-Netzwerk analysiert die Wolkendichte..."):
        # Die KI gibt uns direkt Wetter-Labels wie 'rainy' oder 'cloudy' zurück
        predictions = classifier(image)
        
        # Das Top-Ergebnis herausfiltern
        top_result = predictions[0]
        weather_label = top_result['label'].lower()
        confidence = top_result['score'] * 100

    # Anzeige des echten Wetter-Labels
    st.info(f"**Erkannte Wetterlage:** `{weather_label.capitalize()}` (Sicherheit: {confidence:.1f}%)")
    
    st.markdown("---")
    st.subheader("☂️ Deine Empfehlung für Lübeck:")

    # 5. Alltagsbezug: Regenschirm-Logik basierend auf reinen Wetter-Klassen
    if "rain" in weather_label:
        st.error("🌧️ **Regenschirm-Alarm! (Schauer erkannt)**")
        st.write(
            "Die KI hat das Bild eindeutig als 'Rainy' eingestuft. Es hängen schwere, "
            "wassergeladene Wolken über dir. **Nimm auf jeden Fall einen Regenschirm mit!**"
        )
    elif "cloud" in weather_label:
        st.warning("⚠️ **Späterer Regen möglich (Bewölkt).**")
        st.write(
            "Das Modell erkennt dichten, bewölkten Himmel ('Cloudy'). Auch wenn es im Moment "
            "noch trocken sein sollte: Schauer können sich schnell bilden. "
            "Pack zur Sicherheit lieber eine Regenjacke oder einen kleinen Schirm ein!"
        )
    elif "sun" in weather_label or "clear" in weather_label:
        st.success("😎 **Kein Regenschirm nötig! (Sonnig)**")
        st.write(
            "Das Modell klassifiziert den Himmel als 'Sunny' oder 'Clear'. "
            "Keine Schauerwolken in Sicht. Du kannst den Regenschirm heute zu Hause lassen!"
        )
    else:
        # Für Klassen wie 'foggy' (nebelig)
        st.warning("🌫️ **Dunstig / Nebelige Wetterlage**")
        st.write(
            "Die KI erkennt Nebel. Die Luft ist feucht, aber es droht kein akuter Wolkenbruch. "
            "Ein dicker Schirm ist vermutlich nicht nötig, aber zieh dich warm an!"
        )
