import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. App-Konfiguration & Styling
st.set_page_config(page_title="KI Wetter-Scanner", page_icon="🌦️", layout="centered")

st.title("🌦️ KI Himmels-Scanner")
st.write("Diese App nutzt ein extrem schnelles und leichtgewichtiges KI-Wettermodell von Hugging Face.")

# 2. Das ultra-kleine Wetter-Spezialmodell laden
@st.cache_resource
def load_tiny_weather_model():
    # Dieses Modell wiegt fast nichts und wurde nur für Wetter trainiert
    return pipeline("image-classification", model="Aires/mobilenet_v2_weather_classification")

with st.spinner("🤖 Wetter-KI wird gestartet... Bitte einen kurzen Moment Geduld."):
    try:
        classifier = load_tiny_weather_model()
        st.success("✅ Wetter-KI erfolgreich gestartet!")
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        st.info("Falls die Cloud-Sperre aktiv ist, ist dieses Modell so klein, dass du es super leicht als Alternative testen kannst.")

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
    
    with st.spinner("Das neuronale Netzwerk scannt das Bild..."):
        predictions = classifier(image)
        top_result = predictions[0]
        weather_label = top_result['label'].lower()
        confidence = top_result['score'] * 100

    # Anzeige des gelernten Wetter-Labels
    st.info(f"**Erkannte Wetterlage:** `{weather_label.capitalize()}` (Sicherheit: {confidence:.1f}%)")
    
    st.markdown("---")
    st.subheader("☂️ Deine Empfehlung für den Alltag:")

    # 5. Alltagsbezug: Regenschirm-Logik (Passend zu den 4 Klassen des Modells)
    if "rain" in weather_label:
        st.error("🌧️ **Regenschirm-Alarm! (Schauer erkannt)**")
        st.write("Die Wetter-KI sieht eindeutig Regenwolken. Nimm unbedingt einen Regenschirm mit!")
    elif "cloud" in weather_label:
        st.warning("⚠️ **Späterer Regen möglich (Bewölkt).**")
        st.write("Das Modell erkennt dichten, bewölkten Himmel. Pack zur Sicherheit lieber einen kleinen Schirm ein!")
    elif "shin" in weather_label or "sun" in weather_label:
        st.success("😎 **Kein Regenschirm nötig! (Sonnig)**")
        st.write("Die KI meldet guten Sonnenschein. Du bleibst trocken, genieß den Tag!")
    else:
        # Für 'sunrise' (Sonnenaufgang) oder unerwartete Ergebnisse
        st.success("🌤️ **Freie Sicht / Schöner Tagesstart**")
        st.write("Das Modell erkennt einen freundlichen Himmel. Aktuell ist kein Regenschirm notwendig!")
