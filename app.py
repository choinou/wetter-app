import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort

# 1. App-Konfiguration & Styling
st.set_page_config(page_title="Himmel-Scanner", page_icon="🌦️", layout="centered")

st.title("🌦️ KI Himmels-Scanner")
st.write("Diese App läuft zu 100% offline über ein integriertes ONNX-Wettermodell!")

# Wetter-Klassen, die dieses Modell gelernt hat
CLASSES = ["Cloudy (Bewölkt)", "Foggy (Nebelig)", "Rainy (Regnerisch)", "Sunny (Sonnig)"]

# 2. ONNX Modell laden
@st.cache_resource
def load_onnx_model():
    # Lädt die Datei direkt aus deinem Projektordner
    return ort.InferenceSession("weather_model.onnx")

try:
    session = load_onnx_model()
    st.success("✅ KI-Modell erfolgreich lokal geladen!")
except Exception as e:
    st.error("❌ Die Datei 'weather_model.onnx' wurde im Ordner nicht gefunden!")
    st.write("Bitte stelle sicher, dass du die Modelldatei heruntergeladen und in denselben Ordner wie die app.py gelegt hast.")
    st.stop()

# 3. Bildvorbereitung (Preprocessing für die KI)
def preprocess_image(image):
    # Bild auf die exakte Größe bringen, die das Modell erwartet (224x224 Pixel)
    image = image.resize((224, 224))
    img_data = np.array(image).astype('float32') / 255.0
    
    # Normalisierung (Standard für Vision-Modelle)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_data = (img_data - mean) / std
    
    # Dimensionen anpassen (Vom Bild-Format ins KI-Format: Channels First)
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

st.markdown("---")

# 4. GUI-Auswahlmodus
st.subheader("Wie möchtest du das Foto bereitstellen?")
tab1, tab2 = st.tabs(["📁 Foto hochladen", "📸 Foto machen (Live-Kamera)"])

img_file = None
with tab1:
    uploaded_file = st.file_uploader("Wähle ein Bild vom Himmel aus:", type=["jpg", "jpeg", "png"], key="uploader")
    if uploaded_file:
        img_file = uploaded_file

with tab2:
    camera_file = st.camera_input("Nimm den aktuellen Himmel auf:", key="camera")
    if camera_file:
        img_file = camera_file

# 5. Auswertung & Regenschirm-Entscheidung
if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    
    st.markdown("---")
    st.subheader("🧠 KI-Analyse läuft...")
    
    # Bild für die KI vorbereiten
    input_data = preprocess_image(image)
    
    # KI-Berechnung starten
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    
    # Ergebnisse auswerten (Softmax für Prozentwerte)
    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()
    
    # Das beste Ergebnis finden
    top_class_idx = np.argmax(probabilities)
    weather_label = CLASSES[top_class_idx]
    confidence = probabilities[top_class_idx] * 100

    # Ergebnis anzeigen
    st.info(f"**Erkannte Wetterlage:** `{weather_label}` (Sicherheit: {confidence:.1f}%)")

    # 6. Regenschirm-Logik (Alltagsbezug für Lübeck)
    st.subheader("☂️ Deine Empfehlung:")

    if "Rainy" in weather_label:
        st.error("🚨 **Schauer-Alarm! Nimm UNBEDINGT einen Regenschirm mit.**")
        st.write("Die KI sieht dicke Regenwolken. Geh nicht ohne Schirm aus dem Haus!")
    elif "Cloudy" in weather_label:
        st.warning("⚠️ **Späterer Regen möglich (Bewölkt).**")
        st.write("Es ist aktuell zwar noch trocken, aber der Himmel ist dicht. Ein kleiner Schirm in der Tasche schadet heute nicht.")
    elif "Sunny" in weather_label:
        st.success("😎 **Kein Regenschirm nötig!**")
        st.write("Freie Sicht und Sonne! Du kannst deinen Regenschirm heute beruhigt zu Hause lassen.")
    else:
        st.warning("🌫️ **Nebelig / Feucht**")
        st.write("Es ist nebelig. Pack dich warm ein, ein dicker Regenschirm ist aber vermutlich nicht nötig.")
