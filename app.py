import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. App-Konfiguration & Styling
st.set_page_config(page_title="Himmel-Scanner Lübeck", page_icon="🌦️", layout="centered")

st.title("🌦️ KI Himmels-Scanner")
st.write("Diese App nutzt ein neuronales Netzwerk von Hugging Face, um deine Himmelsfotos auf Schauerwolken zu analysieren.")

# 2. Echtes Hugging-Face-Modell laden (Gecached für maximale Geschwindigkeit)
@st.cache_resource
def load_hf_model():
    # Wir nutzen ein leichtgewichtiges, stabiles Bilderkennungsmodell von Microsoft via Hugging Face
    # Es ist perfekt für Cloud-Server, da es extrem schnell und klein ist.
    return pipeline("image-classification", model="microsoft/resnet-18")

with st.spinner("🤖 Hugging Face KI-Modell wird gestartet... Bitte kurz warten."):
    try:
        classifier = load_hf_model()
        st.success("🤖 KI-Modell erfolgreich geladen!")
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
    uploaded_file = st.file_uploader("Oder wähle ein Bild aus:", type=["jpg", "jpeg", "png"], key="uploader")
    if uploaded_file:
        img_file = uploaded_file

# 4. KI-Vorhersage & Regenschirm-Logik
if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    
    st.markdown("---")
    st.subheader("🧠 KI-Analyse läuft...")
    
    with st.spinner("Das neuronale Netzwerk analysiert das Bild..."):
        # Das Hugging-Face-Modell analysiert das Foto und gibt die Top 5 Objekte zurück
        predictions = classifier(image)
        
        # Wir holen uns alle erkannten Begriffe in eine Liste
        detected_objects = [pred['label'].lower() for pred in predictions]
        detected_string = ", ".join(detected_objects)

    # Unsichtbarer Entwickler-Hinweis auf dem Bildschirm (gut zum Testen)
    st.write(f"*(Erkannte Muster der KI: {detected_string})*")
    
    st.markdown("---")
    st.subheader("☂️ Deine Empfehlung für den Alltag:")

    # 5. Schlaue Alltags-Logik basierend auf den KI-Ergebnissen
    # Wenn das Modell Wolkenstrukturen, Dunst oder graue Muster erkennt:
    if any(word in detected_string for word in ["cloud", "sky", "mist", "fog", "rain", "umbrella"]):
        
        # Eine kleine Zusatz-Unterscheidung: Sieht es nach richtig schweren Wolken aus?
        if any(heavy in detected_string for word in detected_objects for heavy in ["thunder", "dark", "grey", "cumulus"]):
            st.error("🌧️ **Regenschirm-Alarm! (Schauer sehr wahrscheinlich)**")
            st.write(
                "Das Hugging-Face-Modell erkennt sehr dichte, schwere Wolkenstrukturen. "
                "Die Wahrscheinlichkeit für plötzlichen Schauer in Lübeck ist extrem hoch. "
                "**Nimm auf jeden Fall einen Regenschirm oder eine wetterfeste Jacke mit!**"
            )
        else:
            st.warning("⚠️ **Späterer Regen möglich! (Bewölkter Himmel)**")
            st.write(
                "Die KI hat Wolken oder Dunst am Himmel registriert. Aktuell ist es zwar vielleicht "
                "noch trocken, aber das Wetter kann schnell umschlagen. "
                "Sicher ist sicher: Pack lieber einen kleinen Schirm ein!"
            )
    else:
        # Wenn der Himmel hell, klar oder sonnig glänzt
        st.success("😎 **Kein Regenschirm nötig! (Schöner Himmel)**")
        st.write(
            "Das neuronale Netzwerk erkennt keine Anzeichen von dichten Schauerwolken. "
            "Der Himmel sieht freundlich aus. Du kannst deinen Regenschirm heute zu Hause lassen!"
        )
