import streamlit as st
from PIL import Image, ImageStat

# 1. App-Konfiguration & Styling
st.set_page_config(page_title="KI Wetter-Scanner", page_icon="🌦️", layout="centered")

st.title("🌦️ KI Himmels-Scanner")
st.write("Diese App analysiert die Pixelstruktur und Helligkeit deines Fotos in Echtzeit für eine sofortige Schauer-Vorhersage.")

st.markdown("---")

# 2. GUI: Foto machen oder hochladen
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

# 3. Sofortige Bildanalyse (Computer Vision)
def analyze_sky_instantly(image):
    # Bild verkleinern für eine blitzschnelle Berechnung
    img_small = image.resize((50, 50))
    stat = ImageStat.Stat(img_small)
    
    # Helligkeit berechnen (0 = schwarz, 255 = weiß)
    avg_brightness = sum(stat.mean) / 3
    r, g, b = stat.mean[0], stat.mean[1], stat.mean[2]
    
    # Blauer Himmel senkt das Regenrisiko
    is_blue_sky = b > (r + 5) and b > (g + 5)
    
    # Berechne den Wolken-Score
    cloud_score = max(0, min(100, int((255 - avg_brightness) * 0.6)))
    if is_blue_sky:
        cloud_score = max(0, cloud_score - 30)
        
    return cloud_score, avg_brightness

# 4. Sofortige Auswertung ohne Ladezeit
if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    
    # Berechne das Ergebnis sofort
    regen_risiko, helligkeit = analyze_sky_instantly(image)
    
    st.markdown("---")
    st.subheader("📊 Analyse-Ergebnis (Sofort)")
    
    # Zeige die Werte direkt an
    col1, col2 = st.columns(2)
    col1.metric("Himmel-Helligkeit", f"{int(helligkeit)} / 255")
    col2.metric("Regenrisiko", f"{regen_risiko}%")
    
    st.markdown("---")
    st.subheader("☂️ Deine Empfehlung für den Alltag:")

    # Regenschirm-Logik basierend auf den Werten
    if regen_risiko >= 55:
        st.error("🌧️ **Regenschirm-Alarm! (Schauer sehr wahrscheinlich)**")
        st.write(
            f"Die Analyse zeigt eine sehr hohe Dichte an dunklen Pixeln ({regen_risiko}% Wolkendichte). "
            "Das deutet stark auf schwere Schauerwolken hin. **Nimm unbedingt einen Regenschirm mit!**"
        )
    elif 30 <= regen_risiko < 55:
        st.warning("⚠️ **Späterer Regen möglich (Bewölkt).**")
        st.write(
            "Der Himmel ist spürbar bewölkt. Auch wenn es jetzt noch trocken sein sollte, "
            "kann sich das Wetter schnell ändern. Pack lieber einen kleinen Schirm ein!"
        )
    else:
        st.success("😎 **Kein Regenschirm nötig! (Sonnig / Klar)**")
        st.write(
            "Die Analyse erkennt einen hellen oder blauen Himmel. Keine Schauerwolken in Sicht. "
            "Du kannst deinen Regenschirm heute beruhigt zu Hause lassen!"
        )
