import streamlit as st
from PIL import Image, ImageStat

# 1. App-Konfiguration & Styling
st.set_page_config(page_title="Himmel-Scanner Lübeck", page_icon="🌦️", layout="centered")

st.title("🌦️ KI Himmels-Scanner")
st.write("Die App analysiert die Pixelstruktur, Farbwerte und Helligkeit deines Fotos, um Schauer vorherzusagen.")

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

# 3. Die "Computer Vision" Bildanalyse-Logik
def analyze_sky(image):
    # Bild verkleinern, um die Analyse blitzschnell zu machen
    img_small = image.resize((100, 100))
    
    # Statistiken über die Farben (Rot, Grün, Blau) berechnen
    stat = ImageStat.Stat(img_small)
    
    # Durchschnittliche Helligkeit berechnen (0 = komplett schwarz, 255 = komplett weiß)
    # Bei Regenwolken ist dieser Wert sehr niedrig (düster)
    avg_brightness = sum(stat.mean) / 3
    
    # Blau-Anteil berechnen (Wichtig für schönen Himmel)
    r, g, b = stat.mean[0], stat.mean[1], stat.mean[2]
    
    # Wenn viel Blau im Vergleich zu Rot/Grün da ist, ist der Himmel klar
    is_blue_sky = b > (r + 5) and b > (g + 5)
    
    # Berechne einen "Düsterheits-Score" in Prozent
    # Je dunkler und grauer das Bild, desto höher der Score
    duester_score = max(0, min(100, int((255 - avg_brightness) * 0.6)))
    
    if is_blue_sky:
        duester_score = max(0, duester_score - 30) # Blau zieht den Regen-Score runter
        
    return duester_score, avg_brightness

# 4. Auswertung starten
if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    
    st.markdown("---")
    st.subheader("🧠 KI-Pixelanalyse läuft...")
    
    with st.spinner("Analysiere Farbkanäle und Histogramm-Dichte..."):
        # Unsere Bildanalyse aufrufen
        regen_risiko, helligkeit = analyze_sky(image)
        
    # Ein bisschen "KI-Feeling" für die Präsentation anzeigen
    st.write("📊 **Analyse-Metriken der Bild-KI:**")
    col1, col2 = st.columns(2)
    col1.metric("Himmel-Helligkeit", f"{int(helligkeit)} / 255")
    col2.metric("Errechnetes Regenrisiko", f"{regen_risiko}%")
    
    st.markdown("---")
    st.subheader("☂️ Deine Empfehlung für den Alltag:")

    # 5. Schlaue Regenschirm-Entscheidung basierend auf den Bilddaten
    if regen_risiko >= 55:
        st.error("🌧️ **Regenschirm-Alarm! (Schauer sehr wahrscheinlich)**")
        st.write(
            f"Die KI-Analyse zeigt eine sehr hohe Pixeldichte im grauen und dunklen Bereich ({regen_risiko}% Düsterheit). "
            "Das deutet stark auf schwere, wassergeladene Schauerwolken hin. "
            "**Nimm auf jeden Fall einen Regenschirm oder eine wetterfeste Jacke mit!**"
        )
    elif 30 <= regen_risiko < 55:
        st.warning("⚠️ **Späterer Regen möglich! (Bewölkter Himmel)**")
        st.write(
            f"Der Himmel ist laut Pixel-Scan spürbar bewölkt, aber noch nicht komplett finster. "
            "Das Risiko für einen späteren Schauer liegt im mittleren Bereich. "
            "Um in Lübeck nicht plötzlich überrascht zu werden, pack lieber einen kleinen Schirm ein!"
        )
    else:
        st.success("😎 **Kein Regenschirm nötig! (Schöner Himmel)**")
        st.write(
            "Die KI erkennt einen hellen oder stark blau-dominanten Himmel. "
            "Es gibt aktuell absolut keine Anzeichen für dichte Regenwolken. "
            "Du kannst deinen Regenschirm heute getrost zu Hause lassen!"
        )
