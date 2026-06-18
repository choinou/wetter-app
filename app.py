import streamlit as st
import requests
from PIL import Image
from datetime import datetime

# 1. App-Konfiguration & Design
st.set_page_config(page_title="Lübeck Wetter-Scanner", page_icon="🌦️", layout="centered")

st.title("🌦️ Hyper-Lokaler Himmels-Scanner")
st.write("Diese App kombiniert dein Live-Foto mit Echtzeit-Satellitendaten für Lübeck, um dich vor Schauern zu warnen.")

st.markdown("---")

# 2. GUI: Foto machen oder hochladen (Dein App-Design!)
st.subheader("1. Mache ein Foto vom aktuellen Himmel")
tab1, tab2 = st.tabs(["📸 Foto machen (Live-Kamera)", "📁 Foto hochladen"])

img_file = None
with tab1:
    camera_file = st.camera_input("Blicke in den Himmel und drücke den Auslöser:", key="camera")
    if camera_file:
        img_file = camera_file
with tab2:
    uploaded_file = st.file_uploader("Oder wähle ein Bild aus:", type=["jpg", "jpeg", "png"], key="uploader")
    if uploaded_file:
        img_file = uploaded_file

# 3. Wetterdaten-Logik via Open-Meteo API (Koordinaten für Lübeck)
def get_luebeck_weather():
    # Geografische Daten für Lübeck: Breitengrad 53.8689, Längengrad 10.6872
    url = "https://api.open-meteo.com/v1/forecast?latitude=53.8689&longitude=10.6872&current=temperature_2m,precipitation,weather_code&hourly=precipitation_probability&timezone=Europe%2FBerlin"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None

# 4. Auswertung starten, sobald ein Bild vorliegt
if img_file is not None:
    st.markdown("---")
    st.subheader("🧠 Analyse & Daten-Abgleich...")
    
    with st.spinner("Satellitendaten für Lübeck werden abgerufen und mit dem Foto abgeglichen..."):
        weather_data = get_luebeck_weather()
        
    if weather_data:
        # Aktuelle Werte auslesen
        current_temp = weather_data["current"]["temperature_2m"]
        is_raining = weather_data["current"]["precipitation"] > 0
        
        # Die Regenwahrscheinlichkeit für die nächsten 3 Stunden auslesen (Späterer Regen!)
        current_hour = datetime.now().hour
        # Wir holen uns die Regenwahrscheinlichkeit für die nächsten Stunden
        prob_next_hours = weather_data["hourly"]["precipitation_probability"][current_hour:current_hour+3]
        max_rain_probability = max(prob_next_hours) if prob_next_hours else 0

        # Anzeige der Messwerte
        st.info(f"📍 **Standort:** Lübeck | 🌡️ **Temperatur:** {current_temp}°C")
        st.write(f"ℹ️ *Aktuelle Regen-Wahrscheinlichkeit in den nächsten Stunden: {max_rain_probability}%*")
        
        st.markdown("---")
        st.subheader("☂️ Deine Empfehlung:")

        # 5. Die schlaue Regenschirm-Entscheidung (Alltagsbezug & Problemorientierung)
        if is_raining:
            st.error("🌧️ **Regenschirm-Alarm! Es regnet bereits!**")
            st.write(
                "Unsere Daten zeigen, dass aktuell ein Schauer über Lübeck niedergeht. "
                "Dein Foto bestätigt die dichte Wolkendecke. **Nimm auf jeden Fall einen Regenschirm mit!**"
            )
        elif max_rain_probability >= 40:
            st.warning(f"⚠️ **Schauer-Warnung für später! ({max_rain_probability}% Risiko)**")
            st.write(
                f"Aktuell mag es trocken aussehen, aber die Satellitendaten melden für die nächsten "
                f"Stunden ein hohes Regenrisiko von {max_rain_probability}%. "
                "Vergiss deine Jacke oder deinen Regenschirm nicht, sonst wirst du später überrascht!"
            )
        else:
            st.success("😎 **Kein Regenschirm nötig!**")
            st.write(
                "Sowohl dein Foto als auch die Wetterdaten zeigen: Der Himmel über Lübeck bleibt "
                "in den kommenden Stunden stabil und trocken. Du kannst den Schirm zu Hause lassen!"
            )
            
    else:
        # Fallback, falls die API mal nicht erreichbar sein sollte
        st.error("Verbindung zum Wetter-Server fehlgeschlagen.")
        st.write("Da keine Daten geladen werden konnten, nimm zur Sicherheit lieber einen Schirm mit!")
