import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import io
import base64

# 1. App-Konfiguration
st.set_page_config(page_title="KI Wetter-Scanner", page_icon="🌦️", layout="centered")

st.title("🌦️ KI Himmels-Scanner")
st.write("Diese App nutzt ein reines, via Millionen Bilder trainiertes Wetter-Modell direkt in deinem Browser (Sicher vor Server-Sperren!).")

st.markdown("---")

# 2. GUI: Foto machen oder hochladen
st.subheader("1. Himmel fotografieren oder hochladen")
tab1, tab2 = st.tabs(["📸 Foto machen (Live-Kamera)", "📁 Foto hochladen"])

img_file = None
with tab1:
    camera_file = st.camera_input("Nimm den Himmel auf:", key="camera")
    if camera_file:
        img_file = camera_file
with tab2:
    uploaded_file = st.file_uploader("Oder wähle ein Wetterbild aus:", type=["jpg", "jpeg", "png"], key="uploader")
    if uploaded_file:
        img_file = uploaded_file

# 3. Wenn ein Bild da ist, wandeln wir es für das JavaScript-Modell um
if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    
    # Bild in Base64-String umwandeln, damit JavaScript es lesen kann
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    img_data_url = f"data:image/jpeg;base64,{img_str}"

    st.markdown("---")
    st.subheader("🧠 KI-Analyse & Empfehlung:")

    # Wir nutzen jetzt das offizielle, fehlerfreie 'onnx-community' Web-Modell!
    html_code = f"""
    <div id="status" style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; font-family: sans-serif; margin-bottom: 10px;">
        ⏳ Lade reines Wetter-Modell (Transformers.js) in deinem Browser...
    </div>
    <div id="result" style="font-family: sans-serif; line-height: 1.5;"></div>

    <script type="module">
        import {{ pipeline }} from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.14.0';

        const statusDiv = document.getElementById('status');
        const resultDiv = document.getElementById('result');

        async function runKI() {{
            try {{
                statusDiv.innerText = "🧠 KI analysiert das Foto auf Schauer-Muster...";
                
                # Neues, offiziell für Browser optimiertes Wetter-Modell
                const classifier = await pipeline('image-classification', 'onnx-community/vit-base-patch16-224-weather-classification');
                
                const output = await classifier('{img_data_url}');
                const topResult = output[0];
                const label = topResult.label.toLowerCase();
                const confidence = (topResult.score * 100).toFixed(1);

                statusDiv.style.backgroundColor = "#d4edda";
                statusDiv.style.color = "#155724";
                statusDiv.innerHTML = "<b>Erkannte Wetterlage:</b> " + label.toUpperCase() + " (" + confidence + "% Sicherheit)";

                // Regenschirm-Logik ausgeben (für die Klassen: rainy, cloudy, sunny, foggy)
                if (label.includes('rain') || label.includes('schauer')) {{
                    resultDiv.innerHTML = `
                        <div style="padding: 15px; background-color: #f8d7da; color: #721c24; border-radius: 5px; margin-top: 10px;">
                            <h3>🚨 Regenschirm-Alarm! (Schauer erkannt)</h3>
                            <p>Die reine Wetter-KI sieht eindeutig Regenwolken. Nimm unbedingt einen Regenschirm mit!</p>
                        </div>`;
                }} else if (label.includes('cloud') || label.includes('wolke')) {{
                    resultDiv.innerHTML = `
                        <div style="padding: 15px; background-color: #fff3cd; color: #856404; border-radius: 5px; margin-top: 10px;">
                            <h3>⚠️ Späterer Regen möglich (Bewölkt).</h3>
                            <p>Das Modell erkennt dichten, bewölkten Himmel. Pack zur Sicherheit lieber einen kleinen Schirm ein!</p>
                        </div>`;
                }} else if (label.includes('sun') || label.includes('clear') || label.includes('sonne')) {{
                    resultDiv.innerHTML = `
                        <div style="padding: 15px; background-color: #d4edda; color: #155724; border-radius: 5px; margin-top: 10px;">
                            <h3>😎 Kein Regenschirm nötig! (Sonnig)</h3>
                            <p>Die KI meldet guten Sonnenschein. Du bleibst trocken, genieß den Tag!</p>
                        </div>`;
                }} else {{
                    resultDiv.innerHTML = `
                        <div style="padding: 15px; background-color: #e2e3e5; color: #383d41; border-radius: 5px; margin-top: 10px;">
                            <h3>🌫️ Dunstig / Nebelige Wetterlage</h3>
                            <p>Die KI erkennt Nebel oder Dunst. Ein dicker Schirm ist vermutlich nicht nötig.</p>
                        </div>`;
                }}
            }} catch (error) {{
                statusDiv.style.backgroundColor = "#f8d7da";
                statusDiv.style.color = "#721c24";
                statusDiv.innerText = "Fehler bei der KI-Berechnung: " + error.message;
            }}
        }}

        runKI();
    </script>
    """
    
    components.html(html_code, height=250)
