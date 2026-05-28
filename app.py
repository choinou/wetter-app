import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# 1. App-Konfiguration & Styling
st.set_page_config(page_title="Himmel-Scanner", page_icon="🌦️", layout="centered")

st.title("🌦️ KI Himmels-Scanner")
st.write("Diese App läuft komplett lokal und autark auf dem Server – ganz ohne Internet-API!")

# 2. KI-Modell lokal laden (MobileNetV2 - extrem schnell und leicht)
@st.cache_resource
def load_local_model():
    # Lädt das fertige Bilderkennungsmodell aus der Library
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval() # In den Vorhersage-Modus versetzen
    
    # Die originalen ImageNet-Klassennamen laden
    # Wir brauchen eine Liste der 1000 Dinge, die das Modell kennt
    from torchvision.models import MobileNet_V2_Weights
    categories = MobileNet_V2_Weights.DEFAULT.meta["categories"]
    
    return model, categories

with st.spinner("Lokales KI-Modell wird gestartet..."):
    model, categories = load_local_model()

st.markdown("---")

# 3. Auswahlmodus: Hochladen ODER Kamera via Tabs
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

# 4. Auswertung und Logik
if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    
    st.markdown("---")
    st.subheader("🧠 KI-Analyse läuft...")
    
    # Das Bild für das KI-Modell vorbereiten (Größe anpassen & normalisieren)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) # Dimension für das Modell anpassen

    with torch.no_grad():
        output = model(input_batch)
    
    # Wahrscheinlichkeiten berechnen
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 5)
    
    # Das wahrscheinlichste Objekt herausfinden
    best_match_label = categories[top_catid[0]].lower()
    
    # Wir schauen uns auch die Top 5 Ergebnisse an, ob da "Wolke" (cloud) drin vorkommt
    all_top_labels = [categories[top_catid[i]].lower() for i in range(5)]
    detected_labels_string = ", ".join(all_top_labels)

    st.info(f"**Die KI sieht im Bild vor allem:** `{best_match_label.replace('_', ' ')}`")
    # Entwickler-Info für dich zum Testen:
    st.write(f"*(Top-Erkennungen der KI: {detected_labels_string})*")

    # 5. Die Regenschirm-Entscheidungs-Logik
    st.subheader("☂️ Deine Empfehlung:")

    # Logik: Wenn irgendwas mit Wolken (cloud) oder Regen/Schirm im Bild ist
    if any("cloud" in label or "umbrella" in label or "scuba" in label for label in all_top_labels):
        st.error("🚨 **Nimm einen Regenschirm mit!**")
        st.write(
            "Die KI hat Wolkenstrukturen oder graue Muster im Bild erkannt. "
            "Es besteht die Gefahr, dass es bald in Lübeck regnet. Geh lieber auf Nummer sicher!"
        )
    else:
        st.success("😎 **Kein Regenschirm nötig!**")
        st.write(
            "Die KI sieht einen freien Himmel, Licht oder helle Flächen und keine typischen Regenwolken. "
            "Du kannst den Regenschirm heute wahrscheinlich zu Hause lassen!"
        )
