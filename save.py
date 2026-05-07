import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# ================= CONFIG =================
IMG_SIZE = (224, 224)
THRESHOLD = 0.35
CLASS_NAMES = ["anemic", "non-anemic"]

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("uji-6.keras")

model = load_model()

st.title("Klasifikasi Anemia")

# ================= SESSION STATE =================
if "result" not in st.session_state:
    st.session_state.result = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ================= FILE UPLOADER =================
uploaded_file = st.file_uploader(
    "Upload gambar",
    type=["jpg", "jpeg", "png"],
    key=st.session_state.uploader_key
)

# ================= BUTTON =================
col1, col2 = st.columns(2)

with col1:
    classify_btn = st.button(
        "Klasifikasi",
        disabled=(uploaded_file is None)
    )

with col2:
    if st.button("Refresh"):
        st.session_state.result = None
        st.session_state.uploader_key += 1  # reset uploader
        st.rerun()

# ================= PREVIEW =================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image.resize((300, 300)), caption="Preview")

# ================= PROSES =================
if classify_btn and uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # preprocessing cepat
    image = image.resize(IMG_SIZE)
    img = np.array(image)
    img = np.expand_dims(img, axis=0)

    with st.spinner("Menganalisis gambar..."):
        prob_non = model.predict(img, verbose=0)[0][0]

    predicted_class = CLASS_NAMES[1] if prob_non > THRESHOLD else CLASS_NAMES[0]

    # simpan hasil
    st.session_state.result = {
        "prob_non": prob_non,
        "predicted_class": predicted_class
    }

# ================= OUTPUT =================
if st.session_state.result is not None:

    prob_non = st.session_state.result["prob_non"]
    predicted_class = st.session_state.result["predicted_class"]

    st.write("## Hasil Prediksi")

    if predicted_class == "anemic":
        st.error("Anemia")
    else:
        st.success("Non-Anemia")

    # ================= CHART =================
    prob_data = pd.DataFrame({
        "Kelas": ["Anemia", "Non-Anemia"],
        "Probabilitas": [1 - prob_non, prob_non]
    })

    st.bar_chart(prob_data.set_index("Kelas"))

    # ================= ANGKA =================
    st.write(f"Anemia: {1 - prob_non:.4f}")
    st.write(f"Non-Anemia: {prob_non:.4f}")