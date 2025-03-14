import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json

# Load model
@st.cache_resource
def load_my_model():
    return load_model('model_1.h5')

model = load_my_model()

# Konfigurasi
class_labels = ['Mengantuk & Menguap', 'Mengantuk & Tidak Menguap', 'Menguap & Tidak Mengantuk']
FRAME_WINDOW = st.image([])
stop_button = st.button('Hentikan')

# Session state untuk kontrol kamera
if 'running' not in st.session_state:
    st.session_state.running = True

# Sidebar untuk informasi
def setup_sidebar():
    with st.sidebar:
        st.header("Pengaturan & Informasi")
        st.markdown("""
        **Klasifikasi Status Pengemudi:**
        1. 😴 Mengantuk & Menguap
        2. 😪 Mengantuk & Tidak Menguap
        3. 🥱 Menguap & Tidak Mengantuk
        """)
        confidence_bar = st.progress(0)
        status_text = st.empty()
    return confidence_bar, status_text

confidence_bar, status_text = setup_sidebar()

# Fungsi prediksi
def predict_frame(frame):
    resized = cv2.resize(frame, (64, 64))
    normalized = resized / 255.0
    input_frame = np.expand_dims(normalized, axis=0)
    prediction = model.predict(input_frame, verbose=0)
    return prediction

# Fungsi untuk menangani frame video dan prediksi
def process_frame(frame):
    prediction = predict_frame(frame)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]
    label = f"{class_labels[class_index]} ({confidence:.2%})"
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.putText(frame, label, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    return frame, class_index, confidence

# Fungsi utama untuk menjalankan aplikasi
def main():
    cap = cv2.VideoCapture(0)
    
    while st.session_state.running and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal mengambil frame dari kamera")
            break
        
        frame, class_index, confidence = process_frame(frame)
        
        FRAME_WINDOW.image(frame)
        confidence_bar.progress(float(confidence))
        status_text.markdown(f"""
        **Status Terkini:**
        - 🎯 Kategori: {class_labels[class_index]}
        - 🔍 Keyakinan: {confidence:.2%}
        """)
    
    if stop_button:
        st.session_state.running = False
        cap.release()
        st.experimental_rerun()
    
    cap.release()

if __name__ == "__main__":
    main()






      