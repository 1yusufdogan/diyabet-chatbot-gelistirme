import sys
import os

# Üst klasörü Python yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

# MODELLERİ DOĞRU İMPORT ET
from models.huggingface_model_1 import DiabetesModelBERT
from models.huggingface_model_2 import DiabetesModelT5

# Ortam değişkenlerini yükle
load_dotenv()

# Sayfa ayarları
st.set_page_config(
    page_title="🩺 Diyabet Danismani Chatbotu",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar bölümü
with st.sidebar:
    # Logo dosyası yolu
    logo_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'diabetes_logo.png')
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.warning("Logo bulunamadi: data/diabetes_logo.png")

    st.markdown("""
    # 🩺 Diyabet Danismani Chatbotu
    Bu chatbot kan sekeri takibi, beslenme, insulin bilgisi ve genel diyabet danismanligi icin size yardimci olur.
    """)

    st.title("⚙️ Ayarlar")

    if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedback = {}
        st.rerun()

    st.markdown("### Istatistikler")
    if "messages" in st.session_state:
        st.markdown(f"Toplam mesaj: {len(st.session_state.messages)}")

# Model seçimi
model_option = st.sidebar.selectbox("Model Sec:", ["BERT", "T5"])

# Model yükleyici
@st.cache_resource
def load_model():
    if model_option == "T5":
        return DiabetesModelT5()
    else:
        return DiabetesModelBERT()

# Model yükleniyor
with st.spinner("Model yukleniyor..."):
    model = load_model()

# Başlık
st.title("🩺 Diyabet Danismani Chatbotu")

# Sohbet geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# Sohbet geçmişini göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Yeni mesaj girişi
prompt = st.chat_input("Sorunuzu yazin...")
if prompt:
    message_id = str(int(time.time()))
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "id": message_id,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Yanıt hazirlaniyor..."):
            response = model.get_response(prompt)
            st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "id": str(int(time.time())),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
