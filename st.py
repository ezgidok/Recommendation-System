# app.py
import streamlit as st
import pandas as pd

# --- Sayfa başlığı ---
st.set_page_config(page_title="Streamlit Staj Uygulaması", layout="centered")
st.title("FORM SİSTEMİ TEST")

# --- 1. Kullanıcı formu ---
st.header("👤 Kullanıcı Bilgileri")
with st.form("user_form"):
    ad = st.text_input("Adınız")
    soyad = st.text_input("Soyadınız")
    yas = st.number_input("Yaşınız", min_value=0, max_value=120, step=1)
    submit_btn = st.form_submit_button("Gönder")
    if submit_btn:
        st.success(f"Hoş geldiniz {ad} {soyad}, yaşınız: {yas}")

# --- 2. Film seçme bölümü ---
st.header("🎥 Film Seçimi")

# Örnek film dataseti (normalde CSV'den okunabilir)
film_verileri = pd.DataFrame({
    "film": ["Inception", "Interstellar", "The Matrix"],
    "gorsel": [
        "https://upload.wikimedia.org/wikipedia/en/7/7f/Inception_ver3.jpg",
        "https://upload.wikimedia.org/wikipedia/en/b/bc/Interstellar_film_poster.jpg",
        "https://upload.wikimedia.org/wikipedia/en/c/c1/The_Matrix_Poster.jpg"
    ]
})

film_secimi = st.selectbox("Bir film seçin:", film_verileri["film"])

if film_secimi:
    secilen_film = film_verileri[film_verileri["film"] == film_secimi].iloc[0]
    st.image(secilen_film["gorsel"], caption=film_secimi)

# --- 3. Metin analizi ---
st.header("📝 Metin Analizi")
metin = st.text_area("Metninizi buraya yazın:")

if metin:
    kelime_sayisi = len(metin.split())
    karakter_sayisi = len(metin)
    st.write(f"Kelime sayısı: **{kelime_sayisi}**")
    st.write(f"Karakter sayısı: **{karakter_sayisi}**")

# --- 4. Kenar çubuğunda iletişim tercihi ---
st.sidebar.header("📩 İletişim Tercihi")
iletisim = st.sidebar.radio(
    "Tercih ettiğiniz iletişim yöntemi:",
    ["E-posta", "Telefon", "Mesajlaşma Uygulaması"]
)

st.sidebar.write(f"Seçiminiz: **{iletisim}**")

# --- Sonuç ---
st.write("---")
st.success("Uygulama başarıyla çalışıyor ✅")
