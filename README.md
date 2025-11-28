# Movie Recommendation System (Two-Tower Model)

Bu proje, MovieLens veri seti kullanarak **two-tower (user–item)** mimarisiyle bir film öneri sistemi kurar.  
Arka tarafta **FastAPI**, ön tarafta ise **Streamlit** kullanılarak basit bir web arayüzü sağlanır.

## İçerik

- `vektor.py` – Kullanıcı ve film embedding'lerini eğiten PyTorch script'i
- `app.py` – Embedding'leri yükleyip tavsiye üreten FastAPI servisi
- `stream.py` – Kullanıcıya film önerilerini gösteren Streamlit arayüzü
- `embeddings/` – Eğitim sonrası oluşan `.npy` embedding dosyaları 
- `requirements.txt` – Gerekli Python paketleri

---