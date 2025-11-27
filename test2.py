# app.py
import os
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_API_KEY = "a04036f2ffcc835357cdefdc201ef8cf" # .env veya ortam değişkeninden al
TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

st.set_page_config(page_title="TF-IDF + Cosine Film Öneri", layout="centered")
st.title("🎬 Basic İçerik Tabanlı Film Öneri")

def fetch_popular(api_key: str, language: str = "en-US", pages: int = 2):
    """TMDB popüler filmlerini (title/overview/poster) çeker."""
    items = []
    for p in range(1, pages + 1):
        url = f"{TMDB_API_BASE}/movie/popular"
        r = requests.get(url, params={"api_key": api_key, "language": language, "page": p}, timeout=10)
        r.raise_for_status()
        items.extend(r.json().get("results", []))
    # sadeleştir
    seen = set()
    movies = []
    for it in items:
        mid = it.get("id")
        if mid and mid not in seen:
            seen.add(mid)
            movies.append({
                "id": mid,
                "title": it.get("title") or it.get("name") or "Untitled",
                "overview": it.get("overview") or "",
                "poster": it.get("poster_path"),
            })
    return movies

def top_k_similar_texts(texts, k=5, idx=0):
   
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    X = vec.fit_transform(texts)
    sims = cosine_similarity(X[idx], X).flatten()
    order = sims.argsort()[::-1] 
    order = [i for i in order if i != idx][:k]
    return [(i, float(sims[i])) for i in order]

# --- Ana UI ---
with st.expander("🔑 API anahtarı (opsiyonel panel)"):
    st.write("TMDB_API_KEY ortam değişkeni yoksa buraya girebilirsin.")
    if not TMDB_API_KEY:
        TMDB_API_KEY = st.text_input("TMDB_API_KEY", type="password")

lang = st.selectbox("Dil (overview için)", ["en-US", "tr-TR"], index=0)
pages = st.slider("Kaç sayfa popüler film çekilsin? (~20/sayfa)", 1, 3, 2)

if not TMDB_API_KEY:
    st.warning("TMDB_API_KEY gerekli. Ortam değişkeni ayarla veya yukarıdan gir.")
    st.stop()

# Veri çek
try:
    movies = fetch_popular(TMDB_API_KEY, language=lang, pages=pages)
except requests.RequestException as e:
    st.error(f"TMDB isteği başarısız: {e}")
    st.stop()

if len(movies) < 2:
    st.info("Öneri için yeterli film yok.")
    st.stop()

titles = [m["title"] for m in movies]
sel_title = st.selectbox("Referans film seç:", titles, index=0)
sel_idx = titles.index(sel_title)
sel_movie = movies[sel_idx]

# Benzerleri hesapla
indices_scores = top_k_similar_texts([m["overview"] for m in movies], k=5, idx=sel_idx)

# Çıktı
st.subheader("🎯 Seçilen Film")
if sel_movie["poster"]:
    st.image(TMDB_IMG_BASE + sel_movie["poster"], width=250, caption=sel_movie["title"])
st.write(sel_movie["overview"][:300] + ("…" if len(sel_movie["overview"]) > 300 else ""))

st.markdown("### 🔎 En Benzer 5 Film")
cols = st.columns(5)
for i, (idx, score) in enumerate(indices_scores):
    m = movies[idx]
    with cols[i % 5]:
        if m["poster"]:
            st.image(TMDB_IMG_BASE + m["poster"], use_column_width=True)
        st.caption(m["title"])
        st.write(f"Benzerlik: **{score:.3f}**")
