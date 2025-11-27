import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8001"
TMDB_API_KEY = "a04036f2ffcc835357cdefdc201ef8cf"

result_count = st.slider("Rec count", 1, 30, 5)

@st.cache_data
def load_links():
    links_df = pd.read_csv("/Users/ezgidok/Downloads/ml-latest-small/links.csv")
    links_df = links_df.dropna(subset=['tmdbId'])
    links_df['tmdbId'] = links_df['tmdbId'].astype(int)
    return dict(zip(links_df['movieId'], links_df['tmdbId']))

movieid_to_tmdbid = load_links()

def get_similar_users_from_api(user_id, top_k=5):
    try:
        response = requests.get(f"{API_URL}/similar_users/{user_id}?top_k={top_k}")
        response.raise_for_status() 
        return response.json()["similar_users"]
    except requests.exceptions.RequestException as e:
        st.error(f"API'den benzer kullanıcılar çekilemedi: {e}")
        return None
    
def get_user_top_rated_movies_from_api(user_id, top_k=5):
    try:
        response = requests.get(f"{API_URL}/user_top_rated_movies/{user_id}?top_k={top_k}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Kullanıcının en çok puan verdiği filmler çekilemedi: {e}")
        return None

def get_tmdb_movie_details(tmdb_id):
    try:
        response = requests.get(f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&language=en-US")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None
    
def get_recommendations_from_api(user_id, top_k=5):
    try:
        response = requests.get(f"{API_URL}/recommendations/{user_id}?top_k={top_k}")
        response.raise_for_status()
        return response.json()["recommended_movies"]
    except requests.exceptions.RequestException as e:
        st.error(f"API'den önerilen filmler çekilemedi: {e}")
        return None

st.title("film öneri sistemi")
user_id = st.number_input("Kullanıcı ID'si Girin:", min_value=1, value=1, step=1)

if st.button("Benzer kullanıların izlediklerini bul"):
    if user_id:
        with st.spinner("Benzer aranıyor..."):
            similar_users = get_similar_users_from_api(user_id, top_k=3)
            if similar_users:
                st.subheader(f"Kullanıcı {user_id}'ye Benzer Kullanıcılar:")
                for i, similar_user_id in enumerate(similar_users):
                    st.markdown(f"#### {i+1}. Benzer Kullanıcı ID: {similar_user_id}")
                    user_watched_movies = get_user_top_rated_movies_from_api(similar_user_id, top_k=3)
                    
                    if user_watched_movies:
                        for movie_id in user_watched_movies:
                            tmdb_id = movieid_to_tmdbid.get(movie_id)
                            if tmdb_id:
                                movie_details = get_tmdb_movie_details(tmdb_id)
                                if movie_details:
                                    st.write(f"Id: {movie_id}, tmdb_id {tmdb_id}")
                                    st.write(f"- Film Adı: {movie_details['title']}")
                                    if movie_details.get('poster_path'):
                                        st.image(f"https://image.tmdb.org/t/p/w500{movie_details['poster_path']}", width=150)
                                    st.markdown("---")
                            else:
                                st.warning(f"Eşleşmeyen MovieLens ID: {movie_id}")
                    else:
                        st.warning(f"Kullanıcı {similar_user_id} için film detayı bulunamadı.")

if st.button("Kullanıcıya Önerilen Filmleri Göster"):
    if user_id:
        with st.spinner("Öneriler getiriliyor..."):
            recommended_movie_ids = get_recommendations_from_api(user_id, top_k=result_count)
            st.write(f"Recommend ids length: {len(recommended_movie_ids)}")
            if recommended_movie_ids:
                st.subheader(f"Kullanıcı {user_id} için Önerilen Filmler:")
                for movie_id in recommended_movie_ids:
                    tmdb_id = movieid_to_tmdbid.get(movie_id)
                    if tmdb_id:
                        movie_details = get_tmdb_movie_details(tmdb_id)
                        if movie_details:
                            st.write(f"**{movie_details['title']}**")
                            if movie_details.get('poster_path'):
                                st.image(f"https://image.tmdb.org/t/p/w500{movie_details['poster_path']}", width=150)
                        else:
                            st.warning(f"TMDB'den veri alınamadı: {tmdb_id}")
                    else:
                        st.warning(f"Eşleşmeyen MovieLens ID: {movie_id}")
                    st.markdown("---")
            else:
                st.warning("Önerilen film bulunamadı.")
