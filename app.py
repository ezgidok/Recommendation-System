from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
import faiss

app = FastAPI()


try:
    user_embeddings_raw = np.load("embeddings/twotower_user_embeddings.npy")
    item_embeddings_raw = np.load("embeddings/twotower_item_embeddings.npy", allow_pickle=True)
    user_to_idx = np.load("embeddings/twotower_user_to_idx.npy", allow_pickle=True).item()
    movie_to_idx = np.load("embeddings/twotower_movie_to_idx.npy", allow_pickle=True).item()
    ratings_df = pd.read_csv("/Users/ezgidok/Downloads/ml-latest-small/ratings.csv")
    links_df=pd.read_csv("/Users/ezgidok/Downloads/ml-latest-small/links.csv")
except FileNotFoundError as e:
    raise RuntimeError(f"Gerekli veri dosyaları bulunamadı: {e}. Lütfen dosya yollarını kontrol edin.")
except Exception as e:
    raise RuntimeError(f"Veri yüklenirken bir hata oluştu: {e}")

links_df = links_df.dropna(subset=['tmdbId'])        
links_df['tmdbId'] = links_df['tmdbId'].astype(int) 
movieid_to_tmdbid=dict(zip(links_df['movieId'], links_df['tmdbId'])) 
 
user_embeddings = user_embeddings_raw.astype(np.float32)
user_norms = np.linalg.norm(user_embeddings, axis=1, keepdims=True)
user_norms[user_norms == 0] = 1
user_embeddings = user_embeddings / user_norms
user_embeddings = np.nan_to_num(user_embeddings)

item_embeddings = item_embeddings_raw.astype(np.float32)
item_norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
item_norms[item_norms == 0] = 1
item_embeddings = item_embeddings / item_norms
item_embeddings = np.nan_to_num(item_embeddings)

idx_to_user = {v: k for k, v in user_to_idx.items()}
idx_to_movie = {v: k for k, v in movie_to_idx.items()}


try:
    d_user = user_embeddings.shape[1] 
    index_user = faiss.IndexFlatIP(d_user) 
    index_user.add(user_embeddings) 

    d_item = item_embeddings.shape[1] 
    index_item = faiss.IndexFlatIP(d_item) 
    index_item.add(item_embeddings) 
except Exception as e:
    raise RuntimeError(f"FAISS endeksleri oluşturulurken bir hata oluştu: {e}")


@app.get("/")
def root():
    return {"message": "Two-Tower Embedding API calisiyor!"}

##@app.get("/user_embedding/{user_id}")
##def get_user_embedding(user_id: int):
    if user_id not in user_to_idx:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı.")
    idx = user_to_idx[user_id]
    return user_embeddings[idx].tolist()


##@app.get("/movie_embeddings/{movie_id}")
##def get_item_embedding(movie_id: int):
    if movie_id not in movie_to_idx:
        raise HTTPException(status_code=404, detail="Film bulunamadı.")
    idx = movie_to_idx[movie_id]
    return item_embeddings[idx].tolist()


@app.get("/similar_users/{user_id}")
def get_similar_users(user_id: int, top_k: int = 5):

    if user_id not in user_to_idx:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı.")

    try:
        target_idx = user_to_idx[user_id]
        target_vec = user_embeddings[target_idx].reshape(1, -1) # FAISS için 2D dizi bekler

        print(f"DEBUG: similar_users için target_idx: {target_idx}, target_vec.shape: {target_vec.shape}")

       #cosine benzerliğini distances ifade ediyor
        distances, indices = index_user.search(target_vec, top_k + 1) 
        similar_indices = [idx for idx in indices[0] if idx != target_idx][:top_k]

        similar_user_ids = [int(idx_to_user[idx]) for idx in similar_indices if idx in idx_to_user]

        return {"similar_users": similar_user_ids}
    except Exception as e:
        print(f"Hata: /similar_users/{user_id} endpoint'inde bir sorun oluştu: {e}")
        raise HTTPException(status_code=500, detail=f"Benzer kullanıcılar aranırken dahili bir hata oluştu: {e}. Lütfen sunucu loglarını kontrol edin.")


@app.get("/similar_items/{item_id}") 
def get_similar_items(item_id: int, top_k: int = 5):
    if item_id not in movie_to_idx: 
        raise HTTPException(status_code=404, detail="Film bulunamadı.")

    try:
        target_idx = movie_to_idx[item_id]
        target_vec = item_embeddings[target_idx].reshape(1, -1) # FAISS için 2D dizi bekler

        print(f"DEBUG: similar_items için target_idx: {target_idx}, target_vec.shape: {target_vec.shape}")
 
        # FAISS kullanarak benzer filmleri ara
        # D: Uzaklıklar/Benzerlikler, I: Endeksler
        distances, indices = index_item.search(target_vec, top_k + 1) 

        similar_indices = [idx for idx in indices[0] if idx != target_idx][:top_k]

        similar_movie_ids = [int(idx_to_movie[idx]) for idx in similar_indices if idx in idx_to_movie]

        return {"similar_items": similar_movie_ids}
    except Exception as e:
        print(f"Hata: /similar_items/{item_id} endpoint'inde bir sorun oluştu: {e}")
        raise HTTPException(status_code=500, detail=f"Benzer filmler aranırken dahili bir hata oluştu: {e}. Lütfen sunucu loglarını kontrol edin.")


@app.get("/user_top_rated_movies/{userId}")
def get_top_rated_movies(userId: int, top_k: int = 5):
    user_ratings = ratings_df[ratings_df['userId'] == userId]

    if user_ratings.empty:
        raise HTTPException(status_code=404, detail="Kullanıcının değerlendirmesi bulunamadı.")

    top_movies = user_ratings.sort_values(by='rating', ascending=False).head(top_k)

    return [int(movie_id) for movie_id in top_movies['movieId'].tolist()]


@app.get("/recommendations/{user_id}")
def recommend_movies(user_id: int, top_k: int = 10):
    if user_id not in user_to_idx:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı.")

    try:
        target_idx = user_to_idx[user_id]
        target_vec = user_embeddings[target_idx].reshape(1, -1)

        distances, indices = index_item.search(target_vec, 100)  
        candidate_movie_indices = indices[0]
        print(f"Candidate movie count: {len(candidate_movie_indices)}")

        watched_movie_ids = set(ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist())

        recommended_movie_ids = []
        for idx in candidate_movie_indices:
            movie_id = idx_to_movie.get(idx)
            if movie_id is None:
                continue
            if movie_id in watched_movie_ids:
                continue
            recommended_movie_ids.append(movie_id) 
            if len(recommended_movie_ids) >= top_k:
                break

        return {"recommended_movies": [int(movie_id) for movie_id in recommended_movie_ids]}  

    except Exception as e:
        print(f"Hata: /recommendations/{user_id} endpoint'inde sorun oluştu: {e}")
        raise HTTPException(status_code=500, detail=f"Öneri oluşturulurken bir hata oluştu: {e}")


    except Exception as e:
        print(f"Hata: /recommendations/{user_id} endpoint'inde sorun oluştu: {e}")
        raise HTTPException(status_code=500, detail=f"Öneri oluşturulurken bir hata oluştu: {e}")
