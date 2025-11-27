import torch
import torch.nn as nn
import torch.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


rating_file="/Users/ezgidok/Downloads/ml-latest-small/ratings.csv"
movies_file="/Users/ezgidok/Downloads/ml-latest-small/movies.csv"
rating_df=pd.read_csv(rating_file)
movies_df=pd.read_csv(movies_file)


unique_users = rating_df['userId'].unique()
unique_movies = rating_df['movieId'].unique()
user_to_idx = {user: idx for idx, user in enumerate(sorted(unique_users))}
movie_to_idx = {movie: idx for idx, movie in enumerate(sorted(unique_movies))}
idx_to_user = {idx: user for user, idx in user_to_idx.items()}
idx_to_movie = {idx: movie for movie, idx in movie_to_idx.items()}


rating_df['user_idx'] = rating_df['userId'].map(user_to_idx)
rating_df['movie_idx'] = rating_df['movieId'].map(movie_to_idx)

users = torch.tensor(rating_df['user_idx'].values, dtype=torch.long)
movies = torch.tensor(rating_df['movie_idx'].values, dtype=torch.long)
ratings = torch.tensor(rating_df['rating'].values, dtype=torch.float32)

num_users = len(unique_users)
num_movies = len(unique_movies)

all_genres = set()
for g in movies_df['genres']:
    for genre in g.split('|'):
        if genre != '(no genres listed)':
            all_genres.add(genre)

genre_to_idx = {genre: i for i, genre in enumerate(sorted(all_genres))}
num_genres = len(genre_to_idx)

print("Genre mapping:", genre_to_idx)

genre_to_idx = {genre: i for i, genre in enumerate(sorted(all_genres))}
num_genres = len(genre_to_idx)

def encode_genres(genre_str):
    vec = np.zeros(num_genres)
    for genre in genre_str.split('|'):
        if genre in genre_to_idx:
            vec[genre_to_idx[genre]] = 1
    return vec


movies_df['genre_vector'] = movies_df['genres'].apply(encode_genres)
genre_matrix = torch.tensor(np.stack(movies_df['genre_vector']), dtype=torch.float32)
# her rating için ilgili filmin genre vektörünü al
genres_for_batch = genre_matrix[rating_df['movie_idx'].values]



class UserTower(nn.Module):
    def __init__(self, num_users, embed_dim):
        super(UserTower, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)

    def forward(self, user_ids):
        return self.user_embedding(user_ids)

class ItemTower(nn.Module):
    def __init__(self, num_movies,num_genres, embed_dim):
        super(ItemTower, self).__init__()
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)
        self.genre_layer = nn.Linear(num_genres, embed_dim)
       ## self.genre_embeddings = None # multi-hot encoder
       ## self.genre_embeddings_layer = nn.Linear(5)
        #self.fc = nn.Linear(embed_dim*2, embed_dim)


    def forward(self, movie_ids,genre_vectors):
        # concat işlemi
        movie_emb = self.movie_embedding(movie_ids)   
        genre_emb = self.genre_layer(genre_vectors)
        return movie_emb
        x = torch.cat([movie_emb, genre_emb], dim=1)
        return self.fc(x)

class TwoTower(nn.Module):
    def __init__(self, num_users, num_movies,num_genres, embed_dim):
        super(TwoTower, self).__init__()
        self.user_tower = UserTower(num_users, embed_dim)
        self.item_tower = ItemTower(num_movies,num_genres, embed_dim)

    def forward(self, user_ids, movie_ids,genre_vectors):
        user_embedding = self.user_tower(user_ids)
        movie_embedding = self.item_tower(movie_ids,genre_vectors)
        score = torch.sigmoid(torch.sum(user_embedding * movie_embedding, dim=1))
        return score
    
    

embed_dim = 128
model = TwoTower(num_users, num_movies,num_genres, embed_dim)
criter = nn.BCELoss() # Ortalama Kare Hata, regresyon için uygun
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
binary_ratings = (ratings >= 3.5).float()


epochs =300
for epoch in range(epochs):
    predictions = model(users, movies,genres_for_batch)
    loss = criter(predictions, binary_ratings)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

   
with torch.no_grad():
    preds = model(users, movies,genres_for_batch)
    preds_class = (preds >= 0.5).int()       # tahminleri binary
    labels = (ratings >= 3.5).int()          # gerçek etiketleri binary
    
    acc = accuracy_score(labels, preds_class)
    prec = precision_score(labels, preds_class, zero_division=0)
    rec = recall_score(labels, preds_class, zero_division=0)
    f1 = f1_score(labels, preds_class, zero_division=0)


import os

def save_embeddings_and_mappings(model, user_to_idx, movie_to_idx, file_prefix="twotower"):
    os.makedirs("embeddings", exist_ok=True)

    # User embeddings
    user_embeddings = model.user_tower.user_embedding.weight.detach().cpu().numpy()
    np.save(f"embeddings/{file_prefix}_user_embeddings.npy", user_embeddings)

    movie_ids = torch.arange(num_movies)
    genre_vectors = genre_matrix
    with torch.no_grad():
        item_embeddings = model.item_tower(movie_ids, genre_vectors).detach().cpu().numpy()
    np.save(f"embeddings/{file_prefix}_item_embeddings.npy", item_embeddings)

    np.save(f"embeddings/{file_prefix}_user_to_idx.npy", user_to_idx)
    np.save(f"embeddings/{file_prefix}_movie_to_idx.npy", movie_to_idx)

    print("Embeddings ve mapping sözlükleri kaydedildi.")

save_embeddings_and_mappings(model, user_to_idx, movie_to_idx)


print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")


print("Eğitim tamamlandı!")

