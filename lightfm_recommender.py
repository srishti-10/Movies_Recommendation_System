import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
import numpy as np

# --- Load ratings data ---
def load_ratings(csv_path):
    df = pd.read_csv(csv_path)
    return df

def build_lightfm_dataset(df):
    dataset = Dataset()
    users = df['user'].unique().tolist()
    movies = df['movie'].unique().tolist()
    dataset.fit(users, movies)
    interactions, _ = dataset.build_interactions([(row['user'], row['movie'], row['rating']) for _, row in df.iterrows()])
    return dataset, interactions

def train_lightfm(interactions):
    model = LightFM(loss='warp')
    model.fit(interactions, epochs=10, num_threads=2)
    return model

def recommend_lightfm(model, dataset, user_id, movie_ids, n=10):
    user_x = dataset.mapping()[0][user_id]
    movie_xs = [dataset.mapping()[2][mid] for mid in movie_ids]
    scores = model.predict(user_ids=user_x, item_ids=movie_xs)
    top_indices = np.argsort(-scores)[:n]
    return [movie_ids[i] for i in top_indices]

if __name__ == "__main__":
    df = load_ratings('datasets/ratings_small.csv')
    dataset, interactions = build_lightfm_dataset(df)
    model = train_lightfm(interactions)
    # Example: recommend for a random user
    user_id = df['user'].iloc[0]
    movie_ids = df['movie'].unique().tolist()
    recs = recommend_lightfm(model, dataset, user_id, movie_ids)
    print(f"Recommendations for user {user_id}: {recs}")
    # Save model if needed
    # import pickle
    # with open('models/lightfm_model.pkl', 'wb') as f:
    #     pickle.dump((model, dataset), f)
