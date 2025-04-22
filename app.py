import streamlit as st
import pandas as pd
import torch
from recommender import RecSysModel, MovieDataset, evaluate_content_based

st.title("Movie Recommender System")

# --- Load data ---
@st.cache_data
def load_metadata():
    return pd.read_csv('datasets/movies_metadata.csv')

@st.cache_data
def load_keywords():
    import ast
    df = pd.read_csv('datasets/keywords.csv')
    df['keywords'] = df['keywords'].fillna('[]').apply(ast.literal_eval)
    return df

def load_model():
    try:
        model = torch.load('models/recsysmodel.pth', map_location=torch.device('cpu'))
        return model
    except Exception:
        return None

metadata = load_metadata()
keywords_df = load_keywords()

# --- Merge keywords with metadata ---
metadata['id'] = metadata['id'].astype(str)
keywords_df['id'] = keywords_df['id'].astype(str)
metadata = metadata.merge(keywords_df, how='left', left_on='id', right_on='id')
# Use the keywords column from keywords.csv ('keywords_y') if present
if 'keywords_y' in metadata.columns:
    metadata['keywords'] = metadata['keywords_y']
    metadata = metadata.drop(columns=[col for col in ['keywords_x','keywords_y'] if col in metadata.columns])
else:
    metadata['keywords'] = [[] for _ in range(len(metadata))]
metadata['keywords'] = metadata['keywords'].apply(lambda x: x if isinstance(x, list) else [])

# --- Extract all genres and keywords ---
import ast
all_genres = set()
for genres_str in metadata['genres'].fillna('[]'):
    try:
        genres = ast.literal_eval(genres_str)
        for g in genres:
            all_genres.add(g['name'])
    except Exception:
        pass
all_genres = sorted(list(all_genres))

all_keywords = set()
for kwlist in metadata['keywords']:
    for kw in kwlist:
        all_keywords.add(kw['name'])
all_keywords = sorted(list(all_keywords))

# --- User Authentication/Profile ---
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'user_ratings' not in st.session_state:
    st.session_state['user_ratings'] = {}

st.sidebar.header("User Profile")
username = st.sidebar.text_input("Enter your username:", st.session_state['username'])
if st.sidebar.button("Login"):
    st.session_state['username'] = username
    st.success(f"Logged in as {username}")

# --- Filtering options ---
st.sidebar.header('Filter Movies')
selected_genres = st.sidebar.multiselect('Genres', all_genres)
selected_keywords = st.sidebar.multiselect('Keywords', all_keywords)

# Additional filters
def safe_float(val, default=0.0):
    try:
        return float(val)
    except:
        return default

def safe_int(val, default=0):
    try:
        return int(val)
    except:
        return default

vote_avg_min, vote_avg_max = float(metadata['vote_average'].min()), float(metadata['vote_average'].max())
vote_count_min, vote_count_max = safe_int(metadata['vote_count'].min()), safe_int(metadata['vote_count'].max())
year_min, year_max = 1900, 2025
if 'release_date' in metadata.columns:
    years = metadata['release_date'].dropna().apply(lambda x: safe_int(str(x)[:4], 0))
    year_min, year_max = years[years > 0].min(), years[years > 0].max()
orig_langs = sorted(metadata['original_language'].dropna().unique().tolist())
runtime_min, runtime_max = safe_float(metadata['runtime'].min()), safe_float(metadata['runtime'].max())

selected_vote_avg = st.sidebar.slider('Vote Average (Rating)', vote_avg_min, vote_avg_max, (vote_avg_min, vote_avg_max))
selected_vote_count = st.sidebar.slider('Vote Count (Popularity)', vote_count_min, vote_count_max, (vote_count_min, vote_count_max))
selected_year = st.sidebar.slider('Release Year', year_min, year_max, (year_min, year_max))
selected_lang = st.sidebar.multiselect('Original Language', orig_langs)
selected_runtime = st.sidebar.slider('Runtime (min)', runtime_min, runtime_max, (runtime_min, runtime_max))

# Filter movies by genres, keywords, and new filters
filtered_metadata = metadata.copy()
if selected_genres:
    def has_genre(genres_str):
        try:
            genres = ast.literal_eval(genres_str)
            return any(g['name'] in selected_genres for g in genres)
        except Exception:
            return False
    filtered_metadata = filtered_metadata[filtered_metadata['genres'].apply(has_genre)]
if selected_keywords:
    def has_keyword(kwlist):
        return any(kw['name'] in selected_keywords for kw in kwlist)
    filtered_metadata = filtered_metadata[filtered_metadata['keywords'].apply(has_keyword)]
filtered_metadata = filtered_metadata[
    (filtered_metadata['vote_average'] >= selected_vote_avg[0]) &
    (filtered_metadata['vote_average'] <= selected_vote_avg[1]) &
    (filtered_metadata['vote_count'] >= selected_vote_count[0]) &
    (filtered_metadata['vote_count'] <= selected_vote_count[1])
]
if 'release_date' in filtered_metadata.columns:
    filtered_metadata = filtered_metadata[
        filtered_metadata['release_date'].fillna('0').apply(lambda x: safe_int(str(x)[:4], 0)).between(selected_year[0], selected_year[1])
    ]
if selected_lang:
    filtered_metadata = filtered_metadata[filtered_metadata['original_language'].isin(selected_lang)]
filtered_metadata = filtered_metadata[
    (filtered_metadata['runtime'] >= selected_runtime[0]) &
    (filtered_metadata['runtime'] <= selected_runtime[1])
]

movie_list = filtered_metadata['title'].dropna().unique().tolist()
selected_movie = st.selectbox("Select a movie you like:", movie_list)

if username:
    st.write(f"Welcome, {username}! Rate movies to improve your recommendations.")
    rating = st.slider(f"Rate '{selected_movie}' (1-5):", 1, 5, 3)
    if st.button("Save Rating"):
        st.session_state['user_ratings'][selected_movie] = rating
        st.success(f"Saved rating {rating} for '{selected_movie}'!")
    st.write("Your ratings:")
    st.json(st.session_state['user_ratings'])

# --- Content-Based Recommendations ---
if st.button("Get Content-Based Recommendations"):
    st.write("### Content-Based Recommendations:")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    df = metadata.dropna(subset=['overview'])
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['overview'])
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    idx = indices[selected_movie]
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    st.write(df['title'].iloc[movie_indices].tolist())

# --- Collaborative Filtering Recommendations ---
model = load_model()
if username and model is not None and st.button("Get Collaborative Filtering Recommendations"):
    st.write("### Collaborative Filtering Recommendations:")
    # For demo: recommend top movies by predicted rating for this user
    user_idx = 0  # In real app, map username to user index
    movie_indices = list(range(len(movie_list)))
    user_tensor = torch.tensor([user_idx]*len(movie_indices), dtype=torch.long)
    movie_tensor = torch.tensor(movie_indices, dtype=torch.long)
    ratings_tensor = torch.zeros(len(movie_indices), dtype=torch.float)
    with torch.no_grad():
        preds, _, _ = model(user_tensor, movie_tensor, ratings_tensor)
    top_idx = preds.squeeze().argsort(descending=True)[:10]
    rec_movies = [movie_list[i] for i in top_idx]
    st.write(rec_movies)

# --- LightFM Recommendations ---
import os
import pickle
from lightfm import LightFM
from lightfm.data import Dataset as LFM_Dataset

def load_lightfm_model():
    model_path = 'models/lightfm_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model, lfm_dataset, movie_ids = pickle.load(f)
        return model, lfm_dataset, movie_ids
    else:
        # Train on the fly if not present
        df = pd.read_csv('datasets/ratings_small.csv')
        lfm_dataset = LFM_Dataset()
        users = df['user'].unique().tolist()
        movies = df['movie'].unique().tolist()
        lfm_dataset.fit(users, movies)
        interactions, _ = lfm_dataset.build_interactions([(row['user'], row['movie'], row['rating']) for _, row in df.iterrows()])
        model = LightFM(loss='warp')
        model.fit(interactions, epochs=10, num_threads=2)
        movie_ids = movies
        with open(model_path, 'wb') as f:
            pickle.dump((model, lfm_dataset, movie_ids), f)
        return model, lfm_dataset, movie_ids

if username and st.button("Get LightFM Recommendations"):
    st.write("### LightFM Hybrid Recommendations:")
    model, lfm_dataset, movie_ids = load_lightfm_model()
    # For demo: use username as user_id (should match train_v2.csv)
    user_id = username
    if user_id not in lfm_dataset.mapping()[0]:
        st.warning(f"User '{user_id}' not found in training data.")
    else:
        user_x = lfm_dataset.mapping()[0][user_id]
        movie_xs = [lfm_dataset.mapping()[2][mid] for mid in movie_ids]
        scores = model.predict(user_ids=user_x, item_ids=movie_xs)
        top_indices = np.argsort(-scores)[:10]
        rec_movies = [movie_ids[i] for i in top_indices]
        st.write(rec_movies)

# --- Hybrid Recommendation Placeholder ---
if username and st.button("Get Hybrid Recommendations"):
    st.info("Hybrid recommendations coming soon! (Combine content and collaborative filtering)")
