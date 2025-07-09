import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import re

@st.cache_data
def load_and_process_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    # Preprocessing
    movies = movies.head(5000)
    if 'Unnamed: 0' in movies.columns:
        movies.drop(columns=['Unnamed: 0'], inplace=True)

    # Removing year
    movies['title'] = movies['title'].apply(lambda x: re.sub(r'\s*\(\d{4}\)', '', x).strip())

    # Process genres
    movies['genres'] = movies['genres'].fillna('')
    movies['genres_list'] = movies['genres'].apply(lambda x: x.split('|'))
    all_genres = sorted(list(set(g for sublist in movies['genres_list'] for g in sublist if g)))
    for genre in all_genres:
        movies[genre] = movies['genres_list'].apply(lambda x: int(genre in x))
    movies.drop(columns=['genres_list'], inplace=True)

    # Collaborative filtering data
    merged = pd.merge(ratings, movies[['movieId', 'title']], on='movieId')
    user_item_matrix = merged.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    movie_user_matrix = user_item_matrix.T
    sparse_matrix = csr_matrix(movie_user_matrix.values)

    # KNN Model
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
    model_knn.fit(sparse_matrix)

    return movies, all_genres, movie_user_matrix, sparse_matrix, model_knn


# Load data
movies, all_genres, movie_user_matrix, sparse_matrix, model_knn = load_and_process_data()


def get_collab_recommendations(movie_title, n_recommendations=5):
    movie_title = movie_title.strip()
    if movie_title not in movie_user_matrix.index:
        return []

    try:
        movie_idx = movie_user_matrix.index.get_loc(movie_title)
        distances, indices = model_knn.kneighbors(sparse_matrix[movie_idx].reshape(1, -1),
                                                  n_neighbors=n_recommendations + 1)
        recommended = []
        for i in range(1, len(distances[0])):
            if i < len(indices[0]):
                recommended.append(movie_user_matrix.index[indices[0][i]])
        return recommended
    except:
        return []


def recommend_content_based(input_movies, movies, all_genres):
    input_movies = [title.lower().strip() for title in input_movies]
    title_map = {title.lower().strip(): idx for idx, title in enumerate(movies['title'])}
    indices = [title_map[m] for m in input_movies if m in title_map]

    if not indices:
        return ["None of the input movies were found."]

    # Cosine Similarity
    genre_features = movies[all_genres]
    input_features = genre_features.iloc[indices]
    similarities = cosine_similarity(input_features, genre_features)
    if len(indices) > 1:
        sim_scores = np.mean(similarities, axis=0)
    else:
        sim_scores = similarities[0]

    sorted_indices = sorted(list(enumerate(sim_scores)), key=lambda x: x[1], reverse=True)

    added = set(input_movies)
    content_recommendations = []

    for i in sorted_indices:
        title = movies.iloc[i[0]]['title']
        if title.lower().strip() not in added:
            content_recommendations.append(title)
            if len(content_recommendations) >= 5:
                break

    return content_recommendations


# Streamlit
st.title("🎬 Movie Recommender System")
st.markdown("Enter up to 5 movie titles below. We'll use both genre and rating similarity to recommend.")


movie1 = st.text_input("Movie 1", placeholder="e.g., Toy Story, Jumanji, Batman")
movie2 = st.text_input("Movie 2", placeholder="e.g., Forrest Gump, Titanic")
movie3 = st.text_input("Movie 3")
movie4 = st.text_input("Movie 4")
movie5 = st.text_input("Movie 5")

input_movies = [m.strip() for m in [movie1, movie2, movie3, movie4, movie5] if m.strip()]

if st.button("Recommend"):
    if input_movies:
        with st.spinner("Generating recommendations..."):
            content_based = recommend_content_based(input_movies, movies, all_genres)
            collab_based = []
            for m in input_movies:
                collab_recs = get_collab_recommendations(m, 3)
                collab_based.extend(collab_recs)

            combined = list(pd.unique(content_based + collab_based))

            if combined and combined != ["None of the input movies were found."]:
                st.subheader("Top 5 Recommended Movies:")
                for i, movie in enumerate(combined[:5], 1):
                    st.write(f"{i}. {movie}")
            else:
                st.warning("No recommendations found. Check your inputs or try different movies.")
                st.write("Available movies in our dataset:")
                for movie in movies['title'].head(10):
                    st.write(f"• {movie}")
    else:
        st.warning("Please enter at least one movie.")