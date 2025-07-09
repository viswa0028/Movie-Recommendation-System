# 🎬 Movie Recommendation System

A hybrid movie recommendation system built with Python and Streamlit that combines content-based filtering (using movie genres) and collaborative filtering (using user ratings) to suggest movies based on user preferences.

## 🎯 Features

- **Content-Based Filtering**: Recommends movies based on genre similarity
- **Collaborative Filtering**: Uses user ratings and K-Nearest Neighbors to find similar movies
- **Hybrid Approach**: Combines both methods for better recommendations
- **Interactive Web UI**: User-friendly Streamlit interface
- **Real-time Recommendations**: Get top 5 movie suggestions instantly

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/movie-recommender-system.git
cd movie-recommender-system
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the MovieLens dataset:
   - Download from [MovieLens](https://grouplens.org/datasets/movielens/)
   - Place `movies.csv` and `ratings.csv` in the project directory

## 📋 Requirements

Create a `requirements.txt` file with:
```
streamlit
pandas
numpy
scikit-learn
scipy
```

## 🎮 Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Enter up to 5 movie titles in the input fields

3. Click "Recommend" to get personalized movie suggestions

## 🔧 How It Works

### 1. Data Preprocessing
- Removes year suffixes from movie titles (e.g., "Toy Story (1995)" → "Toy Story")
- Processes movie genres into binary features
- Creates user-item matrix for collaborative filtering

### 2. Content-Based Filtering
- Uses genre similarity with cosine similarity
- Finds movies with similar genre combinations
- Recommends based on genre preferences

### 3. Collaborative Filtering
- Uses K-Nearest Neighbors algorithm
- Finds users with similar rating patterns
- Recommends movies liked by similar users

### 4. Hybrid Approach
- Combines both methods
- Provides more diverse and accurate recommendations

## 📊 Dataset

This project uses the MovieLens dataset which includes:
- `movies.csv`: Movie information with genres
- `ratings.csv`: User ratings for movies

**Note**: The system is optimized to work with the first 1000 movies for better performance.

## 🎨 Features

- **Smart Matching**: Handles case-insensitive movie title matching
- **Sample Movies**: Shows available movies in the dataset
- **Real-time Processing**: Fast recommendation generation
- **Responsive UI**: Clean and intuitive interface

## 🔍 Example Usage

```python
# Example movie inputs:
"Toy Story"
"Jumanji" 
"Batman"
"Forrest Gump"
"Titanic"
```

## 🧮 Algorithm Details

### Content-Based Filtering
- Uses one-hot encoding for movie genres
- Calculates cosine similarity between genre vectors
- Recommends movies with highest similarity scores

### Collaborative Filtering
- Creates sparse user-item matrix
- Uses KNN with cosine metric
- Finds similar movies based on user rating patterns

## 🎯 Project Structure

```
movie-recommender-system/
├── app.py                 # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation

```

## 🚀 Future Enhancements

- [ ] Add more sophisticated hybrid approach with weighted combination
- [ ] Include movie posters and additional metadata
- [ ] Add user authentication and personalized profiles
- [ ] Implement matrix factorization techniques (SVD, NMF)
- [ ] Add movie reviews and sentiment analysis
- [ ] Include more filtering options (year, rating, etc.)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) for providing the dataset
- [Streamlit](https://streamlit.io/) for the web framework
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms

