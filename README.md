# Movie Recommendation System

A content-based movie recommendation system built with Python, Streamlit, and scikit-learn. This project demonstrates a real machine learning algorithm (TF-IDF + Cosine Similarity) for recommending movies based on genre similarity.

## 🎯 Project Overview

This system recommends movies similar to a user-selected movie by analyzing genre overlap. It uses **content-based filtering** — specifically, TF-IDF vectorization of movie genres followed by cosine similarity computation — to find movies with similar genre profiles.

**Perfect for an AI/ML internship project:** Clean code, working algorithm, automated tests, and a polished UI.

---

## ✨ Key Features

- **Content-Based Recommendations**: Uses TF-IDF on movie genres + cosine similarity
- **Interactive Streamlit UI**: Search, filter, and explore movies
- **Genre-Based Filtering**: Browse movies by genre
- **Rating Integration**: Shows average ratings and rating counts (when available)
- **Deterministic Results**: Same input → same recommendations every time
- **Comprehensive Tests**: 20+ pytest tests covering edge cases
- **Offline Operation**: Works without internet after initial dataset download

---

## 🧠 How It Works: Content-Based Filtering

### The Algorithm

1. **Genre Representation**: Each movie's genres (e.g., `"Action|Adventure|Sci-Fi"`) are treated as a text document
2. **TF-IDF Vectorization**: Converts genre text into numerical feature vectors
   - **TF (Term Frequency)**: How often a genre appears in a movie's genre list
   - **IDF (Inverse Document Frequency)**: Downweights common genres (like "Drama") that appear in many movies
3. **Cosine Similarity**: Measures the angle between two movie vectors
   - Score of 1.0 = identical genre profiles
   - Score of 0.0 = no genre overlap
4. **Ranking**: Returns top-N most similar movies (excluding the selected movie itself)

### Why TF-IDF + Cosine Similarity?

- **Interpretable**: You can see exactly which genres drive similarity
- **No Training Required**: Works immediately on any movie dataset with genres
- **Cold-Start Friendly**: New movies just need genres — no user history needed
- **Fast**: Computation is O(n) per query, not O(n²) for full matrix

### Limitations

- ❌ **No Personalization**: Same recommendations for every user
- ❌ **Genre-Only**: Ignores plot, actors, directors, mood, themes
- ❌ **Cold-Start for Users**: Cannot recommend to new users without a seed movie
- ❌ **Sparse Genres**: Movies with few/unique genres get poor recommendations

---

## 🛠 Technology Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| UI | Streamlit | Interactive web interface |
| Data | Pandas, NumPy | Data loading, cleaning, manipulation |
| ML | scikit-learn | TF-IDF vectorization, cosine similarity |
| Testing | pytest | Automated test suite |
| Download | urllib | Dataset fetching |

**Python Version**: 3.11+

---

## 📁 Project Structure

```
movie-recommendation-system/
├── app.py                 # Streamlit application
├── recommender.py         # Recommendation engine (TF-IDF + cosine similarity)
├── data_loader.py         # Dataset loading, validation, rating aggregation
├── download_data.py       # One-time dataset download script
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── data/
│   └── .gitkeep           # Placeholder (dataset not committed)
└── tests/
    ├── test_data_loader.py    # Data loading tests
    └── test_recommender.py    # Recommendation engine tests
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or newer
- Git (optional)

### Windows PowerShell

```powershell
# 1. Clone or navigate to project directory
cd C:\path\to\movie-recommendation-system

# 2. Create virtual environment
python -m venv venv
venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset (one-time)
python download_data.py

# 5. Run the application
streamlit run app.py
```

### Linux / macOS / Git Bash

```bash
# 1. Clone or navigate to project directory
cd /path/to/movie-recommendation-system

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset (one-time)
python download_data.py

# 5. Run the application
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📥 Dataset

**Source**: [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/latest/) by GroupLens Research

**Files Used**:
- `movies.csv` — Movie IDs, titles, genres (required)
- `ratings.csv` — User ratings (optional, for rating display)

**License**: MovieLens datasets are provided for **non-commercial research and educational use only**. See [GroupLens Terms](https://grouplens.org/datasets/movielens/) for details.

**Attribution**: 
> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI: http://dx.doi.org/10.1145/2827872

The `download_data.py` script fetches the official dataset from GroupLens. The dataset is **not** included in this repository.

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_recommender.py -v
pytest tests/test_data_loader.py -v

# Run with coverage
pytest --cov=. --cov-report=term-missing
```

**Test Coverage Includes**:
- ✅ Dataset loading and validation
- ✅ Missing column detection
- ✅ Missing/empty genre handling
- ✅ Recommendation generation
- ✅ Self-exclusion from recommendations
- ✅ Correct recommendation count
- ✅ Deterministic/stable output
- ✅ Unknown movie ID handling
- ✅ Identical genre handling
- ✅ No-similarity edge case
- ✅ Rating aggregation
- ✅ Empty/minimal dataset handling

---

## 🎮 Demo Workflow

1. **Launch**: `streamlit run app.py`
2. **Discover Tab**: 
   - Search for "Toy Story" or filter by "Animation"
   - Click a row to select a movie
   - Click **"Get Recommendations"**
3. **Recommendations Tab**:
   - View top 10 similar movies with similarity scores
   - See average ratings and rating counts
   - Click **"Explore"** on any recommendation to recursively find similar movies
4. **Popular Tab** (if ratings.csv downloaded):
   - View highest-rated movies (Bayesian-weighted)
   - View most-rated movies
   - Adjust minimum rating count threshold

---

## ⚙️ Configuration

### Recommender Parameters (in `recommender.py`)

```python
MovieRecommender(
    tfidf_max_features=5000,      # Max vocabulary size
    tfidf_ngram_range=(1, 2)      # Use unigrams + bigrams
)
```

### UI Settings (in `app.py`)

- Default recommendations: 10
- Options: 5, 10, 15, 20
- Similarity threshold: 0.0 – 1.0 (slider)

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `DataLoadError: movies.csv not found` | Run `python download_data.py` |
| `RecommenderError: No genre data` | Dataset may be corrupted; re-download |
| Streamlit won't start | Check port 8501 is free; try `streamlit run app.py --server.port 8502` |
| Tests fail | Ensure you're in project root with venv activated |

---

## 📝 Future Improvements

- [ ] Hybrid filtering (content + collaborative)
- [ ] User profiles & personalized recommendations
- [ ] Movie posters via TMDB API (optional)
- [ ] Export recommendations to CSV
- [ ] Docker containerization
- [ ] Web deployment (Streamlit Cloud, Hugging Face Spaces)

---

## 📄 License

This project is for educational purposes. The MovieLens dataset has its own license (non-commercial research/education only). Code in this repository is MIT-licensed.

---

## 🙏 Acknowledgments

- **GroupLens Research** for the MovieLens dataset
- **scikit-learn** for TF-IDF and cosine similarity implementations
- **Streamlit** for the effortless UI framework