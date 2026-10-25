"""
Recommendation engine module for the Movie Recommendation System.

Implements content-based filtering using TF-IDF vectorization of movie genres
and cosine similarity for finding similar movies.
"""

from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RecommenderError(Exception):
    """Raised when recommendation generation fails."""
    pass


class MovieRecommender:
    """
    Content-based movie recommender using genre similarity.
    
    Uses TF-IDF vectorization on movie genres and cosine similarity
    to find movies similar to a given movie.
    """
    
    def __init__(
        self,
        movies_df: pd.DataFrame,
        rating_stats_df: Optional[pd.DataFrame] = None,
        tfidf_max_features: int = 5000,
        tfidf_ngram_range: Tuple[int, int] = (1, 2)
    ):
        """
        Initialize the recommender.
        
        Args:
            movies_df: DataFrame with columns movieId, title, genres
            rating_stats_df: Optional DataFrame with movieId, avg_rating, rating_count
            tfidf_max_features: Maximum TF-IDF features
            tfidf_ngram_range: N-gram range for TF-IDF
        """
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.rating_stats_df = rating_stats_df
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._movie_id_to_idx: dict = {}
        self._idx_to_movie_id: dict = {}
        self._is_fitted = False
        
        self._build_mappings()
        self._fit()
    
    def _build_mappings(self) -> None:
        """Build mappings between movie IDs and matrix indices."""
        self._movie_id_to_idx = {
            row["movieId"]: idx for idx, row in self.movies_df.iterrows()
        }
        self._idx_to_movie_id = {
            idx: movie_id for movie_id, idx in self._movie_id_to_idx.items()
        }
    
    def _preprocess_genres(self, genres_series: pd.Series) -> List[str]:
        """
        Preprocess genre strings for TF-IDF.
        
        Converts 'Action|Adventure|Sci-Fi' to 'action adventure sci-fi'
        Handles missing/empty genres.
        """
        processed = []
        for genres in genres_series:
            if not genres or genres == "(no genres listed)" or pd.isna(genres):
                processed.append("")
            else:
                # Replace | with space, lowercase
                processed.append(genres.replace("|", " ").lower().strip())
        return processed
    
    def _fit(self) -> None:
        """Fit the TF-IDF vectorizer and compute the feature matrix."""
        genre_texts = self._preprocess_genres(self.movies_df["genres"])
        
        # Check if we have any meaningful genre data
        if all(not g for g in genre_texts):
            raise RecommenderError("No genre data available for any movie")
        
        self._tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=self.tfidf_ngram_range,
            stop_words="english",
            lowercase=True,
            token_pattern=r"[a-zA-Z0-9\-']+",  # Include hyphens for genres like "sci-fi"
            min_df=1,
            max_df=1.0
        )
        
        try:
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(genre_texts)
        except ValueError as e:
            raise RecommenderError(f"TF-IDF vectorization failed: {e}")
        
        self._is_fitted = True
    
    def get_similar_movies(
        self,
        movie_id: int,
        n_recommendations: int = 10,
        min_similarity: float = 0.0
    ) -> pd.DataFrame:
        """
        Get top-N similar movies for a given movie ID.
        
        Args:
            movie_id: The movie ID to find recommendations for
            n_recommendations: Number of recommendations to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            DataFrame with columns: movieId, title, genres, similarity_score,
            avg_rating (optional), rating_count (optional)
            
        Raises:
            RecommenderError: If movie_id not found or recommender not fitted
        """
        if not self._is_fitted:
            raise RecommenderError("Recommender not fitted")
        
        if movie_id not in self._movie_id_to_idx:
            raise RecommenderError(f"Movie ID {movie_id} not found in dataset")
        
        idx = self._movie_id_to_idx[movie_id]
        
        # Compute cosine similarity for this movie against all others
        movie_vector = self._tfidf_matrix[idx]
        similarities = cosine_similarity(movie_vector, self._tfidf_matrix).flatten()
        
        # Exclude the movie itself
        similarities[idx] = -1.0
        
        # Get top N indices by similarity
        # Use argsort for stable sorting (handles ties by index order)
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
        # Filter by minimum similarity
        top_indices = [i for i in top_indices if similarities[i] >= min_similarity]
        
        if not top_indices:
            return self._empty_recommendations_df()
        
        # Build recommendations DataFrame
        recommendations = []
        for i in top_indices:
            movie_id_rec = self._idx_to_movie_id[i]
            movie_row = self.movies_df.iloc[i]
            
            rec = {
                "movieId": movie_id_rec,
                "title": movie_row["title"],
                "genres": movie_row["genres"],
                "similarity_score": round(float(similarities[i]), 4)
            }
            
            # Add rating info if available
            if self.rating_stats_df is not None:
                rating_info = self.rating_stats_df[
                    self.rating_stats_df["movieId"] == movie_id_rec
                ]
                if not rating_info.empty:
                    rec["avg_rating"] = rating_info.iloc[0]["avg_rating"]
                    rec["rating_count"] = int(rating_info.iloc[0]["rating_count"])
                else:
                    rec["avg_rating"] = None
                    rec["rating_count"] = 0
            
            recommendations.append(rec)
        
        return pd.DataFrame(recommendations)
    
    def _empty_recommendations_df(self) -> pd.DataFrame:
        """Return an empty recommendations DataFrame with correct columns."""
        cols = ["movieId", "title", "genres", "similarity_score"]
        if self.rating_stats_df is not None:
            cols.extend(["avg_rating", "rating_count"])
        return pd.DataFrame(columns=cols)
    
    def get_movie_by_id(self, movie_id: int) -> Optional[pd.Series]:
        """Get movie details by ID."""
        if movie_id in self._movie_id_to_idx:
            idx = self._movie_id_to_idx[movie_id]
            return self.movies_df.iloc[idx]
        return None
    
    def get_movie_by_title(self, title: str) -> Optional[pd.Series]:
        """Get movie details by exact title match (case-insensitive)."""
        matches = self.movies_df[
            self.movies_df["title"].str.lower() == title.lower()
        ]
        if not matches.empty:
            return matches.iloc[0]
        return None
    
    @property
    def n_movies(self) -> int:
        """Return the number of movies in the recommender."""
        return len(self.movies_df)
    
    @property
    def feature_names(self) -> List[str]:
        """Return the TF-IDF feature names (genre tokens)."""
        if self._tfidf_vectorizer is not None:
            return self._tfidf_vectorizer.get_feature_names_out().tolist()
        return []


def create_recommender(
    movies_df: pd.DataFrame,
    rating_stats_df: Optional[pd.DataFrame] = None
) -> MovieRecommender:
    """
    Factory function to create a MovieRecommender instance.
    
    Args:
        movies_df: Cleaned movies DataFrame
        rating_stats_df: Optional aggregated rating statistics
        
    Returns:
        Fitted MovieRecommender instance
    """
    return MovieRecommender(movies_df, rating_stats_df)