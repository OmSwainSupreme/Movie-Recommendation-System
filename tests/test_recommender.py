"""
Tests for the recommender module.
"""

import pytest
import pandas as pd
import numpy as np
from recommender import MovieRecommender, create_recommender, RecommenderError


# Module-level fixtures
@pytest.fixture
def sample_movies():
    """Create a sample movies DataFrame for testing."""
    return pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5],
        "title": [
            "Action Movie",
            "Action Comedy",
            "Comedy Drama",
            "Pure Drama",
            "Sci-Fi Action"
        ],
        "genres": [
            "Action|Adventure",
            "Action|Comedy",
            "Comedy|Drama",
            "Drama",
            "Sci-Fi|Action"
        ]
    })


@pytest.fixture
def sample_rating_stats():
    """Create sample rating statistics."""
    return pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5],
        "avg_rating": [4.0, 3.5, 4.2, 3.8, 4.5],
        "rating_count": [100, 80, 120, 90, 150]
    })


@pytest.fixture
def recommender(sample_movies):
    """Create a recommender instance."""
    return create_recommender(sample_movies)


@pytest.fixture
def recommender_with_ratings(sample_movies, sample_rating_stats):
    """Create a recommender with rating stats."""
    return create_recommender(sample_movies, sample_rating_stats)


class TestMovieRecommender:
    """Tests for the MovieRecommender class."""
    
    def test_initialization(self, recommender, sample_movies):
        """Test recommender initializes correctly."""
        assert recommender.n_movies == 5
        assert recommender._is_fitted
        assert len(recommender._movie_id_to_idx) == 5
    
    def test_initialization_with_ratings(self, recommender_with_ratings, sample_rating_stats):
        """Test recommender with rating stats."""
        assert recommender_with_ratings.rating_stats_df is not None
        assert len(recommender_with_ratings.rating_stats_df) == 5
    
    def test_get_movie_by_id(self, recommender):
        """Test retrieving movie by ID."""
        movie = recommender.get_movie_by_id(1)
        assert movie is not None
        assert movie["title"] == "Action Movie"
        assert movie["movieId"] == 1
    
    def test_get_movie_by_id_not_found(self, recommender):
        """Test retrieving non-existent movie by ID."""
        movie = recommender.get_movie_by_id(999)
        assert movie is None
    
    def test_get_movie_by_title(self, recommender):
        """Test retrieving movie by title."""
        movie = recommender.get_movie_by_title("Action Movie")
        assert movie is not None
        assert movie["movieId"] == 1
    
    def test_get_movie_by_title_case_insensitive(self, recommender):
        """Test case-insensitive title search."""
        movie = recommender.get_movie_by_title("action movie")
        assert movie is not None
        assert movie["movieId"] == 1
    
    def test_get_movie_by_title_not_found(self, recommender):
        """Test retrieving non-existent movie by title."""
        movie = recommender.get_movie_by_title("Nonexistent")
        assert movie is None


class TestRecommendations:
    """Tests for recommendation generation."""
    
    def test_recommendations_for_valid_movie(self, recommender):
        """Test getting recommendations for a valid movie."""
        recs = recommender.get_similar_movies(1, n_recommendations=3)
        
        assert len(recs) == 3
        assert "movieId" in recs.columns
        assert "title" in recs.columns
        assert "genres" in recs.columns
        assert "similarity_score" in recs.columns
        assert all(recs["similarity_score"] >= 0)
        assert all(recs["similarity_score"] <= 1)
    
    def test_excludes_selected_movie(self, recommender):
        """Test that selected movie is not in its own recommendations."""
        recs = recommender.get_similar_movies(1, n_recommendations=10)
        assert 1 not in recs["movieId"].values
    
    def test_recommendation_count(self, recommender):
        """Test that correct number of recommendations returned."""
        for n in [1, 2, 3, 4, 5, 10]:
            recs = recommender.get_similar_movies(1, n_recommendations=n)
            assert len(recs) <= n  # May be fewer if not enough movies
            assert len(recs) == min(n, 4)  # 4 other movies in dataset
    
    def test_similarity_scores_sorted_descending(self, recommender):
        """Test that recommendations are sorted by similarity descending."""
        recs = recommender.get_similar_movies(1, n_recommendations=4)
        scores = recs["similarity_score"].values
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    
    def test_min_similarity_threshold(self, recommender):
        """Test minimum similarity threshold filtering."""
        # With high threshold, may get fewer results
        recs = recommender.get_similar_movies(1, n_recommendations=4, min_similarity=0.5)
        assert all(recs["similarity_score"] >= 0.5)
    
    def test_unknown_movie_id_raises_error(self, recommender):
        """Test that unknown movie ID raises RecommenderError."""
        with pytest.raises(RecommenderError, match="not found"):
            recommender.get_similar_movies(999)
    
    def test_recommendations_deterministic(self, recommender):
        """Test that repeated calls return identical results."""
        recs1 = recommender.get_similar_movies(1, n_recommendations=3)
        recs2 = recommender.get_similar_movies(1, n_recommendations=3)
        
        pd.testing.assert_frame_equal(recs1, recs2)
    
    def test_recommendations_with_rating_stats(self, sample_movies, sample_rating_stats):
        """Test recommendations include rating info when available."""
        recommender = create_recommender(sample_movies, sample_rating_stats)
        recs = recommender.get_similar_movies(1, n_recommendations=3)
        
        assert "avg_rating" in recs.columns
        assert "rating_count" in recs.columns
        assert not recs["avg_rating"].isna().all()


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_identical_genres(self):
        """Test movies with identical genres."""
        movies = pd.DataFrame({
            "movieId": [1, 2, 3],
            "title": ["A", "B", "C"],
            "genres": ["Action|Adventure", "Action|Adventure", "Comedy"]
        })
        recommender = create_recommender(movies)
        recs = recommender.get_similar_movies(1, n_recommendations=2)
        
        # Movie 2 has identical genres to movie 1, should have similarity 1.0
        assert len(recs) >= 1
        assert recs.iloc[0]["movieId"] == 2
        assert recs.iloc[0]["similarity_score"] == 1.0
    
    def test_no_meaningful_similarity(self):
        """Test movies with completely different genres."""
        movies = pd.DataFrame({
            "movieId": [1, 2, 3],
            "title": ["Action", "Comedy", "Drama"],
            "genres": ["Action", "Comedy", "Drama"]
        })
        recommender = create_recommender(movies)
        recs = recommender.get_similar_movies(1, n_recommendations=2)
        
        # All similarities should be 0 (or very close)
        assert all(recs["similarity_score"] == 0.0)
    
    def test_missing_genres(self):
        """Test handling of movies with missing/empty genres."""
        movies = pd.DataFrame({
            "movieId": [1, 2, 3],
            "title": ["Action", "No Genres", "Comedy"],
            "genres": ["Action", "", "Comedy"]
        })
        recommender = create_recommender(movies)
        recs = recommender.get_similar_movies(1, n_recommendations=2)
        
        # Should still work, movie with empty genres gets 0 similarity
        assert len(recs) == 2
        # The comedy movie should have 0 similarity
        comedy_rec = recs[recs["movieId"] == 3]
        assert comedy_rec.iloc[0]["similarity_score"] == 0.0
    
    def test_all_movies_no_genres_raises_error(self):
        """Test that all movies with no genres raises error."""
        movies = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["A", "B"],
            "genres": ["", ""]
        })
        with pytest.raises(RecommenderError, match="No genre data"):
            create_recommender(movies)
    
    def test_single_movie_dataset(self):
        """Test with only one movie in dataset."""
        movies = pd.DataFrame({
            "movieId": [1],
            "title": ["Only Movie"],
            "genres": ["Action"]
        })
        recommender = create_recommender(movies)
        recs = recommender.get_similar_movies(1, n_recommendations=5)
        
        assert recs.empty
    
    def test_minimal_dataset_two_movies(self):
        """Test with two movies."""
        movies = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["Action", "Comedy"],
            "genres": ["Action", "Comedy"]
        })
        recommender = create_recommender(movies)
        recs = recommender.get_similar_movies(1, n_recommendations=5)
        
        assert len(recs) == 1
        assert recs.iloc[0]["movieId"] == 2
        assert recs.iloc[0]["similarity_score"] == 0.0
    
    def test_feature_names_available(self, recommender):
        """Test that TF-IDF feature names are accessible."""
        features = recommender.feature_names
        assert isinstance(features, list)
        assert len(features) > 0
        # Should contain genre tokens
        assert any("action" in f.lower() for f in features)


class TestRecommenderFactory:
    """Tests for the create_recommender factory function."""
    
    def test_factory_returns_recommender(self):
        movies = pd.DataFrame({
            "movieId": [1],
            "title": ["Test"],
            "genres": ["Action"]
        })
        recommender = create_recommender(movies)
        assert isinstance(recommender, MovieRecommender)
        assert recommender._is_fitted
    
    def test_factory_with_none_ratings(self):
        movies = pd.DataFrame({
            "movieId": [1],
            "title": ["Test"],
            "genres": ["Action"]
        })
        recommender = create_recommender(movies, rating_stats_df=None)
        assert recommender.rating_stats_df is None
    
    def test_factory_with_empty_ratings(self):
        movies = pd.DataFrame({
            "movieId": [1],
            "title": ["Test"],
            "genres": ["Action"]
        })
        empty_ratings = pd.DataFrame(columns=["movieId", "avg_rating", "rating_count"])
        recommender = create_recommender(movies, rating_stats_df=empty_ratings)
        assert recommender.rating_stats_df is not None
        assert recommender.rating_stats_df.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])