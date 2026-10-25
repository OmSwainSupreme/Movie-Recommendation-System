"""
Tests for the data_loader module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from data_loader import (
    load_movies,
    load_ratings,
    load_dataset,
    clean_movies,
    clean_ratings,
    aggregate_ratings,
    validate_movie_columns,
    validate_rating_columns,
    get_genre_list,
    filter_movies_by_genre,
    search_movies,
    DataLoadError
)


class TestValidateMovieColumns:
    """Tests for movie column validation."""
    
    def test_valid_columns(self):
        df = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["Movie A", "Movie B"],
            "genres": ["Action|Adventure", "Comedy"]
        })
        validate_movie_columns(df)  # Should not raise
    
    def test_missing_movieId(self):
        df = pd.DataFrame({"title": ["A"], "genres": ["Action"]})
        with pytest.raises(DataLoadError, match="movieId"):
            validate_movie_columns(df)
    
    def test_missing_title(self):
        df = pd.DataFrame({"movieId": [1], "genres": ["Action"]})
        with pytest.raises(DataLoadError, match="title"):
            validate_movie_columns(df)
    
    def test_missing_genres(self):
        df = pd.DataFrame({"movieId": [1], "title": ["A"]})
        with pytest.raises(DataLoadError, match="genres"):
            validate_movie_columns(df)


class TestValidateRatingColumns:
    """Tests for rating column validation."""
    
    def test_valid_columns(self):
        df = pd.DataFrame({
            "userId": [1, 2],
            "movieId": [1, 2],
            "rating": [4.0, 5.0]
        })
        validate_rating_columns(df)  # Should not raise
    
    def test_missing_userId(self):
        df = pd.DataFrame({"movieId": [1], "rating": [4.0]})
        with pytest.raises(DataLoadError, match="userId"):
            validate_rating_columns(df)
    
    def test_missing_rating(self):
        df = pd.DataFrame({"userId": [1], "movieId": [1]})
        with pytest.raises(DataLoadError, match="rating"):
            validate_rating_columns(df)


class TestCleanMovies:
    """Tests for movie cleaning."""
    
    def test_removes_duplicates(self):
        df = pd.DataFrame({
            "movieId": [1, 1, 2],
            "title": ["A", "A dup", "B"],
            "genres": ["Action", "Action", "Comedy"]
        })
        cleaned = clean_movies(df)
        assert len(cleaned) == 2
        assert cleaned.iloc[0]["title"] == "A"  # Keeps first
    
    def test_handles_missing_genres(self):
        df = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["A", "B"],
            "genres": [None, "Comedy"]
        })
        cleaned = clean_movies(df)
        assert cleaned.iloc[0]["genres"] == ""
        assert cleaned.iloc[1]["genres"] == "Comedy"
    
    def test_handles_na_genres(self):
        df = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["A", "B"],
            "genres": [np.nan, "Comedy"]
        })
        cleaned = clean_movies(df)
        assert cleaned.iloc[0]["genres"] == ""
    
    def test_strips_whitespace(self):
        df = pd.DataFrame({
            "movieId": [1],
            "title": ["  Movie Title  "],
            "genres": ["Action"]
        })
        cleaned = clean_movies(df)
        assert cleaned.iloc[0]["title"] == "Movie Title"
    
    def test_converts_movieId_to_int(self):
        df = pd.DataFrame({
            "movieId": ["1", "2", "3"],
            "title": ["A", "B", "C"],
            "genres": ["Action", "Comedy", "Drama"]
        })
        cleaned = clean_movies(df)
        # pandas uses nullable Int64 dtype for integer columns with potential NaN
        assert pd.api.types.is_integer_dtype(cleaned["movieId"])
    
    def test_drops_invalid_movieId(self):
        df = pd.DataFrame({
            "movieId": [1, "invalid", 3],
            "title": ["A", "B", "C"],
            "genres": ["Action", "Comedy", "Drama"]
        })
        cleaned = clean_movies(df)
        assert len(cleaned) == 2
        assert all(cleaned["movieId"].isin([1, 3]))


class TestCleanRatings:
    """Tests for rating cleaning."""
    
    def test_removes_duplicates(self):
        df = pd.DataFrame({
            "userId": [1, 1, 2],
            "movieId": [1, 1, 2],
            "rating": [4.0, 4.0, 5.0]
        })
        cleaned = clean_ratings(df)
        assert len(cleaned) == 2
    
    def test_converts_types(self):
        df = pd.DataFrame({
            "userId": ["1", "2"],
            "movieId": ["1", "2"],
            "rating": ["4.0", "5.0"]
        })
        cleaned = clean_ratings(df)
        assert cleaned["userId"].dtype == int
        assert cleaned["movieId"].dtype == int
        assert cleaned["rating"].dtype == float
    
    def test_drops_missing_values(self):
        df = pd.DataFrame({
            "userId": [1, None, 3],
            "movieId": [1, 2, None],
            "rating": [4.0, 5.0, None]
        })
        cleaned = clean_ratings(df)
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["userId"] == 1


class TestAggregateRatings:
    """Tests for rating aggregation."""
    
    def test_basic_aggregation(self):
        df = pd.DataFrame({
            "userId": [1, 2, 3, 4],
            "movieId": [1, 1, 2, 2],
            "rating": [4.0, 5.0, 3.0, 3.0]
        })
        agg = aggregate_ratings(df)
        assert len(agg) == 2
        assert agg[agg["movieId"] == 1].iloc[0]["avg_rating"] == 4.5
        assert agg[agg["movieId"] == 1].iloc[0]["rating_count"] == 2
        assert agg[agg["movieId"] == 2].iloc[0]["avg_rating"] == 3.0
        assert agg[agg["movieId"] == 2].iloc[0]["rating_count"] == 2
    
    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["userId", "movieId", "rating"])
        agg = aggregate_ratings(df)
        assert agg.empty
        assert list(agg.columns) == ["movieId", "avg_rating", "rating_count"]
    
    def test_rounds_average_rating(self):
        df = pd.DataFrame({
            "userId": [1, 2],
            "movieId": [1, 1],
            "rating": [4.0, 4.333333]
        })
        agg = aggregate_ratings(df)
        assert agg.iloc[0]["avg_rating"] == 4.17


class TestGetGenreList:
    """Tests for genre list extraction."""
    
    def test_extracts_unique_genres(self):
        df = pd.DataFrame({
            "movieId": [1, 2, 3],
            "title": ["A", "B", "C"],
            "genres": ["Action|Adventure", "Adventure|Comedy", "Drama"]
        })
        genres = get_genre_list(df)
        assert genres == ["Action", "Adventure", "Comedy", "Drama"]
    
    def test_handles_no_genres_listed(self):
        df = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["A", "B"],
            "genres": ["(no genres listed)", "Action"]
        })
        genres = get_genre_list(df)
        assert genres == ["Action"]
    
    def test_handles_empty_genres(self):
        df = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["A", "B"],
            "genres": ["", "Action"]
        })
        genres = get_genre_list(df)
        assert genres == ["Action"]


class TestFilterMoviesByGenre:
    """Tests for genre filtering."""
    
    def test_filter_all(self):
        df = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["A", "B"],
            "genres": ["Action|Adventure", "Comedy"]
        })
        filtered = filter_movies_by_genre(df, "All")
        assert len(filtered) == 2
    
    def test_filter_specific_genre(self):
        df = pd.DataFrame({
            "movieId": [1, 2, 3],
            "title": ["A", "B", "C"],
            "genres": ["Action|Adventure", "Comedy", "Action"]
        })
        filtered = filter_movies_by_genre(df, "Action")
        assert len(filtered) == 2
        assert all("Action" in g for g in filtered["genres"])
    
    def test_filter_nonexistent_genre(self):
        df = pd.DataFrame({
            "movieId": [1],
            "title": ["A"],
            "genres": ["Action"]
        })
        filtered = filter_movies_by_genre(df, "Horror")
        assert len(filtered) == 0


class TestSearchMovies:
    """Tests for movie search."""
    
    def test_case_insensitive_search(self):
        df = pd.DataFrame({
            "movieId": [1, 2, 3],
            "title": ["Toy Story", "Toy Story 2", "Jumanji"],
            "genres": ["Animation", "Animation", "Adventure"]
        })
        results = search_movies(df, "toy")
        assert len(results) == 2
    
    def test_partial_match(self):
        df = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["The Matrix", "Matrix Reloaded"],
            "genres": ["Action", "Action"]
        })
        results = search_movies(df, "Matrix")
        assert len(results) == 2
    
    def test_empty_query_returns_all(self):
        df = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["A", "B"],
            "genres": ["Action", "Comedy"]
        })
        results = search_movies(df, "")
        assert len(results) == 2
    
    def test_no_matches(self):
        df = pd.DataFrame({
            "movieId": [1],
            "title": ["Toy Story"],
            "genres": ["Animation"]
        })
        results = search_movies(df, "Nonexistent")
        assert len(results) == 0


class TestLoadDataset:
    """Integration tests for dataset loading with temp files."""
    
    def test_load_valid_movies_csv(self, tmp_path):
        csv_content = """movieId,title,genres
1,Toy Story (1995),Action|Adventure|Animation
2,Jumanji (1995),Adventure|Children|Fantasy
"""
        movies_path = tmp_path / "movies.csv"
        movies_path.write_text(csv_content)
        
        movies_df = load_movies(tmp_path)
        assert len(movies_df) == 2
        assert movies_df.iloc[0]["title"] == "Toy Story (1995)"
        assert movies_df.iloc[0]["genres"] == "Action|Adventure|Animation"
    
    def test_load_missing_movies_csv(self, tmp_path):
        with pytest.raises(DataLoadError, match="not found"):
            load_movies(tmp_path)
    
    def test_load_movies_missing_columns(self, tmp_path):
        csv_content = """movieId,title
1,Toy Story
"""
        movies_path = tmp_path / "movies.csv"
        movies_path.write_text(csv_content)
        
        with pytest.raises(DataLoadError, match="genres"):
            load_movies(tmp_path)
    
    def test_load_ratings_csv(self, tmp_path):
        csv_content = """userId,movieId,rating,timestamp
1,1,4.0,123456
2,1,5.0,123457
"""
        movies_path = tmp_path / "movies.csv"
        movies_path.write_text("movieId,title,genres\n1,Toy Story,Action\n")
        ratings_path = tmp_path / "ratings.csv"
        ratings_path.write_text(csv_content)
        
        ratings_df = load_ratings(tmp_path)
        assert len(ratings_df) == 2
        assert ratings_df.iloc[0]["userId"] == 1
    
    def test_load_missing_ratings_csv_returns_none(self, tmp_path):
        movies_path = tmp_path / "movies.csv"
        movies_path.write_text("movieId,title,genres\n1,Toy Story,Action\n")
        
        ratings_df = load_ratings(tmp_path)
        assert ratings_df is None
    
    def test_load_full_dataset(self, tmp_path):
        movies_csv = """movieId,title,genres
1,Toy Story,Action|Adventure
2,Jumanji,Adventure|Fantasy
"""
        ratings_csv = """userId,movieId,rating,timestamp
1,1,4.0,1
2,1,5.0,2
3,2,3.0,3
"""
        (tmp_path / "movies.csv").write_text(movies_csv)
        (tmp_path / "ratings.csv").write_text(ratings_csv)
        
        movies_df, ratings_df, rating_stats_df = load_dataset(tmp_path)
        
        assert len(movies_df) == 2
        assert len(ratings_df) == 3
        assert rating_stats_df is not None
        assert len(rating_stats_df) == 2
        assert rating_stats_df[rating_stats_df["movieId"] == 1].iloc[0]["avg_rating"] == 4.5


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_movies_csv(self, tmp_path):
        (tmp_path / "movies.csv").write_text("movieId,title,genres\n")
        with pytest.raises(DataLoadError, match="no valid data"):
            load_movies(tmp_path)
    
    def test_movies_with_only_no_genres(self, tmp_path):
        csv_content = """movieId,title,genres
1,Movie A,(no genres listed)
2,Movie B,(no genres listed)
"""
        (tmp_path / "movies.csv").write_text(csv_content)
        movies_df = load_movies(tmp_path)
        assert len(movies_df) == 2
        # Should still load, but genres will be empty strings after cleaning
        assert movies_df.iloc[0]["genres"] == "(no genres listed)"
    
    def test_unicode_titles(self, tmp_path):
        csv_content = """movieId,title,genres
1,Amélie,Comedy|Romance
2,北京故事,Drama
"""
        (tmp_path / "movies.csv").write_text(csv_content, encoding="utf-8")
        movies_df = load_movies(tmp_path)
        assert len(movies_df) == 2
        assert movies_df.iloc[0]["title"] == "Amélie"
        assert movies_df.iloc[1]["title"] == "北京故事"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])