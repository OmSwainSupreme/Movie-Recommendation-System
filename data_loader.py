"""
Data loading and validation module for the Movie Recommendation System.

Handles loading movies.csv and ratings.csv from the MovieLens dataset,
validates required columns, cleans data, and aggregates rating statistics.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np


REQUIRED_MOVIE_COLUMNS = ["movieId", "title", "genres"]
REQUIRED_RATING_COLUMNS = ["userId", "movieId", "rating"]


class DataLoadError(Exception):
    """Raised when dataset loading or validation fails."""
    pass


def validate_movie_columns(df: pd.DataFrame) -> None:
    """Validate that the movies DataFrame has all required columns."""
    missing = [col for col in REQUIRED_MOVIE_COLUMNS if col not in df.columns]
    if missing:
        raise DataLoadError(f"movies.csv missing required columns: {missing}")


def validate_rating_columns(df: pd.DataFrame) -> None:
    """Validate that the ratings DataFrame has all required columns."""
    missing = [col for col in REQUIRED_RATING_COLUMNS if col not in df.columns]
    if missing:
        raise DataLoadError(f"ratings.csv missing required columns: {missing}")


def clean_movies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the movies DataFrame.
    
    - Removes duplicates by movieId (keeps first)
    - Handles missing genres (replaces with empty string)
    - Strips whitespace from titles
    - Ensures movieId is integer
    """
    df = df.copy()
    
    # Remove duplicates by movieId
    df = df.drop_duplicates(subset=["movieId"], keep="first")
    
    # Handle missing genres
    df["genres"] = df["genres"].fillna("").astype(str)
    
    # Strip whitespace from titles
    df["title"] = df["title"].astype(str).str.strip()
    
    # Ensure movieId is integer
    df["movieId"] = pd.to_numeric(df["movieId"], errors="coerce").astype("Int64")
    
    # Drop rows with invalid movieId
    invalid_count = df["movieId"].isna().sum()
    if invalid_count > 0:
        df = df.dropna(subset=["movieId"])
        df["movieId"] = df["movieId"].astype(int)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def clean_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the ratings DataFrame.
    
    - Removes duplicates
    - Ensures numeric types
    - Drops rows with missing values in required columns
    """
    df = df.copy()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Ensure numeric types
    for col in ["userId", "movieId", "rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop rows with missing required values
    df = df.dropna(subset=REQUIRED_RATING_COLUMNS)
    
    # Convert to appropriate types
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = df["rating"].astype(float)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def aggregate_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate rating statistics per movie.
    
    Returns a DataFrame with columns: movieId, avg_rating, rating_count
    """
    if ratings_df.empty:
        return pd.DataFrame(columns=["movieId", "avg_rating", "rating_count"])
    
    agg = ratings_df.groupby("movieId")["rating"].agg(
        avg_rating="mean",
        rating_count="count"
    ).reset_index()
    
    agg["avg_rating"] = agg["avg_rating"].round(2)
    
    return agg


def load_movies(data_dir: Path) -> pd.DataFrame:
    """
    Load and clean movies.csv from the data directory.
    
    Args:
        data_dir: Path to the directory containing movies.csv
        
    Returns:
        Cleaned movies DataFrame
        
    Raises:
        DataLoadError: If file is missing or validation fails
    """
    movies_path = data_dir / "movies.csv"
    
    if not movies_path.exists():
        raise DataLoadError(f"movies.csv not found at {movies_path}")
    
    try:
        df = pd.read_csv(movies_path, encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback for potential encoding issues
        df = pd.read_csv(movies_path, encoding="latin-1")
    except Exception as e:
        raise DataLoadError(f"Failed to read movies.csv: {e}")
    
    validate_movie_columns(df)
    df = clean_movies(df)
    
    if df.empty:
        raise DataLoadError("movies.csv contains no valid data after cleaning")
    
    return df


def load_ratings(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load and clean ratings.csv from the data directory.
    
    Args:
        data_dir: Path to the directory containing ratings.csv
        
    Returns:
        Cleaned ratings DataFrame, or None if file doesn't exist
        
    Raises:
        DataLoadError: If file exists but validation fails
    """
    ratings_path = data_dir / "ratings.csv"
    
    if not ratings_path.exists():
        return None
    
    try:
        df = pd.read_csv(ratings_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(ratings_path, encoding="latin-1")
    except Exception as e:
        raise DataLoadError(f"Failed to read ratings.csv: {e}")
    
    validate_rating_columns(df)
    df = clean_ratings(df)
    
    return df


def load_dataset(data_dir: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load the complete dataset: movies, ratings, and aggregated rating stats.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        Tuple of (movies_df, ratings_df, rating_stats_df)
        ratings_df and rating_stats_df are None if ratings.csv is not available
    """
    movies_df = load_movies(data_dir)
    ratings_df = load_ratings(data_dir)
    
    rating_stats_df = None
    if ratings_df is not None and not ratings_df.empty:
        rating_stats_df = aggregate_ratings(ratings_df)
    
    return movies_df, ratings_df, rating_stats_df


def get_genre_list(movies_df: pd.DataFrame) -> list:
    """Extract sorted list of unique genres from the movies DataFrame."""
    all_genres = set()
    for genres_str in movies_df["genres"]:
        if genres_str and genres_str != "(no genres listed)":
            for genre in genres_str.split("|"):
                all_genres.add(genre.strip())
    return sorted(all_genres)


def filter_movies_by_genre(movies_df: pd.DataFrame, genre: str) -> pd.DataFrame:
    """Filter movies by a specific genre."""
    if not genre or genre == "All":
        return movies_df
    return movies_df[movies_df["genres"].str.contains(genre, na=False, regex=False)]


def search_movies(movies_df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Search movies by title (case-insensitive partial match)."""
    if not query:
        return movies_df
    return movies_df[movies_df["title"].str.contains(query, case=False, na=False)]