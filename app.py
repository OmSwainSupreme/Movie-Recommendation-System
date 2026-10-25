#!/usr/bin/env python
"""
Streamlit application for the Movie Recommendation System.

Provides a clean, responsive interface for discovering movies and getting
genre-based content recommendations.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (
    load_dataset,
    get_genre_list,
    filter_movies_by_genre,
    search_movies,
    DataLoadError
)
from recommender import create_recommender, RecommenderError


# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for dark cinema theme
CUSTOM_CSS = """
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1a1d24;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ff6b35 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #ff6b35;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #e85a2a;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #1a1d24;
        border: 1px solid #333;
    }
    
    /* DataFrame */
    .stDataFrame {
        background-color: #1a1d24;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1d24 0%, #23272f 100%);
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Recommendation cards */
    .rec-card {
        background: #1a1d24;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: border-color 0.2s;
    }
    
    .rec-card:hover {
        border-color: #ff6b35;
    }
    
    .rec-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #fafafa;
        margin-bottom: 0.25rem;
    }
    
    .rec-genres {
        font-size: 0.85rem;
        color: #aaa;
        margin-bottom: 0.5rem;
    }
    
    .rec-score {
        font-size: 0.9rem;
        color: #ff6b35;
        font-weight: 500;
    }
    
    .rec-rating {
        font-size: 0.85rem;
        color: #4ec9b0;
    }
    
    /* Info boxes */
    .info-box {
        background: #1a1d24;
        border-left: 4px solid #ff6b35;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .warning-box {
        background: #2d1f1f;
        border-left: 4px solid #ff6b35;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Selected movie highlight */
    .selected-movie {
        background: linear-gradient(135deg, #2d1f1f 0%, #3d2a2a 100%);
        border: 2px solid #ff6b35;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""


@st.cache_data(show_spinner="Loading movie dataset...")
def load_cached_dataset(data_dir: str) -> tuple:
    """Load and cache the dataset."""
    return load_dataset(Path(data_dir))


@st.cache_resource(show_spinner="Building recommendation model...")
def create_cached_recommender(movies_df: pd.DataFrame, rating_stats_df: pd.DataFrame):
    """Create and cache the recommender instance."""
    return create_recommender(movies_df, rating_stats_df)


def display_movie_card(movie_row: pd.Series, show_similarity: bool = False, 
                       similarity_score: float = None, key_prefix: str = "") -> None:
    """Display a movie as a styled card."""
    genres = movie_row.get("genres", "")
    if genres == "(no genres listed)" or not genres:
        genres_display = "No genres listed"
    else:
        genres_display = genres.replace("|", ", ")
    
    title = movie_row.get("title", "Unknown Title")
    movie_id = movie_row.get("movieId", "")
    
    card_html = f"""
    <div class="rec-card" id="{key_prefix}-{movie_id}">
        <div class="rec-title">{title}</div>
        <div class="rec-genres">{genres_display}</div>
    """
    
    if show_similarity and similarity_score is not None:
        card_html += f'<div class="rec-score">Similarity: {similarity_score:.2%}</div>'
    
    # Add rating info if available
    if "avg_rating" in movie_row and pd.notna(movie_row.get("avg_rating")):
        avg_rating = movie_row["avg_rating"]
        rating_count = movie_row.get("rating_count", 0)
        card_html += f'<div class="rec-rating">★ {avg_rating:.1f} ({rating_count:,} ratings)</div>'
    
    card_html += "</div>"
    st.markdown(card_html, unsafe_allow_html=True)


def main():
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown("# 🎬 Movie Recommendation System")
    st.markdown(
        "<div class='info-box'>"
        "<strong>Content-Based Filtering:</strong> Recommendations are based on movie genres "
        "using TF-IDF vectorization and cosine similarity. "
        "This system does <em>not</em> learn your personal preferences—it finds movies "
        "with similar genre profiles to the one you select."
        "</div>",
        unsafe_allow_html=True
    )
    
    # Load dataset
    DATA_DIR = "data"
    
    try:
        movies_df, ratings_df, rating_stats_df = load_cached_dataset(DATA_DIR)
    except DataLoadError as e:
        st.error(f"""
        **Failed to load dataset:** {e}
        
        Please run the download script first:
        ```bash
        python download_data.py
        ```
        """)
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading dataset: {e}")
        st.stop()
    
    # Create recommender
    try:
        recommender = create_cached_recommender(movies_df, rating_stats_df)
    except RecommenderError as e:
        st.error(f"Failed to initialize recommender: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error creating recommender: {e}")
        st.stop()
    
    # Sidebar - Dataset stats and controls
    with st.sidebar:
        st.markdown("## 📊 Dataset Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Movies", f"{len(movies_df):,}")
        with col2:
            if rating_stats_df is not None:
                st.metric("Movies with Ratings", f"{len(rating_stats_df):,}")
            else:
                st.metric("Ratings Data", "Not Available")
        
        # Genre list
        genres = get_genre_list(movies_df)
        st.markdown(f"**Genres Available:** {len(genres)}")
        with st.expander("View All Genres"):
            for genre in genres:
                st.markdown(f"- {genre}")
        
        st.markdown("---")
        
        # Recommendation settings
        st.markdown("## ⚙️ Recommendation Settings")
        n_recommendations = st.selectbox(
            "Number of Recommendations",
            options=[5, 10, 15, 20],
            index=1,
            help="How many similar movies to show"
        )
        
        min_similarity = st.slider(
            "Minimum Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Filter out recommendations below this similarity score"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ How It Works")
        st.markdown("""
        1. **Select a movie** from the search or genre filter
        2. **Click 'Get Recommendations'** to find similar movies
        3. **Explore** - click any recommended movie to see movies like it
        
        Similarity is calculated from **genre overlap only**, not ratings or personal history.
        """)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 Discover Movies", "🎯 Recommendations", "📈 Popular Movies"])
    
    # Session state for selected movie
    if "selected_movie_id" not in st.session_state:
        st.session_state.selected_movie_id = None
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = None
    
    # Tab 1: Movie Discovery
    with tab1:
        st.markdown("## 🔍 Discover Movies")
        
        # Search and filter controls
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input(
                "🔎 Search by Title",
                placeholder="Type a movie title...",
                help="Partial, case-insensitive search"
            )
        
        with col2:
            genre_filter = st.selectbox(
                "🎭 Filter by Genre",
                options=["All"] + genres,
                help="Filter movies by a specific genre"
            )
        
        # Apply filters
        filtered_movies = movies_df
        if search_query:
            filtered_movies = search_movies(filtered_movies, search_query)
        if genre_filter != "All":
            filtered_movies = filter_movies_by_genre(filtered_movies, genre_filter)
        
        # Display results count
        st.markdown(f"**Showing {len(filtered_movies):,} of {len(movies_df):,} movies**")
        
        # Display movies in a selectable table
        if not filtered_movies.empty:
            # Prepare display DataFrame
            display_df = filtered_movies[["movieId", "title", "genres"]].copy()
            display_df["genres"] = display_df["genres"].replace("(no genres listed)", "No genres listed")
            
            # Add rating info if available
            if rating_stats_df is not None:
                display_df = display_df.merge(
                    rating_stats_df[["movieId", "avg_rating", "rating_count"]],
                    on="movieId",
                    how="left"
                )
                display_df["avg_rating"] = display_df["avg_rating"].round(1)
                display_df = display_df.rename(columns={
                    "movieId": "ID",
                    "title": "Title",
                    "genres": "Genres",
                    "avg_rating": "Avg Rating",
                    "rating_count": "# Ratings"
                })
            else:
                display_df = display_df.rename(columns={
                    "movieId": "ID",
                    "title": "Title",
                    "genres": "Genres"
                })
            
            # Show as interactive table with selection
            event = st.dataframe(
                display_df.head(100),
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "ID": st.column_config.NumberColumn("ID", width="small"),
                    "Title": st.column_config.TextColumn("Title", width="large"),
                    "Genres": st.column_config.TextColumn("Genres", width="large"),
                } if rating_stats_df is None else {
                    "ID": st.column_config.NumberColumn("ID", width="small"),
                    "Title": st.column_config.TextColumn("Title", width="large"),
                    "Genres": st.column_config.TextColumn("Genres", width="large"),
                    "Avg Rating": st.column_config.NumberColumn("Avg Rating", format="%.1f", width="small"),
                    "# Ratings": st.column_config.NumberColumn("# Ratings", format="%d", width="small"),
                }
            )
            
            # Handle selection
            if event.selection.rows:
                selected_idx = event.selection.rows[0]
                selected_movie_id = int(display_df.iloc[selected_idx]["ID"])
                st.session_state.selected_movie_id = selected_movie_id
                st.success(f"Selected: {display_df.iloc[selected_idx]['Title']}")
                
                if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
                    with st.spinner("Finding similar movies..."):
                        try:
                            recs = recommender.get_similar_movies(
                                selected_movie_id,
                                n_recommendations=n_recommendations,
                                min_similarity=min_similarity
                            )
                            st.session_state.recommendations = recs
                            st.rerun()
                        except RecommenderError as e:
                            st.error(f"Recommendation error: {e}")
                        except Exception as e:
                            st.error(f"Unexpected error: {e}")
        else:
            st.warning("No movies match your search/filter criteria.")
    
    # Tab 2: Recommendations
    with tab2:
        st.markdown("## 🎯 Recommendations")
        
        if st.session_state.selected_movie_id is None:
            st.markdown("""
            <div class="info-box">
            No movie selected yet. Go to the <strong>Discover Movies</strong> tab, 
            select a movie from the table, and click <strong>Get Recommendations</strong>.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show selected movie
            selected_movie = recommender.get_movie_by_id(st.session_state.selected_movie_id)
            
            if selected_movie is not None:
                genres = selected_movie.get("genres", "")
                if genres == "(no genres listed)" or not genres:
                    genres_display = "No genres listed"
                else:
                    genres_display = genres.replace("|", ", ")
                
                st.markdown(f"""
                <div class="selected-movie">
                    <h3 style="margin: 0 0 0.5rem 0; color: #ff6b35;">{selected_movie['title']}</h3>
                    <p style="margin: 0; color: #aaa;">Genres: {genres_display}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show rating info for selected movie
                if rating_stats_df is not None:
                    rating_info = rating_stats_df[
                        rating_stats_df["movieId"] == st.session_state.selected_movie_id
                    ]
                    if not rating_info.empty:
                        avg_r = rating_info.iloc[0]["avg_rating"]
                        count_r = int(rating_info.iloc[0]["rating_count"])
                        st.markdown(f"**Average Rating:** ★ {avg_r:.1f} ({count_r:,} ratings)")
            
            # Show recommendations
            if st.session_state.recommendations is not None:
                recs = st.session_state.recommendations
                
                if recs.empty:
                    st.markdown("""
                    <div class="warning-box">
                    <strong>No meaningful recommendations found.</strong><br>
                    The selected movie may have unique genres with no similar movies 
                    above the similarity threshold. Try lowering the minimum similarity 
                    or selecting a different movie.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"### Top {len(recs)} Similar Movies")
                    
                    # Display as cards
                    for _, rec in recs.iterrows():
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            display_movie_card(
                                rec, 
                                show_similarity=True, 
                                similarity_score=rec["similarity_score"],
                                key_prefix="rec"
                            )
                        with col2:
                            if st.button("Explore", key=f"explore_{rec['movieId']}", use_container_width=True):
                                st.session_state.selected_movie_id = int(rec["movieId"])
                                st.session_state.recommendations = None
                                st.rerun()
                    
                    # Also show as table for easy comparison
                    with st.expander("📋 View as Table"):
                        table_df = recs[["movieId", "title", "genres", "similarity_score"]].copy()
                        if "avg_rating" in recs.columns:
                            table_df = table_df.merge(
                                recs[["movieId", "avg_rating", "rating_count"]],
                                on="movieId"
                            )
                        table_df["similarity_score"] = (table_df["similarity_score"] * 100).round(1).astype(str) + "%"
                        table_df["genres"] = table_df["genres"].replace("(no genres listed)", "No genres listed")
                        st.dataframe(
                            table_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "movieId": "ID",
                                "title": "Title",
                                "genres": "Genres",
                                "similarity_score": "Similarity",
                                "avg_rating": st.column_config.NumberColumn("Avg Rating", format="%.1f"),
                                "rating_count": "# Ratings"
                            }
                        )
            else:
                st.info("Click **Get Recommendations** in the Discover tab to generate recommendations.")
    
    # Tab 3: Popular Movies (if ratings available)
    with tab3:
        st.markdown("## 📈 Popular & Highly Rated Movies")
        
        if rating_stats_df is None or rating_stats_df.empty:
            st.markdown("""
            <div class="info-box">
            Ratings data is not available. This section requires the <code>ratings.csv</code> 
            file from the MovieLens dataset.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Merge with movies for titles/genres
            popular_df = movies_df.merge(rating_stats_df, on="movieId", how="inner")
            
            # Minimum rating count threshold
            min_ratings = st.slider(
                "Minimum Rating Count",
                min_value=1,
                max_value=200,
                value=50,
                step=10,
                help="Only show movies with at least this many ratings"
            )
            
            popular_df = popular_df[popular_df["rating_count"] >= min_ratings]
            
            if popular_df.empty:
                st.warning(f"No movies have at least {min_ratings} ratings.")
            else:
                # Top rated
                st.markdown("### ⭐ Highest Rated (Bayesian-style)")
                
                # Simple Bayesian average: weighted by rating count
                # Using a prior of global average with weight equivalent to min_ratings
                global_avg = popular_df["avg_rating"].mean()
                popular_df["bayesian_score"] = (
                    (popular_df["avg_rating"] * popular_df["rating_count"] + global_avg * min_ratings) /
                    (popular_df["rating_count"] + min_ratings)
                )
                
                top_rated = popular_df.nlargest(10, "bayesian_score")
                
                for _, movie in top_rated.iterrows():
                    display_movie_card(movie, key_prefix="top")
                
                st.markdown("---")
                
                # Most rated
                st.markdown("### 📊 Most Rated")
                most_rated = popular_df.nlargest(10, "rating_count")
                
                for _, movie in most_rated.iterrows():
                    display_movie_card(movie, key_prefix="popular")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.85rem;'>"
        "Movie Recommendation System | Content-Based Filtering with TF-IDF & Cosine Similarity<br>"
        "Data: MovieLens ml-latest-small (GroupLens Research) | "
        "<a href='https://grouplens.org/datasets/movielens/' style='color: #ff6b35;'>Dataset Source</a>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()