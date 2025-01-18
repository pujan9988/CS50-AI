import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

# Page configuration
st.set_page_config(
    page_title="Article Recommender",
    layout="wide"
)

# Custom CSS with improved visibility
st.markdown("""
    <style>
    .article-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        color: #1f1f1f;  /* Dark text color for contrast */
    }
    .article-title {
        color: #1f1f1f;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .article-preview {
        color: #2d2d2d;  /* Darker grey for better visibility */
        font-size: 1rem;
        line-height: 1.6;
    }
    .article-full-text {
        color: #2d2d2d;
        font-size: 1.1rem;
        line-height: 1.8;
        white-space: pre-wrap;  /* Preserve formatting */
    }
    .recommendation-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    .recommendation-title {
        color: #1f1f1f;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .recommendation-preview {
        color: #2d2d2d;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .page-navigation {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 2rem 0;
    }
    .search-results {
        margin: 1rem 0;
        padding: 0.5rem;
        background-color: #e9ecef;
        border-radius: 5px;
        color: #1f1f1f;
    }
    /* Override Streamlit's default text color */
    .stMarkdown, .stText {
        color: #1f1f1f !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and prepare data functions (unchanged)
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned.csv")
    train_df = pd.read_csv("training.csv")
    return (df,train_df)

@st.cache_resource
def prepare_similarity_matrix(df):
    tfidf = TfidfVectorizer(max_features=5000)
    tf_vectors = tfidf.fit_transform(df["data"]).toarray()
    tf_similarity = cosine_similarity(tf_vectors)
    return tf_similarity

def get_recommended_articles(title, df, tf_similarity):
    title_idx = df[df["title"] == title].index[0]
    similar_idx_scores = list(enumerate(tf_similarity[title_idx]))
    sorted_similar_idx = sorted(similar_idx_scores, key=lambda x: x[1], reverse=True)
    recommended_idx = sorted_similar_idx[1:4]
    return recommended_idx

def truncate_text(text, max_words=50):
    return " ".join(text.split()[:max_words]) + "..."

# Load data
df,train_df = load_data()
tf_similarity = prepare_similarity_matrix(train_df)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Sidebar with improved visibility
with st.sidebar:
    st.title("Navigation")
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = 'home'
        st.rerun()
    
    st.markdown("---")
    search_query = st.text_input("🔍 Search Articles:")

# Main content
if st.session_state.page == 'home':
    st.title("📚 Article Collection")
    
    # Search functionality
    if search_query:
        mask = (df["title"].str.contains(search_query, case=False)) | \
               (df["text"].str.contains(search_query, case=False))
        filtered_df = df[mask]
        st.markdown(f"""
            <div class="search-results">
                📊 Found {len(filtered_df)} articles matching '{search_query}'
            </div>
            """, unsafe_allow_html=True)
    else:
        filtered_df = df

    # Pagination
    articles_per_page = 10
    total_pages = len(filtered_df) // articles_per_page + (1 if len(filtered_df) % articles_per_page > 0 else 0)
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        page_number = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1) - 1
    
    start_idx = page_number * articles_per_page
    end_idx = start_idx + articles_per_page
    page_df = filtered_df.iloc[start_idx:end_idx]

    # Display articles
    for _, row in page_df.iterrows():
        st.markdown(f"""
            <div class="article-card">
                <div class="article-title">{row["title"]}</div>
                <div class="article-preview">{truncate_text(row["text"])}</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("📖 Read Full Article", key=f"read_{_}"):
            st.session_state.page = 'article'
            st.session_state.article_title = row["title"]
            st.rerun()

else:  # Article page
    # Back button in sidebar
    if st.sidebar.button("← Back to Articles", use_container_width=True):
        st.session_state.page = 'home'
        st.rerun()

    # Display full article
    article_data = df[df["title"] == st.session_state.article_title].iloc[0]
    
    st.title(article_data["title"])
    
    # Article container with improved visibility
    st.markdown(f"""
        <div class="article-card">
            <div class="article-full-text">
                {article_data["text"]}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Recommendations section
    st.markdown("---")
    st.subheader("📚 Recommended Articles")
    recommended_articles = get_recommended_articles(st.session_state.article_title, df, tf_similarity)
    
    cols = st.columns(3)
    for idx, (article_idx, similarity_score) in enumerate(recommended_articles):
        with cols[idx]:
            st.markdown(f"""
                <div class="recommendation-card">
                    <div class="recommendation-title">{df['title'].iloc[article_idx]}</div>
                    <div class="recommendation-preview">{truncate_text(df["text"].iloc[article_idx], max_words=30)}</div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("📖 Read This Article", key=f"rec_{article_idx}"):
                st.session_state.article_title = df["title"].iloc[article_idx]
                st.rerun()