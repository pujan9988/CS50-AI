# News Recommendation System - CS50x AI with CS50x Nepal

Welcome to the **News Recommendation System** project taught during **Week 5, Day 1 of CS50x AI** with **CS50x Nepal**.

In this hands-on session, students learned about practical applications of AI and implemented a simple content-based **Recommendation System** using fundamental Natural Language Processing (NLP) techniques.

👉 [Visit CS50x Nepal](https://cs50xnepal.ioepc.edu.np/)

## Project Overview

This project focuses on recommending similar news articles using:

* **Bag of Words (BoW)**
* **TF-IDF (Term Frequency - Inverse Document Frequency)**
* **Cosine Similarity**

We used a dataset of shared articles and went through the complete cycle from raw data to a functional recommender.

---

## Concepts Covered

* Exploratory Data Analysis (EDA)
* Text Cleaning & Preprocessing
* Feature Extraction (BoW, TF-IDF)
* Similarity Calculation (Cosine Similarity)
* Recommendation Logic

---

## Files 

* `new.ipynb`: Complete prototype notebook project code including preprocessing, visualization, and recommendation system logic.
* `gui.py`: GUI implementation using streamlit.
  [Website implementation link](https://pujanpaudel-news-recommendation-system.hf.space)


---

## Dataset Summary

The original dataset includes articles shared by users. It contains various metadata fields, the main usesful ones are:

* `title`: Title of the article
* `text`: Full article content
* `lang`: Language of the article (filtered to English only)

---

## Steps Followed

### 1. **Data Cleaning**

* Removed irrelevant columns and non-English articles
* Removed duplicate entries

### 2. **Text Preprocessing**

* Lowercased text and removed extra spaces
* Removed punctuation and numbers
* Removed stopwords using `sklearn`'s `ENGLISH_STOP_WORDS`
* Applied stemming using `nltk.PorterStemmer`
* Combined `title` and `text` into a single `data` column

### 3. **Exploratory Data Analysis**

* Word count and unique word count per article
* Histograms for word distributions
* Wordcloud visualization
* Most frequent stopwords bar chart

### 4. **Vectorization**

* **Bag of Words**: `CountVectorizer`
* **TF-IDF**: `TfidfVectorizer`

### 5. **Similarity & Recommendation**

* Calculated **cosine similarity** between all articles
* Built a function `recommend_article(title)` to return top 3 similar articles

---

## Example Recommendation

Given article:

> *"The future of financial infrastructure: An ambitious look at how blockchain can reshape financial services"*

Top 3 recommendations:

1. Banca IMI Researcher: Blockchain Won't Work if Business Models Don't Change
2. 4 Blockchain Macro Trends: Where to Place Your Bets
3. Banks' Privacy Concerns Shaping Blockchain Vendors' Strategies

---

## Technologies Used

* Python 
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-learn
* NLTK
* WordCloud

---

##  Learning Outcome

Students learned how AI powers real-world systems like news and product recommendations using simple yet powerful techniques.

This project showed that even without deep neural networks, **smart preprocessing and basic vectorization can build useful AI applications**.

---

## Taught by

**[Pujan Paudel](https://www.linkedin.com/posts/pujanpaudel_ai-recommendationsystems-cs50xnepal-activity-7286786671897960448-Dm89)**
CS50x AI Instructor, CS50x Nepal

---

> "AI isn't a magic. It's just math, models, and a lot of data."
