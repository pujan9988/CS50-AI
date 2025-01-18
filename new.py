import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import re,string 

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("shared_articles.csv")
df = df.drop(columns=['timestamp', 'eventType', 'contentId', 'authorPersonId',
       'authorSessionId', 'authorUserAgent', 'authorRegion', 'authorCountry',
       'contentType', 'url',])
df = df.loc[df["lang"]=="en",["title","text"]]
df = df.drop_duplicates()
df = df.reset_index(drop=True)
df1 = df.copy()

def remove_whitespaces(row):
    pattern = r"\s+"
    row = re.sub(pattern," ",row).lower()
    return row

df["title"] = df["title"].apply(remove_whitespaces)
df["text"] = df["text"].apply(remove_whitespaces)
df["no_of_words"] = df["text"].str.split(" ").apply(len)
df["unique_no_of_words"] = df["text"].str.split(" ").apply(set).apply(len)

def remove_punctuations(row):
    punctuation = re.escape(string.punctuation)
    pattern = f"[{punctuation}0-9]"
    row = re.sub(pattern,"",row)
    return row
df["title"] = df["title"].apply(remove_punctuations)
df["text"] = df["text"].apply(remove_punctuations)


words = stopwords.words('english')
def remove_stopwords(row):
    word_list = row.split(" ")
    new_word_list = [word for word in word_list if word not in ENGLISH_STOP_WORDS]
    return " ".join(new_word_list)

df_new = pd.DataFrame()
df_new["title"] = df["title"].apply(remove_stopwords)
df_new["text"] = df["text"].apply(remove_stopwords)

 #stemming tool
stemmer = PorterStemmer()
def stem_words(row):
    tokens = word_tokenize(row)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

df_new["data"] = df_new["title"] + " " + df_new["text"]

vectorizer = CountVectorizer(max_features=5000)
vectorized_data = vectorizer.fit_transform(df_new["data"]).toarray()

similarity_score = cosine_similarity(vectorized_data)

tfidf = TfidfVectorizer(max_features=5000)
tf_vectors = tfidf.fit_transform(df_new["data"]).toarray()
tf_similarity = cosine_similarity(tf_vectors)


def recommend_article(title:str):
    title_idx = df1[df1["title"]==title].index[0]
    similar_idx_scores = list(enumerate(tf_similarity[title_idx]))
    sorted_similar_idx = sorted(similar_idx_scores,key=lambda x:x[1],reverse=True)
    recommended_idx = sorted_similar_idx[1:4]
    return recommended_idx


tit = df1["title"][1467]

print(f"\nThe clicked article is:\n{tit}\n\n")
recommended_articles = recommend_article(tit)

print("The recommend articles are:")
for i,val in enumerate(recommended_articles):
    print(f"{i+1} --- {df1["title"][val[0]]}\n")


