import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# get random 20 documents
documents = np.random.choice(newsgroups.data, size=20, replace=False)

# Preprocess the documents (TF-IDF vectorization)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Calculate document similarity (cosine similarity)
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", xticklabels=range(1, 21), yticklabels=range(1, 21))
plt.title("Document Similarity Heatmap (First 20 Documents)")
plt.show()
