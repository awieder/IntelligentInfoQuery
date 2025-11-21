import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
#Load Data
data = pd.read_csv("data/Articles.csv", encoding="ISO-8859-1")
#Clean Text:
data = data.dropna(subset=['Article']).reset_index(drop=True)
def cleaner(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'http\S+', '', text)
    return text

data['Clean Articles'] = data['Article'].apply(cleaner)
#Load Matrix
vectorizer = CountVectorizer(max_features=2000, stop_words='english')
X = vectorizer.fit_transform(data['Clean Articles'])
word_freq_matrix = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
labels = data['NewsType']

# print(word_freq_matrix)
#Preform Truncated Decomposition
def TruncateSVD(k, X):
    svd = TruncatedSVD(n_components = k)
    X_svd = svd.fit_transform(X)
    return svd, X_svd

def runQuery(query, svd, k, vectorizer, X):
    query_clean = cleaner(query)
    q_vec = vectorizer.transform([query_clean])
    q_svd = svd.transform(q_vec)
    sims = cosine_similarity(q_svd, X_k)[0]
    top_indices = sims.argsort()[::-1][:5]
    headers = data['Heading'].iloc[top_indices].tolist()
    return headers

def retrieval_precision_at_5(X_k, labels):
    sims = cosine_similarity(X_k)
    precisions = []

    for i in range(len(labels)):
        top = sims[i].argsort()[::-1][1:6]  # skip self
        true_matches = sum(labels[top] == labels[i])
        precisions.append(true_matches / 10)

    return np.mean(precisions)

results = []
for k in range(1000):
    svd, X_k = TruncateSVD(k, X)
    p10 = retrieval_precision_at_5(X_k, labels.to_numpy())
    results.append({"k": k, "P@10": p10})
df_results = pd.DataFrame(results)
print(df_results)

# query = "fuel economy driving car"
# print(runQuery(query, svd, 1, vectorizer, X_k))