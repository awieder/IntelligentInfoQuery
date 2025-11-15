import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
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
print(word_freq_matrix)
#Preform Truncated Decomposition
def TruncateSVD(k, X):
    svd = TruncatedSVD(n_components = k)
    X_svd = svd.fit_transform(X)
    return X_svd

X_low_k = TruncateSVD(5, X)
svd_df = pd.DataFrame(X_low_k)
print(svd_df)
# X_mid_k = TruncateSVD(50, X)
# X_high_k = TruncateSVD(100, X)