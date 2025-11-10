import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
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