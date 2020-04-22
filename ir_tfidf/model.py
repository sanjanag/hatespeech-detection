import pickle
from os import path

import numpy as np
import spacy
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from utils import read_file, preprocess, stem

nlp = spacy.load('en_core_web_sm')
stemmer = PorterStemmer()

# Load processed data: raw -> cleaning -> stemming
processed_datafile = 'processed.pkl'
if path.exists(processed_datafile):
    with open(processed_datafile, 'rb') as f:
        data, y = pickle.load(f)
else:
    data, y = read_file('../hatespeech', True)
    data = [preprocess(text) for text in data]
    with open('processed.pkl', 'wb') as f:
        pickle.dump((data, y), f)


# Define keywords
keywords= {
    1:['new','free','video','check','win'],
    2:['fucked','bitch','pussy','ass','ugly'],
    3:['hate','racist','muslims', 'retarded', 'isis']
}

for class_label, words in keywords.items():
    keywords[class_label] = [stem(w) for w in words]

# get tf idf features
vectorizer = TfidfVectorizer(input='content', encoding='ascii',
                             decode_error='ignore', strip_accents='ascii',
                             stop_words='english', min_df=2)
tfidf_weights = vectorizer.fit_transform(data)
vocabulary = vectorizer.vocabulary_


# get tf idf weights for keywords
keyword_indices = {}
for class_label, words in keywords.items():
    keyword_indices[class_label] = [vocabulary[w] for w in words]
class_weights = []
for class_label in range(1, 4):
    indices = keyword_indices[class_label]
    class_weights.append(
        np.array(tfidf_weights[:, indices].sum(axis=1)).flatten())

# assign label based on aggregate, if aggregate value is 0 for ll other
# labels assign 'normal' label
non_normal_weights = np.vstack(class_weights).T
y_pred = []
max_class = np.argmax(non_normal_weights, axis=1)
for i in range(len(data)):
    if non_normal_weights[i][max_class[i]] > 0:
        y_pred.append(max_class[i])
    else:
        y_pred.append(0)

# print classification report
print(classification_report(y, y_pred))
