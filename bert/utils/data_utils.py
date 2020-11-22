import os
import pickle
import re
import string
import torch
import numpy as np
import preprocessor as p
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

from west.load_data import read_file

tk = TweetTokenizer()
stop_words = set(stopwords.words('english'))


def preprocess(raw_data):
    processed_data = []

    for text in raw_data:
        text = p.clean(text)
        text = re.sub(r'(&#\d+;)+', ' ', text)
        text = re.sub(r'&[\w]*;(\w)?', ' ', text)
        text = re.sub(r':', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = tk.tokenize(text)
        # tokens = [token for token in tokens if token not in stop_words]
        text = " ".join(tokens)
        processed_data.append(text.lower())
    return processed_data


def get_processed_data():
    data_path = "../hatespeech"
    raw_data, y = read_file(data_path, True)
    if not os.path.exists("processed.pkl"):
        data = preprocess(raw_data)
        with open("processed.pkl", "wb") as f:
            pickle.dump([data, y], f)
    else:
        with open("processed.pkl", "rb") as f:
            data, y = pickle.load(f)
    return np.array(data), np.array(y)


def get_sup_docs(data, y):
    doc_path = "../hatespeech/doc_100.txt"
    with open(doc_path, "r") as f:
        lines = f.readlines()

    sup_idx = []
    sup_data = []
    sup_label = []
    for class_id, line in enumerate(lines):
        line = line.strip('\n')
        label, doc_ids = line.split(':')
        assert int(label) == class_id
        seed_idx = doc_ids.split(',')
        seed_idx = [int(idx) for idx in seed_idx]
        sup_idx.append(seed_idx)
        for idx in seed_idx:
            sup_data.append(data[idx])
            assert class_id == y[idx]
            sup_label.append(class_id)
    return sup_idx

def combine_data(d1, d2):
    combined = {}
    combined['input_ids'] = torch.cat((d1['input_ids'], d2['input_ids']),
                                      dim=0)
    combined['attention_mask'] = torch.cat(
        (d1['attention_mask'], d2['attention_mask']), dim=0)
    combined['token_type_ids'] = torch.cat(
        (d1['token_type_ids'], d2['token_type_ids']), dim=0)
    combined['labels'] = torch.cat((d1['labels'], d2['labels']), dim=0)
    combined['pseudolabels'] = torch.cat(
        (d1['pseudolabels'], d2['pseudolabels']), dim=0)
    return combined
