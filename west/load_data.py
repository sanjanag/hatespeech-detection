import csv
import itertools
import re
from collections import Counter
from os.path import join

import numpy as np
from nltk import tokenize

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def read_file(data_dir, with_evaluation):
    data = []
    target = []
    with open(join(data_dir, 'dataset.csv'), 'rt',
              encoding='utf-8') as csvfile:
        csv.field_size_limit(500 * 1024 * 1024)
        reader = csv.reader(csvfile)
        for row in reader:
            if data_dir == './agnews':
                doc = row[1] + '. ' + row[2]
                data.append(doc)
                target.append(int(row[0]) - 1)
            elif data_dir == './yelp':
                data.append(row[1])
                target.append(int(row[0]) - 1)
            elif data_dir == '../hatespeech':
                data.append(row[1])
                target.append(int(row[0]))
    if with_evaluation:
        y = np.asarray(target)
        assert len(data) == len(y)
        assert set(range(len(np.unique(y)))) == set(np.unique(y))
    else:
        y = None
    return data, y

def clean_twitter(text):
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r'\w+@\w+\.\w+', ' ', text)  # remove emails
    text = re.sub(r'(https{0,1}\:\/\/.+?(\s|$))|(www\..+?(\s|$))|('
                  r'\b\w+\.twitter\.com.+?(\s|$))', ' ',
                  text)  # remove urls
    text = re.sub(r'(@[A-Za-z0-9_]+:?(\s|$))', ' ', text)  # remove mentions
    text = re.sub(r'\b(RT|rt)\b', ' ', text)  # remove retweets
    text = re.sub(r'(&#\d+;)+', ' ', text)  # remove retweets
    text = re.sub(r'&\w+;(\w)?', ' ', text)
    text = re.sub(r'(#[A-Za-z0-9_]+)', ' ', text)  # remove hashtags
    text = re.sub(r'(\.){2,}', '.', text)
    text = re.sub(r"\s{2,}", " ", text)
    # text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = text.lower()
    return text

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9\']", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\"", " \" ", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"\.", " . ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\$", " $ ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def preprocess_doc(data, is_tweet=False):
    data = [s.strip() for s in data]
    if is_tweet:
        data = [clean_twitter(s) for s in data]
    data = [clean_str(s) for s in data]
    return data


def pad_sequences(sentences, padding_word="<PAD/>", pad_len=None):
    if pad_len is not None:
        sequence_length = pad_len
    else:
        sequence_length = max(len(x) for x in sentences)

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return word_counts, vocabulary, vocabulary_inv


def build_input_data_cnn(sentences, vocabulary):
    x = np.array(
        [[vocabulary[word] for word in sentence] for sentence in sentences])
    return x


def build_input_data_rnn(data, vocabulary, max_doc_len):
    x = np.zeros((len(data), max_doc_len), dtype='int32')
    for i, doc in enumerate(data):
        for j, word in enumerate(doc):
            x[i, j] = vocabulary[word]
    return x


def extract_keywords(data_path, vocab, class_type, num_keywords, data, perm):
    sup_data = []
    sup_idx = []
    sup_label = []
    file_name = 'doc_id.txt'
    infile = open(join(data_path, file_name), mode='r', encoding='utf-8')
    text = infile.readlines()
    for i, line in enumerate(text):
        line = line.split('\n')[0]
        class_id, doc_ids = line.split(':')
        assert int(class_id) == i
        seed_idx = doc_ids.split(',')
        seed_idx = [int(idx) for idx in seed_idx]
        sup_idx.append(seed_idx)
        for idx in seed_idx:
            sup_data.append(" ".join(data[idx]))
            sup_label.append(i)


    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    import nltk

    with open('stopwords.txt', 'r') as f:
        lines = f.readlines()
    stopwords = [w.strip() for w in lines]

    count_vectorizer = CountVectorizer(input='content',
                                       analyzer='word',
                                       strip_accents='ascii',
                                       ngram_range=(1, 1),
                                       stop_words=stopwords,
                                       token_pattern=r'\b[^\d\W]+\b')
    count = count_vectorizer.fit_transform(sup_data)
    features = np.array(count_vectorizer.get_feature_names())
    freq = count.copy()
    count[count > 0] = 1

    print("\n### Supervision type: Labeled documents ###")
    print("Extracted keywords for each class: ")
    keywords = []
    cnt = 0
    rankingdf = pd.DataFrame(columns=['word', 'rel_doc_freq',
                                      'avg_freq', 'idf'])
    rankingdf['word'] = features
    for i in range(len(sup_idx)):
        start = cnt
        end = cnt + len(sup_idx[i])
        cnt += len(sup_idx[i])
        class_docs = count[start: end]
        rel_doc_freq = np.array(class_docs.sum(axis=0) / class_docs.shape[0])[0]
        avg_freq = np.array(freq[start:end].sum(axis=0) /class_docs.shape[0])[0]
        rankingdf['rel_doc_freq'] = rel_doc_freq
        rankingdf['avg_freq'] = avg_freq
        rankingdf['idf'] = np.log(np.array(count.shape[0] / count.sum(axis=0))[0])

        scaler = MinMaxScaler()
        scaler.fit(rankingdf[['rel_doc_freq', 'idf', 'avg_freq']])
        rankingdf[['rel_doc_freq', 'idf', 'avg_freq']] = scaler.transform(rankingdf[['rel_doc_freq', 'idf', 'avg_freq']])
        rankingdf['comb'] = np.cbrt(rankingdf['rel_doc_freq'] * rankingdf['idf'] * rankingdf['avg_freq'])
        keyword = rankingdf.sort_values(by=['comb'], ascending=False).head(
            num_keywords)['word'].tolist()
        keywords.append(keyword)


    new_sup_idx = []
    m = {v: k for k, v in enumerate(perm)}
    for seed_idx in sup_idx:
        new_seed_idx = []
        for ele in seed_idx:
            new_seed_idx.append(m[ele])
        new_sup_idx.append(new_seed_idx)
    new_sup_idx = np.asarray(new_sup_idx)

    return keywords, new_sup_idx


def load_keywords(data_path, sup_source):
    if sup_source == 'labels':
        file_name = 'classes.txt'
        print("\n### Supervision type: Label Surface Names ###")
        print("Label Names for each class: ")
    elif sup_source == 'keywords':
        file_name = 'keywords.txt'
        print("\n### Supervision type: Class-related Keywords ###")
        print("Keywords for each class: ")
    infile = open(join(data_path, file_name), mode='r', encoding='utf-8')
    text = infile.readlines()

    keywords = []
    for i, line in enumerate(text):
        line = line.split('\n')[0]
        class_id, contents = line.split(':')
        assert int(class_id) == i
        keyword = contents.split(',')
        print("Supervision content of class {}:".format(i))
        print(keyword)
        keywords.append(keyword)
    return keywords


def load_cnn(dataset_name, sup_source, num_keywords=10, with_evaluation=True,
             truncate_len=None):
    data_path = '../' + dataset_name
    data, y = read_file(data_path, with_evaluation)

    sz = len(data)
    np.random.seed(1234)
    perm = np.random.permutation(sz)

    if dataset_name == 'hatespeech':
        data = preprocess_doc(data, True)
    else:
        data = preprocess_doc(data)

    data = [s.split(" ") for s in data]

    tmp_list = [len(doc) for doc in data]
    len_max = max(tmp_list)
    len_avg = np.average(tmp_list)
    len_std = np.std(tmp_list)

    print("\n### Dataset statistics: ###")
    print('Document max length: {} (words)'.format(len_max))
    print('Document average length: {} (words)'.format(len_avg))
    print('Document length std: {} (words)'.format(len_std))

    if truncate_len is None:
        truncate_len = min(int(len_avg + 3 * len_std), len_max)
    print("Defined maximum document length: {} (words)".format(truncate_len))
    print('Fraction of truncated documents: {}'.format(
        sum(tmp > truncate_len for tmp in tmp_list) / len(tmp_list)))

    sequences_padded = pad_sequences(data)
    word_counts, vocabulary, vocabulary_inv = build_vocab(sequences_padded)
    x = build_input_data_cnn(sequences_padded, vocabulary)
    x = x[perm]

    if with_evaluation:
        print("Number of classes: {}".format(len(np.unique(y))))
        print("Number of documents in each class:")
        for i in range(len(np.unique(y))):
            print("Class {}: {}".format(i, len(np.where(y == i)[0])))
        y = y[perm]

    print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

    if sup_source == 'labels' or sup_source == 'keywords':
        keywords = load_keywords(data_path, sup_source)
        return x, y, word_counts, vocabulary, vocabulary_inv, len_avg, \
               len_std, keywords, perm
    elif sup_source == 'docs':
        if dataset_name == 'nyt':
            class_type = 'topic'
        elif dataset_name == 'agnews':
            class_type = 'topic'
        elif dataset_name == 'yelp':
            class_type = 'sentiment'
        elif dataset_name == 'hatespeech':
            class_type = 'topic'
        keywords, sup_idx = extract_keywords(data_path, vocabulary, class_type,
                                             num_keywords, data, perm)
        return x, y, word_counts, vocabulary, vocabulary_inv, len_avg, \
               len_std, keywords, sup_idx, perm


def load_rnn(dataset_name, sup_source, num_keywords=10, with_evaluation=True,
             truncate_len=None):
    data_path = '../' + dataset_name
    data, y = read_file(data_path, with_evaluation)

    sz = len(data)
    np.random.seed(1234)
    perm = np.random.permutation(sz)

    data = preprocess_doc(data, True)
    data = [s.split(" ") for s in data]

    tmp_list = [len(doc) for doc in data]
    len_max = max(tmp_list)
    len_avg = np.average(tmp_list)
    len_std = np.std(tmp_list)

    print("\n### Dataset statistics: ###")
    print('Document max length: {} (words)'.format(len_max))
    print('Document average length: {} (words)'.format(len_avg))
    print('Document length std: {} (words)'.format(len_std))

    if truncate_len is None:
        truncate_len = min(int(len_avg + 3 * len_std), len_max)
    print("Defined maximum document length: {} (words)".format(truncate_len))
    print('Fraction of truncated documents: {}'.format(
        sum(tmp > truncate_len for tmp in tmp_list) / len(tmp_list)))

    data = pad_sequences(data)
    word_counts, vocabulary, vocabulary_inv = build_vocab(data)

    x = build_input_data_rnn(data, vocabulary, len_max)

    x = x[perm]

    if with_evaluation:
        print("Number of classes: {}".format(len(np.unique(y))))
        print("Number of documents in each class:")
        for i in range(len(np.unique(y))):
            print("Class {}: {}".format(i, len(np.where(y == i)[0])))
        y = y[perm]

    print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

    if sup_source == 'labels' or sup_source == 'keywords':
        keywords = load_keywords(data_path, sup_source)
        return x, y, word_counts, vocabulary, vocabulary_inv, len_avg, \
               len_std, keywords, perm
    elif sup_source == 'docs':
        if dataset_name == 'nyt':
            class_type = 'topic'
        elif dataset_name == 'agnews':
            class_type = 'topic'
        elif dataset_name == 'yelp':
            class_type = 'sentiment'
        elif dataset_name == 'hatespeech':
            class_type = 'topic'
        keywords, sup_idx = extract_keywords(data_path, vocabulary, class_type,
                                             num_keywords, data, perm)
        return x, y, word_counts, vocabulary, vocabulary_inv, len_avg, \
               len_std, keywords, sup_idx, perm


def load_dataset(dataset_name, sup_source, model='cnn', with_evaluation=True,
                 truncate_len=None):
    if model == 'cnn':
        return load_cnn(dataset_name, sup_source,
                        with_evaluation=with_evaluation,
                        truncate_len=truncate_len)
    elif model == 'rnn':
        return load_rnn(dataset_name, sup_source,
                        with_evaluation=with_evaluation,
                        truncate_len=truncate_len)
