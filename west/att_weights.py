from keras.models import model_from_json
from sklearn.metrics import classification_report

from load_data import load_dataset
from model import AttentionWithContext, dot_product

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# load weights into new model
loaded_model = model_from_json(loaded_model_json, custom_objects={
    'AttentionWithContext': AttentionWithContext})
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

print(loaded_model.layers)
print(loaded_model.summary())
from keras import backend as K

# with a Sequential model
get_td_layer_output = K.function([loaded_model.layers[0].input],
                                 [loaded_model.layers[3].output])

x, y, word_counts, vocabulary, vocabulary_inv_list, len_avg, \
len_std, word_sup_list, sup_idx, perm = \
    load_dataset('hatespeech', model='rnn',
                 sup_source='docs',
                 with_evaluation=True,
                 truncate_len=114)

print('Loaded data')

print("Computing predicted labels")
y_pred = loaded_model.predict(x).argmax(axis=1)
print(classification_report(y, y_pred))

print("Computing 3rd layer output")
td_output = get_td_layer_output([x])[0]

W, b, u = loaded_model.layers[4].get_weights()


def get_attn_weights(x, W, b, u):
    uit = dot_product(x, W)
    uit += b
    uit = K.tanh(uit)
    ait = dot_product(uit, u)
    a = K.exp(ait)
    a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
    a = K.expand_dims(a)
    return a

print("Computing attention weights")
td_output = K.variable(td_output)
W, b, u = K.variable(W), K.variable(b), K.variable(u)
att_weights = get_attn_weights(td_output, W, b, u)
att_weights = att_weights.numpy().squeeze(axis=2)

import pickle

with open('../seed_word_extraction/attn.pkl', 'wb') as f:
    pickle.dump((x, y, y_pred, att_weights, vocabulary_inv_list), f)
print("Dumped weights")