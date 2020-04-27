from keras.models import model_from_json

from load_data import load_dataset
from model import AttentionWithContext, dot_product

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# load weights into new model
loaded_model = model_from_json(loaded_model_json, custom_objects={
    'AttentionWithContext': AttentionWithContext})
print("Loaded model from disk")

print(loaded_model.layers)

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
td_output = get_td_layer_output([x[:1]])[0]

# print(att_output.shape)
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


td_output = K.variable(td_output)
W, b, u = K.variable(W), K.variable(b), K.variable(u)
att_weights = get_attn_weights(td_output, W, b, u)
att_weights = att_weights.numpy()

print(att_weights.shape)
