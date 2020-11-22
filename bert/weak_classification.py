import sys

import numpy as np
import torch

sys.path.insert(0, '../')
from bert.utils.data_utils import get_processed_data, get_sup_docs, \
    combine_data
from transformers import BertTokenizer, BertForSequenceClassification
import torch.optim as optim
from torch.utils.data import random_split
from bert.utils.model_utils import train, get_dataset, get_dataloader, \
    predict, append_data, concatenate, evaluate, save_checkpoint
import pandas as pd

np.random.seed(1)
############################################
# Read Data
############################################
data, y = get_processed_data()
print("Finished reading data")

############################################
# Set Device
############################################

device = f"cuda:{sys.argv[1]}" if torch.cuda.is_available() else 'cpu'
print(f"Using device {device}")
threshold = float(sys.argv[2])
epochs = int(sys.argv[3])

sup_idx = get_sup_docs(data, y)


def flatten(sup_idx):
    ret_list = []
    for indices in sup_idx:
        ret_list.extend(indices)
    return ret_list


sup_idx = flatten(sup_idx)
############################################
# Bert Tokenizer
############################################

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MAX_SEQ_LEN = 256
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

encoded_inputs = tokenizer(list(data), padding=True, truncation=True,
                           return_tensors="pt")
print("Encoded inputs using Tokenizer")

results_labelled = []
results_labelled_pseudo = []
results_unlabelled = []
results_overall = []

#############################################
# Labelled and Unlabelled sets
############################################
labelled_data = {
    'input_ids': encoded_inputs['input_ids'][sup_idx],
    'attention_mask': encoded_inputs['attention_mask'][sup_idx],
    'token_type_ids': encoded_inputs['token_type_ids'][sup_idx],
    'labels': torch.LongTensor(y[sup_idx]),
    'pseudolabels': torch.LongTensor(y[sup_idx])
}

mask = np.ones((len(encoded_inputs['input_ids'])), dtype=bool)
mask[sup_idx] = False

unlabelled_data = {
    'input_ids': encoded_inputs['input_ids'][mask],
    'attention_mask': encoded_inputs['attention_mask'][mask],
    'token_type_ids': encoded_inputs['token_type_ids'][mask],
    'labels': torch.LongTensor(y[mask]),
    'pseudolabels': torch.LongTensor(y[mask])
}
total = len(labelled_data['input_ids']) + len(unlabelled_data['input_ids'])
print("Split into labelled and unlabelled data")

############################################
# self training
############################################
iter = 0
while len(unlabelled_data['input_ids']) > 0:
    print('======== Iteration {:} ========'.format(iter))

    ############################################
    # Retraining
    ############################################
    print("Training.............")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=4,
        output_attentions=False,
        output_hidden_states=False)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    labelled_dataset = get_dataset(labelled_data)
    train_size = int(0.8 * len(labelled_dataset))
    val_size = len(labelled_dataset) - train_size
    train_dataset, val_dataset = random_split(labelled_dataset, [train_size,
                                                                 val_size])

    train_dataloader = get_dataloader(train_dataset, True)
    validation_dataloader = get_dataloader(val_dataset)

    pred_prob, labels, pseudolabels = train(model, optimizer, device,
                                            train_dataloader,
                                            validation_dataloader,
                                            epochs=epochs)
    print("Labelled Metrics on pseudolabels")
    results_labelled_pseudo.append(evaluate(pred_prob, pseudolabels))

    print("Labelled Metrics on actual labels")
    results_labelled.append(evaluate(pred_prob, labels))

    save_checkpoint(f"./weak_chkpts_{threshold}/{iter}.pt", model)

    ############################################
    # evaluate on whole dataset
    ############################################

    print("Evaluating model on whole dataset.............")
    data = combine_data(labelled_data, unlabelled_data)
    dataset = get_dataset(data)
    dataloader = get_dataloader(dataset)

    pred_prob, labels, pseudolabels = predict(model, dataloader, device)
    pred_labels = np.argmax(pred_prob, axis=1).flatten()

    print("Overall metrics")
    results_overall.append(evaluate(pred_prob, labels))

    ############################################
    # predict
    ############################################
    print("Labelling unlabelled data...........")
    unlabelled_dataset = get_dataset(unlabelled_data)
    dataloader = get_dataloader(unlabelled_dataset)

    pred_prob, labels, pseudolabels = predict(model, dataloader, device)
    pred_labels = np.argmax(pred_prob, axis=1).flatten()
    unlabelled_data['pseudolabels'] = torch.LongTensor(pred_labels)

    print("Unlabelled data metrics")
    results_unlabelled.append(evaluate(pred_prob, labels))

    ############################################
    # split dataset
    ############################################
    print("Splitting dataset.............")
    confident_examples = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': [],
        'labels': [],
        'pseudolabels': []
    }
    nonconfident_examples = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': [],
        'labels': [],
        'pseudolabels': []
    }

    for i in range(len(unlabelled_data['input_ids'])):
        if pred_prob[i][pred_labels[i]] >= threshold:
            confident_examples = append_data(confident_examples,
                                             unlabelled_data['input_ids'][i],
                                             unlabelled_data['attention_mask'][
                                                 i],
                                             unlabelled_data['token_type_ids'][
                                                 i],
                                             unlabelled_data['labels'][i],
                                             unlabelled_data['pseudolabels'][
                                                 i])
        else:
            nonconfident_examples = append_data(nonconfident_examples,
                                                unlabelled_data['input_ids'][
                                                    i],
                                                unlabelled_data[
                                                    'attention_mask'][i],
                                                unlabelled_data[
                                                    'token_type_ids'][i],
                                                unlabelled_data['labels'][i],
                                                unlabelled_data[
                                                    'pseudolabels'][i])

    print(f"Total: {len(unlabelled_data['input_ids'])}, confident: "
          f"{len(confident_examples['input_ids'])}, "
          f"nonconfident:"
          f" {len(nonconfident_examples['input_ids'])}")
    if len(unlabelled_data['input_ids']) != 0:
        unlabelled_data = concatenate(nonconfident_examples)
    else:
        unlabelled_data = nonconfident_examples
    confident_examples = concatenate(confident_examples)
    labelled_data = combine_data(labelled_data, confident_examples)

    assert len(labelled_data['input_ids']) + len(unlabelled_data[
                                                     'input_ids']) == total
    iter += 1

results_labelled = pd.DataFrame(results_labelled)
results_labelled_pseudo = pd.DataFrame(results_labelled_pseudo)
results_unlabelled = pd.DataFrame(results_unlabelled)
results_overall = pd.DataFrame(results_overall)

results_unlabelled.to_csv(f"metrics_{threshold}/unlabelled.csv")
results_labelled.to_csv(f"metrics_{threshold}/labelled.csv")
results_labelled_pseudo.to_csv(f"metrics_{threshold}/labelled_pseudo.csv")
results_overall.to_csv(f"metrics_{threshold}/overall.csv")
