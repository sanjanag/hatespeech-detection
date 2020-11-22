import sys

import torch
from transformers import BertTokenizer, BertForSequenceClassification

sys.path.insert(0, '../')
import torch.optim as optim
from bert.utils.data_utils import get_processed_data
from bert.utils.model_utils import save_checkpoint, evaluate, format_time
from torch.utils.data import DataLoader, TensorDataset, random_split, \
    RandomSampler, SequentialSampler
import time


import numpy as np

############################################
# Read Data
############################################
data, y = get_processed_data()

############################################
# Set Device
############################################

device = "cuda:0" if torch.cuda.is_available() else 'cpu'
print(f"Using device {device}")

############################################
# Bert Tokenizer
############################################

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MAX_SEQ_LEN = 256
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

encoded_inputs = tokenizer(data, padding=True, truncation=True,
                           return_tensors="pt")

############################################
# Train Test Split
############################################
dataset = TensorDataset(encoded_inputs['input_ids'],
                        encoded_inputs['attention_mask'],
                        encoded_inputs['token_type_ids'],
                        torch.Tensor(y).type(torch.LongTensor))
train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))
test_size = len(data) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size,
                                                                  val_size,
                                                                  test_size])

batch_size = 32

train_dataloader = DataLoader(train_dataset,
                              sampler=RandomSampler(train_dataset),
                              batch_size=batch_size)
validation_dataloader = DataLoader(val_dataset,
                                   sampler=SequentialSampler(val_dataset),
                                   batch_size=batch_size)

############################################
# Training
############################################

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=4,  # The number of output labels--2 for binary classification.
    # You can increase this for multi-class tasks.
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=False,  # Whether the model returns all hidden-states.
)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=2e-5)

epochs = 4
total_steps = len(train_dataloader) * epochs
model.train()
loss_values = []
for epoch in range(epochs):
    model.train()
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    total_train_loss = 0.0
    t0 = time.time()
    all_logits = []
    all_labels = []
    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                step, len(train_dataloader), elapsed))

        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        labels = batch[3].to(device)

        loss, logits = model(input_ids=input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask,
                             labels=labels)
        logits = logits.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()
        all_logits.append(logits)
        all_labels.append(labels)

        total_train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_train_loss = total_train_loss / len(train_dataloader)
    loss_values.append(avg_train_loss)
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print("Training metrics")
    evaluate(all_logits, all_labels)

    model.eval()
    all_logits = []
    all_labels = []
    for step, batch in enumerate(validation_dataloader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        labels = batch[3].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()
        all_logits.append(logits)
        all_labels.append(labels)
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print("Validation metrics")
    evaluate(all_logits, all_labels)

    save_checkpoint(f"./model_chkpts/{epoch}.pt", model)
