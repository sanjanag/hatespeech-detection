import datetime
import time

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset, random_split, \
    RandomSampler, SequentialSampler


def save_checkpoint(save_path, model):
    if save_path == None:
        return
    state_dict = {'model_state_dict': model.state_dict()}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def evaluate(pred_prob, labels):
    pred_flat = np.argmax(pred_prob, axis=1).flatten()
    labels_flat = labels.flatten()
    print(classification_report(labels_flat, pred_flat))
    return flatten_report(classification_report(labels_flat, pred_flat,
                                                output_dict=True))


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_dataset(encoded_data):
    dataset = TensorDataset(encoded_data['input_ids'],
                            encoded_data['attention_mask'],
                            encoded_data['token_type_ids'],
                            encoded_data['labels'],
                            encoded_data['pseudolabels'])
    return dataset


def get_dataloader(dataset, is_train=False):
    if is_train:
        dataloader = DataLoader(dataset,
                                sampler=RandomSampler(dataset),
                                batch_size=32)
    else:
        dataloader = DataLoader(dataset,
                                sampler=SequentialSampler(
                                    dataset),
                                batch_size=256)
    return dataloader


def transfer_input_to_device(batch, device):
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    token_type_ids = batch[2].to(device)
    labels = batch[3].to(device)
    pseudolabels = batch[4].to(device)
    return input_ids, attention_mask, token_type_ids, labels, pseudolabels


def predict(model, dataloader, device):
    model.eval()
    all_prob = []
    all_labels = []
    all_pseudolabels = []
    for step, batch in enumerate(dataloader):
        if step % 40 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(dataloader)))
        input_ids, attention_mask, token_type_ids, labels, pseudolabels = \
            transfer_input_to_device(batch, device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        logits = outputs[0]
        pred_prob = torch.nn.functional.softmax(logits, dim=1)
        all_prob.append(pred_prob.detach().cpu().numpy())
        all_labels.append(labels.to('cpu').numpy())
        all_pseudolabels.append(pseudolabels.to('cpu').numpy())
    prob = np.concatenate(all_prob, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    pseudolabels = np.concatenate(all_pseudolabels, axis=0)
    return prob, labels, pseudolabels


def train(model, optimizer, device, train_dataloader, validation_dataloader,
          epochs=3, num_iters=None):
    loss_values = []
    iter = 0
    prob, labels, pseudolabels = None, None, None
    for epoch in range(epochs):
        if num_iters is not None and iter == num_iters:
            break
        model.train()
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        total_train_loss = 0.0
        t0 = time.time()
        all_prob = []
        all_labels = []
        all_pseudolabels = []
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                    step, len(train_dataloader), elapsed))
            input_ids, attention_mask, token_type_ids, labels, pseudolabels = \
                transfer_input_to_device(batch, device)

            loss, logits = model(input_ids=input_ids,
                                 token_type_ids=token_type_ids,
                                 attention_mask=attention_mask,
                                 labels=pseudolabels)

            pred_prob = torch.nn.functional.softmax(logits, dim=1)

            all_prob.append(pred_prob.detach().cpu().numpy())
            all_pseudolabels.append(pseudolabels.to('cpu').numpy())
            all_labels.append(labels.to('cpu').numpy())

            total_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter += 1
            if num_iters is not None and iter == num_iters:
                break
        avg_train_loss = total_train_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        prob = np.concatenate(all_prob, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        pseudolabels = np.concatenate(all_pseudolabels, axis=0)
    return prob, labels, pseudolabels
        # logits, labels, pseudolabels = predict(model, validation_dataloader, device)
        # print("Validation metrics")
        # evaluate(logits, pseudolabels)

def append_data(encoded_data, input_ids, attention_mask, token_type_ids,
                labels, pseudolabels):
    encoded_data['input_ids'].append(input_ids)
    encoded_data['attention_mask'].append(attention_mask)
    encoded_data['token_type_ids'].append(token_type_ids)
    encoded_data['labels'].append(labels)
    encoded_data['pseudolabels'].append(pseudolabels)
    return encoded_data

def concatenate(encoded_data):
    encoded_data['input_ids'] = torch.stack(encoded_data['input_ids'], dim=0)
    encoded_data['attention_mask'] = torch.stack(
        encoded_data['attention_mask'], dim=0)
    encoded_data['token_type_ids'] = torch.stack(
        encoded_data['token_type_ids'], dim=0)
    encoded_data['labels'] = torch.stack(encoded_data['labels'], dim=0)
    encoded_data['pseudolabels'] = torch.stack(encoded_data['pseudolabels'],
                                            dim=0)
    return encoded_data

def flatten_report(report):
    metrics = {'accuracy': round(report['accuracy'], 2),
               'f1-macro': round(report['macro avg']['f1-score'], 2),
               'p-macro': round(report['macro avg']['precision'], 2),
               'r-macro': round(report['macro avg']['recall'], 2),
               'f1-weighted': round(report['weighted avg']['f1-score'], 2),
               'p-weighted': round(report['weighted avg']['precision'], 2),
               'r-weighted': round(report['weighted avg']['recall'], 2),
               'p-0': round(report['0']['precision'], 2),
               'r-0': round(report['0']['recall'], 2),
               'p-1': round(report['0']['precision'], 2),
               'r-1': round(report['0']['recall'], 2),
               'p-2': round(report['0']['precision'], 2),
               'r-2': round(report['0']['recall'], 2),
               'p-3': round(report['0']['precision'], 2),
               'r-3': round(report['0']['recall'], 2)
               }
    return metrics