# -*- coding: utf-8 -*-


import re
import logging
import itertools
import numpy as np
import pandas as pd
from collections import Counter
import string
import torch
from sklearn.metrics import roc_auc_score
import argparse
from sklearn.model_selection import ParameterGrid
import os


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--early_stopping", default=False, action="store_true")
    return parser.parse_args()

args = init_arg()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Path does not exist, ({path}) has been created')
    else:
        print(f'({path}) already exists')


def clean_text(text):
    # Remove puncuation
    text = text.translate(string.punctuation)

    # Convert words to lower case and split them
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\!", "!", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\+", "+", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\-", "-", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"\:", ":", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text

def load_embeddings(vocabulary):
    word_embeddings = {}
    for word in vocabulary:
        word_embeddings[word] = np.random.uniform(-0.25, 0.25, 300)
    word_embeddings["<PAD/>"] = np.zeros((300,)) # zero-padding
    return word_embeddings


def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=None):
    """Pad setences during training or prediction"""
    if forced_sequence_length is None: # Train
        sequence_length = max(len(x) for x in sentences)
    else: # Prediction
        logging.critical('This is prediction, reading the trained sequence length')
        sequence_length = forced_sequence_length
    logging.critical('The maximum length is {}'.format(sequence_length))

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)

        if num_padding < 0: # Prediction: cut off the sentence if it is longer than the sequence length
            logging.info('This sentence has to be cut off because it is longer than trained sequence length')
            padded_sentence = sentence[0:sequence_length]
        else:
            padded_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(padded_sentence)
    return padded_sentences
    

def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv

def load_data(data):
    data = pd.DataFrame(data)

    x_raw = data['text'].apply(lambda x: clean_text(x).split(' ')).tolist()
    y_raw_toxic = data['y']
    x_raw = pad_sentences(x_raw) 
    vocabulary, vocabulary_inv = build_vocab(x_raw)
    
    x = np.array([[vocabulary[word] for word in sentence] for sentence in x_raw])
    y = np.array(y_raw_toxic)

    return x, y, vocabulary, vocabulary_inv, data, x_raw


class text_cnn(torch.nn.Module):
    def __init__(self, word_vec_length, sentence_max_length, num_kernels):
        super().__init__()
        
        self.word_vec_length = word_vec_length
        self.sentence_max_length = sentence_max_length
        self.num_kernels = num_kernels
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=self.num_kernels,
                            kernel_size=(3, self.word_vec_length)),
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d((self.sentence_max_length - 3 + 1, 1))
            )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=self.num_kernels,
                            kernel_size=(4, self.word_vec_length)),
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d((self.sentence_max_length - 4 + 1, 1))
            )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=self.num_kernels,
                            kernel_size=(5, self.word_vec_length)),
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d((self.sentence_max_length - 5 + 1, 1))
            )
        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(self.num_kernels*3, 1), 
            torch.nn.Sigmoid()
            )

    def forward(self, x):
        # x.shape == (batch_size, n_channels, n_rows, n_cols)
        x1 = self.conv1(x) # x1.shape == (batch_size, n_channels, 1, 1)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1).reshape(x.shape[0], self.num_kernels*3)
        x = self.linear1(x)
        return x



class EarlyStopping:
    def __init__(self, patience=10, delta=0, checkpoint_path='.'):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.min_valid_loss = np.inf
        self.delta = delta
        self.path = checkpoint_path
        mkdir(self.path)

    def __call__(self, valid_loss, model):
        if valid_loss < self.min_valid_loss + self.delta:
            self.min_valid_loss = valid_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        '''Save model when validation loss decrease.'''
        torch.save(model, f'{self.path}/early_stopping_model.pt')

dataset = 'skcm'
sentence_df = pd.read_excel('sentence.xlsx')
sentence_df.rename(columns={'sentence': 'text', 'relation': 'y'}, inplace=True)

index_ls = list(range(sentence_df.shape[0]))

np.random.seed(42)
index_ls = np.random.choice(index_ls, sentence_df.shape[0], replace=False) # shuffle


RESULT_PATH = f'{dataset}_result'
mkdir(f'{RESULT_PATH}/training process')


x_, y_, vocabulary, vocabulary_inv, df, x_raw = load_data(sentence_df)
word_embeddings = load_embeddings(vocabulary)
vector_mat = [np.concatenate([word_embeddings[word][None, :] for word in sentence], axis=0) for sentence in x_raw]

vector_mat = np.array(vector_mat)
vector_mat = vector_mat[index_ls]

y_ = np.array(y_)[:, None]
y_ = y_[index_ls]


x_total = torch.tensor(vector_mat).unsqueeze(1).to(torch.float)
y_total = torch.tensor(y_).float()


torch.set_num_threads(32)
input_shape = (300, vector_mat.shape[1])


n_total = len(y_)
n_train_base = 200
n_test = 150

x_test = torch.tensor(vector_mat[n_train_base:n_train_base+n_test]).unsqueeze(1).to(torch.float)
y_test = torch.tensor(y_[n_train_base:n_train_base+n_test]).float()


x_train_base = x_total[:n_train_base,]
x_unlabeled = x_total[n_train_base+n_test:,]

y_train_base = y_total[:n_train_base,]
y_unlabeled = y_total[n_train_base+n_test:,]

original_unlabeled_index = index_ls[n_train_base+n_test:]

param_range = {'lr': [1e-2, 1e-3, 1e-4], 
               'epoch': [300, 400, 500], 
               'batch_size': [50]}
if args.early_stopping is True:
    param_range['epoch'] = [1000]
param_grid = ParameterGrid(param_range)

loss_fn = torch.nn.BCELoss()

n_add = 100

    

"""
active learning
"""

auc_al_ls = []
sen_al_ls = []
spe_al_ls = []
gmeans_al_ls = []


x_train = x_train_base
y_train = y_train_base


n_iter = 1
hyperparam_ls = [] # record best hyperparameter in each active learning iteration
for iter_count in range(n_iter):
    print(iter_count, flush=True)
    model_ls = []
    for index, hyperparam in enumerate(param_grid):
        torch.manual_seed(1)
        model = text_cnn(*input_shape, 100)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=hyperparam['lr'])
        if args.early_stopping is True:
            early_stopper = EarlyStopping(checkpoint_path=RESULT_PATH)
        
        loss_train_ls = [] # record train loss in each epoch
        loss_valid_ls = [] # record valid loss in each epoch
        for epoch in range(hyperparam['epoch']):
            # print(f"{epoch}")
            
            # shuffle the dataset
            permutation = torch.randperm(x_train.shape[0])
            
            loss_train = 0. # record training loss in one epoch
            for i in range(0, x_train.shape[0], hyperparam['batch_size']):
                indices = permutation[i:i + hyperparam['batch_size']]
                batch_x, batch_y = x_train[indices], y_train[indices]
                
                model.train()
                y_train_hat = model(batch_x) 
                loss_batch = loss_fn(y_train_hat, batch_y)
                
                optimizer.zero_grad()
                loss_batch.backward() 
                optimizer.step()

        
                loss_train += loss_batch.item() * len(indices)
                
            # calc loss on validation set
            with torch.no_grad():
                y_valid_hat = model(x_test)
                loss_valid = loss_fn(y_valid_hat, y_test).item()
                
            loss_train_ls.append(loss_train/x_train.shape[0])
            loss_valid_ls.append(loss_valid)
            
            if args.early_stopping is True:
                early_stopper(loss_valid, model)
                if early_stopper.early_stop:
                    # print("Early stopping")
                    loss_valid = early_stopper.min_valid_loss
                    model = torch.load(f'{RESULT_PATH}/early_stopping_model.pt') # the path need to be changed if the path used in class EarlyStopping changed
                    break
        
        model_ls.append([loss_valid, model, hyperparam, epoch, loss_train_ls, loss_valid_ls])
        
    best_model = min(model_ls, key=lambda x: x[0])
    model = best_model[1]
    torch.save(best_model[1], f'{RESULT_PATH}/model_iter={iter_count}.pt')
    hyperparam_ls.append({**best_model[2], 'epoch_used': best_model[3], 'loss_valid': best_model[0]})
    
    hyperparam_df = pd.DataFrame(hyperparam_ls)
    hyperparam_df.to_excel(f'{RESULT_PATH}/hyperparam_df_{dataset}.xlsx', index=False)
    
    
    loss_df = pd.DataFrame({'loss_train': best_model[4], 
                            'loss_valid': best_model[5]})
    loss_df.to_excel(f'{RESULT_PATH}/training process/loss_iteration={iter_count}.xlsx', index=False)
    
    
    model.eval()
    with torch.no_grad():
        y_test_hat = model(x_test)
        threshold = 0.5
        y_test_class = torch.where(y_test_hat >= threshold, torch.tensor(1.), torch.tensor(0.))
    
    
    TP = sum((y_test_class == 1.) & (y_test == 1.)).to(torch.float)
    FP = sum((y_test_class == 1.) & (y_test == 0.)).to(torch.float)
    TN = sum((y_test_class == 0.) & (y_test == 0.)).to(torch.float)
    FN = sum((y_test_class == 0.) & (y_test == 1.)).to(torch.float)
    auc_al_ls.append(roc_auc_score(y_test, y_test_hat))
    sen_al_ls.append((TP/(TP+FN)).item())
    spe_al_ls.append((TN/(TN+FP)).item())
    gmeans_al_ls.append(torch.sqrt((TP/(TP+FN)) * (TN/(TN+FP))).item())



    # query, update labeled and unlabeled data set
    model.eval()
    with torch.no_grad():
        y_unlabeled_hat = model(x_unlabeled)
        add_index = torch.argsort(torch.abs(y_unlabeled_hat - threshold).squeeze())[:n_add]
            
        original_add_index = [original_unlabeled_index[i] for i in add_index.detach().numpy().tolist()]
        print(original_add_index)
        original_unlabeled_index = [original_unlabeled_index[i] for i in range(len(original_unlabeled_index)) if i not in add_index.detach().numpy().tolist()]
        
        x_train = torch.cat([x_train, x_unlabeled[add_index,]])
        y_train = torch.cat([y_train, y_unlabeled[add_index,]])
        
        x_unlabeled = x_unlabeled[[i for i in range(x_unlabeled.shape[0]) if i not in add_index],]
        y_unlabeled = y_unlabeled[[i for i in range(y_unlabeled.shape[0]) if i not in add_index],]


    al_df = pd.DataFrame(zip(auc_al_ls, sen_al_ls, spe_al_ls, gmeans_al_ls), 
                         columns=['auc', 'sen', 'spe', 'gmeans'])
    
    al_df.to_excel(f'{RESULT_PATH}/al_metrics_{dataset}.xlsx', index=False)
    
    