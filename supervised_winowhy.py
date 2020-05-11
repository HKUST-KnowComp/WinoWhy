import ujson as json
import torch

import logging
import argparse
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
import math
import numpy
import collections
import random
from tqdm import tqdm

from transformers import *
import pandas as pd
import numpy as np

from sklearn.utils import shuffle

def output_five_folds(cls_df, method, shuffle,usage):
    """
    method in ["r","wsc"]
    shuffle in [True, False]
    usage in ["wnli", "cls"]
    
    """
    seed = 2019
    info = method + "_" + str(shuffle) + "_" + usage +"_" + str(seed)
    
    five_folds = []
    
    if not shuffle:
        working_df = cls_df
    elif shuffle:
        working_df = shuffle(cls_df,random_state=seed)
    
    if method == "r":
            raw_five_folds = np.array_split(working_df,5)
    elif method == "wsc":
        raw_five_folds = []
        for i in range(5):
             raw_five_folds.append(cls_df.loc[cls_df['fold_num'] == i])           
    
    for fold in raw_five_folds:
        if usage == "cls":
            five_folds.append(fold[["sentence","label"]])
        elif usage == "wnli":
            five_folds.append(fold[["wnli_sent1","wnli_sent2","label"]])

    print("working with classification data with examples:", cls_df.shape[0])

    return five_folds

def pick_training_and_testing_folds(five_folds,fold_num):

    train_folds = []

    for i in range(len(five_folds)):
        if i == fold_num:
            test_fold = five_folds[i]
        else:
            train_folds.append(five_folds[i])

    train_fold = pd.concat(train_folds)

    return test_fold, train_fold

class DataLoader:
    def __init__(self, data_path, args):
        self.args = args
        with open(data_path, 'r') as f:
            self.cls_df = pd.read_csv(f)

        self.five_folds = output_five_folds(self.cls_df, method=args.method, shuffle=False, usage="wnli")

        self.test_df,self.train_df = pick_training_and_testing_folds(self.five_folds, args.fold)

        self.train_set = self.tensorize_example(self.train_df)
        print('successfully loaded %d examples for training data' % len(self.train_set))

        self.test_set = self.tensorize_example(self.test_df)
        print('successfully loaded %d examples for test data' % len(self.test_set))

    def load_embedding_dict(self, path):
        print("Loading word embeddings from {}...".format(path))
        default_embedding = numpy.zeros(300)
        embedding_dict = collections.defaultdict(lambda: default_embedding)
        if len(path) > 0:
            vocab_size = None
            with open(path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    word_end = line.find(" ")
                    word = line[:word_end]
                    embedding = numpy.fromstring(line[word_end + 1:], numpy.float32, sep=" ")
                    assert len(embedding) == 300
                    embedding_dict[word] = embedding
            if vocab_size is not None:
                assert vocab_size == len(embedding_dict)
            print("Done loading word embeddings.")
        return embedding_dict

    def tensorize_example(self, initial_dataframe):
        tensorized_dataset = list()
        
        for i in range(initial_dataframe.shape[0]):

            tensorized_examples_for_one_frame = list()

            sent1 = initial_dataframe.iloc[i]["wnli_sent1"]
            sent2 = initial_dataframe.iloc[i]["wnli_sent2"]
            label = initial_dataframe.iloc[i]["label"]

            lm_tokenized_sent1 = tokenizer.encode(sent1)
            lm_tokenized_sent2 = tokenizer.encode(sent2)
            bert_tokenized_sent1 = tokenizer.encode('[CLS] ' + sent1 + ' . [SEP]')
            bert_tokenized_sent2 = tokenizer.encode('[CLS] ' + sent2 + ' . [SEP]')

            tensorized_examples_for_one_frame.append(
                {'gpt2_sent1':torch.tensor(lm_tokenized_sent1).to(device),
                    'gpt2_sent2': torch.tensor(lm_tokenized_sent2).to(device),
                    'bert_sent1':torch.tensor(bert_tokenized_sent1).to(device),
                    'bert_sent2': torch.tensor(bert_tokenized_sent2).to(device),
                    'label': torch.tensor([int(label)]).to(device)
                    })

            tensorized_dataset += tensorized_examples_for_one_frame

        return tensorized_dataset


class LSTM(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(LSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_size
        self.lstm = torch.nn.LSTM(self.embedding_size, self.hidden_dim)
        self.hidden = self.init_hidden()
        self.dropout = torch.nn.Dropout(0.5)
        self.second_last_layer = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.last_layer = torch.nn.Linear(self.hidden_dim, 2)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sent1, sent2):

        sent1_representation, _ = self.lstm(sent1.unsqueeze(0))
        sent1_representation = torch.mean(sent1_representation, dim=1)
        sent2_representation, _ = self.lstm(sent2.unsqueeze(0))
        sent2_representation = torch.mean(sent2_representation, dim=1)
        overall_representation = torch.cat([sent1_representation, sent2_representation], dim=1)
        overall_representation = self.dropout(overall_representation)
        prediction = self.last_layer(self.second_last_layer(overall_representation))
        return prediction


class GPTCausal(GPT2Model):
    def __init__(self, config):
        super(GPT2Model, self).__init__(config)
        self.lm = GPT2Model(config)
        self.embedding_size = 300
        self.hidden_dim = 200

        self.dropout = torch.nn.Dropout(0.5)
        self.second_last_layer = torch.nn.Linear(768*2, self.hidden_dim)
        self.last_layer = torch.nn.Linear(self.hidden_dim, 2)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sent1, sent2):

        sent1_representation = self.lm(sent1.unsqueeze(0))
        sent2_representation = self.lm(sent2.unsqueeze(0))

        overall_representation = torch.cat(
            [torch.mean(sent1_representation[0].squeeze(), dim=0).unsqueeze(0),
             torch.mean(sent2_representation[0].squeeze(), dim=0).unsqueeze(0)], dim=1)

        prediction = self.last_layer(self.second_last_layer(overall_representation))
        return prediction

class GPTLarge(GPT2Model):
    def __init__(self, config):
        super(GPT2Model, self).__init__(config)
        self.lm = GPT2Model(config)
        self.embedding_size = 300
        self.hidden_dim = 200

        self.dropout = torch.nn.Dropout(0.5)
        self.second_last_layer = torch.nn.Linear(2560, self.hidden_dim)
        self.last_layer = torch.nn.Linear(self.hidden_dim, 2)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sent1, sent2):

        sent1_representation = self.lm(sent1.unsqueeze(0))
        sent2_representation = self.lm(sent2.unsqueeze(0))

        overall_representation = torch.cat(
            [torch.mean(sent1_representation[0].squeeze(), dim=0).unsqueeze(0),
             torch.mean(sent2_representation[0].squeeze(), dim=0).unsqueeze(0)], dim=1)
  
        prediction = self.last_layer(self.second_last_layer(overall_representation))
        return prediction

class Roberta(RobertaModel):
    def __init__(self, config):
        super(RobertaModel, self).__init__(config)
        self.lm = RobertaModel(config)
        self.embedding_size = 300
        self.hidden_dim = 200

        self.dropout = torch.nn.Dropout(0.5)
        self.second_last_layer = torch.nn.Linear(1536, self.hidden_dim)
        self.last_layer = torch.nn.Linear(self.hidden_dim, 2)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sent1, sent2):

        sent1_representation = self.lm(sent1.unsqueeze(0))
        sent2_representation = self.lm(sent2.unsqueeze(0))
        overall_representation = torch.cat(
            [torch.mean(sent1_representation[0].squeeze(), dim=0).unsqueeze(0),
             torch.mean(sent2_representation[0].squeeze(), dim=0).unsqueeze(0)], dim=1)

        prediction = self.last_layer(self.second_last_layer(overall_representation))
        return prediction

class RobertaLarge(RobertaModel):
    def __init__(self, config):
        super(RobertaModel, self).__init__(config)
        self.lm = RobertaModel(config)
        self.embedding_size = 300
        self.hidden_dim = 200

        self.dropout = torch.nn.Dropout(0.5)
        self.second_last_layer = torch.nn.Linear(2048, self.hidden_dim)
        self.last_layer = torch.nn.Linear(self.hidden_dim, 2)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sent1, sent2):

        sent1_representation = self.lm(sent1.unsqueeze(0))
        sent2_representation = self.lm(sent2.unsqueeze(0))

        overall_representation = torch.cat(
            [torch.mean(sent1_representation[0].squeeze(), dim=0).unsqueeze(0),
             torch.mean(sent2_representation[0].squeeze(), dim=0).unsqueeze(0)], dim=1)

        prediction = self.last_layer(self.second_last_layer(overall_representation))
        return prediction



def train(model, data):
    all_loss = 0
    print('training:')
    random.shuffle(data)
    model.train()
    selected_data = data
    for tmp_example in tqdm(selected_data):
        final_prediction = model(sent1=tmp_example['gpt2_sent1'], sent2=tmp_example['gpt2_sent2']) 
        loss = loss_func(final_prediction, tmp_example['label'])
        test_optimizer.zero_grad()
        loss.backward()
        test_optimizer.step()
        all_loss += loss.item()
    print('current loss:', all_loss / len(data))


def test(model, data):
    correct_count = 0
    # print('Testing:')
    model.eval()
    for tmp_example in tqdm(data):
        final_prediction = model(sent1=tmp_example['gpt2_sent1'], sent2=tmp_example['gpt2_sent2'])
        if tmp_example['label'].data[0] == 1:
            # current example is positive
            if final_prediction.data[0][1] >= final_prediction.data[0][0]:
                correct_count += 1
        else:
            # current example is negative
            if final_prediction.data[0][1] <= final_prediction.data[0][0]:
                correct_count += 1

    return correct_count / len(data)


parser = argparse.ArgumentParser()

## parameters
parser.add_argument("--gpu", default='0', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--model", default='gpt2', type=str, required=False,
                    help="choose the model to test")
parser.add_argument("--lr", default=0.0001, type=float, required=False,
                    help="initial learning rate")
parser.add_argument("--lrdecay", default=0.8, type=float, required=False,
                    help="learning rate decay every 5 epochs")
parser.add_argument("--method", default="r", type=str, required=False,
                    help="wsc or r, two kinds of settings")
parser.add_argument("--fold", default=0, type=int, required=False,
                    help="testing which fold")
parser.add_argument("--max_len", default=128, type=int, required=False,
                    help="number of words")
parser.add_argument("--epochs", default=15, type=int, required=False,
                    help="number of epochs")
parser.add_argument("--model_weight", default='gpt2', type=str, required=False,
                    help="gpt2, gpt2-large, roberta-base, roberta-large")

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('current device:', device)
n_gpu = torch.cuda.device_count()
print('number of gpu:', n_gpu)
torch.cuda.get_device_name(0)

if args.model == "gpt2":
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_weight)
    if args.model_weight == 'gpt2':
        current_model = GPTCausal.from_pretrained(args.model_weight)
    elif args.model_weight == 'gpt2-large':
        current_model = GPTLarge.from_pretrained(args.model_weight)

elif args.model == "roberta":
    tokenizer = RobertaTokenizer.from_pretrained(args.model_weight)

    if args.model_weight == 'roberta-base':
        current_model = Roberta.from_pretrained(args.model_weight)
    elif args.model_weight == 'roberta-large':
        current_model = RobertaLarge.from_pretrained(args.model_weight)

current_model.to(device)
test_optimizer = torch.optim.SGD(current_model.parameters(), lr=args.lr)
loss_func = torch.nn.CrossEntropyLoss()


all_data = DataLoader('./dataset/dataset.csv', args)

best_dev_performance = 0
final_performance = 0
accuracy_by_type = dict()

for i in range(args.epochs):
    print('Iteration:', i + 1, '|', 'Current best performance:', final_performance)
    train(current_model, all_data.train_set)
    test_performance = test(current_model, all_data.test_set)
    print('Test accuracy:', test_performance)
    if test_performance >= best_dev_performance:
        print('New best performance!!!')
        best_dev_performance = test_performance
        final_performance = test_performance

print("Best performance:", final_performance)

print('end')
