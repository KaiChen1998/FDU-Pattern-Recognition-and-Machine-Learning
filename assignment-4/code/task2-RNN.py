
# coding: utf-8

# In[1]:


import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time


# In[2]:


import fastNLP
from fastNLP.io.dataset_loader import CSVLoader
from fastNLP import Batch
from fastNLP import Vocabulary
from fastNLP import RandomSampler, SequentialSampler
from fastNLP.io.embed_loader import EmbedLoader


# ## Part1. Text classification using RNN with random word embedding
# ### 1. Hyperparameter

# In[24]:


batch_size = 64
learning_rate = 0.0005
num_epoch = 5
hidden_size = 300
dropout_rate = 0.5
bidirectional = 1
use_pretrain = 1
freeze_pretrain = 1
embed_path = 'data/glove.6B.300d.txt'
embedding_size = 300

loss_history = []
load_address = None
use_cuda = torch.cuda.is_available()
print("use_cuda: ", use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")


# ### 2. data preprocess 

# In[4]:
def get_word(x):
    if(x == ' '):
        return [' ']
    else:
        return x.lower().split()

def load_data(path, is_train = 0):
    loader = CSVLoader(sep='\t')
    dataset = loader.load(path)
    dataset.delete_field('SentenceId')
    dataset.delete_field('PhraseId')
    
    dataset.apply(lambda x: get_word(x['Phrase']), new_field_name = 'words', is_input = True)
    dataset.apply(lambda x: len(x['words']), new_field_name = "length", is_input = True)
    dataset.delete_field('Phrase')
    if(is_train):
        dataset.apply(lambda x: int(x['Sentiment']), new_field_name = "Sentiment")
        dataset.set_target('Sentiment')
    return dataset


# In[5]:


# 1. get dataset
dataset = load_data('data/train.tsv', 1)
train_dataset, val_dataset = dataset.split(0.1)
test_dataset = load_data('data/test.tsv', 0)
print("train_dataset size: ", train_dataset.get_length())
print("val_dataset size: ", val_dataset.get_length())
print("test_dataset size: ", test_dataset.get_length())


# In[6]:


# 2. get vocabulary
if(use_pretrain):
    loader = EmbedLoader()
    pre_embed, vocab = loader.load_without_vocab(embed_path, normalize = False)
    embeddig_size = pre_embed.shape[1]
else:
    vocab = Vocabulary(min_freq=2).from_dataset(dataset, field_name='words')
print("vocabulary size: ", len(vocab))


# In[7]:


# 3. change word to index
vocab.index_dataset(train_dataset, field_name='words',new_field_name='words')
vocab.index_dataset(val_dataset, field_name='words',new_field_name='words')
vocab.index_dataset(test_dataset, field_name='words',new_field_name='words')

# ### 3. Build RNN model

# In[9]:


def bi_fetch(rnn_outs, lengths):
    # rnn_out: [batch_size, seq_len, 2 * hidden_size]
    batch_size = rnn_outs.size(0)
    seq_len = rnn_outs.size(1)
    rnn_outs = rnn_outs.view(batch_size, seq_len, 2, -1) # (batch_size, seq_len, 2, hidden_size)

    fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])).to(device))
    fw_out = fw_out.view(batch_size * seq_len, -1) # 正向
    bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])).to(device))
    bw_out = bw_out.view(batch_size * seq_len, -1)

    batch_range = Variable(torch.LongTensor(range(batch_size))).to(device) * seq_len
    lengths = torch.Tensor(lengths).long().to(device)

    fw_index = batch_range + lengths - 1
    fw_out = torch.index_select(fw_out, 0, fw_index)  # (batch_size, hid)
    bw_index = batch_range
    bw_out = torch.index_select(bw_out, 0, bw_index)
    outs = torch.cat([fw_out, bw_out], dim=1)

    return outs


# In[27]:


class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        if(use_pretrain):
            self.embedding = nn.Embedding.from_pretrained(torch.Tensor(pre_embed), freeze = bool(freeze_pretrain))
        else:
            self.embedding = nn.Embedding(len(vocab), embedding_size, padding_idx = vocab['<pad>'])
        self.LSTM = nn.LSTM(embedding_size, hidden_size, batch_first=True, dropout = dropout_rate, bidirectional = bool(bidirectional))
        h = hidden_size + hidden_size * bidirectional
        self.classifier = nn.Linear(h, 5)

    def forward(self, x, lengths):
        N = x.size(0)
        x = self.embedding(x)
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        
        sorted_inputs = x.index_select(0, idx_sort)
        lengths = list(lengths[idx_sort])
        sequence = pack_padded_sequence(Variable(sorted_inputs), lengths, batch_first=True)
        
        output, (h_t, c_t) = self.LSTM(sequence, None)  # (h_0, c_0) = None
        unpacked = pad_packed_sequence(output, batch_first=True)[0]
        output = bi_fetch(unpacked, lengths)[idx_unsort]

        # output = unpacked[torch.LongTensor(range(len(out_len))), out_len - 1, :][idx_unsort]
        out = self.classifier(output)
        return out

model = RNN().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
if(load_address is not None):
    model = torch.load(load_address)


# In[11]:


print(model.named_modules)


# ### 4. Train Model

# In[12]:


def model_status(training):
    if(training):
        return 'model is in training'
    else:
        return 'model is in testing'


# In[13]:


def pack(batch_x, batch_y, is_train = 1):
    x = batch_x['words'].to(device)
    lengths = batch_x['length'].to(device)
    if(is_train):
        y = batch_y['Sentiment'].to(device)
        return x, lengths, y
    else:
        return x, lengths


# In[14]:


def predict(model, dataset):
    model.eval()
    print(model_status(model.training))
    num_correct = torch.tensor(0.0)
    num_sample = torch.tensor(0.0)
    for batch_x, batch_y in Batch(dataset, sampler = SequentialSampler(), batch_size = batch_size):
        x, lengths, y = pack(batch_x, batch_y)
        score = model(x, lengths)
        y_predict = torch.argmax(score, dim = 1)
        num_correct += torch.sum(y_predict == y)
        num_sample += x.shape[0]
    return 1.0 * num_correct / num_sample


# In[28]:


def train(model, dataset, optimizer, num_epoch = 30):
    loss_history = []
    for i in range(num_epoch):
        start = time.time()
        print("Epoch: {0} start".format(i))
        model.train()
        print(model_status(model.training))
        losses = 0
        for batch_x, batch_y in Batch(dataset, sampler = RandomSampler(), batch_size = batch_size):
            x, lengths, y = pack(batch_x, batch_y)
            optimizer.zero_grad()
            score = model(x, lengths)
            loss_fn = nn.CrossEntropyLoss().to(device)
            loss = loss_fn(score, y)
            loss.backward()
            losses += loss
            optimizer.step()
        end = time.time()
        print("Epoch: {0} finish".format(i))
        loss_history.append(losses)
        acc = predict(model, val_dataset)
        print("Epoch: {0}, loss: {1}, accu: {2}, time: {3}\n".format(i, losses, acc, end - start))
    return loss_history

loss_history_new = train(model, train_dataset, optimizer, num_epoch = num_epoch)


# ### 5. Get Result


# In[30]:


def get_answer(model, dataset):
    answer = []
    print("start to generate result")
    model.eval()
    print(model_status(model.training))
    for batch_x, batch_y in Batch(dataset, sampler = SequentialSampler(), batch_size = batch_size):
        x, lengths = pack(batch_x, batch_y, 0)
        score = model(x, lengths)
        y_predict = torch.argmax(score, dim = 1).cpu().numpy()
        answer += list(y_predict)
    index = [a + 156061 for a in range(len(answer))]
    name = "result/RNN_pretrain" + str(use_pretrain) + "_freeze" + str(freeze_pretrain) + "_dropout" + str(dropout_rate) + "_bidirectional" + str(bidirectional) + "_batch_size" + str(batch_size) + "_lr" + str(learning_rate) + "_epoch" + str(num_epoch) +".csv"
    dataframe = pd.DataFrame({'PhraseId':index, 'Sentiment':answer})
    dataframe.to_csv(name,index=False,sep=',')
    return answer
answer = get_answer(model, test_dataset)
print('RNN text classification finished')
