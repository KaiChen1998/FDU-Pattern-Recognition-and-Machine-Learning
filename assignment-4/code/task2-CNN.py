
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
import time


# In[2]:


import fastNLP
from fastNLP.io.dataset_loader import CSVLoader
from fastNLP import Batch
from fastNLP import Vocabulary
from fastNLP import RandomSampler, SequentialSampler
from fastNLP.io.embed_loader import EmbedLoader


# ## Part1. Text classification using CNN with random word embedding
# ### 1. Hyperparameter

# In[3]:


batch_size = 64
learning_rate = 0.0001
num_epoch = 5
num_kernel = 100
dropout_rate = 0.5
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
    embedding_size = pre_embed.shape[1]
else:
    vocab = Vocabulary(min_freq=2).from_dataset(dataset, field_name='words')
print("vocabulary size: ", len(vocab))


# In[7]:


# 3. word to index
vocab.index_dataset(train_dataset, field_name='words',new_field_name='words')
vocab.index_dataset(val_dataset, field_name='words',new_field_name='words')
vocab.index_dataset(test_dataset, field_name='words',new_field_name='words')


# ### 3. Build CNN model

# In[33]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        if(use_pretrain):
            self.embedding = nn.Embedding.from_pretrained(torch.Tensor(pre_embed), freeze = bool(freeze_pretrain))
        else:
            self.embedding = nn.Embedding(len(vocab), embedding_size, padding_idx = vocab['<pad>'])
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = num_kernel, kernel_size = (2, embedding_size))
        self.conv2 = nn.Conv2d(in_channels = 1, out_channels = num_kernel, kernel_size = (3, embedding_size))
        self.conv3 = nn.Conv2d(in_channels = 1, out_channels = num_kernel, kernel_size = (4, embedding_size))
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(p = dropout_rate)
        self.classifier = nn.Linear(num_kernel * 3, 5)

    def forward(self, x, lengths):
        x = self.embedding(x)# [N, T, E]
        x = torch.unsqueeze(x, 1)
        conv1 = self.pooling(F.relu(self.conv1(x))) # [N, num_kernel, 1, 1]
        conv2 = self.pooling(F.relu(self.conv2(x)))
        conv3 = self.pooling(F.relu(self.conv3(x)))
        x = torch.squeeze(torch.cat((conv1, conv2, conv3), 1))
        x = self.dropout(x)
        score = self.classifier(x)
        return score
    
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
if(load_address is not None):
    model = torch.load(load_address)


# In[11]:


print(model.named_modules)


# ### 4. Train Model

# In[19]:


def model_status(training):
    if(training):
        return 'model is in training'
    else:
        return 'model is in testing'


# In[12]:


def pack(batch_x, batch_y, is_train = 1):
    x = batch_x['words'].to(device)
    lengths = batch_x['length'].to(device)
    if(is_train):
        y = batch_y['Sentiment'].to(device)
        return x, lengths, y
    else:
        return x, lengths


# In[22]:


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


# In[37]:


def train(model, dataset, optimizer, num_epoch = 30):
    loss_history = []
    loss_fn = nn.CrossEntropyLoss().to(device)
    for i in range(num_epoch):
        start = time.time()
        print("Epoch: {0} start".format(i))
        model.train()
        print(model_status(model.training))
        losses = 0

        for batch_x, batch_y in Batch(dataset, sampler = RandomSampler(), batch_size = batch_size):
            x, lengths, y = pack(batch_x, batch_y)
            score = model(x, lengths)
            loss = loss_fn(score, y)
            losses += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_history.append(losses)
        print("Epoch: {0} finish".format(i))
        acc = predict(model, val_dataset)
        end = time.time()
        print("Epoch: {0}, loss: {1}, accu: {2}, time: {3}\n".format(i, losses, acc, end - start))
    return loss_history

loss_history_new = train(model, train_dataset, optimizer, num_epoch = num_epoch)
loss_history += loss_history_new


# In[17]:


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
    name = "result/CNN_pretrain" + str(use_pretrain) + "_freeze" + str(freeze_pretrain) + "dropouot" + str(dropout_rate) + "_batch_size" + str(batch_size) + "_lr" + str(learning_rate) + "_epoch" + str(num_epoch) + ".csv"
    dataframe = pd.DataFrame({'PhraseId':index, 'Sentiment':answer})
    dataframe.to_csv(name,index=False,sep=',')
    return answer
answer = get_answer(model, test_dataset)
print('CNN text classification finished')
