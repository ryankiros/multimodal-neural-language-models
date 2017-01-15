# Trainer module

import numpy as np
import copy
import config
from lm import mlbl, mlblf
from utils import lm_tools
from numpy.random import RandomState

def trainer(z, zd):
    """
    Trainer function for multimodal log-bilinear models

    Dictionary:
    'model' ('add' or 'mul') int:[0,1], cat{add, mul}
    'name' (name of the model, unique to each run)
    'loc' (location to save)
    'context' int:[3,25]
    'learning_rate' float:[0.001, 10]
    'momentum' float:[0, 0.9]
    'batch_size' int:[20, 100]
    'hidden_size' int:[100, 2000]
    'dropout' float:[0, 0.7]
    'word_decay' float:[1e-3, 1e-9]
    'context_decay' float:[1e-3, 1e-9]
    'factors' (mul model only!) int:[50,200], truncate by embedding_size
    """
    d = {}
    d['model'] = 'add'
    d['name'] = 'testrun'
    d['loc'] = './models/' + d['model'] + '_' + d['name']
    d['context'] = 5
    d['learning_rate'] = 0.43
    d['momentum'] = 0.23
    d['batch_size'] = 40
    d['maxepoch'] = 10
    d['hidden_size'] = 441
    d['dropout'] = 0.15
    d['word_decay'] = 3e-7
    d['context_decay'] = 1e-8
    d['factors'] = 50

    print d['loc']
   
    # Load the word embeddings
    embed_map = load_embeddings()

    # Unpack some stuff from the data
    train_ngrams = z['ngrams']
    train_labels = z['labels']
    train_instances = z['instances']
    word_dict = z['word_dict']
    index_dict = z['index_dict']
    context = z['context']
    vocabsize = len(z['word_dict'])
    trainIM = z['IM']
    train_index = z['index']

    dev_ngrams = zd['ngrams']
    dev_labels = zd['labels']
    dev_instances = zd['instances']
    devIM = zd['IM']
    dev_index = zd['index']

    # Initialize the network
    if d['model'] == 'add':
        net = mlbl.MLBL(name=d['name'],
                        loc=d['loc'],
                        seed=1234,
                        dropout=d['dropout'],
                        V=vocabsize,
                        K=100,
                        D=trainIM.shape[1],
                        h=d['hidden_size'],
                        context=d['context'],
                        batchsize=d['batch_size'],
                        maxepoch=d['maxepoch'],
                        eta_t=d['learning_rate'],
                        gamma_r=d['word_decay'],
                        gamma_c=d['context_decay'],
                        f=0.99,
                        p_i=d['momentum'],
                        p_f=d['momentum'],
                        T=20.0,
                        verbose=1)
    elif d['model'] == 'mul':
        net = mlblf.MLBLF(name=d['name'],
                          loc=d['loc'],
                          seed=1234,
                          dropout=d['dropout'],
                          V=vocabsize,
                          K=100,
                          D=trainIM.shape[1],
                          h=d['hidden_size'],
                          factors=d['factors'],
                          context=d['context'],
                          batchsize=d['batch_size'],
                          maxepoch=d['maxepoch'],
                          eta_t=d['learning_rate'],
                          gamma_r=d['word_decay'],
                          gamma_c=d['context_decay'],
                          f=0.99,
                          p_i=d['momentum'],
                          p_f=d['momentum'],
                          T=20.0,
                          verbose=1)

    # Train the network
    X = train_instances
    indX = train_index
    Y = train_labels
    V = dev_instances
    indV = dev_index
    VY = dev_labels

    best = net.train(X, indX, Y, V, indV, VY, trainIM, devIM, index_dict, word_dict, embed_map)
    return best

def load_embeddings():
    """
    Load in the embeddings
    """
    embed_map = {}
    ap = open(config.paths['embedding'], 'r')
    for line in ap:
        entry = line.split(' ')
        key = entry[0]
        value = [float(x) for x in entry[1:]]
        embed_map[key] = value
    return embed_map


