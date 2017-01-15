# Pre-process MSCOCO

import config
import numpy as np
import os
import json
import re
from utils import lm_tools
from collections import Counter
from collections import defaultdict
from scipy.sparse import lil_matrix, sparsetools, csr_matrix
from numpy.random import RandomState

def process(context=5):
    """
    Main process function
    """
    # Load images
    print 'Loading images...'
    (trainIM, devIM, testIM) = load_features_npy()

    # Load sentences
    print 'Loading sentences...'
    d = load_sentences()

    # Load image ids
    print 'Loading image ids...'
    (dx_train, dx_dev) = image_ids()

    # Load splits
    print 'Loading splits...'
    (train_sp, dev_sp, test_sp) = load_splits()

    # Load captions
    print 'Loading captions...'
    train = construct_captions(d, train_sp)
    dev = construct_captions(d, dev_sp)
    test = construct_captions(d, test_sp)

    # Tokenize
    (train_tokens, topwords) = tokenize(train, context=context)
    dev_tokens = tokenize(dev, context=context, topwords=topwords)[0]
    test_tokens = tokenize(test, context=context, topwords=topwords)[0]

    # Index words and create vocabulary
    print 'Creating vocabulary...'
    (word_dict, index_dict) = index_words(train_tokens + dev_tokens)

    # Compute n-grams
    print 'Computing n-grams...'
    train_ngrams = lm_tools.get_ngrams(train_tokens, context=context)
    dev_ngrams = lm_tools.get_ngrams(dev_tokens, context=context)
    test_ngrams = lm_tools.get_ngrams(test_tokens, context=context)

    # Compute sparse label matrix
    print 'Computing labels...'
    train_labels = compute_labels(train_ngrams, word_dict, context=context)
    dev_labels = compute_labels(dev_ngrams, word_dict, context=context)

    # Compute model instances
    print 'Computing model instances...'
    (train_instances, train_index) = lm_tools.model_inputs(train_ngrams, word_dict,
        context=context, include_last=False, include_index=True)
    (dev_instances, dev_index) = lm_tools.model_inputs(dev_ngrams, word_dict,
        context=context, include_last=False, include_index=True)
    (test_instances, test_index) = lm_tools.model_inputs(test_ngrams, word_dict,
        context=context, include_last=False, include_index=True)

    # Save everything into dictionaries
    print 'Packing up...'
    z = {}
    z['text'] = train
    z['tokens'] = train_tokens
    z['word_dict'] = word_dict
    z['index_dict'] = index_dict
    z['ngrams'] = train_ngrams
    z['labels'] = train_labels
    z['instances'] = train_instances
    z['IM'] = trainIM
    z['index'] = train_index
    z['context'] = context

    zd = {}
    zd['text'] = dev
    zd['tokens'] = dev_tokens
    zd['ngrams'] = dev_ngrams
    zd['labels'] = dev_labels
    zd['instances'] = dev_instances
    zd['IM'] = devIM
    zd['index'] = dev_index
    zd['context'] = context

    zt = {}
    zt['text'] = test
    zt['tokens'] = test_tokens
    zt['ngrams'] = test_ngrams
    zt['instances'] = test_instances
    zt['IM'] = testIM
    zt['index'] = test_index
    zt['context'] = context

    return (z, zd, zt)

def load_json():
    """
    Load the JSON annotations
    """
    # Load the training sentences
    f = open(config.paths['sentences_train2014'])
    train_data = json.load(f)
    f.close()

    # Load the validation sentences
    f = open(config.paths['sentences_val2014'])
    val_data = json.load(f)
    f.close()

    return (train_data, val_data)

def uniq(seq):
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if x not in seen and not seen_add(x)]

def load_features_npy():
    """
    Load directly from numpy files
    """
    trainIM = np.load(config.paths['train'])
    devIM = np.load(config.paths['dev'])
    testIM = np.load(config.paths['test'])
    return (trainIM, devIM, testIM)

def load_splits():
    """
    Load train/dev/test splits
    """
    (train, dev, test) = ([], [], [])
    f = open(config.paths['coco_train'])
    for line in f:
        train.append(int(line.strip()[:-4][-12:]))
    f.close()
    f = open(config.paths['coco_val'])
    for line in f:
        dev.append(int(line.strip()[:-4][-12:]))
    f.close()
    f = open(config.paths['coco_test'])
    for line in f:
        test.append(int(line.strip()[:-4][-12:]))
    f.close()
    return (train, dev, test)

def image_ids():
    """
    Return a dictionary mapping image features to their IDs
    """
    dx_train = {}
    dx_dev = {}
    count = 0
    (train_data, val_data) = load_json()

    # Part-1: COCO training data
    tr = []
    for x in train_data['images']:
        tr.append(x['id'])
    tr = sorted(tr)
    for i, x in enumerate(tr):
        dx_train[x] = count
        count += 1

    # Part-2: COCO validation data
    count = 0
    va = []
    for x in val_data['images']:
        va.append(x['id'])
    va = sorted(va)
    for i, x in enumerate(va):
        dx_dev[x] = count
        count += 1
    
    return (dx_train, dx_dev)

def load_sentences():
    """
    Return a dictionary of image ids to sentences
    """
    (train_data, val_data) = load_json()

    # Populate the dictionary
    d = defaultdict(list)
    for x in train_data['sentences']:
        image_id = x['image_id']
        sentence = x['sentence']
        d[image_id].append(sentence)
    for x in val_data['sentences']:
        image_id = x['image_id']
        sentence = x['sentence']
        d[image_id].append(sentence)
    return d

def construct_captions(d, ids):
    """
    Construct captions for entries in ids
    """
    X = []
    for x in ids:
        captions = d[x]
        for s in captions[:5]:
            X.append(s)
    return X

def word_tokenize(text):
    """
    Perform word tokenization (from NLTK)
    """
    CONTRACTIONS2 = [re.compile(r"(?i)\b(can)(not)\b"),
                     re.compile(r"(?i)\b(d)('ye)\b"),
                     re.compile(r"(?i)\b(gim)(me)\b"),
                     re.compile(r"(?i)\b(gon)(na)\b"),
                     re.compile(r"(?i)\b(got)(ta)\b"),
                     re.compile(r"(?i)\b(lem)(me)\b"),
                     re.compile(r"(?i)\b(mor)('n)\b"),
                     re.compile(r"(?i)\b(wan)(na) ")]
    CONTRACTIONS3 = [re.compile(r"(?i) ('t)(is)\b"),
                     re.compile(r"(?i) ('t)(was)\b")]

    #starting quotes
    text = re.sub(r'^\"', r'``', text)
    text = re.sub(r'(``)', r' \1 ', text)
    text = re.sub(r'([ (\[{<])"', r'\1 `` ', text)

    #punctuation
    text = re.sub(r'([:,])([^\d])', r' \1 \2', text)
    text = re.sub(r'\.\.\.', r' ... ', text)
    text = re.sub(r'[;@#$%&]', r' \g<0> ', text)
    text = re.sub(r'([^\.])(\.)([\]\)}>"\']*)\s*$', r'\1 \2\3 ', text)
    text = re.sub(r'[?!]', r' \g<0> ', text)
    text = re.sub(r"([^'])' ", r"\1 ' ", text)

    #parens, brackets, etc.
    text = re.sub(r'[\]\[\(\)\{\}\<\>]', r' \g<0> ', text)
    text = re.sub(r'--', r' -- ', text)

    #add extra space to make things easier
    text = " " + text + " "

    #ending quotes
    text = re.sub(r'"', " '' ", text)
    text = re.sub(r'(\S)(\'\')', r'\1 \2 ', text)

    text = re.sub(r"([^' ])('[sS]|'[mM]|'[dD]|') ", r"\1 \2 ", text)
    text = re.sub(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) ", r"\1 \2 ",
                  text)

    for regexp in CONTRACTIONS2:
        text = regexp.sub(r' \1 \2 ', text)
    for regexp in CONTRACTIONS3:
        text = regexp.sub(r' \1 \2 ', text)

    return text.split()

def tokenize(X, context=5, start='<start>', end='<end>', topwords=None):
    """
    Tokenize each of the captions
    """
    tokens = [word_tokenize(x) for x in X]
    if topwords == None:
        word_counts = get_counts(tokens)
        topwords = [w for w in word_counts.keys() if word_counts[w] >= 5]
        topwords += ['unk']
    d = defaultdict(lambda : 0)
    for w in topwords:
        d[w] = 1
    tokens = [[w if d[w] > 0 else 'unk' for w in t] for t in tokens]
    for i, x in enumerate(tokens):
        tokens[i] = [start] * context + x + [end]
    return (tokens, topwords)

def get_counts(tokens):
    """
    Compute a dictionary of counts from tokens
    """
    flat_tokens = [item for sublist in tokens for item in sublist]
    word_counts = Counter(flat_tokens)
    return word_counts

def index_words(tokens):
    """
    Compute dictionaries for indexing words
    """
    flat_tokens = [item for sublist in tokens for item in sublist]
    word_dict = {}
    for i, w in enumerate(list(set(flat_tokens))):
        word_dict[w] = i
    index_dict = dict((v,k) for k, v in word_dict.iteritems())
    return (word_dict, index_dict)

def compute_labels(ngrams, word_dict, context=5):
    """
    Create matrix of word occurences (labels for the model)
    """
    ngrams_count = [len(x) for x in ngrams]
    uniq_ngrams = uniq([item[:-1] for sublist in ngrams for item in sublist])
    count = 0
    train_dict = {}
    for w in uniq_ngrams:
        train_dict[w] = count
        count = count + 1

    labels = lil_matrix((sum(ngrams_count), len(word_dict.keys())))
    train_ngrams_flat = [item for sublist in ngrams for item in sublist]
    labels_dict = defaultdict(int)
    col_dict = defaultdict(list)

    for w in train_ngrams_flat:
        row_ind = train_dict[w[:context]]
        col_ind = word_dict[w[-1]]
        labels_dict[(row_ind, col_ind)] += 1
        col_dict[row_ind] = list(set(col_dict[row_ind] + [col_ind]))

    count = 0
    for x in ngrams:
        for w in x:
            row_ind = train_dict[w[:context]]
            inds = col_dict[(row_ind)]
            labels[count, word_dict[w[-1]]] = 1
            count = count + 1

    labels_un = labels.tocsr()
    return labels_un


