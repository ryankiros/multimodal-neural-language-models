# Language model tools for LBL and MLBL

import numpy as np
import copy
import random
import sys
import os
import cPickle as pickle
from collections import defaultdict
from scipy.linalg import norm

def save_model(net, loc):
    """
    Save the network model to the specified directory
    """
    output = open(loc, 'wb')
    pickle.dump(net, output)
    output.close()

def load_model(loc):
    """
    Load the network model from the specified directory
    """
    inputs = open(loc, 'rb')
    net = pickle.load(inputs)
    inputs.close()
    return net

def display_phase(phase):
    """
    Print a message displaying the current training phase
    """
    print "============================== Training phase %d ==============================" % (phase)

def compute_ngrams(sequence, n):
    """
    Return n-grams from the input sequence
    """
    sequence = list(sequence)
    count = max(0, len(sequence) - n + 1)
    return [tuple(sequence[i:i+n]) for i in range(count)]

def get_ngrams(X, context=5):
    """
    Extract n-grams from each caption in X
    """
    ngrams = []
    for x in X:
        x_ngrams = compute_ngrams(x, context + 1)
        ngrams.append(x_ngrams)
    return ngrams

def model_inputs(ngrams, word_dict, context=5, include_last=True, include_index=False):
    """
    Maps ngrams to format used for the language model
    include_last=True for evaluation (LL, perplexity)
    Out of vocabulary words are mapped to 'unk' (unknown) token
    """
    d = defaultdict(lambda : 0)
    for w in word_dict.keys():
        d[w] = 1   
    ngrams_count = [len(x) for x in ngrams]
    if include_last:
        instances = np.zeros((sum(ngrams_count), context + 1))
    else:
        instances = np.zeros((sum(ngrams_count), context))
    count = 0
    index = np.zeros((sum(ngrams_count), 1))
    for i in range(len(ngrams)):
        for j in range(len(ngrams[i])):
            values = [word_dict[w] if d[w] > 0 else word_dict['unk']
                for w in ngrams[i][j]]
            if include_last:
                instances[count] = values
            else:
                instances[count] = values[:-1]
            index[count] = i
            count = count + 1
    instances = instances.astype(int)
    if include_index:
        return (instances, index)
    else:
        return instances

def compute_ll(net, instances, Im=None):
    """
    Compute the log-likelihood of instances from net
    """
    if Im != None:
        preds = net.forward(instances[:,:-1], Im)[-1]
    else:
        preds = net.forward(instances[:,:-1])[-1]
    ll = 0
    for i in range(preds.shape[0]):
        ll += np.log2(preds[i, instances[i, -1]] + 1e-20)
    return ll

def perplexity(net, ngrams, word_dict, Im=None, context=5):
    """
    Compute the perplexity of ngrams from net
    """
    ll = 0
    N = 0
    for i, ng in enumerate(ngrams):
        instances = model_inputs([ng], word_dict, context=context)
        if Im != None:
            ll += compute_ll(net, instances, np.tile(Im[i], (len(ng), 1)))
        else:
            ll += compute_ll(net, instances)
        N += len(instances)
    return pow(2, (-1.0 / N) * ll)

def weighted_sample(n_picks, weights):
    """
    Sample from a distribution weighted by 'weights'
    """
    t = np.cumsum(weights)
    s = np.sum(weights)
    return np.searchsorted(t, np.random.rand(n_picks) * s)

def beam_search(net, word_dict, index_dict, num, Im, initial=None, k=2, N=1, lm=None, beta=0.0, rerank=False):
    """
    Return a N-best list of generated captions from a beam width of k
    """
    # Set the initialization
    if initial == None:
        initial = ['<start>'] * net.context
    inputs = np.array([word_dict[w] for w in initial]).reshape(1, net.context)

    # Initialize the beams
    beam_tokens = []
    beam_inputs = []
    beam_scores = []
    for i in range(k):
        beam_scores.append([0])
        beam_tokens.append(['<start>'] * net.context)
        beam_inputs.append(np.array([word_dict[w] for w in initial]).reshape(1, net.context))

    # Start loop
    done = False
    count = 1
    while not done:
        
        # Special case when count = 1
        if count == 1:
            preds = net.forward(inputs[:,inputs.shape[1]-net.context:], [Im])[-1].flatten()
            preds = np.log(preds + 1e-20)
            argpreds = np.argsort(preds)[::-1]
            words = [index_dict[w] for w in argpreds][:k]
            scores = preds[argpreds][:k]
            for i in range(k):
                beam_tokens[i].append(words[i])
                beam_inputs[i] = np.c_[beam_inputs[i], argpreds[i]]
                beam_scores[i].append(scores[i])
                beam_scores[i] = beam_scores[i][1:]
            count += 1

        # Every other case
        if count > 1:

            # Loop over each beam
            candidate_tokens = []
            candidate_scores = []
            candidate_inputs = []
            candidate_norm = []
            for i in range(k):

                # Make predictions and sort
                preds = net.forward(beam_inputs[i][:,beam_inputs[i].shape[1]-net.context:], [Im])[-1].flatten()
                preds = np.log(preds + 1e-20)
                argpreds = np.argsort(preds)[::-1]
                words = [index_dict[w] for w in argpreds][:k]
                scores = preds[argpreds][:k]
                for j in range(k):

                    # First deal with tokens
                    tmp = copy.deepcopy(beam_tokens[i])
                    last_word = tmp[-1]
                    if last_word != '<end>':
                        tmp.append(words[j])
                        candidate_tokens.append(tmp)
                        candidate_norm.append(len(tmp) - net.context)
                    elif j == 0:
                        candidate_tokens.append(tmp)
                        candidate_norm.append(len(tmp) - net.context)

                    # Then scores
                    tmp = copy.deepcopy(beam_scores[i])
                    if last_word != '<end>':
                        tmp.append(scores[j])
                        candidate_scores.append(tmp)
                    elif j == 0:
                        candidate_scores.append(tmp)

                    # Then inputs
                    tmp = copy.deepcopy(beam_inputs[i])
                    if last_word != '<end>':
                        tmp = np.c_[tmp, argpreds[j]]
                        candidate_inputs.append(tmp)
                    elif j == 0:
                        candidate_inputs.append(tmp)

            # Now sort and rescore
            scores = [sum(w) for w in candidate_scores]
            for i in range(len(scores)):
                scores[i] /= candidate_norm[i]
            argscores = np.argsort(scores)[::-1][:k]
            
            # Reset the beams based on the scores
            for i in range(k):
                beam_tokens[i] = candidate_tokens[argscores[i]]
                beam_scores[i] = candidate_scores[argscores[i]]
                beam_inputs[i] = candidate_inputs[argscores[i]]

            # Shallow fusion (if applicable)
            if beta > 0:
                for i in range(k):
                    # Need to reverse the conditionals for SRILM convention
                    lmscore = beta * lm.logprob_strings(beam_tokens[i][-1], beam_tokens[i][:-1][::-1])
                    if lmscore == -np.inf:
                        lmscore = 0.0
                    beam_scores[i][-1] += lmscore
            
            # Check if all beams have produced <end> tokens
            numends = 0
            for i in range(k):
                if beam_tokens[i][-1] == '<end>':
                    numends += 1
            if numends == k:
                done = True

            # If we've gone too long, also finish
            count += 1
            if count == num:
                done = True

    # Return the top-N beams
    topbeams = [b[net.context:-1] for b in beam_tokens[:k]]
    if rerank:
        scores = np.zeros(k)
        for i in range(k):
            scores[i] = lm.total_logprob_strings(topbeams[i][1:])
            scores[i] /= len(topbeams[i][1:])
        argscores = np.argsort(scores)[::-1]
        topbeams = [topbeams[i] for i in argscores]

    return topbeams

def search(net, z, maxlen=50, im=None, init=None, k=2, N=1):
    """
    Generate samples from the net using beam search
    """
    captions = beam_search(net, z['word_dict'], z['index_dict'], num=maxlen, Im=im, initial=init, k=k, N=N)
    return captions

def generate_and_show(net, word_dict, index_dict, IM, k=1, num=5):
    """
    Generate and show results from the model
    """
    inds = range(len(IM))
    random.shuffle(inds)
    for i in inds[:num]:
        caption = beam_search(net, word_dict, index_dict, 50, IM[i], k=k, N=1)[0]
        print ' '.join(caption)

def generate_and_save(net, z, IM, k=1, model='mlblf', dataset='coco', split='dev', extra='c10'):
    """
    Generate and save results
    """
    maxlen=50
    saveloc = model + '_' + dataset + '_' + split + '_' + extra + '_' + 'bw' + str(k) + '.txt'
    print saveloc
    captions = []
    for i in range(0, len(IM), 5):
        c = search(net, z, maxlen, IM[i], k=k, N=1)[0]
        print (i, ' '.join(c))
        captions.append(c)
    f = open(saveloc, 'wb')
    for c in captions:
        f.write(' '.join(c) + '\n')
    f.close()
    return captions

def compute_bleu(net, word_dict, index_dict, IM, k=1, maxlen=50, lm=None, beta=0.0, rerank=False):
    """
    Compute BLEU
    """
    nex = 500
    print '\nComputing BLEU...'
    saveloc = './gen/' + net.name + '_bleu_' + str(k) + '_offdev'
    print saveloc
    captions = []
    for i in range(0, len(IM[:nex]), 1):
        c = beam_search(net, word_dict, index_dict, maxlen, IM[i], k=k, N=1, lm=lm, beta=beta, rerank=rerank)[0]
        print (i, ' '.join(c))
        captions.append(c)
    f = open(saveloc, 'wb')
    for c in captions:
        f.write(' '.join(c) + '\n')
    f.close()   
    os.system("./gen/multi-bleu.perl ./gen/coco_dev_reference < " + saveloc + ' > ' + saveloc + '_scores')
    f = open(saveloc + '_scores', 'rb')
    bleu = f.readline()
    f.close()
    bleu = bleu[7:].split('/')
    bleu[-1] = bleu[-1].split('(')[0]
    bleu = [float(b) for b in bleu]
    return bleu



