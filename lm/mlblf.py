# Multiplicative multimodal log-bilinear model

import numpy as np
import sys
from utils import lm_tools
from utils import svd_tools
from scipy.optimize import check_grad
from scipy.sparse import vstack
from numpy.random import RandomState
import time

class MLBLF(object):
    """
    Factored 3-way Multimodal Log-bilinear language model trained using SGD
    """
    def __init__(self,
                 name='lbl',
                 loc='models/mlblf.pkl',
                 seed=1234,
                 dropout=0.0,
                 k=5,
                 V=10364,
                 K=50,
                 D=10,
                 h=3,
                 factors=10,
                 context=5,
                 batchsize=20,
                 maxepoch=10,
                 eta_t=0.2,
                 gamma_r=0.0,
                 gamma_c=0.0,
                 f=0.995,
                 p_i=0.5,
                 p_f=0.5,
                 T=20,
                 verbose=1):
        """
        name: name of the network
        loc: location to save model files
        seed: random seed
        dropout: probability of dropout
        k: validation interval before stopping
        V: vocabulary size
        K: embedding dimensionality
        D: dimensionality of the image features
        h: intermediate layer dimensionality
        factors: number of factors
        context: word context length
        batchsize: size of the minibatches
        maxepoch: max number of training epochs
        eta_t: learning rate
        gamma_r: weight decay for representations
        gamma_c: weight decay for contexts
        f: learning rate decay
        p_i: initial momentum
        p_f: final momentum
        T: number of epochs until p_f is reached (linearly)
        verbose: display progress
        """
        self.name = name
        self.loc = loc
        self.dropout = dropout
        self.seed = seed
        self.k = k
        self.V = V
        self.K = K
        self.D = D
        self.h = h
        self.factors = factors
        self.context = context
        self.batchsize = batchsize
        self.maxepoch = maxepoch
        self.eta_t = eta_t
        self.gamma_r = gamma_r
        self.gamma_c = gamma_c
        self.f = f
        self.p_i = p_i
        self.p_f = p_f
        self.T = T
        self.verbose = verbose
        self.p_t = (1 - (1 / T)) * p_i + (1 / T) * p_f

    def init_params(self, embed_map, count_dict):
        """
        Initializes embeddings and context matricies
        """
        prng = RandomState(self.seed)

        # Pre-trained word embedding matrix
        if embed_map != None:
            R = np.zeros((self.K, self.V))
            for i in range(self.V):
                word = count_dict[i]
                if word in embed_map:
                    R[:,i] = embed_map[word]
                else:
                    R[:,i] = embed_map['*UNKNOWN*']
        else:
            r = np.sqrt(6) / np.sqrt(self.K + self.V + 1)
            R = prng.rand(self.K, self.V) * 2 * r - r
        bw = np.zeros((1, self.V))

        # Context 
        C = 0.01 * prng.randn(self.context, self.K, self.K)

        # Image context
        M = 0.01 * prng.randn(self.h, self.K)

        # Hidden layer
        r = np.sqrt(6) / np.sqrt(self.D + self.h + 1)
        J = prng.rand(self.D, self.h) * 2 * r - r
        bj = np.zeros((1, self.h))

        # Decomposition matricies
        Wfx, Whf = svd_tools.svd(R, n_components=self.factors, transpose='false')
        Wfv = 0.01 * prng.randn(self.h, self.factors)

        # Initial deltas used for SGD
        deltaC = np.zeros(np.shape(C))
        deltaB = np.zeros(np.shape(bw))
        deltaM = np.zeros(np.shape(M))
        deltaJ = np.zeros(np.shape(J))
        deltaBj = np.zeros(np.shape(bj))
        deltaWfx = np.zeros(np.shape(Wfx))
        deltaWhf = np.zeros(np.shape(Whf))
        deltaWfv = np.zeros(np.shape(Wfv))

        # Pack up
        self.C = C
        self.bw = bw
        self.M = M
        self.J = J
        self.bj = bj
        self.Wfx = Wfx
        self.Whf = Whf
        self.Wfv = Wfv
        self.deltaC = deltaC
        self.deltaB = deltaB
        self.deltaM = deltaM
        self.deltaJ = deltaJ
        self.deltaBj = deltaBj
        self.deltaWfx = deltaWfx
        self.deltaWhf = deltaWhf
        self.deltaWfv = deltaWfv

    def forward(self, X, Im, test=True):
        """
        Feed-forward pass through the model
        """
        batchsize = X.shape[0]

        # Forwardprop images
        IF = np.dot(Im, self.J) + self.bj
        IF = np.multiply(IF, (IF > 0))

        # Dropout (if applicable)
        if self.dropout > 0 and not test:
            dropmask = np.random.rand(batchsize, self.h) > self.dropout
            IF = IF * dropmask

        # Obtain word features
        R = np.dot(self.Wfx, self.Whf)
        tmp = R[:,X.flatten()].flatten(order='F').reshape((batchsize, self.K * self.context))
        words = np.zeros((batchsize, self.K, self.context))
        for i in range(batchsize):
            words[i] = tmp[i].reshape((self.K, self.context), order='F')

        # Compute the hidden layer (predicted next word representation)
        acts = np.zeros((batchsize, self.K))
        for i in range(self.context):
            acts += np.dot(words[:,:,i], self.C[i])
        if test:
            acts += np.dot(IF, self.M * (1 - self.dropout))
        else:
            acts += np.dot(IF, self.M)

        # Multiplicative interaction
        if test:
            F = np.multiply(np.dot(acts, self.Wfx), np.dot(IF, self.Wfv * (1 - self.dropout)))
        else:
            F = np.multiply(np.dot(acts, self.Wfx), np.dot(IF, self.Wfv))

        # Compute softmax
        preds = np.dot(F, self.Whf) + self.bw
        preds = np.exp(preds - preds.max(1).reshape(batchsize, 1))
        preds /= preds.sum(1).reshape(batchsize, 1)

        return (words, acts, IF, F, preds)

    def objective(self, Y, preds):
        """
        Compute the objective function
        """
        batchsize = Y.shape[0]

        # Cross-entropy
        C = -np.sum(Y.multiply(np.log(preds + 1e-20))) / batchsize
        return C

    def backward(self, Y, preds, F, IF, acts, words, X, Im):
        """
        Backward pass through the network
        """
        batchsize = preds.shape[0]

        # Compute part of df/dWhf
        Y = np.array(Y.todense())
        Ix = (preds - Y) / batchsize
        dWhf = np.dot(F.T, Ix)
        db = np.sum(Ix, 0)

        # Compute df/Wfv and part of df/Wfx
        Ix = np.dot(Ix, self.Whf.T)
        dWfv = np.dot(IF.T, np.multiply(Ix, np.dot(acts, self.Wfx)))
        dWfx = np.dot(acts.T, np.multiply(Ix, np.dot(IF, self.Wfv)))

        # Compute df/dC and word inputs for df/dR
        Ix_word = np.dot(np.multiply(Ix, np.dot(IF, self.Wfv)), self.Wfx.T)
        dC = np.zeros(np.shape(self.C))
        dR = np.zeros((self.K, self.V))
        for i in range(self.context):
            dC[i] = np.dot(words[:,:,i].T, Ix_word)
            delta = np.dot(Ix_word, self.C[i].T)
            for j in range(X.shape[0]):
                dR[:, X[j,i]] += delta.T[:,j]
        dWfx += np.dot(dR, self.Whf.T)
        dWhf += np.dot(self.Wfx.T, dR)

        # Compute df/dM
        dM = np.dot(IF.T, Ix_word)

        # Compute df/dJ
        Ix = np.multiply(np.dot(np.multiply(Ix, np.dot(acts, self.Wfx)), self.Wfv.T), (IF > 0)) + \
             np.multiply(np.dot(Ix_word, self.M.T), (IF > 0))
        dJ = np.dot(Im.T, Ix)
        dBj = np.sum(Ix, 0)

        # Weight decay terms
        dWhf += self.gamma_r * self.Whf
        dWfv += self.gamma_r * self.Wfv
        dWfx += self.gamma_r * self.Wfx
        dC += self.gamma_c * self.C
        dM += self.gamma_c * self.M
        dJ += self.gamma_c * self.J

        # Pack
        self.db = db
        self.dC = dC
        self.dM = dM
        self.dJ = dJ
        self.dBj = dBj
        self.dWhf = dWhf
        self.dWfv = dWfv
        self.dWfx = dWfx

    def update_params(self, X):
        """
        Update the network parameters using the computed gradients
        """
        batchsize = X.shape[0]
        self.deltaC = self.p_t * self.deltaC - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.dC
        self.deltaB = self.p_t * self.deltaB - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.db
        self.deltaM = self.p_t * self.deltaM - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.dM
        self.deltaJ = self.p_t * self.deltaJ - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.dJ
        self.deltaBj = self.p_t * self.deltaBj - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.dBj
        self.deltaWhf = self.p_t * self.deltaWhf - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.dWhf
        self.deltaWfv = self.p_t * self.deltaWfv - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.dWfv
        self.deltaWfx = self.p_t * self.deltaWfx - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.dWfx

        self.C = self.C + self.deltaC
        self.bw = self.bw + self.deltaB
        self.M = self.M + self.deltaM
        self.J = self.J + self.deltaJ
        self.bj = self.bj + self.deltaBj
        self.Wfv = self.Wfv + self.deltaWfv
        self.Wfx = self.Wfx + self.deltaWfx
        self.Whf = self.Whf + self.deltaWhf

    def update_hyperparams(self):
        """
        Updates the learning rate and momentum schedules
        """
        self.eta_t *= self.f
        if self.step < self.T:
            self.p_t = (1 - ((self.step + 1) / self.T)) * self.p_i + \
                ((self.step + 1) / self.T) * self.p_f
        else:
            self.p_t = self.p_f

    def compute_obj(self, X, Im, Y):
        """
        Perform a forward pass and compute the objective
        """
        preds = self.forward(X, Im)[-1]
        obj = self.objective(Y, preds)
        return obj

    def train(self, X, indX, XY, V, indV, VY, IM, VIM, count_dict, word_dict, embed_map):
        """
        Trains the LBL
        """
        self.start = self.seed
        self.init_params(embed_map, count_dict)
        self.step = 0
        inds = np.arange(len(X))
        numbatches = len(inds) / self.batchsize
        tic = time.time()
        bleu = [0.0]*4
        best = 0.0
        scores = '/'.join([str(b) for b in bleu])
        patience = 10
        count = 0
        done = False

        # Progress printing times
        _details = 1000
        _samples = 10000
        _update = 100000
        _bleu = 1000000

        # Main loop
        lm_tools.display_phase(1)
        for epoch in range(self.maxepoch):
            if done:
                break
            self.epoch = epoch
            prng = RandomState(self.seed + epoch + 1)
            prng.shuffle(inds)
            for minibatch in range(numbatches):

                batchX = X[inds[minibatch::numbatches]]
                batchY = XY[inds[minibatch::numbatches]]
                batchindX = indX[inds[minibatch::numbatches]].astype(int).flatten()
                batchindX = np.floor(batchindX/5).astype(int)
                batchIm = IM[batchindX]

                (words, acts, IF, F, preds) = self.forward(batchX, batchIm, test=False)
                self.backward(batchY, preds, F, IF, acts, words, batchX, batchIm)
                self.update_params(batchX)
                if np.sum(np.isnan(acts)) > 0:
                    print 'NaNs... breaking out'
                    done = True
                    break

                # Print out progress
                if np.mod(minibatch * self.batchsize, _details) == 0 and minibatch > 0:
                    print "epoch: %d, pts: %d, time: %.2f" % (epoch, minibatch * self.batchsize, (time.time()-tic)/60)
                if np.mod(minibatch * self.batchsize, _samples) == 0 and minibatch > 0:
                    print "best: %s" % (scores)
                    print '\nSamples:'
                    lm_tools.generate_and_show(self, word_dict, count_dict, VIM, k=3)
                    print ' '
                if np.mod(minibatch * self.batchsize, _update) == 0 and minibatch > 0:
                    self.update_hyperparams()
                    self.step += 1
                    print "learning rate: %.4f, momentum: %.4f" % (self.eta_t, self.p_t)

                # Compute BLEU
                if np.mod(minibatch * self.batchsize, _bleu) == 0 and minibatch > 0:
                    bleu = lm_tools.compute_bleu(self, word_dict, count_dict, VIM, k=3)
                    if bleu[-1] >= best:
                        count = 0
                        best = bleu[-1]
                        scores = '/'.join([str(b) for b in bleu])
                        print scores + '\n'
                        lm_tools.save_model(self, self.loc)
                    else:
                        count += 1
                        if count == patience:
                            done = True
                            break

            self.update_hyperparams()
            self.step += 1
        return best

def main():
    pass

if __name__ == '__main__':
    main()


