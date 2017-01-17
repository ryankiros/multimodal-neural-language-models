# Tester module

from utils import lm_tools

def tester(loc, z, zd, k=3, neval=500, evaldev=True):
    """
    Trainer function for multimodal log-bilinear models
    loc: location of model to evaluate
    k: the beam width to use for inference
    neval: Number of images to evaluate
    evaldev: True if evaluating on dev set, False for test set
    """
    prog = {}
    prog['_neval'] = neval
    prog['_evaldev'] = evaldev

    print 'Loading model...'
    net = lm_tools.load_model(loc)

    print 'Evaluating...'
    bleu = lm_tools.compute_bleu(net, z['word_dict'], z['index_dict'], zd['IM'], prog, k=k)
    print bleu


