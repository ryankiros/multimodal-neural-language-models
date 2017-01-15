"""
Dataset configuration
"""
#-----------------------------------------------------------------------------#
# Paths to MSCOCO
#-----------------------------------------------------------------------------#
paths = dict()

# JSON annotations
paths['sentences_train2014'] = '/ais/gobi3/u/rkiros/coco/annotations/sentences_train2014.json'
paths['sentences_val2014'] = '/ais/gobi3/u/rkiros/coco/annotations/sentences_val2014.json'

# VGG19 features
paths['train'] = '/ais/gobi3/u/rkiros/coco/splits/train.npy'
paths['dev'] = '/ais/gobi3/u/rkiros/coco/splits/dev.npy'
paths['test'] = '/ais/gobi3/u/rkiros/coco/splits/test.npy'

# Data splits
paths['coco_train'] = '/ais/gobi3/u/rkiros/coco/coco_train.txt'
paths['coco_val'] = '/ais/gobi3/u/rkiros/coco/coco_val.txt'
paths['coco_test'] = '/ais/gobi3/u/rkiros/coco/coco_test.txt'

# Word embeddings
paths['embedding'] = '/ais/gobi3/u/rkiros/iaprtc12/embeddings-scaled.EMBEDDING_SIZE=100.txt'


