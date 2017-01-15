# multimodal-neural-language-models

A bare-bones NumPy implementation of "Multimodal Neural Language Models" (Kiros et al, ICML 2014)

## Dependencies

This code is written in python. To use it you will need:

* Python 2.7
* A recent version of [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/)

## Quickstart for Toronto users

To train an additive log-bilinear model with the default settings, open IPython and run the following:

    import coco_proc, trainer
    z, zd, zt = coco_proc.process(context=5)
    trainer.trainer(z, zd)
    
this will store trained models in the models directory and periodically compute BLEU using the Perl code and reference captions in the gen directory. All the hyperparameters settings can be tuned in trainer.py. Links to MSCOCO data are in config.py.
    

    
