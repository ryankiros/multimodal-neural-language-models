# multimodal-neural-language-models

A bare-bones NumPy implementation of "Multimodal Neural Language Models" (Kiros et al, ICML 2014), containing additive and multiplicative log-bilinear image caption generators. These models differ from most other image caption generators in that they do not use recurrent neural networks.

This code may be useful to you if you're looking for a simple, bare-bones image caption generator that can be trained on the CPU. It may also be useful for teaching purposes. This code was used as part of an assignment for the undergraduate neural networks class at the University of Toronto.

On MSCOCO using VGG19 features, a single model can achieve BLEU4 score of 25. An ensemble can achieve near 27. For comparison, a "Show and Tell" LSTM with the same features achieves a score of 27.x. The state of the art is currently around 34. Thus these models are quite far from the current state of the art. I am releasing this code for completeness as part of my PhD thesis.

## Visualization

Here are [results](http://www.cs.toronto.edu/~rkiros/bayescapgen.html) on 1000 images using an ensemble of additive log-bilinear models trained using this code.

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

## Getting started

You will first need to download the pre-processed MSCOCO data. All necessary files can be downloaded by running:

    wget http://www.cs.toronto.edu/~rkiros/data/mnlm.zip
    
After unpacking, open config.py and set the paths accordingly. Then you can proceed to the quickstart instructions. All training settings can be found in trainer.py. Testing trained models is done with tester.py. The lm directory contains classes for the additive and multiplicative log-bilinear models. Helper functions, such as beam search, is found in the utils directory.

## Reference

If you found this code useful, please cite the following paper:

Ryan Kiros, Ruslan Salakhutdinov, Richard S. Zemel. **"Multimodal Neural Language Models."** *ICML (2014).*

    @inproceedings{kiros2014multimodal,
      title={Multimodal Neural Language Models.},
      author={Kiros, Ryan and Salakhutdinov, Ruslan and Zemel, Richard S},
      booktitle={ICML},
      volume={14},
      pages={595--603},
      year={2014}
    }
    
## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

    
