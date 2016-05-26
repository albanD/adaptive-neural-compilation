# Adaptive Neural Compilation

This repository contains the code associated to the Adaptive Neural Compilation paper that can be found on [arxiv](http://arxiv.org/abs/1605.07969).

In this repository, you can find two projects:
* The compiler in the [`compilation/`](compilation) folder allows to transform a program written in a low level language (examples can be found [here](compilation/tests)) into a set of weights for our model. These weights are represented as a configuration file that will be used by the adaptation project.
* The learning part in the [`adaptation/`](adaptation) folder provides the execution model, the training script and all the utilities to run the experiments. Examples of configurations files can be found [here](adaptation/examples) and examples of training commands can be found [here](adaptation/paper-experiment/run-experiments.sh)


If you use this work, please cite:
```
@article{anc,
    title={Adaptive Neural Compilation},
    author={Bunel, Rudy and Desmaison, Alban and Kohli, Pushmeet and Torr, Philip H.S and Kumar, M. Pawan},
    journal={arXiv preprint arXiv:1605.07969},
    year={2016}
}
```